#!/bin/bash
# 双节点实验启动脚本 - 基于成功经验
# 用法: bash run_dual_node_exp.sh <experiment_name> <TP> <PP> <DP> <freq_mhz>

set -euo pipefail

# 解析参数
EXP_NAME="${1:-test_exp}"
TP="${2:-2}"
PP="${3:-2}"
DP="${4:-4}"
FREQ_MHZ="${5:-1200}"

BASE_PATH="/home/sd/Megatron-DeepSpeed"
cd "$BASE_PATH"

# 创建日志目录
mkdir -p logs

# 使用系统 Python (成功经验)
PYTHON="/usr/bin/python3"
DEEPSPEED="/home/sd/.local/bin/deepspeed"

# 设置关键环境变量 (成功经验)
export NCCL_RAS_ENABLE=0
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export PYTHONPATH="/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:"
export PDSH_RCMD_TYPE=ssh

# 创建 hostfile (成功经验)
HOSTFILE="/tmp/dual_node_hostfile_$$"
cat > "$HOSTFILE" << EOF
localhost slots=8
192.168.205.202 slots=8
EOF

echo "=============================================="
echo "双节点实验: $EXP_NAME"
echo "拓扑: TP=$TP, PP=$PP, DP=$DP"
echo "频率: ${FREQ_MHZ}MHz"
echo "Python: $PYTHON"
echo "Hostfile: $HOSTFILE"
echo "=============================================="

# 清理旧检查点
rm -rf "${BASE_PATH}/checkpoints/${EXP_NAME}"

# 创建 DeepSpeed 配置
DS_CONFIG="/tmp/ds_config_${EXP_NAME}_$$.json"
cat > "$DS_CONFIG" << EOF
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 1,
  "steps_per_print": 1,
  "zero_optimization": {"stage": 1},
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "wall_clock_breakdown": false
}
EOF

# 计算 warmup steps
if (( 20 <= 1 )); then
    WARMUP=0
elif (( 20 <= 50 )); then
    WARMUP=19
else
    WARMUP=50
fi

# 锁定 GPU 频率 (如果指定)
if [[ -n "$FREQ_MHZ" ]]; then
    echo "锁定 GPU 频率到 ${FREQ_MHZ}MHz..."
    for i in {0..7}; do
        sudo nvidia-smi -i $i -lgc "$FREQ_MHZ" 2>/dev/null || true
    done
fi

# 启动训练 (使用成功经验中的 deepspeed 命令)
echo "启动训练..."
$PYTHON $DEEPSPEED \
    --hostfile "$HOSTFILE" \
    --num_nodes 2 \
    --num_gpus 8 \
    pretrain_gpt.py \
    --tensor-model-parallel-size "$TP" \
    --pipeline-model-parallel-size "$PP" \
    --num-layers 28 \
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --num-key-value-heads 4 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 20 \
    --data-path /home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document \
    --data-impl mmap \
    --tokenizer-type HFTokenizer \
    --tokenizer-model /home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat \
    --split 98,2,0 \
    --distributed-backend nccl \
    --lr 1e-5 \
    --lr-decay-style cosine \
    --min-lr 1e-6 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --lr-warmup-iters "$WARMUP" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --log-interval 1 \
    --save-interval 0 \
    --eval-interval 100 \
    --eval-iters 10 \
    --no-query-key-layer-scaling \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --recompute-granularity full \
    --recompute-method uniform \
    --deepspeed-activation-checkpointing \
    --zero-stage=1 \
    --deepspeed_config="$DS_CONFIG" \
    --deepspeed \
    --experiment-run-id "${EXP_NAME}_$(date +%Y%m%d_%H%M%S)" \
    --experiment-name "$EXP_NAME" \
    --experiment-root-dir /home/sd/Megatron-DeepSpeed/experiments \
    --bf16 2>&1 | tee "logs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=${PIPESTATUS[0]}

# 重置 GPU 频率
echo "重置 GPU 频率..."
for i in {0..7}; do
    sudo nvidia-smi -i $i -rgc 2>/dev/null || true
done

# 清理临时文件
rm -f "$HOSTFILE" "$DS_CONFIG"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ 实验完成: $EXP_NAME"
else
    echo "✗ 实验失败 (exit $EXIT_CODE): $EXP_NAME"
fi

exit $EXIT_CODE
