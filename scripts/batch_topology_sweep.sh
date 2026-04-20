#!/bin/bash
# 批量拓扑补测脚本

set -euo pipefail

BASE_PATH="/home/sd/Megatron-DeepSpeed"
cd "$BASE_PATH"

# 环境变量
export NCCL_RAS_ENABLE=0
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export PYTHONPATH="/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:"
export PDSH_RCMD_TYPE=ssh

# 创建 hostfile
HOSTFILE="/tmp/sweep_hostfile"
echo "localhost slots=8" > "$HOSTFILE"
echo "192.168.205.202 slots=8" >> "$HOSTFILE"

# 定义实验配置: NAME:TP:PP:DP:FREQ:ITERS
EXPERIMENTS=(
    # TP2PP2DP4 - 方差大的拓扑，重点补测
    "sweep_tp2pp2dp4_f1072_r1:2:2:4:1072:20"
    "sweep_tp2pp2dp4_f1072_r2:2:2:4:1072:20"
    "sweep_tp2pp2dp4_f1072_r3:2:2:4:1072:20"
    "sweep_tp2pp2dp4_f1080_r1:2:2:4:1080:20"
    "sweep_tp2pp2dp4_f1080_r2:2:2:4:1080:20"
    "sweep_tp2pp2dp4_f1080_r3:2:2:4:1080:20"
    "sweep_tp2pp2dp4_f1185_r1:2:2:4:1185:20"
    "sweep_tp2pp2dp4_f1185_r2:2:2:4:1185:20"
    "sweep_tp2pp2dp4_f1185_r3:2:2:4:1185:20"
    "sweep_tp2pp2dp4_f1200_r1:2:2:4:1200:20"
    "sweep_tp2pp2dp4_f1200_r2:2:2:4:1200:20"
    "sweep_tp2pp2dp4_f1200_r3:2:2:4:1200:20"
)

echo "=========================================="
echo "批量拓扑补测开始"
echo "总实验数: ${#EXPERIMENTS[@]}"
echo "开始时间: $(date)"
echo "=========================================="

SUCCESS=0
FAILED=0

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    IFS=':' read -r NAME TP PP DP FREQ ITERS <<< "$exp"
    
    echo ""
    echo "[$((i+1))/${#EXPERIMENTS[@]}] 实验: $NAME"
    echo "  拓扑: TP=$TP, PP=$PP, DP=$DP, 频率=${FREQ}MHz, 迭代=$ITERS"
    
    # DeepSpeed 配置
    DS_CONFIG="/tmp/ds_config_${NAME}.json"
    cat > "$DS_CONFIG" << EOF
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 1,
  "steps_per_print": 1,
  "zero_optimization": {"stage": 1},
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0
}
EOF
    
    # 启动实验
    if timeout 600 /home/sd/.local/bin/deepspeed \
      --hostfile "$HOSTFILE" --num_nodes 2 --num_gpus 8 \
      --master_addr 192.168.205.201 --master_port 29500 \
      pretrain_gpt.py \
      --tensor-model-parallel-size $TP --pipeline-model-parallel-size $PP \
      --num-layers 28 --hidden-size 3584 --ffn-hidden-size 18944 \
      --num-attention-heads 28 --num-key-value-heads 4 \
      --micro-batch-size 1 --global-batch-size 16 --seq-length 2048 \
      --max-position-embeddings 2048 --train-iters $ITERS \
      --data-path /home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document \
      --data-impl mmap --tokenizer-type HFTokenizer \
      --tokenizer-model /home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat \
      --split 98,2,0 --distributed-backend nccl --lr 1e-5 \
      --lr-decay-style cosine --min-lr 1e-6 --weight-decay 0.01 \
      --clip-grad 1.0 --lr-warmup-iters $((ITERS-1)) --optimizer adam \
      --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
      --log-interval 1 --save-interval 0 --eval-interval 100 --eval-iters 10 \
      --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 \
      --use-rotary-position-embeddings --untie-embeddings-and-output-weights \
      --swiglu --normalization rmsnorm --disable-bias-linear \
      --no-position-embedding --no-masked-softmax-fusion \
      --no-bias-gelu-fusion --no-bias-dropout-fusion \
      --recompute-granularity full --recompute-method uniform \
      --deepspeed-activation-checkpointing --zero-stage=1 \
      --deepspeed_config "$DS_CONFIG" --deepspeed --bf16 \
      --experiment-name "$NAME" \
      --experiment-root-dir /home/sd/Megatron-DeepSpeed/experiments \
      2>&1 | tee "/home/sd/Megatron-DeepSpeed/logs/${NAME}.log" | tail -10; then
        echo "  ✓ 成功"
        ((SUCCESS++))
    else
        echo "  ✗ 失败或超时"
        ((FAILED++))
    fi
    
    rm -f "$DS_CONFIG"
    sleep 5
done

echo ""
echo "=========================================="
echo "批量补测完成"
echo "成功: $SUCCESS, 失败: $FAILED"
echo "结束时间: $(date)"
echo "=========================================="
