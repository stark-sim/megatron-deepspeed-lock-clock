                                                                                #!/bin/bash
# Qwen2.5-14B 3D Parallel Training Script
# TP=8, PP=2, DP=2 on 2 nodes (32 GPUs total)
set -ex

######################################
# 基础路径配置
BASE_PATH=/home/sd/Megatron-DeepSpeed
DS_CONFIG=${BASE_PATH}/scripts/ds_config_3d.json
DATASET="${BASE_PATH}/data/chinese_wiki_megatron_text_document"
CHECKPOINT_PATH=${BASE_PATH}/checkpoints/qwen14b_3d
TOKENIZER_PATH=/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9

# 3D并行配置
TP=8    # 张量并行度
PP=2    # 流水线并行度
DP=2    # 数据并行度 (自动计算: WORLD_SIZE / TP / PP)
ZERO_STAGE=1  # ZeRO优化阶段 (PP>1时建议用0或1)

# 集群配置
GPUS_PER_NODE=16
MASTER_ADDR=${MASTER_ADDR:-100.64.0.90}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}

# Qwen2.5-14B 模型架构参数
HIDDEN_SIZE=5120
FFN_HIDDEN_SIZE=13824  # intermediate_size
NUM_LAYERS=48
NUM_HEADS=40
NUM_KV_HEADS=8  # GQA: num_key_value_heads
SEQ_LENGTH=2048
VOCAB_SIZE=152064

# 训练超参数
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32
TRAIN_STEPS=1000
LR=1e-5
MIN_LR=1e-6
LR_WARMUP_STEPS=100
WEIGHT_DECAY=0.01
GRAD_CLIP=1.0

# 激活检查点 (节省显存)
activation_checkpoint="true"

######################################
# 创建输出目录
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${BASE_PATH}/logs

# 生成DeepSpeed配置
cat <<EOT > $DS_CONFIG
{
  "train_batch_size": ${GLOBAL_BATCH_SIZE},
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": ${ZERO_STAGE}
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": ${GRAD_CLIP},
  "prescale_gradients": false,
  "wall_clock_breakdown": false
}
EOT

# DeepSpeed参数
ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=${DS_CONFIG} ${ds_args}"
ds_args=" --zero-stage=${ZERO_STAGE} ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
  ds_args="--deepspeed-activation-checkpointing ${ds_args}"
  ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
fi

# 分布式启动参数
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

echo "=========================================="
echo "3D Parallel Configuration:"
echo "  TP=${TP}, PP=${PP}, DP=${DP}"
echo "  Total GPUs: $((GPUS_PER_NODE * NNODES))"
echo "  Nodes: ${NNODES}, GPUs per node: ${GPUS_PER_NODE}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Node Rank: ${NODE_RANK}"
echo "=========================================="

# NCCL环境变量 - 增加超时和调试
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=tailscale0
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# 启动训练
/home/sd/.local/bin/torchrun $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --num-key-value-heads $NUM_KV_HEADS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --train-iters $TRAIN_STEPS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATASET \
    --data-impl mmap \
    --tokenizer-type HFTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --split 98,2,0 \
    --distributed-backend nccl \
    --lr $LR \
    --lr-decay-style cosine \
    --min-lr $MIN_LR \
    --weight-decay $WEIGHT_DECAY \
    --clip-grad $GRAD_CLIP \
    --lr-warmup-iters $LR_WARMUP_STEPS \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    --bf16 \
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
    --enable-comm-freq-scaling \
    --comm-low-freq 800 \
    $ds_args \
    2>&1 | tee ${BASE_PATH}/logs/train_freq_scaling_$(date +%Y%m%d_%H%M%S).log
