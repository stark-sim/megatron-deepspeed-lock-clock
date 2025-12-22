#!/bin/bash
# Qwen2.5-7B Single Node Training with Freq Scaling
# TP=4, PP=1, DP=4 on 1 node (16 GPUs)
set -ex

BASE_PATH=/home/sd/Megatron-DeepSpeed
DS_CONFIG=${BASE_PATH}/scripts/ds_config_7b_tp4.json
DATASET="${BASE_PATH}/data/chinese_wiki_megatron_text_document"
CHECKPOINT_PATH=${BASE_PATH}/checkpoints/qwen7b_tp4_baseline
TOKENIZER_PATH=/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9

# 单节点配置: TP=4, PP=1, DP=4
TP=4
PP=1
ZERO_STAGE=1

GPUS_PER_NODE=16
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500

# Qwen2.5-7B 模型架构
# NUM_HEADS=28, NUM_KV_HEADS=4, 需要 TP 能整除
HIDDEN_SIZE=3584
FFN_HIDDEN_SIZE=18944
NUM_LAYERS=28
NUM_HEADS=28
NUM_KV_HEADS=4
SEQ_LENGTH=2048

# 训练参数
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TRAIN_STEPS=100
LR=1e-5
MIN_LR=1e-6

mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${BASE_PATH}/logs

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
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "wall_clock_breakdown": false
}
EOT

WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
echo "Single node training: TP=$TP, PP=$PP, DP=$((WORLD_SIZE / TP / PP))"

/home/sd/.local/bin/torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
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
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --lr-warmup-iters 50 \
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
     \
     \
    --recompute-granularity full \
    --recompute-method uniform \
    --deepspeed-activation-checkpointing \
    --zero-stage=${ZERO_STAGE} \
    --deepspeed_config=${DS_CONFIG} \
    --deepspeed \
    2>&1 | tee ${BASE_PATH}/logs/qwen7b_tp4_baseline_baseline_$(date +%Y%m%d_%H%M%S).log
