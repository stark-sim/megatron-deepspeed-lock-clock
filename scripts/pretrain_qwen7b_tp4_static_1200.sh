#!/bin/bash
# Qwen2.5-7B Single Node Training with STATIC 1200MHz
# 全程锁定 1200MHz，不进行动态调频
set -ex

BASE_PATH=/home/sd/Megatron-DeepSpeed
DS_CONFIG=${BASE_PATH}/scripts/ds_config_7b_tp4.json
DATASET="${BASE_PATH}/data/chinese_wiki_megatron_text_document"
CHECKPOINT_PATH=${BASE_PATH}/checkpoints/qwen7b_tp4_static
TOKENIZER_PATH=/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9

TP=4
PP=1
ZERO_STAGE=1

GPUS_PER_NODE=16
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500

HIDDEN_SIZE=3584
FFN_HIDDEN_SIZE=18944
NUM_LAYERS=28
NUM_HEADS=28
NUM_KV_HEADS=4
SEQ_LENGTH=2048

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TRAIN_STEPS=500
LR=1e-5
MIN_LR=1e-6

mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${BASE_PATH}/logs

# 训练开始前锁定所有 GPU 到 1200MHz
echo "Locking all GPUs to 1200MHz..."
python3 -c "
import sys
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml
pynvml.nvmlInit()
for i in range(16):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    pynvml.nvmlDeviceSetGpuLockedClocks(h, 1200, 1200)
    print(f'GPU {i} locked to 1200MHz')
pynvml.nvmlShutdown()
"

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
echo "Single node training (STATIC 1200MHz): TP=$TP, PP=$PP, DP=$((WORLD_SIZE / TP / PP))"

# 运行训练（不启用动态调频）
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
    --recompute-granularity full \
    --recompute-method uniform \
    --deepspeed-activation-checkpointing \
    --zero-stage=${ZERO_STAGE} \
    --deepspeed_config=${DS_CONFIG} \
    --deepspeed

# 训练结束后恢复默认频率
echo "Resetting GPU frequencies to default..."
python3 -c "
import sys
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml
pynvml.nvmlInit()
for i in range(16):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    pynvml.nvmlDeviceResetGpuLockedClocks(h)
    pynvml.nvmlDeviceResetApplicationsClocks(h)
    print(f'GPU {i} reset to default')
pynvml.nvmlShutdown()
"
echo "Done!"
