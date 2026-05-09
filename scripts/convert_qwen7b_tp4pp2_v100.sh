#!/usr/bin/env bash
# Re-convert Qwen2.5-7B-Instruct HF weights to Megatron-DeepSpeed TP4PP2 format on V100
# This fixes the previous shape mismatch when loading TP2PP2 checkpoint with TP4PP2 runtime.
# Usage: bash scripts/convert_qwen7b_tp4pp2_v100.sh [GPU_START] [GPU_END]
#   GPU_START/GPU_END: GPU range for conversion (default: 0-7 for TP4PP2=8 processes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${BASE_PATH}:${PYTHONPATH:-}"

TP=4
PP=2
NPROC_PER_NODE=$((TP * PP))

if [ "$#" -ge 2 ]; then
    GPU_START="${1}"
    GPU_END="${2}"
else
    GPU_START=0
    GPU_END=$((NPROC_PER_NODE - 1))
fi

HF_MODEL="${HF_MODEL:-/home/sd/models/Qwen2.5-7B-Instruct-full}"

# Qwen2.5-7B architecture
NUM_LAYERS=28
HIDDEN_SIZE=3584
FFN_HIDDEN_SIZE=18944
NUM_HEADS=28
NUM_KV_HEADS=4
VOCAB_SIZE=152064
SEQ_LENGTH=2048

# Validate TP divides heads and kv_heads
if [ $((NUM_HEADS % TP)) -ne 0 ]; then
    echo "[Error] NUM_HEADS (${NUM_HEADS}) must be divisible by TP (${TP})" >&2
    exit 1
fi
if [ $((NUM_KV_HEADS % TP)) -ne 0 ]; then
    echo "[Error] NUM_KV_HEADS (${NUM_KV_HEADS}) must be divisible by TP (${TP})" >&2
    exit 1
fi

RUN_ID="qwen25_7b_instruct_hf2megads_tp${TP}pp${PP}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${BASE_PATH}/checkpoints/${RUN_ID}"
mkdir -p "${OUTPUT_PATH}"

echo "========================================"
echo "HF → Megatron-DS Weight Conversion"
echo "Model: Qwen2.5-7B-Instruct"
echo "HF path: ${HF_MODEL}"
echo "TP=${TP}, PP=${PP} (fixed)"
echo "Output: ${OUTPUT_PATH}"
echo "GPUs: ${GPU_START}-${GPU_END}"
echo "========================================"

# Determine number of safetensor shards
NUM_SHARDS=$(ls "${HF_MODEL}"/model-*.safetensors 2>/dev/null | wc -l)
if [ "${NUM_SHARDS}" -eq 0 ]; then
    NUM_SHARDS=1
fi
echo "Detected ${NUM_SHARDS} safetensor shard(s)"

# Run conversion
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29507 \
    "${BASE_PATH}/tools/hf2megads_weight_converter.py" \
    --hf-ckpt-dir "${HF_MODEL}" \
    --hf-ckpt-num-shards "${NUM_SHARDS}" \
    --load-mode safetensor \
    --num-layers "${NUM_LAYERS}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
    --num-attention-heads "${NUM_HEADS}" \
    --num-key-value-heads "${NUM_KV_HEADS}" \
    --seq-length "${SEQ_LENGTH}" \
    --max-position-embeddings "${SEQ_LENGTH}" \
    --vocab-size "${VOCAB_SIZE}" \
    --make-vocab-size-divisible-by 128 \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 1 \
    --lr-decay-iters 1 \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --lr-warmup-fraction 0.01 \
    --log-interval 1 \
    --save-interval 1 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --fp16 \
    --swiglu \
    --normalization rmsnorm \
    --use-rotary-position-embeddings \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --tokenizer-type HFTokenizer \
    --tokenizer-model "${HF_MODEL}" \
    --deepspeed \
    --deepspeed_config examples_deepspeed/finetune_hf_llama/ds_config_empty.json \
    --save "${OUTPUT_PATH}" \
    --distributed-backend nccl \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --no-one-logger \
    --exit-on-missing-checkpoint \
    --use-mcore-models

echo "========================================"
echo "Conversion complete!"
echo "Checkpoint saved to: ${OUTPUT_PATH}"
echo ""
echo "To use this checkpoint in training, set:"
echo "  LOAD_CHECKPOINT_PATH=${OUTPUT_PATH}"
echo ""
echo "Note: Update run_real_qwen25_7b_tp4pp2dp{1,2}_v100.sh"
echo "      to point LOAD_CHECKPOINT_PATH to this new checkpoint."
echo "========================================"
