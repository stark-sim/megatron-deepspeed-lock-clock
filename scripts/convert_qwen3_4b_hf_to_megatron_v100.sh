#!/usr/bin/env bash
# Convert Qwen3-4B HF weights to Megatron-DeepSpeed format on V100
# Usage: bash scripts/convert_qwen3_4b_hf_to_megatron_v100.sh [TP] [PP] [GPU_START] [GPU_END]
#   TP: tensor parallel size (default: 2)
#   PP: pipeline parallel size (default: 2)
#   GPU_START/GPU_END: GPU range for conversion (default: 0-3 for TP2PP2, 0-7 for TP4PP2)
#
# Example:
#   bash scripts/convert_qwen3_4b_hf_to_megatron_v100.sh 2 2   # TP2PP2 on GPU 0-3
#   bash scripts/convert_qwen3_4b_hf_to_megatron_v100.sh 4 2   # TP4PP2 on GPU 0-7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${BASE_PATH}:${PYTHONPATH:-}"

TP="${1:-2}"
PP="${2:-2}"

# Auto-determine GPU range if not provided
NPROC_PER_NODE=$((TP * PP))
if [ "$#" -ge 4 ]; then
    GPU_START="${3}"
    GPU_END="${4}"
else
    GPU_START=0
    GPU_END=$((NPROC_PER_NODE - 1))
fi

HF_MODEL="${HF_MODEL:-/home/sd/models/Qwen3-4B}"

# Qwen3-4B architecture
NUM_LAYERS=36
HIDDEN_SIZE=2560
FFN_HIDDEN_SIZE=9728
NUM_HEADS=32
NUM_KV_HEADS=8
VOCAB_SIZE=151936
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

RUN_ID="qwen3_4b_hf2megads_tp${TP}pp${PP}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${BASE_PATH}/checkpoints/${RUN_ID}"
mkdir -p "${OUTPUT_PATH}"

echo "========================================"
echo "HF → Megatron-DS Weight Conversion"
echo "Model: Qwen3-4B"
echo "HF path: ${HF_MODEL}"
echo "TP=${TP}, PP=${PP}"
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
    --master_port=29506 \
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
echo "========================================"
