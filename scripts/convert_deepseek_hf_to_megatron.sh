#!/usr/bin/env bash
# Convert DeepSeek-R1-Distill-Qwen-7B HF weights to Megatron-DeepSpeed format
# Usage: bash scripts/convert_deepseek_hf_to_megatron.sh [TP] [PP]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${BASE_PATH}:${PYTHONPATH:-}"

TP="${1:-2}"
PP="${2:-2}"
HF_MODEL="${HF_MODEL:-/home/sd/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"

# Model params matching DeepSeek-R1-Distill-Qwen-7B / Qwen2.5-7B
NUM_LAYERS=28
HIDDEN_SIZE=3584
FFN_HIDDEN_SIZE=18944
NUM_HEADS=28
NUM_KV_HEADS=4
VOCAB_SIZE=152064
SEQ_LENGTH=2048

# Output path
RUN_ID="deepseek_r1_qwen7b_hf2megads_tp${TP}pp${PP}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${BASE_PATH}/checkpoints/${RUN_ID}"
mkdir -p "${OUTPUT_PATH}"

echo "========================================"
echo "HF → Megatron-DS Weight Conversion"
echo "Model: ${HF_MODEL}"
echo "TP=${TP}, PP=${PP}"
echo "Output: ${OUTPUT_PATH}"
echo "========================================"

# Determine number of safetensor shards
NUM_SHARDS=$(ls "${HF_MODEL}"/model-*.safetensors 2>/dev/null | wc -l)
if [ "${NUM_SHARDS}" -eq 0 ]; then
    NUM_SHARDS=1
fi
echo "Detected ${NUM_SHARDS} safetensor shard(s)"

# Run conversion with torchrun
# Need TP*PP processes
NPROC_PER_NODE=$((TP * PP))

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29505 \
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
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --use-flash-attn-triton \
    --transformer-impl local \
    --tokenizer-type NullTokenizer \
    --tokenizer-model "" \
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
echo "========================================"
