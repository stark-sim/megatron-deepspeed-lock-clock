#!/usr/bin/env bash
# Convert Qwen2.5-7B-Instruct HF weights to Megatron-DeepSpeed TP2PP4 format on V100
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${BASE_PATH}:${PYTHONPATH:-}"
TP=2; PP=4; NPROC_PER_NODE=$((TP * PP))
if [ "$#" -ge 2 ]; then GPU_START="${1}"; GPU_END="${2}"
else GPU_START=8; GPU_END=$((GPU_START + NPROC_PER_NODE - 1)); fi
HF_MODEL="${HF_MODEL:-/home/sd/models/Qwen2.5-7B-Instruct-full}"
NUM_LAYERS=28; HIDDEN_SIZE=3584; FFN_HIDDEN_SIZE=18944; NUM_HEADS=28; NUM_KV_HEADS=4; VOCAB_SIZE=152064; SEQ_LENGTH=2048
[ $((NUM_HEADS % TP)) -ne 0 ] && { echo "[Error] NUM_HEADS not divisible by TP"; exit 1; }
[ $((NUM_KV_HEADS % TP)) -ne 0 ] && { echo "[Error] NUM_KV_HEADS not divisible by TP"; exit 1; }
RUN_ID="qwen25_7b_instruct_hf2megads_tp${TP}pp${PP}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${BASE_PATH}/checkpoints/${RUN_ID}"; mkdir -p "${OUTPUT_PATH}"
echo "Converting Qwen7B TP=${TP} PP=${PP} -> ${OUTPUT_PATH} GPUs=${GPU_START}-${GPU_END}"
NUM_SHARDS=$(ls "${HF_MODEL}"/model-*.safetensors 2>/dev/null | wc -l); [ "${NUM_SHARDS}" -eq 0 ] && NUM_SHARDS=1
CUDA_VISIBLE_DEVICES=$(seq -s, ${GPU_START} ${GPU_END}) CUDA_DEVICE_MAX_CONNECTIONS=1 python3 -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29507 \
    "${BASE_PATH}/tools/hf2megads_weight_converter.py" \
    --hf-ckpt-dir "${HF_MODEL}" --hf-ckpt-num-shards "${NUM_SHARDS}" --load-mode safetensor \
    --num-layers "${NUM_LAYERS}" --hidden-size "${HIDDEN_SIZE}" --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
    --num-attention-heads "${NUM_HEADS}" --num-key-value-heads "${NUM_KV_HEADS}" --seq-length "${SEQ_LENGTH}" \
    --max-position-embeddings "${SEQ_LENGTH}" --vocab-size "${VOCAB_SIZE}" --make-vocab-size-divisible-by 128 \
    --tensor-model-parallel-size "${TP}" --pipeline-model-parallel-size "${PP}" --micro-batch-size 1 --global-batch-size 8 \
    --train-iters 1 --lr-decay-iters 1 --lr 1e-5 --min-lr 1e-6 --lr-warmup-fraction 0.01 --log-interval 1 \
    --save-interval 1 --eval-interval 1000 --eval-iters 0 --fp16 --swiglu --normalization rmsnorm \
    --use-rotary-position-embeddings --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion --tokenizer-type HFTokenizer --tokenizer-model "${HF_MODEL}" --deepspeed \
    --deepspeed_config examples_deepspeed/finetune_hf_llama/ds_config_empty.json --save "${OUTPUT_PATH}" \
    --distributed-backend nccl --no-load-optim --no-load-rng --no-save-optim --no-save-rng --disable-bias-linear --exit-on-missing-checkpoint
echo "Done: ${OUTPUT_PATH}"
