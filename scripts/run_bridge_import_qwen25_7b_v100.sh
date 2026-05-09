#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"

MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-}"
if [[ -z "${MEGATRON_BRIDGE_ROOT}" ]]; then
    echo "[Error] MEGATRON_BRIDGE_ROOT is required." >&2
    echo "Example: export MEGATRON_BRIDGE_ROOT=/home/sd/Megatron-Bridge" >&2
    exit 1
fi

HF_MODEL="${HF_MODEL:-/home/sd/models/Qwen2.5-7B-Instruct-full}"
TP="${TP:-2}"
PP="${PP:-2}"
VP="${VP:-}"
CP="${CP:-}"
LOCAL_GPU_INDICES="${LOCAL_GPU_INDICES:-8,9,10,11}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$((TP * PP))}"
MASTER_PORT="${MASTER_PORT:-31441}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
RUN_ID="${RUN_ID:-qwen25_7b_instruct_bridge_tp${TP}pp${PP}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-${BASE_PATH}/checkpoints/${RUN_ID}}"

CMD=(
    torchrun
    --nproc-per-node "${NPROC_PER_NODE}"
    --master-port "${MASTER_PORT}"
    "${BASE_PATH}/scripts/convert_hf_to_megatron_bridge.py"
    --bridge-root "${MEGATRON_BRIDGE_ROOT}"
    --hf-model "${HF_MODEL}"
    --megatron-path "${OUTPUT_PATH}"
    --tp "${TP}"
    --pp "${PP}"
    --torch-dtype "${TORCH_DTYPE}"
)

if [[ -n "${VP}" ]]; then
    CMD+=(--vp "${VP}")
fi
if [[ -n "${CP}" ]]; then
    CMD+=(--cp "${CP}")
fi

echo "[Bridge] HF_MODEL=${HF_MODEL}"
echo "[Bridge] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[Bridge] TP=${TP} PP=${PP} NPROC=${NPROC_PER_NODE}"

CUDA_VISIBLE_DEVICES="${LOCAL_GPU_INDICES}" "${CMD[@]}"
