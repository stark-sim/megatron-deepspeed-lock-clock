#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ "${HOSTNAME%%.*}" != "v100x16-1" && "${HOSTNAME%%.*}" != "DGX2-1" ]]; then
    echo "[Error] Run this script from DGX2-1 / v100x16-1." >&2
    exit 1
fi

FREQS="${FREQS:-1260 1350 1455 1530}"
STEPS="${STEPS:-20}"
CKPT_PATH="${CKPT_PATH:-}"

if [[ -z "${CKPT_PATH}" ]]; then
    CKPT_PATH=$(ls -dt "${BASE_PATH}"/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp4_* 2>/dev/null | head -1 || true)
fi

if [[ -z "${CKPT_PATH}" || ! -d "${CKPT_PATH}" ]]; then
    echo "[Error] No TP2PP4 checkpoint found. Please run convert_qwen7b_tp2pp4_v100.sh first." >&2
    exit 1
fi

BASE_PORT=32331
RUN_TAG="tp2pp4dp2_formal${STEPS}_finetune_nosave"

echo "========================================"
echo "Qwen2.5-7B TP2PP4DP2 Sweep"
echo "Checkpoint: ${CKPT_PATH}"
echo "Freqs: ${FREQS}"
echo "Steps: ${STEPS}"
echo "GPUs: 16 (8 per node)"
echo "========================================"

EXPERIMENT_NAME="ib_real_qwen25_7b_tp2pp4dp2_baseline_${RUN_TAG}" \
EXPERIMENT_MODE=baseline \
MASTER_PORT="${BASE_PORT}" \
TRAIN_STEPS="${STEPS}" \
LOAD_CHECKPOINT_PATH="${CKPT_PATH}" \
bash "${SCRIPT_DIR}/run_real_qwen25_7b_tp2pp4dp2_v100.sh"

i=2
for freq in ${FREQS}; do
    echo ""
    echo ">>> [${i}/$(( $(echo $FREQS | wc -w) + 1 ))] Running STATIC ${freq} MHz ..."
    EXPERIMENT_NAME="ib_real_qwen25_7b_tp2pp4dp2_static${freq}_${RUN_TAG}" \
    EXPERIMENT_MODE=static \
    STATIC_CLOCK_MHZ="${freq}" \
    MASTER_PORT=$((BASE_PORT + i * 10)) \
    TRAIN_STEPS="${STEPS}" \
    LOAD_CHECKPOINT_PATH="${CKPT_PATH}" \
    bash "${SCRIPT_DIR}/run_real_qwen25_7b_tp2pp4dp2_v100.sh"
    i=$((i + 1))
done

echo ""
echo "========================================"
echo "Sweep complete!"
echo "========================================"
