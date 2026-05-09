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
CKPT_PATH="${CKPT_PATH:-${BASE_PATH}/checkpoints/llama7b_hf2megads_tp1pp4_latest}"

if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "[WARN] Checkpoint not found: ${CKPT_PATH}" >&2
    echo "[WARN] Run: bash scripts/convert_llama7b_hf_to_megatron_v100.sh 1 4" >&2
fi

BASE_PORT=32131

echo "========================================"
echo "LLaMA7B Dual-Node TP1PP4DP2 Baseline + Static Sweep"
echo "Checkpoint: ${CKPT_PATH}"
echo "Freqs: ${FREQS} | Steps: ${STEPS}"
echo "========================================"

EXPERIMENT_NAME="ib_llama7b_tp1pp4dp2_baseline_formal${STEPS}_finetune_nosave" \
EXPERIMENT_MODE=baseline \
MASTER_PORT="${BASE_PORT}" \
TRAIN_STEPS="${STEPS}" \
LOAD_CHECKPOINT_PATH="${CKPT_PATH}" \
bash "${SCRIPT_DIR}/run_v100_llama7b_tp1pp4dp2.sh"

i=2
for freq in ${FREQS}; do
    EXPERIMENT_NAME="ib_llama7b_tp1pp4dp2_static${freq}_formal${STEPS}_finetune_nosave" \
    EXPERIMENT_MODE=static \
    STATIC_CLOCK_MHZ="${freq}" \
    MASTER_PORT=$((BASE_PORT + i - 1)) \
    TRAIN_STEPS="${STEPS}" \
    LOAD_CHECKPOINT_PATH="${CKPT_PATH}" \
    bash "${SCRIPT_DIR}/run_v100_llama7b_tp1pp4dp2.sh"
    i=$((i + 1))
done

echo "========================================"
echo "Sweep complete!"
echo "========================================"
