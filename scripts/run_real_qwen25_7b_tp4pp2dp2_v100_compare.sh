#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ "${HOSTNAME%%.*}" != "v100x16-1" && "${HOSTNAME%%.*}" != "DGX2-1" ]]; then
    echo "[Error] Run this script from DGX2-1 / v100x16-1." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Configurable sweep parameters
# ---------------------------------------------------------------------------
FREQS="${FREQS:-1260 1350 1455 1530}"
STEPS="${STEPS:-20}"
CKPT_PATH="${CKPT_PATH:-${BASE_PATH}/checkpoints/qwen25_7b_instruct_hf2megads_tp4pp2_20260506_214108}"

# Verify checkpoint exists
if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "[WARN] Checkpoint not found: ${CKPT_PATH}" >&2
    echo "[WARN] Please run scripts/convert_qwen7b_tp4pp2_v100.sh first." >&2
fi

BASE_PORT=31731

echo "========================================"
echo "Qwen2.5-7B TP4PP2DP2 Baseline + Static Sweep"
echo "Checkpoint: ${CKPT_PATH}"
echo "Freqs: ${FREQS}"
echo "Steps: ${STEPS}"
echo "GPUs: 16 (8 per node)"
echo "========================================"

# --- Baseline ---
echo ""
echo ">>> [1/$(( $(echo $FREQS | wc -w) + 1 ))] Running BASELINE ..."
EXPERIMENT_NAME="ib_real_qwen25_7b_tp4pp2dp2_baseline_formal${STEPS}_finetune_nosave" \
EXPERIMENT_MODE=baseline \
MASTER_PORT="${BASE_PORT}" \
TRAIN_STEPS="${STEPS}" \
LOAD_CHECKPOINT_PATH="${CKPT_PATH}" \
bash "${SCRIPT_DIR}/run_real_qwen25_7b_tp4pp2dp2_v100.sh"

# --- Static sweep ---
i=2
for freq in ${FREQS}; do
    echo ""
    echo ">>> [${i}/$(( $(echo $FREQS | wc -w) + 1 ))] Running STATIC ${freq} MHz ..."
    EXPERIMENT_NAME="ib_real_qwen25_7b_tp4pp2dp2_static${freq}_formal${STEPS}_finetune_nosave" \
    EXPERIMENT_MODE=static \
    STATIC_CLOCK_MHZ="${freq}" \
    MASTER_PORT=$((BASE_PORT + i)) \
    TRAIN_STEPS="${STEPS}" \
    LOAD_CHECKPOINT_PATH="${CKPT_PATH}" \
    bash "${SCRIPT_DIR}/run_real_qwen25_7b_tp4pp2dp2_v100.sh"
    i=$((i + 1))
done

echo ""
echo "========================================"
echo "Sweep complete!"
echo "========================================"
