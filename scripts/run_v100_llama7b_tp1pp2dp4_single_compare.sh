#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

FREQS="${FREQS:-1260 1350 1455 1530}"
STEPS="${STEPS:-20}"
CKPT_PATH="${CKPT_PATH:-${BASE_PATH}/checkpoints/llama7b_hf2megads_tp1pp2_latest}"

if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "[WARN] Checkpoint not found: ${CKPT_PATH}" >&2
    echo "[WARN] Run: bash scripts/convert_llama7b_hf_to_megatron_v100.sh 1 2" >&2
fi

BASE_PORT=31831

echo "========================================"
echo "LLaMA7B Single-Node TP1PP2DP4 Baseline + Static Sweep"
echo "Checkpoint: ${CKPT_PATH}"
echo "Freqs: ${FREQS} | Steps: ${STEPS}"
echo "========================================"

i=1
total=$(( $(echo $FREQS | wc -w) + 1 ))

EXPERIMENT_NAME="v100_llama7b_tp1pp2dp4_single_baseline_formal${STEPS}_finetune_nosave" \
EXPERIMENT_MODE=baseline \
MASTER_PORT="${BASE_PORT}" \
TRAIN_STEPS="${STEPS}" \
LOAD_CHECKPOINT_PATH="${CKPT_PATH}" \
bash "${SCRIPT_DIR}/run_v100_llama7b_tp1pp2dp4_single.sh"

i=2
for freq in ${FREQS}; do
    EXPERIMENT_NAME="v100_llama7b_tp1pp2dp4_single_static${freq}_formal${STEPS}_finetune_nosave" \
    EXPERIMENT_MODE=static \
    STATIC_CLOCK_MHZ="${freq}" \
    MASTER_PORT=$((BASE_PORT + i - 1)) \
    TRAIN_STEPS="${STEPS}" \
    LOAD_CHECKPOINT_PATH="${CKPT_PATH}" \
    bash "${SCRIPT_DIR}/run_v100_llama7b_tp1pp2dp4_single.sh"
    i=$((i + 1))
done

echo "========================================"
echo "Sweep complete!"
echo "========================================"
