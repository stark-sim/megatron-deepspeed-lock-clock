#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LAUNCH_SCRIPT="${BASE_PATH}/scripts/run_v100_llama7b_tp1pp2dp8_16card.sh"

if [[ "${HOSTNAME%%.*}" != "v100x16-1" && "${HOSTNAME%%.*}" != "DGX2-1" ]]; then
    echo "[Error] Run this script from DGX2-1 / v100x16-1." >&2
    exit 1
fi

source "${BASE_PATH}/scripts/experiment_utils.sh" 2>/dev/null || true

# ========== 可调参数 ==========
export TRAIN_STEPS="${TRAIN_STEPS:-20}"
FREQS="${FREQS:-1260}"          # 多个频点用空格分隔，如 "1260 1350 1455"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-31531}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-v100_llama7b_16card}"
# =============================

LOG_DIR="${BASE_PATH}/temp"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_PREFIX}_tp1pp2_compare_${TIMESTAMP}.log"

cat <<BANNER | tee -a "${LOG_FILE}"
==================================================================
${EXPERIMENT_PREFIX} TP=1/PP=2/DP=8 Baseline vs Static Compare
Steps=${TRAIN_STEPS}  Freqs=[${FREQS}]  Timestamp=${TIMESTAMP}
==================================================================
BANNER

reset_clocks() {
    echo "[$(date +%H:%M:%S)] Resetting GPU clocks..." | tee -a "${LOG_FILE}"
    sudo -n nvidia-smi -rgc 2>/dev/null || true
    sleep 5
}

run_one() {
    local mode="$1"
    local clock="$2"
    local port="$3"
    reset_clocks
    if [[ "${mode}" == "baseline" ]]; then
        export EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_baseline${TRAIN_STEPS}"
        export EXPERIMENT_MODE="baseline"
        echo "[$(date +%H:%M:%S)] Running ${EXPERIMENT_NAME} (default clocks)..." | tee -a "${LOG_FILE}"
        MASTER_PORT="${port}" bash "${LAUNCH_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
    else
        export EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_static${clock}_${TRAIN_STEPS}"
        export EXPERIMENT_MODE="static"
        export STATIC_CLOCK_MHZ="${clock}"
        echo "[$(date +%H:%M:%S)] Running ${EXPERIMENT_NAME} (${clock} MHz)..." | tee -a "${LOG_FILE}"
        MASTER_PORT="${port}" bash "${LAUNCH_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
    fi
}

# 1) Baseline
run_one baseline "" "${MASTER_PORT_BASE}"

# 2) Static sweep
port="${MASTER_PORT_BASE}"
for clock in ${FREQS}; do
    port=$((port + 10))
    run_one static "${clock}" "${port}"
done

echo "[$(date +%H:%M:%S)] All runs completed." | tee -a "${LOG_FILE}"
