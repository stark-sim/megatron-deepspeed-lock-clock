#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LAUNCH_SCRIPT="${BASE_PATH}/scripts/run_real_qwen25_7b_tp4pp2dp1_v100.sh"

echo "========================================"
echo "TP4PP2DP1 8-card low-frequency sweep"
echo "Frequencies: baseline + 990/1080/1155/1200"
echo "========================================"

TRAIN_STEPS="${TRAIN_STEPS:-20}"
MASTER_PORT_BASE=32231
RUN_TAG="${RUN_TAG:-lowfreq_formal20_finetune_nosave}"

run_one() {
    local mode="$1"
    local clock="$2"
    local port="$3"
    local experiment_name

    if [[ "${mode}" == "baseline" ]]; then
        experiment_name="ib_real_qwen25_7b_tp4pp2dp1_baseline_${RUN_TAG}"
        MASTER_PORT="${port}" \
        EXPERIMENT_NAME="${experiment_name}" \
        EXPERIMENT_MODE="baseline" \
        TRAIN_STEPS="${TRAIN_STEPS}" \
        bash "${LAUNCH_SCRIPT}" 2>&1 | tee "/tmp/qwen7b_tp4pp2dp1_lowfreq_baseline_$(date +%Y%m%d_%H%M%S).log"
        return
    fi

    experiment_name="ib_real_qwen25_7b_tp4pp2dp1_static${clock}_${RUN_TAG}"
    MASTER_PORT="${port}" \
    EXPERIMENT_NAME="${experiment_name}" \
    EXPERIMENT_MODE="static" \
    STATIC_CLOCK_MHZ="${clock}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    bash "${LAUNCH_SCRIPT}" 2>&1 | tee "/tmp/qwen7b_tp4pp2dp1_lowfreq_static${clock}_$(date +%Y%m%d_%H%M%S).log"
}

run_one baseline "" "${MASTER_PORT_BASE}"
sleep 60
run_one static "1200" "$((MASTER_PORT_BASE + 10))"
sleep 60
run_one static "1155" "$((MASTER_PORT_BASE + 20))"
sleep 60
run_one static "1080" "$((MASTER_PORT_BASE + 30))"
sleep 60
run_one static "990" "$((MASTER_PORT_BASE + 40))"

echo "========================================"
echo "All frequencies completed."
echo "========================================"
