#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LAUNCH_SCRIPT="${BASE_PATH}/scripts/run_real_qwen25_7b_tp4pp2dp4_gbs32_v100.sh"

echo "========================================"
echo "TP4PP2DP4 GBS=32 32-card high-frequency sweep"
echo "Frequencies: baseline + 1350/1455/1530"
echo "========================================"

TRAIN_STEPS="${TRAIN_STEPS:-20}"
MASTER_PORT_BASE=32631
RUN_TAG="${RUN_TAG:-highfreq_formal20_finetune_nosave}"

run_one() {
    local mode="$1"
    local clock="$2"
    local port="$3"
    local experiment_name

    if [[ "${mode}" == "baseline" ]]; then
        experiment_name="ib_real_qwen25_7b_tp4pp2dp4_full32_gbs32_baseline_${RUN_TAG}"
        MASTER_PORT="${port}" \
        EXPERIMENT_NAME="${experiment_name}" \
        EXPERIMENT_MODE="baseline" \
        TRAIN_STEPS="${TRAIN_STEPS}" \
        bash "${LAUNCH_SCRIPT}" 2>&1 | tee "/tmp/qwen7b_tp4pp2dp4_gbs32_highfreq_baseline_$(date +%Y%m%d_%H%M%S).log"
        return
    fi

    experiment_name="ib_real_qwen25_7b_tp4pp2dp4_full32_gbs32_static${clock}_${RUN_TAG}"
    MASTER_PORT="${port}" \
    EXPERIMENT_NAME="${experiment_name}" \
    EXPERIMENT_MODE="static" \
    STATIC_CLOCK_MHZ="${clock}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    bash "${LAUNCH_SCRIPT}" 2>&1 | tee "/tmp/qwen7b_tp4pp2dp4_gbs32_highfreq_static${clock}_$(date +%Y%m%d_%H%M%S).log"
}

run_one baseline "" "${MASTER_PORT_BASE}"
sleep 60
run_one static "1350" "$((MASTER_PORT_BASE + 10))"
sleep 60
run_one static "1455" "$((MASTER_PORT_BASE + 20))"
sleep 60
run_one static "1530" "$((MASTER_PORT_BASE + 30))"

echo "========================================"
echo "All frequencies completed."
echo "========================================"
