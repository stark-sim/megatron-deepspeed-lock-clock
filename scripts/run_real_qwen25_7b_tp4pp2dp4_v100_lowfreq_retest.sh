#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LAUNCH_SCRIPT="${BASE_PATH}/scripts/run_real_qwen25_7b_tp4pp2dp4_v100.sh"

if [[ ! -x "${LAUNCH_SCRIPT}" ]]; then
    echo "[Error] Missing executable launcher: ${LAUNCH_SCRIPT}" >&2
    exit 1
fi

TRAIN_STEPS="${TRAIN_STEPS:-20}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-32101}"
RUN_TAG="${RUN_TAG:-lowfreq_retest_formal20_finetune_nosave}"

run_one() {
    local mode="$1"
    local clock="$2"
    local port="$3"
    local experiment_name

    if [[ "${mode}" == "baseline" ]]; then
        experiment_name="ib_real_qwen25_7b_tp4pp2dp4_full32_baseline_${RUN_TAG}"
        MASTER_PORT="${port}" \
        EXPERIMENT_NAME="${experiment_name}" \
        EXPERIMENT_MODE="baseline" \
        TRAIN_STEPS="${TRAIN_STEPS}" \
        bash "${LAUNCH_SCRIPT}"
        return
    fi

    experiment_name="ib_real_qwen25_7b_tp4pp2dp4_full32_static${clock}_${RUN_TAG}"
    MASTER_PORT="${port}" \
    EXPERIMENT_NAME="${experiment_name}" \
    EXPERIMENT_MODE="static" \
    STATIC_CLOCK_MHZ="${clock}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    bash "${LAUNCH_SCRIPT}"
}

run_one baseline "" "${MASTER_PORT_BASE}"
sleep 10
run_one static "1200" "$((MASTER_PORT_BASE + 10))"
sleep 10
run_one static "1155" "$((MASTER_PORT_BASE + 20))"
sleep 10
run_one static "1080" "$((MASTER_PORT_BASE + 30))"
sleep 10
run_one static "990" "$((MASTER_PORT_BASE + 40))"
