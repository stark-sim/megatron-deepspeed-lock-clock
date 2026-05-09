#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LAUNCH_SCRIPT="${BASE_PATH}/scripts/run_real_qwen25_7b_tp2pp2dp2_v100.sh"

if [[ ! -x "${LAUNCH_SCRIPT}" ]]; then
    echo "[Error] Missing executable launcher: ${LAUNCH_SCRIPT}" >&2
    exit 1
fi

TRAIN_STEPS="${TRAIN_STEPS:-20}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-31331}"
RUN_TAG="${RUN_TAG:-formal20_finetune_nosave}"
RUN_BASELINE="${RUN_BASELINE:-0}"
# Existing: baseline + static1500 done. Need: 1260/1350/1455/1530/1590
FREQS="${FREQS:-1260 1350 1455 1530 1590}"

run_one() {
    local mode="$1"
    local clock="$2"
    local port="$3"
    local experiment_name

    if [[ "${mode}" == "baseline" ]]; then
        experiment_name="ib_real_qwen25_7b_tp2pp2dp2_baseline_${RUN_TAG}"
        MASTER_PORT="${port}" \
        EXPERIMENT_NAME="${experiment_name}" \
        EXPERIMENT_MODE="baseline" \
        TRAIN_STEPS="${TRAIN_STEPS}" \
        bash "${LAUNCH_SCRIPT}"
        return
    fi

    experiment_name="ib_real_qwen25_7b_tp2pp2dp2_static${clock}_${RUN_TAG}"
    MASTER_PORT="${port}" \
    EXPERIMENT_NAME="${experiment_name}" \
    EXPERIMENT_MODE="static" \
    STATIC_CLOCK_MHZ="${clock}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    bash "${LAUNCH_SCRIPT}"
}

port="${MASTER_PORT_BASE}"
if [[ "${RUN_BASELINE}" == "1" ]]; then
    run_one baseline "" "${port}"
    port=$((port + 10))
    sleep 10
fi

for clock in ${FREQS}; do
    run_one static "${clock}" "${port}"
    port=$((port + 10))
    sleep 10
done
