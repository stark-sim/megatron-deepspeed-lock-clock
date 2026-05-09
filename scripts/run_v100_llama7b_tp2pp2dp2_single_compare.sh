#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LAUNCH_SCRIPT="${BASE_PATH}/scripts/run_v100_llama7b_tp2pp2dp2_single.sh"

if [[ "${HOSTNAME%%.*}" != "v100x16-1" && "${HOSTNAME%%.*}" != "DGX2-1" ]]; then
    echo "[Error] Run this script from DGX2-1 / v100x16-1." >&2
    exit 1
fi

if [[ ! -x "${LAUNCH_SCRIPT}" ]]; then
    echo "[Error] Missing executable launcher: ${LAUNCH_SCRIPT}" >&2
    exit 1
fi

TRAIN_STEPS="${TRAIN_STEPS:-20}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-31431}"
RUN_TAG="${RUN_TAG:-formal20}"
RUN_BASELINE="${RUN_BASELINE:-1}"
FREQS="${FREQS:-1260 1350 1455 1530}"
COOLDOWN_SEC="${COOLDOWN_SEC:-30}"

reset_clocks() {
    local gpu
    IFS=',' read -r -a gpus <<< "${LOCAL_GPU_INDICES:-0,1,2,3,4,5,6,7}"
    for gpu in "${gpus[@]}"; do
        sudo -n nvidia-smi -i "${gpu}" -rgc >/dev/null 2>&1 || true
    done
}

run_one() {
    local mode="$1"
    local clock="$2"
    local port="$3"
    local experiment_name

    reset_clocks

    if [[ "${mode}" == "baseline" ]]; then
        experiment_name="v100_llama7b_realweight_baseline_${RUN_TAG}"
        MASTER_PORT="${port}" \
        EXPERIMENT_NAME="${experiment_name}" \
        EXPERIMENT_MODE="baseline" \
        TRAIN_STEPS="${TRAIN_STEPS}" \
        bash "${LAUNCH_SCRIPT}"
        return
    fi

    experiment_name="v100_llama7b_realweight_static${clock}_${RUN_TAG}"
    MASTER_PORT="${port}" \
    EXPERIMENT_NAME="${experiment_name}" \
    EXPERIMENT_MODE="static" \
    STATIC_CLOCK_MHZ="${clock}" \
    TRAIN_STEPS="${TRAIN_STEPS}" \
    bash "${LAUNCH_SCRIPT}"
}

echo "LLaMA-7B V100 single-node compare"
echo "Base path: ${BASE_PATH}"
echo "Train steps: ${TRAIN_STEPS}"
echo "Static clocks: ${FREQS}"

port="${MASTER_PORT_BASE}"
if [[ "${RUN_BASELINE}" == "1" ]]; then
    run_one baseline "" "${port}"
    port=$((port + 10))
    sleep "${COOLDOWN_SEC}"
fi

for clock in ${FREQS}; do
    run_one static "${clock}" "${port}"
    port=$((port + 10))
    reset_clocks
    sleep "${COOLDOWN_SEC}"
done

reset_clocks
