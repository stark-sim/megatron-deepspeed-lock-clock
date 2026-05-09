#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

FREQS="${FREQS:-baseline 1260 1350 1455 1530}"

echo "========================================"
echo "V100 Single-Node Qwen7B TP2PP2DP2 Sweep"
echo "FREQS: ${FREQS}"
echo "========================================"

for freq in ${FREQS}; do
    if [[ "${freq}" == "baseline" ]]; then
        mode="baseline"
        clock_mhz=""
        exp_name="v100_qwen7b_tp2pp2dp2_single_baseline_formal20_finetune_nosave"
    else
        mode="static"
        clock_mhz="${freq}"
        exp_name="v100_qwen7b_tp2pp2dp2_single_static${freq}_formal20_finetune_nosave"
    fi

    echo ""
    echo ">>> Starting ${exp_name} ..."
    echo "    Mode: ${mode}, Clock: ${clock_mhz:-default}"

    export EXPERIMENT_MODE="${mode}"
    export EXPERIMENT_NAME="${exp_name}"
    export STATIC_CLOCK_MHZ="${clock_mhz}"
    export MASTER_PORT="$((30791 + $(date +%s) % 1000))"

    bash "${BASE_PATH}/scripts/run_v100_qwen7b_tp2pp2dp2_single.sh" || {
        echo "[Warning] ${exp_name} failed, continuing to next frequency..." >&2
    }

    echo "<<< Finished ${exp_name}"
    sleep 5
done

echo ""
echo "========================================"
echo "Sweep completed. All runs under:"
echo "  ${BASE_PATH}/experiments/"
echo "========================================"
