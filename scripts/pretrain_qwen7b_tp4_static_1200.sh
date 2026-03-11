#!/bin/bash
set -euo pipefail
BASE_PATH="${BASE_PATH:-/home/sd/Megatron-DeepSpeed}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen7b_tp4_static_1200}"
export EXPERIMENT_MODE="${EXPERIMENT_MODE:-static}"
export STATIC_CLOCK_MHZ="${STATIC_CLOCK_MHZ:-1200}"
export TP="${TP:-4}"
export PP="${PP:-1}"
exec bash "${BASE_PATH}/scripts/run_experiment.sh"
