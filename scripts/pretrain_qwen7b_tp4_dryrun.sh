#!/bin/bash
set -euo pipefail
BASE_PATH="${BASE_PATH:-/home/sd/Megatron-DeepSpeed}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen7b_tp4_freq_dryrun}"
export EXPERIMENT_MODE="${EXPERIMENT_MODE:-dryrun}"
export TP="${TP:-4}"
export PP="${PP:-1}"
export COMM_LOW_FREQ="${COMM_LOW_FREQ:-800}"
export COMM_MIN_ELEMENTS="${COMM_MIN_ELEMENTS:-104857600}"
export TRAIN_STEPS="${TRAIN_STEPS:-100}"
exec bash "${BASE_PATH}/scripts/run_experiment.sh"
