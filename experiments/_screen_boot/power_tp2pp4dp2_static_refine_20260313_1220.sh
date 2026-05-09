#!/usr/bin/env bash
set -euo pipefail

BASE_PATH=/home/sd/Megatron-DeepSpeed
HOSTFILE=/tmp/pattaya_localhost_16.hostfile
PATH_PREFIX=/home/sd/.local/bin:/usr/local/cuda/bin:/opt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY_SITE=/home/sd/.local/lib/python3.10/site-packages

frequencies=(1185 1192 1207 1215)

cd "$BASE_PATH"

for freq in "${frequencies[@]}"; do
  echo "==== RUN ${freq} MHz ===="
  env -i \
    HOME=/home/sd \
    USER=sd \
    TERM=dumb \
    PATH="$PATH_PREFIX" \
    PYTHONPATH="$PY_SITE" \
    BASE_PATH="$BASE_PATH" \
    HOSTFILE="$HOSTFILE" \
    LAUNCHER=deepspeed \
    EXPERIMENT_MODE=static \
    EXPERIMENT_NAME="v100_tp2pp4dp2_static_${freq}" \
    TP=2 \
    PP=4 \
    TRAIN_STEPS=20 \
    MICRO_BATCH_SIZE=1 \
    GLOBAL_BATCH_SIZE=16 \
    LR_WARMUP_ITERS=19 \
    EVAL_INTERVAL=21 \
    EVAL_ITERS=0 \
    ZERO_STAGE=1 \
    PRECISION_MODE=bf16 \
    STATIC_CLOCK_MHZ="$freq" \
    DISABLE_CHECKPOINT=1 \
    bash "$BASE_PATH/run_experiment.sh"
done
