#!/usr/bin/env bash
set -euo pipefail

BASE_PATH=/home/sd/Megatron-DeepSpeed
HOSTFILE=/tmp/pattaya_localhost_16.hostfile
PATH_PREFIX=/home/sd/.local/bin:/usr/local/cuda/bin:/opt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY_SITE=/home/sd/.local/lib/python3.10/site-packages
frequencies=(1185 1192 1207 1215)

for freq in "${frequencies[@]}"; do
  echo "==== RUN ${freq} MHz ===="
  env -i HOME=/home/sd USER=sd TERM=dumb PATH="$PATH_PREFIX" PYTHONPATH="$PY_SITE" bash -c '
    set -euo pipefail
    export BASE_PATH="'$BASE_PATH'"
    export HOSTFILE="'$HOSTFILE'"
    export LAUNCHER=deepspeed
    export EXPERIMENT_MODE=static
    export EXPERIMENT_NAME="v100_tp2pp4dp2_static_'"$freq"'"
    export TP=2
    export PP=4
    export TRAIN_STEPS=20
    export MICRO_BATCH_SIZE=1
    export GLOBAL_BATCH_SIZE=16
    export LR_WARMUP_ITERS=19
    export EVAL_INTERVAL=21
    export EVAL_ITERS=0
    export ZERO_STAGE=1
    export PRECISION_MODE=bf16
    export STATIC_CLOCK_MHZ='"$freq"'
    export DISABLE_CHECKPOINT=1
    exec bash "'$BASE_PATH'/scripts/run_experiment.sh"
  '
done
