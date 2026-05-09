#!/usr/bin/env bash
set -euo pipefail

BASE="/home/sd/Megatron-DeepSpeed"
HOSTS_TEXT=$'v100x16-1 slots=8\nv100x16-2 slots=8\n'
FREQS=(1080 1155)
PORTS=(30032 30033)

cd "${BASE}"

export PATH="/home/sd/.local/bin:${PATH}"
export PYTHONPATH="/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PDSH_RCMD_TYPE="ssh"
export GLOO_SOCKET_IFNAME="enp6s0"
export NCCL_SOCKET_IFNAME="enp6s0"
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
export NCCL_IB_DISABLE="0"
export NCCL_DEBUG="WARN"
export NCCL_RAS_ENABLE="0"
export TORCH_NCCL_BLOCKING_WAIT="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LAUNCHER="deepspeed"
export EXPERIMENT_MODE="static"
export TP="4"
export PP="1"
export NNODES="2"
export DS_INCLUDE="v100x16-1:8,9,10,11,12,13,14,15@v100x16-2:8,9,10,11,12,13,14,15"
export LOCAL_GPU_INDICES="8,9,10,11,12,13,14,15"
export MASTER_ADDR="192.168.205.201"
export MODEL_SIZE="qwen7b"
export MICRO_BATCH_SIZE="1"
export GLOBAL_BATCH_SIZE="16"
export TRAIN_STEPS="20"
export ZERO_STAGE="1"
export DISABLE_CHECKPOINT="1"
export EVAL_INTERVAL="0"
export EVAL_ITERS="0"
export SAVE_INTERVAL="0"

for idx in "${!FREQS[@]}"; do
  freq="${FREQS[$idx]}"
  port="${PORTS[$idx]}"
  ts="$(date +%Y%m%d_%H%M%S)"

  export RUN_ID="ib_dual16_tp4pp1dp4_formal_${freq}_${ts}_DGX2-1"
  export EXPERIMENT_NAME="ib_dual16_tp4pp1dp4_formal_${freq}_20260410"
  export STATIC_CLOCK_MHZ="${freq}"
  export MASTER_PORT="${port}"
  export HOSTFILE="/tmp/hostfile_${RUN_ID}"

  printf '%s' "${HOSTS_TEXT}" > "${HOSTFILE}"

  echo "RUN_ID=${RUN_ID}"
  echo "STATIC_CLOCK_MHZ=${STATIC_CLOCK_MHZ}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "HOSTFILE=${HOSTFILE}"
  echo "START_AT=$(date '+%F %T')"

  timeout 5400 ./scripts/run_experiment.sh

  echo "END_AT=$(date '+%F %T')"
  echo "DONE_FREQ=${freq}"
  echo

  sleep 10
done
