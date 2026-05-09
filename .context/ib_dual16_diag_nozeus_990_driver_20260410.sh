#!/usr/bin/env bash
set -euo pipefail

BASE="/home/sd/Megatron-DeepSpeed"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ib_dual16_tp4pp1dp4_diag_nozeus_990_${TS}_DGX2-1"
HOSTFILE="/tmp/hostfile_${RUN_ID}"

echo "RUN_ID=${RUN_ID}"
echo "HOSTFILE=${HOSTFILE}"
echo "START_AT=$(date '+%F %T')"

printf 'v100x16-1 slots=8\nv100x16-2 slots=8\n' > "${HOSTFILE}"

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
export DISABLE_ZEUS_MONITORING="1"
export RUN_ID
export LAUNCHER="deepspeed"
export EXPERIMENT_MODE="static"
export EXPERIMENT_NAME="ib_dual16_tp4pp1dp4_diag_nozeus_990_20260410"
export STATIC_CLOCK_MHZ="990"
export TP="4"
export PP="1"
export NNODES="2"
export HOSTFILE
export DS_INCLUDE="v100x16-1:8,9,10,11,12,13,14,15@v100x16-2:8,9,10,11,12,13,14,15"
export LOCAL_GPU_INDICES="8,9,10,11,12,13,14,15"
export MASTER_ADDR="192.168.205.201"
export MASTER_PORT="30041"
export MODEL_SIZE="qwen7b"
export MICRO_BATCH_SIZE="1"
export GLOBAL_BATCH_SIZE="16"
export TRAIN_STEPS="20"
export ZERO_STAGE="1"
export DISABLE_CHECKPOINT="1"
export EVAL_INTERVAL="0"
export EVAL_ITERS="0"
export SAVE_INTERVAL="0"

timeout 5400 ./scripts/run_experiment.sh
rc=$?
echo "EXIT_CODE=${rc}"
echo "END_AT=$(date '+%F %T')"
exit "${rc}"
