#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ "${HOSTNAME%%.*}" != "v100x16-1" && "${HOSTNAME%%.*}" != "DGX2-1" ]]; then
    echo "[Error] Run this script from DGX2-1 / v100x16-1." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Topology: TP=2, PP=2, DP=4, 16 GPUs total (8 per node)
# ---------------------------------------------------------------------------
export TP=2
export PP=2
export DP=4
export NNODES=2
export LOCAL_GPU_INDICES="8,9,10,11,12,13,14,15"
export DS_INCLUDE="v100x16-1:8,9,10,11,12,13,14,15@v100x16-2:8,9,10,11,12,13,14,15"
export MASTER_ADDR="${MASTER_ADDR:-192.168.205.201}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp6s0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9}"

export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export TRAIN_STEPS="${TRAIN_STEPS:-20}"

# Checkpoint path must be set externally or defaults to TP2PP2 converted checkpoint
export LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-}"
export NO_UNTIE_EMBEDDINGS=1
export DISABLE_SAVE_CHECKPOINT=1

# ---------------------------------------------------------------------------
# Hostfile
# ---------------------------------------------------------------------------
HOSTFILE="/tmp/hostfile_tp2pp2dp4_$(date +%s).txt"
echo "v100x16-1 slots=8" > "${HOSTFILE}"
echo "v100x16-2 slots=8" >> "${HOSTFILE}"
export HOSTFILE="${HOSTFILE}"

# Launch via canonical run_experiment.sh
bash "${SCRIPT_DIR}/run_experiment.sh"
