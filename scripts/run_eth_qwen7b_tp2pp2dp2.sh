#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ "${HOSTNAME}" != "sd-1" ]]; then
    echo "[Error] Run this script from sd-1" >&2
    exit 1
fi

export PATH="/home/user/miniconda3/envs/tp4bit/bin:${PATH}"

if [[ -f "${BASE_PATH}/scripts/activate_runtime_env.sh" ]]; then
    source "${BASE_PATH}/scripts/activate_runtime_env.sh" >/dev/null
fi

export PYTHON_BIN="/home/user/miniconda3/envs/tp4bit/bin/python3"
export LAUNCHER="deepspeed"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-eth_qwen7b_tp2pp2dp2_baseline_formal20}"
export EXPERIMENT_MODE="${EXPERIMENT_MODE:-baseline}"

export NNODES="${NNODES:-2}"
export NODE_RANK="${NODE_RANK:-0}"
export TP="${TP:-2}"
export PP="${PP:-2}"
export LOCAL_GPU_INDICES="${LOCAL_GPU_INDICES:-0,1,2,3}"
export DS_INCLUDE="${DS_INCLUDE:-sd-1:0,1,2,3@sd-2:0,1,2,3}"

export MASTER_ADDR="${MASTER_ADDR:-192.168.66.121}"
export MASTER_PORT="${MASTER_PORT:-32251}"
export HOSTFILE="${HOSTFILE:-/tmp/hostfile_eth_qwen7b_tp2pp2dp2}"

cat > "${HOSTFILE}" <<'EOF_HOSTFILE'
sd-1 slots=4
sd-2 slots=4
EOF_HOSTFILE

export CONDA_ENV="${CONDA_ENV:-tp4bit}"

export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export MODEL_SIZE="${MODEL_SIZE:-qwen7b}"
export NUM_LAYERS="${NUM_LAYERS:-28}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-3584}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-18944}"
export NUM_HEADS="${NUM_HEADS:-28}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"

export DATASET="${DATASET:-${BASE_PATH}/data/qwen_data_text_document}"
export DATA_CACHE_PATH="${DATA_CACHE_PATH:-${BASE_PATH}/data/index-cache}"

export TOKENIZER_PATH="${TOKENIZER_PATH:-/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SEQ_LENGTH=2048
export MAX_POSITION_EMBEDDINGS=2048
export TRAIN_STEPS="${TRAIN_STEPS:-20}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-0}"

export ZERO_STAGE="${ZERO_STAGE:-1}"
export USE_CPU_OPTIMIZER="${USE_CPU_OPTIMIZER:-1}"
export OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE:-cpu}"
export OFFLOAD_OPTIMIZER_PIN_MEMORY="${OFFLOAD_OPTIMIZER_PIN_MEMORY:-0}"
export PRECISION_MODE="${PRECISION_MODE:-bf16}"

export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-1}"
export LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-${BASE_PATH}/checkpoints/qwen25_7b_instruct_hf2megads_tp2pp2_real_main}"
export FINETUNE="${FINETUNE:-1}"
export DISABLE_SAVE_CHECKPOINT="${DISABLE_SAVE_CHECKPOINT:-1}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec bash "${BASE_PATH}/scripts/run_experiment.sh"
