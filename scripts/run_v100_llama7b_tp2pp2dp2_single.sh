#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ "${HOSTNAME%%.*}" != "v100x16-1" && "${HOSTNAME%%.*}" != "DGX2-1" ]]; then
    echo "[Error] Run this script from DGX2-1 / v100x16-1." >&2
    exit 1
fi

if [[ -f "${BASE_PATH}/scripts/activate_runtime_env.sh" ]]; then
    source "${BASE_PATH}/scripts/activate_runtime_env.sh" >/dev/null
fi

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
export LAUNCHER="${LAUNCHER:-deepspeed}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-v100_llama7b_realweight_baseline_formal20}"
export EXPERIMENT_MODE="${EXPERIMENT_MODE:-baseline}"

export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export TP="${TP:-2}"
export PP="${PP:-2}"
export LOCAL_GPU_INDICES="${LOCAL_GPU_INDICES:-0,1,2,3,4,5,6,7}"
export DS_INCLUDE="${DS_INCLUDE:-v100x16-1:0,1,2,3,4,5,6,7}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-31331}"
export HOSTFILE="${HOSTFILE:-/tmp/hostfile_v100_llama7b_tp2pp2dp2_single}"

cat > "${HOSTFILE}" <<'EOF_HOSTFILE'
v100x16-1 slots=8
EOF_HOSTFILE

export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp6s0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp6s0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

export MODEL_SIZE="${MODEL_SIZE:-llama7b}"
export NUM_LAYERS="${NUM_LAYERS:-32}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-11008}"
export NUM_HEADS="${NUM_HEADS:-32}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-32}"
export VOCAB_SIZE="${VOCAB_SIZE:-32000}"
export MAKE_VOCAB_SIZE_DIVISIBLE_BY="${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-128}"

export DATASET="${DATASET:-${BASE_PATH}/data/chinese_wiki_llama_megatron_text_document}"
export DATA_CACHE_PATH="${DATA_CACHE_PATH:-${BASE_PATH}/data/index-cache}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/home/sd/models/llama-7b-hf}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export TRAIN_STEPS="${TRAIN_STEPS:-20}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export EVAL_ITERS="${EVAL_ITERS:-0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-0}"

export ZERO_STAGE="${ZERO_STAGE:-1}"
export USE_CPU_OPTIMIZER="${USE_CPU_OPTIMIZER:-1}"
export OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE:-cpu}"
export OFFLOAD_OPTIMIZER_PIN_MEMORY="${OFFLOAD_OPTIMIZER_PIN_MEMORY:-0}"
export PRECISION_MODE="${PRECISION_MODE:-bf16}"

export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-1}"
export LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-${BASE_PATH}/checkpoints/llama7b_hf2megads_tp2pp2_v100}"
export FINETUNE="${FINETUNE:-1}"
export DISABLE_SAVE_CHECKPOINT="${DISABLE_SAVE_CHECKPOINT:-1}"

exec bash "${BASE_PATH}/scripts/run_experiment.sh"
