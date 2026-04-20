#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${PACKAGE_DIR}/.." && pwd)}"

export LAUNCHER="${LAUNCHER:-deepspeed}"
export EXPERIMENT_MODE="${EXPERIMENT_MODE:-baseline}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen1p5b_demo_${EXPERIMENT_MODE}}"

export TP="${TP:-2}"
export PP="${PP:-1}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export HIDDEN_SIZE="${HIDDEN_SIZE:-1536}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-8960}"
export NUM_LAYERS="${NUM_LAYERS:-28}"
export NUM_HEADS="${NUM_HEADS:-12}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
export SEQ_LENGTH="${SEQ_LENGTH:-2048}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export TRAIN_STEPS="${TRAIN_STEPS:-20}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
export EVAL_ITERS="${EVAL_ITERS:-0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
export ZERO_STAGE="${ZERO_STAGE:-1}"
export PRECISION_MODE="${PRECISION_MODE:-fp16}"
export DISABLE_CHECKPOINT="${DISABLE_CHECKPOINT:-1}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-0}"

export DATASET="${DATASET:-${BASE_PATH}/data/chinese_wiki_megatron_text_document}"

if [[ "${EXPERIMENT_MODE}" == "static" && -z "${STATIC_CLOCK_MHZ:-}" ]]; then
    echo "[错误] EXPERIMENT_MODE=static 时必须设置 STATIC_CLOCK_MHZ" >&2
    exit 1
fi

cat <<EOF
[演示预设] qwen1p5b-demo
[演示预设] 模式=${EXPERIMENT_MODE}
[演示预设] tp=${TP} pp=${PP} 节点数=${NNODES}
[演示预设] layers=${NUM_LAYERS} hidden=${HIDDEN_SIZE} ffn=${FFN_HIDDEN_SIZE} heads=${NUM_HEADS} kv_heads=${NUM_KV_HEADS}
[演示预设] micro_batch=${MICRO_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE} train_steps=${TRAIN_STEPS}
[演示预设] precision=${PRECISION_MODE} disable_checkpoint=${DISABLE_CHECKPOINT}
[演示预设] dataset=${DATASET}
[演示预设] tokenizer=${TOKENIZER_PATH:-<自动解析>}
EOF

exec bash "${BASE_PATH}/scripts/run_experiment.sh"
