#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/experiment_utils.sh"
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-}}"

resolve_default_dataset() {
    local candidates=(
        "${BASE_PATH}/data/chinese_wiki_megatron_text_document"
        "${BASE_PATH}/data/qwen_data_text_document"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}.bin" && -f "${candidate}.idx" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done
    printf '%s\n' "${BASE_PATH}/data/chinese_wiki_megatron_text_document"
}

resolve_default_tokenizer() {
    local candidate
    if [[ -d "${BASE_PATH}/.context/qwen25_tokenizer_flat" ]]; then
        printf '%s\n' "${BASE_PATH}/.context/qwen25_tokenizer_flat"
        return
    fi

    shopt -s nullglob
    for candidate in \
        "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"* \
        "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/"* \
        "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"* \
        "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/"* \
        "/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/"*
    do
        if [[ -f "${candidate}/tokenizer.json" ]]; then
            printf '%s\n' "${candidate}"
            shopt -u nullglob
            return
        fi
    done
    shopt -u nullglob

    printf '%s\n' "${BASE_PATH}/.context/qwen25_tokenizer_flat"
}

dataset_exists() {
    local dataset_prefix="$1"
    [[ -n "${dataset_prefix}" && -f "${dataset_prefix}.bin" && -f "${dataset_prefix}.idx" ]]
}

tokenizer_exists() {
    local tokenizer_path="$1"
    [[ -n "${tokenizer_path}" && -d "${tokenizer_path}" && ( -f "${tokenizer_path}/tokenizer.json" || -f "${tokenizer_path}/tokenizer_config.json" ) ]]
}

declare -a REMOTE_SELECTED_HOSTS=()

resolve_host_gpu_indices() {
    local host="$1"
    local short_host="${host%%.*}"
    local local_host_short="${HOSTNAME%%.*}"
    if [[ "$host" == "localhost" || "$host" == "$HOSTNAME" || "$host" == "$local_host_short" ]]; then
        printf '%s\n' "${LOCAL_GPU_INDICES}"
        return
    fi
    if [[ -n "${DS_INCLUDE:-}" ]]; then
        local chunk entry_host gpu_indices entry_short
        IFS='@' read -r -a chunks <<< "${DS_INCLUDE}"
        for chunk in "${chunks[@]}"; do
            entry_host="${chunk%%:*}"
            gpu_indices="${chunk#*:}"
            entry_short="${entry_host%%.*}"
            if [[ "${entry_host}" == "${host}" || "${entry_short}" == "${short_host}" ]]; then
                printf '%s\n' "${gpu_indices}"
                return
            fi
        done
    fi
    printf '%s\n' "${LOCAL_GPU_INDICES}"
}

LAUNCHER="${LAUNCHER:-deepspeed}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-baseline}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen7b_experiment}"
TP="${TP:-4}"
PP="${PP:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
HOSTFILE="${HOSTFILE:-}"
DS_INCLUDE="${DS_INCLUDE:-}"

MODEL_SIZE="${MODEL_SIZE:-qwen7b}"
DS_CONFIG_TEMPLATE="${DS_CONFIG_TEMPLATE:-${BASE_PATH}/scripts/ds_config_7b_tp4.json}"
DATASET="${DATASET:-$(resolve_default_dataset)}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$(resolve_default_tokenizer)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${BASE_PATH}/checkpoints/${EXPERIMENT_NAME}}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-0}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
DISABLE_CHECKPOINT="${DISABLE_CHECKPOINT:-0}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-}"

HIDDEN_SIZE="${HIDDEN_SIZE:-3584}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-18944}"
NUM_LAYERS="${NUM_LAYERS:-28}"
NUM_HEADS="${NUM_HEADS:-28}"
NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
TRAIN_STEPS="${TRAIN_STEPS:-500}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_STEPS}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_ITERS="${EVAL_ITERS:-10}"
ZERO_STAGE="${ZERO_STAGE:-1}"
PRECISION_MODE="${PRECISION_MODE:-bf16}"
USE_CPU_OPTIMIZER="${USE_CPU_OPTIMIZER:-0}"
OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE:-}"
OFFLOAD_OPTIMIZER_PIN_MEMORY="${OFFLOAD_OPTIMIZER_PIN_MEMORY:-0}"

STATIC_CLOCK_MHZ="${STATIC_CLOCK_MHZ:-}"
COMM_LOW_FREQ="${COMM_LOW_FREQ:-1200}"
COMM_HIGH_FREQ="${COMM_HIGH_FREQ:-}"
COMM_MIN_ELEMENTS="${COMM_MIN_ELEMENTS:-300000000}"

if [[ -z "$LR_WARMUP_ITERS" ]]; then
    if (( TRAIN_STEPS <= 1 )); then
        LR_WARMUP_ITERS=0
    elif (( TRAIN_STEPS <= 50 )); then
        LR_WARMUP_ITERS=$(( TRAIN_STEPS - 1 ))
    else
        LR_WARMUP_ITERS=50
    fi
fi

if (( EVAL_ITERS <= 0 )) && (( EVAL_INTERVAL <= 0 )); then
    EVAL_INTERVAL=$(( TRAIN_STEPS + 1 ))
fi

LOCAL_GPU_INDICES="${LOCAL_GPU_INDICES:-$(parse_cuda_visible_devices)}"
LOCAL_GPU_COUNT="$(count_csv_items "$LOCAL_GPU_INDICES")"
WORLD_SIZE="$(( LOCAL_GPU_COUNT * NNODES ))"

setup_experiment_run "${BASE_PATH}" "${EXPERIMENT_NAME}"

RUN_DS_CONFIG="${RUN_DIR}/ds_config.json"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/${RUN_ID}}"
if [[ "$DISABLE_CHECKPOINT" == "1" ]]; then
    CHECKPOINT_PATH=""
    SAVE_INTERVAL=0
    LOAD_CHECKPOINT=0
    echo "[Checkpoint] DISABLE_CHECKPOINT=1; skipping checkpoint save/load"
else
    mkdir -p "${CHECKPOINT_PATH}"
fi

export EXPERIMENT_NAME RUN_ID RUN_DIR RUN_LOG_DIR EXPERIMENT_ROOT
export MEGATRON_RUN_ID="${RUN_ID}"
export MEGATRON_EXPERIMENT_ROOT="${EXPERIMENT_ROOT}"
export MEGATRON_EXPERIMENT_MODE="${EXPERIMENT_MODE}"
export MEGATRON_REQUESTED_TP="${TP}"
export MEGATRON_REQUESTED_PP="${PP}"

if [[ -z "$LOCAL_GPU_INDICES" || "$LOCAL_GPU_COUNT" -eq 0 ]]; then
    echo "[Error] No visible GPUs found via CUDA_VISIBLE_DEVICES or nvidia-smi" >&2
    exit 1
fi

if (( TP > LOCAL_GPU_COUNT )); then
    echo "[Error] TP=${TP} exceeds local visible GPU count ${LOCAL_GPU_COUNT}; TP must fit in a single node" >&2
    exit 1
fi

if (( WORLD_SIZE % (TP * PP) != 0 )); then
    echo "[Error] WORLD_SIZE=${WORLD_SIZE} is not divisible by TP*PP=$((TP * PP))" >&2
    exit 1
fi

if (( NUM_HEADS % TP != 0 )); then
    echo "[Error] NUM_HEADS=${NUM_HEADS} is not divisible by TP=${TP}" >&2
    exit 1
fi

if (( NUM_KV_HEADS % TP != 0 )); then
    echo "[Error] NUM_KV_HEADS=${NUM_KV_HEADS} is not divisible by TP=${TP}; choose a TP-compatible GQA topology or increase NUM_KV_HEADS" >&2
    exit 1
fi

if (( NNODES > 1 )) && [[ -z "$HOSTFILE" ]]; then
    echo "[Error] HOSTFILE is required when NNODES>1" >&2
    exit 1
fi

if (( PP > 1 )) && [[ -z "$MASTER_ADDR" || -z "$MASTER_PORT" ]]; then
    echo "[Error] MASTER_ADDR and MASTER_PORT are required when PP>1" >&2
    exit 1
fi

if ! dataset_exists "$DATASET"; then
    echo "[Error] DATASET prefix is invalid: ${DATASET} (.bin/.idx missing)" >&2
    exit 1
fi

if ! tokenizer_exists "$TOKENIZER_PATH"; then
    echo "[Error] TOKENIZER_PATH is invalid: ${TOKENIZER_PATH}" >&2
    exit 1
fi

json_bool() {
    local value="${1:-}"
    case "${value,,}" in
        1|true|yes|on)
            printf 'true\n'
            ;;
        0|false|no|off|'')
            printf 'false\n'
            ;;
        *)
            echo "[Error] Invalid boolean value: ${value}" >&2
            exit 1
            ;;
    esac
}

OFFLOAD_OPTIMIZER_PIN_MEMORY_JSON="$(json_bool "$OFFLOAD_OPTIMIZER_PIN_MEMORY")"

cat > "$RUN_DS_CONFIG" <<JSON
{
  "train_batch_size": ${GLOBAL_BATCH_SIZE},
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": ${ZERO_STAGE}$( if [[ -n "$OFFLOAD_OPTIMIZER_DEVICE" ]]; then cat <<EOF
,
    "offload_optimizer": {
      "device": "${OFFLOAD_OPTIMIZER_DEVICE}",
      "pin_memory": ${OFFLOAD_OPTIMIZER_PIN_MEMORY_JSON}
    }
EOF
fi )
  },
  "bf16": {
    "enabled": $( [[ "$PRECISION_MODE" == "bf16" ]] && echo true || echo false )
  },
  "fp16": {
    "enabled": $( [[ "$PRECISION_MODE" == "fp16" ]] && echo true || echo false )
  },
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "wall_clock_breakdown": false
}
JSON

HOSTFILE_JSON_PATH="${RUN_DIR}/hostfile_snapshot.json"
PREFLIGHT_JSON_PATH="${RUN_DIR}/preflight.json"
TOPOLOGY_JSON_PATH="${RUN_DIR}/topology.json"

if [[ -n "$HOSTFILE" ]]; then
    parse_hostfile_to_json "$HOSTFILE" "$NNODES" > "$HOSTFILE_JSON_PATH"
    export MEGATRON_HOSTFILE_PATH="$HOSTFILE"
    export MEGATRON_HOSTFILE_JSON="$HOSTFILE_JSON_PATH"
fi

write_topology_json "$TOPOLOGY_JSON_PATH" "$LOCAL_GPU_INDICES" "$NNODES" "$NODE_RANK" "$MASTER_ADDR" "$MASTER_PORT" "$TP" "$PP" "$LAUNCHER"
export MEGATRON_TOPOLOGY_JSON="$TOPOLOGY_JSON_PATH"

run_local_preflight() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
import socket
import subprocess
import sys

result = {
    'hostname': socket.gethostname(),
    'repo_path': os.getcwd(),
    'checks': {},
}

def command_ok(cmd):
    completed = subprocess.run(cmd, capture_output=True, text=True)
    return completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip()

visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if visible.strip():
    gpu_indices = [int(x.strip()) for x in visible.split(',') if x.strip()]
else:
    ok, stdout, _ = command_ok(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'])
    gpu_indices = [int(x.strip()) for x in stdout.splitlines() if x.strip()] if ok else []

result['cuda_visible_devices'] = visible
result['gpu_indices'] = gpu_indices
result['local_gpu_count'] = len(gpu_indices)
expected_tp = int(os.environ['TP'])
result['checks']['tp_fits_local_node'] = len(gpu_indices) >= expected_tp
result['checks']['repo_exists'] = os.path.exists(os.getcwd())
result['checks']['python_available'] = command_ok(['python3', '--version'])[0]
result['checks']['nvidia_smi_available'] = command_ok(['bash', '-lc', 'command -v nvidia-smi >/dev/null 2>&1'])[0]
dataset_prefix = os.environ.get('DATASET', '')
tokenizer_path = os.environ.get('TOKENIZER_PATH', '')
result['checks']['dataset_exists'] = (
    bool(dataset_prefix)
    and os.path.exists(f'{dataset_prefix}.bin')
    and os.path.exists(f'{dataset_prefix}.idx')
)
result['checks']['tokenizer_exists'] = (
    bool(tokenizer_path)
    and os.path.isdir(tokenizer_path)
    and (
        os.path.exists(os.path.join(tokenizer_path, 'tokenizer.json'))
        or os.path.exists(os.path.join(tokenizer_path, 'tokenizer_config.json'))
    )
)
if os.environ.get('LAUNCHER', 'deepspeed') == 'deepspeed':
    result['checks']['launcher_available'] = command_ok(['bash', '-lc', 'command -v deepspeed >/dev/null 2>&1'])[0]
else:
    result['checks']['launcher_available'] = command_ok(['python3', '-m', 'torch.distributed.run', '--help'])[0]
try:
    fallback = os.path.expanduser('~/.local/lib/python3.10/site-packages')
    if os.path.isdir(fallback) and fallback not in sys.path:
        sys.path.insert(0, fallback)
    import pynvml
    result['checks']['pynvml_available'] = True
except Exception:
    result['checks']['pynvml_available'] = False

mode = os.environ.get('EXPERIMENT_MODE', '')
static_clock = os.environ.get('STATIC_CLOCK_MHZ', '')
if mode == 'static' and static_clock and result['checks']['pynvml_available'] and gpu_indices:
    import pynvml
    pynvml.nvmlInit()
    try:
        clock = int(static_clock)
        support = True
        unsupported = []
        for index in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)[0]
            graphics = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, mem)
            if clock not in graphics:
                support = False
                unsupported.append({'gpu_index': index, 'supported_graphics_clocks_mhz': graphics})
        result['checks']['static_clock_supported'] = support
        if unsupported:
            result['static_clock_unsupported'] = unsupported
    finally:
        pynvml.nvmlShutdown()

result['ok'] = all(bool(v) for v in result['checks'].values())
print(json.dumps(result, sort_keys=True))
PY
}

LOCAL_PREFLIGHT="$(CUDA_VISIBLE_DEVICES="$LOCAL_GPU_INDICES" TP="$TP" LAUNCHER="$LAUNCHER" EXPERIMENT_MODE="$EXPERIMENT_MODE" STATIC_CLOCK_MHZ="$STATIC_CLOCK_MHZ" DATASET="$DATASET" TOKENIZER_PATH="$TOKENIZER_PATH" run_local_preflight)"
REMOTE_PREFLIGHTS=()

if (( NNODES > 1 )); then
    mapfile -t SELECTED_HOSTS < <(python3 - "$HOSTFILE_JSON_PATH" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
for entry in payload.get('selected_hosts', []):
    print(entry['hostname'])
PY
)
    for host in "${SELECTED_HOSTS[@]}"; do
        if [[ "$(is_local_host_alias "$host")" == "1" ]]; then
            continue
        fi
        REMOTE_SELECTED_HOSTS+=("$host")
        host_gpu_indices="$(resolve_host_gpu_indices "$host")"
        REMOTE_PREFLIGHTS+=("$(run_remote_preflight "$host" "$BASE_PATH" "$TP" "$EXPERIMENT_MODE" "$STATIC_CLOCK_MHZ" "$host_gpu_indices" "$LAUNCHER" "$DATASET" "$TOKENIZER_PATH" "$CONDA_ENV")")
    done
fi

for host in "${REMOTE_SELECTED_HOSTS[@]}"; do
    remote_dirs=("$RUN_DIR" "$RUN_LOG_DIR")
    [[ -n "$DATA_CACHE_PATH" ]] && remote_dirs+=("$DATA_CACHE_PATH")
    [[ -n "${TORCH_EXTENSIONS_DIR:-}" ]] && remote_dirs+=("$TORCH_EXTENSIONS_DIR")
    [[ -n "${TMPDIR:-}" ]] && remote_dirs+=("$TMPDIR")
    [[ -n "${PYTHONPYCACHEPREFIX:-}" ]] && remote_dirs+=("$PYTHONPYCACHEPREFIX")
    remote_mkdir_cmd="mkdir -p"
    for remote_dir in "${remote_dirs[@]}"; do
        remote_mkdir_cmd+=" '$remote_dir'"
    done
    ssh "$host" "$remote_mkdir_cmd"
    sync_file_to_host "$RUN_DS_CONFIG" "$host" "$RUN_DS_CONFIG"
done

local_dirs=()
[[ -n "$DATA_CACHE_PATH" ]] && local_dirs+=("$DATA_CACHE_PATH")
[[ -n "${TORCH_EXTENSIONS_DIR:-}" ]] && local_dirs+=("$TORCH_EXTENSIONS_DIR")
[[ -n "${TMPDIR:-}" ]] && local_dirs+=("$TMPDIR")
[[ -n "${PYTHONPYCACHEPREFIX:-}" ]] && local_dirs+=("$PYTHONPYCACHEPREFIX")
if (( ${#local_dirs[@]} > 0 )); then
    mkdir -p "${local_dirs[@]}"
fi

python3 - "$PREFLIGHT_JSON_PATH" "$LOCAL_PREFLIGHT" "${REMOTE_PREFLIGHTS[@]}" <<'PY'
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
node_results = [json.loads(item) for item in sys.argv[2:]]
payload = {
    'ok': all(node.get('ok', False) for node in node_results),
    'node_results': node_results,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if not payload['ok']:
    raise SystemExit(1)
PY
export MEGATRON_PREFLIGHT_JSON="$PREFLIGHT_JSON_PATH"

DEEPSPEED_ENV_PATH="${BASE_PATH}/.deepspeed_env"
DEEPSPEED_ENV_BACKUP=""
DEEPSPEED_MANAGED_ENV_VARS=(
    MEGATRON_RUN_ID
    MEGATRON_EXPERIMENT_ROOT
    MEGATRON_EXPERIMENT_MODE
    MEGATRON_REQUESTED_TP
    MEGATRON_REQUESTED_PP
    MEGATRON_HOSTFILE_PATH
    MEGATRON_HOSTFILE_JSON
    MEGATRON_TOPOLOGY_JSON
    MEGATRON_PREFLIGHT_JSON
    STATIC_CLOCK_MHZ
    PATH
    PYTHONPATH
    PYTHON_BIN
    CONDA_ENV
    CONDA_DEFAULT_ENV
    LD_LIBRARY_PATH
    CUDA_HOME
    CUDACXX
    GLOO_SOCKET_IFNAME
    NCCL_SOCKET_IFNAME
    NCCL_IB_HCA
    NCCL_IB_DISABLE
    NCCL_DEBUG
    NCCL_RAS_ENABLE
    TORCH_NCCL_BLOCKING_WAIT
    PYTORCH_CUDA_ALLOC_CONF
    TORCH_EXTENSIONS_DIR
    TMPDIR
    PYTHONPYCACHEPREFIX
    PYTHONDONTWRITEBYTECODE
    DATA_CACHE_PATH
    MAX_JOBS
)

is_managed_deepspeed_env_key() {
    local key="$1"
    local managed_key
    for managed_key in "${DEEPSPEED_MANAGED_ENV_VARS[@]}"; do
        if [[ "$managed_key" == "$key" ]]; then
            return 0
        fi
    done
    return 1
}

append_deepspeed_env_var() {
    local key="$1"
    local value="${!key:-}"
    if [[ -n "$value" ]]; then
        printf '%s=%s\n' "$key" "$value"
    fi
}

prepare_deepspeed_env() {
    if [[ "$LAUNCHER" != "deepspeed" ]]; then
        return
    fi

    DEEPSPEED_ENV_BACKUP="${RUN_DIR}/.deepspeed_env.backup"
    if [[ -f "$DEEPSPEED_ENV_PATH" ]]; then
        cp "$DEEPSPEED_ENV_PATH" "$DEEPSPEED_ENV_BACKUP"
    else
        : > "$DEEPSPEED_ENV_BACKUP"
    fi

    {
        if [[ -s "$DEEPSPEED_ENV_BACKUP" ]]; then
            local line key
            while IFS= read -r line || [[ -n "$line" ]]; do
                [[ -n "$line" && "$line" == *=* ]] || continue
                key="${line%%=*}"
                if is_managed_deepspeed_env_key "$key"; then
                    continue
                fi
                printf '%s\n' "$line"
            done < "$DEEPSPEED_ENV_BACKUP"
        fi
        local env_key
        for env_key in "${DEEPSPEED_MANAGED_ENV_VARS[@]}"; do
            append_deepspeed_env_var "$env_key"
        done
    } > "$DEEPSPEED_ENV_PATH"
}

restore_deepspeed_env() {
    if [[ "$LAUNCHER" != "deepspeed" ]]; then
        return
    fi

    if [[ -n "$DEEPSPEED_ENV_BACKUP" && -f "$DEEPSPEED_ENV_BACKUP" ]]; then
        if [[ -s "$DEEPSPEED_ENV_BACKUP" ]]; then
            mv "$DEEPSPEED_ENV_BACKUP" "$DEEPSPEED_ENV_PATH"
        else
            rm -f "$DEEPSPEED_ENV_PATH" "$DEEPSPEED_ENV_BACKUP"
        fi
    fi
}

prepare_deepspeed_env

if [[ "$VALIDATE_ONLY" == "1" ]]; then
    restore_deepspeed_env
    echo "[Validate] topology and preflight succeeded"
    echo "[Validate] run_dir=${RUN_DIR}"
    exit 0
fi

cleanup() {
    local exit_code=$?
    restore_deepspeed_env
    if [[ -n "${LOCAL_GPU_INDICES:-}" ]]; then
        case "$EXPERIMENT_MODE" in
            static)
                if [[ -n "$STATIC_CLOCK_MHZ" ]]; then
                    for host in "${REMOTE_SELECTED_HOSTS[@]}"; do
                        remote_gpu_indices="$(resolve_host_gpu_indices "$host")"
                        reset_remote_gpu_clocks "$host" "$CONDA_ENV" "$remote_gpu_indices" || echo "[Cleanup] Warning: Failed to reset GPU clocks on ${host}:${remote_gpu_indices}" >&2
                    done
                    reset_gpu_clocks "$LOCAL_GPU_INDICES" || echo "[Cleanup] Warning: Failed to reset GPU clocks for ${LOCAL_GPU_INDICES}" >&2
                fi
                ;;
            dynamic|dryrun)
                reset_gpu_clocks "$LOCAL_GPU_INDICES" || echo "[Cleanup] Warning: Failed to reset GPU clocks for ${LOCAL_GPU_INDICES}" >&2
                ;;
        esac
    fi
    return "$exit_code"
}
trap cleanup EXIT

if [[ "$EXPERIMENT_MODE" == "static" ]]; then
    if [[ -z "$STATIC_CLOCK_MHZ" ]]; then
        echo "[Error] STATIC_CLOCK_MHZ is required for static mode" >&2
        exit 1
    fi
    assert_static_clock_supported "$LOCAL_GPU_INDICES" "$STATIC_CLOCK_MHZ"
    for host in "${REMOTE_SELECTED_HOSTS[@]}"; do
        remote_gpu_indices="$(resolve_host_gpu_indices "$host")"
        assert_remote_static_clock_supported "$host" "$CONDA_ENV" "$remote_gpu_indices" "$STATIC_CLOCK_MHZ"
    done
    lock_gpu_clocks "$LOCAL_GPU_INDICES" "$STATIC_CLOCK_MHZ"
    for host in "${REMOTE_SELECTED_HOSTS[@]}"; do
        remote_gpu_indices="$(resolve_host_gpu_indices "$host")"
        lock_remote_gpu_clocks "$host" "$CONDA_ENV" "$remote_gpu_indices" "$STATIC_CLOCK_MHZ"
    done
elif [[ "$EXPERIMENT_MODE" != "baseline" && "$EXPERIMENT_MODE" != "dynamic" && "$EXPERIMENT_MODE" != "dryrun" ]]; then
    echo "[Error] Unsupported EXPERIMENT_MODE=${EXPERIMENT_MODE}" >&2
    exit 1
fi

TRAIN_CMD=(
    pretrain_gpt.py
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_HEADS"
    --num-key-value-heads "$NUM_KV_HEADS"
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$SEQ_LENGTH"
    --train-iters "$TRAIN_STEPS"
    --data-path "$DATASET"
    $( [[ -n "$DATA_CACHE_PATH" ]] && printf -- '--data-cache-path\n%s\n' "$DATA_CACHE_PATH" )
    --data-impl mmap
    --tokenizer-type HFTokenizer
    --tokenizer-model "$TOKENIZER_PATH"
    --split 98,2,0
    --distributed-backend nccl
    --lr "$LR"
    --lr-decay-style cosine
    --min-lr "$MIN_LR"
    --weight-decay 0.01
    --clip-grad 1.0
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --optimizer adam
    $( [[ "$USE_CPU_OPTIMIZER" == "1" ]] && printf -- '--cpu-optimizer\n' )
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --log-interval 1
    --save-interval "$SAVE_INTERVAL"
    --eval-interval "$EVAL_INTERVAL"
    --eval-iters "$EVAL_ITERS"
    --no-query-key-layer-scaling
    --attention-dropout 0
    --hidden-dropout 0
    --use-rotary-position-embeddings
    --untie-embeddings-and-output-weights
    --swiglu
    --normalization rmsnorm
    --disable-bias-linear
    --no-position-embedding
    --no-masked-softmax-fusion
    --no-bias-gelu-fusion
    --no-bias-dropout-fusion
    --recompute-granularity full
    --recompute-method uniform
    --deepspeed-activation-checkpointing
    --zero-stage="${ZERO_STAGE}"
    --deepspeed_config="${RUN_DS_CONFIG}"
    --deepspeed
    --experiment-run-id "${RUN_ID}"
    --experiment-name "${EXPERIMENT_NAME}"
    --experiment-root-dir "${EXPERIMENT_ROOT}"
)

if [[ "$DISABLE_CHECKPOINT" != "1" ]]; then
    TRAIN_CMD+=(--save "$CHECKPOINT_PATH")
fi

if [[ "$PRECISION_MODE" == "bf16" ]]; then
    TRAIN_CMD+=(--bf16)
elif [[ "$PRECISION_MODE" == "fp16" ]]; then
    TRAIN_CMD+=(--fp16)
else
    echo "[Error] Unsupported PRECISION_MODE=${PRECISION_MODE}" >&2
    exit 1
fi

if [[ "$LOAD_CHECKPOINT" == "1" ]]; then
    TRAIN_CMD+=(--load "$CHECKPOINT_PATH")
fi

if [[ "$EXPERIMENT_MODE" == "dynamic" ]]; then
    TRAIN_CMD+=(--enable-comm-freq-scaling --comm-low-freq "$COMM_LOW_FREQ" --comm-min-elements "$COMM_MIN_ELEMENTS")
    if [[ -n "$COMM_HIGH_FREQ" ]]; then
        TRAIN_CMD+=(--comm-high-freq "$COMM_HIGH_FREQ")
    fi
elif [[ "$EXPERIMENT_MODE" == "dryrun" ]]; then
    TRAIN_CMD+=(--enable-comm-freq-scaling --comm-freq-dry-run --comm-low-freq "$COMM_LOW_FREQ" --comm-min-elements "$COMM_MIN_ELEMENTS")
    if [[ -n "$COMM_HIGH_FREQ" ]]; then
        TRAIN_CMD+=(--comm-high-freq "$COMM_HIGH_FREQ")
    fi
fi

if [[ "$LAUNCHER" == "deepspeed" ]]; then
    LAUNCH_CMD=(deepspeed --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
    if [[ -n "$DS_INCLUDE" ]]; then
        # --include encodes both node and GPU selection; do not add num_nodes/num_gpus
        if [[ -n "$HOSTFILE" ]]; then
            LAUNCH_CMD+=(--hostfile "$HOSTFILE" --include "$DS_INCLUDE")
        else
            LAUNCH_CMD+=(--include "$DS_INCLUDE")
        fi
    else
        if [[ -n "$HOSTFILE" ]]; then
            LAUNCH_CMD+=(--hostfile "$HOSTFILE" --num_nodes "$NNODES" --num_gpus "$LOCAL_GPU_COUNT")
        else
            LAUNCH_CMD+=(--num_gpus "$LOCAL_GPU_COUNT")
        fi
    fi
    FINAL_CMD=("${LAUNCH_CMD[@]}" "${TRAIN_CMD[@]}")
elif [[ "$LAUNCHER" == "torchrun" ]]; then
    FINAL_CMD=(
        python3 -m torch.distributed.run
        --nproc_per_node "$LOCAL_GPU_COUNT"
        --nnodes "$NNODES"
        --node_rank "$NODE_RANK"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
        "${TRAIN_CMD[@]}"
    )
else
    echo "[Error] Unsupported LAUNCHER=${LAUNCHER}" >&2
    exit 1
fi

write_command_snapshot "${RUN_DIR}/command.sh" "${FINAL_CMD[@]}"
export MEGATRON_LAUNCH_COMMAND="$(printf '%q ' "${FINAL_CMD[@]}")"

cd "$BASE_PATH"
"${FINAL_CMD[@]}" 2>&1 | tee "${RUN_LOG_DIR}/${RUN_ID}.log"
