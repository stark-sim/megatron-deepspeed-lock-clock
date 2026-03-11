#!/bin/bash
set -euo pipefail

BASE_PATH="${BASE_PATH:-/home/sd/Megatron-DeepSpeed}"
source "${BASE_PATH}/scripts/experiment_utils.sh"

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

MODEL_SIZE="${MODEL_SIZE:-qwen7b}"
DS_CONFIG_TEMPLATE="${DS_CONFIG_TEMPLATE:-${BASE_PATH}/scripts/ds_config_7b_tp4.json}"
DATASET="${DATASET:-${BASE_PATH}/data/chinese_wiki_megatron_text_document}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${BASE_PATH}/checkpoints/${EXPERIMENT_NAME}}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-0}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

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
ZERO_STAGE="${ZERO_STAGE:-1}"
PRECISION_MODE="${PRECISION_MODE:-bf16}"

STATIC_CLOCK_MHZ="${STATIC_CLOCK_MHZ:-}"
COMM_LOW_FREQ="${COMM_LOW_FREQ:-1200}"
COMM_HIGH_FREQ="${COMM_HIGH_FREQ:-}"
COMM_MIN_ELEMENTS="${COMM_MIN_ELEMENTS:-300000000}"

LOCAL_GPU_INDICES="$(parse_cuda_visible_devices)"
LOCAL_GPU_COUNT="$(count_csv_items "$LOCAL_GPU_INDICES")"
WORLD_SIZE="$(( LOCAL_GPU_COUNT * NNODES ))"

setup_experiment_run "${BASE_PATH}" "${EXPERIMENT_NAME}"

RUN_DS_CONFIG="${RUN_DIR}/ds_config.json"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/${RUN_ID}}"
mkdir -p "${CHECKPOINT_PATH}"

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

if (( NNODES > 1 )) && [[ -z "$HOSTFILE" ]]; then
    echo "[Error] HOSTFILE is required when NNODES>1" >&2
    exit 1
fi

if (( PP > 1 )) && [[ -z "$HOSTFILE" ]]; then
    echo "[Error] HOSTFILE is required when PP>1" >&2
    exit 1
fi

if (( PP > 1 )) && [[ -z "$MASTER_ADDR" || -z "$MASTER_PORT" ]]; then
    echo "[Error] MASTER_ADDR and MASTER_PORT are required when PP>1" >&2
    exit 1
fi

cat > "$RUN_DS_CONFIG" <<JSON
{
  "train_batch_size": ${GLOBAL_BATCH_SIZE},
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": ${ZERO_STAGE}
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
    python3 - <<'PY'
import json
import os
import socket
import subprocess
import sys

sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')

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
if os.environ.get('LAUNCHER', 'deepspeed') == 'deepspeed':
    result['checks']['launcher_available'] = command_ok(['bash', '-lc', 'command -v deepspeed >/dev/null 2>&1'])[0]
else:
    result['checks']['launcher_available'] = command_ok(['python3', '-m', 'torch.distributed.run', '--help'])[0]
try:
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

LOCAL_PREFLIGHT="$(TP="$TP" LAUNCHER="$LAUNCHER" EXPERIMENT_MODE="$EXPERIMENT_MODE" STATIC_CLOCK_MHZ="$STATIC_CLOCK_MHZ" run_local_preflight)"
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
        local_host_short="${HOSTNAME%%.*}"
        if [[ "$host" == "localhost" || "$host" == "$HOSTNAME" || "$host" == "$local_host_short" ]]; then
            continue
        fi
        REMOTE_PREFLIGHTS+=("$(run_remote_preflight "$host" "$BASE_PATH" "$TP" "$EXPERIMENT_MODE" "$STATIC_CLOCK_MHZ" "${CUDA_VISIBLE_DEVICES:-}" "$LAUNCHER")")
    done
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

if [[ "$VALIDATE_ONLY" == "1" ]]; then
    echo "[Validate] topology and preflight succeeded"
    echo "[Validate] run_dir=${RUN_DIR}"
    exit 0
fi

cleanup() {
    if [[ "$EXPERIMENT_MODE" == "static" && -n "$STATIC_CLOCK_MHZ" ]]; then
        echo "[Cleanup] Resetting GPU clocks for ${LOCAL_GPU_INDICES}"
        reset_gpu_clocks "$LOCAL_GPU_INDICES"
    fi
}
trap cleanup EXIT

if [[ "$EXPERIMENT_MODE" == "static" ]]; then
    if [[ -z "$STATIC_CLOCK_MHZ" ]]; then
        echo "[Error] STATIC_CLOCK_MHZ is required for static mode" >&2
        exit 1
    fi
    assert_static_clock_supported "$LOCAL_GPU_INDICES" "$STATIC_CLOCK_MHZ"
    lock_gpu_clocks "$LOCAL_GPU_INDICES" "$STATIC_CLOCK_MHZ"
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
    --save "$CHECKPOINT_PATH"
    --data-path "$DATASET"
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
    --lr-warmup-iters 50
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --log-interval 1
    --save-interval "$TRAIN_STEPS"
    --eval-interval 100
    --eval-iters 10
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
    LAUNCH_CMD=(deepspeed)
    if [[ -n "$HOSTFILE" ]]; then
        LAUNCH_CMD+=(--hostfile "$HOSTFILE" --num_nodes "$NNODES" --num_gpus "$LOCAL_GPU_COUNT")
    else
        LAUNCH_CMD+=(--num_gpus "$LOCAL_GPU_COUNT")
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
