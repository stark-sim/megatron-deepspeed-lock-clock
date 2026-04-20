#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
BASE="/home/user/Megatron-DeepSpeed"
BATCH_ID="eth_dual4_common_qwen3b_tp1pp2dp4_20260419"
HOSTS_TEXT=$'sd-1 slots=4\nsd-2 slots=4\n'
MODES=(baseline static static static)
CLOCKS=("" 1005 1200 1395)
PORTS=(30201 30202 30203 30204)

cd "${BASE}"

export PATH="/home/user/miniconda3/envs/tp4bit/bin:/home/user/.local/bin:${PATH}"
export PYTHON_BIN="/home/user/miniconda3/envs/tp4bit/bin/python3.10"
export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
export CONDA_ENV="tp4bit"
export PDSH_RCMD_TYPE="ssh"
export GLOO_SOCKET_IFNAME="eth0"
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_DISABLE="1"
export NCCL_DEBUG="WARN"
export NCCL_RAS_ENABLE="0"
export TORCH_NCCL_BLOCKING_WAIT="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LAUNCHER="deepspeed"
export TP="1"
export PP="2"
export NNODES="2"
export DS_INCLUDE="sd-1:0,1,2,3@sd-2:0,1,2,3"
export LOCAL_GPU_INDICES="0,1,2,3"
export MASTER_ADDR="192.168.66.121"
export HIDDEN_SIZE="2048"
export FFN_HIDDEN_SIZE="11008"
export NUM_LAYERS="36"
export NUM_HEADS="16"
export NUM_KV_HEADS="2"
export MICRO_BATCH_SIZE="1"
export GLOBAL_BATCH_SIZE="4"
export TRAIN_STEPS="20"
export ZERO_STAGE="1"
export DISABLE_CHECKPOINT="1"
export EVAL_INTERVAL="100"
export EVAL_ITERS="0"
export SAVE_INTERVAL="0"
export DATASET="${BASE}/data/qwen_data_text_document"
export DATA_CACHE_PATH="${BASE}/data/index-cache"
export TORCH_EXTENSIONS_DIR="/dev/shm/megatron_common_qwen3b_20260419/torch_extensions_tp4bit"
export TMPDIR="/dev/shm/megatron_common_qwen3b_20260419/tmp"
export PYTHONPYCACHEPREFIX="/dev/shm/megatron_common_qwen3b_20260419/pycache"
export PYTHONDONTWRITEBYTECODE="1"
export USE_CPU_OPTIMIZER="1"
export OFFLOAD_OPTIMIZER_DEVICE="cpu"
export OFFLOAD_OPTIMIZER_PIN_MEMORY="0"
export EXPERIMENT_NAME="${BATCH_ID}"

mkdir -p "${DATA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}" "${PYTHONPYCACHEPREFIX}"

TS="$(date +%Y%m%d_%H%M%S)"
MANIFEST="${BASE}/experiments/${BATCH_ID}_manifest_${TS}.txt"

echo "BATCH_ID=${BATCH_ID}" | tee -a "${MANIFEST}"
echo "MANIFEST=${MANIFEST}" | tee -a "${MANIFEST}"
echo "START_AT=$(date '+%F %T')" | tee -a "${MANIFEST}"

run_one() {
    local mode="$1"
    local clock="$2"
    local port="$3"
    local label run_ts run_id

    label="${mode}"
    if [[ -n "${clock}" ]]; then
        label="${mode}${clock}"
    fi

    run_ts="$(date +%Y%m%d_%H%M%S)"
    run_id="${BATCH_ID}_${label}_${run_ts}_sd-1"

    export RUN_ID="${run_id}"
    export MASTER_PORT="${port}"
    export HOSTFILE="/tmp/hostfile_${RUN_ID}"
    export VALIDATE_ONLY="0"

    if [[ "${mode}" == "baseline" ]]; then
        export EXPERIMENT_MODE="baseline"
        unset STATIC_CLOCK_MHZ
    else
        export EXPERIMENT_MODE="static"
        export STATIC_CLOCK_MHZ="${clock}"
    fi

    printf '%s' "${HOSTS_TEXT}" > "${HOSTFILE}"

    {
        echo "RUN_ID=${RUN_ID}"
        echo "MODE=${EXPERIMENT_MODE}"
        echo "STATIC_CLOCK_MHZ=${STATIC_CLOCK_MHZ:-baseline}"
        echo "MASTER_PORT=${MASTER_PORT}"
        echo "HOSTFILE=${HOSTFILE}"
        echo "START_AT=$(date '+%F %T')"
    } | tee -a "${MANIFEST}"

    timeout 7200 ./scripts/run_experiment.sh 2>&1 | tee -a "${MANIFEST}"

    echo "END_AT=$(date '+%F %T')" | tee -a "${MANIFEST}"
    echo | tee -a "${MANIFEST}"

    unset RUN_ID HOSTFILE VALIDATE_ONLY
}

if [[ "${MODE}" == "validate" ]]; then
    export RUN_ID="${BATCH_ID}_validate_$(date +%Y%m%d_%H%M%S)_sd-1"
    export MASTER_PORT="${PORTS[0]}"
    export HOSTFILE="/tmp/hostfile_${RUN_ID}"
    export EXPERIMENT_MODE="baseline"
    export VALIDATE_ONLY="1"
    unset STATIC_CLOCK_MHZ
    printf '%s' "${HOSTS_TEXT}" > "${HOSTFILE}"
    ./scripts/run_experiment.sh
    exit 0
fi

for idx in "${!MODES[@]}"; do
    run_one "${MODES[$idx]}" "${CLOCKS[$idx]}" "${PORTS[$idx]}"
    sleep 10
done

echo "FINISHED_AT=$(date '+%F %T')" | tee -a "${MANIFEST}"
