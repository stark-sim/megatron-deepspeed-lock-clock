#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
BASE="/home/sd/Megatron-DeepSpeed"
BATCH_ID="ib_topology_static_compare_tpge2_kv4_20260419"
HOSTS_TEXT=$'v100x16-1 slots=4\nv100x16-2 slots=4\n'
TOPO_NAMES=(tp2pp2dp2 tp4pp2dp1)
TPS=(2 4)
PPS=(2 2)
MODES=(baseline static static static)
CLOCKS=("" 990 1080 1155)
PORT_BASE=30731

cd "${BASE}"

export PATH="/home/sd/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin:${PATH}"
export PYTHON_BIN="/usr/bin/python3"
export PYTHONPATH="${BASE}:/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export CUDA_HOME="/usr/local/cuda"
export CUDACXX="/usr/local/cuda/bin/nvcc"
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
export NNODES="2"
export DS_INCLUDE="v100x16-1:8,9,10,11@v100x16-2:8,9,10,11"
export LOCAL_GPU_INDICES="8,9,10,11"
export MASTER_ADDR="192.168.205.201"
export HIDDEN_SIZE="2048"
export FFN_HIDDEN_SIZE="11008"
export NUM_LAYERS="36"
export NUM_HEADS="16"
export NUM_KV_HEADS="4"
export MICRO_BATCH_SIZE="1"
export GLOBAL_BATCH_SIZE="4"
export TRAIN_STEPS="20"
export ZERO_STAGE="1"
export DISABLE_CHECKPOINT="1"
export EVAL_INTERVAL="100"
export EVAL_ITERS="0"
export SAVE_INTERVAL="0"
export DATASET="${BASE}/data/qwen_data_text_document"
export TOKENIZER_PATH="${BASE}/.context/qwen25_tokenizer_flat"
export DATA_CACHE_PATH="${BASE}/data/index-cache"
export TORCH_EXTENSIONS_DIR="/dev/shm/megatron_tpge2_static_20260419/torch_extensions_v100"
export TMPDIR="/dev/shm/megatron_tpge2_static_20260419/tmp"
export PYTHONPYCACHEPREFIX="/dev/shm/megatron_tpge2_static_20260419/pycache"
export PYTHONDONTWRITEBYTECODE="1"
export MAX_JOBS="8"
export USE_CPU_OPTIMIZER="1"
export OFFLOAD_OPTIMIZER_DEVICE="cpu"
export OFFLOAD_OPTIMIZER_PIN_MEMORY="0"
export EXPERIMENT_NAME="${BATCH_ID}"

mkdir -p "${DATA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}" "${PYTHONPYCACHEPREFIX}"

TS="$(date +%Y%m%d_%H%M%S)"
MANIFEST="${BASE}/experiments/${BATCH_ID}_manifest_${TS}.txt"

echo "BATCH_ID=${BATCH_ID}" | tee -a "${MANIFEST}"
echo "MANIFEST=${MANIFEST}" | tee -a "${MANIFEST}"
echo "MODEL=36L hidden2048 ffn11008 heads16 kv4" | tee -a "${MANIFEST}"
echo "DATASET=${DATASET}" | tee -a "${MANIFEST}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}" | tee -a "${MANIFEST}"
echo "TRAIN_STEPS=${TRAIN_STEPS}" | tee -a "${MANIFEST}"
echo "START_AT=$(date '+%F %T')" | tee -a "${MANIFEST}"

run_one() {
    local topo_name="$1"
    local tp="$2"
    local pp="$3"
    local mode="$4"
    local clock="$5"
    local port="$6"
    local label run_ts run_id

    label="${mode}"
    if [[ -n "${clock}" ]]; then
        label="${mode}${clock}"
    fi

    run_ts="$(date +%Y%m%d_%H%M%S)"
    run_id="${BATCH_ID}_${topo_name}_${label}_${run_ts}_v100x16-1"

    export RUN_ID="${run_id}"
    export TP="${tp}"
    export PP="${pp}"
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
        echo "TOPOLOGY=${topo_name}"
        echo "TP=${TP}"
        echo "PP=${PP}"
        echo "EXPECTED_DP=$(( 8 / (tp * pp) ))"
        echo "MODE=${EXPERIMENT_MODE}"
        echo "STATIC_CLOCK_MHZ=${STATIC_CLOCK_MHZ:-baseline}"
        echo "MASTER_PORT=${MASTER_PORT}"
        echo "HOSTFILE=${HOSTFILE}"
        echo "START_AT=$(date '+%F %T')"
    } | tee -a "${MANIFEST}"

    timeout 7200 ./scripts/run_experiment.sh 2>&1 | tee -a "${MANIFEST}"

    echo "END_AT=$(date '+%F %T')" | tee -a "${MANIFEST}"
    echo | tee -a "${MANIFEST}"

    unset RUN_ID TP PP HOSTFILE VALIDATE_ONLY MASTER_PORT
}

if [[ "${MODE}" == "validate" ]]; then
    export RUN_ID="${BATCH_ID}_validate_$(date +%Y%m%d_%H%M%S)_v100x16-1"
    export TP="${TPS[0]}"
    export PP="${PPS[0]}"
    export MASTER_PORT="${PORT_BASE}"
    export HOSTFILE="/tmp/hostfile_${RUN_ID}"
    export EXPERIMENT_MODE="baseline"
    export VALIDATE_ONLY="1"
    unset STATIC_CLOCK_MHZ
    printf '%s' "${HOSTS_TEXT}" > "${HOSTFILE}"
    ./scripts/run_experiment.sh
    exit 0
fi

for topo_idx in "${!TOPO_NAMES[@]}"; do
    for mode_idx in "${!MODES[@]}"; do
        run_one \
            "${TOPO_NAMES[$topo_idx]}" \
            "${TPS[$topo_idx]}" \
            "${PPS[$topo_idx]}" \
            "${MODES[$mode_idx]}" \
            "${CLOCKS[$mode_idx]}" \
            "$(( PORT_BASE + topo_idx * 10 + mode_idx ))"
        sleep 10
    done
done

echo "FINISHED_AT=$(date '+%F %T')" | tee -a "${MANIFEST}"
