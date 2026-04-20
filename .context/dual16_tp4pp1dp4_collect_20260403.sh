#!/usr/bin/env bash
# =============================================================================
# 2x8 (dual16) clean source collection with IB enabled, GPUs 8-15
# =============================================================================
set -euo pipefail

BASE="/home/sd/Megatron-DeepSpeed"
PEER="192.168.205.202"
SESSION="dual16_tp4pp1dp4_collect_20260403"
LOG="$BASE/.context/${SESSION}.log"
HOSTFILE="/tmp/hostfile_${SESSION}"

TARGET_GPUS="8,9,10,11,12,13,14,15"
NODE_LIST="localhost,${PEER}"
DS_INCLUDE="localhost:${TARGET_GPUS}@${PEER}:${TARGET_GPUS}"

TP=4
PP=1
NNODES=2
GBS=16
MBS=1
TRAIN_STEPS=20
MASTER_ADDR="192.168.205.201"
MASTER_PORT="29960"

FREQUENCIES=(990 1080 1155)

export PATH="/home/sd/.local/bin:${PATH}"
export PYTHONPATH="/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export NCCL_RAS_ENABLE="0"
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT="1"
export PDSH_RCMD_TYPE="ssh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$BASE/.context"

echo "============================================" | tee -a "$LOG"
echo "[collect] $(date '+%Y-%m-%d %H:%M:%S') Starting 2x8 collection" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# Preflight
if ! "$BASE/scripts/preflight_check.sh" --node-list "$NODE_LIST" --gpu-indices "$TARGET_GPUS" >> "$LOG" 2>&1; then
    echo "[FATAL] Preflight failed. Aborting." | tee -a "$LOG"
    exit 1
fi

cat > "$HOSTFILE" <<EOF
localhost slots=8
${PEER} slots=8
EOF

lock_node() {
    local target="$1"
    local clock="$2"
    ssh "$target" "sudo -n /usr/bin/python3 - $clock <<'PY'
import sys
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml
clock=int(sys.argv[1])
pynvml.nvmlInit()
try:
    for index in [8,9,10,11,12,13,14,15]:
        h=pynvml.nvmlDeviceGetHandleByIndex(index)
        pynvml.nvmlDeviceSetGpuLockedClocks(h, clock, clock)
        print(f'lock GPU{index} -> {clock} MHz')
finally:
    pynvml.nvmlShutdown()
PY"
}

reset_node() {
    local target="$1"
    ssh "$target" "sudo -n /usr/bin/python3 - <<'PY'
import sys
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml
pynvml.nvmlInit()
try:
    for index in [8,9,10,11,12,13,14,15]:
        h=pynvml.nvmlDeviceGetHandleByIndex(index)
        try: pynvml.nvmlDeviceResetGpuLockedClocks(h)
        except Exception: pass
        try: pynvml.nvmlDeviceResetApplicationsClocks(h)
        except Exception: pass
        print(f'reset GPU{index}')
finally:
    pynvml.nvmlShutdown()
PY"
}

cleanup_procs() {
    local target="${1:-localhost}"
    if [[ "$target" == "localhost" ]]; then
        /usr/bin/python3 - <<'PY'
import os, signal, subprocess
for line in subprocess.run(['ps','-ef'], capture_output=True, text=True).stdout.splitlines():
    if ('pretrain_gpt.py' in line or 'deepspeed.launcher.launch' in line or 'pdsh -S -f 1024' in line) and 'python3 - <<' not in line:
        try: os.kill(int(line.split()[1]), signal.SIGKILL)
        except Exception: pass
PY
    else
        ssh "$target" "/usr/bin/python3 - <<'PY'
import os, signal, subprocess
for line in subprocess.run(['ps','-ef'], capture_output=True, text=True).stdout.splitlines():
    if ('pretrain_gpt.py' in line or 'deepspeed.launcher.launch' in line or 'pdsh -S -f 1024' in line) and 'python3 - <<' not in line:
        try: os.kill(int(line.split()[1]), signal.SIGKILL)
        except Exception: pass
PY"
    fi
}

trap 'set +e; reset_node localhost; reset_node "$PEER"; cleanup_procs localhost; cleanup_procs "$PEER"' EXIT

for CLOCK in "${FREQUENCIES[@]}"; do
    RUN_ID="dual16_tp4pp1dp4_static${CLOCK}_$(date +%Y%m%d_%H%M%S)_DGX2-1"
    RUN_DIR="$BASE/experiments/$RUN_ID"
    LOG_DIR="$RUN_DIR/logs"
    DSCFG="$BASE/.context/dsconfig_${RUN_ID}.json"

    echo "" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"
    echo "[collect] Freq=${CLOCK}MHz | run_id=${RUN_ID}" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"

    cat > "$DSCFG" <<JSON
{
  "train_batch_size": ${GBS},
  "train_micro_batch_size_per_gpu": ${MBS},
  "steps_per_print": 1,
  "zero_optimization": {"stage": 1},
  "bf16": {"enabled": true},
  "fp16": {"enabled": false},
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "wall_clock_breakdown": false
}
JSON
    ssh "$PEER" "mkdir -p '$BASE/.context' && cat > '$DSCFG'" < "$DSCFG"
    mkdir -p "$RUN_DIR" "$LOG_DIR"

    lock_node localhost "$CLOCK"
    lock_node "$PEER" "$CLOCK"

    cd "$BASE"
    deepspeed \
        --hostfile "$HOSTFILE" \
        --include "$DS_INCLUDE" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        pretrain_gpt.py \
        --tensor-model-parallel-size "$TP" \
        --pipeline-model-parallel-size "$PP" \
        --num-layers 28 \
        --hidden-size 3584 \
        --ffn-hidden-size 18944 \
        --num-attention-heads 28 \
        --num-key-value-heads 4 \
        --micro-batch-size "$MBS" \
        --global-batch-size "$GBS" \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --train-iters "$TRAIN_STEPS" \
        --data-path "$BASE/data/chinese_wiki_megatron_text_document" \
        --data-impl mmap \
        --tokenizer-type HFTokenizer \
        --tokenizer-model "$BASE/.context/qwen25_tokenizer_flat" \
        --split 98,2,0 \
        --distributed-backend nccl \
        --lr 1e-5 \
        --lr-decay-style cosine \
        --min-lr 1e-6 \
        --weight-decay 0.01 \
        --clip-grad 1.0 \
        --lr-warmup-iters 19 \
        --optimizer adam \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --adam-eps 1e-8 \
        --log-interval 1 \
        --save-interval 0 \
        --eval-interval 100 \
        --eval-iters 10 \
        --no-query-key-layer-scaling \
        --attention-dropout 0 \
        --hidden-dropout 0 \
        --use-rotary-position-embeddings \
        --untie-embeddings-and-output-weights \
        --swiglu \
        --normalization rmsnorm \
        --disable-bias-linear \
        --no-position-embedding \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion \
        --recompute-granularity full \
        --recompute-method uniform \
        --deepspeed-activation-checkpointing \
        --zero-stage=1 \
        --deepspeed_config="$DSCFG" \
        --deepspeed \
        --experiment-run-id "$RUN_ID" \
        --experiment-name "dual16_tp4pp1dp4_static${CLOCK}" \
        --experiment-root-dir "$BASE/experiments" \
        --bf16 2>&1 | tee -a "$LOG_DIR/$RUN_ID.log"

    echo "[collect] Freq=${CLOCK}MHz completed." | tee -a "$LOG"
    reset_node localhost
    reset_node "$PEER"
    cleanup_procs localhost
    cleanup_procs "$PEER"
    sleep 15
done

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "[collect] All 2x8 collection completed." | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
