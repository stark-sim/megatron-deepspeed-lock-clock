#!/usr/bin/env bash
set -euo pipefail
BASE=/home/sd/Megatron-DeepSpeed
TOKENIZER=$BASE/.context/qwen25_tokenizer_flat
HOSTFILE=/tmp/hostfile_dual32_smoke_20260325
MASTER_ADDR=192.168.205.201
MASTER_PORT=29825
CLOCK=1050
RUN_ID="dual32_tp4pp1dp8_smoke1050_$(date +%Y%m%d_%H%M%S)_DGX2-1"
RUN_DIR="$BASE/experiments/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
DSCFG="$BASE/.context/dsconfig_${RUN_ID}.json"
export PATH=/home/sd/.local/bin:$PATH
export PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/sd/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PDSH_RCMD_TYPE=ssh
cat > "$HOSTFILE" <<HF
localhost slots=16
192.168.205.202 slots=16
HF
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
    for index in range(16):
        h=pynvml.nvmlDeviceGetHandleByIndex(index)
        pynvml.nvmlDeviceSetGpuLockedClocks(h, clock, clock)
        print(f'lock {index} -> {clock}')
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
    for index in range(16):
        h=pynvml.nvmlDeviceGetHandleByIndex(index)
        try: pynvml.nvmlDeviceResetGpuLockedClocks(h)
        except Exception: pass
        try: pynvml.nvmlDeviceResetApplicationsClocks(h)
        except Exception: pass
        print(f'reset {index}')
finally:
    pynvml.nvmlShutdown()
PY"
}
cleanup_procs() {
  /usr/bin/python3 - <<'PY'
import os, signal, subprocess
for line in subprocess.run(['ps','-ef'], capture_output=True, text=True).stdout.splitlines():
    if ('pretrain_gpt.py' in line or 'deepspeed.launcher.launch' in line or 'pdsh -S -f 1024' in line) and 'python3 - <<' not in line:
        try: os.kill(int(line.split()[1]), signal.SIGKILL)
        except Exception: pass
PY
  ssh 192.168.205.202 "/usr/bin/python3 - <<'PY'
import os, signal, subprocess
for line in subprocess.run(['ps','-ef'], capture_output=True, text=True).stdout.splitlines():
    if ('pretrain_gpt.py' in line or 'deepspeed.launcher.launch' in line or 'pdsh -S -f 1024' in line) and 'python3 - <<' not in line:
        try: os.kill(int(line.split()[1]), signal.SIGKILL)
        except Exception: pass
PY"
}
trap 'set +e; reset_node localhost; reset_node 192.168.205.202; cleanup_procs' EXIT
mkdir -p "$RUN_DIR" "$LOG_DIR"
cat > "$DSCFG" <<JSON
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 1,
  "steps_per_print": 1,
  "zero_optimization": {"stage": 1},
  "bf16": {"enabled": true},
  "fp16": {"enabled": false},
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "wall_clock_breakdown": false
}
JSON
ssh 192.168.205.202 "mkdir -p '$BASE/.context' '$RUN_DIR' '$LOG_DIR' && cat > '$DSCFG'" < "$DSCFG"
echo "[dual32-smoke] run_id=$RUN_ID" | tee "$LOG_DIR/$RUN_ID.log"
lock_node localhost "$CLOCK"
lock_node 192.168.205.202 "$CLOCK"
cd "$BASE"
deepspeed --hostfile "$HOSTFILE" --num_nodes 2 --num_gpus 16 --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" pretrain_gpt.py \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --num-layers 28 \
  --hidden-size 3584 \
  --ffn-hidden-size 18944 \
  --num-attention-heads 28 \
  --num-key-value-heads 4 \
  --micro-batch-size 1 \
  --global-batch-size 32 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --train-iters 2 \
  --data-path "$BASE/data/chinese_wiki_megatron_text_document" \
  --data-impl mmap \
  --tokenizer-type HFTokenizer \
  --tokenizer-model "$TOKENIZER" \
  --split 98,2,0 \
  --distributed-backend nccl \
  --lr 1e-5 \
  --lr-decay-style cosine \
  --min-lr 1e-6 \
  --weight-decay 0.01 \
  --clip-grad 1.0 \
  --lr-warmup-iters 1 \
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
  --experiment-name dual32_tp4pp1dp8_smoke1050 \
  --experiment-root-dir "$BASE/experiments" \
  --bf16 2>&1 | tee -a "$LOG_DIR/$RUN_ID.log"
