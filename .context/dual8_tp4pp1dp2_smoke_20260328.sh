#!/usr/bin/env bash
set -euo pipefail
BASE=/home/sd/Megatron-DeepSpeed
TOKENIZER=$BASE/.context/qwen25_tokenizer_flat
HOSTFILE=/tmp/hostfile_tp4pp1dp2_dual8_smoke
CLOCK=990
MASTER_ADDR=192.168.205.201
MASTER_PORT=29990
GPU_IDS=(8 9 10 11)
CUDA_DEVICES=$(IFS=,; echo "${GPU_IDS[*]}")
RUN_ID="dual8_tp4pp1dp2_smoke990_$(date +%Y%m%d_%H%M%S)_DGX2-1"
RUN_DIR="$BASE/experiments/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
DSCFG="$BASE/.context/dsconfig_dual8_tp4pp1dp2_smoke990_20260328.json"
mkdir -p "$BASE/.context" "$RUN_DIR" "$LOG_DIR"
export PATH=/home/sd/.local/bin:$PATH
export PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/sd/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
cat > "$HOSTFILE" <<HF
localhost slots=4
192.168.205.202 slots=4
HF
cat > "$DSCFG" <<JSON
{
  "train_batch_size": 8,
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

sudo -n python3 - "$CLOCK" "$CUDA_DEVICES" <<'PY'
import sys
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml
clock = int(sys.argv[1])
gpu_ids = [int(x) for x in sys.argv[2].split(',')]
pynvml.nvmlInit()
try:
  for i in gpu_ids:
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    pynvml.nvmlDeviceSetGpuLockedClocks(h, clock, clock)
finally:
  pynvml.nvmlShutdown()
PY
ssh 192.168.205.202 "sudo -n python3 - $CLOCK $CUDA_DEVICES <<'PY'
import sys
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml
clock = int(sys.argv[1])
gpu_ids = [int(x) for x in sys.argv[2].split(',')]
pynvml.nvmlInit()
try:
  for i in gpu_ids:
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    pynvml.nvmlDeviceSetGpuLockedClocks(h, clock, clock)
finally:
  pynvml.nvmlShutdown()
PY"

set +e
(cd "$BASE" && deepspeed --hostfile "$HOSTFILE" --num_nodes 2 --num_gpus 4 --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" pretrain_gpt.py \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --num-layers 28 \
  --hidden-size 3584 \
  --ffn-hidden-size 18944 \
  --num-attention-heads 28 \
  --num-key-value-heads 4 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
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
  --eval-iters 1 \
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
  --experiment-name "dual8_tp4pp1dp2_smoke990" \
  --experiment-root-dir "$BASE/experiments" \
  --bf16 2>&1 | tee "$LOG_DIR/$RUN_ID.log")
RC=$?
set -e

echo "RC=$RC RUN_ID=$RUN_ID"
echo "$RUN_ID" > "$BASE/.context/dual8_tp4pp1dp2_smoke_last_runid.txt"

sudo -n python3 - "$CUDA_DEVICES" <<'PY'
import sys
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml
gpu_ids = [int(x) for x in sys.argv[1].split(',')]
pynvml.nvmlInit()
try:
  for i in gpu_ids:
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    try: pynvml.nvmlDeviceResetGpuLockedClocks(h)
    except Exception: pass
    try: pynvml.nvmlDeviceResetApplicationsClocks(h)
    except Exception: pass
finally:
  pynvml.nvmlShutdown()
PY
ssh 192.168.205.202 "sudo -n python3 - $CUDA_DEVICES <<'PY'
import sys
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml
gpu_ids = [int(x) for x in sys.argv[1].split(',')]
pynvml.nvmlInit()
try:
  for i in gpu_ids:
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    try: pynvml.nvmlDeviceResetGpuLockedClocks(h)
    except Exception: pass
    try: pynvml.nvmlDeviceResetApplicationsClocks(h)
    except Exception: pass
finally:
  pynvml.nvmlShutdown()
PY"

exit "$RC"
