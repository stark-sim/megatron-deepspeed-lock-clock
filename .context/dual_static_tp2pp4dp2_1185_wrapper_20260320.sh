#!/usr/bin/env bash
set -euo pipefail
BASE=/home/sd/Megatron-DeepSpeed
RUN_ID=dual_tp2pp4dp2_static1185_fix_20260320_$(date +%Y%m%d_%H%M%S)_DGX2-1
RUN_DIR=$BASE/experiments/$RUN_ID
LOG_DIR=$RUN_DIR/logs
HOSTFILE=/tmp/hostfile_tp2pp4dp2_dual
DSCFG=$BASE/.context/dsconfig_dual_tp2pp4dp2_static1185_20260320.json
TOKENIZER=$BASE/.context/qwen25_tokenizer_flat
CLOCK=1185
export PATH=/home/sd/.local/bin:$PATH
export PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p "$RUN_DIR" "$LOG_DIR" "$BASE/.context"
cat > "$HOSTFILE" <<HF
localhost slots=8
192.168.205.202 slots=8
HF
cat > "$DSCFG" <<JSON
{
  "train_batch_size": 16,
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
lock_local() {
sudo -n python3 - "$CLOCK" <<'PY'
import sys
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml
clock=int(sys.argv[1])
pynvml.nvmlInit()
try:
    for index in range(8):
        h=pynvml.nvmlDeviceGetHandleByIndex(index)
        pynvml.nvmlDeviceSetGpuLockedClocks(h, clock, clock)
        print(f'lock local gpu {index} -> {clock}')
finally:
    pynvml.nvmlShutdown()
PY
}
lock_remote() {
ssh 192.168.205.202 'sudo -n python3 - 1185 <<'"'"'PY'"'"'
import sys
sys.path.insert(0, "/home/sd/.local/lib/python3.10/site-packages")
import pynvml
clock=int(sys.argv[1])
pynvml.nvmlInit()
try:
    for index in range(8):
        h=pynvml.nvmlDeviceGetHandleByIndex(index)
        pynvml.nvmlDeviceSetGpuLockedClocks(h, clock, clock)
        print(f"lock remote gpu {index} -> {clock}")
finally:
    pynvml.nvmlShutdown()
PY'
}
reset_local() {
sudo -n python3 - <<'PY'
import sys
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml
pynvml.nvmlInit()
try:
    for index in range(8):
        h=pynvml.nvmlDeviceGetHandleByIndex(index)
        try: pynvml.nvmlDeviceResetGpuLockedClocks(h)
        except Exception: pass
        try: pynvml.nvmlDeviceResetApplicationsClocks(h)
        except Exception: pass
        print(f'reset local gpu {index}')
finally:
    pynvml.nvmlShutdown()
PY
}
reset_remote() {
ssh 192.168.205.202 'sudo -n python3 - <<'"'"'PY'"'"'
import sys
sys.path.insert(0, "/home/sd/.local/lib/python3.10/site-packages")
import pynvml
pynvml.nvmlInit()
try:
    for index in range(8):
        h=pynvml.nvmlDeviceGetHandleByIndex(index)
        try: pynvml.nvmlDeviceResetGpuLockedClocks(h)
        except Exception: pass
        try: pynvml.nvmlDeviceResetApplicationsClocks(h)
        except Exception: pass
        print(f"reset remote gpu {index}")
finally:
    pynvml.nvmlShutdown()
PY'
}
trap 'reset_local; reset_remote' EXIT
lock_local
lock_remote
cd "$BASE"
echo "[wrapper] run_id=$RUN_ID"
echo "[wrapper] ds_config=$DSCFG"
deepspeed --hostfile "$HOSTFILE" --num_nodes 2 --num_gpus 8 --master_addr 192.168.205.201 --master_port 29610 pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 4 \
  --num-layers 28 \
  --hidden-size 3584 \
  --ffn-hidden-size 18944 \
  --num-attention-heads 28 \
  --num-key-value-heads 4 \
  --micro-batch-size 1 \
  --global-batch-size 16 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --train-iters 20 \
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
  --experiment-name dual_tp2pp4dp2_static1185_fix_20260320 \
  --experiment-root-dir "$BASE/experiments" \
  --bf16 | tee "$LOG_DIR/$RUN_ID.log"
