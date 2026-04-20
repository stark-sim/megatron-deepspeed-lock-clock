#!/usr/bin/env bash
# TP2PP2DP4 重新测量脚本 - 基于成功配置

set -euo pipefail
BASE=/home/sd/Megatron-DeepSpeed
TOKENIZER=$BASE/.context/qwen25_tokenizer_flat
HOSTFILE=/tmp/hostfile_tp2pp2dp4_remeasure

# 测试频率和重复次数
FREQS=(1072 1080 1087 1185 1200)
REPEATS=3

MASTER_ADDR=192.168.205.201
BASE_PORT=29700

export PATH=/home/sd/.local/bin:$PATH
export PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PDSH_RCMD_TYPE=ssh

mkdir -p "$BASE/experiments"

# 创建 hostfile
cat > "$HOSTFILE" <<'EOF'
localhost slots=8
192.168.205.202 slots=8
EOF

# 频率锁定函数
lock_local() {
  local clock="$1"
  sudo -n python3 - "$clock" <<'PY'
import sys, pynvml
clock=int(sys.argv[1])
pynvml.nvmlInit()
for i in range(8):
    h=pynvml.nvmlDeviceGetHandleByIndex(i)
    pynvml.nvmlDeviceSetGpuLockedClocks(h, clock, clock)
    print(f'lock local gpu {i} -> {clock}')
pynvml.nvmlShutdown()
PY
}

lock_remote() {
  local clock="$1"
  ssh 192.168.205.202 "sudo -n python3 - $clock <<'PY'
import sys, pynvml
clock=int(sys.argv[1])
pynvml.nvmlInit()
for i in range(8):
    h=pynvml.nvmlDeviceGetHandleByIndex(i)
    pynvml.nvmlDeviceSetGpuLockedClocks(h, clock, clock)
    print(f'lock remote gpu {i} -> {clock}')
pynvml.nvmlShutdown()
PY"
}

reset_local() {
  sudo -n python3 <<'PY'
import pynvml
pynvml.nvmlInit()
for i in range(8):
    h=pynvml.nvmlDeviceGetHandleByIndex(i)
    try: pynvml.nvmlDeviceResetGpuLockedClocks(h)
    except: pass
PY
}

reset_remote() {
  ssh 192.168.205.202 "sudo -n python3 <<'PY'
import pynvml
pynvml.nvmlInit()
for i in range(8):
    h=pynvml.nvmlDeviceGetHandleByIndex(i)
    try: pynvml.nvmlDeviceResetGpuLockedClocks(h)
    except: pass
pynvml.nvmlShutdown()
PY"
}

cleanup_procs() {
  python3 <<'PY'
import os, signal, subprocess
for line in subprocess.run(['ps','-ef'],capture_output=True,text=True).stdout.splitlines():
    if 'pretrain_gpt.py' in line or 'deepspeed.launcher.launch' in line:
        try: os.kill(int(line.split()[1]), signal.SIGKILL)
        except: pass
PY
  ssh 192.168.205.202 "python3 <<'PY'
import os, signal, subprocess
for line in subprocess.run(['ps','-ef'],capture_output=True,text=True).stdout.splitlines():
    if 'pretrain_gpt.py' in line or 'deepspeed.launcher.launch' in line:
        try: os.kill(int(line.split()[1]), signal.SIGKILL)
        except: pass
PY"
}

trap 'reset_local; reset_remote; cleanup_procs' EXIT

# 运行单个实验
run_one() {
  local clock="$1"
  local port="$2"
  local repeat="$3"
  local run_id="tp2pp2dp4_static${clock}_r${repeat}_$(date +%Y%m%d_%H%M%S)_DGX2-1"
  local run_dir="$BASE/experiments/$run_id"
  local log_dir="$run_dir/logs"
  local dscfg="$BASE/.context/dsconfig_tp2pp2dp4_${clock}_r${repeat}.json"
  
  mkdir -p "$run_dir" "$log_dir"
  
  cat > "$dscfg" <<JSON
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 1,
  "steps_per_print": 1,
  "zero_optimization": {"stage": 1},
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0
}
JSON
  
  ssh 192.168.205.202 "mkdir -p '$BASE/.context' '$run_dir' '$log_dir'" </dev/null
  
  echo "[run] freq=$clock port=$port repeat=$repeat run_id=$run_id"
  
  lock_local "$clock"
  lock_remote "$clock"
  
  (cd "$BASE" && deepspeed --hostfile "$HOSTFILE" --num_nodes 2 --num_gpus 8 \
    --master_addr "$MASTER_ADDR" --master_port "$port" \
    pretrain_gpt.py \
    --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 \
    --num-layers 28 --hidden-size 3584 --ffn-hidden-size 18944 \
    --num-attention-heads 28 --num-key-value-heads 4 \
    --micro-batch-size 1 --global-batch-size 16 --seq-length 2048 \
    --max-position-embeddings 2048 --train-iters 20 \
    --data-path "$BASE/data/chinese_wiki_megatron_text_document" \
    --data-impl mmap --tokenizer-type HFTokenizer --tokenizer-model "$TOKENIZER" \
    --split 98,2,0 --distributed-backend nccl --lr 1e-5 \
    --lr-decay-style cosine --min-lr 1e-6 --weight-decay 0.01 \
    --clip-grad 1.0 --lr-warmup-iters 19 --optimizer adam \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --log-interval 1 --save-interval 0 --eval-interval 100 --eval-iters 10 \
    --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 \
    --use-rotary-position-embeddings --untie-embeddings-and-output-weights \
    --swiglu --normalization rmsnorm --disable-bias-linear \
    --no-position-embedding --no-masked-softmax-fusion \
    --no-bias-gelu-fusion --no-bias-dropout-fusion \
    --recompute-granularity full --recompute-method uniform \
    --deepspeed-activation-checkpointing --zero-stage=1 \
    --deepspeed_config="$dscfg" --deepspeed --bf16 \
    --experiment-run-id "$run_id" \
    --experiment-name "tp2pp2dp4_static${clock}_r${repeat}" \
    --experiment-root-dir "$BASE/experiments" \
    | tee "$log_dir/$run_id.log")
  
  reset_local
  reset_remote
  
  echo "[run] completed $run_id"
}

# 主循环
echo "=========================================="
echo "TP2PP2DP4 重新测量"
echo "频率: ${FREQS[@]}"
echo "每频率重复: $REPEATS 次"
echo "开始时间: $(date)"
echo "=========================================="

idx=0
for freq in "${FREQS[@]}"; do
  for rep in $(seq 1 $REPEATS); do
    port=$((BASE_PORT + idx))
    echo ""
    echo "[$((idx+1))/$((5*REPEATS))] freq=$freq repeat=$rep port=$port"
    run_one "$freq" "$port" "$rep"
    ((idx++))
    sleep 5
  done
done

echo ""
echo "=========================================="
echo "全部完成: $(date)"
echo "=========================================="
