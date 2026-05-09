#!/usr/bin/env bash
# TP2PP2DP4 重新测量脚本 V2 - 使用 nvidia-smi 锁定频率

set -euo pipefail
BASE=/home/sd/Megatron-DeepSpeed
TOKENIZER=$BASE/.context/qwen25_tokenizer_flat
HOSTFILE=/tmp/hostfile_tp2pp2dp4_v2

FREQS=(1072 1080 1087 1185 1200)
REPEATS=3

MASTER_ADDR=192.168.205.201
BASE_PORT=29800

export PATH=/home/sd/.local/bin:$PATH
export PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PDSH_RCMD_TYPE=ssh

mkdir -p "$BASE/experiments"

cat > "$HOSTFILE" <<'EOF'
localhost slots=8
192.168.205.202 slots=8
EOF

# 使用 nvidia-smi 锁定频率
lock_gpus() {
  local clock="$1"
  echo "Locking GPUs to ${clock}MHz..."
  sudo -n nvidia-smi -lgc "$clock" 2>/dev/null || echo "Warning: Failed to lock local GPUs"
  ssh 192.168.205.202 "sudo -n nvidia-smi -lgc $clock" 2>/dev/null || echo "Warning: Failed to lock remote GPUs"
}

reset_gpus() {
  echo "Resetting GPUs..."
  sudo -n nvidia-smi -rgc 2>/dev/null || true
  ssh 192.168.205.202 "sudo -n nvidia-smi -rgc" 2>/dev/null || true
}

cleanup_procs() {
  python3 <<'PY' 2>/dev/null || true
import os, signal, subprocess
for line in subprocess.run(['ps','-ef'],capture_output=True,text=True).stdout.splitlines():
    if 'pretrain_gpt.py' in line or 'deepspeed.launcher.launch' in line:
        try: os.kill(int(line.split()[1]), signal.SIGKILL)
        except: pass
PY
  ssh 192.168.205.202 "python3 <<'PY' 2>/dev/null || true
import os, signal, subprocess
for line in subprocess.run(['ps','-ef'],capture_output=True,text=True).stdout.splitlines():
    if 'pretrain_gpt.py' in line or 'deepspeed.launcher.launch' in line:
        try: os.kill(int(line.split()[1]), signal.SIGKILL)
        except: pass
PY"
}

trap 'reset_gpus; cleanup_procs' EXIT

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
  
  lock_gpus "$clock"
  
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
  
  reset_gpus
  
  echo "[run] completed $run_id"
}

echo "=========================================="
echo "TP2PP2DP4 重新测量 (V2)"
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
