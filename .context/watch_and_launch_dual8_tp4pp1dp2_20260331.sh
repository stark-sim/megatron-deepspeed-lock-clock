#!/usr/bin/env bash
set -euo pipefail
TARGET_SCRIPT=/home/sd/Megatron-DeepSpeed/.context/dual8_tp4pp1dp2_collect_20260328.sh
POLL_SEC=60
LOG=/home/sd/Megatron-DeepSpeed/.context/watch_and_launch_dual8_tp4pp1dp2_20260331.log
REMOTE_HOST=192.168.205.202
GPU_MEM_THRESHOLD_MB=2000
GPU_IDS_CSV=8,9,10,11

check_gpu_free() {
  local host="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "python3 - <<'PY'
import subprocess, sys
wanted = {8, 9, 10, 11}
threshold = 2000
out = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'
], text=True)
busy = False
for line in out.strip().splitlines():
    idx_s, mem_s = [part.strip() for part in line.split(',', 1)]
    idx = int(idx_s)
    mem = int(mem_s)
    if idx in wanted and mem > threshold:
        busy = True
        break
sys.exit(1 if busy else 0)
PY"
}

snapshot() {
  local host="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "echo host=$host target_gpus=${GPU_IDS_CSV}; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '9,12p'; echo apps:; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true"
}

while true; do
  ts=$(date '+%F %T')
  local_gpu_ok=0
  remote_gpu_ok=0

  if check_gpu_free localhost; then local_gpu_ok=1; fi
  if check_gpu_free "$REMOTE_HOST"; then remote_gpu_ok=1; fi

  if [[ $local_gpu_ok -eq 1 && $remote_gpu_ok -eq 1 ]]; then
    echo "[$ts] both hosts look free on GPUs ${GPU_IDS_CSV}; launching $TARGET_SCRIPT" | tee -a "$LOG"
    bash -lc "$TARGET_SCRIPT" >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] launch script exited" | tee -a "$LOG"
    exit 0
  fi

  echo "[$ts] waiting: local_gpu=$local_gpu_ok remote_gpu=$remote_gpu_ok target_gpus=${GPU_IDS_CSV}" | tee -a "$LOG"
  snapshot localhost >> "$LOG" 2>&1 || true
  snapshot "$REMOTE_HOST" >> "$LOG" 2>&1 || true
  sleep "$POLL_SEC"
done
