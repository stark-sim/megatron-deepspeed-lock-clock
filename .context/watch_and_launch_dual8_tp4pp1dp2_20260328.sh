#!/usr/bin/env bash
set -euo pipefail
TARGET_SCRIPT=/home/sd/Megatron-DeepSpeed/.context/dual8_tp4pp1dp2_collect_20260328.sh
POLL_SEC=60
LOG=/home/sd/Megatron-DeepSpeed/.context/watch_and_launch_dual8_tp4pp1dp2_20260328.log
is_free() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, 'NR<=4 {gsub(/ /,"",$2); if ($2>2000) busy=1} END{exit busy}'
}
while true; do
  ts=$(date '+%F %T')
  if is_free; then
    echo "[$ts] GPUs 0-3 appear free; launching $TARGET_SCRIPT" | tee -a "$LOG"
    bash -lc "$TARGET_SCRIPT" >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] launch script exited" | tee -a "$LOG"
    exit 0
  fi
  echo "[$ts] GPUs 0-3 busy; retry in ${POLL_SEC}s" >> "$LOG"
  sleep "$POLL_SEC"
done
