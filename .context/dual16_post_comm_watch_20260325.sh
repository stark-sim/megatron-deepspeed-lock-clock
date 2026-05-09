#!/usr/bin/env bash
set -euo pipefail
BASE=/home/sd/Megatron-DeepSpeed
while screen -ls | grep -q 'dual16_tp4pp1dp4_20260325'; do
  sleep 60
done
PATH=/home/sd/.local/bin:$PATH \
PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages \
MASTER_PORT=29940 \
$BASE/.context/comm_bench_torchrun_dual8.sh |& tee $BASE/.context/comm_bench_dual8_tailscale_20260325.log
