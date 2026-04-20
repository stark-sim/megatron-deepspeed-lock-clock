#!/usr/bin/env bash
set -euo pipefail
BASE=/home/sd/Megatron-DeepSpeed
SCRIPT=$BASE/.context/torch_nccl_comm_bench.py
PORT=29980
python3 - <<'PY'
import os, signal, subprocess
patterns = ["torch_nccl_comm_bench.py", "torch.distributed.run --nnodes=2 --nproc_per_node=8", "29980", "29960"]
for host in [None, "192.168.205.202"]:
    cmd = ["ssh", host, "ps -ef"] if host else ["ps", "-ef"]
    lines = subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()
    for line in lines:
        if any(p in line for p in patterns):
            try:
                pid = int(line.split()[1])
            except Exception:
                continue
            if host:
                subprocess.run(["ssh", host, f"kill -9 {pid}"], check=False)
            else:
                try: os.kill(pid, signal.SIGKILL)
                except Exception: pass
PY
rm -f "$BASE/.context/comm_bench_dual8_tailscale_20260326.json" "$BASE/.context/comm_bench_dual8_tailscale_20260326.peer.json"
ssh 192.168.205.202 "mkdir -p $BASE/.context && PATH=/home/sd/.local/bin:\$PATH PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages LD_LIBRARY_PATH=/home/sd/.local/lib/python3.10/site-packages/torch/lib NCCL_SOCKET_IFNAME=tailscale0 NCCL_IB_DISABLE=1 NCCL_RAS_ENABLE=0 TORCH_NCCL_BLOCKING_WAIT=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 COMM_BENCH_OUT=$BASE/.context/comm_bench_dual8_tailscale_20260326.peer.json nohup /usr/bin/python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=192.168.205.201 --master_port=$PORT $SCRIPT >/tmp/comm_bench_dual8_peer_20260326.log 2>&1 &"
sleep 5
export PATH=/home/sd/.local/bin:$PATH
export PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/sd/.local/lib/python3.10/site-packages/torch/lib
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export COMM_BENCH_OUT=$BASE/.context/comm_bench_dual8_tailscale_20260326.json
exec /usr/bin/python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=192.168.205.201 --master_port=$PORT $SCRIPT
