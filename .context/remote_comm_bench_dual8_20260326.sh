#!/usr/bin/env bash
set -euo pipefail
BASE=/home/sd/Megatron-DeepSpeed
SCRIPT=$BASE/.context/torch_nccl_comm_bench.py
PORT=30010
python3 - <<'PY'
import os, signal, subprocess
patterns = ["torch_nccl_comm_bench.py", "torch.distributed.run --nnodes=2 --nproc_per_node=8", "30010", "29990", "29980", "29960"]
for host in [None, "192.168.205.202"]:
    cmd = ["ssh", host, "ps -ef"] if host else ["ps", "-ef"]
    for line in subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines():
        if any(p in line for p in patterns):
            try:
                pid = int(line.split()[1])
            except Exception:
                continue
            if host:
                subprocess.run(["ssh", host, f"kill -9 {pid}"], check=False)
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
PY
rm -f "$BASE/.context/comm_bench_dual8_tailscale_20260326.json" "$BASE/.context/comm_bench_dual8_tailscale_20260326.peer.json"
rm -f /tmp/comm_bench_dual8_peer_20260326.log
ssh 192.168.205.202 "bash -lc 'mkdir -p $BASE/.context; export PATH=/home/sd/.local/bin:\$PATH; export PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages; export LD_LIBRARY_PATH=/home/sd/.local/lib/python3.10/site-packages/torch/lib; export NCCL_SOCKET_IFNAME=tailscale0; export NCCL_IB_DISABLE=1; export NCCL_RAS_ENABLE=0; export TORCH_NCCL_BLOCKING_WAIT=1; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; export COMM_BENCH_OUT=$BASE/.context/comm_bench_dual8_tailscale_20260326.peer.json; nohup /usr/bin/python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=192.168.205.201 --master_port=$PORT $SCRIPT </dev/null >/tmp/comm_bench_dual8_peer_20260326.log 2>&1 & echo peer_started'"
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
/usr/bin/python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=192.168.205.201 --master_port=$PORT $SCRIPT
