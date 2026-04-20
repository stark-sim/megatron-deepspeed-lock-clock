#!/usr/bin/env bash
set -euo pipefail
BASE=/home/sd/Megatron-DeepSpeed
SCRIPT=$BASE/.context/torch_nccl_comm_bench.py
MASTER_ADDR=192.168.205.201
MASTER_PORT=${MASTER_PORT:-29940}
export PATH=/home/sd/.local/bin:$PATH
export PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/sd/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export COMM_BENCH_OUT=$BASE/.context/comm_bench_dual8_tailscale_20260325.json
ssh 192.168.205.202 "mkdir -p $BASE/.context && MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT PATH=/home/sd/.local/bin:\$PATH PYTHONPATH=/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages LD_LIBRARY_PATH=/home/sd/.local/lib/python3.10/site-packages/torch/lib NCCL_SOCKET_IFNAME=tailscale0 NCCL_IB_DISABLE=1 NCCL_RAS_ENABLE=0 TORCH_NCCL_BLOCKING_WAIT=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 COMM_BENCH_OUT=$BASE/.context/comm_bench_dual8_tailscale_20260325.peer.json nohup /usr/bin/python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $SCRIPT >/tmp/comm_bench_dual8_peer.log 2>&1 &"
sleep 3
/usr/bin/python3 -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $SCRIPT
