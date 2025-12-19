#!/bin/bash
# 多节点启动脚本 - 在两个节点上启动3D并行训练
set -ex

BASE_PATH=/home/sd/Megatron-DeepSpeed
SCRIPT=${BASE_PATH}/scripts/pretrain_qwen_3d_parallel.sh

# 节点配置
NODE0=100.64.0.90
NODE1=100.64.0.103
MASTER_ADDR=${NODE0}
MASTER_PORT=29500

echo "=========================================="
echo "Starting 3D Parallel Training on 2 Nodes"
echo "  Node 0 (Master): ${NODE0}"
echo "  Node 1: ${NODE1}"
echo "=========================================="

# 先在Node 0上启动 (后台)
echo "Starting Node 0 (Master)..."
cd ${BASE_PATH}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export NNODES=2
export NODE_RANK=0
nohup bash ${SCRIPT} > ${BASE_PATH}/logs/node0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
NODE0_PID=$!

# 等待master节点启动
sleep 10

# 在Node 1上启动
echo "Starting Node 1..."
ssh ${NODE1} "cd ${BASE_PATH} && \
    export MASTER_ADDR=${MASTER_ADDR} && \
    export MASTER_PORT=${MASTER_PORT} && \
    export NNODES=2 && \
    export NODE_RANK=1 && \
    bash ${SCRIPT}" &
NODE1_PID=$!

# 等待所有进程
echo "Waiting for training to complete..."
wait $NODE0_PID $NODE1_PID
echo "Training completed!"
