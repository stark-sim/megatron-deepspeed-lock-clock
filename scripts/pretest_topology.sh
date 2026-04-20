#!/bin/bash
# 预测试：验证双节点连通性和基础功能

set -euo pipefail

BASE_PATH="/home/sd/Megatron-DeepSpeed"

# 创建 hostfile
HOSTFILE="/tmp/pretest_hostfile"
cat > "$HOSTFILE" << EOF
v100x16-1 slots=8
v100x16-2 slots=8
EOF

echo "=== 预测试：双节点拓扑验证 ==="
echo ""

# 测试 1: 基础双节点连通性
echo "[测试 1/3] 验证双节点连通性 (TP2PP2DP4 @ 1200MHz)..."
export EXPERIMENT_NAME="pretest_tp2pp2dp4_1200"
export TP=2
export PP=2
export STATIC_CLOCK_MHZ=1200
export TRAIN_STEPS=5  # 只跑 5 步验证
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=16
export ZERO_STAGE=1
export PRECISION_MODE="bf16"
export EXPERIMENT_MODE="static"
export DISABLE_CHECKPOINT=1
export NNODES=2
export MASTER_ADDR="100.64.0.90"
export MASTER_PORT="29510"
export HOSTFILE="$HOSTFILE"
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

rm -rf "${BASE_PATH}/checkpoints/${EXPERIMENT_NAME}"

if timeout 300 bash "${BASE_PATH}/scripts/run_experiment.sh" 2>&1 | tee "${BASE_PATH}/logs/pretest_1.log" | tail -20; then
    echo "✓ 测试 1 通过"
else
    echo "✗ 测试 1 失败，检查日志: ${BASE_PATH}/logs/pretest_1.log"
    exit 1
fi

echo ""
echo "[测试 2/3] 验证纯 DP 跨节点 (TP2PP1DP8 @ 1185MHz)..."
export EXPERIMENT_NAME="pretest_tp2pp1dp8_1185"
export TP=2
export PP=1
export STATIC_CLOCK_MHZ=1185
export TRAIN_STEPS=5

rm -rf "${BASE_PATH}/checkpoints/${EXPERIMENT_NAME}"

if timeout 300 bash "${BASE_PATH}/scripts/run_experiment.sh" 2>&1 | tee "${BASE_PATH}/logs/pretest_2.log" | tail -20; then
    echo "✓ 测试 2 通过"
else
    echo "✗ 测试 2 失败，检查日志: ${BASE_PATH}/logs/pretest_2.log"
    exit 1
fi

echo ""
echo "[测试 3/3] 验证极限 DP (TP1PP2DP8 @ 1072MHz)..."
export EXPERIMENT_NAME="pretest_tp1pp2dp8_1072"
export TP=1
export PP=2
export STATIC_CLOCK_MHZ=1072
export TRAIN_STEPS=5

rm -rf "${BASE_PATH}/checkpoints/${EXPERIMENT_NAME}"

if timeout 300 bash "${BASE_PATH}/scripts/run_experiment.sh" 2>&1 | tee "${BASE_PATH}/logs/pretest_3.log" | tail -20; then
    echo "✓ 测试 3 通过"
else
    echo "✗ 测试 3 失败，检查日志: ${BASE_PATH}/logs/pretest_3.log"
    exit 1
fi

echo ""
echo "=== 预测试全部通过 ==="
echo "可以开始执行完整补测方案"
