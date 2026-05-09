#!/bin/bash
# 跨节点拓扑全面补测脚本
# 用法: bash run_topology_sweep.sh [start_idx] [end_idx]

set -euo pipefail

BASE_PATH="/home/sd/Megatron-DeepSpeed"
source "${BASE_PATH}/scripts/experiment_utils.sh"

# 主机配置
MASTER_ADDR="${MASTER_ADDR:-100.64.0.90}"
MASTER_PORT="${MASTER_PORT:-29500}"
HOSTFILE="${HOSTFILE:-/home/sd/hostfile_v100}"
NNODES=2

# 模型配置 (Qwen 7B)
HIDDEN_SIZE=3584
FFN_HIDDEN_SIZE=18944
NUM_LAYERS=28
NUM_HEADS=28
NUM_KV_HEADS=4
SEQ_LENGTH=2048

# 实验配置
TRAIN_STEPS=20
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
ZERO_STAGE=1
PRECISION_MODE="bf16"

# 解析命令行参数
START_IDX="${1:-0}"
END_IDX="${2:-9999}"

# 定义实验数组 (拓扑, TP, PP, DP, 频率, 重复, 描述)
declare -a EXPERIMENTS=(
    # === 高优先级：TP2PP2DP4 补测 (25个) ===
    "TP2PP2DP4_1072_1:2:2:4:1072:1:PP=2浅流水DP=4"
    "TP2PP2DP4_1072_2:2:2:4:1072:2:PP=2浅流水DP=4"
    "TP2PP2DP4_1072_3:2:2:4:1072:3:PP=2浅流水DP=4"
    "TP2PP2DP4_1072_4:2:2:4:1072:4:PP=2浅流水DP=4"
    "TP2PP2DP4_1072_5:2:2:4:1072:5:PP=2浅流水DP=4"
    "TP2PP2DP4_1080_1:2:2:4:1080:1:PP=2浅流水DP=4"
    "TP2PP2DP4_1080_2:2:2:4:1080:2:PP=2浅流水DP=4"
    "TP2PP2DP4_1080_3:2:2:4:1080:3:PP=2浅流水DP=4"
    "TP2PP2DP4_1080_4:2:2:4:1080:4:PP=2浅流水DP=4"
    "TP2PP2DP4_1080_5:2:2:4:1080:5:PP=2浅流水DP=4"
    "TP2PP2DP4_1087_1:2:2:4:1087:1:PP=2浅流水DP=4"
    "TP2PP2DP4_1087_2:2:2:4:1087:2:PP=2浅流水DP=4"
    "TP2PP2DP4_1087_3:2:2:4:1087:3:PP=2浅流水DP=4"
    "TP2PP2DP4_1087_4:2:2:4:1087:4:PP=2浅流水DP=4"
    "TP2PP2DP4_1087_5:2:2:4:1087:5:PP=2浅流水DP=4"
    "TP2PP2DP4_1185_1:2:2:4:1185:1:PP=2浅流水DP=4"
    "TP2PP2DP4_1185_2:2:2:4:1185:2:PP=2浅流水DP=4"
    "TP2PP2DP4_1185_3:2:2:4:1185:3:PP=2浅流水DP=4"
    "TP2PP2DP4_1185_4:2:2:4:1185:4:PP=2浅流水DP=4"
    "TP2PP2DP4_1185_5:2:2:4:1185:5:PP=2浅流水DP=4"
    "TP2PP2DP4_1200_1:2:2:4:1200:1:PP=2浅流水DP=4"
    "TP2PP2DP4_1200_2:2:2:4:1200:2:PP=2浅流水DP=4"
    "TP2PP2DP4_1200_3:2:2:4:1200:3:PP=2浅流水DP=4"
    "TP2PP2DP4_1200_4:2:2:4:1200:4:PP=2浅流水DP=4"
    "TP2PP2DP4_1200_5:2:2:4:1200:5:PP=2浅流水DP=4"
    
    # === 中优先级：新拓扑 (27个) ===
    "TP1PP2DP8_1072_1:1:2:8:1072:1:PP=2DP=8极限"
    "TP1PP2DP8_1072_2:1:2:8:1072:2:PP=2DP=8极限"
    "TP1PP2DP8_1072_3:1:2:8:1072:3:PP=2DP=8极限"
    "TP1PP2DP8_1185_1:1:2:8:1185:1:PP=2DP=8极限"
    "TP1PP2DP8_1185_2:1:2:8:1185:2:PP=2DP=8极限"
    "TP1PP2DP8_1185_3:1:2:8:1185:3:PP=2DP=8极限"
    "TP1PP2DP8_1200_1:1:2:8:1200:1:PP=2DP=8极限"
    "TP1PP2DP8_1200_2:1:2:8:1200:2:PP=2DP=8极限"
    "TP1PP2DP8_1200_3:1:2:8:1200:3:PP=2DP=8极限"
    
    "TP4PP2DP2_1072_1:4:2:2:1072:1:TP=4PP=2"
    "TP4PP2DP2_1072_2:4:2:2:1072:2:TP=4PP=2"
    "TP4PP2DP2_1072_3:4:2:2:1072:3:TP=4PP=2"
    "TP4PP2DP2_1185_1:4:2:2:1185:1:TP=4PP=2"
    "TP4PP2DP2_1185_2:4:2:2:1185:2:TP=4PP=2"
    "TP4PP2DP2_1185_3:4:2:2:1185:3:TP=4PP=2"
    "TP4PP2DP2_1200_1:4:2:2:1200:1:TP=4PP=2"
    "TP4PP2DP2_1200_2:4:2:2:1200:2:TP=4PP=2"
    "TP4PP2DP2_1200_3:4:2:2:1200:3:TP=4PP=2"
    
    "TP2PP1DP8_1072_1:2:1:8:1072:1:纯DP跨节点"
    "TP2PP1DP8_1072_2:2:1:8:1072:2:纯DP跨节点"
    "TP2PP1DP8_1072_3:2:1:8:1072:3:纯DP跨节点"
    "TP2PP1DP8_1185_1:2:1:8:1185:1:纯DP跨节点"
    "TP2PP1DP8_1185_2:2:1:8:1185:2:纯DP跨节点"
    "TP2PP1DP8_1185_3:2:1:8:1185:3:纯DP跨节点"
    "TP2PP1DP8_1200_1:2:1:8:1200:1:纯DP跨节点"
    "TP2PP1DP8_1200_2:2:1:8:1200:2:纯DP跨节点"
    "TP2PP1DP8_1200_3:2:1:8:1200:3:纯DP跨节点"
    
    # === 低优先级：验证性测量 (8个) ===
    "TP1PP4DP4_1185_1:1:4:4:1185:1:已有数据验证"
    "TP1PP4DP4_1200_1:1:4:4:1200:1:已有数据验证"
    "TP2PP4DP2_1185_1:2:4:2:1185:1:已有数据验证"
    "TP2PP4DP2_1200_1:2:4:2:1200:1:已有数据验证"
    "TP4PP4DP1_1072_1:4:4:1:1072:1:无DP开销"
    "TP4PP4DP1_1072_2:4:4:1:1072:2:无DP开销"
    "TP4PP4DP1_1200_1:4:4:1:1200:1:无DP开销"
    "TP4PP4DP1_1200_2:4:4:1:1200:2:无DP开销"
)

TOTAL_EXP=${#EXPERIMENTS[@]}
echo "=================================================="
echo "跨节点拓扑补测"
echo "=================================================="
echo "总实验数: $TOTAL_EXP"
echo "执行范围: [$START_IDX, $END_IDX]"
echo "开始时间: $(date)"
echo "=================================================="

# 执行实验
for i in "${!EXPERIMENTS[@]}"; do
    if (( i < START_IDX )) || (( i > END_IDX )); then
        continue
    fi
    
    exp="${EXPERIMENTS[$i]}"
    # 解析: name:tp:pp:dp:freq:repeat:desc
    IFS=':' read -r name tp pp dp freq repeat desc <<< "$exp"
    
    echo ""
    echo "[$i/$TOTAL_EXP] 运行: $name (TP=$tp, PP=$pp, DP=$dp, FREQ=$freq)"
    echo "描述: $desc"
    
    export EXPERIMENT_NAME="topo_sweep_${name}"
    export TP="$tp"
    export PP="$pp"
    export STATIC_CLOCK_MHZ="$freq"
    export TRAIN_STEPS="$TRAIN_STEPS"
    export MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE"
    export GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE"
    export ZERO_STAGE="$ZERO_STAGE"
    export PRECISION_MODE="$PRECISION_MODE"
    export EXPERIMENT_MODE="static"
    export NNODES="$NNODES"
    export MASTER_ADDR="$MASTER_ADDR"
    export MASTER_PORT="$MASTER_PORT"
    export HOSTFILE="$HOSTFILE"
    export DISABLE_CHECKPOINT=1
    
    # 启动双节点实验
    # Node 0 (当前节点)
    export NODE_RANK=0
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    
    # 清理旧检查点
    rm -rf "${BASE_PATH}/checkpoints/${EXPERIMENT_NAME}"
    
    # 运行实验
    if ! timeout 600 bash "${BASE_PATH}/scripts/run_experiment.sh" 2>&1 | tee -a "${BASE_PATH}/logs/${EXPERIMENT_NAME}.log"; then
        echo "警告: $name 可能失败，继续下一个..."
    fi
    
    echo "完成: $name"
    echo "$(date) - 进度: $((i+1))/$TOTAL_EXP" > "${BASE_PATH}/logs/topology_sweep_progress.txt"
done

echo ""
echo "=================================================="
echo "补测完成!"
echo "结束时间: $(date)"
echo "=================================================="
