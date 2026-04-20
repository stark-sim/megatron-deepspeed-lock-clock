#!/bin/bash
# 跨节点拓扑补测脚本 V2 - 使用 DeepSpeed hostfile
# 用法: bash run_topology_sweep_v2.sh [start_idx] [end_idx]

set -euo pipefail

BASE_PATH="/home/sd/Megatron-DeepSpeed"

# 主机配置
MASTER_ADDR="${MASTER_ADDR:-100.64.0.90}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES=2
GPUS_PER_NODE=8

# 创建 hostfile
HOSTFILE="/tmp/topology_sweep_hostfile"
cat > "$HOSTFILE" << EOF
v100x16-1 slots=$GPUS_PER_NODE
v100x16-2 slots=$GPUS_PER_NODE
EOF

# 模型配置 (Qwen 7B)
export HIDDEN_SIZE=3584
export FFN_HIDDEN_SIZE=18944
export NUM_LAYERS=28
export NUM_HEADS=28
export NUM_KV_HEADS=4
export SEQ_LENGTH=2048

# 实验配置
export TRAIN_STEPS=20
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=16
export ZERO_STAGE=1
export PRECISION_MODE="bf16"
export EXPERIMENT_MODE="static"
export DISABLE_CHECKPOINT=1
export LAUNCHER="deepspeed"

# 解析命令行参数
START_IDX="${1:-0}"
END_IDX="${2:-9999}"

# 定义实验数组 (name:tp:pp:dp:freq:desc)
declare -a EXPERIMENTS=(
    # === TP2PP2DP4 补测 (方差大，重点验证) ===
    "TP2PP2DP4_f1072_r1:2:2:4:1072:PP2浅流水DP4验证"
    "TP2PP2DP4_f1072_r2:2:2:4:1072:PP2浅流水DP4验证"
    "TP2PP2DP4_f1072_r3:2:2:4:1072:PP2浅流水DP4验证"
    "TP2PP2DP4_f1080_r1:2:2:4:1080:PP2浅流水DP4验证"
    "TP2PP2DP4_f1080_r2:2:2:4:1080:PP2浅流水DP4验证"
    "TP2PP2DP4_f1080_r3:2:2:4:1080:PP2浅流水DP4验证"
    "TP2PP2DP4_f1185_r1:2:2:4:1185:PP2浅流水DP4验证"
    "TP2PP2DP4_f1185_r2:2:2:4:1185:PP2浅流水DP4验证"
    "TP2PP2DP4_f1185_r3:2:2:4:1185:PP2浅流水DP4验证"
    "TP2PP2DP4_f1200_r1:2:2:4:1200:PP2浅流水DP4验证"
    "TP2PP2DP4_f1200_r2:2:2:4:1200:PP2浅流水DP4验证"
    "TP2PP2DP4_f1200_r3:2:2:4:1200:PP2浅流水DP4验证"
    
    # === 新拓扑：极限压力测试 ===
    "TP1PP2DP8_f1072_r1:1:2:8:1072:PP2DP8极限"
    "TP1PP2DP8_f1185_r1:1:2:8:1185:PP2DP8极限"
    "TP1PP2DP8_f1200_r1:1:2:8:1200:PP2DP8极限"
    
    "TP4PP2DP2_f1072_r1:4:2:2:1072:TP4宽张量"
    "TP4PP2DP2_f1185_r1:4:2:2:1185:TP4宽张量"
    "TP4PP2DP2_f1200_r1:4:2:2:1200:TP4宽张量"
    
    "TP2PP1DP8_f1072_r1:2:1:8:1072:纯DP跨节点"
    "TP2PP1DP8_f1185_r1:2:1:8:1185:纯DP跨节点"
    "TP2PP1DP8_f1200_r1:2:1:8:1200:纯DP跨节点"
    
    # === 验证性测量 (已有数据对比) ===
    "TP1PP4DP4_f1185_v1:1:4:4:1185:已有数据验证"
    "TP2PP4DP2_f1185_v1:2:4:2:1185:已有数据验证"
    
    # === 无DP开销场景 ===
    "TP4PP4DP1_f1072_r1:4:4:1:1072:无DP开销"
    "TP4PP4DP1_f1200_r1:4:4:1:1200:无DP开销"
)

TOTAL_EXP=${#EXPERIMENTS[@]}
echo "=================================================="
echo "跨节点拓扑补测 (V2)"
echo "=================================================="
echo "总实验数: $TOTAL_EXP"
echo "执行范围: [$START_IDX, $END_IDX]"
echo "Hostfile: $HOSTFILE"
echo "开始时间: $(date)"
echo "=================================================="

# 创建日志目录
mkdir -p "${BASE_PATH}/logs"

# 执行实验
SUCCESS=0
FAILED=0
for i in "${!EXPERIMENTS[@]}"; do
    if (( i < START_IDX )) || (( i > END_IDX )); then
        continue
    fi
    
    exp="${EXPERIMENTS[$i]}"
    IFS=':' read -r name tp pp dp freq desc <<< "$exp"
    
    echo ""
    echo "[$((i+1))/$TOTAL_EXP] 实验: $name"
    echo "  拓扑: TP=$tp, PP=$pp, DP=$dp, 频率=${freq}MHz"
    echo "  描述: $desc"
    
    export EXPERIMENT_NAME="$name"
    export TP="$tp"
    export PP="$pp"
    export STATIC_CLOCK_MHZ="$freq"
    export NNODES="$NNODES"
    export MASTER_ADDR="$MASTER_ADDR"
    export MASTER_PORT="$MASTER_PORT"
    export HOSTFILE="$HOSTFILE"
    export NODE_RANK=0  # 当前节点
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    
    # 清理旧检查点
    rm -rf "${BASE_PATH}/checkpoints/${EXPERIMENT_NAME}"
    
    # 运行实验 (超时 15 分钟)
    EXP_LOG="${BASE_PATH}/logs/${name}.log"
    echo "  日志: $EXP_LOG"
    
    if timeout 900 bash "${BASE_PATH}/scripts/run_experiment.sh" 2>&1 | tee "$EXP_LOG"; then
        echo "  状态: ✓ 成功"
        ((SUCCESS++))
    else
        echo "  状态: ✗ 失败或超时"
        ((FAILED++))
    fi
    
    # 记录进度
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 进度: $((i+1))/$TOTAL_EXP (成功:$SUCCESS 失败:$FAILED)" >> "${BASE_PATH}/logs/topology_sweep_progress.log"
    
    # 短暂休息，让系统冷却
    sleep 5
done

echo ""
echo "=================================================="
echo "补测完成!"
echo "成功: $SUCCESS, 失败: $FAILED, 总计: $((SUCCESS+FAILED))"
echo "结束时间: $(date)"
echo "=================================================="
