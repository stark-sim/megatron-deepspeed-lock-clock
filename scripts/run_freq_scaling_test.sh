#!/bin/bash
# 频率缩放对比测试脚本
# 需要用 sudo 运行

set -ex

cd /home/sd/Megatron-DeepSpeed

# 清理旧的 checkpoint
rm -rf checkpoints/qwen14b_3d/*

# 设置环境变量
export PYTHONPATH=/home/sd/.local/lib/python3.10/site-packages:$PYTHONPATH
export PATH=/home/sd/.local/bin:$PATH

# 启动训练
NODE_RANK=${NODE_RANK:-0} bash scripts/pretrain_qwen_3d_freq_scaling.sh
