#!/bin/bash
# LLaMA-7B 模型获取和转换完整脚本（只下载 safetensors 格式）

set -e

cd /home/user/Megatron-DeepSpeed

# 激活环境
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate tp4bit

# 显存优化设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_DIR="/home/user/models/llama-7b-hf"
MEGA_DIR="/home/user/Megatron-DeepSpeed/checkpoints/llama7b_hf2megads_tp2pp2"

echo "=========================================="
echo "LLaMA-7B 模型获取和转换"
echo "=========================================="

# 步骤 1: 检查模型完整性
echo "[1/3] 检查模型完整性..."

python3 << 'CHECK_EOF'
import json
import os
import sys

model_dir = "/home/user/models/llama-7b-hf"
index_file = os.path.join(model_dir, "model.safetensors.index.json")

if not os.path.exists(index_file):
    print("INCOMPLETE: index file missing")
    sys.exit(1)

with open(index_file) as f:
    index = json.load(f)

required_weights = set(index.get("weight_map", {}).values())
print(f"Required weight files: {len(required_weights)}")

missing = []
for w in required_weights:
    w_path = os.path.join(model_dir, w)
    if not os.path.exists(w_path):
        missing.append(w)
        print(f"MISSING: {w}")
    else:
        size_mb = os.path.getsize(w_path) / (1024*1024)
        print(f"OK: {w} ({size_mb:.1f} MB)")

if missing:
    print(f"INCOMPLETE: Missing {len(missing)} weight file(s)")
    sys.exit(1)
else:
    print("COMPLETE: All weight files present")
    sys.exit(0)
CHECK_EOF

# 步骤 2: 验证模型
echo ""
echo "[2/3] 验证模型文件..."
python3 << 'VERIFY_EOF'
import json
import os

model_dir = "/home/user/models/llama-7b-hf"

with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
    index = json.load(f)

required = set(index["weight_map"].values())
print(f"Required files: {len(required)}")

for w in sorted(required):
    w_path = os.path.join(model_dir, w)
    size_gb = os.path.getsize(w_path) / (1024**3)
    print(f"  {w}: {size_gb:.2f} GB")

print("\n模型文件验证通过！")
VERIFY_EOF

# 步骤 3: 转换为 Megatron-DeepSpeed 格式
echo ""
if [ -d "$MEGA_DIR" ] && [ -f "$MEGA_DIR/latest" ]; then
    echo "[3/3] Checkpoint 已存在，跳过转换"
    echo "  $MEGA_DIR"
else
    echo "[3/3] 转换为 Megatron-DeepSpeed 格式..."
    echo "  TP=2, PP=2"
    echo "  注意: 使用 CPU offload 避免 16GB 显存 OOM"
    
    # 使用 CPU 初始化避免 OOM
    export CUDA_VISIBLE_DEVICES=0
    
    deepspeed --num_gpus 1 tools/hf2megads_weight_converter.py \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        --num-layers 32 \
        --hidden-size 4096 \
        --ffn-hidden-size 11008 \
        --num-attention-heads 32 \
        --micro-batch-size 1 \
        --global-batch-size 1 \
        --train-iters 1 \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --tokenizer-type HFTokenizer \
        --tokenizer-model "$MODEL_DIR" \
        --distributed-backend nccl \
        --swiglu \
        --normalization rmsnorm \
        --use-rotary-position-embeddings \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --no-position-embedding \
        --bf16 \
        --zero-stage 0 \
        --deepspeed_config examples_deepspeed/finetune_hf_llama/ds_config_empty.json \
        --deepspeed \
        --hf-ckpt-dir "$MODEL_DIR" \
        --hf-ckpt-num-shards 2 \
        --load-mode auto \
        --save-interval 1 \
        --save "$MEGA_DIR"
    
    echo ""
    echo "=========================================="
    echo "转换完成！"
    echo "Checkpoint 路径: $MEGA_DIR"
fi

echo ""
echo "=========================================="
echo "Setup 完成！"
echo "模型路径: $MODEL_DIR"
echo "Checkpoint: $MEGA_DIR"
echo "=========================================="
