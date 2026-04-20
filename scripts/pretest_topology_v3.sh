#!/bin/bash
# 预测试 V3 - 使用系统 Python (和之前成功的实验一致)

set -euo pipefail

BASE_PATH="/home/sd/Megatron-DeepSpeed"

echo "=== 预测试：双节点拓扑验证 (系统 Python) ==="
echo "Python: $(/usr/bin/python3 --version)"
echo "PyTorch: $(/usr/bin/python3 -c 'import torch; print(torch.__version__)')"
echo ""

# 设置环境
export PYTHONPATH="/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:"
export NCCL_RAS_ENABLE=0
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export PDSH_RCMD_TYPE=ssh

# 测试 1: TP2PP2DP4
echo "[测试 1/3] 验证双节点连通性 (TP2PP2DP4 @ 1200MHz)..."
/usr/bin/python3 -m deepspeed.launcher.launch \
    --world_info='{"v100x16-1": [0,1,2,3,4,5,6,7], "v100x16-2": [0,1,2,3,4,5,6,7]}' \
    --node_rank=0 --master_addr=v100x16-1 --master_port=29500 \
    pretrain_gpt.py \
    --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 \
    --num-layers 28 --hidden-size 3584 --ffn-hidden-size 18944 \
    --num-attention-heads 28 --num-key-value-heads 4 \
    --micro-batch-size 1 --global-batch-size 16 --seq-length 2048 \
    --max-position-embeddings 2048 --train-iters 5 \
    --data-path /home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document \
    --data-impl mmap --tokenizer-type HFTokenizer \
    --tokenizer-model /home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat \
    --split 98,2,0 --distributed-backend nccl --lr 1e-5 \
    --lr-decay-style cosine --min-lr 1e-6 --weight-decay 0.01 \
    --clip-grad 1.0 --lr-warmup-iters 4 --optimizer adam \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --log-interval 1 --save-interval 0 --eval-interval 100 --eval-iters 10 \
    --no-query-key-layer-scaling --attention-dropout 0 --hidden-dropout 0 \
    --use-rotary-position-embeddings --untie-embeddings-and-output-weights \
    --swiglu --normalization rmsnorm --disable-bias-linear \
    --no-position-embedding --no-masked-softmax-fusion \
    --no-bias-gelu-fusion --no-bias-dropout-fusion \
    --recompute-granularity full --recompute-method uniform \
    --deepspeed-activation-checkpointing --zero-stage=1 \
    --deepspeed_config /home/sd/Megatron-DeepSpeed/.context/ds_config_pretest.json \
    --deepspeed --bf16 2>&1 | tee /home/sd/Megatron-DeepSpeed/logs/pretest_v3_1.log | tail -20

echo ""
echo "测试完成，检查日志..."
grep -E 'iteration.*time|step.*time|throughput|tokens' /home/sd/Megatron-DeepSpeed/logs/pretest_v3_1.log | tail -5 || echo "未找到性能数据"
