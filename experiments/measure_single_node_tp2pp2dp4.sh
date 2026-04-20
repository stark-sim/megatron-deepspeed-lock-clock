#!/bin/bash
# Single-node TP=2,PP=2,DP=4 measurement script for V100 16-GPU node
# This gets the real single_node_time_s for cross-node calibration
# Zeus monitoring is automatically started by training.py for rank 0

set -e

# Environment setup
export NCCL_RAS_ENABLE=0
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:"

# Model config (matches calibration points)
NUM_LAYERS=28
HIDDEN_SIZE=3584
FFN_HIDDEN_SIZE=18944
NUM_HEADS=28
NUM_KV_HEADS=4
SEQ_LENGTH=2048
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16

# Topology: TP=2, PP=2, DP=4 (2*2*4=16 GPUs)
TP=2
PP=2
DP=4

# Static frequencies to measure (matching dual-node measurements)
FREQUENCIES=(1072 1080 1087)

echo "=========================================="
echo "Single-Node TP=2,PP=2,DP=4 Measurement"
echo "GPUs: 16x V100 on v100x16-1"
echo "Frequencies: ${FREQUENCIES[@]}"
echo "=========================================="

for FREQ in "${FREQUENCIES[@]}"; do
    echo ""
    echo ">>> Running with static frequency: ${FREQ} MHz"

    # Set static GPU frequency using nvidia-smi
    echo "Setting GPU frequency to ${FREQ} MHz..."
    sudo nvidia-smi -lgc ${FREQ},${FREQ} 2>/dev/null || echo "Warning: Could not lock frequency (may need sudo)"

    # Run training
    # Note: Zeus monitoring starts automatically in training.py for rank 0
    /home/sd/.local/bin/deepspeed \
        --num_gpus 16 \
        pretrain_gpt.py \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
        --num-attention-heads ${NUM_HEADS} \
        --num-key-value-heads ${NUM_KV_HEADS} \
        --seq-length ${SEQ_LENGTH} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --train-iters 20 \
        --lr 0.0001 \
        --min-lr 0.00001 \
        --lr-decay-style linear \
        --lr-warmup-fraction 0.01 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --optimizer adam \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --adam-eps 1e-8 \
        --fp16 \
        --zero-stage 1 \
        --log-interval 5 \
        --eval-iters 0 \
        --exit-on-missing-checkpoint \
        --max-position-embeddings 2048 \
        --data-path /home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document \
        --data-impl mmap \
        --tokenizer-type HFTokenizer \
        --tokenizer-model /home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat \
        --split 98,2,0 \
        --distributed-backend nccl \
        --no-query-key-layer-scaling \
        --attention-dropout 0 \
        --hidden-dropout 0 \
        --use-rotary-position-embeddings \
        --untie-embeddings-and-output-weights \
        --swiglu \
        --normalization rmsnorm \
        --disable-bias-linear \
        --no-position-embedding \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion \
        --recompute-granularity full \
        --recompute-method uniform \
        --deepspeed-activation-checkpointing \
        2>&1 | tee "single_node_tp2pp2dp4_freq${FREQ}.log"

    echo "<<< Completed frequency ${FREQ} MHz"
    echo ""
done

# Reset GPU frequencies
echo "Resetting GPU frequencies to default..."
sudo nvidia-smi -rgc 2>/dev/null || echo "Warning: Could not reset frequency"

echo "=========================================="
echo "All measurements complete!"
echo "Log files:"
ls -la single_node_tp2pp2dp4_freq*.log
echo "=========================================="
