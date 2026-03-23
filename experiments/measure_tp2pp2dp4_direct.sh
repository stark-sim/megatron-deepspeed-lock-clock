#!/bin/bash
# Direct measurement script for TP=2,PP=2,DP=4 single-node

set -e

BASE_PATH="/home/sd/Megatron-DeepSpeed"
FREQUENCIES=(1072 1080 1087)

cd "$BASE_PATH"

# Create DeepSpeed config for each frequency
for FREQ in "${FREQUENCIES[@]}"; do
    cat > "/tmp/dsconfig_tp2pp2dp4_${FREQ}.json" << EOF
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 1,
  "steps_per_print": 5,
  "zero_optimization": {"stage": 1},
  "bf16": {"enabled": true},
  "fp16": {"enabled": false},
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "wall_clock_breakdown": false
}
EOF
done

for FREQ in "${FREQUENCIES[@]}"; do
    echo ""
    echo ">>> Running TP=2,PP=2,DP=4 at ${FREQ} MHz"

    # Set static GPU frequency
    sudo nvidia-smi -lgc ${FREQ},${FREQ}

    # Run using torchrun (single node)
    /usr/bin/python3 -m torch.distributed.run \
        --nproc_per_node=16 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        pretrain_gpt.py \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        --num-layers 28 \
        --hidden-size 3584 \
        --ffn-hidden-size 18944 \
        --num-attention-heads 28 \
        --num-key-value-heads 4 \
        --micro-batch-size 1 \
        --global-batch-size 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --train-iters 20 \
        --data-path ${BASE_PATH}/data/chinese_wiki_megatron_text_document \
        --data-impl mmap \
        --tokenizer-type HFTokenizer \
        --tokenizer-model ${BASE_PATH}/.context/qwen25_tokenizer_flat \
        --split 98,2,0 \
        --distributed-backend nccl \
        --lr 0.0001 \
        --lr-decay-style linear \
        --min-lr 0.00001 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --lr-warmup-fraction 0.01 \
        --optimizer adam \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --adam-eps 1e-8 \
        --log-interval 5 \
        --save-interval 0 \
        --eval-interval 100 \
        --eval-iters 0 \
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
        --zero-stage=1 \
        --deepspeed_config=/tmp/dsconfig_tp2pp2dp4_${FREQ}.json \
        --deepspeed \
        --bf16 \
        2>&1 | tee "single_node_tp2pp2dp4_freq${FREQ}_final.log"

    echo "<<< Completed ${FREQ} MHz"
done

# Reset frequencies
echo ""
echo "Resetting GPU frequencies..."
sudo nvidia-smi -rgc

echo ""
echo "All measurements complete!"
