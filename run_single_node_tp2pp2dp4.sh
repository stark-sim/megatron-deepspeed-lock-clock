#!/bin/bash
# Single-node TP=2,PP=2,DP=4 measurement using experiment framework

BASE_PATH="/home/sd/Megatron-DeepSpeed"
FREQUENCIES=(1072 1080 1087)

cd "$BASE_PATH"

for FREQ in "${FREQUENCIES[@]}"; do
    echo "=========================================="
    echo "Running TP=2,PP=2,DP=4 at ${FREQ} MHz"
    echo "=========================================="

    # Set static GPU frequency
    sudo nvidia-smi -lgc ${FREQ},${FREQ}

    # Run experiment using framework
    EXPERIMENT_NAME="single_tp2pp2dp4_static${FREQ}" \
    TP=2 \
    PP=2 \
    NNODES=1 \
    NODE_RANK=0 \
    MASTER_ADDR=localhost \
    MASTER_PORT=29500 \
    MICRO_BATCH_SIZE=1 \
    GLOBAL_BATCH_SIZE=16 \
    TRAIN_STEPS=20 \
    ZERO_STAGE=1 \
    PRECISION_MODE=fp16 \
    STATIC_CLOCK_MHZ=${FREQ} \
    DISABLE_CHECKPOINT=1 \
    EVAL_ITERS=0 \
    bash run_experiment.sh 2>&1 | tee "single_node_tp2pp2dp4_freq${FREQ}_new.log"

    echo "<<< Completed ${FREQ} MHz"
done

# Reset frequencies
sudo nvidia-smi -rgc
echo "All measurements complete!"
