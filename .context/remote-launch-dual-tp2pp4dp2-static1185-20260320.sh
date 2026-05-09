#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/sd/.local/bin:$PATH
export PYTHONPATH=/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}
export NCCL_SOCKET_IFNAME=tailscale0
export NCCL_IB_DISABLE=1
export NCCL_RAS_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cat > /tmp/hostfile_tp2pp4dp2_dual <<HF
localhost slots=8
192.168.205.202 slots=8
HF
cd /home/sd/Megatron-DeepSpeed
EXPERIMENT_NAME=dual_tp2pp4dp2_static1185_20260320 \
EXPERIMENT_MODE=static STATIC_CLOCK_MHZ=1185 \
TP=2 PP=4 NNODES=2 HOSTFILE=/tmp/hostfile_tp2pp4dp2_dual \
MASTER_ADDR=100.64.0.90 MASTER_PORT=29610 ZERO_STAGE=1 \
MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=16 TRAIN_STEPS=20 DISABLE_CHECKPOINT=1 \
TOKENIZER_PATH=/home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat \
bash scripts/run_experiment.sh
