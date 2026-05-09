#!/usr/bin/env bash
# Run communication benchmark for 2x4 (dual8) configuration
set -euo pipefail

BASE="/home/sd/Megatron-DeepSpeed"
PEER="192.168.205.202"
HOSTFILE="/tmp/hostfile_comm_bench_2x4"
TARGET_GPUS="8,9,10,11"
MASTER_ADDR="192.168.205.201"
MASTER_PORT="29998"
OUTPUT="$BASE/.context/comm_bench_2x4_$(date +%Y%m%d_%H%M%S).json"

export PATH="/home/sd/.local/bin:${PATH}"
export PYTHONPATH="/home/sd/Megatron-DeepSpeed:/home/sd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export NCCL_RAS_ENABLE="0"
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
export NCCL_DEBUG=WARN

cat > "$HOSTFILE" <<EOF2
localhost slots=4
${PEER} slots=4
EOF2

echo "Running 2x4 communication benchmark..."
echo "Output: $OUTPUT"

cd "$BASE"
deepspeed \
    --hostfile "$HOSTFILE" \
    --include "localhost:${TARGET_GPUS}@${PEER}:${TARGET_GPUS}" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    .context/torch_nccl_comm_bench.py \
    --sizes-mb 1 4 16 64 256 \
    --output "$OUTPUT"

echo "Benchmark complete: $OUTPUT"
