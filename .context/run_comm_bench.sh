#!/bin/bash
# Multi-node NCCL communication benchmark launcher
# Usage: run_comm_bench.sh [2x4|2x8]

set -e

MODE=${1:-2x4}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "NCCL Communication Benchmark - ${MODE}"
echo "=============================================="

# Configuration
export PATH="/home/sd/.local/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="/home/sd/Megatron-DeepSpeed:$PYTHONPATH"

# NCCL IB Configuration
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Host configuration
MASTER_ADDR="192.168.205.202"
MASTER_PORT="29560"

if [ "$MODE" = "2x4" ]; then
    GPUS_PER_NODE=4
    DS_INCLUDE="v100x16-1:8,9,10,11@v100x16-2:8,9,10,11"
    OUTPUT="/home/sd/Megatron-DeepSpeed/.context/comm_bench_2x4_${TIMESTAMP}.json"
    HOSTFILE="/tmp/bench_hostfile_2x4"
elif [ "$MODE" = "2x8" ]; then
    GPUS_PER_NODE=8
    DS_INCLUDE="v100x16-1:0,1,2,3,4,5,6,7@v100x16-2:0,1,2,3,4,5,6,7"
    OUTPUT="/home/sd/Megatron-DeepSpeed/.context/comm_bench_2x8_${TIMESTAMP}.json"
    HOSTFILE="/tmp/bench_hostfile_2x8"
else
    echo "Unknown mode: $MODE (use 2x4 or 2x8)"
    exit 1
fi

# Create hostfile
cat > "$HOSTFILE" << EOF
v100x16-1 slots=${GPUS_PER_NODE}
v100x16-2 slots=${GPUS_PER_NODE}
EOF

echo ""
echo "Configuration:"
echo "  Mode: ${MODE}"
echo "  GPUs per node: ${GPUS_PER_NODE}"
echo "  Total GPUs: $((GPUS_PER_NODE * 2))"
echo "  Include: ${DS_INCLUDE}"
echo "  Output: ${OUTPUT}"
echo ""

# Run benchmark
echo "Starting benchmark..."
cd /home/sd/Megatron-DeepSpeed/.context

timeout 300 deepspeed --hostfile "$HOSTFILE" \
    --include "$DS_INCLUDE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    torch_nccl_comm_bench.py \
    --sizes-mb 1 4 16 64 256 \
    --output "$OUTPUT" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Benchmark completed successfully!"
    echo "Results saved to: ${OUTPUT}"
    echo "=============================================="
    
    # Display summary
    if [ -f "$OUTPUT" ]; then
        echo ""
        echo "Summary:"
        python3 << EOF
import json
with open("$OUTPUT") as f:
    data = json.load(f)
print(f"  World size: {data['world_size']}")
print(f"  Timestamp: {data['timestamp']}")
print(f"  NCCL IB HCA: {data['nccl_env'].get('NCCL_IB_HCA', 'N/A')}")
print("")
print("  Bandwidth results:")
for r in data['results']:
    print(f"    {r['size_mb']:3.0f} MB: busbw={r['busbw_gbps']:6.2f} Gbps, "
          f"time={r['time_ms']:6.2f} ms, cv={r['cv']:.3f}")
EOF
    fi
else
    echo ""
    echo "=============================================="
    echo "Benchmark failed with exit code: ${EXIT_CODE}"
    echo "=============================================="
fi

# Cleanup
rm -f "$HOSTFILE"

exit $EXIT_CODE
