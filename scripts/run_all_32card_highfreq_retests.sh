#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "32-card High-Frequency Retest Sequence"
echo "========================================"
echo ""

cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

# 1. TP2PP4DP4 highfreq
log1="/tmp/all_highfreq_$(date +%Y%m%d_%H%M%S)_tp2pp4dp4.log"
echo "[1/3] Starting TP2PP4DP4 highfreq sweep... (log: $log1)"
RUN_TAG="highfreq_formal20_finetune_nosave_retest" TRAIN_STEPS="20" \
    bash scripts/run_real_qwen25_7b_tp2pp4dp4_v100_highfreq_compare.sh > "$log1" 2>&1
echo "[1/3] TP2PP4DP4 done."

# 2. TP2PP2DP8 highfreq
log2="/tmp/all_highfreq_$(date +%Y%m%d_%H%M%S)_tp2pp2dp8.log"
echo "[2/3] Starting TP2PP2DP8 highfreq sweep... (log: $log2)"
RUN_TAG="highfreq_formal20_finetune_nosave_retest" TRAIN_STEPS="20" \
    bash scripts/run_real_qwen25_7b_tp2pp2dp8_v100_highfreq_compare.sh > "$log2" 2>&1
echo "[2/3] TP2PP2DP8 done."

# 3. TP4PP2DP4 GBS=32 highfreq
log3="/tmp/all_highfreq_$(date +%Y%m%d_%H%M%S)_tp4pp2dp4_gbs32.log"
echo "[3/3] Starting TP4PP2DP4 GBS=32 highfreq sweep... (log: $log3)"
RUN_TAG="highfreq_formal20_finetune_nosave_retest" TRAIN_STEPS="20" \
    bash scripts/run_real_qwen25_7b_tp4pp2dp4_gbs32_v100_highfreq_compare.sh > "$log3" 2>&1
echo "[3/3] TP4PP2DP4 GBS=32 done."

echo ""
echo "========================================"
echo "All 32-card high-frequency sweeps done."
echo "========================================"
