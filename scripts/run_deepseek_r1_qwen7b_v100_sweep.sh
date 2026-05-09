#!/bin/bash
set -euo pipefail

# DeepSeek-R1-Distill-Qwen-7B V100 单节点多频点批量实验脚本
# 顺序运行 baseline + 多个 static 频点
#
# 使用方式:
#   bash scripts/run_deepseek_r1_qwen7b_v100_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ "${HOSTNAME%%.*}" != "v100x16-1" && "${HOSTNAME%%.*}" != "DGX2-1" ]]; then
    echo "[Error] Run this script from DGX2-1 / v100x16-1." >&2
    exit 1
fi

# 频点列表 (MHz) — 从低到高，避免温度累积影响
# 已跑: baseline(1380), 1260
# 待跑: 1155, 1080, 990
FREQS=(1155 1080 990)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/deepseek_sweep_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "DeepSeek-R1-Distill-Qwen-7B V100 Sweep"
echo "========================================"
echo "Base path: ${BASE_PATH}"
echo "Log dir:   ${LOG_DIR}"
echo "Freqs:     ${FREQS[*]}"
echo ""

# 每个频点之间等待 GPU 冷却
COOLDOWN_SEC=30

for freq in "${FREQS[@]}"; do
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] Starting static ${freq} MHz ..."
    
    export EXPERIMENT_NAME="deepseek_static_${freq}_20steps"
    export EXPERIMENT_MODE="static"
    export STATIC_CLOCK_MHZ="${freq}"
    export TRAIN_STEPS=20
    
    LOG_FILE="${LOG_DIR}/deepseek_static_${freq}_20steps.log"
    
    if bash "${BASE_PATH}/scripts/run_deepseek_r1_distill_qwen7b_v100_single_node.sh" >"${LOG_FILE}" 2>&1; then
        echo "    [$(date '+%H:%M:%S')] static ${freq} MHz completed OK"
        # 提取 Zeus 汇总
        grep "\[Zeus\] Steps" "${LOG_FILE}" || echo "    (no Zeus summary found)"
    else
        echo "    [$(date '+%H:%M:%S')] static ${freq} MHz FAILED (exit $?)"
        echo "    Log: ${LOG_FILE}"
    fi
    
    # 确认 GPU 时钟已重置
    echo "    Resetting GPU clocks..."
    for i in {0..7}; do
        sudo -n nvidia-smi -i "${i}" -rgc >/dev/null 2>&1 || true
    done
    
    # 冷却等待
    if [[ "${freq}" != "${FREQS[-1]}" ]]; then
        echo "    Cooling down for ${COOLDOWN_SEC}s ..."
        sleep "${COOLDOWN_SEC}"
    fi
done

echo ""
echo "========================================"
echo "Sweep completed. Logs in: ${LOG_DIR}"
echo "========================================"

# 汇总所有结果
echo ""
echo "=== Zeus Summary ==="
for freq in "${FREQS[@]}"; do
    LOG_FILE="${LOG_DIR}/deepseek_static_${freq}_20steps.log"
    if [[ -f "${LOG_FILE}" ]]; then
        echo "--- static ${freq} MHz ---"
        grep "\[Zeus\] Steps" "${LOG_FILE}" || echo "  (no Zeus summary)"
    fi
done
