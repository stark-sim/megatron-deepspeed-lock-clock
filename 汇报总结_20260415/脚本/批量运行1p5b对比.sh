#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_PATH="${BASE_PATH:-$(cd "${PACKAGE_DIR}/.." && pwd)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${BASE_PATH}/experiments}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PACKAGE_DIR}/结果}"
REPORT_DIR="${REPORT_DIR:-${RESULT_ROOT}/1p5b_demo_sweep_${STAMP}}"
STATIC_FREQS="${STATIC_FREQS:-1080 1200}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-qwen1p5b_demo}"

mkdir -p "${REPORT_DIR}"

MANIFEST_TSV="${REPORT_DIR}/运行清单.tsv"
COMPARE_MD="${REPORT_DIR}/对比结果.md"

printf '标签\t运行ID\t运行目录\n' > "${MANIFEST_TSV}"

declare -a RUN_DIRS=()

launch_one() {
    local label="$1"
    local mode="$2"
    local static_clock="${3:-}"
    local run_id="${EXPERIMENT_PREFIX}_${label}_${STAMP}"
    local run_dir="${EXPERIMENT_ROOT}/${run_id}"

    echo "[批量演示] 启动 ${label} -> ${run_id}"

    if [[ -n "${static_clock}" ]]; then
        EXPERIMENT_MODE="${mode}" \
        EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${label}" \
        RUN_ID="${run_id}" \
        STATIC_CLOCK_MHZ="${static_clock}" \
        bash "${PACKAGE_DIR}/脚本/运行1p5b演示.sh"
    else
        EXPERIMENT_MODE="${mode}" \
        EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${label}" \
        RUN_ID="${run_id}" \
        bash "${PACKAGE_DIR}/脚本/运行1p5b演示.sh"
    fi

    printf '%s\t%s\t%s\n' "${label}" "${run_id}" "${run_dir}" >> "${MANIFEST_TSV}"
    RUN_DIRS+=("${run_dir}")
}

launch_one baseline baseline

for freq in ${STATIC_FREQS}; do
    launch_one "static${freq}" static "${freq}"
done

python3 "${PACKAGE_DIR}/脚本/对比Zeus结果.py" "${RUN_DIRS[@]}" > "${COMPARE_MD}"

cat <<EOF
[批量演示] 运行清单: ${MANIFEST_TSV}
[批量演示] 对比表: ${COMPARE_MD}
EOF
