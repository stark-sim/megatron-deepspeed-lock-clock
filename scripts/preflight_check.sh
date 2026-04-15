#!/usr/bin/env bash
# =============================================================================
# scripts/preflight_check.sh
# 实验前控制变量检查脚本（双节点兼容）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# 默认阈值
# ---------------------------------------------------------------------------
GPU_UTIL_THRESHOLD=5          # GPU 利用率应低于此值 (%)
GPU_MEM_THRESHOLD_MB=1024     # GPU 显存使用应低于此值 (MB)
LOAD_THRESHOLD_PER_GPU=1.0    # 系统 load average / GPU 数量应低于此值
NCCL_TEST_TIMEOUT=30          # NCCL 网络测试超时 (秒)

# ---------------------------------------------------------------------------
# 用法
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") --node-list <nodes> --gpu-indices <indices> [options]

Required:
  --node-list <n1,n2,...>      逗号分隔的节点主机名或 IP（SSH 可达）
  --gpu-indices <i1,i2,...>    逗号分隔的目标 GPU 索引

Optional:
  --util-threshold <pct>       GPU 利用率阈值 (默认: ${GPU_UTIL_THRESHOLD}%)
  --mem-threshold-mb <mb>      GPU 显存阈值 (默认: ${GPU_MEM_THRESHOLD_MB} MB)
  --load-threshold <ratio>     系统负载 / GPU数 阈值 (默认: ${LOAD_THRESHOLD_PER_GPU})
  --nccl-test <script>         NCCL 通信测试脚本路径
  --json                       输出 JSON 格式报告
  --verbose                    输出详细检查过程
  -h, --help                   显示此帮助

Example:
  $(basename "$0") --node-list "v100x16-1,v100x16-2" --gpu-indices "8,9,10,11" --verbose
EOF
}

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
NODE_LIST=""
GPU_INDICES=""
UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD}"
MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB}"
LOAD_THRESHOLD="${LOAD_THRESHOLD_PER_GPU}"
NCCL_TEST_SCRIPT=""
OUTPUT_JSON=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-list) NODE_LIST="$2"; shift 2 ;;
        --gpu-indices) GPU_INDICES="$2"; shift 2 ;;
        --util-threshold) UTIL_THRESHOLD="$2"; shift 2 ;;
        --mem-threshold-mb) MEM_THRESHOLD_MB="$2"; shift 2 ;;
        --load-threshold) LOAD_THRESHOLD="$2"; shift 2 ;;
        --nccl-test) NCCL_TEST_SCRIPT="$2"; shift 2 ;;
        --json) OUTPUT_JSON=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$NODE_LIST" || -z "$GPU_INDICES" ]]; then
    echo "Error: --node-list and --gpu-indices are required." >&2
    usage >&2
    exit 1
fi

IFS=',' read -ra NODES <<< "$NODE_LIST"
IFS=',' read -ra GPUS <<< "$GPU_INDICES"
NUM_GPUS="${#GPUS[@]}"

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
logv() { if $VERBOSE; then echo "[PREFLIGHT] $*" >&2; fi; }

# 在远程节点执行命令（如果节点是 localhost/127.0.0.1，则本地执行）
run_on_node() {
    local node="$1"
    shift
    if [[ "$node" == "localhost" || "$node" == "127.0.0.1" ]]; then
        bash -c "$*"
    else
        ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no "$node" "$*"
    fi
}

# 收集某个节点的 GPU 状态（CSV 格式）
query_gpu_status() {
    local node="$1"
    local gpu_list="$2"
    run_on_node "$node" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,clocks.current.graphics,clocks.applications.graphics,name --format=csv,noheader,nounits -i ${gpu_list}"
}

# 检查系统负载
query_load() {
    local node="$1"
    run_on_node "$node" "cat /proc/loadavg | awk '{print \$1,\$2,\$3}'"
}

# 检查残留 CUDA/ML 进程
query_residue_processes() {
    local node="$1"
    run_on_node "$node" "ps aux | grep -E '(python|python3|cuda|vllm|nccl)' | grep -v grep | grep -v 'preflight_check' || true"
}

# 检查 nvidia-smi 中的计算进程
query_compute_processes() {
    local node="$1"
    local gpu_list="$2"
    run_on_node "$node" "nvidia-smi pmon -c 1 -i ${gpu_list} 2>/dev/null | grep -v '^#' | awk '\$3 != \"-\" {print}' || true"
}

# ---------------------------------------------------------------------------
# 状态变量
# ---------------------------------------------------------------------------
OVERALL_PASS=true
NODE_RESULTS=()

# ---------------------------------------------------------------------------
# 主检查循环
# ---------------------------------------------------------------------------
GPU_LIST_STR="${GPUS[*]}"
GPU_LIST_STR="${GPU_LIST_STR// /,}"

for node in "${NODES[@]}"; do
    logv "Checking node: $node"
    NODE_PASS=true
    NODE_MSGS=()
    
    # ---- 0. SSH 可达性 -----------------------------------------------------
    if ! run_on_node "$node" "true" >/dev/null 2>&1; then
        NODE_PASS=false
        OVERALL_PASS=false
        NODE_MSGS+=("FAIL: cannot reach node via SSH (or localhost exec failed)")
        NODE_RESULTS+=("{\"node\":\"$node\",\"reachable\":false,\"pass\":false,\"checks\":[]}")
        continue
    fi
    NODE_MSGS+=("PASS: node reachable")
    
    # ---- 1. GPU 空闲状态检查 ------------------------------------------------
    GPU_CSV=$(query_gpu_status "$node" "$GPU_LIST_STR" 2>/dev/null || true)
    if [[ -z "$GPU_CSV" ]]; then
        NODE_PASS=false
        OVERALL_PASS=false
        NODE_MSGS+=("FAIL: nvidia-smi query returned empty for GPUs [$GPU_LIST_STR]")
    else
        while IFS=',' read -r idx util mem temp cur_clk app_clk name; do
            idx=$(echo "$idx" | xargs)
            util=$(echo "$util" | xargs)
            mem=$(echo "$mem" | xargs)
            temp=$(echo "$temp" | xargs)
            cur_clk=$(echo "$cur_clk" | xargs)
            app_clk=$(echo "$app_clk" | xargs)
            name=$(echo "$name" | xargs)
            
            if (( $(echo "$util > $UTIL_THRESHOLD" | bc -l) )); then
                NODE_PASS=false
                OVERALL_PASS=false
                NODE_MSGS+=("FAIL: GPU $idx utilization=${util}% > threshold=${UTIL_THRESHOLD}%")
            else
                NODE_MSGS+=("PASS: GPU $idx utilization=${util}% (<= ${UTIL_THRESHOLD}%)")
            fi
            
            if (( mem > MEM_THRESHOLD_MB )); then
                NODE_PASS=false
                OVERALL_PASS=false
                NODE_MSGS+=("FAIL: GPU $idx memory.used=${mem}MB > threshold=${MEM_THRESHOLD_MB}MB")
            else
                NODE_MSGS+=("PASS: GPU $idx memory.used=${mem}MB (<= ${MEM_THRESHOLD_MB}MB)")
            fi
            
            NODE_MSGS+=("INFO: GPU $idx temp=${temp}°C clock=${cur_clk}/${app_clk}MHz name=${name}")
        done <<< "$GPU_CSV"
    fi
    
    # ---- 2. 系统负载检查 ----------------------------------------------------
    LOAD_AVG=$(query_load "$node" 2>/dev/null | awk '{print $1}')
    if [[ -n "$LOAD_AVG" ]]; then
        LOAD_OK=$(echo "$LOAD_AVG < $LOAD_THRESHOLD * $NUM_GPUS" | bc -l)
        if [[ "$LOAD_OK" -eq 0 ]]; then
            NODE_PASS=false
            OVERALL_PASS=false
            NODE_MSGS+=("FAIL: loadavg=${LOAD_AVG} >= threshold=${LOAD_THRESHOLD}*${NUM_GPUS}")
        else
            NODE_MSGS+=("PASS: loadavg=${LOAD_AVG} (< ${LOAD_THRESHOLD}*${NUM_GPUS})")
        fi
    else
        NODE_PASS=false
        OVERALL_PASS=false
        NODE_MSGS+=("WARN: could not read system load")
    fi
    
    # ---- 3. 残留进程检查 ----------------------------------------------------
    RESIDUE=$(query_residue_processes "$node")
    if [[ -n "$RESIDUE" ]]; then
        # 只警告，不强制失败（用户自己的 preflight ssh/ps 可能也在其中）
        NODE_MSGS+=("WARN: found running python/CUDA processes on node")
        while IFS= read -r line; do
            NODE_MSGS+=("INFO:   $line")
        done <<< "$RESIDUE"
    else
        NODE_MSGS+=("PASS: no obvious residue python/CUDA processes")
    fi
    
    # ---- 4. nvidia-smi pmon 计算进程检查 ------------------------------------
    COMPUTE_PROCS=$(query_compute_processes "$node" "$GPU_LIST_STR")
    if [[ -n "$COMPUTE_PROCS" ]]; then
        NODE_PASS=false
        OVERALL_PASS=false
        NODE_MSGS+=("FAIL: active compute processes found on target GPUs [$GPU_LIST_STR]")
        while IFS= read -r line; do
            NODE_MSGS+=("INFO:   $line")
        done <<< "$COMPUTE_PROCS"
    else
        NODE_MSGS+=("PASS: no active compute processes on target GPUs [$GPU_LIST_STR]")
    fi
    
    # ---- 5. 跨节点网络连通性（仅当节点数 > 1）-------------------------------
    if [[ ${#NODES[@]} -gt 1 ]]; then
        # 简单的 ping 测试
        if command -v ping >/dev/null 2>&1; then
            PING_TARGET="$node"
            if ping -c 1 -W 2 "$PING_TARGET" >/dev/null 2>&1; then
                NODE_MSGS+=("PASS: network ping to $PING_TARGET OK")
            else
                NODE_MSGS+=("WARN: network ping to $PING_TARGET failed")
            fi
        fi
    fi
    
    # 组装 JSON 片段
    CHECKS_JSON=$(printf '%s\n' "${NODE_MSGS[@]}" | python3 -c '
import sys, json
lines = [l.rstrip() for l in sys.stdin if l.strip()]
checks = []
for line in lines:
    parts = line.split(": ", 1)
    if len(parts) == 2:
        checks.append({"status": parts[0].lower(), "message": parts[1]})
    else:
        checks.append({"status": "info", "message": line})
print(json.dumps(checks))
')
    NODE_RESULTS+=("{\"node\":\"$node\",\"reachable\":true,\"pass\":$NODE_PASS,\"checks\":$CHECKS_JSON}")
done

# ---------------------------------------------------------------------------
# NCCL 通信测试（可选）
# ---------------------------------------------------------------------------
NCCL_PASS=true
NCCL_MSG=""
if [[ -n "$NCCL_TEST_SCRIPT" && -f "$NCCL_TEST_SCRIPT" ]]; then
    logv "Running NCCL test: $NCCL_TEST_SCRIPT"
    if bash "$NCCL_TEST_SCRIPT"; then
        NCCL_MSG="PASS: NCCL test completed"
    else
        NCCL_PASS=false
        OVERALL_PASS=false
        NCCL_MSG="FAIL: NCCL test failed"
    fi
else
    NCCL_MSG="SKIP: no NCCL test script provided"
fi

# ---------------------------------------------------------------------------
# 输出报告
# ---------------------------------------------------------------------------
if $OUTPUT_JSON; then
    printf '{%s"overall_pass":%s,"nodes":[%s],"nccl":{"pass":%s,"message":"%s"}}\n' \
        "" "$OVERALL_PASS" "$(IFS=,; echo "${NODE_RESULTS[*]}")" "$NCCL_PASS" "$NCCL_MSG"
else
    echo "=============================================="
    echo "  Preflight Check Report"
    echo "=============================================="
    echo "Target GPUs : [$GPU_LIST_STR]"
    echo "Nodes       : [${NODE_LIST}]"
    echo ""
    
    for i in "${!NODES[@]}"; do
        node="${NODES[$i]}"
        # 从 JSON 片段中提取可读信息
        json_frag="${NODE_RESULTS[$i]}"
        node_pass=$(echo "$json_frag" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("PASS" if d["pass"] else "FAIL")')
        echo "--- Node: $node [$node_pass] ---"
        echo "$json_frag" | python3 -c '
import sys, json
d = json.load(sys.stdin)
for c in d.get("checks", []):
    print("  [%4s] %s" % (c["status"].upper(), c["message"]))
'
        echo ""
    done
    
    echo "--- NCCL / Network ---"
    echo "  [$NCCL_PASS] $NCCL_MSG"
    echo ""
    
    if $OVERALL_PASS; then
        echo "=============================================="
        echo "  RESULT: ALL CHECKS PASSED"
        echo "=============================================="
    else
        echo "=============================================="
        echo "  RESULT: SOME CHECKS FAILED"
        echo "=============================================="
        echo "Please resolve the issues above before launching experiments."
    fi
fi

$OVERALL_PASS
