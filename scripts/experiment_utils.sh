#!/bin/bash

json_escape() {
    python3 - <<'PY' "$1"
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}


generate_run_id() {
    local prefix="$1"
    local ts host sanitized_prefix
    ts="$(date +%Y%m%d_%H%M%S)"
    host="${HOSTNAME:-$(hostname -s)}"
    host="${host%%.*}"
    sanitized_prefix="$(echo "$prefix" | tr ' /' '__' | tr -cd '[:alnum:]_.-')"
    printf '%s_%s_%s' "$sanitized_prefix" "$ts" "$host"
}


parse_cuda_visible_devices() {
    local raw="${CUDA_VISIBLE_DEVICES:-}"
    if [[ -z "$raw" ]]; then
        python3 - <<'PY'
import subprocess
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print('')
else:
    print(','.join(line.strip() for line in result.stdout.splitlines() if line.strip()))
PY
        return
    fi
    echo "$raw" | tr -d ' '
}


count_csv_items() {
    local csv="$1"
    if [[ -z "$csv" ]]; then
        echo 0
        return
    fi
    awk -F',' '{print NF}' <<< "$csv"
}


setup_experiment_run() {
    local base_path="$1"
    local experiment_name="$2"

    export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$experiment_name}"
    export MEGATRON_LAUNCHER_SCRIPT="$(basename "$0")"
    export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${base_path}/experiments}"
    export RUN_ID="${RUN_ID:-$(generate_run_id "$EXPERIMENT_NAME")}"
    export RUN_DIR="${EXPERIMENT_ROOT}/${RUN_ID}"
    export RUN_LOG_DIR="${RUN_DIR}/logs"

    mkdir -p "${RUN_DIR}" "${RUN_LOG_DIR}"

    echo "[Experiment] experiment_name=${EXPERIMENT_NAME}"
    echo "[Experiment] run_id=${RUN_ID}"
    echo "[Experiment] run_dir=${RUN_DIR}"
}


write_command_snapshot() {
    local destination="$1"
    shift

    {
        echo '#!/usr/bin/env bash'
        printf 'export MEGATRON_RUN_ID=%q\n' "$RUN_ID"
        printf 'export MEGATRON_EXPERIMENT_ROOT=%q\n' "$EXPERIMENT_ROOT"
        printf 'export MEGATRON_LAUNCHER_SCRIPT=%q\n' "$MEGATRON_LAUNCHER_SCRIPT"
        printf '%q ' "$@"
        echo
    } > "$destination"
    chmod +x "$destination"
}


lock_gpu_clocks() {
    local gpu_indices_csv="$1"
    local clock_mhz="$2"

    sudo -n python3 - "$gpu_indices_csv" "$clock_mhz" <<'PY'
import sys

sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml

indices = [int(x) for x in sys.argv[1].split(',') if x.strip()]
clock_mhz = int(sys.argv[2])

pynvml.nvmlInit()
try:
    for index in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, clock_mhz, clock_mhz)
        print(f'GPU {index} locked to {clock_mhz} MHz')
finally:
    pynvml.nvmlShutdown()
PY
}


reset_gpu_clocks() {
    local gpu_indices_csv="$1"

    sudo -n python3 - "$gpu_indices_csv" <<'PY'
import sys

sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml

indices = [int(x) for x in sys.argv[1].split(',') if x.strip()]

pynvml.nvmlInit()
try:
    for index in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        except pynvml.NVMLError:
            pass
        try:
            pynvml.nvmlDeviceResetApplicationsClocks(handle)
        except pynvml.NVMLError:
            pass
        print(f'GPU {index} reset to default clocks')
finally:
    pynvml.nvmlShutdown()
PY
}


assert_static_clock_supported() {
    local gpu_indices_csv="$1"
    local clock_mhz="$2"

    sudo -n python3 - "$gpu_indices_csv" "$clock_mhz" <<'PY'
import sys

sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')
import pynvml

indices = [int(x) for x in sys.argv[1].split(',') if x.strip()]
clock_mhz = int(sys.argv[2])

pynvml.nvmlInit()
try:
    for index in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
        mem_clock = mem_clocks[0]
        graphics = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, mem_clock)
        if clock_mhz not in graphics:
            raise SystemExit(f'GPU {index} does not support static clock {clock_mhz} MHz; supported: {graphics}')
finally:
    pynvml.nvmlShutdown()
PY
}


get_hostfile_raw_text() {
    local hostfile_path="$1"
    python3 - "$hostfile_path" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
PY
}


parse_hostfile_to_json() {
    local hostfile_path="$1"
    local nnodes="$2"
    python3 - "$hostfile_path" "$nnodes" <<'PY'
import hashlib
import json
import pathlib
import sys

hostfile_path = pathlib.Path(sys.argv[1])
requested_nnodes = int(sys.argv[2])
raw = hostfile_path.read_text(encoding='utf-8')
entries = []
for raw_line in raw.splitlines():
    line = raw_line.strip()
    if not line or line.startswith('#'):
        continue
    parts = line.split()
    hostname = parts[0]
    slots = None
    for part in parts[1:]:
        if part.startswith('slots='):
            slots = int(part.split('=', 1)[1])
    entries.append({'hostname': hostname, 'slots': slots})
selected_hosts = entries[:requested_nnodes] if requested_nnodes > 0 else entries
payload = {
    'path': str(hostfile_path),
    'sha1': hashlib.sha1(raw.encode('utf-8')).hexdigest(),
    'raw_text': raw,
    'entries': entries,
    'selected_hosts': selected_hosts,
    'selection_reason': f'first {requested_nnodes} host(s) from hostfile order' if requested_nnodes > 0 else 'all hosts',
    'preflight_required': requested_nnodes > 1,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}


write_topology_json() {
    local output_path="$1"
    local local_gpu_indices="$2"
    local nnodes="$3"
    local node_rank="$4"
    local master_addr="$5"
    local master_port="$6"
    local tp="$7"
    local pp="$8"
    local launcher="$9"
    python3 - "$output_path" "$local_gpu_indices" "$nnodes" "$node_rank" "$master_addr" "$master_port" "$tp" "$pp" "$launcher" <<'PY'
import json
import pathlib
import sys

output_path = pathlib.Path(sys.argv[1])
indices = [int(x) for x in sys.argv[2].split(',') if x.strip()]
payload = {
    'launcher': sys.argv[9],
    'nproc_per_node': len(indices),
    'visible_gpu_indices': indices,
    'nnodes': int(sys.argv[3]),
    'node_rank': int(sys.argv[4]),
    'master_addr': sys.argv[5],
    'master_port': sys.argv[6],
    'world_size': len(indices) * int(sys.argv[3]),
    'tp': int(sys.argv[7]),
    'pp': int(sys.argv[8]),
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
}


run_remote_preflight() {
    local host="$1"
    local base_path="$2"
    local expected_tp="$3"
    local mode="$4"
    local static_clock="$5"
    local cuda_visible_devices="$6"
    local launcher="$7"

    ssh "$host" "BASE_PATH='$base_path' EXPECTED_TP='$expected_tp' EXPERIMENT_MODE='$mode' STATIC_CLOCK_MHZ='$static_clock' CUDA_VISIBLE_DEVICES='$cuda_visible_devices' LAUNCHER='$launcher' bash -s" <<'EOS'
set -euo pipefail
cd "$BASE_PATH"
python3 - <<'PY'
import json
import os
import socket
import subprocess
import sys

sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')

result = {
    'hostname': socket.gethostname(),
    'repo_path': os.getcwd(),
    'checks': {},
}

def command_ok(cmd):
    completed = subprocess.run(cmd, capture_output=True, text=True)
    return completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip()

visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if visible.strip():
    gpu_indices = [int(x.strip()) for x in visible.split(',') if x.strip()]
else:
    ok, stdout, _ = command_ok(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'])
    gpu_indices = [int(x.strip()) for x in stdout.splitlines() if x.strip()] if ok else []

result['cuda_visible_devices'] = visible
result['gpu_indices'] = gpu_indices
result['local_gpu_count'] = len(gpu_indices)
expected_tp = int(os.environ['EXPECTED_TP'])
result['checks']['tp_fits_local_node'] = len(gpu_indices) >= expected_tp
result['checks']['repo_exists'] = os.path.exists(os.getcwd())
result['checks']['deepspeed_available'] = command_ok(['bash', '-lc', 'command -v deepspeed >/dev/null 2>&1'])[0]
result['checks']['python_available'] = command_ok(['python3', '--version'])[0]
result['checks']['nvidia_smi_available'] = command_ok(['bash', '-lc', 'command -v nvidia-smi >/dev/null 2>&1'])[0]
try:
    import pynvml
    result['checks']['pynvml_available'] = True
except Exception:
    result['checks']['pynvml_available'] = False

mode = os.environ.get('EXPERIMENT_MODE', '')
static_clock = os.environ.get('STATIC_CLOCK_MHZ', '')
if mode == 'static' and static_clock and result['checks']['pynvml_available'] and gpu_indices:
    import pynvml
    pynvml.nvmlInit()
    try:
        clock = int(static_clock)
        support = True
        unsupported = []
        for index in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)[0]
            graphics = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, mem)
            if clock not in graphics:
                support = False
                unsupported.append({'gpu_index': index, 'supported_graphics_clocks_mhz': graphics})
        result['checks']['static_clock_supported'] = support
        if unsupported:
            result['static_clock_unsupported'] = unsupported
    finally:
        pynvml.nvmlShutdown()

result['ok'] = all(bool(v) for v in result['checks'].values())
print(json.dumps(result, sort_keys=True))
PY
EOS
}


write_preflight_json() {
    local output_path="$1"
    shift
    printf '%s\n' "$1" > "$output_path"
}
