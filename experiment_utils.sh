#!/bin/bash

json_escape() {
    python3 - <<'PY' "$1"
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}


resolve_python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "${PYTHON_BIN}"
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    command -v python
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
    local python_bin
    python_bin="$(resolve_python_bin)"

    sudo -n "$python_bin" - "$gpu_indices_csv" "$clock_mhz" <<'PY'
import os
import sys

site_candidates = [
    os.path.expanduser('~/.local/lib/python3.10/site-packages'),
    '/home/sd/.local/lib/python3.10/site-packages',
]
sandbox_user = os.environ.get('SUDO_USER') or os.environ.get('USER')
if sandbox_user:
    site_candidates.insert(0, f'/home/{sandbox_user}/.local/lib/python3.10/site-packages')
for fallback in site_candidates:
    if os.path.isdir(fallback) and fallback not in sys.path:
        sys.path.insert(0, fallback)
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
    local python_bin
    python_bin="$(resolve_python_bin)"

    sudo -n "$python_bin" - "$gpu_indices_csv" <<'PY'
import os
import sys

site_candidates = [
    os.path.expanduser('~/.local/lib/python3.10/site-packages'),
    '/home/sd/.local/lib/python3.10/site-packages',
]
sandbox_user = os.environ.get('SUDO_USER') or os.environ.get('USER')
if sandbox_user:
    site_candidates.insert(0, f'/home/{sandbox_user}/.local/lib/python3.10/site-packages')
for fallback in site_candidates:
    if os.path.isdir(fallback) and fallback not in sys.path:
        sys.path.insert(0, fallback)
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
    local python_bin
    python_bin="$(resolve_python_bin)"

    sudo -n "$python_bin" - "$gpu_indices_csv" "$clock_mhz" <<'PY'
import os
import sys

site_candidates = [
    os.path.expanduser('~/.local/lib/python3.10/site-packages'),
    '/home/sd/.local/lib/python3.10/site-packages',
]
sandbox_user = os.environ.get('SUDO_USER') or os.environ.get('USER')
if sandbox_user:
    site_candidates.insert(0, f'/home/{sandbox_user}/.local/lib/python3.10/site-packages')
for fallback in site_candidates:
    if os.path.isdir(fallback) and fallback not in sys.path:
        sys.path.insert(0, fallback)
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


sync_file_to_host() {
    local source_path="$1"
    local host="$2"
    local destination_path="$3"
    local destination_dir
    destination_dir="$(dirname "$destination_path")"

    ssh "$host" "mkdir -p '$destination_dir'"
    ssh "$host" "cat > '$destination_path'" < "$source_path"
}


is_local_host_alias() {
    local host="$1"
    python3 - "$host" <<'PY'
import os
import socket
import subprocess
import sys

candidate = sys.argv[1].strip()
if not candidate:
    print("0")
    raise SystemExit(0)

loopback_names = {"localhost", "127.0.0.1", "::1"}
if candidate in loopback_names:
    print("1")
    raise SystemExit(0)

local_names = set()
for value in [os.environ.get("HOSTNAME")]:
    if value:
        local_names.add(value)
        local_names.add(value.split(".", 1)[0])

for getter in (socket.gethostname, socket.getfqdn):
    try:
        value = getter()
    except Exception:
        continue
    if value:
        local_names.add(value)
        local_names.add(value.split(".", 1)[0])

candidate_names = {candidate, candidate.split(".", 1)[0]}
if local_names & candidate_names:
    print("1")
    raise SystemExit(0)

def resolve_all(name: str):
    resolved = set()
    try:
        infos = socket.getaddrinfo(name, None, proto=socket.IPPROTO_TCP)
    except Exception:
        infos = []
    for info in infos:
        addr = info[4][0]
        resolved.add(addr)
    return resolved

local_ips = {"127.0.0.1", "::1"}
for name in list(local_names):
    local_ips.update(resolve_all(name))

try:
    hostname_i = subprocess.run(
        ["hostname", "-I"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
except Exception:
    hostname_i = ""

for addr in hostname_i.split():
    local_ips.add(addr.strip())

candidate_ips = resolve_all(candidate)
print("1" if (candidate_ips and local_ips & candidate_ips) else "0")
PY
}


_run_remote_gpu_clock_action() {
    local host="$1"
    local conda_env="$2"
    local action="$3"
    local gpu_indices_csv="$4"
    local clock_mhz="${5:-}"

    ssh "$host" "CONDA_ENV='$conda_env' ACTION='$action' GPU_INDICES='$gpu_indices_csv' CLOCK_MHZ='$clock_mhz' bash -s" <<'EOS'
set -euo pipefail

if [[ -n "${CONDA_ENV:-}" && -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

python_bin="$(command -v python3 2>/dev/null || command -v python)"
sudo -n ACTION="${ACTION}" GPU_INDICES="${GPU_INDICES}" CLOCK_MHZ="${CLOCK_MHZ}" "$python_bin" - <<'PY'
import os
import sys

site_candidates = [
    os.path.expanduser('~/.local/lib/python3.10/site-packages'),
    '/home/sd/.local/lib/python3.10/site-packages',
]
sandbox_user = os.environ.get('SUDO_USER') or os.environ.get('USER')
if sandbox_user:
    site_candidates.insert(0, f'/home/{sandbox_user}/.local/lib/python3.10/site-packages')
for fallback in site_candidates:
    if os.path.isdir(fallback) and fallback not in sys.path:
        sys.path.insert(0, fallback)
import pynvml

action = os.environ['ACTION']
indices = [int(x) for x in os.environ.get('GPU_INDICES', '').split(',') if x.strip()]
clock_mhz = int(os.environ.get('CLOCK_MHZ') or 0)

pynvml.nvmlInit()
try:
    for index in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        if action == 'assert':
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
            mem_clock = mem_clocks[0]
            graphics = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, mem_clock)
            if clock_mhz not in graphics:
                raise SystemExit(f'GPU {index} does not support static clock {clock_mhz} MHz; supported: {graphics}')
            print(f'GPU {index} supports {clock_mhz} MHz')
        elif action == 'lock':
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, clock_mhz, clock_mhz)
            print(f'GPU {index} locked to {clock_mhz} MHz')
        elif action == 'reset':
            try:
                pynvml.nvmlDeviceResetGpuLockedClocks(handle)
            except pynvml.NVMLError:
                pass
            try:
                pynvml.nvmlDeviceResetApplicationsClocks(handle)
            except pynvml.NVMLError:
                pass
            print(f'GPU {index} reset to default clocks')
        else:
            raise SystemExit(f'Unknown action: {action}')
finally:
    pynvml.nvmlShutdown()
PY
EOS
}


assert_remote_static_clock_supported() {
    local host="$1"
    local conda_env="$2"
    local gpu_indices_csv="$3"
    local clock_mhz="$4"
    _run_remote_gpu_clock_action "$host" "$conda_env" "assert" "$gpu_indices_csv" "$clock_mhz"
}


lock_remote_gpu_clocks() {
    local host="$1"
    local conda_env="$2"
    local gpu_indices_csv="$3"
    local clock_mhz="$4"
    _run_remote_gpu_clock_action "$host" "$conda_env" "lock" "$gpu_indices_csv" "$clock_mhz"
}


reset_remote_gpu_clocks() {
    local host="$1"
    local conda_env="$2"
    local gpu_indices_csv="$3"
    _run_remote_gpu_clock_action "$host" "$conda_env" "reset" "$gpu_indices_csv"
}


run_remote_preflight() {
    local host="$1"
    local base_path="$2"
    local expected_tp="$3"
    local mode="$4"
    local static_clock="$5"
    local cuda_visible_devices="$6"
    local launcher="$7"
    local dataset="$8"
    local tokenizer_path="$9"
    local conda_env="${10}"

    ssh "$host" "BASE_PATH='$base_path' EXPECTED_TP='$expected_tp' EXPERIMENT_MODE='$mode' STATIC_CLOCK_MHZ='$static_clock' CUDA_VISIBLE_DEVICES='$cuda_visible_devices' LAUNCHER='$launcher' DATASET='$dataset' TOKENIZER_PATH='$tokenizer_path' CONDA_ENV='$conda_env' bash -s" <<'EOS'
set -euo pipefail

if [[ -n "${CONDA_ENV:-}" && -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

cd "$BASE_PATH"
python3 - <<'PY'
import json
import os
import socket
import subprocess
import sys

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
result['checks']['python_available'] = command_ok(['python3', '--version'])[0]
result['checks']['nvidia_smi_available'] = command_ok(['bash', '-lc', 'command -v nvidia-smi >/dev/null 2>&1'])[0]
dataset_prefix = os.environ.get('DATASET', '')
tokenizer_path = os.environ.get('TOKENIZER_PATH', '')
result['checks']['dataset_exists'] = (
    bool(dataset_prefix)
    and os.path.exists(f'{dataset_prefix}.bin')
    and os.path.exists(f'{dataset_prefix}.idx')
)
result['checks']['tokenizer_exists'] = (
    bool(tokenizer_path)
    and os.path.isdir(tokenizer_path)
    and (
        os.path.exists(os.path.join(tokenizer_path, 'tokenizer.json'))
        or os.path.exists(os.path.join(tokenizer_path, 'tokenizer_config.json'))
    )
)
if os.environ.get('LAUNCHER', 'deepspeed') == 'deepspeed':
    result['checks']['launcher_available'] = command_ok(['bash', '-lc', 'command -v deepspeed >/dev/null 2>&1'])[0]
else:
    result['checks']['launcher_available'] = command_ok(['python3', '-m', 'torch.distributed.run', '--help'])[0]
try:
    fallback = os.path.expanduser('~/.local/lib/python3.10/site-packages')
    if os.path.isdir(fallback) and fallback not in sys.path:
        sys.path.insert(0, fallback)
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
