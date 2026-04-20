#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_TAG="${RUNTIME_TAG:-megatron_lock_clock}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/dev/shm/${RUNTIME_TAG}}"

if [[ ! -d /dev/shm || ! -w /dev/shm ]]; then
    RUNTIME_ROOT="/tmp/${RUNTIME_TAG}"
fi

export PYTHONPATH="${BASE_PATH}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export MAX_JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUNTIME_ROOT}/torch_extensions}"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${RUNTIME_ROOT}/pycache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_ROOT}/triton}"

mkdir -p \
    "${TORCH_EXTENSIONS_DIR}" \
    "${TMPDIR}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${TRITON_CACHE_DIR}"

echo "[runtime-env] BASE_PATH=${BASE_PATH}"
echo "[runtime-env] TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "[runtime-env] TMPDIR=${TMPDIR}"
echo "[runtime-env] PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX}"
echo "[runtime-env] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
