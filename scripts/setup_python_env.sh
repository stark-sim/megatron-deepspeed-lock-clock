#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_BIN="${CONDA_BIN:-}"
ENV_NAME="${ENV_NAME:-megatron-lock-clock}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
STACK_PROFILE="${STACK_PROFILE:-sd-eth}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_APEX="${INSTALL_APEX:-1}"
INSTALL_ZEUS="${INSTALL_ZEUS:-1}"
RUN_VERIFY="${RUN_VERIFY:-1}"
MAX_JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
APEX_REPO_URL="${APEX_REPO_URL:-https://github.com/NVIDIA/apex.git}"
APEX_REF="${APEX_REF:-master}"
APEX_SRC_DIR="${APEX_SRC_DIR:-${BASE_PATH}/.build/apex-${APEX_REF//\//-}}"

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[setup] missing required command: $cmd" >&2
        exit 1
    fi
}

resolve_conda_bin() {
    if [[ -n "$CONDA_BIN" ]]; then
        printf '%s\n' "$CONDA_BIN"
        return
    fi
    if command -v mamba >/dev/null 2>&1; then
        command -v mamba
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return
    fi
    echo "[setup] neither mamba nor conda is available" >&2
    exit 1
}

run_in_env() {
    "$CONDA_BIN" run -n "$ENV_NAME" "$@"
}

ensure_env_exists() {
    if "$CONDA_BIN" run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
        echo "[setup] conda env already exists: $ENV_NAME"
        return
    fi
    echo "[setup] creating conda env $ENV_NAME (python=$PYTHON_VERSION)"
    "$CONDA_BIN" create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
}

select_profile_versions() {
    case "$STACK_PROFILE" in
        sd-eth)
            TORCH_VERSION="2.10.0+cu128"
            DEEPSPEED_VERSION="0.14.0"
            EINOPS_VERSION="0.8.2"
            ZEUS_VERSION="0.11.0.post1"
            ;;
        dgx-v100)
            TORCH_VERSION="2.9.1+cu128"
            DEEPSPEED_VERSION="0.18.3"
            EINOPS_VERSION="0.8.1"
            ZEUS_VERSION="0.11.0.post1"
            ;;
        *)
            echo "[setup] unsupported STACK_PROFILE=$STACK_PROFILE" >&2
            echo "[setup] supported profiles: sd-eth, dgx-v100" >&2
            exit 1
            ;;
    esac

    TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.3}"
    SENTENCEPIECE_VERSION="${SENTENCEPIECE_VERSION:-0.2.1}"
    PYNVML_VERSION="${PYNVML_VERSION:-12.0.0}"
    DATASETS_VERSION="${DATASETS_VERSION:-3.6.0}"
    TQDM_VERSION="${TQDM_VERSION:-4.67.1}"
}

install_base_build_tools() {
    local -a packages=(
        pip
        setuptools
        wheel
        packaging
        ninja
        cmake
        pybind11
        "numpy<2"
        psutil
        regex
    )
    if [[ -n "${TQDM_VERSION:-}" ]]; then
        packages+=("tqdm==${TQDM_VERSION}")
    fi

    echo "[setup] installing base build tools"
    run_in_env python -m pip install --upgrade "${packages[@]}"
}

install_pytorch_stack() {
    echo "[setup] installing torch from $TORCH_INDEX_URL"
    run_in_env python -m pip install \
        --index-url "$TORCH_INDEX_URL" \
        "torch==${TORCH_VERSION}"
}

install_python_packages() {
    local -a packages=(
        "transformers==${TRANSFORMERS_VERSION}"
        "einops==${EINOPS_VERSION}"
        "sentencepiece==${SENTENCEPIECE_VERSION}"
        "pynvml==${PYNVML_VERSION}"
        "datasets==${DATASETS_VERSION}"
    )

    echo "[setup] installing Python packages"
    run_in_env python -m pip install "${packages[@]}"

    if [[ "$INSTALL_ZEUS" == "1" ]]; then
        run_in_env python -m pip install "zeus-ml==${ZEUS_VERSION}"
    fi
}

prebuild_deepspeed_ops() {
    require_command nvcc
    echo "[setup] prebuilding DeepSpeed CPUAdam/FusedAdam"
    run_in_env env \
        DS_BUILD_CPU_ADAM=1 \
        DS_BUILD_FUSED_ADAM=1 \
        DS_BUILD_UTILS=1 \
        MAX_JOBS="$MAX_JOBS" \
        python -m pip install --force-reinstall --no-build-isolation "deepspeed==${DEEPSPEED_VERSION}"
}

prepare_apex_source() {
    require_command git
    mkdir -p "$(dirname "$APEX_SRC_DIR")"
    if [[ -f "${APEX_SRC_DIR}/setup.py" ]]; then
        echo "[setup] reusing apex source at $APEX_SRC_DIR"
        return
    fi
    echo "[setup] cloning apex (${APEX_REF}) to $APEX_SRC_DIR"
    rm -rf "$APEX_SRC_DIR"
    git clone --depth 1 --branch "$APEX_REF" "$APEX_REPO_URL" "$APEX_SRC_DIR"
}

install_apex() {
    if [[ "$INSTALL_APEX" != "1" ]]; then
        echo "[setup] skipping apex installation"
        return
    fi
    require_command nvcc
    prepare_apex_source
    echo "[setup] installing apex with CUDA extensions"
    run_in_env env \
        APEX_CPP_EXT=1 \
        APEX_CUDA_EXT=1 \
        MAX_JOBS="$MAX_JOBS" \
        python -m pip install --no-build-isolation -v "$APEX_SRC_DIR"
}

install_repo_editable() {
    echo "[setup] installing repo in editable mode"
    run_in_env python -m pip install --no-build-isolation -e "$BASE_PATH"
}

run_verify() {
    if [[ "$RUN_VERIFY" != "1" ]]; then
        return
    fi
    echo "[setup] running runtime verification and warmup"
    "$CONDA_BIN" run -n "$ENV_NAME" bash -lc "cd '$BASE_PATH' && source scripts/activate_runtime_env.sh >/dev/null && python scripts/verify_python_env.py --warmup"
}

main() {
    CONDA_BIN="$(resolve_conda_bin)"
    select_profile_versions

    echo "[setup] base_path=$BASE_PATH"
    echo "[setup] env_name=$ENV_NAME"
    echo "[setup] stack_profile=$STACK_PROFILE"
    echo "[setup] torch=$TORCH_VERSION deepspeed=$DEEPSPEED_VERSION"

    ensure_env_exists
    install_base_build_tools
    install_pytorch_stack
    install_python_packages
    prebuild_deepspeed_ops
    install_apex
    install_repo_editable
    run_verify

    cat <<EOF
[setup] done
[setup] activate with:
  conda activate $ENV_NAME
  source ${BASE_PATH}/scripts/activate_runtime_env.sh
[setup] verify again with:
  python ${BASE_PATH}/scripts/verify_python_env.py --warmup
EOF
}

main "$@"
