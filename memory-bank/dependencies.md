# Dependencies

## Hardware / Runtime Inventory Notes
- [2026-04-28] `user@sd-1` 与 `user@sd-2` 当前登录环境均暴露 `2 × AMD EPYC 7K62`（`96` 物理核 / `192` 线程），但内存由 cgroup 限为 `100 GiB`。`sd-1` 容器 rootfs 容量约 `5.3T`，同时可见额外 `2 × 21.8T HDD` 与 `2 × 1.8T NVMe`；`sd-2` 当前登录环境只可见约 `1.7T` rootfs 磁盘。

## Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | `2.9.1+cu128` | Core training runtime and distributed execution |
| `deepspeed` | `0.18.3` | Distributed training launcher and ZeRO runtime |
| `pynvml` | installed in remote user site-packages | GPU clock control and NVML queries |
| `transformers` | `4.57.3` | Tokenizer/model ecosystem support |
| `einops` | `0.8.1` | Tensor reshaping utilities |
| `sentencepiece` | `0.2.1` | Tokenization support |
| `regex` | installed | Tokenizer/runtime dependency |

## Reproducible Profiles
| Profile | Python | `torch` | `deepspeed` | `einops` | `zeus-ml` | Intended Hosts |
|---------|--------|---------|-------------|----------|-----------|----------------|
| `sd-eth` | `3.10` | `2.10.0+cu128` | `0.14.0` | `0.8.2` | `0.11.0.post1` | `sd-1` / `sd-2` Ethernet line |
| `dgx-v100` | `3.10` | `2.9.1+cu128` | `0.18.3` | `0.8.1` | `0.11.0.post1` | `v100x16-1` / `v100x16-2` |

## Build-Time Requirements
| Tool / Package | Purpose |
|----------------|---------|
| `conda` or `mamba` | Create the canonical Python runtime |
| `git` | Fetch `NVIDIA/apex` source when not pre-provided |
| `nvcc` | Build `apex` CUDA extensions and prebuild DeepSpeed Adam ops |
| `APEX_CPP_EXT=1`, `APEX_CUDA_EXT=1` | Required for the Megatron CUDA apex path used by this repo |
| `DS_BUILD_CPU_ADAM=1`, `DS_BUILD_FUSED_ADAM=1` | Prebuild DeepSpeed Adam ops during environment setup |

## Dev Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| `bash` | system | Launcher scripting |
| `screen` | `4.09.00` | Detached remote experiment sessions |
| `~/.local` Python env | Python `3.10` user-site environment exposed via `~/.local/bin` and `~/.local/lib/python3.10/site-packages` | Canonical remote runtime for this project, including previously successful Megatron-DeepSpeed launches |
| `tp4bit` Conda env on `sd-1` / `sd-2` | Python `3.10`, `torch 2.10.0+cu128`, `deepspeed 0.14.0`, `einops 0.8.2`, `zeus-ml 0.11.0.post1` | Currently the only verified runtime on `sd-1` / `sd-2` that can import training dependencies, see 4 GPUs, and emit Zeus-backed Ethernet artifacts |
| `BasicTeX` | TeX Live `2026` basic distribution | Local paper build runtime providing `/Library/TeX/texbin/pdflatex` and `/Library/TeX/texbin/bibtex` |
| user-mode `tlmgr` packages | `ieeetran`, `algorithms`, `multirow`, `subfigure`, `courier` | Additional LaTeX packages/fonts required for compiling `.context/paper/main.tex` under BasicTeX |
| local Miniconda Python | `$(HOME)/miniconda3/bin/python3` with `matplotlib` and `numpy` available | Current stable local interpreter for `.context/paper/generate_figures.py`, because the default login-shell `python3` can resolve to an Xcode shim without plotting packages |
| `Qwen2.5 tokenizer flat copy` | snapshot-derived local files | Use `/home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat` on both DGX2 nodes to avoid Hugging Face snapshot symlink/blob breakage on `sd@v100x16-2` |
| `nvidia-smi` | system | GPU inspection and validation |

## Version Constraints
- [2026-04-24] `DGX2-1` 上用户新建的 `conda` 环境 `megatron-bridge`（Python `3.12`）目前至少缺少 `transformer_engine`，Bridge 导入会在 `/home/sd/Megatron-Bridge/src/megatron/bridge/peft/lora_layers.py` 处报 `ModuleNotFoundError: No module named 'transformer_engine'`。
- [2026-04-24] 同一个 `megatron-bridge` Python `3.12` 环境当前拿到的 `torch` wheel 已不支持 V100 (`CC 7.0`)；`torch.cuda` 会明确警告该 wheel 仅支持 `sm75+ / sm80+ / sm86+ / sm90+ / sm100+ / sm120+`。这意味着即便补齐 `transformer_engine`，现有 wheel 也不应被视为 `DGX2-1` 上的稳定 V100 运行时。
- [2026-04-21] On `sd@v100x16-2`, successful real-checkpoint bring-up now depends on the local-workspace version of `megatron/tokenizer/tokenizer.py`, not only on the flat tokenizer files. Without that code sync, `HFTokenizer` still reports the smaller Qwen tokenizer vocab and stage-1 `LMHeadPipe` is built with `75904` rows instead of the checkpoint's `76032`.
- [2026-04-21] The V100 real-checkpoint smoke currently depends on per-dataset `index-cache` hashes being present on **both** DGX2 nodes. For the current `qwen_data_text_document` smoke, `DGX2-2` needed `e652788a584bd8acc28746e4a39bd45b_{doc,sample,shuffle}_idx.npy` synced from `DGX2-1`; otherwise startup fails during `GPTDataset` initialization with `FileNotFoundError`.
- [2026-04-20] Fresh `DGX2-1` validation confirms the live `/usr/bin/python3` runtime can still import `torch 2.9.1+cu128` and `deepspeed 0.18.3`, so current V100 conversion bring-up is not blocked by missing core Python packages.
- [2026-04-20] On `sd@v100x16-1`, `hf2megads` can rely on the already-synced core conversion files (`tools/hf2megads_weight_converter.py`, `megatron/arguments.py`, `megatron/training.py`, `pretrain_gpt.py`, `scripts/experiment_utils.sh`), but the newer helper scripts `scripts/activate_runtime_env.sh` and `scripts/setup_python_env.sh` are not present in the live tree yet. For clean-shell runs, either sync those scripts first or export equivalent cache/JIT env vars manually.
- [2026-04-10] On `sd@v100x16-{1,2}`, live communication benchmarking currently depends on the local-workspace version of `.context/torch_nccl_comm_bench.py`; the remote copy can lag behind and fail on newer flags such as `--warmup-iters` / `--iters`. Sync the local script to `/home/sd/Megatron-DeepSpeed/.context/` before short-form DGX2 benchmarks.
- [2026-04-10] The DGX2 `.local` training runtime currently needs a `psutil` sanity check before fresh bring-up if DeepSpeed starts failing during model build: on `DGX2-2`, `deepspeed.runtime.utils.see_memory_usage()` hit `AttributeError: module 'psutil' has no attribute 'virtual_memory'` even though `import psutil` succeeded. The immediate cause is a broken user-site `psutil` tree under `/home/sd/.local/lib/python3.10/site-packages/psutil/` that contains only `__pycache__/` and `tests/`, so Python imports it as an empty namespace package. Reconfirm `python3 -c "import psutil; print(psutil, getattr(psutil, '__file__', None), hasattr(psutil, 'virtual_memory'))"` in the active remote shell before formal reruns if this reproduces.
- [2026-03-26] Same-transport communication probing uses `/usr/bin/python3 -m torch.distributed.run` from the same `.local` user-site PyTorch environment as training; no external `nccl-tests` binary is required for this benchmark path.
- Remote experiments assume Python `3.10.12` from `/usr/bin/python3`.
- Remote Python packages are primarily installed under `~/.local/lib/python3.10/site-packages`.
- Do not assume `tp4bit` is relevant to this repo unless a future experiment explicitly opts into it.
- `sudo` may not inherit user `PYTHONPATH`, so explicit environment propagation may be required for Python-based admin helpers.
- [2026-04-05] On `user@sd-1` / `user@sd-2`, default `/usr/bin/python3` (`3.12.3`) does **not** provide `torch`; preflight must activate `tp4bit` explicitly before any Torch/DeepSpeed/NCCL checks.
- [2026-04-07] The current `sd-1` / `sd-2` login context should be treated as Ethernet-only. Use `NCCL_SOCKET_IFNAME=eth0` and `NCCL_IB_DISABLE=1` for benchmarking on these hosts.
- [2026-04-07] On `sd-1` / `sd-2`, successful short Megatron runs currently rely on `tp4bit` plus `TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_user` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- [2026-04-08] On `sd-1` / `sd-2`, the synced runtime path must include `megatron/gpu_freq_manager.py` together with `megatron/experiment_tracker.py`; otherwise `collect_nvml_device_snapshot` import fails before training startup.
- [2026-04-08] Manual `deepspeed` launches on `sd-1` / `sd-2` can now emit `run.json/events.jsonl`, but topology and hostfile snapshots remain empty unless `MEGATRON_HOSTFILE_JSON` and `MEGATRON_TOPOLOGY_JSON` are exported alongside the launch.
- [2026-04-20] Fresh-machine bring-up should now use `scripts/setup_python_env.sh` plus `scripts/activate_runtime_env.sh` rather than ad-hoc `pip install` notes. The second script is part of the dependency contract because `CPUAdam` and other JIT paths depend on writable low-latency cache directories (`TORCH_EXTENSIONS_DIR`, `TMPDIR`, `PYTHONPYCACHEPREFIX`, `TRITON_CACHE_DIR`).

## Upgrade Notes
- If DeepSpeed or Torch versions change, revalidate checkpoint, Zeus integration, and distributed exit behavior.
- If the remote environment changes, reconfirm NVML lock/reset behavior under `sudo -n`.

## Internal Dependencies
- `scripts/run_experiment.sh`
- `scripts/experiment_utils.sh`
- `megatron/power_monitor.py`
- `megatron/training.py`
- `megatron/experiment_tracker.py`
- `megatron/gpu_freq_manager.py`
