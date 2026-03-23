# Dependencies

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

## Dev Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| `bash` | system | Launcher scripting |
| `screen` | `4.09.00` | Detached remote experiment sessions |
| `tp4bit` Conda env | Python `3.12` + `torch 2.9.1+cu128` + `deepspeed 0.18.5` | Current dual-node DGX2 bring-up environment on `sd@v100x16-{1,2}` |
| `Qwen2.5 tokenizer flat copy` | snapshot-derived local files | Use `/home/sd/Megatron-DeepSpeed/.context/qwen25_tokenizer_flat` on both DGX2 nodes to avoid Hugging Face snapshot symlink/blob breakage on `sd@v100x16-2` |
| `nvidia-smi` | system | GPU inspection and validation |

## Version Constraints
- Remote experiments assume Python `3.10.12` from `/usr/bin/python3`.
- Remote Python packages are primarily installed under `~/.local/lib/python3.10/site-packages`.
- `sudo` may not inherit user `PYTHONPATH`, so explicit environment propagation may be required for Python-based admin helpers.

## Upgrade Notes
- If DeepSpeed or Torch versions change, revalidate checkpoint, Zeus integration, and distributed exit behavior.
- If the remote environment changes, reconfirm NVML lock/reset behavior under `sudo -n`.

## Internal Dependencies
- `scripts/run_experiment.sh`
- `scripts/experiment_utils.sh`
- `megatron/power_monitor.py`
- `megatron/training.py`
- `megatron/experiment_tracker.py`
