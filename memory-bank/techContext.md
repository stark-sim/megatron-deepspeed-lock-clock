# Tech Context

## Repository
- Local workspace: `/Users/stark_sim/conductor/workspaces/megatron-deepspeed-lock-clock/dar-es-salaam`
- Remote workspace: `/home/sd/Megatron-DeepSpeed`
- Current local branch: `stark-sim/init-workspace`
- Remote git branch observed during runs: `main`

## Core Stack
- Python-based Megatron-DeepSpeed training stack.
- DeepSpeed launcher for distributed training.
- NVML-based GPU frequency control.
- Zeus-based power monitoring integrated into training.

## Remote Runtime Context
- Host: `DGX2-1`
- GPUs: `16 x Tesla V100-SXM3-32GB`
- Remote Python: `/usr/bin/python3` (`3.10.12`)
- Remote package installs primarily live under `~/.local/lib/python3.10/site-packages`

## Data / Tokenizer Context
- Dataset prefix: `/home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document`
- Tokenizer snapshot: `/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9`

## Current Runtime Conventions
- `LR_WARMUP_ITERS` is made short-run friendly.
- `EVAL_INTERVAL` and `EVAL_ITERS` can be disabled safely for short runs.
- `DISABLE_CHECKPOINT=1` is the preferred mode for energy-only sweep jobs.
- `screen` sessions should use `env -i` with explicit `PATH` and `PYTHONPATH` to avoid environment leakage.
