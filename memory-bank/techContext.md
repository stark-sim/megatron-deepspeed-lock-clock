# Tech Context

## Repository
- Local workspace: `/Users/stark_sim/conductor/workspaces/megatron-deepspeed-lock-clock/chicago`
- Remote workspace: `/home/sd/Megatron-DeepSpeed`
- Current local branch: `stark-sim/review-progress`
- Remote git branch observed during runs: `main`

## Core Stack
- Python-based Megatron-DeepSpeed training stack.
- DeepSpeed launcher for distributed training.
- NVML-based GPU frequency control.
- Zeus-based power monitoring integrated into training.
- Offline Python analysis layer for frequency sweet-spot prediction implemented in `analysis/freq_model/` and `scripts/predict_freq_sweet_spot.py`.

## Remote Runtime Context
- Primary V100 transfer-validation host: `sd@v100x16-1` (`DGX2-1`, `16 x Tesla V100-SXM3-32GB`, NvLink, Python `3.10.12`).
- Secondary host checked on [2026-03-16]: `user@sd-1` (`4 x RTX 4080 SUPER 16GB`, Python `3.12.3`, repo path `/home/user/Megatron-DeepSpeed`). Keep it out of the active predictor-validation loop until the model is stable or we intentionally define a 4-GPU validation set.
## Data / Tokenizer Context
- Dataset prefix: `/home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document`
- Tokenizer snapshot: `/home/sd/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9`

## Current Runtime Conventions
- `LR_WARMUP_ITERS` is made short-run friendly.
- `EVAL_INTERVAL` and `EVAL_ITERS` can be disabled safely for short runs.
- `DISABLE_CHECKPOINT=1` is the preferred mode for energy-only sweep jobs.
- `screen` sessions should use `env -i` with explicit `PATH` and `PYTHONPATH` to avoid environment leakage.
- Local offline freq-model unit tests currently need `python -m pytest --noconftest ...` because repo-level `tests/conftest.py` imports `deepspeed`, which is not available in this workspace environment.

- Remote repo note: `/home/sd/Megatron-DeepSpeed/run_experiment.sh` is stale relative to `/home/sd/Megatron-DeepSpeed/scripts/run_experiment.sh`; use the `scripts/` launcher for current short-run energy workflows.
