# Active Context

## Current Focus
[2026-03-11] Find the V100 fixed-frequency sweet spot for Megatron-DeepSpeed short training runs using Zeus power metrics, while avoiding unnecessary checkpoint writes.

## Recent Changes
- [2026-03-11] Added short-run-friendly warmup and eval controls in `scripts/run_experiment.sh`.
- [2026-03-11] Updated NVML lock/reset helpers in `scripts/experiment_utils.sh` to use elevated remote execution.
- [2026-03-11] Extended Zeus integration in `megatron/power_monitor.py` and `megatron/training.py` to preserve `samples/Wh` and `tokens/J`, including leftover/final interval capture.
- [2026-03-11] Added `DISABLE_CHECKPOINT=1` support to `scripts/run_experiment.sh` and validated it remotely.
- [2026-03-11] Switched remote experiment launches to isolated `screen` sessions and recorded session IDs in `.context/remote-screen-sessions.tsv`.

## Active Decisions
- [2026-03-11] V100 work is focused on static frequency energy efficiency, not quantization.
- [2026-03-11] Real runs should happen on `sd@v100x16-1` first.
- [2026-03-11] Energy-only sweeps should avoid checkpoint writes unless explicitly requested.

## Next Steps
- Continue the V100 short-run sweep with remaining candidate frequencies such as `1500 MHz`.
- Decide whether the final recommendation should prefer absolute energy efficiency (`1200 MHz`) or a performance/efficiency balance (`1305 MHz`).
- Optionally improve tracker finalize behavior when runs exit without checkpointing or when finalization is partially incomplete.

## Important Patterns & Preferences
- Use remote `screen` for runs that take more than a quick validation.
- Record `screen` session IDs and attachment commands immediately after launch.
- Preserve both `samples/Wh` and `tokens/J` in experiment summaries.
- Keep experiment conditions identical except for the frequency policy being tested.

## Current Blockers
- `run.json` finalization is still less reliable than log/events parsing for some runs.
- Sweep coverage is still incomplete without a `1500 MHz` data point.
