# System Patterns

## Experiment Execution Pattern
- Launcher entry point: `scripts/run_experiment.sh`.
- Shell helpers: `scripts/experiment_utils.sh`.
- Training/power integration: `megatron/training.py` and `megatron/power_monitor.py`.
- Experiment metadata is stored in `experiments/<run_id>/` as JSON, JSONL, notes, command snapshots, and logs.

## Remote-First Pattern
- Real GPU experiments are run on `sd@v100x16-1`.
- Remote repo path is `/home/sd/Megatron-DeepSpeed`.
- Remote runs should be launched inside isolated `screen` sessions to survive SSH disconnects.
- Session IDs and attachment commands are recorded in `.context/remote-screen-sessions.tsv`.

## Frequency-Lock Pattern
- `EXPERIMENT_MODE=baseline` keeps default GPU behavior.
- `EXPERIMENT_MODE=static` with `STATIC_CLOCK_MHZ=<mhz>` uses NVML clock locking.
- Clock locking/reset uses `sudo -n python3` on the remote host because NVML lock/reset requires elevated permission.

## Energy-Only Sweep Pattern
- Short sweep runs use a fixed topology and training length to isolate clock effects.
- `DISABLE_CHECKPOINT=1` is preferred for sweep jobs because checkpoint artifacts are large and unnecessary for energy analysis.
- Zeus power monitoring is used to generate energy, average power, `samples/Wh`, and `tokens/J`.

## Comparison Pattern
- Keep model, tokenizer, data path, TP/PP/DP, ZeRO stage, and train steps constant across variants.
- V100 short-run sweep baseline uses `TP=4`, `PP=1`, effective `DP=4`, `ZeRO-1`, `TRAIN_STEPS=20`.
- Use log-derived steady-state iteration time plus Zeus metrics for direct comparison.

## Artifact Pattern
- `run.json` is the canonical run summary when finalize succeeds.
- `events.jsonl` is a reliable backup source for interval metrics, especially when final checkpointing or finalize logic is incomplete.
- The training log remains the fallback truth source for final `[Zeus] Steps 1-20 ...` summaries.
