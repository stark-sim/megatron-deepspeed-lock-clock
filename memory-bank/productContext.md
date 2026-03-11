# Product Context

## Problem Statement
NVIDIA default boost and thermal throttling behavior is not necessarily optimal for long, steady LLM training workloads. The project exists to test whether static GPU clocks produce better performance stability and better energy efficiency.

## User Goals
- Run comparable Megatron-DeepSpeed training jobs on remote GPU machines.
- Measure not only speed and throughput, but also power and energy.
- Find a non-maximum fixed frequency that beats both baseline and max-lock on energy efficiency.

## Success Criteria
- Each sweep point produces consistent topology, training-step, and power metrics.
- Results are easy to compare with minimal manual cleanup.
- Short validation runs can be launched safely without filling disks with checkpoints.

## User Preferences
- Real experiment execution should default to the remote host first, not local-only runs.
- Current focus is V100 static-frequency energy efficiency, not quantization.
- Both `samples/Wh` and `tokens/J` must be preserved in experiment results.
- `screen` should be used for long-running remote jobs, and session IDs should be recorded.

## Workflow Summary
- Prepare or patch launcher logic locally.
- Sync critical launcher changes to `/home/sd/Megatron-DeepSpeed` on the remote host.
- Launch isolated `screen` sessions with explicit environment variables.
- Collect results from `run.json`, `events.jsonl`, and experiment logs.
- Update sweep conclusions and next candidate frequencies.
