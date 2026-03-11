# Observability

## Logging
- Primary execution logs are written under `experiments/<run_id>/logs/<run_id>.log`.
- Detached remote `screen` launch logs are also captured under `experiments/_screen_boot/<session>.log`.
- `run.json` stores finalized metadata when tracker finalization succeeds.
- `events.jsonl` stores interval/checkpoint/status events and is a key fallback for incomplete finalization cases.

## Monitoring
- Zeus is the primary power/energy monitor for current V100 experiments.
- Required comparison metrics:
  - `energy_wh`
  - `avg_power_w`
  - `samples/Wh`
  - `tokens/J`
  - iteration time / throughput from training logs

## Tracing
- No dedicated distributed tracing system is currently used.
- Timing is inferred from Megatron/DeepSpeed logs and Zeus summaries.

## Alerting
| Alert | Condition | Severity | Channel |
|-------|-----------|----------|---------|
| Disk nearly full | Remote filesystem approaches 100% usage | High | Manual CLI inspection |
| Finalize mismatch | `run.json` incomplete but training log shows completion | Medium | Manual result validation |
| Static clock not reset | GPUs remain locked after run exit | High | Manual `nvidia-smi`/NVML reset |

## Health Checks
- Use `VALIDATE_ONLY=1` before launch for topology and preflight validation.
- Verify `screen -ls` and `tail` on the boot log after launching detached runs.
- Confirm final Zeus summary appears in the training log for short runs.

## Error Tracking
- No dedicated Sentry/Bugsnag integration.
- Important failure modes are currently tracked through logs, `events.jsonl`, and memory-bank notes.
