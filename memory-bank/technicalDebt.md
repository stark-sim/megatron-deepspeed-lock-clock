# Technical Debt

## Known Debt Items
| Item | Severity | Effort | Impact | Added |
|------|----------|--------|--------|-------|
| `run.json` finalize is inconsistent for some completed runs | Medium | Medium | Requires log/event fallback during analysis | 2026-03-11 |
| Distributed teardown emits noisy TCPStore/NCCL warnings after successful completion | Low | Medium | Makes logs noisier and can obscure true failures | 2026-03-11 |
| Remote launcher changes must be synced manually from local workspace | Medium | Low | Easy to forget and causes validation drift | 2026-03-11 |
| Experiment comparison still relies on ad-hoc parsing commands | Medium | Medium | Slower analysis across many sweep points | 2026-03-11 |
| Offline freq model still ignores `num_attention_heads` / `num_key_value_heads` / `swiglu` in analytic feature formulas | Medium | Medium | Guardrails now reject mixed workloads, but true cross-shape transfer still lacks explicit modeling | 2026-03-15 |

## Refactoring Priorities
1. Add a small comparison/report script for multiple `run_id`s.
2. Improve tracker finalization robustness for no-checkpoint runs.
3. Consider reducing teardown warning noise if it masks genuine failures.

## Code Smells
- Important experiment truths are split across `run.json`, `events.jsonl`, and logs.
- Remote and local script versions can temporarily diverge during active iteration.

## Cleanup Plans
- Continue deleting obsolete checkpoint directories after verifying power-only results are preserved in logs/artifacts.
- Keep using `DISABLE_CHECKPOINT=1` for future sweep runs unless checkpoint recovery is explicitly needed.

## Dependency Debt
- None urgent beyond keeping Torch/DeepSpeed/Zeus integration validated after environment changes.

## Architecture Debt
- Experiment execution and experiment analysis are still coupled through manual shell snippets rather than a dedicated summary tool.
