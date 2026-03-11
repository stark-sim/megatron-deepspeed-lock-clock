# Progress

## Completed Features
- [x] Unified experiment launcher and experiment tracking workflow.
- [x] Remote NVML static clock lock/reset support for V100 runs.
- [x] Zeus power metrics integrated into experiment outputs.
- [x] Short-run-compatible warmup and eval configuration handling.
- [x] `screen`-based remote launch workflow with recorded session IDs.
- [x] `DISABLE_CHECKPOINT=1` mode for energy-only sweep runs.
- [x] Remote cleanup of large, obsolete checkpoint directories.

## In Progress
- [ ] Complete the V100 static-frequency sweep with additional candidate points.
- [ ] Decide how to present the sweet-spot conclusion: max efficiency vs best efficiency/performance balance.
- [ ] Optionally harden experiment finalization/reporting after no-checkpoint runs.

## Known Issues
- Some completed runs still rely on log/event parsing because `run.json` finalize did not fully complete.
- Distributed teardown can emit TCPStore/NCCL warnings even when training completed successfully.
- Final checkpoint writing was previously the dominant disk-space failure mode for sweep jobs.

## Milestones

| Milestone | Status | Target Date |
|-----------|--------|-------------|
| Remote short-run power workflow validated | Done | 2026-03-11 |
| Initial V100 sweep baseline/static comparison | Done | 2026-03-11 |
| No-checkpoint sweep mode validated | Done | 2026-03-11 |
| Remaining V100 sweep points collected | In Progress | 2026-03-12 |
| Sweet-spot recommendation documented | Pending | 2026-03-12 |
