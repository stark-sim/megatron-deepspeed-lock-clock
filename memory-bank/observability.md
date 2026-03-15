# Observability

## Logging
- Primary execution logs are written under `experiments/<run_id>/logs/<run_id>.log`.
- Detached remote `screen` launch logs are also captured under `experiments/_screen_boot/<session>.log`.
- `run.json` stores finalized metadata when tracker finalization succeeds.
- `events.jsonl` stores interval/checkpoint/status events and is a key fallback for incomplete finalization cases.

## Monitoring
- Zeus is the primary power/energy monitor for current V100 experiments.
- Required comparison metrics:
  - total energy to finish the fixed workload
  - total wall-clock time to finish the fixed workload
  - `energy_wh`
  - `avg_power_w`
  - `samples/Wh`
  - `tokens/J`
  - iteration time / throughput from training logs
- Recommendation review should consider the full-task energy/time tradeoff first; short-horizon power and per-step latency are supporting signals, not the final objective.
- When the run is steady from start to finish, per-step power and step time can be used as anchored observational proxies for total energy and total wall-clock comparisons across candidate frequencies.

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
- `TP=8, DP=2` validation failed immediately in remote boot log `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp8dp2_baseline_20260312_1930.log` with `AssertionError: 28 is not divisible by 8`; treat this as a topology feasibility failure rather than a runtime instability.
- The fallback `TP=2, PP=4, DP=2` baseline is running in remote `screen` session `power_tp2pp4dp2_baseline_20260312_1944`; the boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp2pp4dp2_baseline_20260312_1944.log` and the attach command is recorded in `.context/remote-screen-sessions.tsv`.
- The fallback `TP=2, PP=4, DP=2` baseline completed with local synced artifact `.context/real-v100-tp2pp4dp2-baseline-20260312/v100_tp2pp4dp2_baseline_20260312_194435_DGX2-1/` and a Zeus summary of `419.508 s`, `1515838.132 J`, `3613.370 W`, `0.432342 tokens/J`.
- Zero-shot transfer prediction artifacts for the new topology are staged under `.context/real-v100-tp2pp4dp2-baseline-20260312/zero-shot-transfer-v1/`.
- Partial measured-vs-predicted comparison for the first three completed `TP=2, PP=4, DP=2` static points is staged at `.context/real-v100-tp2pp4dp2-static-20260312/measured_vs_predicted_partial.tsv`, with a short narrative summary in `.context/real-v100-tp2pp4dp2-static-20260312/partial_analysis.md`.
- The completed measured-vs-predicted comparison for `TP=2, PP=4, DP=2` is staged at `.context/real-v100-tp2pp4dp2-static-20260312/measured_vs_predicted.tsv`, with the final narrative summary at `.context/real-v100-tp2pp4dp2-static-20260312/final_analysis.md`.
- The detached static validation sweep session `power_tp2pp4dp2_static_validate_20260312_2000` has already produced all targeted Zeus summaries in `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp2pp4dp2_static_validate_20260312_2000.log`.
- Refactored hardware-first prediction outputs are staged under `.context/real-v100-power-static-20260312-merged/predictions-hw-first-v3/`; this local smoke run uses the fresh unfixed baseline rerun and currently recommends `1245/1252/1260 MHz`.
- `pytest` cannot currently be used as the first-line validator in this workspace because `tests/conftest.py` imports `deepspeed`; use `python -m py_compile` plus direct module/CLI smoke runs unless the runtime environment is provisioned with DeepSpeed.
- `scripts/predict_freq_sweet_spot.py` can be smoke-tested locally with synthetic artifacts under `.context/freq-model-smoke/` before using real sweep outputs.
- Real prediction outputs are currently staged under `.context/real-v100-power-static-20260312/predictions/` for review before integrating with remote sweep workflows.
- Guardrail sweep outputs for the merged 8-point V100 dataset are staged under `.context/real-v100-power-static-20260312-merged/guardrail-sweep/`, with a measured-vs-modeled summary table at `.context/real-v100-power-static-20260312-merged/guardrail_comparison.tsv`.
- Recalibrated overlay-based guardrail outputs are staged under `.context/real-v100-power-static-20260312-merged/guardrail-sweep-overlay/`, with the updated summary table at `.context/real-v100-power-static-20260312-merged/guardrail_comparison_overlay.tsv`.
- Baseline-relative tradeoff output versus `power_baseline_20260311_20b` is staged at `.context/real-v100-power-static-20260312-merged/baseline_tradeoff_vs_power_baseline_20b.tsv`.
- The fresh baseline rerun is logging via remote `screen` boot log `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_baseline_rerun_20260312_1512.log`, and the local attach record is in `.context/remote-screen-sessions.tsv`.
- The refreshed baseline-relative table against the completed rerun is staged at `.context/real-v100-power-static-20260312-merged/baseline_tradeoff_vs_power_baseline_rerun_20260312_151032.tsv`.
- Pareto skyline chart artifacts are staged under `.context/real-v100-power-static-20260312-merged/pareto-charts/`.
- Serial validation launch log for the low-frequency confirmation sweep is stored at `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_lowfreq_20260312_113029b.log`, and the local session record is `.context/remote-screen-sessions.tsv`.
- Use `VALIDATE_ONLY=1` before launch for topology and preflight validation.
- Verify `screen -ls` and `tail` on the boot log after launching detached runs.
- Confirm final Zeus summary appears in the training log for short runs.

## Error Tracking
- [2026-03-12] The first baseline-relative hardware-first refactor compiled but still produced unrealistic pure-analytic recommendations (first `727 MHz`, then `1597 MHz`) on the merged V100 dataset; adding throughput-scale centering plus a throughput-saturation prior corrected the balanced recommendation to about `1252 MHz` while keeping low runtime/energy-ratio error.
- [2026-03-12] Synthetic smoke validation showed low power-fit error but high throughput-fit error, so calibration quality should be checked on real sweep artifacts before trusting the recommended sweet spot.
- [2026-03-12] The first real V100 prediction achieved `throughput_mape≈10.79%` and `power_mape≈2.37%`, but still extrapolated the optimum below the sampled static range.
- [2026-03-12] Real follow-up runs at `1020/1027/1035 MHz` all exceeded the old `1200 MHz` point in `tokens/J`, validating the model's low-frequency search direction even though absolute predicted scales remain mismatched.
- [2026-03-12] Pure analytic calibration still mis-ranked guardrail-optimal frequencies on the merged 8-point dataset, but the observed-frequency interpolation overlay now restores the correct transition pattern across the main guardrail thresholds.
- No dedicated Sentry/Bugsnag integration.
- Important failure modes are currently tracked through logs, `events.jsonl`, and memory-bank notes.
- [2026-03-13] Remote `TP=2, PP=4, DP=2` refinement sweep session: `power_tp2pp4dp2_static_refine_20260313_1238c`. Completed Zeus summaries: `1185 MHz -> 449.706 s / 1089279.7 J / 2422.2 W / 0.601645 tokens/J`, `1192 MHz -> 449.1 s / 1099588.2 J / 2448.6 W / 0.596`, `1207 MHz -> 444.1 s / 1106275.4 J / 2490.9 W / 0.592`, `1215 MHz -> 440.6 s / 1111404.2 J / 2522.3 W / 0.590`.
- [2026-03-13] Full merged Pareto-first accuracy artifacts for `TP=2, PP=4, DP=2` live under `.context/real-v100-tp2pp4dp2-static-merged-20260313/predictions-pareto-first-full-20260313/`; use `prediction_report.md` for human review and `prediction.json` for per-frequency ratio error extraction.
- [2026-03-15] Updated PP-bubble rerun artifacts live under `.context/real-v100-tp2pp4dp2-rerun-20260315/`; the fresh analytic rerun reports `supported_sweet_spot=1020 MHz`, `runtime_ratio_mape≈1.7303%`, `energy_ratio_mape≈1.2042%`, and sampled-frequency Pareto overlap on `{1185, 1200, 1207, 1215, 1245, 1252, 1260}` with extra predicted frontier points at `1192` and `930 MHz`.
- [2026-03-15] Two-band corrected rerun artifacts live under `.context/real-v100-tp2pp4dp2-rerun-20260315-corrected/`; the corrected curve reports `supported_sweet_spot=1125 MHz`, `total_time_mape≈1.1971%`, `total_energy_mape≈0.9045%`, and sharply improves low-frequency sampled-point fit (`900/915/930 MHz`) while preserving the observed Pareto overlap on sampled frontier points except for one extra predicted `1192 MHz` point.

- [2026-03-15] Remote 50-step validation session: `power_tp2pp4dp2_pred50_20260315_134758` on `sd@v100x16-1`; launcher script is `.context/remote-launch-tp2pp4dp2-predicted50-20260315.sh`, boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp2pp4dp2_pred50_20260315_134758.log`.
- [2026-03-15] Final Zeus summaries for the corrected-band validation points: `1117 MHz -> 1187.3 s / 2649800.5 J / 2231.8 W / 0.618 tokens/J` from `/home/sd/Megatron-DeepSpeed/experiments/v100_tp2pp4dp2_pred50_1117_20260315_134807_DGX2-1/logs/v100_tp2pp4dp2_pred50_1117_20260315_134807_DGX2-1.log`, and `1125 MHz -> 1178.7 s / 2662089.4 J / 2258.5 W` from `/home/sd/Megatron-DeepSpeed/experiments/v100_tp2pp4dp2_pred50_1125_20260315_140919_DGX2-1/logs/v100_tp2pp4dp2_pred50_1125_20260315_140919_DGX2-1.log`.
- [2026-03-15] For quick curve-quality checks on a running job, a stable fallback is to combine mean recent iteration time with a 1-minute `nvidia-smi --query-gpu=power.draw` sample; on the in-flight `1125 MHz` run this produced `1175.7 s / 2274.6 W` equivalent estimates, already close to the final Zeus `1178.7 s / 2258.5 W` summary.

- [2026-03-15] Final Zeus summary for the third corrected-band validation point: `1132 MHz -> 1172.9 s / 2674848.9 J / 2280.5 W / 0.613 tokens/J` from `/home/sd/Megatron-DeepSpeed/experiments/v100_tp2pp4dp2_pred50_1132_20260315_143024_DGX2-1/logs/v100_tp2pp4dp2_pred50_1132_20260315_143024_DGX2-1.log`.
