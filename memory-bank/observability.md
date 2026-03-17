# Observability

## 2026-03-17 Local/Remote Replay Check
- Local focused validation command: `pytest -q --noconftest tests/unit_tests/test_freq_model.py` -> `19 passed in 60.96s`.
- Local replay artifacts: `.context/transfer-debug-20260317/local-tp1pp4dp4-topodist/prediction.json` and `.context/transfer-debug-20260317/local-tp1pp2dp8-topodist/prediction.json`.
- Replay summary on measured diagnostic points: `TP=1, PP=4, DP=4` stays at about `1.25%` runtime MAPE / `1.05%` power MAPE / `0.34%` energy MAPE; `TP=1, PP=2, DP=8` stays at about `2.26%` runtime MAPE / `1.66%` power MAPE / `0.65%` energy MAPE.
- Remote regenerated artifacts using `python3 .context/regen_transfer_bundle.py`: `/home/sd/Megatron-DeepSpeed/.context/canonical-transfer-tp1pp4dp4-20260317-topodist/prediction.json` and `/home/sd/Megatron-DeepSpeed/.context/canonical-transfer-tp1pp2dp8-20260317-topodist/prediction.json`. Both now recommend `[1185, 1192, 1200]` with default `1185 MHz`, matching local replay exactly.


- [2026-03-17] Local-only gap-regularization experiment artifacts live under `.context/transfer-debug-20260317/local-tp1pp4dp4-gapfix/` and `.context/transfer-debug-20260317/local-tp1pp2dp8-gapfix/`. It preserves measured-band fit (`TP=1, PP=4, DP=4` still about `1.25%` time / `1.05%` power MAPE) but is not promoted because the regenerated `TP=1, PP=4, DP=4` default still falls to `930 MHz`.
- [2026-03-17] After adding source-observed-band tail regularization, local `TP=1, PP=2, DP=8` transfer regeneration under `.context/transfer-debug-20260317/local-tp1pp2dp8-tailfix/` now recommends `[900, 907, 915]` instead of `[757, 765, 772]` while leaving the measured `1177/1185/1192 MHz` diagnostics effectively unchanged.
- [2026-03-17] Remote regeneration confirms the same result for `TP=1, PP=2, DP=8`: `/home/sd/Megatron-DeepSpeed/.context/canonical-transfer-tp1pp2dp8-20260317-tailfix/prediction.json` now defaults to `900 MHz` with calibration reference band ratios `~0.564-0.789`.
- [2026-03-17] Remote regeneration for `TP=1, PP=4, DP=4` still defaults to `997 MHz` even with the same tail regularization (`/home/sd/Megatron-DeepSpeed/.context/canonical-transfer-tp1pp4dp4-20260317-tailfix/prediction.json`), which isolates a broader low-frequency curve-shape issue that lives inside the source-observed band.
- [2026-03-17] Remote transfer artifact regenerated after syncing the new `analysis/freq_model/model.py`: `/home/sd/Megatron-DeepSpeed/.context/canonical-transfer-tp1pp2dp8-20260317-anchorfix/prediction.json`.
- [2026-03-17] Reanalysis against the existing `TP=1, PP=2, DP=8` measured trio is stored locally under `.context/real-v100-tp1pp2dp8-reanalysis-20260317-anchorfix/`; diagnostic-band MAPE is now `total_time≈2.07%`, `avg_power_w≈1.68%`, `total_energy≈0.44%`.
- [2026-03-17] Important caveat: the regenerated energy-first default for `TP=1, PP=2, DP=8` is still `757 MHz` with neighborhood `[757, 765, 772]`, so do not interpret this model change as full low-frequency-tail validation.
- [2026-03-17] Copied the minimal transfer-debug bundles from `sd@v100x16-1` into `.context/transfer-debug-20260316/` so the current local predictor can be replayed against the saved source/target `prediction.json` files without touching the remote working tree.
- [2026-03-17] Local offline replay with the new low-TP transfer-anchor adjustment yields: `TP=1, PP=4, DP=4` -> `total_time MAPE≈1.25%`, `avg_power_w MAPE≈1.05%`, `total_energy MAPE≈0.34%`; `TP=1, PP=2, DP=8` -> `total_time MAPE≈2.07%`, `avg_power_w MAPE≈1.68%`, `total_energy MAPE≈0.44%`.
- [2026-03-17] Focused local unit coverage for the new low-TP response runs with `pytest --noconftest tests/unit_tests/test_freq_model.py -k ...`; repo-wide pytest still requires `deepspeed` through `tests/conftest.py`.
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
- [2026-03-17] The detached `TP=1, PP=2, DP=8` static validation trio completed successfully. Zeus summaries: `1177 MHz -> 1130.8 s / 2725506.6 J / 2410.3 W / 0.601 tokens/J`, `1185 MHz -> 1119.4 s / 2729647.6 J / 2438.4 W / 0.600 tokens/J`, `1192 MHz -> 1118.4 s / 2754717.5 J / 2463.0 W / 0.595 tokens/J`.
- [2026-03-17] Local comparison artifacts for the completed `TP=1, PP=2, DP=8` trio live under `.context/real-v100-tp1pp2dp8-static-20260316/`; use `measured_vs_predicted.tsv` for curve errors and `final_analysis.md` for the diagnostic conclusion.
- [2026-03-17] Curve-accuracy summary for the completed `TP=1, PP=2, DP=8` trio: `total_time` MAPE `≈11.73%`, `avg_power_w` MAPE `≈5.74%`, and `total_energy` MAPE `≈5.33%`. This is worse than the `TP=1, PP=4, DP=4` diagnostic, so the current transfer issue appears broader than pipeline depth alone.
- [2026-03-16] The first `TP=1, PP=2, DP=8` unfixed baseline attempt failed with CUDA OOM at step 5 on later pipeline-stage ranks. Retrying with allocator expansion (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) resolved the issue and produced final Zeus summary `1137.3 s / 3935271.5 J / 3460.1 W / 0.416 tokens/J`.
- [2026-03-16] Canonical zero-shot transfer artifacts for `TP=1, PP=2, DP=8` now live under `.context/canonical-transfer-tp1pp2dp8-20260316/`; default recommendation is `1177 MHz` with neighborhood `[1177, 1185, 1192]`.
- [2026-03-16] Remote `TP=1, PP=2, DP=8` static validation sequence completed in detached session `tp1pp2dp8_static_seq_20260316`; boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/tp1pp2dp8_static_seq_20260316.log`, and the completed frequencies were `1177/1185/1192 MHz`.
- [2026-03-16] Curve-accuracy summary for the completed `TP=1, PP=4, DP=4` trio: `total_time` MAPE is about `7.17%`, `avg_power_w` MAPE about `5.30%`, and `total_energy` MAPE about `1.48%`. Interpret the low energy error carefully: it is likely dominated by cancellation between time and power biases rather than by a truly correct curve.
- [2026-03-16] The detached `TP=1, PP=4, DP=4` static validation trio completed successfully. Zeus summaries: `1252 MHz -> 1300.1 s / 3078625.8 J / 2367.9 W / 0.532 tokens/J`, `1260 MHz -> 1287.9 s / 3067834.2 J / 2382.0 W / 0.534 tokens/J`, `1267 MHz -> 1286.3 s / 3097007.8 J / 2407.7 W / 0.529 tokens/J`.
- [2026-03-16] Local comparison artifacts for the completed `TP=1, PP=4, DP=4` trio live under `.context/real-v100-tp1pp4dp4-static-20260316/`; use `measured_vs_predicted.tsv` for ratio errors and `final_analysis.md` for the transfer takeaway.
- [2026-03-16] The `TP=1, PP=4, DP=4` 50-step unfixed baseline completed on `sd@v100x16-1` with Zeus summary `1316.7 s / 4174627.4 J / 3170.6 W / 0.392 tokens/J` from `/home/sd/Megatron-DeepSpeed/experiments/v100_tp1pp4dp4_baseline50_20260316_185249_DGX2-1/logs/v100_tp1pp4dp4_baseline50_20260316_185249_DGX2-1.log`.
- [2026-03-16] Canonical zero-shot transfer artifacts for `TP=1, PP=4, DP=4` now live under `.context/canonical-transfer-tp1pp4dp4-20260316/`; default recommendation is `1252 MHz` with neighborhood `[1252, 1260, 1267]` and predicted baseline-relative energy ratio near `0.749x`.
- [2026-03-16] Remote `TP=1, PP=4, DP=4` static validation sequence completed in detached session `tp1pp4dp4_static_seq_20260316`; boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/tp1pp4dp4_static_seq_20260316.log`, and the completed frequencies were `1252/1260/1267 MHz`.
- [2026-03-16] Canonical predictor-quality checkpoint on `sd@v100x16-1`: outputs live under `.context/canonical-transfer-tp2pp2dp4-20260316/` and `.context/canonical-transfer-tp2pp1dp8-20260316/`. Current canonical defaults are `1072 MHz` for `TP=2, PP=2, DP=4` and `1005 MHz` for `TP=2, PP=1, DP=8`.
- [2026-03-16] Transfer-time rescaling recheck on `sd@v100x16-1` completed successfully with the new target-topology low-frequency amplitude rule. Outputs live under `.context/remote-transfer-rescaled-tp2pp2dp4-20260316-r2/` and `.context/remote-transfer-rescaled-tp2pp1dp8-20260316-r2/`; defaults are now `1072 MHz` for `TP=2, PP=2, DP=4` and `1005 MHz` for `TP=2, PP=1, DP=8`.
- [2026-03-16] Second V100 recheck after restoring the low-frequency intensity baseline completed successfully. Outputs live under `.context/remote-recheck-tp2pp2dp4-20260316-v100-r2/` and `.context/remote-recheck-tp2pp1dp8-20260316-v100-r2/`; defaults improved to `1057 MHz` for `TP=2, PP=2, DP=4` and `960 MHz` for `TP=2, PP=1, DP=8`.
- [2026-03-16] Remote offline transfer recheck on `sd@v100x16-1` completed after syncing the updated `analysis/freq_model/` package plus baseline artifacts. Outputs live under `.context/remote-recheck-tp2pp2dp4-20260316-v100/` and `.context/remote-recheck-tp2pp1dp8-20260316-v100/`; the new defaults are `1035 MHz` for `TP=2, PP=2, DP=4` and `907 MHz` for `TP=2, PP=1, DP=8`.
- [2026-03-15] The detached `TP=2, PP=1, DP=8` static validation session `power_tp2pp1dp8_pred50_20260315_211016` completed successfully; synced artifacts live under `.context/real-v100-tp2pp1dp8-static-20260315/`, with measured-vs-predicted summary at `.context/real-v100-tp2pp1dp8-static-20260315/measured_vs_predicted.tsv` and narrative summary at `.context/real-v100-tp2pp1dp8-static-20260315/final_analysis.md`.
- [2026-03-15] Zeus summaries for `TP=2, PP=1, DP=8` 50-step validation: `967 MHz -> 912.1 s / 2298437.0 J / 2520.0 W`, `975 MHz -> 904.9 s / 2295007.9 J / 2536.1 W`, `982 MHz -> 898.0 s / 2290348.0 J / 2550.4 W`, `1050 MHz -> 841.8 s / 2265331.2 J / 2691.1 W`.
- [2026-03-15] The `TP=2, PP=1, DP=8` 50-step unfixed baseline completed with local synced artifact `.context/real-v100-tp2pp1dp8-baseline-20260315/v100_tp2pp1dp8_baseline50_20260315_203404_DGX2-1/` and Zeus summary `851.9 s`, `3549352.3 J`, `4166.4 W`, `0.462 tokens/J`.
- [2026-03-15] Transferred prediction artifacts for `TP=2, PP=1, DP=8` are staged under `.context/real-v100-tp2pp1dp8-transfer-20260315-with-baseline/`; default recommendation is `975 MHz`, balanced comparator is baseline-like (`1597 MHz`), and the recommended neighborhood is `[967, 975, 982]`.
- [2026-03-15] Remote `TP=2, PP=1, DP=8` static validation session: `power_tp2pp1dp8_pred50_20260315_211016`; launcher script is `.context/remote-launch-tp2pp1dp8-predicted50-20260315.sh`, boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp2pp1dp8_pred50_20260315_211016.log`, and frequencies are `967/975/982/1050 MHz`.
- [2026-03-15] The detached `TP=2, PP=2, DP=4` static validation session `power_tp2pp2dp4_pred50_20260315_161856` completed successfully; synced artifacts live under `.context/real-v100-tp2pp2dp4-static-20260315/`, with measured-vs-predicted summary at `.context/real-v100-tp2pp2dp4-static-20260315/measured_vs_predicted.tsv` and narrative summary at `.context/real-v100-tp2pp2dp4-static-20260315/final_analysis.md`.
- [2026-03-15] Zeus summaries for `TP=2, PP=2, DP=4` 50-step validation: `1072 MHz -> 1053.6 s / 2475841.1 J / 2349.8 W`, `1080 MHz -> 1046.1 s / 2471497.2 J / 2362.5 W`, `1087 MHz -> 1040.0 s / 2482053.4 J / 2386.6 W`, `1125 MHz -> 1006.1 s / 2492545.3 J / 2477.3 W`.
- [2026-03-15] `TP=2, PP=2, DP=4` preflight on `sd@v100x16-1` passed via `VALIDATE_ONLY=1`; the baseline launcher script is `.context/remote-launch-tp2pp2dp4-baseline50-20260315.sh`.
- [2026-03-15] The `TP=2, PP=2, DP=4` 50-step unfixed baseline completed with local synced artifact `.context/real-v100-tp2pp2dp4-baseline-20260315/v100_tp2pp2dp4_baseline50_20260315_155856_DGX2-1/` and Zeus summary `1030.6 s`, `3798903.8 J`, `3686.1 W`, `0.431 tokens/J`.
- [2026-03-15] Transferred prediction artifacts for `TP=2, PP=2, DP=4` are staged under `.context/real-v100-tp2pp2dp4-transfer-20260315-with-baseline/`; default recommendation is `1080 MHz`, balanced comparator is baseline-like (`1597 MHz`), and the recommended neighborhood is `[1072, 1080, 1087]`.
- [2026-03-15] Remote `TP=2, PP=2, DP=4` static validation session: `power_tp2pp2dp4_pred50_20260315_161856`; launcher script is `.context/remote-launch-tp2pp2dp4-predicted50-20260315.sh`, boot log is `/home/sd/Megatron-DeepSpeed/experiments/_screen_boot/power_tp2pp2dp4_pred50_20260315_161856.log`, and frequencies are `1072/1080/1087/1125 MHz`.
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
- [2026-03-12] Synthetic smoke validation showed low power-fit error but high throughput-fit error, so calibration quality should be checked on real sweep artifacts before trusting any curve-derived default recommendation.
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
