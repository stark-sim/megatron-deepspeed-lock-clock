# Progress

## Completed Features
- [x] Full 11-point `TP=2, PP=4, DP=2` Pareto-first accuracy rerun completed; sampled-point prediction error is low (`runtime_ratio_mape≈1.77%`, `energy_ratio_mape≈1.05%`), but the unsampled continuous default still drifts toward `1027 MHz`, so curve-shape/extrapolation remains the main modeling issue.
- [x] Pareto-first prediction reporting completed: CLI/report output now foregrounds the full frontier and an analytic observed-vs-predicted accuracy table for total-time / total-energy ratios, while keeping the default recommendation as a secondary summary.
- [x] Updated the frequency recommendation objective to `total_energy` first and `total_time` second, while preserving the previous balanced sweet spot as a secondary comparator in prediction outputs and reports.
- [x] Completed the remote refinement sweep `1185/1192/1207/1215 MHz` for `TP=2, PP=4, DP=2`; `1185 MHz` is the current best measured discrete point (`tokens/J=0.601645`, runtime ratio `1.0720`, energy ratio `0.7186`).
- [x] Feasibility triage for topology-transfer validation: proved `TP=8, DP=2` is invalid for the current Qwen2.5-7B head layout and established `TP=2, PP=4, DP=2` as a runnable, model-visible fallback on 16 V100s.
- [x] Completed the fallback `TP=2, PP=4, DP=2` unfixed baseline run and generated the first zero-shot baseline-relative transfer prediction (`~915 MHz`, recommendation neighborhood `900-930 MHz`).
- [x] Completed the first `TP=2, PP=4, DP=2` static transfer-validation sweep (`900/915/930/1200 MHz`) and showed that the best tested point is `1200 MHz`, not the zero-shot recommended `900-930 MHz` neighborhood.
- [x] Revised the transfer predictor to make `PP` communication cost comparable to `TP` and added a pipeline bubble efficiency factor; local smoke checks now keep the old-topology sweet spot near `1252 MHz` while moving the new-topology zero-shot sweet spot from `~915 MHz` back to `~1252 MHz`.
- [x] Baseline-relative predictor refactor completed: the recommendation bundle now reports Pareto frontier, balanced sweet spot, and baseline-relative runtime/energy ratios without a hard slowdown guardrail.
- [x] Hardware-first calibration improved with throughput-scale centering and a throughput-saturation prior, reducing real merged-V100 prediction drift and moving the analytic sweet spot into the measured `1200-1305 MHz` neighborhood.
- [x] Added targeted freq-model regression tests for `PP` bubble efficiency, throughput-saturation clamping, and Zeus-log fallback loading; local focused run now passes with `python -m pytest --noconftest -q tests/unit_tests/test_freq_model.py` (`8 passed`).
- [x] Focused validation completed via `py_compile`, direct module smoke, and a real local CLI rerun against `.context/real-v100-power-static-20260312-merged` plus `.context/real-v100-baseline-rerun-20260312`.
- [x] Initial offline frequency-modeling scaffold with calibration, discrete recommendation, CLI output, and targeted tests.
- [x] Synthetic smoke validation of the prediction CLI and output artifacts.
- [x] First real-V100 offline prediction run against five static sweep artifacts.
- [x] Serial validation sweep of predicted low-frequency candidates `1020/1027/1035 MHz` on real V100 hardware.
- [x] Unified experiment launcher and experiment tracking workflow.
- [x] Remote NVML static clock lock/reset support for V100 runs.
- [x] Zeus power metrics integrated into experiment outputs.
- [x] Short-run-compatible warmup and eval configuration handling.
- [x] `screen`-based remote launch workflow with recorded session IDs.
- [x] `DISABLE_CHECKPOINT=1` mode for energy-only sweep runs.
- [x] Remote cleanup of large, obsolete checkpoint directories.
- [x] Runtime-guarded sweet-spot recommendation scoring plus CLI/report output for fixed-workload total energy/time estimates.
- [x] Local sync of the `1020/1027/1035 MHz` validation artifacts plus a measured-vs-modeled guardrail comparison table for the merged 8-point V100 sweep.
- [x] Observed-frequency interpolation overlay that aligns the merged 8-point guardrail recommendations with measured V100 sweet-spot transitions.
- [x] Baseline-relative energy/runtime tradeoff table versus `power_baseline_20260311_20b`, including the current Pareto frontier across the merged 8-point sweep.
- [x] Fresh unfixed-baseline rerun completed and a refreshed tradeoff table was written against `v100_power_baseline_rerun_20260312_151032_DGX2-1`.
- [x] Dual Pareto skyline chart artifacts (paper view + analysis view) generated from the refreshed baseline-relative tradeoff table.

## In Progress
- [ ] Preserve the now-low sampled-point ratio error (`runtime_ratio≈1.77%`, `energy_ratio≈1.05%`) while fixing the continuous low-frequency drift that still pulls the default recommendation toward `1027 MHz` instead of the measured `1185 MHz` winner.
- [ ] 在新硬件拓扑下复用当前 hardware-first 模型，检查其对 baseline-relative runtime/energy tradeoff 与甜点频率的预测能力是否仍然稳定；原始首选 `TP=8, DP=2` 已确认对当前 Qwen2.5-7B 不可行，当前改用 `TP=2, PP=4, DP=2` 继续推进首轮 `DP=2` 泛化验证。
- [ ] Refine the hardware-first transfer model so it captures `PP=4`/multi-stage topology overhead well enough to avoid overly optimistic low-frequency recommendations on NvLink-equipped single-node runs.
- [ ] Validate the bubble-aware zero-shot recommendation (`~1245/1252/1260 MHz`) on `TP=2, PP=4, DP=2`, since the current measured sweep only covered `900/915/930/1200 MHz`.
- [ ] Validate whether the accepted continuous sweet spot around `1250 MHz` still transfers well after changing topology to `TP=8, DP=2`.
- [ ] Improve prediction accuracy and calibration robustness using real V100 sweep artifacts, especially for sub-`1200 MHz` extrapolation.
- [ ] Complete the V100 static-frequency sweep with additional candidate points.
- [ ] Refactor the predictor toward a hardware-first analytic prior with light sample calibration over the full frequency band.
- [ ] Optionally harden experiment finalization/reporting after no-checkpoint runs.

## Known Issues
- `TP=8, DP=2` cannot be run on the current Qwen2.5-7B configuration because `28` attention heads and `4` KV heads are incompatible with `tensor-model-parallel-size=8` in this Megatron setup.
- Under the current stable topology proxy, several 16-GPU decompositions collapse to the same communication feature (`TP=4,DP=4`, `TP=8,DP=2`, `TP=4,PP=2,DP=2`), so topology-transfer validation should avoid those pairings unless the analytic communication model is refined.
- 当前 `~1250 MHz` 更像连续理论甜点；由于实际 NVML 支持频点是离散的，后续解读时应区分“理论甜点”与“可锁定最佳离散频点”。
- The pure hardware-first model is now much more stable, but its supported-frequency Pareto frontier is still broader than the measured discrete frontier because it predicts a smooth plateau from about `1250 MHz` downward; more real points are still needed to tighten that frontier confidently.
- The first prediction model can overestimate throughput materially on synthetic calibration data even when power fit is tight.
- The first real V100 prediction extrapolated the sweet spot down to about `1025 MHz`; follow-up runs at `1020/1027/1035 MHz` now confirm that the low-frequency neighborhood outperforms the old `1200 MHz` point, but the model still needs recalibration for absolute-scale accuracy.
- After adding the low-frequency artifacts, measured guardrail-optimal choices are `1305 MHz` at `<=1.08x`, `1200 MHz` at `<=1.20x`, and about `1035-1020 MHz` only once the runtime budget reaches roughly `1.25-1.30x`; the observed-frequency overlay now tracks these transitions closely, though it can prefer interpolated in-between clocks (for example near `1185 MHz`) instead of exact sampled points.
- Some completed runs still rely on log/event parsing because `run.json` finalize did not fully complete.
- Local repo-wide `pytest` still imports `tests/conftest.py`, which requires `deepspeed`; focused offline analysis tests should currently be run with `--noconftest` unless that dependency is restored locally.
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

- [2026-03-13] Synced the completed `1245/1252/1260 MHz` `TP=2, PP=4, DP=2` static runs into `.context/real-v100-tp2pp4dp2-static-20260312/` and generated `.context/real-v100-tp2pp4dp2-static-20260312/measured_vs_predicted_v2_pp_bubble_full.tsv` plus `final_analysis_v2_pp_bubble.md`.
- [x] Waited for remote refinement sweep `power_tp2pp4dp2_static_refine_20260313_1238c` (`1185 / 1192 / 1207 / 1215 MHz`) to finish, synced the artifacts, and compared them against the previous `1200 MHz` best point.
