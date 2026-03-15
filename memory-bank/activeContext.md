# Active Context

## Current Focus
[2026-03-13] The objective is now explicit: treat the predictor primarily as a Pareto estimator for `total_energy_ratio` and `total_runtime_ratio` versus the unfixed baseline, then improve continuous curve shape/extrapolation so the default recommendation stops drifting below the measured `1185 MHz` discrete winner on `TP=2, PP=4, DP=2`.

## Recent Changes
- [2026-03-13] Re-ran the Pareto-first accuracy report on the merged `TP=2, PP=4, DP=2` full 11-point sweep (`900/915/930/1185/1192/1200/1207/1215/1245/1252/1260 MHz`). Analytic prediction accuracy on sampled points is now quantified at about `1.77%` MAPE for `runtime_ratio_vs_baseline` and `1.05%` MAPE for `energy_ratio_vs_baseline`, but the continuous default recommendation still drifts too low (`1027 MHz`) because the curve remains slightly too optimistic about low-frequency energy savings and too pessimistic about high-frequency energy cost.
- [2026-03-13] Reframed `scripts/predict_freq_sweet_spot.py` outputs around Pareto-first review: the report now leads with the full predicted Pareto frontier and adds an explicit observed-vs-predicted accuracy section for `runtime_ratio_vs_baseline` and `energy_ratio_vs_baseline`, computed from analytic predictions without observed-overlay interpolation.
- [2026-03-13] Updated the prediction-layer objective so the default `supported_sweet_spot` now selects from the Pareto frontier by `total_energy` first and `total_time` second, while preserving the old equal-weight runtime/energy choice as `supported_balanced_sweet_spot` for comparison.
- [2026-03-13] Updated the expanded transfer comparison artifacts at `.context/real-v100-tp2pp4dp2-static-20260312/measured_vs_predicted_v2_pp_bubble_full_with_refine.tsv` and `final_analysis_v2_pp_bubble_with_refine.md`; the bubble-aware model still overshoots upward (`~1252 MHz` predicted sweet spot), but it now clearly points into the correct high-efficiency band rather than the original failed `900-930 MHz` neighborhood.
- [2026-03-13] The remote refinement sweep `power_tp2pp4dp2_static_refine_20260313_1238c` completed successfully for `1185 / 1192 / 1207 / 1215 MHz`; the measured winner is now `1185 MHz` with Zeus summary `449.706 s`, `1089279.7 J`, `2422.2 W`, `0.601645 tokens/J`, beating the previous `1200 MHz` point on energy efficiency while staying near the same runtime band.
- [2026-03-13] Restored context from the memory bank, synced the completed `1245/1252/1260 MHz` remote artifacts for `TP=2, PP=4, DP=2`, and confirmed they do **not** beat the existing `1200 MHz` point on measured `tokens/J`; the bubble-aware predictor identified the right higher-frequency neighborhood directionally, but still overshot the measured discrete winner.
- [2026-03-13] Launched a narrower remote follow-up sweep around the measured `1200 MHz` region in detached `screen` session `power_tp2pp4dp2_static_refine_20260313_1238c`, covering `1185 / 1192 / 1207 / 1215 MHz` with `DISABLE_CHECKPOINT=1` restored.
- [2026-03-13] Found that the remote repo still contains a stale top-level `run_experiment.sh` without `DISABLE_CHECKPOINT` support; energy-only remote launches must call `scripts/run_experiment.sh` explicitly.
- [2026-03-12] Attempted the originally preferred `TP=8, DP=2` topology on Qwen2.5-7B, but the real run failed immediately because the model shape is incompatible with `TP=8`: `num_attention_heads=28` is not divisible by `8`, and grouped-query `num_key_value_heads=4` would also be incompatible with `TP=8`.
- [2026-03-12] Checked several 16-GPU fallback decompositions against the current stable topology proxy and found that `TP=4, DP=4`, `TP=8, DP=2`, and `TP=4, PP=2, DP=2` all collapse to the same communication feature under the existing analytic model, so they are poor choices for a meaningful topology-transfer test without further modeling changes.
- [2026-03-12] Identified `TP=2, PP=4, DP=2` as the nearest feasible and currently model-visible `DP=2` fallback topology for Qwen2.5-7B on 16 V100s; `VALIDATE_ONLY=1` passed and a detached unfixed-baseline run was launched in remote `screen` session `power_tp2pp4dp2_baseline_20260312_1944`.
- [2026-03-12] The fallback `TP=2, PP=4, DP=2` unfixed baseline completed cleanly with Zeus summary `419.508 s`, `1515838.132 J`, `3613.370 W`, `0.432342 tokens/J`, which is materially slower but lower-power than the old `TP=4, PP=1, DP=4` baseline.
- [2026-03-12] Reused the old-topology hardware-first calibration (`predictions-hw-first-v3`) for a first zero-shot transfer prediction on the new `TP=2, PP=4, DP=2` baseline; the baseline-relative supported sweet spot came out around `915 MHz`, with neighborhood recommendation `[900, 907, 915, 922, 930]`.
- [2026-03-12] Started a detached validation sweep for the transferred recommendation on `TP=2, PP=4, DP=2` covering `900/915/930 MHz` plus a `1200 MHz` control in remote `screen` session `power_tp2pp4dp2_static_validate_20260312_2000`.
- [2026-03-12] Synced the first three completed static validation points (`900/915/930 MHz`) for `TP=2, PP=4, DP=2` and wrote `.context/real-v100-tp2pp4dp2-static-20260312/measured_vs_predicted_partial.tsv`; all three beat the new baseline on total energy, but the zero-shot transfer model materially overpredicted throughput and underestimated runtime ratio (predicted `~0.93-0.96x`, measured `~1.36-1.40x`).
- [2026-03-12] Completed the `1200 MHz` control for `TP=2, PP=4, DP=2` and finalized `.context/real-v100-tp2pp4dp2-static-20260312/measured_vs_predicted.tsv`: among the tested points, `1200 MHz` is best (`tokens/J=0.596836`, runtime ratio `1.0605`, energy ratio `0.7244`), outperforming the zero-shot recommended `900-930 MHz` band.
- [2026-03-12] The first topology-transfer verdict is now clear: the current hardware-first zero-shot model keeps the energy-saving direction but does not retain good sweet-spot ranking after shifting to `TP=2, PP=4, DP=2`; it overestimates low-frequency throughput and likely misses PP/bubble-style topology overhead even on this NvLink-equipped machine.
- [2026-03-12] Adjusted the predictor so `PP` communication cost is no longer discounted relative to `TP` in the workload proxy, and added an explicit pipeline-parallel efficiency term based on microbatches-per-step vs pipeline depth. With this bubble-aware revision, the old topology still predicts `~1252 MHz` and the `TP=2, PP=4, DP=2` zero-shot sweet spot moves back to `~1252 MHz`, much closer to the measured `1200 MHz` control winner.
- [2026-03-12] Refactored the offline frequency predictor toward a baseline-relative, hardware-first objective by calibrating against runtime/energy ratios versus the unfixed baseline, making observed-overlay use opt-in rather than default.
- [2026-03-12] Added a throughput-saturation prior to `analysis/freq_model/model.py` and centered calibration candidates on observed throughput scale rather than raw theoretical token-rate anchors, which prevents the pure analytic model from collapsing to unrealistic low clocks or over-trusting max-clock throughput.
- [2026-03-12] Smoke-validated the refactored predictor manually (without pytest, because local test bootstrap still requires `deepspeed`) and re-ran the real merged V100 CLI locally; the hardware-first prediction now recommends about `1245/1252/1260 MHz` with balanced sweet spot `1252 MHz` against the fresh unfixed baseline.
- [2026-03-12] Added focused freq-model regression coverage for three recent risk points: exact `PP` bubble-efficiency math, throughput-saturation early-clamp behavior, and `load_experiment_samples()` fallback parsing from Zeus log lines when `events.jsonl` is missing.
- [2026-03-12] Refactored the offline frequency predictor toward a baseline-relative, hardware-first objective by calibrating against runtime/energy ratios versus the unfixed baseline, making observed-overlay use opt-in rather than default.
- [2026-03-12] Added a throughput-saturation prior to `analysis/freq_model/model.py` and centered calibration candidates on observed throughput scale rather than raw theoretical token-rate anchors, which prevents the pure analytic model from collapsing to unrealistically low clocks or over-trusting max-clock throughput.
- [2026-03-12] Smoke-validated the refactored predictor manually (without pytest, because local test bootstrap still requires `deepspeed`) and re-ran the real merged V100 CLI locally; the hardware-first prediction now recommends about `1245/1252/1260 MHz` with balanced sweet spot `1252 MHz` against the fresh unfixed baseline.
- [2026-03-11] Added short-run-friendly warmup and eval controls in `scripts/run_experiment.sh`.
- [2026-03-11] Updated NVML lock/reset helpers in `scripts/experiment_utils.sh` to use elevated remote execution.
- [2026-03-11] Extended Zeus integration in `megatron/power_monitor.py` and `megatron/training.py` to preserve `samples/Wh` and `tokens/J`, including leftover/final interval capture.
- [2026-03-11] Added `DISABLE_CHECKPOINT=1` support to `scripts/run_experiment.sh` and validated it remotely.
- [2026-03-11] Switched remote experiment launches to isolated `screen` sessions and recorded session IDs in `.context/remote-screen-sessions.tsv`.
- [2026-03-11] Defined a theory-led, calibration-light frequency sweet-spot prediction workflow and documented it in `docs/plans/2026-03-11-frequency-model-design.md`.
- [2026-03-12] Implemented the first offline prediction scaffold in `analysis/freq_model/` plus `scripts/predict_freq_sweet_spot.py`, with targeted unit coverage.
- [2026-03-12] Smoke-validated the prediction CLI on synthetic experiment artifacts in `.context/freq-model-smoke/`; it produced a discrete recommendation set `[1110, 1200, 1305]` but showed large throughput error on synthetic data.
- [2026-03-12] Ran the first real V100 prediction on `.context/real-v100-power-static-20260312` using five static sweep points (`1200/1305/1380/1410/1597 MHz`). The model used real V100 NVML frequency tables and extrapolated a lower-frequency optimum near `1025 MHz`, recommending `[1020, 1027, 1035]`.
- [2026-03-12] Executed a serial real-V100 validation sweep at `1020/1027/1035 MHz` in remote `screen` session `power_lowfreq_20260312_113029b`. All three runs completed successfully and each beat the earlier `1200 MHz` point on `tokens/J`.
- [2026-03-12] Updated `analysis/freq_model/recommend.py` and `scripts/predict_freq_sweet_spot.py` so recommendations now maximize efficiency only within a configurable runtime-slowdown guardrail, and report estimated total time/energy for a fixed comparison workload.
- [2026-03-12] Synced the real low-frequency validation artifacts (`1020/1027/1035 MHz`) into `.context/real-v100-power-static-20260312-lowfreq/`, built an 8-point merged sweep under `.context/real-v100-power-static-20260312-merged/`, and compared measured-vs-modeled guardrail choices.
- [2026-03-12] Added an observed-frequency interpolation overlay in `analysis/freq_model/recommend.py` and now trust the observed frequency range once enough real points exist; on the merged 8-point V100 sweep this brings guardrail recommendations back in line with measured sweet spots (`1305 MHz` at `1.00x`, around `1200 MHz` near `1.10x`, and `1035/1020 MHz` by `1.25x/1.30x`).
- [2026-03-12] Reframed the analysis around baseline-relative tradeoffs: the usable unfixed baseline is `power_baseline_20260311_20b`, and the current Pareto frontier versus that baseline is `1305/1200/1035/1020 MHz` from faster-to-greener.
- [2026-03-12] Launched a fresh unfixed-baseline rerun in remote `screen` session `power_baseline_rerun_20260312_1512`, with run ID `v100_power_baseline_rerun_20260312_151032_DGX2-1` and `DISABLE_CHECKPOINT=1` for clean energy/time comparison.
- [2026-03-12] The fresh baseline rerun completed cleanly with Zeus summary: `309.165 s`, `1387540.447 J`, `4488.029 W`, `0.472318 tokens/J`. Relative to this rerun, the Pareto frontier remains `1305/1200/1035/1020 MHz` from faster-to-greener.
- [2026-03-12] Added `scripts/plot_baseline_pareto.py` and generated both paper-style and analysis-style 2D Pareto skyline charts from the refreshed baseline-relative TSV.
- [2026-03-12] Modeling direction updated: keep the second-layer predictor hardware-first across the full frequency band, treat frequency itself as part of hardware capability, and avoid overfitting the current small sample set; use samples mainly to calibrate an analytic prior rather than to dominate it.

## Active Decisions
- [2026-03-12] 当前无需把 balanced sweet spot 人为偏向已观测 Pareto 频点；`~1250 MHz` 可以视为合理的理论甜点频率，只是当前 V100 可锁定离散频点中没有该值。
- [2026-03-12] 下一阶段优先改变硬件拓扑来检验模型泛化能力，首选对比场景是 `TP=8, DP=2`（保持其余条件尽量可比），观察 baseline-relative 预测与实测 Pareto/甜点是否仍保持较好一致性。
- [2026-03-12] Because `TP=8, DP=2` is structurally invalid for the current Qwen2.5-7B head layout, the working fallback for the first transferable `DP=2` validation is `TP=2, PP=4, DP=2` on the same 16-GPU node.
- [2026-03-12] The user clarified that this DGX/V100 machine has `NvLink`, so topology-transfer analysis should treat inter-GPU communication as stronger than plain PCIe and explicitly watch for the current model overestimating communication penalties on multi-stage topologies.
- [2026-03-12] For the current transfer miss, prioritize modeling pipeline/topology overhead (for example `PP=4` bubble or stage-schedule effects) before further tweaking raw communication-bandwidth assumptions.
- [2026-03-12] For now, treat `PP` and `TP` communication pressure as same-order effects in the topology proxy, then use a separate pipeline bubble efficiency term to account for deeper `PP` throughput loss without forcing a radically different sweet-spot frequency.
- [2026-03-11] V100 work is focused on static frequency energy efficiency, not quantization.
- [2026-03-11] Real runs should happen on `sd@v100x16-1` first.
- [2026-03-11] Energy-only sweeps should avoid checkpoint writes unless explicitly requested.
- [2026-03-11] The new prediction layer will model `throughput(f)` and `power(f)` first, then derive efficiency metrics such as `tokens/J`.
- [2026-03-11] The prediction layer should prioritize hardware/task features and use only a small number of interpretable calibration parameters.
- [2026-03-12] Real V100 prediction runs should start from the comparable `power_static*` sweep artifacts before mixing in baseline or short-run variants.
- [2026-03-12] The current sweet-spot analysis should use the unfixed baseline (default GPU frequency behavior) as the comparison anchor and report each static frequency by its total-energy ratio and total-runtime ratio versus baseline.
- [2026-03-12] Do not impose a hard runtime-acceptance guardrail for now; instead present the baseline-relative energy/time tradeoff so different trainers can choose according to whether they care more about energy or elapsed time.
- [2026-03-12] Because the end-to-end training regime is expected to stay stable, per-step power and step-time measurements are acceptable proxy observables for total-energy and total-runtime behavior, as long as comparisons keep workload and setup fixed.

## Next Steps
- Keep using `python -m pytest --noconftest tests/unit_tests/test_freq_model.py` for local freq-model regression checks until the repo-wide pytest bootstrap can import `deepspeed` again.
- For the invalid `TP=8, DP=2` request, preserve the failure as a hard feasibility note (`28` attention heads / `4` KV heads cannot be partitioned by `TP=8`).
- Wait for the `TP=2, PP=4, DP=2` static validation sweep (`900/915/930/1200 MHz`) to finish, sync the artifacts locally, and compare measured runtime/energy ratios against the zero-shot transfer prediction.
- If the `1200 MHz` control confirms the same pattern, update the topology-transfer conclusion to emphasize missing PP/topology overhead (not just raw communication bandwidth) in the current hardware-first analytic prior.
- Rework the transfer-oriented analytic prior so topology changes with higher pipeline depth do not inherit overly optimistic low-frequency throughput from the old topology calibration.
- For the next validation round on `TP=2, PP=4, DP=2`, center the follow-up frequency search around the newly observed `1200 MHz` region rather than the failed zero-shot `900-930 MHz` neighborhood.
- `~1250 MHz` is currently accepted as a reasonable continuous/theoretical sweet spot; do not bias the model toward sampled Pareto points unless later topology-transfer validation shows a clear need.
- Build an offline artifact loader that extracts workload, topology, and Zeus metrics from `run.json`, `events.jsonl`, and logs.
- Refit and reassess the prediction layer now that the local merged 8-point dataset (`1020/1027/1035/1200/1305/1380/1410/1597 MHz`) is available for calibration.
- Tighten the interpolation/overlay policy and calibration reporting using the real 8-point V100 sweep, especially around the `1.02x-1.20x` guardrail region where the model now tracks measured choices closely but still prefers interpolated in-between clocks rather than exact sampled points.
- Refit the new runtime-guarded recommendation layer using the expanded real V100 sweep so the guardrail and energy ranking stay well calibrated below `1200 MHz`.
- Continue the V100 short-run sweep with remaining candidate frequencies such as `1500 MHz`.
- For the current topology, keep reporting both the theoretical continuous sweet spot and the measured discrete Pareto options, rather than collapsing them into a single preferred static clock.
- Optionally improve tracker finalize behavior when runs exit without checkpointing or when finalization is partially incomplete.

## Important Patterns & Preferences
- Use remote `screen` for runs that take more than a quick validation.
- Record `screen` session IDs and attachment commands immediately after launch.
- Preserve both `samples/Wh` and `tokens/J` in experiment summaries.
- Keep experiment conditions identical except for the frequency policy being tested.
- When comparing candidate frequencies, report baseline-relative total energy and total elapsed time first; avoid collapsing the tradeoff into a single hard constraint before the user decides the preferred balance.

## Current Blockers
- `run.json` finalization is still less reliable than log/events parsing for some runs.
- Sweep coverage is still incomplete without a `1500 MHz` data point.
