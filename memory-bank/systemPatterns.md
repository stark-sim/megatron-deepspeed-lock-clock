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
- [2026-03-15] Clock cleanup for experiment runs should be layered: keep the in-process NVML reset in `megatron/gpu_freq_manager.py`, but also add a launcher-level `EXIT` trap in `scripts/run_experiment.sh` for any mode that may touch clocks (`static`, `dynamic`, `dryrun`) so node-local GPUs are unlocked even if the training process exits without reaching the normal Python shutdown path. Keep this path quiet on success and warn only on reset failures unless actively debugging clock cleanup.

## Artifact Pattern
- `run.json` is the canonical run summary when finalize succeeds.
- `events.jsonl` is a reliable backup source for interval metrics, especially when final checkpointing or finalize logic is incomplete.
- The training log remains the fallback truth source for final `[Zeus] Steps 1-20 ...` summaries.

## Predictive Modeling Pattern
- [2026-03-16] For cross-topology transfer, keep the source topology's learned low/high curve shape, but do not inherit low-frequency correction amplitude blindly. Rescale low-frequency throughput correction on the target by its `pipeline_exposed_fraction` relative to the source topology: when pipeline-exposed waiting shrinks, low frequency should hurt throughput more, not less. Low-frequency power correction can be reduced more gently than throughput correction.
- [2026-03-16] The topology-correction logic is now explicitly split into three interpretable channels in code: `pipeline_exposed_fraction` for exposed PP waiting/activation cost, `dp_overlapable_fraction` for mostly overlap-friendly DP/ZeRO cost, and `tp_sync_fraction` for TP synchronization pressure. Preserve these as long-lived concepts even if later coefficient tuning changes.
- [2026-03-16] Low-frequency correction should be treated as a gate on exposed waiting, not a generic topology multiplier: PP-heavy topologies should receive stronger low-frequency correction pressure, while `PP=1` topologies should retain only a much weaker residual term from TP/DP.
- [2026-03-16] Communication frequency sensitivity should use an exposed-communication proxy rather than raw `communication_share`, so aggregate communication volume and critical-path exposure stay separable in future model revisions.
- [2026-03-17] For low-TP transfer, do not let the source topology's clock elasticity carry over unchanged. Dampen sub-max throughput frequency elasticity using the target topology's `tp_sync_fraction`, `pipeline_exposed_fraction`, and `dp_overlapable_fraction`, and add a small low-frequency power-retention floor so `TP=1` targets do not inherit unrealistically large power savings from the source curve.
- [2026-03-17] Full-curve transfer also needs an extrapolation trust boundary: below the source topology's observed minimum frequency, do not keep extending the same softened low-frequency elasticity indefinitely. Store the source observed frequency band in calibration params and blend deep-low-frequency throughput back toward the base hardware scaling once predictions leave that band.
- [2026-03-15] The prediction layer should now be treated as a continuous `throughput(f)` / `power(f)` curve first, not a single-point picker. A lightweight two-band residual correction layer can sit on top of the hardware-first prior: one low-frequency band plus one smooth mid/high-frequency band blended by a sigmoid transition, with calibration targeted directly at sampled-point `total_time` and `total_energy` rather than only throughput/power magnitudes.
- [2026-03-15] Refined the `PP` topology proxy so pipeline bubble exposure now inflates the `PP` share of communication pressure: keep the classic throughput-side bubble efficiency term `m / (m + PP - 1)`, but also scale the `PP` communication component by `1 + bubble_fraction`, where `bubble_fraction = (PP - 1) / (m + PP - 1)`. This keeps `PP=1` unchanged while making deeper pipelines and scarcer microbatches more strongly penalize low-frequency throughput.
- 拓扑变更验证前，先检查模型结构约束是否允许目标 TP：对于当前 Qwen2.5-7B 配置，`num_attention_heads=28` / `num_key_value_heads=4` 使得 `TP=8` 在当前 Megatron 实现下不可行。
- 如果首选拓扑因模型形状约束不可运行，应优先选择“同样改变 DP、且当前 analytic proxy 能区分”的最近可行拓扑；当前 16×V100/Qwen2.5-7B 的首选 fallback 是 `TP=2, PP=4, DP=2`。
- 在 `DGX2-1` 这类带 `NvLink` 的单机拓扑上，pipeline/TP 通信惩罚不应默认按弱互联假设解释；如果 transfer 预测明显偏向过低频点，要优先检查是否把通信成本估高了。
- 当拓扑从 `TP=4,PP=1,DP=4` 转到 `TP=2,PP=4,DP=2` 时，即使机器具备 `NvLink`，transfer 误差仍可能主要来自 pipeline depth / bubble / stage schedule 等未建模开销，而不是裸通信带宽不足；这类拓扑迁移不能只靠“更强互联”来解释。
- 在当前 analytic proxy 中，`PP` 通信项不应被显著低估；更稳妥的做法是把 `PP` 与 `TP` 视为同数量级的通信压力来源，再额外乘上一个基于 `microbatches_per_step / (microbatches_per_step + PP - 1)` 的 pipeline bubble efficiency 去表达深层 `PP` 的吞吐折减。
- 当连续模型的能耗/耗时曲线在不可锁定的中间频点（例如 `~1250 MHz`）附近表现最好时，应把它解释为“连续曲线上的局部最优区域”，用于理解曲线形状，而不是把它当作主要优化目标去追逐单一离散频点。
- 模型下一阶段验证应优先通过拓扑变化（例如 `TP=8, DP=2`）来测试泛化，而不是只在同一拓扑上继续微调到更贴合现有样本。
- 在进入下一轮泛化验证时，应优先选择“结构可行且当前 analytic proxy 能区分”的 16-GPU 拓扑；对于已知会在 proxy 中退化成旧特征的分解（例如 `TP=4, PP=2, DP=2`），暂时不值得优先投入真实机测试。
- [2026-03-15] 当前下一站泛化验证已具体落到 `TP=2, PP=2, DP=4`：它保持 16×V100 单机、结构可行，并且比 `TP=2, PP=4, DP=2` 明显减轻 pipeline bubble，适合检验当前 correction layer 是否会把整条低频曲线再次错误地压向过低频。
- [2026-03-15] `TP=2, PP=2, DP=4` 结果说明：当前 correction layer 在“较浅 PP”拓扑上的整条曲线排序泛化明显更稳，低误差区域能覆盖正确的 `1080 MHz` 邻域；剩余误差主要是绝对 runtime/energy 规模仍偏保守，而不是曲线方向错误。
- [2026-03-15] 再下一站优先用 `TP=2, PP=1, DP=8`：它把 `PP` 从 2 降到 1，能进一步检验当前模型在“无 pipeline bubble、较高 DP”条件下是否还会保持正确的曲线形状排序。
- [2026-03-15] `TP=2, PP=1, DP=8` 结果表明：当前 correction layer 在 `PP=1` 的 no-bubble 拓扑上会再次把低频曲线压得过于乐观；虽然 baseline-relative 总能耗比值拟合仍很紧，但绝对 runtime / power 尺度偏保守，导致频点排序被错误地推向低频。
- [2026-03-15] 长期设计概念应升级为“拓扑开销三分法”：把 distributed-training 开销拆成 `pipeline_exposed_cost`、`dp_overlapable_cost`、`tp_sync_cost`，并明确区分“通信体量”与“关键路径暴露度”。低频 correction 主要由 `pipeline_exposed_cost` 驱动，而不是由总 distributed complexity 驱动。详见 `docs/plans/2026-03-15-topology-explanatory-adjustment-design.md`。
- Baseline-relative prediction should be the default review mode: estimate each static frequency as `runtime_ratio_vs_baseline` and `energy_ratio_vs_baseline`, then construct the Pareto skyline in that 2D ratio plane.
- The hardware-first throughput prior now includes a calibrated throughput-saturation ratio: below the saturation knee, throughput scales analytically with effective clock; above it, extra clock mostly increases power while throughput plateaus.
- Calibration candidate generation should be centered on observed throughput scale (effective utilization of hardware anchors), not on raw theoretical token/s anchors, so theory shapes the curve but samples still set the achievable magnitude.
- Frequency recommendation should be generated by an offline prediction layer, not embedded directly in the training loop.
- The prediction layer should consume structured experiment artifacts first (`run.json`, `events.jsonl`) and use logs only as a fallback.
- Model `throughput(f)` and `power(f)` as the primary continuous quantities, then derive `tokens/J`, `tokens/Wh`, and `samples/Wh`.
- The optimization target is total energy to complete the same training workload, with an explicit runtime guardrail so lower clocks are rejected if they stretch wall-clock time too much.
- Predictive modeling should remain hardware-first over the entire supported frequency band: encode frequency as part of effective hardware capability and let samples act as light calibration anchors, not as the primary shape of the curve.
- Prefer analytic priors plus small residual calibration over sample-heavy interpolation when estimating unsampled frequencies, especially for cross-run or cross-hardware transfer.
- Use hardware capability and workload features as inputs, while keeping calibration parameters few and interpretable.
- The first implementation lives in `analysis/freq_model/` and is invoked through `scripts/predict_freq_sweet_spot.py`.
- When a continuous curve suggests an interesting low-error region, map that region onto the real supported graphics-clock set for validation, but evaluate success by full-curve `total_time` / power accuracy rather than by hitting one preferred point.

- On the remote V100 host, prefer `scripts/run_experiment.sh` as the canonical launcher path; a stale top-level `run_experiment.sh` may still exist and can silently drop newer controls such as `DISABLE_CHECKPOINT=1`.
- Prediction outputs may still expose a default operating point for convenience, but that field is secondary: the primary artifact is the full Pareto-front `runtime_ratio_vs_baseline` / `energy_ratio_vs_baseline` curve and its associated `total_time` / power predictions.
- Prediction reports should be Pareto-first: present the full `runtime_ratio_vs_baseline` / `energy_ratio_vs_baseline` frontier as the main decision surface, and treat any single default frequency as only a convenience field layered on top.
- Prediction-accuracy summaries must be computed from analytic supported-frequency predictions without observed-overlay interpolation; otherwise overlay can hide model error on already-sampled frequencies.
- The new source-gap low-band throughput regularizer is only valid for true transfer cases: persist source topology fractions during calibration and gate topology-distance blending on `reference_topology_features_present` so no-reference or synthetic single-topology evaluations keep the older behavior.
- [2026-03-21] Cross-node calibration points should be matched single-vs-dual measurements at the same frequency and world size whenever possible, and the fit should weight points by fidelity instead of treating rough legacy estimates as equal evidence.
- [2026-03-21] Cross-node TP/PP/DP proxies must be span-aware: derive them from the actual rank-to-node layout and only charge a communication family when that family truly crosses nodes in the resolved topology.
- [2026-03-21] For PP cross-node waits, byte volume alone is insufficient across topologies. Use a reference-topology residual correction so the PP penalty can increase with additional pipeline replicas per node and with pipeline bubble growth relative to the calibrated reference layout.
- [2026-03-21] Cross-node power error on the validated dual-node curve is strongly frequency-dependent: a light PP-scaled low-frequency multiplier works better than a topology-only constant drop. Activate it only below a calibrated frequency ratio threshold instead of de-rating all clocks equally.
- [2026-03-21] When only a few cross-topology calibration points exist, prefer a single topology-pressure term over multiple free residual coefficients. In this project, `PP wait pressure = replicas_per_node × pipeline_bubble_fraction` (gated on PP really crossing nodes) is a better cross-topology prior than fitting separate replica and bubble residual coefficients independently.
