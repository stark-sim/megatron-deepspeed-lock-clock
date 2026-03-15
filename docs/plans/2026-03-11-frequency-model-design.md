# Frequency Sweet-Spot Prediction Design

## Goal

Add an offline prediction layer that estimates a continuous training-efficiency curve over GPU core frequency, then recommends a small set of real lockable frequencies near the predicted optimum for validation runs.

The design is intentionally theory-led and calibration-light. It should generalize beyond the current V100 short-run experiments by expressing the problem in terms of hardware capability and workload characteristics, while still using measured sweep data to correct a small number of interpretable parameters.

## Scope

### In Scope
- Predict `throughput(f)` and `power(f)` for fixed-frequency training runs.
- Derive `tokens/J`, `tokens/Wh`, and `samples/Wh` from the predicted base quantities.
- Consume existing experiment artifacts such as `run.json`, `events.jsonl`, and Zeus summaries.
- Recommend a narrow discrete validation set around the predicted optimum.
- Update the model incrementally as new sweep points are collected.

### Out of Scope
- Online control inside the training loop.
- Architecture-specific hand tuning for every GPU generation.
- A black-box regressor that memorizes environment-specific frequency outcomes.

## Design Principles

- Use hardware and workload features as first-class inputs.
- Keep calibration parameters few, interpretable, and weakly environment-specific.
- Model `throughput` and `power` directly; derive efficiency metrics afterward.
- Prefer stable proxies over fragile exact counters in the first version.
- Keep the prediction layer off the critical training path until validation is mature.

## Model Structure

The prediction layer is split into three levels:

1. `hardware_features`
   - GPU model, count, supported graphics clocks, `f_max`, theoretical compute capability, memory bandwidth, optional power-cap/TDP metadata.
2. `workload_features`
   - Model size, layer count, hidden size, FFN size, sequence length, micro/global batch, TP/PP/DP, ZeRO stage, precision mode.
3. `calibration_params`
   - Small interpretable coefficients such as effective compute utilization, memory efficiency, communication penalty, static-power term, and dynamic-power exponent.

The continuous curve uses two submodels:

- `Throughput(f)` is a smooth bottleneck model combining compute-limited, memory-limited, and communication-limited terms.
- `Power(f)` is decomposed into a frequency-weak static term and a nonlinear dynamic term.

Efficiency is derived afterward:

- `efficiency(f) = throughput(f) / power(f)`

## Feature Extraction

The first implementation should consume structured experiment artifacts before parsing raw logs.

### Primary Sources
- `run.json` for model, training, parallelism, frequency policy, topology, and environment metadata.
- `events.jsonl` for interval metrics when finalization is incomplete.
- Main training log Zeus summaries as the final fallback.

### Derived Features
- Approximate per-step FLOPs.
- Approximate per-step memory traffic.
- Arithmetic intensity proxy.
- Tokens per step and samples per step.
- Communication proxy built from topology and workload dimensions.
- Observed steady-state step time, throughput, energy, and average power.

The first version should favor robust approximations over exact architecture-specific counters.

## Candidate Recommendation

The model produces a continuous optimum `f*` over the valid frequency interval. That optimum is never treated as final truth. Instead, it is projected onto the real supported frequency set and expanded into a small neighborhood, for example the nearest point plus one or two neighbors on each side.

The experiment loop then validates only those discrete candidates. If the best real point lands on the edge of the recommended neighborhood, the search expands outward once more.

## Calibration Strategy

Calibration fits a small number of interpretable parameters to existing sweep data.

- Fit `throughput` and `power` jointly.
- Use derived efficiency metrics only for evaluation and ranking.
- Regularize toward smooth, single-peak or near-single-peak efficiency behavior.
- Refit incrementally when a new frequency point is added.

This keeps the model grounded in physics-inspired structure while still adapting to measured training behavior.

## Repository Integration

The first implementation should be offline analysis code, separate from the training launcher:

- `analysis/freq_model/hardware.py`
- `analysis/freq_model/workload.py`
- `analysis/freq_model/features.py`
- `analysis/freq_model/model.py`
- `analysis/freq_model/calibrate.py`
- `analysis/freq_model/recommend.py`
- `scripts/predict_freq_sweet_spot.py`

Outputs:

- `prediction.json` for machine-readable results.
- `prediction_report.md` for human review.

## Validation Criteria

- The continuous efficiency curve is physically plausible.
- The recommended discrete set contains the true best point or one of its immediate neighbors.
- Throughput and power prediction errors are bounded and tracked separately.
- Model updates remain stable as additional sweep points are added.

## Implementation Phases

1. Build artifact loader and feature extraction pipeline.
2. Implement the theory-led throughput and power models.
3. Add lightweight calibration against existing V100 sweep data.
4. Recommend discrete validation frequencies near the predicted optimum.
5. Generate comparison reports between predicted and observed results.
6. Use new validation runs to refine parameters and confidence.
