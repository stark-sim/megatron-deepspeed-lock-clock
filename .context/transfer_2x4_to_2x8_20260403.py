#!/usr/bin/env python3
"""2x4 -> 2x8 transfer validation using clean IB-enabled source data."""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path("/home/sd/Megatron-DeepSpeed")
sys.path.insert(0, str(REPO))

from analysis.freq_model.calibrate import calibrate_frequency_model
from analysis.freq_model.features import derive_model_features
from analysis.freq_model.hardware import build_hardware_features
from analysis.freq_model.model import CalibrationParams
from analysis.freq_model.recommend import build_prediction_bundle
from analysis.freq_model.workload import load_experiment_samples, LoadedRunSample

EXPERIMENT_ROOT = REPO / "experiments"
OUT_DIR = REPO / ".context" / "transfer_2x4_to_2x8_20260403"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _mean(values):
    filtered = [v for v in values if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def _ape(obs, pred):
    if obs is None or pred is None:
        return None
    return abs(obs - pred) / max(abs(obs), 1e-9)


# Load all samples
collection = load_experiment_samples(str(EXPERIMENT_ROOT), include_baseline=True)

# Filter 2x4 source samples (latest clean IB-enabled runs: 20260403)
source_samples = [
    s for s in collection.samples
    if s.run_id.startswith("dual8_tp4pp1dp2_static990_20260403_114053")
    or s.run_id.startswith("dual8_tp4pp1dp2_static1080_20260403_015926")
    or s.run_id.startswith("dual8_tp4pp1dp2_static1155_20260403_020750")
]
source_samples.sort(key=lambda s: s.observed.frequency_mhz or 0)
print(f"Source (2x4) samples: {[(s.run_id, s.observed.frequency_mhz) for s in source_samples]}")

# Filter 2x8 target samples (latest clean IB-enabled runs)
target_samples = [
    s for s in collection.samples
    if s.run_id.startswith("dual16_tp4pp1dp4_static990_20260403_102608")
    or s.run_id.startswith("dual16_tp4pp1dp4_static1080_20260403_103520")
    or s.run_id.startswith("dual16_tp4pp1dp4_static1155_20260403_104355")
]
target_samples.sort(key=lambda s: s.observed.frequency_mhz or 0)
print(f"Target (2x8) samples: {[(s.run_id, s.observed.frequency_mhz) for s in target_samples]}")

if len(source_samples) < 3:
    print("ERROR: Not enough source samples (need 3)")
    sys.exit(1)
if len(target_samples) < 3:
    print("ERROR: Not enough target samples (need 3)")
    sys.exit(1)

# Derive features for source (calibration) topology
source_hardware = build_hardware_features(source_samples[0].run_payload)
source_features = derive_model_features(source_hardware, source_samples[0].workload)

# Calibrate on 2x4
source_baseline = None
for s in source_samples:
    if s.observed.frequency_mhz is None:
        source_baseline = s
        break
calibration = calibrate_frequency_model(
    source_samples, source_hardware, [source_features] * len(source_samples), baseline_sample=source_baseline
)
params = calibration.params
print(f"\n=== Calibration on 2x4 ===")
print(f"Default recommendation (source): {calibration.prediction['supported_sweet_spot']['frequency_mhz']} MHz")
print(f"Recommended frequencies: {calibration.prediction['recommended_frequencies_mhz']}")

# Derive features for target (2x8) topology
target_hardware = build_hardware_features(target_samples[0].run_payload)
target_features = derive_model_features(target_hardware, target_samples[0].workload)

print(f"\n=== 2x4 vs 2x8 Topology Comparison ===")
print(f"2x4: TP={source_samples[0].workload.tensor_model_parallel_size}, "
      f"PP={source_samples[0].workload.pipeline_model_parallel_size}, "
      f"DP={source_samples[0].workload.data_parallel_size}")
print(f"2x8: TP={target_samples[0].workload.tensor_model_parallel_size}, "
      f"PP={target_samples[0].workload.pipeline_model_parallel_size}, "
      f"DP={target_samples[0].workload.data_parallel_size}")

# Pick a baseline for target (use lowest freq observed run as baseline anchor)
target_baseline = target_samples[0]  # 990 MHz
comparison_steps = target_baseline.workload.train_iters or 20

# Build prediction bundle for 2x8 using 2x4 calibration params
transfer_prediction = build_prediction_bundle(
    hardware=target_hardware,
    features=target_features,
    params=params,
    metric="tokens_per_j",
    neighborhood=1,
    comparison_steps=comparison_steps,
    baseline_sample=target_baseline,
    observed_samples=target_samples,
    use_observed_overlay=False,
)

print(f"\n=== Transfer Prediction (2x4 -> 2x8) ===")
print(f"Default recommendation (2x8): {transfer_prediction['supported_sweet_spot']['frequency_mhz']} MHz")
print(f"Recommended frequencies: {transfer_prediction['recommended_frequencies_mhz']}")
print(f"Pareto frontier: {transfer_prediction['pareto_frontier_frequencies_mhz']}")

# Accuracy assessment: compare predicted vs observed on 2x8 target frequencies
predicted_by_freq = {int(p["frequency_mhz"]): p for p in transfer_prediction["supported_predictions"]}

accuracy_points = []
print(f"\n=== Accuracy Assessment ===")
print(f"{'Freq':>6} | {'ObsTime':>8} | {'PredTime':>8} | {'TimeAPE':>8} | {'ObsPower':>9} | {'PredPower':>9} | {'PowerAPE':>8}")
print("-" * 85)

for sample in target_samples:
    freq = int(sample.observed.frequency_mhz or 0)
    pred = predicted_by_freq.get(freq)
    if pred is None:
        continue

    observed_steps = max(sample.observed.num_steps, 1)
    observed_step_time_s = sample.observed.time_s / observed_steps
    observed_total_time_s = observed_step_time_s * comparison_steps
    observed_total_energy_j = sample.observed.avg_power_w * observed_total_time_s

    baseline_total_time_s = target_baseline.observed.time_s / max(target_baseline.observed.num_steps, 1) * comparison_steps
    baseline_total_energy_j = target_baseline.observed.avg_power_w * baseline_total_time_s

    observed_runtime_ratio = observed_total_time_s / max(baseline_total_time_s, 1e-9)
    observed_energy_ratio = observed_total_energy_j / max(baseline_total_energy_j, 1e-9)

    predicted_runtime_ratio = pred.get("runtime_ratio_vs_baseline")
    predicted_energy_ratio = pred.get("energy_ratio_vs_baseline")

    time_ape = _ape(observed_total_time_s, pred["estimated_total_time_s"])
    power_ape = _ape(sample.observed.avg_power_w, pred["estimated_avg_power_w"])

    point = {
        "frequency_mhz": freq,
        "observed_total_time_s": observed_total_time_s,
        "predicted_total_time_s": pred["estimated_total_time_s"],
        "observed_avg_power_w": sample.observed.avg_power_w,
        "predicted_avg_power_w": pred["estimated_avg_power_w"],
        "total_time_ape": time_ape,
        "avg_power_ape": power_ape,
    }
    accuracy_points.append(point)
    print(f"{freq:>6} | {observed_total_time_s:>8.1f} | {pred['estimated_total_time_s']:>8.1f} | "
          f"{time_ape:>7.2%} | {sample.observed.avg_power_w:>9.1f} | {pred['estimated_avg_power_w']:>9.1f} | "
          f"{power_ape:>7.2%}")

accuracy_summary = {
    "comparison_steps": comparison_steps,
    "point_count": len(accuracy_points),
    "total_time_mape": _mean([p["total_time_ape"] for p in accuracy_points]),
    "avg_power_mape": _mean([p["avg_power_ape"] for p in accuracy_points]),
    "points": accuracy_points,
}

print("-" * 85)
print(f"{'MAPE':>6} | {'':>8} | {'':>8} | {accuracy_summary['total_time_mape']:>7.2%} | "
      f"{'':>9} | {'':>9} | {accuracy_summary['avg_power_mape']:>7.2%}")

payload = {
    "source_topology": "dual8_tp4pp1dp2",
    "target_topology": "dual16_tp4pp1dp4",
    "source_samples": [s.run_id for s in source_samples],
    "target_samples": [s.run_id for s in target_samples],
    "calibration_params": params.to_dict(),
    "transfer_prediction": transfer_prediction,
    "accuracy_summary": accuracy_summary,
}

out_json = OUT_DIR / "transfer_comparison.json"
out_json.write_text(json.dumps(payload, indent=2) + "\n")
print(f"\n✅ Results saved to {out_json}")
