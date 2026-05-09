#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.freq_model.hardware import HardwareFeatures, build_hardware_features
from analysis.freq_model.workload import WorkloadFeatures, build_workload_features, load_experiment_samples
from analysis.freq_model.features import derive_model_features
from analysis.freq_model.model import CalibrationParams
from analysis.freq_model.recommend import build_prediction_bundle

if len(sys.argv) != 3:
    raise SystemExit('usage: predict-transfer-tp2pp2dp4.py BASELINE_ROOT OUTPUT_DIR')


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _transfer_rescaled_params(source: dict, target_features) -> CalibrationParams:
    source_params = CalibrationParams(**source['calibration']['params'])
    source_features = derive_model_features(
        HardwareFeatures(**source['hardware']),
        WorkloadFeatures(**source['workload']),
    )
    source_pipeline = max(source_features.pipeline_exposed_fraction, 1e-9)
    pipeline_ratio = _clamp(target_features.pipeline_exposed_fraction / source_pipeline, 0.0, 1.25)
    distributed_factor = _clamp(
        target_features.pipeline_exposed_fraction
        + target_features.tp_sync_fraction
        + target_features.dp_overlapable_fraction,
        0.0,
        1.0,
    )
    throughput_low_scale = _clamp(
        1.0 + (0.75 * (1.0 - pipeline_ratio) * (0.55 + (0.45 * distributed_factor))),
        1.0,
        1.75,
    )
    power_low_scale = _clamp(0.75 + (0.25 * pipeline_ratio), 0.75, 1.00)
    return CalibrationParams(
        **{
            **source_params.to_dict(),
            'throughput_low_freq_correction': source_params.throughput_low_freq_correction * throughput_low_scale,
            'power_low_freq_correction': source_params.power_low_freq_correction * power_low_scale,
        }
    )


baseline_root = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
source_prediction_path = REPO / '.context/real-v100-tp2pp4dp2-rerun-20260315-corrected-50steps/prediction.json'
source = json.loads(source_prediction_path.read_text())
collection = load_experiment_samples(str(baseline_root), include_baseline=True)
baseline_candidates = [sample for sample in collection.samples if sample.observed.frequency_mhz is None]
if not baseline_candidates:
    raise SystemExit(f'no baseline sample found under {baseline_root}')
baseline_sample = sorted(baseline_candidates, key=lambda s: s.run_id)[-1]
hardware = build_hardware_features(baseline_sample.run_payload)
workload = build_workload_features(baseline_sample.run_payload)
features = derive_model_features(hardware, workload)
params = _transfer_rescaled_params(source, features)
prediction = build_prediction_bundle(
    hardware=hardware,
    features=features,
    params=params,
    metric='tokens_per_j',
    neighborhood=1,
    comparison_steps=50,
    baseline_sample=baseline_sample,
    observed_samples=[],
    use_observed_overlay=False,
)
payload = {
    'source_prediction': str(source_prediction_path),
    'baseline_run': baseline_sample.to_dict(),
    'hardware': hardware.to_dict(),
    'workload': workload.to_dict(),
    'derived_features': features.to_dict(),
    'calibration_params': params.to_dict(),
    'prediction': prediction,
}
(out_dir / 'prediction.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
(out_dir / 'prediction_report.md').write_text(
    '# TP=2 PP=2 DP=4 Transfer Prediction\n\n'
    + f"- source_prediction: `{source_prediction_path}`\n"
    + f"- baseline_run_id: `{baseline_sample.run_id}`\n"
    + f"- default_frequency_mhz: `{prediction['supported_sweet_spot']['frequency_mhz']}`\n"
    + f"- balanced_frequency_mhz: `{prediction['supported_balanced_sweet_spot']['frequency_mhz']}`\n"
    + f"- recommended_frequencies_mhz: `{prediction['recommended_frequencies_mhz']}`\n"
    + f"- pareto_frontier_frequencies_mhz: `{prediction['pareto_frontier_frequencies_mhz']}`\n"
)
print(json.dumps({
    'baseline_run_id': baseline_sample.run_id,
    'default_frequency_mhz': prediction['supported_sweet_spot']['frequency_mhz'],
    'balanced_frequency_mhz': prediction['supported_balanced_sweet_spot']['frequency_mhz'],
    'recommended_frequencies_mhz': prediction['recommended_frequencies_mhz'],
}, indent=2))
