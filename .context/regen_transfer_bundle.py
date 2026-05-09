import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.freq_model.features import DerivedModelFeatures, derive_model_features
from analysis.freq_model.hardware import HardwareFeatures
from analysis.freq_model.model import CalibrationParams
from analysis.freq_model.recommend import build_prediction_bundle
from analysis.freq_model.workload import LoadedRunSample, ObservedMetrics, WorkloadFeatures

if len(sys.argv) != 4:
    raise SystemExit('usage: regen_transfer_bundle.py <source_prediction> <base_target_prediction> <out_dir>')

source_path = Path(sys.argv[1])
base_target_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

source = json.loads(source_path.read_text())
base_target = json.loads(base_target_path.read_text())

hardware = HardwareFeatures(**base_target['hardware'])
workload = WorkloadFeatures(**base_target['workload'])
derived = derive_model_features(hardware, workload)
params = CalibrationParams(**source['calibration']['params'])
source_workload = WorkloadFeatures(**source['workload']) if source.get('workload') else None
source_derived = None
if source.get('derived_features') and all(key in source['derived_features'] for key in ['pipeline_exposed_fraction', 'dp_overlapable_fraction', 'tp_sync_fraction']):
    source_derived = DerivedModelFeatures(**source['derived_features'])
elif source_workload is not None:
    source_derived = derive_model_features(hardware, source_workload)
if source.get('samples') and params.reference_min_frequency_ratio <= 0.0:
    observed_frequencies = [
        int(sample['observed']['frequency_mhz'])
        for sample in source['samples']
        if sample.get('observed', {}).get('frequency_mhz') is not None
    ]
    if observed_frequencies:
        max_frequency = float(hardware.max_frequency_mhz or max(observed_frequencies) or 1.0)
        params = CalibrationParams(
            **{
                **params.to_dict(),
                'reference_min_frequency_ratio': min(observed_frequencies) / max_frequency,
                'reference_max_frequency_ratio': max(observed_frequencies) / max_frequency,
                'reference_observed_frequency_ratios': tuple(sorted(freq / max_frequency for freq in observed_frequencies)),
                'reference_pipeline_exposed_fraction': (source_derived.pipeline_exposed_fraction if source_derived is not None else params.reference_pipeline_exposed_fraction),
                'reference_dp_overlapable_fraction': (source_derived.dp_overlapable_fraction if source_derived is not None else params.reference_dp_overlapable_fraction),
                'reference_tp_sync_fraction': (source_derived.tp_sync_fraction if source_derived is not None else params.reference_tp_sync_fraction),
                'reference_topology_features_present': bool(source_derived is not None),
            }
        )
baseline_data = base_target['baseline_run']
baseline_sample = LoadedRunSample(
    run_id=baseline_data['run_id'],
    run_dir=Path(baseline_data['run_dir']),
    run_payload={},
    workload=WorkloadFeatures(**baseline_data['workload']),
    observed=ObservedMetrics(**baseline_data['observed']),
)

prediction = build_prediction_bundle(
    hardware=hardware,
    features=derived,
    params=params,
    metric=base_target['prediction'].get('metric', 'tokens_per_j'),
    neighborhood=1,
    comparison_steps=baseline_sample.observed.num_steps,
    baseline_sample=baseline_sample,
    observed_samples=None,
    use_observed_overlay=False,
)

payload = {
    'source_prediction': str(source_path),
    'baseline_run': baseline_sample.to_dict(),
    'hardware': hardware.to_dict(),
    'workload': workload.to_dict(),
    'derived_features': derived.to_dict(),
    'calibration_params': params.to_dict(),
    'prediction': prediction,
}
(out_dir / 'prediction.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps({
    'out_dir': str(out_dir),
    'recommended_frequencies_mhz': prediction['recommended_frequencies_mhz'],
    'default_frequency_mhz': prediction['supported_sweet_spot']['frequency_mhz'],
    'tp1_points': [
        {
            'frequency_mhz': point['frequency_mhz'],
            'runtime_ratio_vs_baseline': point.get('runtime_ratio_vs_baseline'),
            'energy_ratio_vs_baseline': point.get('energy_ratio_vs_baseline'),
            'power_ratio_vs_baseline': point.get('power_ratio_vs_baseline'),
        }
        for point in prediction['supported_predictions']
        if point['frequency_mhz'] in {1177, 1185, 1192, 1252, 1260, 1267}
    ],
}, indent=2))
