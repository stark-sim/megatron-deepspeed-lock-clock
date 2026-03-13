#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.freq_model.calibrate import calibrate_frequency_model
from analysis.freq_model.features import derive_model_features
from analysis.freq_model.hardware import build_hardware_features
from analysis.freq_model.recommend import build_prediction_bundle
from analysis.freq_model.workload import ExperimentCollection, LoadedRunSample, load_experiment_samples


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fixed-frequency sweet spots from experiment artifacts")
    parser.add_argument("--experiment-root", default="experiments", help="Directory containing static-frequency experiment run folders")
    parser.add_argument("--baseline-root", default=None, help="Directory containing baseline (unfixed-frequency) experiment run folders")
    parser.add_argument("--baseline-run-id", default=None, help="Optional baseline run ID to force when baseline-root has multiple candidates")
    parser.add_argument("--output-dir", default=None, help="Directory for prediction outputs")
    parser.add_argument("--metric", choices=["tokens_per_j", "tokens_per_wh", "samples_per_wh"], default="tokens_per_j")
    parser.add_argument("--include-baseline", action="store_true", help="Include non-static runs found inside --experiment-root")
    parser.add_argument("--peak-fp16-tflops-per-gpu", type=float, default=None, help="Override GPU FP16 tensor TFLOPs per GPU")
    parser.add_argument("--memory-bandwidth-gbps-per-gpu", type=float, default=None, help="Override GPU memory bandwidth per GPU")
    parser.add_argument("--power-limit-w-per-gpu", type=float, default=None, help="Override GPU board/power limit per GPU")
    parser.add_argument("--neighborhood", type=int, default=1, help="How many supported frequencies to include on each side of the balanced sweet spot")
    parser.add_argument(
        "--comparison-steps",
        type=int,
        default=None,
        help="Fixed number of training steps used to estimate total time and total energy for prediction reporting",
    )
    parser.add_argument(
        "--use-observed-overlay",
        action="store_true",
        help="Blend observed static points back into the curve with interpolation; disabled by default for hardware-first prediction",
    )
    return parser.parse_args()


def _workload_signature(sample: LoadedRunSample) -> tuple:
    workload = sample.workload
    return (
        workload.num_layers,
        workload.hidden_size,
        workload.ffn_hidden_size,
        workload.seq_length,
        workload.global_batch_size,
        workload.tensor_model_parallel_size,
        workload.pipeline_model_parallel_size,
        workload.data_parallel_size,
        workload.zero_stage,
        workload.precision_mode,
    )


def _validate_workload_consistency(samples: List[LoadedRunSample]) -> None:
    signatures = {_workload_signature(sample) for sample in samples}
    if len(signatures) > 1:
        raise SystemExit("prediction currently expects equivalent workloads across selected runs")


def _default_output_dir(experiment_root: Path) -> Path:
    return experiment_root / "predictions"


def _resolve_comparison_steps(sample: LoadedRunSample, cli_value: int | None) -> int:
    if cli_value is not None:
        return max(cli_value, 1)
    if sample.workload.train_iters > 0:
        return sample.workload.train_iters
    if sample.observed.num_steps > 0:
        return sample.observed.num_steps
    return 1


def _load_baseline_sample(baseline_root: str | None, run_id: str | None) -> LoadedRunSample | None:
    if not baseline_root:
        return None
    collection = load_experiment_samples(baseline_root, include_baseline=True)
    candidates = [sample for sample in collection.samples if sample.observed.frequency_mhz is None]
    if run_id:
        matches = [sample for sample in candidates if sample.run_id == run_id]
        if not matches:
            raise SystemExit(f"baseline run_id not found: {run_id}")
        return matches[0]
    if not candidates:
        raise SystemExit(f"no usable baseline samples found under {baseline_root}")
    return sorted(candidates, key=lambda sample: sample.run_id)[-1]


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _absolute_percentage_error(observed: float | None, predicted: float | None) -> float | None:
    if observed is None or predicted is None:
        return None
    return abs(observed - predicted) / max(abs(observed), 1e-9)


def _mean(values: List[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _build_accuracy_assessment(
    samples: List[LoadedRunSample],
    analytic_prediction: Dict[str, Any],
    baseline_reference: Dict[str, Any] | None,
    comparison_steps: int,
) -> Dict[str, Any]:
    predicted_by_freq = {
        int(point["frequency_mhz"]): point
        for point in analytic_prediction["supported_predictions"]
    }
    points: List[Dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: int(item.observed.frequency_mhz or 0)):
        observed = sample.observed
        if observed.frequency_mhz is None:
            continue
        frequency_mhz = int(observed.frequency_mhz)
        predicted = predicted_by_freq.get(frequency_mhz)
        if predicted is None:
            continue
        observed_steps = max(observed.num_steps, 1)
        observed_step_time_s = observed.time_s / observed_steps
        observed_total_time_s = observed_step_time_s * comparison_steps
        observed_total_energy_j = observed.avg_power_w * observed_total_time_s
        observed_runtime_ratio = None
        observed_energy_ratio = None
        if baseline_reference is not None:
            observed_runtime_ratio = observed_total_time_s / max(baseline_reference["total_time_s"], 1e-9)
            observed_energy_ratio = observed_total_energy_j / max(baseline_reference["total_energy_j"], 1e-9)
        point = {
            "frequency_mhz": frequency_mhz,
            "observed_total_time_s": observed_total_time_s,
            "predicted_total_time_s": predicted["estimated_total_time_s"],
            "observed_total_energy_j": observed_total_energy_j,
            "predicted_total_energy_j": predicted["estimated_total_energy_j"],
            "observed_runtime_ratio_vs_baseline": observed_runtime_ratio,
            "predicted_runtime_ratio_vs_baseline": predicted.get("runtime_ratio_vs_baseline"),
            "observed_energy_ratio_vs_baseline": observed_energy_ratio,
            "predicted_energy_ratio_vs_baseline": predicted.get("energy_ratio_vs_baseline"),
        }
        point["total_time_ape"] = _absolute_percentage_error(point["observed_total_time_s"], point["predicted_total_time_s"])
        point["total_energy_ape"] = _absolute_percentage_error(point["observed_total_energy_j"], point["predicted_total_energy_j"])
        point["runtime_ratio_ape"] = _absolute_percentage_error(
            point["observed_runtime_ratio_vs_baseline"],
            point["predicted_runtime_ratio_vs_baseline"],
        )
        point["energy_ratio_ape"] = _absolute_percentage_error(
            point["observed_energy_ratio_vs_baseline"],
            point["predicted_energy_ratio_vs_baseline"],
        )
        points.append(point)

    return {
        "source": "analytic_supported_predictions_without_overlay",
        "comparison_steps": comparison_steps,
        "point_count": len(points),
        "overlay_enabled_for_primary_prediction": bool(analytic_prediction["overlay"]["enabled"]),
        "runtime_ratio_mape": _mean([point["runtime_ratio_ape"] for point in points]),
        "energy_ratio_mape": _mean([point["energy_ratio_ape"] for point in points]),
        "total_time_mape": _mean([point["total_time_ape"] for point in points]),
        "total_energy_mape": _mean([point["total_energy_ape"] for point in points]),
        "points": points,
    }


def _write_report(output_path: Path, payload: dict) -> None:
    prediction = payload["prediction"]
    objective = prediction["objective"]
    overlay = prediction["overlay"]
    baseline_reference = prediction["baseline_reference"]
    sweet_spot = prediction["supported_sweet_spot"]
    balanced_sweet_spot = prediction.get("supported_balanced_sweet_spot")
    accuracy = payload["prediction_accuracy"]
    lines = [
        "# Frequency Tradeoff Prediction",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- experiment_root: `{payload['experiment_root']}`",
        f"- metric: `{payload['metric']}`",
        f"- runs_used: `{len(payload['samples'])}`",
        "",
        "## Objective",
        f"- mode: `{objective['mode']}`",
        f"- comparison_steps: `{objective['comparison_steps']}`",
        f"- comparison_tokens: `{objective['comparison_tokens']:.0f}`",
        f"- primary: `{objective['primary']}`",
        f"- secondary: `{objective['secondary']}`",
        "",
        "## Baseline Reference",
        f"- run_id: `{baseline_reference['run_id'] if baseline_reference else None}`",
        f"- total_time_s: `{baseline_reference['total_time_s']:.3f}`" if baseline_reference else "- total_time_s: `None`",
        f"- total_energy_j: `{baseline_reference['total_energy_j']:.3f}`" if baseline_reference else "- total_energy_j: `None`",
        f"- avg_power_w: `{baseline_reference['avg_power_w']:.3f}`" if baseline_reference else "- avg_power_w: `None`",
        f"- tokens_per_j: `{baseline_reference['tokens_per_j']:.6f}`" if baseline_reference else "- tokens_per_j: `None`",
        "",
        "## Overlay",
        f"- enabled_for_primary_prediction: `{overlay['enabled']}`",
        f"- uses_observed_interpolation: `{overlay['uses_observed_interpolation']}`",
        f"- observed_frequency_range_mhz: `{overlay['observed_frequency_range_mhz']}`",
        f"- observed_frequency_count: `{overlay['observed_frequency_count']}`",
        "",
        "## Pareto Frontier (Primary View)",
        "- interpretation: `choose along this frontier based on your own energy-vs-time preference`",
        f"- frequencies_mhz: `{prediction['pareto_frontier_frequencies_mhz']}`",
    ]
    for point in prediction["pareto_frontier_predictions"]:
        marker = " default_recommendation" if point["frequency_mhz"] == sweet_spot["frequency_mhz"] else ""
        lines.append(
            "- "
            f"`{point['frequency_mhz']}` MHz:{marker} runtime_ratio_vs_baseline={_format_ratio(point.get('runtime_ratio_vs_baseline'))}, "
            f"energy_ratio_vs_baseline={_format_ratio(point.get('energy_ratio_vs_baseline'))}, "
            f"estimated_total_time_s={_format_float(point.get('estimated_total_time_s'))}, "
            f"estimated_total_energy_j={_format_float(point.get('estimated_total_energy_j'))}"
        )

    lines.extend([
        "",
        "## Prediction Accuracy",
        "- assessment_scope: `observed static-frequency runs only`",
        f"- assessment_source: `{accuracy['source']}`",
        "- note: `accuracy uses analytic predictions without observed overlay, so interpolation does not hide model error`",
        f"- runtime_ratio_mape: `{_format_ratio(accuracy.get('runtime_ratio_mape'))}`",
        f"- energy_ratio_mape: `{_format_ratio(accuracy.get('energy_ratio_mape'))}`",
        f"- total_time_mape: `{_format_ratio(accuracy.get('total_time_mape'))}`",
        f"- total_energy_mape: `{_format_ratio(accuracy.get('total_energy_mape'))}`",
        f"- point_count: `{accuracy['point_count']}`",
        "",
        "## Observed vs Predicted Ratios",
    ])
    for point in accuracy["points"]:
        lines.append(
            "- "
            f"`{point['frequency_mhz']}` MHz: observed_runtime_ratio={_format_ratio(point.get('observed_runtime_ratio_vs_baseline'))}, "
            f"predicted_runtime_ratio={_format_ratio(point.get('predicted_runtime_ratio_vs_baseline'))}, "
            f"runtime_ratio_ape={_format_ratio(point.get('runtime_ratio_ape'))}, "
            f"observed_energy_ratio={_format_ratio(point.get('observed_energy_ratio_vs_baseline'))}, "
            f"predicted_energy_ratio={_format_ratio(point.get('predicted_energy_ratio_vs_baseline'))}, "
            f"energy_ratio_ape={_format_ratio(point.get('energy_ratio_ape'))}"
        )

    lines.extend([
        "",
        "## Default Recommendation",
        "- selection_priority: `total_energy first, total_time second`",
        f"- frequency_mhz: `{sweet_spot['frequency_mhz']}`",
        f"- runtime_ratio_vs_baseline: `{_format_ratio(sweet_spot.get('runtime_ratio_vs_baseline'))}`",
        f"- energy_ratio_vs_baseline: `{_format_ratio(sweet_spot.get('energy_ratio_vs_baseline'))}`",
        f"- estimated_total_time_s: `{sweet_spot['estimated_total_time_s']:.3f}`",
        f"- estimated_total_energy_j: `{sweet_spot['estimated_total_energy_j']:.3f}`",
        "",
        "## Recommended Frequencies",
        f"- frequencies_mhz: `{prediction['recommended_frequencies_mhz']}`",
        "",
        "## Calibration",
        f"- throughput_mape: `{payload['calibration']['throughput_mape']:.4f}`",
        f"- power_mape: `{payload['calibration']['power_mape']:.4f}`",
        f"- runtime_ratio_mape: `{payload['calibration']['runtime_ratio_mape']:.4f}`",
        f"- energy_ratio_mape: `{payload['calibration']['energy_ratio_mape']:.4f}`",
        f"- objective: `{payload['calibration']['objective']:.4f}`",
    ])
    if balanced_sweet_spot is not None:
        lines.extend([
            "",
            "## Balanced Comparator",
            f"- frequency_mhz: `{balanced_sweet_spot['frequency_mhz']}`",
            f"- runtime_ratio_vs_baseline: `{_format_ratio(balanced_sweet_spot.get('runtime_ratio_vs_baseline'))}`",
            f"- energy_ratio_vs_baseline: `{_format_ratio(balanced_sweet_spot.get('energy_ratio_vs_baseline'))}`",
        ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    collection: ExperimentCollection = load_experiment_samples(args.experiment_root, include_baseline=args.include_baseline)
    if not collection.samples:
        raise SystemExit("no usable experiment samples were found")

    _validate_workload_consistency(collection.samples)

    first_sample = collection.samples[0]
    hardware = build_hardware_features(
        first_sample.run_payload,
        peak_fp16_tflops_per_gpu=args.peak_fp16_tflops_per_gpu,
        memory_bandwidth_gbps_per_gpu=args.memory_bandwidth_gbps_per_gpu,
        power_limit_w_per_gpu=args.power_limit_w_per_gpu,
    )
    derived_features = [derive_model_features(hardware, sample.workload) for sample in collection.samples]
    comparison_steps = _resolve_comparison_steps(first_sample, args.comparison_steps)
    baseline_sample = _load_baseline_sample(args.baseline_root, args.baseline_run_id)
    calibration = calibrate_frequency_model(collection.samples, hardware, derived_features, baseline_sample=baseline_sample)
    analytic_prediction = build_prediction_bundle(
        hardware=hardware,
        features=derived_features[0],
        params=calibration.params,
        metric=args.metric,
        neighborhood=args.neighborhood,
        comparison_steps=comparison_steps,
        baseline_sample=baseline_sample,
        observed_samples=collection.samples,
        use_observed_overlay=False,
    )
    prediction = build_prediction_bundle(
        hardware=hardware,
        features=derived_features[0],
        params=calibration.params,
        metric=args.metric,
        neighborhood=args.neighborhood,
        comparison_steps=comparison_steps,
        baseline_sample=baseline_sample,
        observed_samples=collection.samples,
        use_observed_overlay=args.use_observed_overlay,
    )
    prediction_accuracy = _build_accuracy_assessment(
        collection.samples,
        analytic_prediction=analytic_prediction,
        baseline_reference=prediction["baseline_reference"],
        comparison_steps=comparison_steps,
    )

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(collection.root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": _utc_now(),
        "experiment_root": str(collection.root_dir),
        "metric": args.metric,
        "hardware": hardware.to_dict(),
        "workload": first_sample.workload.to_dict(),
        "samples": [sample.to_dict() for sample in collection.samples],
        "skipped_runs": collection.skipped_runs,
        "derived_features": derived_features[0].to_dict(),
        "baseline_sample": baseline_sample.to_dict() if baseline_sample is not None else None,
        "calibration": {
            "params": calibration.params.to_dict(),
            "throughput_mape": calibration.throughput_mape,
            "power_mape": calibration.power_mape,
            "runtime_ratio_mape": calibration.runtime_ratio_mape,
            "energy_ratio_mape": calibration.energy_ratio_mape,
            "objective": calibration.objective,
        },
        "prediction_accuracy": prediction_accuracy,
        "prediction": prediction,
    }

    json_path = output_dir / "prediction.json"
    report_path = output_dir / "prediction_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, payload)

    print(f"[Prediction] Wrote {json_path}")
    print(f"[Prediction] Wrote {report_path}")
    if prediction["baseline_reference"] is not None:
        print(f"[Prediction] Baseline reference: {prediction['baseline_reference']['run_id']}")
    print(f"[Prediction] Pareto frontier: {prediction['pareto_frontier_frequencies_mhz']}")
    print(f"[Prediction] Recommended frequencies: {prediction['recommended_frequencies_mhz']}")
    print(f"[Prediction] Default recommendation: {prediction['supported_sweet_spot']['frequency_mhz']} MHz")


if __name__ == "__main__":
    main()
