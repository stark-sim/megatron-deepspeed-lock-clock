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
from analysis.freq_model.model import predict_point
from analysis.freq_model.network import apply_network_quality, load_network_quality_observation
from analysis.freq_model.recommend import build_prediction_bundle
from analysis.freq_model.workload import LoadedRunSample, load_experiment_samples


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate source->target transfer prediction accuracy from experiment artifacts")
    parser.add_argument("--source-root", required=True, help="Directory containing source static-frequency run folders")
    parser.add_argument("--target-root", required=True, help="Directory containing target static-frequency run folders")
    parser.add_argument("--output-dir", default=None, help="Directory for transfer evaluation outputs")
    parser.add_argument("--network-benchmark-json", default=None, help="Optional network benchmark JSON used to annotate multi-node target features")
    parser.add_argument("--peak-fp16-tflops-per-gpu", type=float, default=None)
    parser.add_argument("--memory-bandwidth-gbps-per-gpu", type=float, default=None)
    parser.add_argument("--power-limit-w-per-gpu", type=float, default=None)
    return parser.parse_args()


def _non_topology_signature(sample: LoadedRunSample) -> tuple:
    workload = sample.workload
    return (
        workload.num_layers,
        workload.hidden_size,
        workload.ffn_hidden_size,
        workload.num_attention_heads,
        workload.num_key_value_heads,
        workload.seq_length,
        workload.micro_batch_size,
        workload.global_batch_size,
        workload.train_iters,
        workload.zero_stage,
        workload.precision_mode,
        workload.swiglu,
    )


def _validate_samples(samples: List[LoadedRunSample], label: str) -> None:
    if not samples:
        raise SystemExit(f"no usable samples found under {label}")
    signatures = {_non_topology_signature(sample) for sample in samples}
    if len(signatures) > 1:
        raise SystemExit(f"{label} contains mismatched non-topology workloads")


def _validate_source_target_compatibility(source_samples: List[LoadedRunSample], target_samples: List[LoadedRunSample]) -> None:
    if _non_topology_signature(source_samples[0]) != _non_topology_signature(target_samples[0]):
        raise SystemExit("source and target workloads differ outside topology fields")


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _ape(observed: float, predicted: float) -> float:
    return abs(observed - predicted) / max(abs(observed), 1e-9)


def _default_output_dir(source_root: Path, target_root: Path) -> Path:
    name = f"transfer_eval_{source_root.name}_to_{target_root.name}"
    return REPO_ROOT / ".context" / name


def _load_static_samples(root: Path) -> List[LoadedRunSample]:
    collection = load_experiment_samples(str(root), include_baseline=False)
    return sorted(
        [sample for sample in collection.samples if sample.observed.frequency_mhz is not None],
        key=lambda sample: int(sample.observed.frequency_mhz or 0),
    )


def _evaluate_target_points(
    target_samples: List[LoadedRunSample],
    hardware,
    calibration_params,
    network_observation,
) -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []

    for sample in target_samples:
        observed = sample.observed
        if observed.frequency_mhz is None:
            continue
        features = derive_model_features(hardware, sample.workload)
        if network_observation is not None and sample.workload.node_count > 1:
            features = apply_network_quality(features, network_observation)

        prediction = predict_point(int(observed.frequency_mhz), hardware, features, calibration_params)
        interval_tokens = observed.interval_tokens
        if interval_tokens is None:
            interval_tokens = features.tokens_per_step * max(observed.num_steps, 1)
        predicted_total_time_s = float(interval_tokens) / max(prediction.throughput_tokens_per_s, 1e-9)
        predicted_total_energy_j = prediction.power_w * predicted_total_time_s

        point = {
            "run_id": sample.run_id,
            "frequency_mhz": int(observed.frequency_mhz),
            "observed_total_time_s": observed.time_s,
            "predicted_total_time_s": predicted_total_time_s,
            "observed_total_energy_j": observed.energy_j,
            "predicted_total_energy_j": predicted_total_energy_j,
            "observed_avg_power_w": observed.avg_power_w,
            "predicted_avg_power_w": prediction.power_w,
            "observed_step_time_s": observed.step_time_s,
            "predicted_step_time_s": predicted_total_time_s / max(observed.num_steps, 1),
            "observed_steps": observed.num_steps,
            "observed_interval_tokens": interval_tokens,
        }
        point["total_time_ape"] = _ape(point["observed_total_time_s"], point["predicted_total_time_s"])
        point["total_energy_ape"] = _ape(point["observed_total_energy_j"], point["predicted_total_energy_j"])
        point["avg_power_ape"] = _ape(point["observed_avg_power_w"], point["predicted_avg_power_w"])
        point["step_time_ape"] = _ape(point["observed_step_time_s"], point["predicted_step_time_s"])
        points.append(point)

    return {
        "point_count": len(points),
        "total_time_mape": _mean([point["total_time_ape"] for point in points]),
        "total_energy_mape": _mean([point["total_energy_ape"] for point in points]),
        "avg_power_mape": _mean([point["avg_power_ape"] for point in points]),
        "step_time_mape": _mean([point["step_time_ape"] for point in points]),
        "points": points,
    }


def _build_report(payload: Dict[str, Any]) -> str:
    lines = [
        "# Transfer Prediction Evaluation",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- source_root: `{payload['source_root']}`",
        f"- target_root: `{payload['target_root']}`",
        f"- network_benchmark_json: `{payload['network_benchmark_json']}`",
        "",
        "## Source Calibration",
        f"- sample_count: `{payload['source_sample_count']}`",
        f"- throughput_mape: `{payload['source_calibration']['throughput_mape']:.4f}`",
        f"- power_mape: `{payload['source_calibration']['power_mape']:.4f}`",
        f"- total_time_mape: `{payload['source_calibration']['total_time_mape']:.4f}`",
        f"- total_energy_mape: `{payload['source_calibration']['total_energy_mape']:.4f}`",
        "",
        "## Target Transfer Accuracy",
        f"- point_count: `{payload['target_accuracy']['point_count']}`",
        f"- total_time_mape: `{payload['target_accuracy']['total_time_mape']:.4f}`",
        f"- step_time_mape: `{payload['target_accuracy']['step_time_mape']:.4f}`",
        f"- avg_power_mape: `{payload['target_accuracy']['avg_power_mape']:.4f}`",
        f"- total_energy_mape: `{payload['target_accuracy']['total_energy_mape']:.4f}`",
        "",
        "## Cross-Node Coefficients",
        f"- alpha_pp_s_per_byte: `{payload['cross_node']['alpha_pp_s_per_byte']:.6e}`",
        f"- alpha_dp_s_per_byte: `{payload['cross_node']['alpha_dp_s_per_byte']:.6e}`",
        f"- alpha_tp_s_per_byte: `{payload['cross_node']['alpha_tp_s_per_byte']:.6e}`",
        "",
        "## Target Points",
    ]

    for point in payload["target_accuracy"]["points"]:
        lines.append(
            "- "
            f"`{point['frequency_mhz']}` MHz: "
            f"time observed/predicted=`{point['observed_total_time_s']:.3f}/{point['predicted_total_time_s']:.3f}` s, "
            f"time APE=`{point['total_time_ape']:.4f}`, "
            f"step observed/predicted=`{point['observed_step_time_s']:.3f}/{point['predicted_step_time_s']:.3f}` s"
        )

    recommendation = payload.get("target_prediction_summary") or {}
    if recommendation:
        lines.extend(
            [
                "",
                "## Predicted Sweet Spot",
                f"- default_frequency_mhz: `{recommendation['default_frequency_mhz']}`",
                f"- recommended_frequencies_mhz: `{recommendation['recommended_frequencies_mhz']}`",
                f"- pareto_frontier_frequencies_mhz: `{recommendation['pareto_frontier_frequencies_mhz']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(source_root, target_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_samples = _load_static_samples(source_root)
    target_samples = _load_static_samples(target_root)
    _validate_samples(source_samples, str(source_root))
    _validate_samples(target_samples, str(target_root))
    _validate_source_target_compatibility(source_samples, target_samples)

    hardware = build_hardware_features(
        source_samples[0].run_payload,
        peak_fp16_tflops_per_gpu=args.peak_fp16_tflops_per_gpu,
        memory_bandwidth_gbps_per_gpu=args.memory_bandwidth_gbps_per_gpu,
        power_limit_w_per_gpu=args.power_limit_w_per_gpu,
    )

    source_features = [derive_model_features(hardware, sample.workload) for sample in source_samples]
    network_payload = None
    network_observation = None
    if args.network_benchmark_json:
        network_path = Path(args.network_benchmark_json).expanduser().resolve()
        network_payload = json.loads(network_path.read_text(encoding="utf-8"))
        network_observation = load_network_quality_observation(network_path)

    calibration = calibrate_frequency_model(
        source_samples,
        hardware,
        source_features,
        network_bench_result=network_payload,
    )

    target_accuracy = _evaluate_target_points(
        target_samples,
        hardware,
        calibration.params,
        network_observation,
    )

    reference_target_features = derive_model_features(hardware, target_samples[0].workload)
    if network_observation is not None and target_samples[0].workload.node_count > 1:
        reference_target_features = apply_network_quality(reference_target_features, network_observation)
    target_prediction = build_prediction_bundle(
        hardware,
        reference_target_features,
        calibration.params,
        comparison_steps=max(target_samples[0].observed.num_steps, 1),
    )

    payload: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "network_benchmark_json": (
            str(Path(args.network_benchmark_json).expanduser().resolve())
            if args.network_benchmark_json
            else None
        ),
        "network_observation": network_observation.to_dict() if network_observation is not None else None,
        "hardware": hardware.to_dict(),
        "source_sample_count": len(source_samples),
        "target_sample_count": len(target_samples),
        "source_calibration": {
            "throughput_mape": calibration.throughput_mape,
            "power_mape": calibration.power_mape,
            "total_time_mape": calibration.total_time_mape,
            "total_energy_mape": calibration.total_energy_mape,
            "runtime_ratio_mape": calibration.runtime_ratio_mape,
            "energy_ratio_mape": calibration.energy_ratio_mape,
            "objective": calibration.objective,
        },
        "cross_node": {
            "alpha_pp_s_per_byte": calibration.params.cross_node_alpha_pp_s_per_byte,
            "alpha_dp_s_per_byte": calibration.params.cross_node_alpha_dp_s_per_byte,
            "alpha_tp_s_per_byte": calibration.params.cross_node_alpha_tp_s_per_byte,
        },
        "target_accuracy": target_accuracy,
        "target_prediction_summary": {
            "default_frequency_mhz": target_prediction["supported_sweet_spot"]["frequency_mhz"],
            "recommended_frequencies_mhz": target_prediction["recommended_frequencies_mhz"],
            "pareto_frontier_frequencies_mhz": target_prediction["pareto_frontier_frequencies_mhz"],
        },
    }

    json_path = output_dir / "transfer_prediction.json"
    report_path = output_dir / "transfer_prediction_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_build_report(payload), encoding="utf-8")

    print(f"[TransferEval] Source samples: {len(source_samples)}")
    print(f"[TransferEval] Target samples: {len(target_samples)}")
    print(f"[TransferEval] Target total_time_mape: {target_accuracy['total_time_mape']:.4f}")
    print(f"[TransferEval] Output JSON: {json_path}")
    print(f"[TransferEval] Output report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
