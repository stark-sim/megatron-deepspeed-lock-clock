from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Sequence

from analysis.freq_model.features import DerivedModelFeatures
from analysis.freq_model.hardware import HardwareFeatures
from analysis.freq_model.model import CalibrationParams, PredictionPoint, build_continuous_grid, predict_point, select_metric_value
from analysis.freq_model.workload import LoadedRunSample


def _estimate_total_time_s(comparison_tokens: float, throughput_tokens_per_s: float) -> float:
    return comparison_tokens / max(throughput_tokens_per_s, 1e-9)


def _build_baseline_reference(
    baseline_sample: LoadedRunSample | None,
    features: DerivedModelFeatures,
    comparison_steps: int,
) -> Dict[str, Any] | None:
    if baseline_sample is None:
        return None

    observed = baseline_sample.observed
    observed_steps = max(observed.num_steps, 1)
    step_time_s = observed.time_s / observed_steps
    total_time_s = step_time_s * comparison_steps
    total_energy_j = observed.avg_power_w * total_time_s
    tokens_per_j = (observed.tokens_per_j or 0.0)
    if tokens_per_j <= 0:
        baseline_tokens = observed.interval_tokens or (features.tokens_per_step * observed_steps)
        tokens_per_j = float(baseline_tokens) / max(observed.energy_j, 1e-9)

    return {
        "run_id": baseline_sample.run_id,
        "comparison_steps": comparison_steps,
        "comparison_tokens": features.tokens_per_step * comparison_steps,
        "total_time_s": total_time_s,
        "total_energy_j": total_energy_j,
        "avg_power_w": observed.avg_power_w,
        "throughput_tokens_per_s": observed.throughput_tokens_per_s,
        "tokens_per_j": tokens_per_j,
    }


def _build_observed_overlay(samples: Sequence[LoadedRunSample] | None) -> List[Dict[str, float]]:
    if not samples:
        return []
    buckets: Dict[int, Dict[str, float]] = {}
    for sample in samples:
        observed = sample.observed
        if observed.frequency_mhz is None or observed.throughput_tokens_per_s is None:
            continue
        freq = int(observed.frequency_mhz)
        bucket = buckets.setdefault(
            freq,
            {"frequency_mhz": float(freq), "throughput_sum": 0.0, "power_sum": 0.0, "count": 0.0},
        )
        bucket["throughput_sum"] += float(observed.throughput_tokens_per_s)
        bucket["power_sum"] += float(observed.avg_power_w)
        bucket["count"] += 1.0
    points = []
    for freq in sorted(buckets):
        bucket = buckets[freq]
        count = max(bucket["count"], 1.0)
        points.append(
            {
                "frequency_mhz": float(freq),
                "throughput_tokens_per_s": bucket["throughput_sum"] / count,
                "power_w": bucket["power_sum"] / count,
            }
        )
    return points


def _interpolate_overlay_value(observed_overlay: Sequence[Dict[str, float]], frequency_mhz: int, key: str) -> float | None:
    if len(observed_overlay) < 2:
        return None
    target = float(frequency_mhz)
    if target < observed_overlay[0]["frequency_mhz"] or target > observed_overlay[-1]["frequency_mhz"]:
        return None
    for left, right in zip(observed_overlay, observed_overlay[1:]):
        left_freq = left["frequency_mhz"]
        right_freq = right["frequency_mhz"]
        if target == left_freq:
            return left[key]
        if target == right_freq:
            return right[key]
        if left_freq <= target <= right_freq:
            span = max(right_freq - left_freq, 1e-9)
            ratio = (target - left_freq) / span
            return left[key] + ratio * (right[key] - left[key])
    return None


def _predict_point_with_overlay(
    frequency_mhz: int,
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
    observed_overlay: Sequence[Dict[str, float]],
) -> PredictionPoint:
    analytic_point = predict_point(frequency_mhz, hardware, features, params)
    throughput_tokens_per_s = _interpolate_overlay_value(observed_overlay, frequency_mhz, "throughput_tokens_per_s")
    power_w = _interpolate_overlay_value(observed_overlay, frequency_mhz, "power_w")
    if throughput_tokens_per_s is None or power_w is None:
        return analytic_point

    throughput_samples_per_s = throughput_tokens_per_s / max(features.tokens_per_step / max(features.samples_per_step, 1e-9), 1e-9)
    step_time_s = features.tokens_per_step / max(throughput_tokens_per_s, 1e-9)
    tokens_per_j = throughput_tokens_per_s / max(power_w, 1e-9)
    tokens_per_wh = tokens_per_j * 3600.0
    samples_per_wh = throughput_samples_per_s * 3600.0 / max(power_w, 1e-9)
    return replace(
        analytic_point,
        throughput_tokens_per_s=throughput_tokens_per_s,
        throughput_samples_per_s=throughput_samples_per_s,
        power_w=power_w,
        step_time_s=step_time_s,
        tokens_per_j=tokens_per_j,
        tokens_per_wh=tokens_per_wh,
        samples_per_wh=samples_per_wh,
    )


def _sweep_prediction_points(
    frequencies_mhz: Sequence[int],
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
    observed_overlay: Sequence[Dict[str, float]],
) -> List[PredictionPoint]:
    return [
        _predict_point_with_overlay(freq, hardware, features, params, observed_overlay)
        if observed_overlay
        else predict_point(freq, hardware, features, params)
        for freq in frequencies_mhz
    ]


def _annotate_point(
    point: PredictionPoint,
    metric: str,
    comparison_tokens: float,
    baseline_reference: Dict[str, Any] | None,
) -> Dict[str, Any]:
    estimated_total_time_s = _estimate_total_time_s(comparison_tokens, point.throughput_tokens_per_s)
    estimated_total_energy_j = point.power_w * estimated_total_time_s
    estimated_total_energy_wh = estimated_total_energy_j / 3600.0
    objective_metric_value = select_metric_value(point, metric)

    annotated = {
        **point.to_dict(),
        "objective_metric_value": objective_metric_value,
        "estimated_total_time_s": estimated_total_time_s,
        "estimated_total_energy_j": estimated_total_energy_j,
        "estimated_total_energy_wh": estimated_total_energy_wh,
    }
    if baseline_reference is not None:
        annotated.update(
            {
                "runtime_ratio_vs_baseline": estimated_total_time_s / max(baseline_reference["total_time_s"], 1e-9),
                "energy_ratio_vs_baseline": estimated_total_energy_j / max(baseline_reference["total_energy_j"], 1e-9),
                "power_ratio_vs_baseline": point.power_w / max(baseline_reference["avg_power_w"], 1e-9),
                "tokens_per_j_ratio_vs_baseline": point.tokens_per_j / max(baseline_reference["tokens_per_j"], 1e-9),
            }
        )
    return annotated


def _pareto_coords(point: Dict[str, Any], baseline_reference: Dict[str, Any] | None) -> tuple[float, float]:
    if baseline_reference is not None:
        return point["runtime_ratio_vs_baseline"], point["energy_ratio_vs_baseline"]
    return point["estimated_total_time_s"], point["estimated_total_energy_j"]


def _is_dominated(candidate: Dict[str, Any], others: Sequence[Dict[str, Any]], baseline_reference: Dict[str, Any] | None) -> bool:
    cand_x, cand_y = _pareto_coords(candidate, baseline_reference)
    for other in others:
        if other is candidate:
            continue
        other_x, other_y = _pareto_coords(other, baseline_reference)
        if other_x <= cand_x and other_y <= cand_y and (other_x < cand_x or other_y < cand_y):
            return True
    return False


def _build_pareto_frontier(predictions: List[Dict[str, Any]], baseline_reference: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    frontier = [point for point in predictions if not _is_dominated(point, predictions, baseline_reference)]
    return sorted(frontier, key=lambda point: _pareto_coords(point, baseline_reference)[0])


def _balanced_sweet_spot_score(point: Dict[str, Any], baseline_reference: Dict[str, Any] | None) -> float:
    if baseline_reference is not None:
        runtime_ratio = point["runtime_ratio_vs_baseline"]
        energy_ratio = point["energy_ratio_vs_baseline"]
        return math.sqrt((runtime_ratio - 1.0) ** 2 + (energy_ratio - 1.0) ** 2)
    return -point["objective_metric_value"]


def _energy_then_time_key(point: Dict[str, Any], baseline_reference: Dict[str, Any] | None) -> tuple[float, float]:
    if baseline_reference is not None:
        return point["energy_ratio_vs_baseline"], point["runtime_ratio_vs_baseline"]
    return point["estimated_total_energy_j"], point["estimated_total_time_s"]


def _nearest_frequency_index(supported_frequencies: List[int], target_frequency: int) -> int:
    return min(range(len(supported_frequencies)), key=lambda index: abs(supported_frequencies[index] - target_frequency))


def build_prediction_bundle(
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
    metric: str = "tokens_per_j",
    neighborhood: int = 1,
    comparison_steps: int | None = None,
    baseline_sample: LoadedRunSample | None = None,
    observed_samples: Sequence[LoadedRunSample] | None = None,
    use_observed_overlay: bool = False,
) -> Dict[str, Any]:
    continuous_grid = build_continuous_grid(hardware)
    if not continuous_grid:
        raise ValueError("hardware must provide a non-empty frequency range")

    supported_frequencies = hardware.supported_frequency_mhz or continuous_grid
    effective_steps = max(int(comparison_steps or 0), 1)
    comparison_tokens = features.tokens_per_step * effective_steps
    baseline_reference = _build_baseline_reference(baseline_sample, features, effective_steps)
    observed_overlay = _build_observed_overlay(observed_samples) if use_observed_overlay else []

    continuous_predictions = _sweep_prediction_points(continuous_grid, hardware, features, params, observed_overlay)
    supported_predictions = _sweep_prediction_points(supported_frequencies, hardware, features, params, observed_overlay)

    annotated_continuous = [
        _annotate_point(point, metric=metric, comparison_tokens=comparison_tokens, baseline_reference=baseline_reference)
        for point in continuous_predictions
    ]
    annotated_supported = [
        _annotate_point(point, metric=metric, comparison_tokens=comparison_tokens, baseline_reference=baseline_reference)
        for point in supported_predictions
    ]

    continuous_frontier = _build_pareto_frontier(annotated_continuous, baseline_reference)
    supported_frontier = _build_pareto_frontier(annotated_supported, baseline_reference)
    sweet_spot_continuous = min(continuous_frontier, key=lambda point: _energy_then_time_key(point, baseline_reference))
    sweet_spot_supported = min(supported_frontier, key=lambda point: _energy_then_time_key(point, baseline_reference))
    balanced_sweet_spot_continuous = min(continuous_frontier, key=lambda point: _balanced_sweet_spot_score(point, baseline_reference))
    balanced_sweet_spot_supported = min(supported_frontier, key=lambda point: _balanced_sweet_spot_score(point, baseline_reference))

    best_index = _nearest_frequency_index(supported_frequencies, int(sweet_spot_supported["frequency_mhz"]))
    desired_width = max(1, (2 * max(neighborhood, 0)) + 1)
    lower = max(0, best_index - max(neighborhood, 0))
    upper = min(len(supported_frequencies), best_index + max(neighborhood, 0) + 1)
    while (upper - lower) < desired_width and lower > 0:
        lower -= 1
    while (upper - lower) < desired_width and upper < len(supported_frequencies):
        upper += 1

    top_supported_predictions = sorted(
        annotated_supported,
        key=lambda point: (
            point["frequency_mhz"] not in {frontier_point["frequency_mhz"] for frontier_point in supported_frontier},
            _energy_then_time_key(point, baseline_reference),
        ),
    )

    return {
        "metric": metric,
        "objective": {
            "mode": "baseline_relative_energy_then_time_tradeoff" if baseline_reference is not None else "absolute_energy_then_time_tradeoff",
            "comparison_steps": effective_steps,
            "comparison_tokens": comparison_tokens,
            "primary": "total_energy",
            "secondary": "total_time",
        },
        "baseline_reference": baseline_reference,
        "overlay": {
            "enabled": use_observed_overlay,
            "uses_observed_interpolation": bool(observed_overlay),
            "observed_frequency_range_mhz": [
                int(observed_overlay[0]["frequency_mhz"]),
                int(observed_overlay[-1]["frequency_mhz"]),
            ] if observed_overlay else None,
            "observed_frequency_count": len(observed_overlay),
        },
        "continuous_sweet_spot": sweet_spot_continuous,
        "supported_sweet_spot": sweet_spot_supported,
        "continuous_balanced_sweet_spot": balanced_sweet_spot_continuous,
        "supported_balanced_sweet_spot": balanced_sweet_spot_supported,
        "pareto_frontier_frequencies_mhz": [point["frequency_mhz"] for point in supported_frontier],
        "pareto_frontier_predictions": supported_frontier,
        "recommended_frequencies_mhz": supported_frequencies[lower:upper],
        "supported_predictions": annotated_supported,
        "top_supported_predictions": top_supported_predictions[:5],
    }
