from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from analysis.freq_model.features import DerivedModelFeatures
from analysis.freq_model.hardware import HardwareFeatures


@dataclass(frozen=True)
class CalibrationParams:
    compute_limit_at_max_tokens_per_s: float
    memory_limit_tokens_per_s: float
    communication_limit_tokens_per_s: float
    communication_penalty: float
    static_power_w: float
    dynamic_power_w: float
    dynamic_power_exponent: float
    throughput_saturation_ratio: float = 1.0
    correction_split_ratio: float = 0.72
    correction_transition_width: float = 0.06
    throughput_low_freq_correction: float = 0.0
    throughput_high_freq_correction: float = 0.0
    power_low_freq_correction: float = 0.0
    power_high_freq_correction: float = 0.0
    correction_topology_weight: float = 1.0
    correction_communication_weight: float = 0.5
    reference_min_frequency_ratio: float = 0.0
    reference_max_frequency_ratio: float = 1.0
    reference_observed_frequency_ratios: tuple[float, ...] = ()
    reference_pipeline_exposed_fraction: float = 0.0
    reference_dp_overlapable_fraction: float = 0.0
    reference_tp_sync_fraction: float = 0.0
    reference_topology_features_present: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionPoint:
    frequency_mhz: int
    frequency_ratio: float
    throughput_tokens_per_s: float
    throughput_samples_per_s: float
    power_w: float
    step_time_s: float
    tokens_per_j: float
    tokens_per_wh: float
    samples_per_wh: float
    component_limits: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _harmonic_blend(limits: Iterable[float], weights: Iterable[float]) -> float:
    denominator = 0.0
    for limit, weight in zip(limits, weights):
        denominator += weight / max(limit, 1e-9)
    return 1.0 / max(denominator, 1e-9)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def infer_initial_anchors(
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    observed_max_throughput_tokens_s: float,
) -> Dict[str, Optional[float]]:
    compute_anchor = None
    if hardware.total_peak_fp16_tensor_tflops is not None and features.approx_flops_per_token > 0:
        compute_anchor = (
            hardware.total_peak_fp16_tensor_tflops * 1_000_000_000_000.0
        ) / features.approx_flops_per_token

    memory_anchor = None
    if hardware.total_memory_bandwidth_gbps is not None and features.tokens_per_step > 0:
        bytes_per_token = features.approx_memory_bytes_per_step / max(features.tokens_per_step, 1e-9)
        memory_anchor = (
            hardware.total_memory_bandwidth_gbps * 1_000_000_000.0
        ) / max(bytes_per_token, 1e-9)

    if compute_anchor is None:
        compute_anchor = observed_max_throughput_tokens_s * 1.2
    if memory_anchor is None:
        memory_anchor = observed_max_throughput_tokens_s * 1.1

    communication_anchor = max(observed_max_throughput_tokens_s * 1.05, min(compute_anchor, memory_anchor))

    return {
        "compute_anchor": compute_anchor,
        "memory_anchor": memory_anchor,
        "communication_anchor": communication_anchor,
    }


def _base_compute_frequency_ratio(frequency_ratio: float, params: CalibrationParams) -> float:
    saturation_ratio = _clamp(params.throughput_saturation_ratio, 0.05, 1.0)
    return _clamp(frequency_ratio / saturation_ratio, 0.05, 1.0)


def _throughput_frequency_elasticity(features: DerivedModelFeatures) -> float:
    return _clamp(
        0.55
        + (0.35 * features.tp_sync_fraction)
        + (0.25 * features.pipeline_exposed_fraction)
        - (0.10 * features.dp_overlapable_fraction),
        0.55,
        1.0,
    )


def _low_frequency_extrapolation_blend(
    frequency_ratio: float,
    params: CalibrationParams,
) -> float:
    reference_min = _clamp(params.reference_min_frequency_ratio, 0.0, 1.0)
    if reference_min <= 0.0 or frequency_ratio >= reference_min:
        return 0.0
    return _clamp((reference_min - frequency_ratio) / max(reference_min, 1e-9), 0.0, 1.0)


def _reference_observed_ratios(params: CalibrationParams) -> list[float]:
    return sorted(
        _clamp(float(ratio), 0.0, 1.0)
        for ratio in params.reference_observed_frequency_ratios
        if ratio is not None
    )


def _primary_large_gap_bounds(params: CalibrationParams) -> tuple[float, float] | None:
    observed = _reference_observed_ratios(params)
    if len(observed) < 2:
        return None
    gaps = [(left, right, right - left) for left, right in zip(observed, observed[1:])]
    positive_gaps = [gap for _, _, gap in gaps if gap > 0.0]
    if not positive_gaps:
        return None
    median_gap = sorted(positive_gaps)[len(positive_gaps) // 2]
    large_gap_threshold = max(0.08, 3.0 * median_gap)
    left, right, gap = max(gaps, key=lambda item: item[2])
    if gap < large_gap_threshold:
        return None
    return left, right


def _has_reference_topology_features(params: CalibrationParams) -> bool:
    return bool(params.reference_topology_features_present)


def _transfer_topology_distance(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    if not _has_reference_topology_features(params):
        return 0.0
    return _clamp(
        (0.60 * abs(features.tp_sync_fraction - params.reference_tp_sync_fraction))
        + (0.25 * abs(features.pipeline_exposed_fraction - params.reference_pipeline_exposed_fraction))
        + (0.15 * abs(features.dp_overlapable_fraction - params.reference_dp_overlapable_fraction)),
        0.0,
        1.0,
    )


def _in_gap_frequency_regularization_blend(
    frequency_ratio: float,
    params: CalibrationParams,
) -> float:
    bounds = _primary_large_gap_bounds(params)
    if bounds is None:
        return 0.0
    left, right = bounds
    if not (left < frequency_ratio < right):
        return 0.0
    midpoint = 0.5 * (left + right)
    half_gap = max(0.5 * (right - left), 1e-9)
    distance_to_mid = abs(frequency_ratio - midpoint)
    trust = _clamp(distance_to_mid / half_gap, 0.0, 1.0)
    return 1.0 - trust


def _low_band_transfer_regularization_blend(
    frequency_ratio: float,
    params: CalibrationParams,
    features: DerivedModelFeatures,
) -> float:
    bounds = _primary_large_gap_bounds(params)
    if bounds is None:
        return 0.0
    low_edge, high_edge = bounds
    if frequency_ratio >= high_edge:
        return 0.0
    band_position = _clamp((high_edge - frequency_ratio) / max(high_edge - low_edge, 1e-9), 0.0, 1.0)
    topology_mismatch = _clamp(1.8 * _transfer_topology_distance(features, params), 0.0, 1.0)
    return band_position * topology_mismatch


def _compute_frequency_ratio(
    frequency_ratio: float,
    params: CalibrationParams,
    features: DerivedModelFeatures,
) -> float:
    base_ratio = _base_compute_frequency_ratio(frequency_ratio, params)
    elasticity = _throughput_frequency_elasticity(features)
    elastic_ratio = 1.0 - (elasticity * (1.0 - base_ratio))
    extrapolation_blend = _low_frequency_extrapolation_blend(frequency_ratio, params)
    in_gap_blend = _in_gap_frequency_regularization_blend(frequency_ratio, params)
    low_band_blend = _low_band_transfer_regularization_blend(frequency_ratio, params, features)
    regularization_blend = max(extrapolation_blend, in_gap_blend, low_band_blend)
    return ((1.0 - regularization_blend) * elastic_ratio) + (regularization_blend * base_ratio)


def _exposed_communication_share(features: DerivedModelFeatures) -> float:
    exposure_mix = _clamp(
        (0.60 * features.pipeline_exposed_fraction)
        + (0.30 * features.tp_sync_fraction)
        + (0.10 * features.dp_overlapable_fraction),
        0.0,
        1.0,
    )
    return features.communication_share * exposure_mix


def _low_freq_topology_mix(features: DerivedModelFeatures) -> float:
    return _clamp(
        (0.85 * features.pipeline_exposed_fraction)
        + (0.10 * features.tp_sync_fraction)
        + (0.05 * features.dp_overlapable_fraction),
        0.0,
        1.0,
    )


def _high_freq_topology_mix(features: DerivedModelFeatures) -> float:
    return _clamp(
        (0.45 * features.pipeline_exposed_fraction)
        + (0.40 * features.tp_sync_fraction)
        + (0.15 * features.dp_overlapable_fraction),
        0.0,
        1.0,
    )


def _pipeline_low_freq_gate(features: DerivedModelFeatures) -> float:
    return _clamp(
        0.15 + (1.70 * features.pipeline_exposed_fraction),
        0.15,
        1.0,
    )


def _low_freq_correction_intensity(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    pipeline_gate = _pipeline_low_freq_gate(features)
    return _clamp(
        1.0
        + (pipeline_gate * params.correction_topology_weight * _low_freq_topology_mix(features))
        + (pipeline_gate * 0.35 * params.correction_communication_weight * _exposed_communication_share(features)),
        1.0,
        2.0,
    )


def _high_freq_correction_intensity(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    return _clamp(
        1.0
        + (params.correction_topology_weight * _high_freq_topology_mix(features))
        + (params.correction_communication_weight * _exposed_communication_share(features)),
        1.0,
        2.5,
    )


def _power_frequency_retention_bias(features: DerivedModelFeatures) -> float:
    return _clamp(
        (1.0 - features.tp_sync_fraction)
        * (
            0.10
            + (0.25 * features.pipeline_exposed_fraction)
            + (0.15 * features.dp_overlapable_fraction)
        ),
        0.0,
        0.25,
    )


def _correction_positions(frequency_ratio: float, params: CalibrationParams) -> tuple[float, float, float, float]:
    split_ratio = _clamp(params.correction_split_ratio, 0.55, 0.90)
    transition_width = _clamp(params.correction_transition_width, 0.01, 0.20)
    low_position = _clamp((split_ratio - frequency_ratio) / max(split_ratio, 1e-9), 0.0, 1.0)
    high_position = _clamp((frequency_ratio - split_ratio) / max(1.0 - split_ratio, 1e-9), 0.0, 1.0)
    gate_high = _sigmoid((frequency_ratio - split_ratio) / transition_width)
    gate_low = 1.0 - gate_high
    return low_position, high_position, gate_low, gate_high


def _throughput_correction_factor(
    frequency_ratio: float,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    low_position, high_position, gate_low, gate_high = _correction_positions(frequency_ratio, params)
    low_intensity = _low_freq_correction_intensity(features, params)
    high_intensity = _high_freq_correction_intensity(features, params)
    low_factor = 1.0 + (low_intensity * params.throughput_low_freq_correction * low_position)
    high_factor = 1.0 + (high_intensity * params.throughput_high_freq_correction * (high_position ** 2))
    factor = (gate_low * low_factor) + (gate_high * high_factor)
    return _clamp(factor, 0.75, 1.25)


def _power_correction_factor(
    frequency_ratio: float,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    low_position, high_position, gate_low, gate_high = _correction_positions(frequency_ratio, params)
    low_intensity = _low_freq_correction_intensity(features, params)
    high_intensity = _high_freq_correction_intensity(features, params)
    low_factor = 1.0 + (low_intensity * params.power_low_freq_correction * low_position)
    high_factor = 1.0 + (high_intensity * params.power_high_freq_correction * (high_position ** 2))
    factor = (gate_low * low_factor) + (gate_high * high_factor)
    factor *= 1.0 + (_power_frequency_retention_bias(features) * (1.0 - frequency_ratio))
    return _clamp(factor, 0.85, 1.20)


def predict_throughput_tokens_per_s(
    frequency_mhz: float,
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    max_frequency = float(hardware.max_frequency_mhz or frequency_mhz or 1.0)
    frequency_ratio = _clamp(float(frequency_mhz) / max_frequency, 0.05, 1.0)
    compute_frequency_ratio = _compute_frequency_ratio(frequency_ratio, params, features)
    compute_limit = params.compute_limit_at_max_tokens_per_s * compute_frequency_ratio
    memory_limit = params.memory_limit_tokens_per_s
    exposed_communication_share = _exposed_communication_share(features)
    communication_limit = params.communication_limit_tokens_per_s / (
        1.0 + params.communication_penalty * exposed_communication_share * ((1.0 / frequency_ratio) - 1.0)
    )
    throughput = _harmonic_blend(
        [compute_limit, memory_limit, communication_limit],
        [features.compute_weight, features.memory_weight, features.communication_weight],
    )
    throughput *= max(features.pipeline_parallel_efficiency, 1e-9)
    throughput *= _throughput_correction_factor(frequency_ratio, features, params)
    return throughput


def predict_power_w(
    frequency_mhz: float,
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    max_frequency = float(hardware.max_frequency_mhz or frequency_mhz or 1.0)
    frequency_ratio = _clamp(float(frequency_mhz) / max_frequency, 0.05, 1.0)
    compute_frequency_ratio = _compute_frequency_ratio(frequency_ratio, params, features)
    compute_limit = max(params.compute_limit_at_max_tokens_per_s * compute_frequency_ratio, 1e-9)
    throughput = predict_throughput_tokens_per_s(frequency_mhz, hardware, features, params)
    utilization = _clamp(throughput / compute_limit, 0.05, 1.2)
    dynamic = params.dynamic_power_w * utilization * (frequency_ratio ** params.dynamic_power_exponent)
    power_w = params.static_power_w + dynamic
    power_w *= _power_correction_factor(frequency_ratio, features, params)
    return power_w


def predict_point(
    frequency_mhz: int,
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> PredictionPoint:
    max_frequency = float(hardware.max_frequency_mhz or frequency_mhz or 1.0)
    frequency_ratio = _clamp(float(frequency_mhz) / max_frequency, 0.05, 1.0)
    compute_frequency_ratio = _compute_frequency_ratio(frequency_ratio, params, features)
    throughput_tokens_per_s = predict_throughput_tokens_per_s(frequency_mhz, hardware, features, params)
    throughput_samples_per_s = throughput_tokens_per_s / max(features.tokens_per_step / max(features.samples_per_step, 1e-9), 1e-9)
    power_w = predict_power_w(frequency_mhz, hardware, features, params)
    step_time_s = features.tokens_per_step / max(throughput_tokens_per_s, 1e-9)
    tokens_per_j = throughput_tokens_per_s / max(power_w, 1e-9)
    tokens_per_wh = tokens_per_j * 3600.0
    samples_per_wh = throughput_samples_per_s * 3600.0 / max(power_w, 1e-9)
    exposed_communication_share = _exposed_communication_share(features)
    component_limits = {
        "compute_tokens_per_s": params.compute_limit_at_max_tokens_per_s * compute_frequency_ratio,
        "memory_tokens_per_s": params.memory_limit_tokens_per_s,
        "communication_tokens_per_s": params.communication_limit_tokens_per_s / (
            1.0 + params.communication_penalty * exposed_communication_share * ((1.0 / frequency_ratio) - 1.0)
        ),
    }
    return PredictionPoint(
        frequency_mhz=int(frequency_mhz),
        frequency_ratio=frequency_ratio,
        throughput_tokens_per_s=throughput_tokens_per_s,
        throughput_samples_per_s=throughput_samples_per_s,
        power_w=power_w,
        step_time_s=step_time_s,
        tokens_per_j=tokens_per_j,
        tokens_per_wh=tokens_per_wh,
        samples_per_wh=samples_per_wh,
        component_limits=component_limits,
    )


def sweep_prediction_points(
    frequencies_mhz: Iterable[int],
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> List[PredictionPoint]:
    return [predict_point(freq, hardware, features, params) for freq in frequencies_mhz]


def select_metric_value(point: PredictionPoint, metric: str) -> float:
    if metric == "tokens_per_wh":
        return point.tokens_per_wh
    if metric == "samples_per_wh":
        return point.samples_per_wh
    return point.tokens_per_j


def build_continuous_grid(hardware: HardwareFeatures, step_mhz: int = 5) -> List[int]:
    min_frequency = int(hardware.min_frequency_mhz or 0)
    max_frequency = int(hardware.max_frequency_mhz or min_frequency)
    if min_frequency <= 0 or max_frequency <= 0:
        return []
    return list(range(min_frequency, max_frequency + 1, max(1, step_mhz)))
