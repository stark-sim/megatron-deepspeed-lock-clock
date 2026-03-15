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


def _compute_frequency_ratio(frequency_ratio: float, params: CalibrationParams) -> float:
    saturation_ratio = _clamp(params.throughput_saturation_ratio, 0.05, 1.0)
    return _clamp(frequency_ratio / saturation_ratio, 0.05, 1.0)


def _correction_intensity(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    bubble_exposure = max(1.0 - features.pipeline_parallel_efficiency, 0.0)
    return 1.0 + (params.correction_topology_weight * bubble_exposure) + (
        params.correction_communication_weight * features.communication_share
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
    intensity = _correction_intensity(features, params)
    low_factor = 1.0 + (intensity * params.throughput_low_freq_correction * low_position)
    high_factor = 1.0 + (intensity * params.throughput_high_freq_correction * (high_position ** 2))
    factor = (gate_low * low_factor) + (gate_high * high_factor)
    return _clamp(factor, 0.75, 1.25)


def _power_correction_factor(
    frequency_ratio: float,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    low_position, high_position, gate_low, gate_high = _correction_positions(frequency_ratio, params)
    intensity = _correction_intensity(features, params)
    low_factor = 1.0 + (intensity * params.power_low_freq_correction * low_position)
    high_factor = 1.0 + (intensity * params.power_high_freq_correction * (high_position ** 2))
    factor = (gate_low * low_factor) + (gate_high * high_factor)
    return _clamp(factor, 0.85, 1.20)


def predict_throughput_tokens_per_s(
    frequency_mhz: float,
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    max_frequency = float(hardware.max_frequency_mhz or frequency_mhz or 1.0)
    frequency_ratio = _clamp(float(frequency_mhz) / max_frequency, 0.05, 1.0)
    compute_frequency_ratio = _compute_frequency_ratio(frequency_ratio, params)
    compute_limit = params.compute_limit_at_max_tokens_per_s * compute_frequency_ratio
    memory_limit = params.memory_limit_tokens_per_s
    communication_limit = params.communication_limit_tokens_per_s / (
        1.0 + params.communication_penalty * features.communication_share * ((1.0 / frequency_ratio) - 1.0)
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
    compute_frequency_ratio = _compute_frequency_ratio(frequency_ratio, params)
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
    compute_frequency_ratio = _compute_frequency_ratio(frequency_ratio, params)
    throughput_tokens_per_s = predict_throughput_tokens_per_s(frequency_mhz, hardware, features, params)
    throughput_samples_per_s = throughput_tokens_per_s / max(features.tokens_per_step / max(features.samples_per_step, 1e-9), 1e-9)
    power_w = predict_power_w(frequency_mhz, hardware, features, params)
    step_time_s = features.tokens_per_step / max(throughput_tokens_per_s, 1e-9)
    tokens_per_j = throughput_tokens_per_s / max(power_w, 1e-9)
    tokens_per_wh = tokens_per_j * 3600.0
    samples_per_wh = throughput_samples_per_s * 3600.0 / max(power_w, 1e-9)
    component_limits = {
        "compute_tokens_per_s": params.compute_limit_at_max_tokens_per_s * compute_frequency_ratio,
        "memory_tokens_per_s": params.memory_limit_tokens_per_s,
        "communication_tokens_per_s": params.communication_limit_tokens_per_s / (
            1.0 + params.communication_penalty * features.communication_share * ((1.0 / frequency_ratio) - 1.0)
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
