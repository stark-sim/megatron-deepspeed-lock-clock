from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from analysis.freq_model.features import DerivedModelFeatures
from analysis.freq_model.hardware import HardwareFeatures, HardwareFingerprint
from analysis.freq_model.network import NetworkConfig


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
    reference_total_gpu_count: int = 1
    reference_gpus_per_node: int = 1
    reference_pipeline_parallel_efficiency: float = 1.0
    reference_pipeline_exposed_fraction: float = 0.0
    reference_dp_overlapable_fraction: float = 0.0
    reference_tp_sync_fraction: float = 0.0
    reference_topology_dispersion: float = 0.0
    reference_topology_count: int = 1
    reference_multi_topology_calibration: bool = False
    reference_topology_features_present: bool = False
    cross_node_alpha_pp_s_per_byte: float = 0.0
    cross_node_alpha_dp_s_per_byte: float = 0.0
    cross_node_alpha_tp_s_per_byte: float = 0.0
    cross_node_reference_cross_node_pp_bytes: float = 0.0
    cross_node_reference_pp_cross_node_wait_pressure: float = 0.0
    cross_node_reference_cross_node_dp_bytes: float = 0.0
    cross_node_beta_pp_wait_s: float = 0.0
    cross_node_beta_pp_edge_s: float = 0.0
    cross_node_power_base_drop: float = 0.0
    cross_node_power_low_freq_reference_ratio: float = 0.782
    cross_node_power_low_freq_gamma: float = 0.0
    cross_node_reference_transport_label: str = "tailscale0|ib_disable=1"
    cross_node_reference_bandwidth_gbps: float = 0.2075
    cross_node_reference_jitter_cv: float = 0.136
    cross_node_benchmark_world_size: int = 0
    cross_node_benchmark_message_sizes_mb: tuple[float, ...] = ()
    cross_node_benchmark_avg_times_ms: tuple[float, ...] = ()
    cross_node_benchmark_bus_bandwidth_gbps: tuple[float, ...] = ()
    cross_node_use_benchmark_time_model: bool = False
    cross_node_bandwidth_sensitivity: float = 1.0
    cross_node_jitter_sensitivity: float = 0.5
    cross_node_pp_exposure_sensitivity: float = 1.0
    cross_node_dp_exposure_sensitivity: float = 1.0
    cross_node_tp_exposure_sensitivity: float = 1.0
    cross_node_dp_group_scale_gain: float = 0.5
    cross_node_pp_bubble_exposure_gain: float = 0.5
    base_topology_throughput_sensitivity: float = 0.0
    base_topology_communication_sensitivity: float = 0.0
    base_topology_power_sensitivity: float = 0.0
    base_topology_shape_sensitivity: float = 0.0
    base_topology_min_throughput_scale: float = 0.18
    base_topology_max_power_scale: float = 1.75
    thermal_throttle_threshold: float = 1.0
    thermal_throttle_coefficient: float = 0.0
    power_utilization_exponent: float = 1.0

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


# Transport-type → communication-penalty mapping for physics-driven derivation.
_TRANSPORT_PENALTY: dict[str, float] = {
    "nvlink": 0.01,
    "ib": 0.10,
    "ethernet": 0.75,
    "pcie": 0.50,
}


def derive_calibration_params(
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    network: NetworkConfig,
    fingerprint: HardwareFingerprint,
) -> CalibrationParams:
    """Derive CalibrationParams from hardware specs + workload features + network + fingerprint.

    This is the physics-driven derivation layer.  Instead of per-scenario hard-coded
    parameters, the throughput limits are computed from first principles:

      * compute_limit  → Roofline model (peak TFLOPS × efficiency × AI/hw_balance)
      * memory_limit   → Memory bandwidth × efficiency / bytes_per_token
      * comm_limit     → tokens_per_step / (exposed_comm_bytes / net_bw / efficiency)

    The power-model parameters (static_power_w, dynamic_power_w, exponent) are NOT
    derived from TDP because training power is typically far below thermal design
    power.  They are calibrated from observed power-frequency data and stored in
    the fingerprint.

    The fingerprint is calibrated **once per hardware platform** and reused for
    any model or topology on that platform.

    Default fingerprint values (all 0.0) produce near-zero limits — callers must
    supply a calibrated fingerprint or use the legacy hand-crafted path.
    """
    # --- Compute limit (Roofline model) ---
    peak_compute_tokens_s = 0.0
    if hardware.peak_fp16_tensor_tflops_per_gpu and features.approx_flops_per_token > 0:
        peak_compute_tokens_s = (
            hardware.peak_fp16_tensor_tflops_per_gpu * hardware.gpu_count * 1_000_000_000_000.0
        ) / features.approx_flops_per_token

    roofline_scale = 1.0
    if features.arithmetic_intensity_flops_per_byte > 0 and features.hardware_balance_flops_per_byte > 0:
        roofline_scale = min(
            1.0,
            features.arithmetic_intensity_flops_per_byte / max(features.hardware_balance_flops_per_byte, 1e-9),
        )

    compute_limit = peak_compute_tokens_s * roofline_scale * max(fingerprint.compute_efficiency, 1e-9)

    # --- Memory limit ---
    memory_limit = 0.0
    if hardware.memory_bandwidth_gbps_per_gpu and features.tokens_per_step > 0:
        bytes_per_token = features.approx_memory_bytes_per_step / max(features.tokens_per_step, 1e-9)
        peak_memory_tokens_s = (
            hardware.memory_bandwidth_gbps_per_gpu * hardware.gpu_count * 1_000_000_000.0
        ) / max(bytes_per_token, 1e-9)
        memory_limit = peak_memory_tokens_s * max(fingerprint.memory_efficiency, 1e-9)

    # --- Communication limit (from network bandwidth + derived comm bytes) ---
    communication_limit = 1e9  # Single-node default: not bottlenecked
    if features.node_count > 1 and network.effective_bandwidth_gbps > 0:
        total_comm_bytes = (
            features.cross_node_dp_bytes
            + features.cross_node_tp_bytes
            + features.cross_node_pp_bytes
        )
        exposed_comm_bytes = total_comm_bytes * features.communication_share
        comm_time_s = exposed_comm_bytes / max(
            network.effective_bandwidth_gbps * 1_000_000_000.0 * max(fingerprint.network_efficiency, 1e-9),
            1e-9,
        )
        communication_limit = features.tokens_per_step / max(comm_time_s, 1e-9)

    # --- Power: from fingerprint (calibrated, NOT derived from TDP) ---
    static_power = fingerprint.static_power_w
    dynamic_power = fingerprint.dynamic_power_w

    # --- Communication penalty from transport type ---
    communication_penalty = _TRANSPORT_PENALTY.get(network.transport_type.lower(), 0.30)

    return CalibrationParams(
        compute_limit_at_max_tokens_per_s=compute_limit,
        memory_limit_tokens_per_s=memory_limit,
        communication_limit_tokens_per_s=communication_limit,
        communication_penalty=communication_penalty,
        static_power_w=static_power,
        dynamic_power_w=dynamic_power,
        dynamic_power_exponent=fingerprint.dynamic_power_exponent or 1.5,
        power_utilization_exponent=fingerprint.power_utilization_exponent or 1.0,
        throughput_saturation_ratio=1.0,
        thermal_throttle_threshold=fingerprint.thermal_throttle_threshold,
        thermal_throttle_coefficient=fingerprint.thermal_throttle_coefficient,
        reference_total_gpu_count=hardware.gpu_count,
        reference_gpus_per_node=features.gpus_per_node,
        reference_pipeline_parallel_efficiency=features.pipeline_parallel_efficiency,
        cross_node_reference_bandwidth_gbps=network.effective_bandwidth_gbps,
    )


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
    raw_distance = _clamp(
        (0.60 * abs(features.tp_sync_fraction - params.reference_tp_sync_fraction))
        + (0.25 * abs(features.pipeline_exposed_fraction - params.reference_pipeline_exposed_fraction))
        + (0.15 * abs(features.dp_overlapable_fraction - params.reference_dp_overlapable_fraction)),
        0.0,
        1.0,
    )
    if not params.reference_multi_topology_calibration:
        return raw_distance
    dispersion = _clamp(params.reference_topology_dispersion, 0.0, 1.0)
    if raw_distance <= dispersion:
        return 0.0
    return _clamp((raw_distance - dispersion) / max(1.0 - dispersion, 1e-9), 0.0, 1.0)


def _network_quality_distance(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    if features.node_count <= 1:
        return 0.0

    observed_bandwidth = float(features.network_effective_bandwidth_gbps or params.cross_node_reference_bandwidth_gbps or 0.0)
    observed_jitter = float(features.network_jitter_cv or params.cross_node_reference_jitter_cv or 0.0)
    reference_bandwidth = float(params.cross_node_reference_bandwidth_gbps or 0.0)
    reference_jitter = float(params.cross_node_reference_jitter_cv or 0.0)

    bandwidth_distance = 0.0
    if observed_bandwidth > 0.0 and reference_bandwidth > 0.0:
        bandwidth_distance = abs(math.log(observed_bandwidth / max(reference_bandwidth, 1e-9)))

    jitter_distance = 0.0
    if observed_jitter > 0.0 and reference_jitter > 0.0:
        jitter_distance = abs(math.log(observed_jitter / max(reference_jitter, 1e-9)))

    return _clamp((0.70 * bandwidth_distance) + (0.30 * jitter_distance), 0.0, 1.0)


def _transfer_correction_damping(
    features: DerivedModelFeatures,
    params: CalibrationParams,
    *,
    emphasis: str,
) -> float:
    topology_distance = _transfer_topology_distance(features, params)
    network_distance = _network_quality_distance(features, params)
    if topology_distance <= 0.0 and network_distance <= 0.0:
        return 1.0

    large_gap_bounds = _primary_large_gap_bounds(params)
    sparsity_boost = 1.0 if large_gap_bounds is not None else 0.65
    mismatch = _clamp((0.80 * topology_distance) + (0.20 * network_distance), 0.0, 1.0)
    if emphasis == 'low':
        return _clamp(1.0 - ((1.00 + (0.60 * sparsity_boost)) * mismatch), 0.10, 1.0)
    return _clamp(1.0 - ((0.45 + (0.35 * sparsity_boost)) * mismatch), 0.30, 1.0)


def _scale_transfer_correction(correction: float, damping: float, *, preserve_penalty: bool) -> float:
    if correction > 0.0:
        return correction * damping
    if correction < 0.0 and preserve_penalty:
        return correction * (0.70 + (0.30 * damping))
    return correction * damping


def _topology_shape_pressure(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    signed_pressure = (
        0.75 * (features.tp_sync_fraction - params.reference_tp_sync_fraction)
        + 0.25 * (params.reference_pipeline_exposed_fraction - features.pipeline_exposed_fraction)
    )
    return _clamp(signed_pressure, -1.0, 1.0)


def _multi_topology_prior_strength(params: CalibrationParams) -> float:
    if not params.reference_multi_topology_calibration:
        return 0.0
    return _clamp(params.reference_topology_dispersion, 0.0, 1.0)


def _topology_anchor_pressure(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    if not _has_reference_topology_features(params):
        return 0.0
    return _clamp(
        (0.70 * max(features.tp_sync_fraction - params.reference_tp_sync_fraction, 0.0))
        + (0.25 * max(params.reference_pipeline_exposed_fraction - features.pipeline_exposed_fraction, 0.0))
        + (0.05 * max(params.reference_dp_overlapable_fraction - features.dp_overlapable_fraction, 0.0)),
        0.0,
        1.0,
    )


def _topology_frequency_shape_scale(
    frequency_ratio: float,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    positive_shape_pressure = max(_topology_shape_pressure(features, params), 0.0)
    if positive_shape_pressure <= 0.0:
        return 1.0
    effective_sensitivity = params.base_topology_shape_sensitivity
    if params.reference_multi_topology_calibration:
        effective_sensitivity += 0.50 + (1.50 * _multi_topology_prior_strength(params))
    if effective_sensitivity <= 0.0:
        return 1.0
    low_freq_span = max(1.0 - frequency_ratio, 0.0)
    adjustment = effective_sensitivity * positive_shape_pressure * (0.35 + (0.65 * low_freq_span))
    return _clamp(1.0 - adjustment, 0.55, 1.10)


def _topology_base_pressure(features: DerivedModelFeatures) -> float:
    return _clamp(
        (0.55 * features.tp_sync_fraction)
        + (0.30 * features.pipeline_exposed_fraction)
        + (0.15 * (1.0 - features.dp_overlapable_fraction)),
        0.0,
        1.0,
    )


def _topology_base_throughput_scale(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    anchor_pressure = _topology_anchor_pressure(features, params)
    effective_sensitivity = params.base_topology_throughput_sensitivity
    if params.reference_multi_topology_calibration:
        effective_sensitivity += 1.00 + (2.50 * _multi_topology_prior_strength(params))
    if anchor_pressure <= 0.0 or effective_sensitivity <= 0.0:
        return 1.0
    scale = 1.0 / (1.0 + (2.5 * effective_sensitivity * anchor_pressure))
    return _clamp(scale, params.base_topology_min_throughput_scale, 1.02)


def _topology_communication_scale(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    anchor_pressure = _topology_anchor_pressure(features, params)
    effective_sensitivity = params.base_topology_communication_sensitivity
    if params.reference_multi_topology_calibration:
        effective_sensitivity += 0.15 + (1.20 * _multi_topology_prior_strength(params))
    if anchor_pressure <= 0.0 or effective_sensitivity <= 0.0:
        return 1.0
    amplification = 2.0 * effective_sensitivity * anchor_pressure
    return _clamp(1.0 + amplification, 1.0, 4.0)


def _topology_power_scale(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    anchor_pressure = _topology_anchor_pressure(features, params)
    effective_sensitivity = params.base_topology_power_sensitivity
    if params.reference_multi_topology_calibration:
        effective_sensitivity += 0.10 + (0.80 * _multi_topology_prior_strength(params))
    if anchor_pressure <= 0.0 or effective_sensitivity <= 0.0:
        return 1.0
    boost = 0.60 * effective_sensitivity * anchor_pressure
    return _clamp(1.0 + boost, 0.90, max(params.base_topology_max_power_scale, 1.0))


def _cluster_capacity_scale(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    reference_total_gpus = max(int(params.reference_total_gpu_count or 0), 1)
    target_total_gpus = max(int(features.node_count) * max(int(features.gpus_per_node), 1), 1)
    gpu_scale = target_total_gpus / reference_total_gpus

    reference_pipeline_efficiency = max(float(params.reference_pipeline_parallel_efficiency or 0.0), 1e-9)
    target_pipeline_efficiency = max(float(features.pipeline_parallel_efficiency or 0.0), 1e-9)
    pipeline_scale = target_pipeline_efficiency / reference_pipeline_efficiency

    return _clamp(gpu_scale * pipeline_scale, 0.25, 8.0)


def _local_gpu_count_power_scale(features: DerivedModelFeatures, params: CalibrationParams) -> float:
    reference_gpus_per_node = max(int(params.reference_gpus_per_node or 0), 1)
    target_gpus_per_node = max(int(features.gpus_per_node), 1)
    return _clamp(target_gpus_per_node / reference_gpus_per_node, 0.25, 8.0)


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
    low_damping = _transfer_correction_damping(features, params, emphasis='low')
    high_damping = _transfer_correction_damping(features, params, emphasis='high')
    low_intensity = _low_freq_correction_intensity(features, params) * low_damping
    high_intensity = _high_freq_correction_intensity(features, params) * high_damping
    low_correction = _scale_transfer_correction(params.throughput_low_freq_correction, low_damping, preserve_penalty=True)
    high_correction = _scale_transfer_correction(params.throughput_high_freq_correction, high_damping, preserve_penalty=False)
    low_factor = 1.0 + (low_intensity * low_correction * low_position)
    high_factor = 1.0 + (high_intensity * high_correction * (high_position ** 2))
    factor = (gate_low * low_factor) + (gate_high * high_factor)
    return _clamp(factor, 0.75, 1.25)


def _power_correction_factor(
    frequency_ratio: float,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    low_position, high_position, gate_low, gate_high = _correction_positions(frequency_ratio, params)
    low_damping = _transfer_correction_damping(features, params, emphasis='low')
    high_damping = _transfer_correction_damping(features, params, emphasis='high')
    low_intensity = _low_freq_correction_intensity(features, params) * low_damping
    high_intensity = _high_freq_correction_intensity(features, params) * high_damping
    low_correction = _scale_transfer_correction(params.power_low_freq_correction, low_damping, preserve_penalty=True)
    high_correction = _scale_transfer_correction(params.power_high_freq_correction, high_damping, preserve_penalty=False)
    low_factor = 1.0 + (low_intensity * low_correction * low_position)
    high_factor = 1.0 + (high_intensity * high_correction * (high_position ** 2))
    factor = (gate_low * low_factor) + (gate_high * high_factor)
    factor *= 1.0 + (_power_frequency_retention_bias(features) * (1.0 - frequency_ratio))
    return _clamp(factor, 0.85, 1.20)


def _cross_node_bandwidth_factor(
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    if features.node_count <= 1:
        return 1.0

    observed_bandwidth = float(features.network_effective_bandwidth_gbps or params.cross_node_reference_bandwidth_gbps or 0.0)
    if params.cross_node_reference_bandwidth_gbps <= 0.0 or observed_bandwidth <= 0.0:
        return 1.0

    bandwidth_ratio = params.cross_node_reference_bandwidth_gbps / max(observed_bandwidth, 1e-9)
    return _clamp(1.0 + (params.cross_node_bandwidth_sensitivity * (bandwidth_ratio - 1.0)), 0.75, 3.0)


def _cross_node_benchmark_bandwidth_scale(
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    if features.node_count <= 1:
        return 1.0

    observed_bandwidth = float(features.network_effective_bandwidth_gbps or 0.0)
    reference_bandwidth = float(params.cross_node_reference_bandwidth_gbps or 0.0)
    if observed_bandwidth <= 0.0 or reference_bandwidth <= 0.0:
        return 1.0
    return _clamp(observed_bandwidth / max(reference_bandwidth, 1e-9), 0.02, 1024.0)


def _cross_node_jitter_multiplier(
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    if features.node_count <= 1:
        return 1.0

    observed_jitter = float(features.network_jitter_cv or params.cross_node_reference_jitter_cv or 0.0)
    if params.cross_node_reference_jitter_cv <= 0.0 or observed_jitter <= 0.0:
        return 1.0

    jitter_ratio = observed_jitter / max(params.cross_node_reference_jitter_cv, 1e-9)
    return _clamp(1.0 + (params.cross_node_jitter_sensitivity * (jitter_ratio - 1.0)), 0.85, 2.5)


def _cross_node_network_quality_multiplier(
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    return _clamp(
        _cross_node_bandwidth_factor(features, params) * _cross_node_jitter_multiplier(features, params),
        0.75,
        4.0,
    )


def _has_cross_node_benchmark_curve(params: CalibrationParams) -> bool:
    return (
        params.cross_node_use_benchmark_time_model
        and len(params.cross_node_benchmark_message_sizes_mb) > 0
        and len(params.cross_node_benchmark_message_sizes_mb) == len(params.cross_node_benchmark_bus_bandwidth_gbps)
    )


def _cross_node_benchmark_points(params: CalibrationParams) -> list[tuple[float, float]]:
    points = []
    for size_mb, busbw_gbps in zip(
        params.cross_node_benchmark_message_sizes_mb,
        params.cross_node_benchmark_bus_bandwidth_gbps,
    ):
        size_mb = float(size_mb)
        busbw_gbps = float(busbw_gbps)
        if size_mb > 0.0 and busbw_gbps > 0.0:
            points.append((size_mb, busbw_gbps))
    return sorted(points, key=lambda item: item[0])


def _interpolate_cross_node_benchmark_bandwidth_gbps(
    message_bytes: float,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    points = _cross_node_benchmark_points(params)
    if not points or message_bytes <= 0.0:
        return 0.0

    target_size_mb = max(message_bytes / 1_000_000.0, 1e-6)
    if target_size_mb <= points[0][0]:
        base_bandwidth_gbps = points[0][1]
    elif target_size_mb >= points[-1][0]:
        base_bandwidth_gbps = points[-1][1]
    else:
        base_bandwidth_gbps = points[-1][1]
        for (left_size, left_bandwidth), (right_size, right_bandwidth) in zip(points, points[1:]):
            if left_size <= target_size_mb <= right_size:
                left_log_size = math.log(max(left_size, 1e-9))
                right_log_size = math.log(max(right_size, 1e-9))
                target_log_size = math.log(target_size_mb)
                if abs(right_log_size - left_log_size) < 1e-12:
                    base_bandwidth_gbps = right_bandwidth
                else:
                    blend = (target_log_size - left_log_size) / (right_log_size - left_log_size)
                    left_log_bandwidth = math.log(max(left_bandwidth, 1e-9))
                    right_log_bandwidth = math.log(max(right_bandwidth, 1e-9))
                    base_bandwidth_gbps = math.exp(
                        left_log_bandwidth + (blend * (right_log_bandwidth - left_log_bandwidth))
                    )
                break

    return max(base_bandwidth_gbps * _cross_node_benchmark_bandwidth_scale(features, params), 0.0)


def _allreduce_transfer_bytes(message_bytes: float, group_size: int) -> float:
    if message_bytes <= 0.0 or group_size <= 1:
        return 0.0
    return 2.0 * message_bytes * max(group_size - 1, 0) / max(group_size, 1)


def _estimate_benchmark_transport_time_s(
    message_bytes: float,
    *,
    collective_group_size: int,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    if message_bytes <= 0.0:
        return 0.0

    bandwidth_gbps = _interpolate_cross_node_benchmark_bandwidth_gbps(message_bytes, features, params)
    if bandwidth_gbps <= 0.0:
        return 0.0

    transfer_bytes = (
        _allreduce_transfer_bytes(message_bytes, collective_group_size)
        if collective_group_size > 1
        else message_bytes
    )
    return transfer_bytes / max(bandwidth_gbps * 1_000_000_000.0, 1e-9)


def estimate_cross_node_time_penalty_s(
    frequency_mhz: float,
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    """Estimate cross-node time penalty with communication-complexity-aware model.

    This model explicitly accounts for:
    1. DP group size effect on allreduce latency (larger groups = slower)
    2. Frequency-dependent communication exposure (higher freq = less overlap)
    3. Topology structure (PP stages, TP groups, cross-node boundaries)

    The penalty is ADDITIVE to base step time: corrected_step_time = base_step_time + penalty.
    """
    if features.node_count <= 1:
        return 0.0

    max_frequency = float(hardware.max_frequency_mhz or frequency_mhz or 1.0)
    frequency_ratio = _clamp(float(frequency_mhz) / max_frequency, 0.05, 1.0)

    # Extract coefficients from params
    alpha_pp = max(params.cross_node_alpha_pp_s_per_byte, 0.0)
    alpha_dp = max(params.cross_node_alpha_dp_s_per_byte, 0.0)
    alpha_tp = max(params.cross_node_alpha_tp_s_per_byte, 0.0)
    beta_pp_wait = max(params.cross_node_beta_pp_wait_s, 0.0)
    beta_pp_edge = max(params.cross_node_beta_pp_edge_s, 0.0)

    # --- Explicit TP/PP/DP exposed communication time model ---
    # Higher frequency means less compute slack to hide communication waits, so exposure grows with frequency_ratio.
    pp_exposure_factor = _clamp(
        0.65
        + (0.35 * frequency_ratio)
        + (params.cross_node_pp_bubble_exposure_gain * features.pipeline_bubble_fraction),
        0.50,
        1.80,
    )
    dp_exposure_factor = _clamp(
        0.55
        + (0.45 * frequency_ratio)
        + (0.25 * (1.0 - features.dp_overlapable_fraction))
        + (0.20 * features.dp_cross_node_group_fraction),
        0.40,
        2.00,
    )
    tp_exposure_factor = _clamp(
        0.65
        + (0.50 * frequency_ratio)
        + (0.30 * features.tp_sync_fraction)
        + (0.20 * features.tp_cross_node_group_fraction),
        0.50,
        2.20,
    )
    dp_group_scale = 1.0 + (features.dp_cross_node_group_fraction * params.cross_node_dp_group_scale_gain)

    dp_penalty_s = (
        alpha_dp
        * features.cross_node_dp_bytes
        * dp_group_scale
        * dp_exposure_factor
        * max(params.cross_node_dp_exposure_sensitivity, 0.0)
    )
    pp_penalty_s = (
        alpha_pp
        * features.cross_node_pp_bytes
        * pp_exposure_factor
        * max(params.cross_node_pp_exposure_sensitivity, 0.0)
    ) + (beta_pp_wait * features.pp_cross_node_wait_pressure) + (beta_pp_edge * features.pp_cross_node_edge_fraction)
    tp_penalty_s = (
        alpha_tp
        * features.cross_node_tp_bytes
        * tp_exposure_factor
        * max(params.cross_node_tp_exposure_sensitivity, 0.0)
    )
    network_quality_multiplier = _cross_node_network_quality_multiplier(features, params)

    total_penalty_s = network_quality_multiplier * (dp_penalty_s + pp_penalty_s + tp_penalty_s)
    return max(total_penalty_s, 0.0)


def _predict_base_throughput_tokens_per_s(
    frequency_mhz: float,
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    max_frequency = float(hardware.max_frequency_mhz or frequency_mhz or 1.0)
    frequency_ratio = _clamp(float(frequency_mhz) / max_frequency, 0.05, 1.0)
    compute_frequency_ratio = _compute_frequency_ratio(frequency_ratio, params, features)
    compute_frequency_ratio *= _topology_frequency_shape_scale(frequency_ratio, features, params)
    topology_throughput_scale = _topology_base_throughput_scale(features, params)
    topology_communication_scale = _topology_communication_scale(features, params)
    cluster_capacity_scale = _cluster_capacity_scale(features, params)
    compute_limit = (
        params.compute_limit_at_max_tokens_per_s
        * compute_frequency_ratio
        * topology_throughput_scale
        * cluster_capacity_scale
    )
    memory_limit = params.memory_limit_tokens_per_s * cluster_capacity_scale
    exposed_communication_share = _exposed_communication_share(features)
    communication_limit = params.communication_limit_tokens_per_s / (
        (1.0 + params.communication_penalty * exposed_communication_share * ((1.0 / frequency_ratio) - 1.0))
        * topology_communication_scale
    )
    throughput = _harmonic_blend(
        [compute_limit, memory_limit, communication_limit],
        [features.compute_weight, features.memory_weight, features.communication_weight],
    )
    throughput *= max(features.pipeline_parallel_efficiency, 1e-9)
    throughput *= _throughput_correction_factor(frequency_ratio, features, params)
    return throughput


def _thermal_throttle_factor(
    frequency_ratio: float,
    params: CalibrationParams,
) -> float:
    threshold = params.thermal_throttle_threshold
    if threshold >= 1.0 or frequency_ratio <= threshold:
        return 1.0
    overshoot = frequency_ratio - threshold
    max_overshoot = 1.0 - threshold
    if max_overshoot <= 1e-9:
        return 1.0
    thermal_drop = params.thermal_throttle_coefficient * (overshoot / max_overshoot) ** 2
    return max(0.5, 1.0 - thermal_drop)


def predict_throughput_tokens_per_s(
    frequency_mhz: float,
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
    mode: str = "static",
) -> float:
    """Predict throughput at a given frequency.

    Args:
        mode: "static" for fixed-frequency prediction (no thermal throttling),
              "baseline" for dynamic-boost prediction (with thermal throttling).
    """
    base_throughput = _predict_base_throughput_tokens_per_s(frequency_mhz, hardware, features, params)
    cross_node_penalty_s = estimate_cross_node_time_penalty_s(frequency_mhz, hardware, features, params)
    if cross_node_penalty_s <= 0.0:
        throughput = base_throughput
    else:
        base_step_time_s = features.tokens_per_step / max(base_throughput, 1e-9)
        corrected_step_time_s = base_step_time_s + cross_node_penalty_s
        throughput = features.tokens_per_step / max(corrected_step_time_s, 1e-9)
    max_frequency = float(hardware.max_frequency_mhz or frequency_mhz or 1.0)
    frequency_ratio = _clamp(float(frequency_mhz) / max_frequency, 0.05, 1.0)
    # Thermal throttling only applies to baseline (dynamic boost) mode.
    # Static locking eliminates thermal throttling because temperature stabilizes.
    if mode == "baseline":
        throughput *= _thermal_throttle_factor(frequency_ratio, params)
    return throughput




def _cross_node_power_multiplier(
    frequency_ratio: float,
    features: DerivedModelFeatures,
    params: CalibrationParams,
) -> float:
    if features.node_count <= 1:
        return 1.0
    reference_cross_node_dp_bytes = max(params.cross_node_reference_cross_node_dp_bytes, 1e-9)
    dp_scale = features.cross_node_dp_bytes / reference_cross_node_dp_bytes
    low_freq_gap = max(params.cross_node_power_low_freq_reference_ratio - frequency_ratio, 0.0)
    network_quality_multiplier = _cross_node_network_quality_multiplier(features, params)
    drop = (params.cross_node_power_base_drop * dp_scale * network_quality_multiplier) + (
        params.cross_node_power_low_freq_gamma * dp_scale * low_freq_gap * network_quality_multiplier
    )
    return _clamp(1.0 - drop, 0.85, 1.05)

def predict_power_w(
    frequency_mhz: float,
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
    mode: str = "static",
) -> float:
    """Predict average power at a given frequency.

    Args:
        mode: "static" for fixed-frequency prediction (no thermal throttling),
              "baseline" for dynamic-boost prediction (with thermal throttling).
    """
    max_frequency = float(hardware.max_frequency_mhz or frequency_mhz or 1.0)
    frequency_ratio = _clamp(float(frequency_mhz) / max_frequency, 0.05, 1.0)
    compute_frequency_ratio = _compute_frequency_ratio(frequency_ratio, params, features)
    compute_limit = max(params.compute_limit_at_max_tokens_per_s * compute_frequency_ratio, 1e-9)
    throughput = predict_throughput_tokens_per_s(frequency_mhz, hardware, features, params, mode=mode)
    utilization = _clamp(throughput / compute_limit, 0.05, 1.2)
    utilization_factor = utilization ** params.power_utilization_exponent
    dynamic = params.dynamic_power_w * utilization_factor * (frequency_ratio ** params.dynamic_power_exponent)
    power_w = params.static_power_w + dynamic
    power_w *= _topology_power_scale(features, params)
    power_w *= _local_gpu_count_power_scale(features, params)
    power_w *= _power_correction_factor(frequency_ratio, features, params)
    power_w *= _cross_node_power_multiplier(frequency_ratio, features, params)
    return power_w


def predict_point(
    frequency_mhz: int,
    hardware: HardwareFeatures,
    features: DerivedModelFeatures,
    params: CalibrationParams,
    mode: str = "static",
) -> PredictionPoint:
    """Predict a single point.

    Args:
        mode: "static" for fixed-frequency prediction (no thermal throttling),
              "baseline" for dynamic-boost prediction (with thermal throttling).
    """
    max_frequency = float(hardware.max_frequency_mhz or frequency_mhz or 1.0)
    frequency_ratio = _clamp(float(frequency_mhz) / max_frequency, 0.05, 1.0)
    compute_frequency_ratio = _compute_frequency_ratio(frequency_ratio, params, features)
    throughput_tokens_per_s = predict_throughput_tokens_per_s(frequency_mhz, hardware, features, params, mode=mode)
    throughput_samples_per_s = throughput_tokens_per_s / max(features.tokens_per_step / max(features.samples_per_step, 1e-9), 1e-9)
    power_w = predict_power_w(frequency_mhz, hardware, features, params, mode=mode)
    step_time_s = features.tokens_per_step / max(throughput_tokens_per_s, 1e-9)
    tokens_per_j = throughput_tokens_per_s / max(power_w, 1e-9)
    tokens_per_wh = tokens_per_j * 3600.0
    samples_per_wh = throughput_samples_per_s * 3600.0 / max(power_w, 1e-9)
    exposed_communication_share = _exposed_communication_share(features)
    cluster_capacity_scale = _cluster_capacity_scale(features, params)
    component_limits = {
        "compute_tokens_per_s": params.compute_limit_at_max_tokens_per_s * compute_frequency_ratio * cluster_capacity_scale,
        "memory_tokens_per_s": params.memory_limit_tokens_per_s * cluster_capacity_scale,
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
    mode: str = "static",
) -> List[PredictionPoint]:
    """Sweep prediction over a range of frequencies.

    Args:
        mode: "static" for fixed-frequency prediction (no thermal throttling),
              "baseline" for dynamic-boost prediction (with thermal throttling).
    """
    return [predict_point(freq, hardware, features, params, mode=mode) for freq in frequencies_mhz]


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
