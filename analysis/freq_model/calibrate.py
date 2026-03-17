from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Tuple

from analysis.freq_model.features import DerivedModelFeatures
from analysis.freq_model.hardware import HardwareFeatures
from analysis.freq_model.model import CalibrationParams, infer_initial_anchors, predict_power_w, predict_throughput_tokens_per_s
from analysis.freq_model.workload import LoadedRunSample


@dataclass(frozen=True)
class CalibrationResult:
    params: CalibrationParams
    throughput_mape: float
    power_mape: float
    total_time_mape: float
    total_energy_mape: float
    runtime_ratio_mape: float
    energy_ratio_mape: float
    objective: float


@dataclass(frozen=True)
class CalibrationMetrics:
    throughput_mape: float
    power_mape: float
    total_time_mape: float
    total_energy_mape: float
    runtime_ratio_mape: float
    energy_ratio_mape: float


def _mean_absolute_percentage_error(pairs: List[Tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    total = 0.0
    for observed, predicted in pairs:
        baseline = max(abs(observed), 1e-9)
        total += abs(observed - predicted) / baseline
    return total / len(pairs)


def _candidate_limits(anchor: float | None, observed_reference: float, scales: List[float], min_ratio: float = 0.01) -> List[float]:
    if observed_reference <= 0:
        observed_reference = 1.0
    if anchor is None or anchor <= 0:
        return [observed_reference * scale for scale in scales]
    centered_ratio = observed_reference / max(anchor, 1e-9)
    centered_ratio = min(max(centered_ratio, min_ratio), 1.0)
    return [anchor * centered_ratio * scale for scale in scales]


def _tokens_per_j(throughput_tokens_per_s: float, power_w: float) -> float:
    return throughput_tokens_per_s / max(power_w, 1e-9)


def _sample_interval_tokens(sample: LoadedRunSample, features: DerivedModelFeatures) -> float:
    if sample.observed.interval_tokens is not None and sample.observed.interval_tokens > 0:
        return float(sample.observed.interval_tokens)
    return features.tokens_per_step * max(sample.observed.num_steps, 1)


def _solve_dynamic_power(
    frequencies_mhz: List[int],
    predicted_throughputs: List[float],
    observed_powers: List[float],
    hardware: HardwareFeatures,
    params: CalibrationParams,
) -> float:
    max_frequency = float(hardware.max_frequency_mhz or max(frequencies_mhz) or 1.0)
    numerator = 0.0
    denominator = 0.0
    for frequency_mhz, throughput, observed_power in zip(frequencies_mhz, predicted_throughputs, observed_powers):
        frequency_ratio = max(float(frequency_mhz) / max_frequency, 0.05)
        compute_limit = max(params.compute_limit_at_max_tokens_per_s * frequency_ratio, 1e-9)
        utilization = min(max(throughput / compute_limit, 0.05), 1.2)
        basis = utilization * (frequency_ratio ** params.dynamic_power_exponent)
        numerator += basis * max(observed_power - params.static_power_w, 0.0)
        denominator += basis * basis
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _reference_throughput_and_efficiency(
    samples: List[LoadedRunSample],
    baseline_sample: LoadedRunSample | None,
) -> tuple[float, float]:
    if baseline_sample is not None:
        baseline_throughput = baseline_sample.observed.throughput_tokens_per_s or 0.0
        baseline_tokens_per_j = baseline_sample.observed.tokens_per_j or _tokens_per_j(
            baseline_throughput,
            baseline_sample.observed.avg_power_w,
        )
        return baseline_throughput, baseline_tokens_per_j

    fastest_sample = max(samples, key=lambda sample: sample.observed.throughput_tokens_per_s or 0.0)
    ref_throughput = fastest_sample.observed.throughput_tokens_per_s or 1.0
    ref_tokens_per_j = fastest_sample.observed.tokens_per_j or _tokens_per_j(ref_throughput, fastest_sample.observed.avg_power_w)
    return ref_throughput, ref_tokens_per_j


def _evaluate_metrics(
    samples: List[LoadedRunSample],
    hardware: HardwareFeatures,
    derived_features: List[DerivedModelFeatures],
    params: CalibrationParams,
    baseline_throughput: float,
    baseline_tokens_per_j: float,
) -> CalibrationMetrics:
    throughput_pairs: List[Tuple[float, float]] = []
    power_pairs: List[Tuple[float, float]] = []
    total_time_pairs: List[Tuple[float, float]] = []
    total_energy_pairs: List[Tuple[float, float]] = []
    runtime_ratio_pairs: List[Tuple[float, float]] = []
    energy_ratio_pairs: List[Tuple[float, float]] = []

    for sample, features in zip(samples, derived_features):
        frequency_mhz = int(sample.observed.frequency_mhz or 0)
        predicted_throughput = predict_throughput_tokens_per_s(frequency_mhz, hardware, features, params)
        predicted_power = predict_power_w(frequency_mhz, hardware, features, params)
        observed_throughput = sample.observed.throughput_tokens_per_s or 0.0
        observed_power = sample.observed.avg_power_w
        interval_tokens = _sample_interval_tokens(sample, features)
        observed_total_time = sample.observed.time_s
        predicted_total_time = interval_tokens / max(predicted_throughput, 1e-9)
        observed_total_energy = sample.observed.energy_j
        predicted_total_energy = predicted_power * predicted_total_time

        throughput_pairs.append((observed_throughput, predicted_throughput))
        power_pairs.append((observed_power, predicted_power))
        total_time_pairs.append((observed_total_time, predicted_total_time))
        total_energy_pairs.append((observed_total_energy, predicted_total_energy))

        observed_runtime_ratio = baseline_throughput / max(observed_throughput, 1e-9)
        predicted_runtime_ratio = baseline_throughput / max(predicted_throughput, 1e-9)
        observed_energy_ratio = baseline_tokens_per_j / max(_tokens_per_j(observed_throughput, observed_power), 1e-9)
        predicted_energy_ratio = baseline_tokens_per_j / max(_tokens_per_j(predicted_throughput, predicted_power), 1e-9)
        runtime_ratio_pairs.append((observed_runtime_ratio, predicted_runtime_ratio))
        energy_ratio_pairs.append((observed_energy_ratio, predicted_energy_ratio))

    return CalibrationMetrics(
        throughput_mape=_mean_absolute_percentage_error(throughput_pairs),
        power_mape=_mean_absolute_percentage_error(power_pairs),
        total_time_mape=_mean_absolute_percentage_error(total_time_pairs),
        total_energy_mape=_mean_absolute_percentage_error(total_energy_pairs),
        runtime_ratio_mape=_mean_absolute_percentage_error(runtime_ratio_pairs),
        energy_ratio_mape=_mean_absolute_percentage_error(energy_ratio_pairs),
    )


def _base_objective(metrics: CalibrationMetrics, throughput_saturation_ratio: float, memory_limit: float, max_observed_throughput: float) -> float:
    smoothness_penalty = 0.0
    if memory_limit < max_observed_throughput * 0.85:
        smoothness_penalty += 0.02
    if throughput_saturation_ratio < 0.78:
        smoothness_penalty += 0.01
    return (
        (0.10 * metrics.throughput_mape)
        + (0.10 * metrics.power_mape)
        + (0.30 * metrics.total_time_mape)
        + (0.30 * metrics.total_energy_mape)
        + (0.10 * metrics.runtime_ratio_mape)
        + (0.10 * metrics.energy_ratio_mape)
        + smoothness_penalty
    )


def _correction_objective(metrics: CalibrationMetrics, params: CalibrationParams) -> float:
    magnitude_penalty = 0.02 * (
        abs(params.throughput_low_freq_correction)
        + abs(params.throughput_high_freq_correction)
        + abs(params.power_low_freq_correction)
        + abs(params.power_high_freq_correction)
    )
    return (
        (0.10 * metrics.throughput_mape)
        + (0.10 * metrics.power_mape)
        + (0.32 * metrics.total_time_mape)
        + (0.32 * metrics.total_energy_mape)
        + (0.08 * metrics.runtime_ratio_mape)
        + (0.08 * metrics.energy_ratio_mape)
        + magnitude_penalty
    )


def _fit_curve_corrections(
    samples: List[LoadedRunSample],
    hardware: HardwareFeatures,
    derived_features: List[DerivedModelFeatures],
    base_params: CalibrationParams,
    baseline_throughput: float,
    baseline_tokens_per_j: float,
) -> tuple[CalibrationParams, CalibrationMetrics, float]:
    best_params = base_params
    best_metrics = _evaluate_metrics(samples, hardware, derived_features, base_params, baseline_throughput, baseline_tokens_per_j)
    best_objective = _correction_objective(best_metrics, base_params)

    split_ratios = [0.68, 0.72, 0.76]
    transition_widths = [0.04, 0.06, 0.08]
    throughput_low_values = [-0.12, -0.08, -0.04, 0.0, 0.04]
    throughput_high_values = [-0.04, -0.02, 0.0, 0.02]
    power_low_values = [-0.02, 0.0, 0.02]
    power_high_values = [0.0, 0.02, 0.04, 0.06]
    topology_weights = [0.5, 1.0]
    communication_weights = [0.25, 0.5]

    for split_ratio in split_ratios:
        for transition_width in transition_widths:
            for throughput_low in throughput_low_values:
                for throughput_high in throughput_high_values:
                    for power_low in power_low_values:
                        for power_high in power_high_values:
                            for topology_weight in topology_weights:
                                for communication_weight in communication_weights:
                                    candidate_params = replace(
                                        base_params,
                                        correction_split_ratio=split_ratio,
                                        correction_transition_width=transition_width,
                                        throughput_low_freq_correction=throughput_low,
                                        throughput_high_freq_correction=throughput_high,
                                        power_low_freq_correction=power_low,
                                        power_high_freq_correction=power_high,
                                        correction_topology_weight=topology_weight,
                                        correction_communication_weight=communication_weight,
                                    )
                                    metrics = _evaluate_metrics(
                                        samples,
                                        hardware,
                                        derived_features,
                                        candidate_params,
                                        baseline_throughput,
                                        baseline_tokens_per_j,
                                    )
                                    objective = _correction_objective(metrics, candidate_params)
                                    if objective < best_objective:
                                        best_params = candidate_params
                                        best_metrics = metrics
                                        best_objective = objective

    return best_params, best_metrics, best_objective


def calibrate_frequency_model(
    samples: List[LoadedRunSample],
    hardware: HardwareFeatures,
    derived_features: List[DerivedModelFeatures],
    baseline_sample: LoadedRunSample | None = None,
) -> CalibrationResult:
    if not samples:
        raise ValueError("at least one sample is required for calibration")

    throughputs = [sample.observed.throughput_tokens_per_s or 0.0 for sample in samples]
    powers = [sample.observed.avg_power_w for sample in samples]
    frequencies = [int(sample.observed.frequency_mhz or 0) for sample in samples]
    max_observed_throughput = max(throughputs)
    base_features = derived_features[0]
    anchors = infer_initial_anchors(hardware, base_features, max_observed_throughput)
    baseline_throughput, baseline_tokens_per_j = _reference_throughput_and_efficiency(samples, baseline_sample)

    compute_candidates = _candidate_limits(
        anchors["compute_anchor"],
        observed_reference=max_observed_throughput,
        scales=[0.70, 0.85, 1.00, 1.15, 1.35, 1.60],
        min_ratio=0.01,
    )
    memory_candidates = _candidate_limits(
        anchors["memory_anchor"],
        observed_reference=max_observed_throughput,
        scales=[0.85, 1.00, 1.20, 1.50, 1.90, 2.40],
        min_ratio=0.002,
    )
    communication_candidates = _candidate_limits(
        anchors["communication_anchor"],
        observed_reference=max_observed_throughput,
        scales=[0.70, 0.85, 1.00, 1.15, 1.35, 1.60],
        min_ratio=0.01,
    )
    communication_penalties = [0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2]
    saturation_ratios = [0.72, 0.78, 0.84, 0.90, 0.96, 1.0]
    static_power_candidates = [min(powers) * ratio for ratio in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]]
    dynamic_exponents = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]

    best_params: CalibrationParams | None = None
    best_metrics: CalibrationMetrics | None = None
    best_objective: float | None = None

    for compute_limit in compute_candidates:
        for memory_limit in memory_candidates:
            for communication_limit in communication_candidates:
                for communication_penalty in communication_penalties:
                    for throughput_saturation_ratio in saturation_ratios:
                        base_params = CalibrationParams(
                            compute_limit_at_max_tokens_per_s=compute_limit,
                            memory_limit_tokens_per_s=memory_limit,
                            communication_limit_tokens_per_s=communication_limit,
                            communication_penalty=communication_penalty,
                            static_power_w=0.0,
                            dynamic_power_w=0.0,
                            dynamic_power_exponent=1.6,
                            throughput_saturation_ratio=throughput_saturation_ratio,
                        )
                        predicted_throughputs = [
                            predict_throughput_tokens_per_s(freq, hardware, features, base_params)
                            for freq, features in zip(frequencies, derived_features)
                        ]
                        for static_power_w in static_power_candidates:
                            for dynamic_power_exponent in dynamic_exponents:
                                power_params = replace(
                                    base_params,
                                    static_power_w=static_power_w,
                                    dynamic_power_exponent=dynamic_power_exponent,
                                )
                                dynamic_power_w = _solve_dynamic_power(
                                    frequencies,
                                    predicted_throughputs,
                                    powers,
                                    hardware,
                                    power_params,
                                )
                                params = replace(power_params, dynamic_power_w=dynamic_power_w)
                                metrics = _evaluate_metrics(
                                    samples,
                                    hardware,
                                    derived_features,
                                    params,
                                    baseline_throughput,
                                    baseline_tokens_per_j,
                                )
                                objective = _base_objective(metrics, throughput_saturation_ratio, memory_limit, max_observed_throughput)
                                if best_objective is None or objective < best_objective:
                                    best_params = params
                                    best_metrics = metrics
                                    best_objective = objective

    if best_params is None or best_metrics is None or best_objective is None:
        raise RuntimeError("failed to calibrate frequency model")

    corrected_params, corrected_metrics, corrected_objective = _fit_curve_corrections(
        samples,
        hardware,
        derived_features,
        best_params,
        baseline_throughput,
        baseline_tokens_per_j,
    )

    max_frequency = float(hardware.max_frequency_mhz or max(frequencies) or 1.0)
    corrected_params = replace(
        corrected_params,
        reference_min_frequency_ratio=min(frequencies) / max_frequency,
        reference_max_frequency_ratio=max(frequencies) / max_frequency,
        reference_observed_frequency_ratios=tuple(sorted(freq / max_frequency for freq in frequencies)),
        reference_pipeline_exposed_fraction=base_features.pipeline_exposed_fraction,
        reference_dp_overlapable_fraction=base_features.dp_overlapable_fraction,
        reference_tp_sync_fraction=base_features.tp_sync_fraction,
        reference_topology_features_present=True,
    )

    return CalibrationResult(
        params=corrected_params,
        throughput_mape=corrected_metrics.throughput_mape,
        power_mape=corrected_metrics.power_mape,
        total_time_mape=corrected_metrics.total_time_mape,
        total_energy_mape=corrected_metrics.total_energy_mape,
        runtime_ratio_mape=corrected_metrics.runtime_ratio_mape,
        energy_ratio_mape=corrected_metrics.energy_ratio_mape,
        objective=corrected_objective,
    )
