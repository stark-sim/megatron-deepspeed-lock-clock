from __future__ import annotations

from dataclasses import dataclass
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
    runtime_ratio_mape: float
    energy_ratio_mape: float
    objective: float


def _mean_absolute_percentage_error(pairs: List[Tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    total = 0.0
    for observed, predicted in pairs:
        baseline = max(abs(observed), 1e-9)
        total += abs(observed - predicted) / baseline
    return total / len(pairs)


def _candidate_scales(anchor: float, scales: List[float]) -> List[float]:
    return [anchor * scale for scale in scales]


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
    observed_runtime_ratios = [baseline_throughput / max(throughput, 1e-9) for throughput in throughputs]
    observed_energy_ratios = [baseline_tokens_per_j / max(_tokens_per_j(throughput, power), 1e-9) for throughput, power in zip(throughputs, powers)]

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

    best: CalibrationResult | None = None

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
                        throughput_error = _mean_absolute_percentage_error(list(zip(throughputs, predicted_throughputs)))
                        for static_power_w in static_power_candidates:
                            for dynamic_power_exponent in dynamic_exponents:
                                power_params = CalibrationParams(
                                    compute_limit_at_max_tokens_per_s=compute_limit,
                                    memory_limit_tokens_per_s=memory_limit,
                                    communication_limit_tokens_per_s=communication_limit,
                                    communication_penalty=communication_penalty,
                                    static_power_w=static_power_w,
                                    dynamic_power_w=0.0,
                                    dynamic_power_exponent=dynamic_power_exponent,
                                    throughput_saturation_ratio=throughput_saturation_ratio,
                                )
                                dynamic_power_w = _solve_dynamic_power(
                                    frequencies,
                                    predicted_throughputs,
                                    powers,
                                    hardware,
                                    power_params,
                                )
                                params = CalibrationParams(
                                    compute_limit_at_max_tokens_per_s=compute_limit,
                                    memory_limit_tokens_per_s=memory_limit,
                                    communication_limit_tokens_per_s=communication_limit,
                                    communication_penalty=communication_penalty,
                                    static_power_w=static_power_w,
                                    dynamic_power_w=dynamic_power_w,
                                    dynamic_power_exponent=dynamic_power_exponent,
                                    throughput_saturation_ratio=throughput_saturation_ratio,
                                )
                                predicted_powers = [
                                    predict_power_w(freq, hardware, features, params)
                                    for freq, features in zip(frequencies, derived_features)
                                ]
                                power_error = _mean_absolute_percentage_error(list(zip(powers, predicted_powers)))
                                predicted_runtime_ratios = [baseline_throughput / max(throughput, 1e-9) for throughput in predicted_throughputs]
                                predicted_energy_ratios = [
                                    baseline_tokens_per_j / max(_tokens_per_j(throughput, power), 1e-9)
                                    for throughput, power in zip(predicted_throughputs, predicted_powers)
                                ]
                                runtime_ratio_error = _mean_absolute_percentage_error(
                                    list(zip(observed_runtime_ratios, predicted_runtime_ratios))
                                )
                                energy_ratio_error = _mean_absolute_percentage_error(
                                    list(zip(observed_energy_ratios, predicted_energy_ratios))
                                )
                                smoothness_penalty = 0.0
                                if memory_limit < max_observed_throughput * 0.85:
                                    smoothness_penalty += 0.02
                                if throughput_saturation_ratio < 0.78:
                                    smoothness_penalty += 0.01
                                objective = (
                                    (0.20 * throughput_error)
                                    + (0.15 * power_error)
                                    + (0.35 * runtime_ratio_error)
                                    + (0.30 * energy_ratio_error)
                                    + smoothness_penalty
                                )
                                candidate = CalibrationResult(
                                    params=params,
                                    throughput_mape=throughput_error,
                                    power_mape=power_error,
                                    runtime_ratio_mape=runtime_ratio_error,
                                    energy_ratio_mape=energy_ratio_error,
                                    objective=objective,
                                )
                                if best is None or candidate.objective < best.objective:
                                    best = candidate
    if best is None:
        raise RuntimeError("failed to calibrate frequency model")
    return best
