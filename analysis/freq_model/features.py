from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from analysis.freq_model.hardware import HardwareFeatures
from analysis.freq_model.workload import WorkloadFeatures


@dataclass(frozen=True)
class DerivedModelFeatures:
    tokens_per_step: float
    samples_per_step: float
    approx_model_params: float
    approx_flops_per_token: float
    approx_flops_per_step: float
    approx_memory_bytes_per_step: float
    approx_communication_bytes_per_step: float
    arithmetic_intensity_flops_per_byte: float
    communication_share: float
    pipeline_parallel_efficiency: float
    pipeline_exposed_fraction: float
    dp_overlapable_fraction: float
    tp_sync_fraction: float
    hardware_balance_flops_per_byte: float
    compute_weight: float
    memory_weight: float
    communication_weight: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _precision_bytes(precision_mode: str) -> int:
    if precision_mode in {"bf16", "fp16"}:
        return 2
    return 4


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _dp_overlap_prior(zero_stage: int) -> float:
    if zero_stage >= 3:
        return 0.50
    if zero_stage == 2:
        return 0.35
    return 0.20


def derive_model_features(hardware: HardwareFeatures, workload: WorkloadFeatures) -> DerivedModelFeatures:
    bytes_per_element = _precision_bytes(workload.precision_mode)
    tokens_per_step = float(workload.global_batch_size * workload.seq_length)
    samples_per_step = float(workload.global_batch_size)

    approx_attention_params = 4.0 * workload.hidden_size * workload.hidden_size
    approx_mlp_params = 3.0 * workload.hidden_size * workload.ffn_hidden_size
    approx_model_params = float(workload.num_layers * (approx_attention_params + approx_mlp_params))

    approx_flops_per_token = float(
        (6.0 * approx_model_params)
        + (12.0 * workload.num_layers * workload.seq_length * workload.hidden_size)
    )
    approx_flops_per_step = tokens_per_step * approx_flops_per_token

    activation_bytes = (
        tokens_per_step
        * workload.hidden_size
        * workload.num_layers
        * bytes_per_element
        * 6.0
    )
    parameter_bytes = approx_model_params * bytes_per_element

    tp_penalty = max(workload.tensor_model_parallel_size - 1, 0) / max(workload.tensor_model_parallel_size, 1)
    dp_penalty = max(workload.data_parallel_size - 1, 0) / max(workload.data_parallel_size, 1)
    pp_penalty = max(workload.pipeline_model_parallel_size - 1, 0) / max(workload.pipeline_model_parallel_size, 1)
    microbatches_per_step = max(
        float(workload.global_batch_size) / max(float(workload.micro_batch_size * workload.data_parallel_size), 1.0),
        1.0,
    )
    pipeline_bubble_fraction = 0.0
    if workload.pipeline_model_parallel_size > 1:
        pipeline_bubble_fraction = max(
            float(max(workload.pipeline_model_parallel_size - 1, 0))
            / max(microbatches_per_step + max(workload.pipeline_model_parallel_size - 1, 0), 1.0),
            0.0,
        )
    pp_communication_multiplier = 1.0 + pipeline_bubble_fraction
    communication_penalty_denominator = max((2.0 * tp_penalty) + dp_penalty + (pp_penalty * pp_communication_multiplier), 1e-9)
    communication_bytes = (
        tokens_per_step
        * workload.hidden_size
        * workload.num_layers
        * bytes_per_element
        * communication_penalty_denominator
    )

    approx_memory_bytes_per_step = activation_bytes + (2.0 * parameter_bytes) + communication_bytes
    arithmetic_intensity = approx_flops_per_step / max(approx_memory_bytes_per_step, 1.0)

    hardware_balance = 40.0
    if hardware.peak_fp16_tensor_tflops_per_gpu and hardware.memory_bandwidth_gbps_per_gpu:
        hardware_balance = (
            hardware.peak_fp16_tensor_tflops_per_gpu * 1_000.0
        ) / max(hardware.memory_bandwidth_gbps_per_gpu, 1e-6)

    communication_share = communication_bytes / max(approx_memory_bytes_per_step, 1.0)
    pipeline_parallel_efficiency = 1.0
    if workload.pipeline_model_parallel_size > 1:
        pipeline_parallel_efficiency = microbatches_per_step / (
            microbatches_per_step + max(workload.pipeline_model_parallel_size - 1, 0)
        )

    pipeline_exposed_fraction = pp_penalty * (0.40 + (0.60 * pipeline_bubble_fraction))
    dp_overlapable_fraction = dp_penalty * _dp_overlap_prior(workload.zero_stage)
    tp_sync_fraction = (2.0 * tp_penalty) / communication_penalty_denominator if communication_penalty_denominator > 0 else 0.0

    pipeline_exposed_fraction = _clamp(pipeline_exposed_fraction, 0.0, 1.0)
    dp_overlapable_fraction = _clamp(dp_overlapable_fraction, 0.0, 1.0)
    tp_sync_fraction = _clamp(tp_sync_fraction, 0.0, 1.0)

    communication_weight = _clamp(communication_share * 1.5, 0.05, 0.35)
    compute_base = arithmetic_intensity / (arithmetic_intensity + hardware_balance)
    compute_base = _clamp(compute_base, 0.15, 0.85)
    memory_base = 1.0 - compute_base
    remaining = 1.0 - communication_weight
    compute_weight = remaining * compute_base
    memory_weight = remaining * memory_base

    return DerivedModelFeatures(
        tokens_per_step=tokens_per_step,
        samples_per_step=samples_per_step,
        approx_model_params=approx_model_params,
        approx_flops_per_token=approx_flops_per_token,
        approx_flops_per_step=approx_flops_per_step,
        approx_memory_bytes_per_step=approx_memory_bytes_per_step,
        approx_communication_bytes_per_step=communication_bytes,
        arithmetic_intensity_flops_per_byte=arithmetic_intensity,
        communication_share=communication_share,
        pipeline_parallel_efficiency=pipeline_parallel_efficiency,
        pipeline_exposed_fraction=pipeline_exposed_fraction,
        dp_overlapable_fraction=dp_overlapable_fraction,
        tp_sync_fraction=tp_sync_fraction,
        hardware_balance_flops_per_byte=hardware_balance,
        compute_weight=compute_weight,
        memory_weight=memory_weight,
        communication_weight=communication_weight,
    )
