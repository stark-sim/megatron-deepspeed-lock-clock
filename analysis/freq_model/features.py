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
    microbatches_per_step: float = 1.0
    pipeline_bubble_fraction: float = 0.0
    replicas_per_node: float = 1.0
    pp_cross_node_wait_pressure: float = 0.0
    parameter_bytes: float = 0.0
    activation_bytes_per_microbatch: float = 0.0
    dp_exposed_fraction: float = 0.0
    node_count: int = 1
    gpus_per_node: int = 0
    pp_cross_node_edge_fraction: float = 0.0
    dp_cross_node_group_fraction: float = 0.0
    tp_cross_node_group_fraction: float = 0.0
    cross_node_pp_bytes: float = 0.0
    cross_node_dp_bytes: float = 0.0
    cross_node_tp_bytes: float = 0.0
    network_transport_label: str = ""
    network_effective_bandwidth_gbps: float = 0.0
    network_jitter_cv: float = 0.0
    network_large_message_jitter_cv: float = 0.0
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    data_parallel_size: int = 1

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


def _global_rank(pipe_index: int, data_index: int, model_index: int, workload: WorkloadFeatures) -> int:
    # Megatron rank assignment: rank = tp_rank + tp_size * (pp_rank + pp_size * dp_rank)
    return model_index + workload.tensor_model_parallel_size * (pipe_index + workload.pipeline_model_parallel_size * data_index)


def _node_index_for_rank(global_rank: int, workload: WorkloadFeatures) -> int:
    gpus_per_node = max(int(workload.gpus_per_node), 1)
    return global_rank // gpus_per_node


def _tp_cross_node_group_fraction(workload: WorkloadFeatures) -> float:
    if workload.node_count <= 1 or workload.tensor_model_parallel_size <= 1 or workload.gpus_per_node <= 0:
        return 0.0
    total = 0.0
    count = 0
    for pipe_index in range(workload.pipeline_model_parallel_size):
        for data_index in range(workload.data_parallel_size):
            nodes = {
                _node_index_for_rank(_global_rank(pipe_index, data_index, model_index, workload), workload)
                for model_index in range(workload.tensor_model_parallel_size)
            }
            total += (len(nodes) - 1) / max(workload.tensor_model_parallel_size - 1, 1)
            count += 1
    return total / max(count, 1)


def _dp_cross_node_group_fraction(workload: WorkloadFeatures) -> float:
    if workload.node_count <= 1 or workload.data_parallel_size <= 1 or workload.gpus_per_node <= 0:
        return 0.0
    total = 0.0
    count = 0
    for pipe_index in range(workload.pipeline_model_parallel_size):
        for model_index in range(workload.tensor_model_parallel_size):
            nodes = {
                _node_index_for_rank(_global_rank(pipe_index, data_index, model_index, workload), workload)
                for data_index in range(workload.data_parallel_size)
            }
            total += (len(nodes) - 1) / max(workload.data_parallel_size - 1, 1)
            count += 1
    return total / max(count, 1)


def _pp_cross_node_edge_fraction(workload: WorkloadFeatures) -> float:
    if workload.node_count <= 1 or workload.pipeline_model_parallel_size <= 1 or workload.gpus_per_node <= 0:
        return 0.0
    cross_edges = 0
    total_edges = 0
    for pipe_index in range(workload.pipeline_model_parallel_size - 1):
        for data_index in range(workload.data_parallel_size):
            for model_index in range(workload.tensor_model_parallel_size):
                left_rank = _global_rank(pipe_index, data_index, model_index, workload)
                right_rank = _global_rank(pipe_index + 1, data_index, model_index, workload)
                total_edges += 1
                if _node_index_for_rank(left_rank, workload) != _node_index_for_rank(right_rank, workload):
                    cross_edges += 1
    return cross_edges / max(total_edges, 1)


def derive_model_features(hardware: HardwareFeatures, workload: WorkloadFeatures) -> DerivedModelFeatures:
    bytes_per_element = _precision_bytes(workload.precision_mode)
    tokens_per_step = float(workload.global_batch_size * workload.seq_length)
    samples_per_step = float(workload.global_batch_size)

    # Attention params with GQA support: q_proj + k_proj + v_proj + o_proj
    # k_proj and v_proj scale by (num_kv_heads / num_attention_heads)
    kv_ratio = float(workload.num_key_value_heads) / max(float(workload.num_attention_heads), 1.0)
    approx_attention_params = float(
        workload.hidden_size * workload.hidden_size * (2.0 + 2.0 * kv_ratio)
    )
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
    activation_bytes_per_microbatch = (
        workload.micro_batch_size
        * workload.seq_length
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
    dp_exposed_fraction = _clamp(dp_penalty - dp_overlapable_fraction, 0.0, 1.0)

    node_count = max(int(workload.node_count), 1)
    replicas_per_node = workload.data_parallel_size / max(float(node_count), 1.0)
    pp_cross_node_edge_fraction = _pp_cross_node_edge_fraction(workload)
    dp_cross_node_group_fraction = _dp_cross_node_group_fraction(workload)
    tp_cross_node_group_fraction = _tp_cross_node_group_fraction(workload)
    cross_node_pp_bytes = 0.0
    cross_node_dp_bytes = 0.0
    cross_node_tp_bytes = 0.0
    pp_cross_node_wait_pressure = 0.0
    if node_count > 1:
        cross_node_pp_bytes = (
            activation_bytes_per_microbatch
            * microbatches_per_step
            * pipeline_exposed_fraction
            * max(workload.pipeline_model_parallel_size - 1, 0)
            * pp_cross_node_edge_fraction
        )
        # DP allreduce overlaps with backward pass computation. More microbatches
        # means a longer backward pass and more hiding opportunity → exposure ∝ 1/m.
        dp_cross_node_exposure = dp_cross_node_group_fraction / max(microbatches_per_step, 1.0)
        cross_node_dp_bytes = (
            (parameter_bytes / max(workload.tensor_model_parallel_size, 1))
            * dp_cross_node_exposure
        )
        cross_node_tp_bytes = (
            parameter_bytes
            * microbatches_per_step
            * tp_sync_fraction
            * tp_cross_node_group_fraction
        )
        if pp_cross_node_edge_fraction > 0.0:
            pp_cross_node_wait_pressure = replicas_per_node * pipeline_bubble_fraction

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
        microbatches_per_step=microbatches_per_step,
        pipeline_bubble_fraction=pipeline_bubble_fraction,
        replicas_per_node=replicas_per_node,
        pp_cross_node_wait_pressure=pp_cross_node_wait_pressure,
        parameter_bytes=parameter_bytes,
        activation_bytes_per_microbatch=activation_bytes_per_microbatch,
        dp_exposed_fraction=dp_exposed_fraction,
        node_count=node_count,
        gpus_per_node=workload.gpus_per_node,
        pp_cross_node_edge_fraction=pp_cross_node_edge_fraction,
        dp_cross_node_group_fraction=dp_cross_node_group_fraction,
        tp_cross_node_group_fraction=tp_cross_node_group_fraction,
        cross_node_pp_bytes=cross_node_pp_bytes,
        cross_node_dp_bytes=cross_node_dp_bytes,
        cross_node_tp_bytes=cross_node_tp_bytes,
        tensor_model_parallel_size=workload.tensor_model_parallel_size,
        pipeline_model_parallel_size=workload.pipeline_model_parallel_size,
        data_parallel_size=workload.data_parallel_size,
    )
