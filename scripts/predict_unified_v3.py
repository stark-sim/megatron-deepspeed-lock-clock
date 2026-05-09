#!/usr/bin/env python3
"""Unified predictor v3: physics-driven derivation, hardware-agnostic prediction.

Core principle: CalibrationParams are DERIVED from hardware specs + workload
features + network config + HardwareFingerprint.  No per-scenario hard-coding.

Changing the model (e.g., Qwen2-0.5B) or topology (e.g., dual-node 2×2 cards)
automatically produces new predictions via derive_calibration_params().

The HardwareFingerprint is calibrated once per hardware platform from observed
sweeps and reused for any model/topology on that platform.
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.freq_model.hardware import HardwareFeatures, HardwareFingerprint
from analysis.freq_model.network import NetworkConfig
from analysis.freq_model.workload import WorkloadFeatures
from analysis.freq_model.features import DerivedModelFeatures, derive_model_features
from analysis.freq_model.model import (
    CalibrationParams,
    derive_calibration_params,
    predict_point,
    sweep_prediction_points,
    build_continuous_grid,
)

# =============================================================================
# Hardware definitions
# =============================================================================

RTX4080S = HardwareFeatures(
    gpu_name="NVIDIA GeForce RTX 4080 SUPER",
    gpu_count=8,
    supported_frequency_mhz=list(range(210, 3106, 15)),
    min_frequency_mhz=210,
    max_frequency_mhz=2505,
    power_limit_w_per_gpu=320.0,
    peak_fp16_tensor_tflops_per_gpu=50.0,
    peak_fp32_tflops_per_gpu=25.0,
    memory_bandwidth_gbps_per_gpu=736.0,
)

RTX4080S_DUAL2 = HardwareFeatures(
    gpu_name="NVIDIA GeForce RTX 4080 SUPER",
    gpu_count=4,
    supported_frequency_mhz=list(range(210, 3106, 15)),
    min_frequency_mhz=210,
    max_frequency_mhz=2505,
    power_limit_w_per_gpu=320.0,
    peak_fp16_tensor_tflops_per_gpu=50.0,
    peak_fp32_tflops_per_gpu=25.0,
    memory_bandwidth_gbps_per_gpu=736.0,
)

RTX4080S_BASELINE = HardwareFeatures(
    gpu_name="NVIDIA GeForce RTX 4080 SUPER",
    gpu_count=8,
    supported_frequency_mhz=list(range(210, 3106, 15)),
    min_frequency_mhz=210,
    max_frequency_mhz=3105,
    power_limit_w_per_gpu=320.0,
    peak_fp16_tensor_tflops_per_gpu=50.0,
    peak_fp32_tflops_per_gpu=25.0,
    memory_bandwidth_gbps_per_gpu=736.0,
)

V100 = HardwareFeatures(
    gpu_name="Tesla V100-SXM3-32GB",
    gpu_count=16,
    supported_frequency_mhz=list(range(765, 1530 + 1, 15)),
    min_frequency_mhz=765,
    max_frequency_mhz=1530,
    power_limit_w_per_gpu=300.0,
    peak_fp16_tensor_tflops_per_gpu=125.0,
    peak_fp32_tflops_per_gpu=15.7,
    memory_bandwidth_gbps_per_gpu=900.0,
)

V100_SINGLE8 = HardwareFeatures(
    gpu_name="Tesla V100-SXM3-32GB",
    gpu_count=8,
    supported_frequency_mhz=list(range(765, 1530 + 1, 15)),
    min_frequency_mhz=765,
    max_frequency_mhz=1530,
    power_limit_w_per_gpu=300.0,
    peak_fp16_tensor_tflops_per_gpu=125.0,
    peak_fp32_tflops_per_gpu=15.7,
    memory_bandwidth_gbps_per_gpu=900.0,
)

V100_DUAL8 = HardwareFeatures(
    gpu_name="Tesla V100-SXM3-32GB",
    gpu_count=8,
    supported_frequency_mhz=list(range(765, 1530 + 1, 15)),
    min_frequency_mhz=765,
    max_frequency_mhz=1530,
    power_limit_w_per_gpu=300.0,
    peak_fp16_tensor_tflops_per_gpu=125.0,
    peak_fp32_tflops_per_gpu=15.7,
    memory_bandwidth_gbps_per_gpu=900.0,
)

# =============================================================================
# Pre-calibrated HardwareFingerprints
#
# Calibrated via calibrate_hardware_fingerprint() from observed sweeps:
#   - RTX 4080S: joint Qwen 9-pt + LLaMA 4-pt (dual-node 8-GPU, Ethernet)
#   - V100:      joint LLaMA 2-pt + Qwen 2-pt (single-node 16-GPU, NVLink)
# =============================================================================

RTX4080S_FINGERPRINT = HardwareFingerprint(
    compute_efficiency=0.30,
    memory_efficiency=0.005,
    network_efficiency=0.05,
    static_power_w=212.7,
    dynamic_power_w=161.25,
    dynamic_power_exponent=1.5,
    power_utilization_exponent=0.5,
    thermal_throttle_threshold=0.70,
    thermal_throttle_coefficient=0.65,
)

V100_FINGERPRINT = HardwareFingerprint(
    compute_efficiency=0.02,
    memory_efficiency=0.80,
    network_efficiency=0.20,
    static_power_w=1409.1,
    dynamic_power_w=1134.0,
    dynamic_power_exponent=5.0,
    power_utilization_exponent=0.5,
    thermal_throttle_threshold=0.80,
    thermal_throttle_coefficient=0.30,
)

# Network configs
ETHERNET_02GBPS = NetworkConfig(transport_type="ethernet", effective_bandwidth_gbps=0.2075)
NVLINK_50GBPS = NetworkConfig(transport_type="nvlink", effective_bandwidth_gbps=50.0)
IB_18GBPS = NetworkConfig(transport_type="ib", effective_bandwidth_gbps=18.8)

# =============================================================================
# Workload definitions (existing + new)
# =============================================================================

QWEN7B_4080S = WorkloadFeatures(
    num_layers=28, hidden_size=3584, ffn_hidden_size=18944,
    num_attention_heads=28, num_key_value_heads=4,
    seq_length=2048, micro_batch_size=1, global_batch_size=4, train_iters=20,
    tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=2,
    zero_stage=1, precision_mode="bf16", swiglu=True, node_count=2, gpus_per_node=4,
)

LLAMA7B_4080S = WorkloadFeatures(
    num_layers=32, hidden_size=4096, ffn_hidden_size=11008,
    num_attention_heads=32, num_key_value_heads=32,
    seq_length=2048, micro_batch_size=1, global_batch_size=4, train_iters=20,
    tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=2,
    zero_stage=1, precision_mode="bf16", swiglu=True, node_count=2, gpus_per_node=4,
)

QWEN7B_V100 = WorkloadFeatures(
    num_layers=28, hidden_size=3584, ffn_hidden_size=18944,
    num_attention_heads=28, num_key_value_heads=4,
    seq_length=2048, micro_batch_size=1, global_batch_size=16, train_iters=20,
    tensor_model_parallel_size=1, pipeline_model_parallel_size=2, data_parallel_size=8,
    zero_stage=1, precision_mode="bf16", swiglu=True, node_count=1, gpus_per_node=16,
)

LLAMA7B_V100 = WorkloadFeatures(
    num_layers=32, hidden_size=4096, ffn_hidden_size=11008,
    num_attention_heads=32, num_key_value_heads=32,
    seq_length=2048, micro_batch_size=1, global_batch_size=16, train_iters=20,
    tensor_model_parallel_size=1, pipeline_model_parallel_size=2, data_parallel_size=8,
    zero_stage=1, precision_mode="bf16", swiglu=True, node_count=1, gpus_per_node=16,
)

# ---- V100 8-GPU topologies (for cross-topology generalization validation) ----
LLAMA7B_V100_SINGLE8 = WorkloadFeatures(
    num_layers=32, hidden_size=4096, ffn_hidden_size=11008,
    num_attention_heads=32, num_key_value_heads=32,
    seq_length=2048, micro_batch_size=1, global_batch_size=4, train_iters=20,
    tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=2,
    zero_stage=1, precision_mode="bf16", swiglu=True, node_count=1, gpus_per_node=8,
)

LLAMA7B_V100_DUAL8 = WorkloadFeatures(
    num_layers=32, hidden_size=4096, ffn_hidden_size=11008,
    num_attention_heads=32, num_key_value_heads=32,
    seq_length=2048, micro_batch_size=1, global_batch_size=4, train_iters=20,
    tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=2,
    zero_stage=1, precision_mode="bf16", swiglu=True, node_count=2, gpus_per_node=4,
)

QWEN7B_V100_DUAL8 = WorkloadFeatures(
    num_layers=28, hidden_size=3584, ffn_hidden_size=18944,
    num_attention_heads=28, num_key_value_heads=4,
    seq_length=2048, micro_batch_size=1, global_batch_size=4, train_iters=20,
    tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=2,
    zero_stage=1, precision_mode="bf16", swiglu=True, node_count=2, gpus_per_node=4,
)

# ---- NEW: Qwen2-0.5B (hypothetical scenario) ----
QWEN2_05B_4080S = WorkloadFeatures(
    num_layers=24, hidden_size=896, ffn_hidden_size=4864,
    num_attention_heads=14, num_key_value_heads=2,
    seq_length=2048, micro_batch_size=1, global_batch_size=4, train_iters=20,
    tensor_model_parallel_size=2, pipeline_model_parallel_size=1, data_parallel_size=2,
    zero_stage=1, precision_mode="bf16", swiglu=True, node_count=2, gpus_per_node=2,
)

# ---- NEW: Qwen2.5-1.5B + dual-node 2×2 cards ----
QWEN15B_4080S_DUAL2 = WorkloadFeatures(
    num_layers=28, hidden_size=1536, ffn_hidden_size=8960,
    num_attention_heads=12, num_key_value_heads=2,
    seq_length=2048, micro_batch_size=1, global_batch_size=4, train_iters=20,
    tensor_model_parallel_size=2, pipeline_model_parallel_size=2, data_parallel_size=1,
    zero_stage=1, precision_mode="bf16", swiglu=True, node_count=2, gpus_per_node=2,
)

# =============================================================================
# Prediction engine
# =============================================================================

def predict_scenario(
    hardware: HardwareFeatures,
    workload: WorkloadFeatures,
    network: NetworkConfig,
    fingerprint: HardwareFingerprint,
    mode: str = "static",
) -> Tuple[List[Dict], DerivedModelFeatures, CalibrationParams]:
    """Predict a full frequency sweep for an arbitrary hardware+workload+network scenario.

    Returns:
        points: List of prediction dicts (one per supported frequency)
        features: DerivedModelFeatures (for inspection)
        params: CalibrationParams (for inspection)
    """
    features = derive_model_features(hardware, workload)
    params = derive_calibration_params(hardware, features, network, fingerprint)

    # For baseline mode, use theoretical boost ceiling hardware for thermal
    # throttle calculation, but keep the same frequency sweep range as static.
    hw_for_pred = hardware
    if mode == "baseline":
        if hardware.max_frequency_mhz == 2505 and "4080" in hardware.gpu_name.lower():
            hw_for_pred = RTX4080S_BASELINE

    frequencies = build_continuous_grid(hardware, step_mhz=15)
    frequencies = [f for f in frequencies if f >= hardware.min_frequency_mhz]

    points = sweep_prediction_points(frequencies, hw_for_pred, features, params, mode=mode)
    return [p.to_dict() for p in points], features, params


def print_prediction_table(points: List[Dict], title: str):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"{'Freq(MHz)':>10} {'Time(s)':>10} {'Power(W)':>10} {'Energy(J)':>12} {'tok/J':>8}")
    print("-" * 70)
    for p in points:
        print(
            f"{p['frequency_mhz']:>10} "
            f"{p['step_time_s'] * 20:>10.1f} "
            f"{p['power_w']:>10.1f} "
            f"{p['power_w'] * p['step_time_s'] * 20:>12.0f} "
            f"{p['tokens_per_j']:>8.3f}"
        )


def main():
    print("Unified Predictor v3 — Physics-Driven Derivation")
    print("=" * 70)

    # ---- Scenario 1: 4080S + Qwen7B (baseline validation) ----
    points, features, params = predict_scenario(
        RTX4080S, QWEN7B_4080S, ETHERNET_02GBPS, RTX4080S_FINGERPRINT, mode="static"
    )
    print_prediction_table(points, "RTX 4080S + Qwen2.5-7B (dual-node 8-GPU, Ethernet)")
    print(f"\nDerived limits:  compute={params.compute_limit_at_max_tokens_per_s:,.0f}  "
          f"memory={params.memory_limit_tokens_per_s:,.0f}  comm={params.communication_limit_tokens_per_s:,.0f}")
    print(f"Power model:     static={params.static_power_w:.1f}W  dynamic={params.dynamic_power_w:.1f}W  "
          f"exp={params.dynamic_power_exponent:.1f}  pue={params.power_utilization_exponent:.1f}")

    # ---- Scenario 2: V100 + LLaMA7B (baseline validation) ----
    points_v100, features_v100, params_v100 = predict_scenario(
        V100, LLAMA7B_V100, NVLINK_50GBPS, V100_FINGERPRINT, mode="static"
    )
    print_prediction_table(points_v100, "V100 + LLaMA-7B (single-node 16-GPU, NVLink)")
    print(f"\nDerived limits:  compute={params_v100.compute_limit_at_max_tokens_per_s:,.0f}  "
          f"memory={params_v100.memory_limit_tokens_per_s:,.0f}  comm={params_v100.communication_limit_tokens_per_s:,.0f}")
    print(f"Power model:     static={params_v100.static_power_w:.1f}W  dynamic={params_v100.dynamic_power_w:.1f}W  "
          f"exp={params_v100.dynamic_power_exponent:.1f}  pue={params_v100.power_utilization_exponent:.1f}")

    # ---- Scenario 3: NEW — 4080S + Qwen2-0.5B (dual-node 2×2 cards) ----
    points_new, features_new, params_new = predict_scenario(
        RTX4080S, QWEN2_05B_4080S, ETHERNET_02GBPS, RTX4080S_FINGERPRINT, mode="static"
    )
    print_prediction_table(points_new, "RTX 4080S + Qwen2-0.5B (dual-node 2×2 cards, Ethernet)")
    print(f"\nDerived limits:  compute={params_new.compute_limit_at_max_tokens_per_s:,.0f}  "
          f"memory={params_new.memory_limit_tokens_per_s:,.0f}  comm={params_new.communication_limit_tokens_per_s:,.0f}")
    print(f"Power model:     static={params_new.static_power_w:.1f}W  dynamic={params_new.dynamic_power_w:.1f}W  "
          f"exp={params_new.dynamic_power_exponent:.1f}  pue={params_new.power_utilization_exponent:.1f}")
    print(f"\nModel features:")
    print(f"  approx_params:     {features_new.approx_model_params / 1e6:.2f}M")
    print(f"  flops_per_token:   {features_new.approx_flops_per_token / 1e9:.2f}B")
    print(f"  comm_share:        {features_new.communication_share:.2%}")
    print(f"  cross_node_dp:     {features_new.cross_node_dp_bytes / 1e6:.1f} MB")
    print(f"  cross_node_tp:     {features_new.cross_node_tp_bytes / 1e6:.1f} MB")

    # ---- Scenario 4: NEW — 4080S + Qwen2.5-1.5B (dual-node 2×2 cards, TP2PP2DP1) ----
    points_15b, features_15b, params_15b = predict_scenario(
        RTX4080S_DUAL2, QWEN15B_4080S_DUAL2, ETHERNET_02GBPS, RTX4080S_FINGERPRINT, mode="static"
    )
    print_prediction_table(points_15b, "RTX 4080S + Qwen2.5-1.5B (dual-node 2×2 cards, Ethernet)")
    print(f"\nDerived limits:  compute={params_15b.compute_limit_at_max_tokens_per_s:,.0f}  "
          f"memory={params_15b.memory_limit_tokens_per_s:,.0f}  comm={params_15b.communication_limit_tokens_per_s:,.0f}")
    print(f"Power model:     static={params_15b.static_power_w:.1f}W  dynamic={params_15b.dynamic_power_w:.1f}W  "
          f"exp={params_15b.dynamic_power_exponent:.1f}  pue={params_15b.power_utilization_exponent:.1f}")
    print(f"\nModel features:")
    print(f"  approx_params:     {features_15b.approx_model_params / 1e6:.2f}M")
    print(f"  flops_per_token:   {features_15b.approx_flops_per_token / 1e9:.2f}B")
    print(f"  comm_share:        {features_15b.communication_share:.2%}")
    print(f"  cross_node_dp:     {features_15b.cross_node_dp_bytes / 1e6:.1f} MB")
    print(f"  cross_node_tp:     {features_15b.cross_node_tp_bytes / 1e6:.1f} MB")

    # ---- Baseline vs Static comparison for new scenario ----
    points_baseline, _, _ = predict_scenario(
        RTX4080S, QWEN2_05B_4080S, ETHERNET_02GBPS, RTX4080S_FINGERPRINT, mode="baseline"
    )
    static_max = points_new[-1]  # highest frequency static point
    baseline_max = points_baseline[-1]
    print(f"\n--- Baseline vs Static (Qwen2-0.5B @ max freq) ---")
    print(f"Static:   {static_max['step_time_s']*20:.1f}s  {static_max['power_w']:.1f}W")
    print(f"Baseline: {baseline_max['step_time_s']*20:.1f}s  {baseline_max['power_w']:.1f}W")
    print(f"Theta:    {baseline_max['throughput_tokens_per_s'] / static_max['throughput_tokens_per_s']:.3f}")

    # ---- Baseline vs Static comparison for 1.5B scenario ----
    points_15b_baseline, _, _ = predict_scenario(
        RTX4080S_DUAL2, QWEN15B_4080S_DUAL2, ETHERNET_02GBPS, RTX4080S_FINGERPRINT, mode="baseline"
    )
    static_15b_max = points_15b[-1]
    baseline_15b_max = points_15b_baseline[-1]
    print(f"\n--- Baseline vs Static (Qwen2.5-1.5B @ max freq) ---")
    print(f"Static:   {static_15b_max['step_time_s']*20:.1f}s  {static_15b_max['power_w']:.1f}W")
    print(f"Baseline: {baseline_15b_max['step_time_s']*20:.1f}s  {baseline_15b_max['power_w']:.1f}W")
    print(f"Theta:    {baseline_15b_max['throughput_tokens_per_s'] / static_15b_max['throughput_tokens_per_s']:.3f}")

    # =============================================================================
    # V100 Cross-Topology Generalization Validation
    # =============================================================================
    print("\n" + "=" * 70)
    print("V100 Cross-Topology Generalization Validation")
    print("=" * 70)

    # ---- Scenario A: LLaMA7B single-node 8-GPU (NVLink) ----
    points_v100_single8, features_v100_single8, params_v100_single8 = predict_scenario(
        V100_SINGLE8, LLAMA7B_V100_SINGLE8, NVLINK_50GBPS, V100_FINGERPRINT, mode="static"
    )
    print_prediction_table(points_v100_single8, "V100 + LLaMA-7B (single-node 8-GPU, NVLink)")
    print(f"\nDerived limits:  compute={params_v100_single8.compute_limit_at_max_tokens_per_s:,.0f}  "
          f"memory={params_v100_single8.memory_limit_tokens_per_s:,.0f}  "
          f"comm={params_v100_single8.communication_limit_tokens_per_s:,.0f}")
    print(f"Model features:")
    print(f"  comm_share:        {features_v100_single8.communication_share:.2%}")
    print(f"  pipeline_exposed:  {features_v100_single8.pipeline_exposed_fraction:.2%}")
    print(f"  tp_sync:           {features_v100_single8.tp_sync_fraction:.2%}")

    # ---- Scenario B: LLaMA7B dual-node 8-GPU (IB) ----
    points_v100_dual8_llama, features_v100_dual8_llama, params_v100_dual8_llama = predict_scenario(
        V100_DUAL8, LLAMA7B_V100_DUAL8, IB_18GBPS, V100_FINGERPRINT, mode="static"
    )
    print_prediction_table(points_v100_dual8_llama, "V100 + LLaMA-7B (dual-node 8-GPU, IB)")
    print(f"\nDerived limits:  compute={params_v100_dual8_llama.compute_limit_at_max_tokens_per_s:,.0f}  "
          f"memory={params_v100_dual8_llama.memory_limit_tokens_per_s:,.0f}  "
          f"comm={params_v100_dual8_llama.communication_limit_tokens_per_s:,.0f}")
    print(f"Model features:")
    print(f"  comm_share:        {features_v100_dual8_llama.communication_share:.2%}")
    print(f"  cross_node_dp:     {features_v100_dual8_llama.cross_node_dp_bytes / 1e6:.1f} MB")
    print(f"  cross_node_tp:     {features_v100_dual8_llama.cross_node_tp_bytes / 1e6:.1f} MB")

    # ---- Scenario C: Qwen7B dual-node 8-GPU (IB) ----
    points_v100_dual8_qwen, features_v100_dual8_qwen, params_v100_dual8_qwen = predict_scenario(
        V100_DUAL8, QWEN7B_V100_DUAL8, IB_18GBPS, V100_FINGERPRINT, mode="static"
    )
    print_prediction_table(points_v100_dual8_qwen, "V100 + Qwen2.5-7B (dual-node 8-GPU, IB)")
    print(f"\nDerived limits:  compute={params_v100_dual8_qwen.compute_limit_at_max_tokens_per_s:,.0f}  "
          f"memory={params_v100_dual8_qwen.memory_limit_tokens_per_s:,.0f}  "
          f"comm={params_v100_dual8_qwen.communication_limit_tokens_per_s:,.0f}")
    print(f"Model features:")
    print(f"  comm_share:        {features_v100_dual8_qwen.communication_share:.2%}")
    print(f"  cross_node_dp:     {features_v100_dual8_qwen.cross_node_dp_bytes / 1e6:.1f} MB")
    print(f"  cross_node_tp:     {features_v100_dual8_qwen.cross_node_tp_bytes / 1e6:.1f} MB")

    # ---- Baseline predictions for all three V100 scenarios ----
    for scenario_name, hw, wl, net in [
        ("LLaMA7B single-node 8-GPU", V100_SINGLE8, LLAMA7B_V100_SINGLE8, NVLINK_50GBPS),
        ("LLaMA7B dual-node 8-GPU", V100_DUAL8, LLAMA7B_V100_DUAL8, IB_18GBPS),
        ("Qwen7B dual-node 8-GPU", V100_DUAL8, QWEN7B_V100_DUAL8, IB_18GBPS),
    ]:
        pts_static, _, _ = predict_scenario(hw, wl, net, V100_FINGERPRINT, mode="static")
        pts_baseline, _, _ = predict_scenario(hw, wl, net, V100_FINGERPRINT, mode="baseline")
        s_max = pts_static[-1]
        b_max = pts_baseline[-1]
        theta = b_max['throughput_tokens_per_s'] / s_max['throughput_tokens_per_s']
        print(f"\n--- Baseline vs Static ({scenario_name}) ---")
        print(f"Static:   {s_max['step_time_s']*20:.1f}s  {s_max['power_w']:.1f}W")
        print(f"Baseline: {b_max['step_time_s']*20:.1f}s  {b_max['power_w']:.1f}W")
        print(f"Theta:    {theta:.3f}")

    # ---- Cross-topology comparison summary ----
    print("\n" + "=" * 70)
    print("Cross-Topology Comparison Summary (Selected Frequencies)")
    print("=" * 70)
    print(f"{'Topology':<40} {'1260MHz':>12} {'1380MHz':>12} {'1530MHz':>12}")
    print("-" * 80)
    for label, hw, wl, net in [
        ("LLaMA7B single-node 8-GPU (NVLink)", V100_SINGLE8, LLAMA7B_V100_SINGLE8, NVLINK_50GBPS),
        ("LLaMA7B dual-node 8-GPU (IB)", V100_DUAL8, LLAMA7B_V100_DUAL8, IB_18GBPS),
        ("Qwen7B dual-node 8-GPU (IB)", V100_DUAL8, QWEN7B_V100_DUAL8, IB_18GBPS),
    ]:
        pts, _, _ = predict_scenario(hw, wl, net, V100_FINGERPRINT, mode="static")
        t_1260 = next((p for p in pts if p['frequency_mhz'] == 1260), None)
        t_1380 = next((p for p in pts if p['frequency_mhz'] == 1380), None)
        t_1530 = next((p for p in pts if p['frequency_mhz'] == 1530), None)
        s_1260 = f"{t_1260['step_time_s']*20:.0f}s/{t_1260['power_w']:.0f}W" if t_1260 else "N/A"
        s_1380 = f"{t_1380['step_time_s']*20:.0f}s/{t_1380['power_w']:.0f}W" if t_1380 else "N/A"
        s_1530 = f"{t_1530['step_time_s']*20:.0f}s/{t_1530['power_w']:.0f}W" if t_1530 else "N/A"
        print(f"{label:<40} {s_1260:>12} {s_1380:>12} {s_1530:>12}")


if __name__ == "__main__":
    main()
