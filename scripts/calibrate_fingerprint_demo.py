#!/usr/bin/env python3
"""Demo script: calibrate HardwareFingerprint from hard-coded observed data.

This is a transitional tool to validate the physics-driven derivation layer.
It uses the same OBSERVED data as predict_unified_v2.py and searches for the
optimal HardwareFingerprint per platform via grid search.

Usage:
    python scripts/calibrate_fingerprint_demo.py

Output:
    Prints the calibrated fingerprint + validation MAPE for each hardware platform.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.freq_model.hardware import HardwareFeatures, HardwareFingerprint
from analysis.freq_model.network import NetworkConfig
from analysis.freq_model.workload import WorkloadFeatures, ObservedMetrics, LoadedRunSample
from analysis.freq_model.features import derive_model_features
from analysis.freq_model.calibrate import calibrate_hardware_fingerprint

# ---------------------------------------------------------------------------
# Hardware definitions (same as predict_unified_v2.py)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Observed data (from predict_unified_v2.py)
# ---------------------------------------------------------------------------
OBSERVED = {
    ("rtx4080s", "qwen7b", 1005): {"time_s": 267.6, "power_w": 213.7},
    ("rtx4080s", "qwen7b", 1155): {"time_s": 262.0, "power_w": 221.9},
    ("rtx4080s", "qwen7b", 1305): {"time_s": 254.4, "power_w": 231.6},
    ("rtx4080s", "qwen7b", 1455): {"time_s": 248.9, "power_w": 245.1},
    ("rtx4080s", "qwen7b", 1650): {"time_s": 238.3, "power_w": 264.0},
    ("rtx4080s", "qwen7b", 1800): {"time_s": 239.2, "power_w": 276.2},
    ("rtx4080s", "qwen7b", 1950): {"time_s": 237.6, "power_w": 289.1},
    ("rtx4080s", "qwen7b", 2205): {"time_s": 239.1, "power_w": 303.1},
    ("rtx4080s", "qwen7b", 2505): {"time_s": 229.5, "power_w": 320.2},
    ("rtx4080s", "llama7b", 1200): {"time_s": 299.2, "power_w": 212.7},
    ("rtx4080s", "llama7b", 1650): {"time_s": 300.8, "power_w": 219.9},
    ("rtx4080s", "llama7b", 1800): {"time_s": 276.1, "power_w": 226.2},
    ("rtx4080s", "llama7b", 2505): {"time_s": 281.0, "power_w": 309.1},
    ("v100", "llama7b", 1260): {"time_s": 620.5, "power_w": 2118.0},
    ("v100", "llama7b", 1380): {"time_s": 659.0, "power_w": 2769.0},
    ("v100", "qwen7b", 1260): {"time_s": 717.6, "power_w": 2013.0},
    ("v100", "qwen7b", 1380): {"time_s": 724.6, "power_w": 2685.0},
}


def _make_samples(hw_key: str, model_key: str, hardware: HardwareFeatures, workload: WorkloadFeatures):
    """Convert OBSERVED dict entries into LoadedRunSample list."""
    features = derive_model_features(hardware, workload)
    samples = []
    for (h, m, freq), data in OBSERVED.items():
        if h != hw_key or m != model_key:
            continue
        observed = ObservedMetrics(
            iteration=1,
            frequency_mhz=freq,
            num_steps=20,
            time_s=data["time_s"],
            step_time_s=data["time_s"] / 20.0,
            avg_power_w=data["power_w"],
            energy_j=data["power_w"] * data["time_s"],
            energy_wh=data["power_w"] * data["time_s"] / 3600.0,
            interval_tokens=features.tokens_per_step * 20,
            interval_samples=features.samples_per_step * 20,
            throughput_tokens_per_s=(features.tokens_per_step * 20) / data["time_s"],
            throughput_samples_per_s=(features.samples_per_step * 20) / data["time_s"],
            tokens_per_j=(features.tokens_per_step * 20) / (data["power_w"] * data["time_s"]),
            tokens_per_wh=(features.tokens_per_step * 20) / (data["power_w"] * data["time_s"] / 3600.0),
            samples_per_wh=(features.samples_per_step * 20) / (data["power_w"] * data["time_s"] / 3600.0),
        )
        samples.append(LoadedRunSample(
            run_id=f"{hw_key}_{model_key}_{freq}",
            run_dir=Path("."),
            run_payload={},
            workload=workload,
            observed=observed,
        ))
    return samples, [features] * len(samples)


def main():
    # ---- RTX 4080S: joint calibration from Qwen + LLaMA ----
    print("=" * 70)
    print("RTX 4080S HardwareFingerprint Calibration")
    print("=" * 70)

    qwen_samples, qwen_features = _make_samples("rtx4080s", "qwen7b", RTX4080S, QWEN7B_4080S)
    llama_samples, llama_features = _make_samples("rtx4080s", "llama7b", RTX4080S, LLAMA7B_4080S)
    all_samples = qwen_samples + llama_samples
    all_features = qwen_features + llama_features

    network = NetworkConfig(transport_type="ethernet", effective_bandwidth_gbps=0.2075)
    fingerprint, result = calibrate_hardware_fingerprint(
        all_samples, RTX4080S, all_features, network,
        baseline_sample=qwen_samples[-1],  # 2505 MHz as baseline
    )

    print(f"\nCalibrated fingerprint:")
    for k, v in fingerprint.to_dict().items():
        print(f"  {k}: {v}")
    print(f"\nValidation MAPE:")
    print(f"  Time MAPE:   {result.total_time_mape * 100:.2f}%")
    print(f"  Power MAPE:  {result.power_mape * 100:.2f}%")
    print(f"  Energy MAPE: {result.total_energy_mape * 100:.2f}%")
    print(f"  Objective:   {result.objective:.4f}")

    # ---- V100: joint calibration from LLaMA + Qwen ----
    print("\n" + "=" * 70)
    print("V100 HardwareFingerprint Calibration")
    print("=" * 70)

    llama_samples_v100, llama_features_v100 = _make_samples("v100", "llama7b", V100, LLAMA7B_V100)
    qwen_samples_v100, qwen_features_v100 = _make_samples("v100", "qwen7b", V100, QWEN7B_V100)
    all_samples_v100 = llama_samples_v100 + qwen_samples_v100
    all_features_v100 = llama_features_v100 + qwen_features_v100

    network_v100 = NetworkConfig(transport_type="nvlink", effective_bandwidth_gbps=50.0)
    fingerprint_v100, result_v100 = calibrate_hardware_fingerprint(
        all_samples_v100, V100, all_features_v100, network_v100,
        baseline_sample=llama_samples_v100[-1],  # 1380 MHz as baseline
    )

    print(f"\nCalibrated fingerprint:")
    for k, v in fingerprint_v100.to_dict().items():
        print(f"  {k}: {v}")
    print(f"\nValidation MAPE:")
    print(f"  Time MAPE:   {result_v100.total_time_mape * 100:.2f}%")
    print(f"  Power MAPE:  {result_v100.power_mape * 100:.2f}%")
    print(f"  Energy MAPE: {result_v100.total_energy_mape * 100:.2f}%")
    print(f"  Objective:   {result_v100.objective:.4f}")


if __name__ == "__main__":
    main()
