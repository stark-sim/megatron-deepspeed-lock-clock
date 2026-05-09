#!/usr/bin/env python3
"""Unified predictor v2: one physics model, all hardware platforms.

Core principle: same `analysis.freq_model` physics engine for both RTX 4080S and V100.
Model structural differences (layers, hidden size, attention type) are handled
automatically by `derive_model_features()`.  Hardware differences are captured by:
  1. HardwareFeatures (peak TFLOPS, memory BW, GPU count, topology, network)
  2. Calibrated util coefficients anchored to `infer_initial_anchors()`
  3. Power model shape (frequency exponent, utilization coupling)

Calibration strategy per hardware (NOT per model):
  - RTX 4080S: dual-node 8-GPU, Ethernet (~0.2 Gbps effective).
    Communication is the bottleneck → throughput barely changes with frequency.
    Power is frequency-dominated (GPUs stay active waiting for network).
    Calibrated from 9-freq Qwen sweep: time MAPE 0.83%, power MAPE 1.36%.
    LLaMA (same hardware) inherits the same params: time MAPE ~10%, power ~13%.
    The residual error comes from MHA vs GQA communication-pattern differences
    that `derive_model_features()` weights but cannot fully capture.
  - V100-SXM3: single-node 16-GPU, NVLink (~50 GB/s).
    Compute is closer to the bottleneck → power tracks utilization × frequency^5.
    Calibrated from 5-freq LLaMA sweep: time MAPE <5%, power MAPE <1%.
    Qwen inherits same params: time MAPE ~12% (embedding layer not counted).

Key insight: `power_utilization_exponent` (added to CalibrationParams) decouples
power from throughput when communication dominates (exponent=0 for 4080S), while
keeping the coupling for compute-bound cases (exponent=1 for V100).  This single
degree of freedom lets the same `predict_power_w()` formula cover both regimes.
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.freq_model.hardware import HardwareFeatures, resolve_hardware_spec
from analysis.freq_model.workload import WorkloadFeatures
from analysis.freq_model.features import derive_model_features
from analysis.freq_model.model import (
    CalibrationParams,
    infer_initial_anchors,
    predict_point,
)

# =============================================================================
# Hardware definitions
# =============================================================================

RTX4080S = HardwareFeatures(
    gpu_name="NVIDIA GeForce RTX 4080 SUPER",
    gpu_count=8,
    supported_frequency_mhz=list(range(210, 3106, 15)),
    min_frequency_mhz=210,
    max_frequency_mhz=2505,  # Experimental locking ceiling for static mode
    power_limit_w_per_gpu=320.0,
    peak_fp16_tensor_tflops_per_gpu=50.0,
    peak_fp32_tflops_per_gpu=25.0,
    memory_bandwidth_gbps_per_gpu=736.0,
)

# Baseline mode uses theoretical boost ceiling for thermal throttle calculation
RTX4080S_BASELINE = HardwareFeatures(
    gpu_name="NVIDIA GeForce RTX 4080 SUPER",
    gpu_count=8,
    supported_frequency_mhz=list(range(210, 3106, 15)),
    min_frequency_mhz=210,
    max_frequency_mhz=3105,  # Theoretical boost ceiling for baseline thermal throttle
    power_limit_w_per_gpu=320.0,
    peak_fp16_tensor_tflops_per_gpu=50.0,
    peak_fp32_tflops_per_gpu=25.0,
    memory_bandwidth_gbps_per_gpu=736.0,
)

V100_SPEC = resolve_hardware_spec("tesla v100-sxm3-32gb")
V100 = HardwareFeatures(
    gpu_name="Tesla V100-SXM3-32GB",
    gpu_count=16,
    supported_frequency_mhz=list(range(765, 1530 + 1, 15)),
    min_frequency_mhz=765,
    max_frequency_mhz=1530,
    power_limit_w_per_gpu=V100_SPEC.board_power_w_per_gpu,
    peak_fp16_tensor_tflops_per_gpu=V100_SPEC.peak_fp16_tensor_tflops_per_gpu,
    peak_fp32_tflops_per_gpu=V100_SPEC.peak_fp32_tflops_per_gpu,
    memory_bandwidth_gbps_per_gpu=V100_SPEC.memory_bandwidth_gbps_per_gpu,
)

# =============================================================================
# Workload definitions (model structure → derive_model_features handles the rest)
# =============================================================================

QWEN7B = WorkloadFeatures(
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

# =============================================================================
# Calibration parameter builders
#
# Each builder follows the same pattern:
#   1. Call infer_initial_anchors(hardware, features, observed_max_tput)
#   2. Apply hardware-specific util coefficients calibrated from observed data
#   3. Fit power model (P_static, P_dynamic, exponent) from power observations
# =============================================================================

def build_rtx4080s_params(features, model_label: str) -> CalibrationParams:
    """Build 4080S CalibrationParams.

    Calibrated from Qwen 9-freq sweep (1005–2505 MHz).
    Power model: P(f) = 155 + 180·(f/2505)^1.5  (frequency-only, no util coupling).
    Throughput: compute=7000, memory=600, comm=300, comm_penalty=0.75.
    """
    anchors = infer_initial_anchors(RTX4080S, features, observed_max_throughput_tokens_s=700.0)
    # Util coefficients derived from grid-search calibration:
    #   compute_limit = 9611 * 0.728 ≈ 7000
    #   memory_limit  = 1_220_245 * 0.00049 ≈ 600
    #   comm_limit    = 300  (Ethernet bottleneck, heuristic from observed throughput)
    return CalibrationParams(
        compute_limit_at_max_tokens_per_s=7000.0,
        memory_limit_tokens_per_s=600.0,
        communication_limit_tokens_per_s=300.0,
        communication_penalty=0.75,
        static_power_w=155.0,
        dynamic_power_w=180.0,
        dynamic_power_exponent=1.5,
        power_utilization_exponent=0.0,  # KEY: power independent of throughput
        throughput_saturation_ratio=1.0,
        thermal_throttle_threshold=0.7,
        thermal_throttle_coefficient=0.65,
        reference_total_gpu_count=8,
        reference_gpus_per_node=4,
        # NOTE: reference_pipeline_parallel_efficiency NOT set here,
        # matching the calibration script conditions (default=1.0).
    )


def build_v100_params(features, model_label: str) -> CalibrationParams:
    """Build V100 CalibrationParams.

    Calibrated from LLaMA 5-freq sweep (1260–1530 MHz).
    Power model: P(f) = 592 + 4069·(f/1530)^5.0  (utilization-coupled).
    Throughput: compute=1201, memory=623326, comm=7208.
    """
    anchors = infer_initial_anchors(V100, features, observed_max_throughput_tokens_s=1000.0)
    ca = anchors["compute_anchor"] or 50000.0
    ma = anchors["memory_anchor"] or 40000.0
    coa = anchors["communication_anchor"] or 30000.0
    # Util coefficients from grid-search calibration:
    #   compute_util=0.025, memory_util=0.10, comm_util=0.15
    return CalibrationParams(
        compute_limit_at_max_tokens_per_s=ca * 0.025,
        memory_limit_tokens_per_s=ma * 0.10,
        communication_limit_tokens_per_s=coa * 0.15,
        communication_penalty=0.01,
        static_power_w=592.0,
        dynamic_power_w=4069.0,
        dynamic_power_exponent=5.0,
        power_utilization_exponent=1.0,  # KEY: power tracks utilization
        throughput_saturation_ratio=1.0,
        thermal_throttle_threshold=0.80,
        thermal_throttle_coefficient=0.30,
        reference_total_gpu_count=16,
        reference_gpus_per_node=16,
        reference_pipeline_parallel_efficiency=features.pipeline_parallel_efficiency,
    )


# =============================================================================
# Observed data (for validation)
# =============================================================================

OBSERVED: Dict[Tuple[str, str, int], Dict[str, float]] = {
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

# =============================================================================
# Prediction engine
# =============================================================================

def predict_hw_model(hw_key: str, model_key: str, freq: int, mode: str = "static"):
    if hw_key == "rtx4080s":
        # Baseline mode uses theoretical boost ceiling for thermal throttle calc
        hardware = RTX4080S_BASELINE if mode == "baseline" else RTX4080S
        workload = QWEN7B if model_key == "qwen7b" else LLAMA7B_4080S
        builder = build_rtx4080s_params
    else:
        hardware = V100
        workload = QWEN7B_V100 if model_key == "qwen7b" else LLAMA7B_V100
        builder = build_v100_params

    features = derive_model_features(hardware, workload)
    params = builder(features, model_key)
    pred = predict_point(freq, hardware, features, params, mode=mode)
    return pred, features, params, workload


def run_comparison(hw_key: str, model_key: str):
    hw_name = "RTX 4080S (8×, dual-node, Ethernet)" if hw_key == "rtx4080s" else "V100 (16×, single-node, NVLink)"
    model_name = model_key.upper()

    # Print anchors
    hardware = RTX4080S if hw_key == "rtx4080s" else V100
    workload = (
        QWEN7B if (hw_key == "rtx4080s" and model_key == "qwen7b") else
        LLAMA7B_4080S if (hw_key == "rtx4080s" and model_key == "llama7b") else
        QWEN7B_V100 if (hw_key == "v100" and model_key == "qwen7b") else
        LLAMA7B_V100
    )
    features = derive_model_features(hardware, workload)
    anchors = infer_initial_anchors(hardware, features, observed_max_throughput_tokens_s=700.0 if hw_key == "rtx4080s" else 1000.0)

    print(f"\n{'='*100}")
    print(f"  {hw_name} | {model_name}")
    print(f"{'='*100}")
    print(f"  Hardware anchors from infer_initial_anchors():")
    print(f"    compute_anchor       = {anchors['compute_anchor']:,.0f} tok/s")
    print(f"    memory_anchor        = {anchors['memory_anchor']:,.0f} tok/s")
    print(f"    communication_anchor = {anchors['communication_anchor']:,.0f} tok/s")

    # Observed points
    observed_freqs = sorted(k[2] for k in OBSERVED if k[:2] == (hw_key, model_key))
    if not observed_freqs:
        print("  (no observed data)")
        return [], []

    results = []
    for freq in observed_freqs:
        pred_static, features, params, workload = predict_hw_model(hw_key, model_key, freq, mode="static")
        # Baseline prediction: same nominal frequency but with thermal throttling
        pred_baseline, _, _, _ = predict_hw_model(hw_key, model_key, freq, mode="baseline")
        obs = OBSERVED[(hw_key, model_key, freq)]
        results.append({
            "freq": freq,
            "pred_step_static": pred_static.step_time_s,
            "pred_step_baseline": pred_baseline.step_time_s,
            "pred_power_static": pred_static.power_w,
            "pred_power_baseline": pred_baseline.power_w,
            "pred_tok_j_static": pred_static.tokens_per_j,
            "pred_tok_j_baseline": pred_baseline.tokens_per_j,
            "obs_step": obs["time_s"],
            "obs_power": obs["power_w"],
        })

    # Static validation table
    print(f"\n  STATIC PREDICTIONS (fixed frequency, no thermal throttling)")
    print(f"  {'Freq':>6} | {'PredTotal':>10} | {'ObsTotal':>10} | {'TimeErr':>8} | "
          f"{'PredPow':>8} | {'ObsPow':>8} | {'PowErr':>7} | {'Tok/J':>7}")
    print("  " + "-" * 90)

    time_errs = []
    power_errs = []
    for r in results:
        pred_total = r["pred_step_static"] * workload.train_iters
        ste = (pred_total - r["obs_step"]) / r["obs_step"] * 100
        pe = (r["pred_power_static"] - r["obs_power"]) / r["obs_power"] * 100
        time_errs.append(abs(ste))
        power_errs.append(abs(pe))
        print(f"  {r['freq']:>6} | {pred_total:>9.1f}s | {r['obs_step']:>9.1f}s | {ste:>+7.1f}% | "
              f"{r['pred_power_static']:>7.1f}W | {r['obs_power']:>7.1f}W | {pe:>+6.1f}% | {r['pred_tok_j_static']:>6.2f}")

    print(f"\n  Time MAPE:  {sum(time_errs)/len(time_errs):.2f}%")
    print(f"  Power MAPE: {sum(power_errs)/len(power_errs):.2f}%")

    # Baseline vs Static comparison (at max observed frequency)
    max_freq = max(r["freq"] for r in results)
    max_r = next(r for r in results if r["freq"] == max_freq)
    static_total = max_r["pred_step_static"] * workload.train_iters
    baseline_total = max_r["pred_step_baseline"] * workload.train_iters
    static_power = max_r["pred_power_static"]
    baseline_power = max_r["pred_power_baseline"]
    static_energy = static_total * static_power
    baseline_energy = baseline_total * baseline_power
    static_tok_j = max_r["pred_tok_j_static"]
    baseline_tok_j = max_r["pred_tok_j_baseline"]

    # Thermal throttle factor at max frequency (use baseline hardware for correct ratio)
    baseline_hw = RTX4080S_BASELINE if hw_key == "rtx4080s" else V100
    freq_ratio = max_freq / baseline_hw.max_frequency_mhz
    overshoot = max(0.0, freq_ratio - params.thermal_throttle_threshold)
    max_overshoot = max(0.001, 1.0 - params.thermal_throttle_threshold)
    theta = 1.0 - params.thermal_throttle_coefficient * ((overshoot / max_overshoot) ** 2)
    theta = max(0.5, theta)

    print(f"\n  BASELINE vs STATIC COMPARISON @ {max_freq} MHz (full GPU count)")
    print(f"  {'Mode':>12} | {'TotalTime':>10} | {'Power':>8} | {'Energy':>10} | {'Tok/J':>7} | {'Throttle':>8}")
    print("  " + "-" * 72)
    print(f"  {'Static':>12} | {static_total:>9.1f}s | {static_power:>7.1f}W | {static_energy:>9.0f}J | {static_tok_j:>6.2f} | {'—':>8}")
    print(f"  {'Baseline':>12} | {baseline_total:>9.1f}s | {baseline_power:>7.1f}W | {baseline_energy:>9.0f}J | {baseline_tok_j:>6.2f} | {theta:>7.3f}θ")
    time_delta = (baseline_total / static_total - 1) * 100
    energy_delta = (baseline_energy / static_energy - 1) * 100
    tok_j_delta = (baseline_tok_j / static_tok_j - 1) * 100
    print(f"  {'BaselineΔ':>12} | {time_delta:>+8.1f}% | {'—':>8} | {energy_delta:>+8.1f}% | {tok_j_delta:>+6.1f}% | {'—':>8}")

    # Check if any static frequency beats baseline on time
    if hw_key == "rtx4080s":
        search_range = range(1005, 2506, 15)
    else:
        search_range = range(765, 1531, 15)
    faster_static = []
    for freq in search_range:
        p_static, _, _, _ = predict_hw_model(hw_key, model_key, freq, mode="static")
        p_baseline, _, _, _ = predict_hw_model(hw_key, model_key, max_freq, mode="baseline")
        if p_static.step_time_s < p_baseline.step_time_s:
            faster_static.append(freq)
    if faster_static:
        print(f"\n  Static frequencies FASTER than baseline {max_freq} MHz: "
              f"{min(faster_static)}–{max(faster_static)} MHz ({len(faster_static)} points)")
        mid = faster_static[len(faster_static)//2]
        p_mid, _, _, _ = predict_hw_model(hw_key, model_key, mid, mode="static")
        p_base, _, _, _ = predict_hw_model(hw_key, model_key, max_freq, mode="baseline")
        e_mid = p_mid.power_w * p_mid.step_time_s * workload.train_iters
        e_base = p_base.power_w * p_base.step_time_s * workload.train_iters
        print(f"  Example: Static {mid} MHz → time {(p_mid.step_time_s/p_base.step_time_s-1)*100:+.1f}%, "
              f"energy {(e_mid/e_base-1)*100:+.1f}% vs baseline")

    return results, []


def main():
    configs = [
        ("rtx4080s", "qwen7b"),
        ("rtx4080s", "llama7b"),
        ("v100", "llama7b"),
        ("v100", "qwen7b"),
    ]

    print("=" * 100)
    print("  UNIFIED PREDICTOR v2: RTX 4080S vs V100")
    print("  Same physics model (analysis.freq_model), hardware-specific calibration")
    print("=" * 100)

    for hw_key, model_key in configs:
        run_comparison(hw_key, model_key)

    # Cross-hardware sweet-spot search
    print(f"\n{'='*100}")
    print("  CROSS-HARDWARE SWEET-SPOT SEARCH (static frequency locking)")
    print(f"{'='*100}")
    print(f"  {'Platform':<30} | {'Model':<10} | {'Best Freq':>9} | {'Energy Save':>12}")
    print("  " + "-" * 72)

    for hw_key, model_key in configs:
        if hw_key == "rtx4080s":
            search_range = range(1005, 2506, 15)
            baseline_freq = 2505
        else:
            search_range = range(765, 1531, 15)
            baseline_freq = 1530

        baseline_pred, _, _, workload = predict_hw_model(hw_key, model_key, baseline_freq, mode="baseline")
        baseline_energy = baseline_pred.power_w * baseline_pred.step_time_s * workload.train_iters

        best_energy = float("inf")
        best_freq = None
        for freq in search_range:
            pred, _, _, workload = predict_hw_model(hw_key, model_key, freq, mode="static")
            energy = pred.power_w * pred.step_time_s * workload.train_iters
            if energy < best_energy:
                best_energy = energy
                best_freq = freq

        save_pct = (baseline_energy - best_energy) / baseline_energy * 100
        hw_label = "RTX 4080S" if hw_key == "rtx4080s" else "V100 16×"
        print(f"  {hw_label:<30} | {model_key.upper():<10} | {best_freq:>9} MHz | {save_pct:>+11.1f}%")


if __name__ == "__main__":
    main()
