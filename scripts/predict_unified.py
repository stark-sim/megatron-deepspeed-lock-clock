#!/usr/bin/env python3
"""Unified predictor: one script, multiple hardware platforms.

Each hardware uses its best-suited prediction method:
- RTX 4080S: analytical fit model (T(f)=a·(f_max/f)^b+c, P(f)=P_s+P_d·(f/f_max)^exp)
  because 9-freq sweep data yields very accurate 3-param fits (MAPE <2%).
- V100: full physics model via analysis.freq_model (compute/memory/communication bottleneck
  + harmonic blend) because grid-searched throughput params match observed 2-point data.

Usage:
    python scripts/predict_unified.py
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from analysis.freq_model.hardware import HardwareFeatures, resolve_hardware_spec
from analysis.freq_model.features import derive_model_features
from analysis.freq_model.model import CalibrationParams, predict_point
from analysis.freq_model.workload import WorkloadFeatures

# =============================================================================
# 4080S Analytical Fit Models (from 9-freq Qwen sweep + 4-freq LLaMA sweep)
# =============================================================================

class AnalyticalModel:
    """Simplified analytical model: T(f) = a*(f_max/f)^b + c, P(f) = P_s + P_d*(f/f_max)^exp."""
    def __init__(self, f_max: float, a: float, b: float, c: float,
                 p_static: float, p_dynamic: float, exp: float, train_iters: int = 20):
        self.f_max = f_max
        self.a, self.b, self.c = a, b, c
        self.p_static, self.p_dynamic, self.exp = p_static, p_dynamic, exp
        self.train_iters = train_iters

    def predict(self, freq: float, tokens_per_step: float = 8192.0):
        ratio = self.f_max / freq
        step_time = self.a * (ratio ** self.b) + self.c
        total_time = step_time * self.train_iters
        power = self.p_static + self.p_dynamic * (freq / self.f_max) ** self.exp
        energy = power * total_time
        tok_j = tokens_per_step / step_time / power
        return {
            "step_time_s": step_time,
            "total_time_s": total_time,
            "power_w": power,
            "energy_j": energy,
            "tokens_per_j": tok_j,
        }

# 4080S Qwen: from .context/raw_predict_4080_v3.py (9-freq fit, MAPE 1.08% time, 2.40% power)
RTX4080S_QWEN_MODEL = AnalyticalModel(
    f_max=2505.0, a=23.637, b=1.0458, c=207.924,
    p_static=218.67, p_dynamic=93.78, exp=8.0,
)

# 4080S LLaMA: from .context/raw_predict_4080_llama32l_v2.py (4-freq fit, MAPE 2.16% time)
RTX4080S_LLAMA_MODEL = AnalyticalModel(
    f_max=2505.0, a=28.101, b=0.754, c=251.370,
    p_static=211.23, p_dynamic=97.89, exp=5.731,
)

# =============================================================================
# V100 Full Physics Model (via analysis.freq_model)
# =============================================================================

V100_SPEC = resolve_hardware_spec("tesla v100-sxm3-32gb")
V100_HARDWARE = HardwareFeatures(
    gpu_name="Tesla V100-SXM3-32GB", gpu_count=16,
    supported_frequency_mhz=list(range(765, 1530 + 1, 15)),
    min_frequency_mhz=765, max_frequency_mhz=1530,
    power_limit_w_per_gpu=V100_SPEC.board_power_w_per_gpu,
    peak_fp16_tensor_tflops_per_gpu=V100_SPEC.peak_fp16_tensor_tflops_per_gpu,
    peak_fp32_tflops_per_gpu=V100_SPEC.peak_fp32_tflops_per_gpu,
    memory_bandwidth_gbps_per_gpu=V100_SPEC.memory_bandwidth_gbps_per_gpu,
)

V100_WORKLOADS = {
    "qwen7b": WorkloadFeatures(
        num_layers=28, hidden_size=3584, ffn_hidden_size=18944,
        num_attention_heads=28, num_key_value_heads=4,
        seq_length=2048, micro_batch_size=1, global_batch_size=16, train_iters=20,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=2, data_parallel_size=8,
        zero_stage=1, precision_mode="bf16", swiglu=True, node_count=1, gpus_per_node=16,
    ),
    "llama7b": WorkloadFeatures(
        num_layers=32, hidden_size=4096, ffn_hidden_size=11008,
        num_attention_heads=32, num_key_value_heads=32,
        seq_length=2048, micro_batch_size=1, global_batch_size=16, train_iters=20,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=2, data_parallel_size=8,
        zero_stage=1, precision_mode="bf16", swiglu=True, node_count=1, gpus_per_node=16,
    ),
}

def build_v100_params(features):
    """Build V100 CalibrationParams using infer_initial_anchors + calibrated coefficients."""
    from analysis.freq_model.model import infer_initial_anchors
    anchors = infer_initial_anchors(V100_HARDWARE, features, observed_max_throughput_tokens_s=1000.0)
    CA = anchors["compute_anchor"] or 50000.0
    MA = anchors["memory_anchor"] or 40000.0
    COA = anchors["communication_anchor"] or 30000.0
    # Grid-searched throughput params + 2-point power fit
    return CalibrationParams(
        compute_limit_at_max_tokens_per_s=CA * 0.025,
        memory_limit_tokens_per_s=MA * 0.10,
        communication_limit_tokens_per_s=COA * 0.15,
        communication_penalty=0.01,
        static_power_w=592.0,
        dynamic_power_w=4069.0,
        dynamic_power_exponent=5.0,
        throughput_saturation_ratio=1.0,
        thermal_throttle_threshold=0.80,
        thermal_throttle_coefficient=0.30,
        reference_total_gpu_count=16,
        reference_gpus_per_node=16,
        reference_pipeline_parallel_efficiency=features.pipeline_parallel_efficiency,
    )

# =============================================================================
# Observed Data (for validation)
# =============================================================================

OBSERVED = {
    ("rtx4080s", "qwen7b"): {
        1005: {"time_s": 267.6, "power_w": 213.7, "energy_j": 57177.5},
        1200: {"time_s": 262.0, "power_w": 218.2, "energy_j": 57168.6},
        1395: {"time_s": 254.4, "power_w": 221.6, "energy_j": 56385.5},
        1500: {"time_s": 248.9, "power_w": 223.2, "energy_j": 55554.4},
        1650: {"time_s": 238.3, "power_w": 228.2, "energy_j": 54380.4},
        1800: {"time_s": 239.2, "power_w": 231.5, "energy_j": 55380.9},
        1950: {"time_s": 237.6, "power_w": 235.0, "energy_j": 55839.8},
        2100: {"time_s": 239.1, "power_w": 237.1, "energy_j": 56694.1},
        2250: {"time_s": 238.3, "power_w": 239.8, "energy_j": 57151.3},
        2505: {"time_s": 229.5, "power_w": 320.2, "energy_j": 73478.5},
    },
    ("rtx4080s", "llama7b"): {
        1200: {"time_s": 299.2, "power_w": 212.7, "energy_j": 63638.0},
        1650: {"time_s": 300.8, "power_w": 219.9, "energy_j": 66154.0},
        1800: {"time_s": 276.1, "power_w": 226.2, "energy_j": 62459.0},
        2505: {"time_s": 281.0, "power_w": 309.1, "energy_j": 86842.0},
    },
    ("v100", "llama7b"): {
        1260: {"time_s": 620.5, "power_w": 2118.0, "energy_j": 1314427.0},
        1380: {"time_s": 659.0, "power_w": 2769.0, "energy_j": 1824888.0},
        1350: {"time_s": 488.2, "power_w": 1306.8, "energy_j": 638014.0},
        1455: {"time_s": 467.3, "power_w": 1509.1, "energy_j": 705273.0},
        1530: {"time_s": 452.4, "power_w": 1681.2, "energy_j": 760476.0},
    },
    ("v100", "qwen7b"): {
        1260: {"time_s": 717.6, "power_w": 2013.0, "energy_j": 1444301.0},
        1380: {"time_s": 724.6, "power_w": 2685.0, "energy_j": 1945838.0},
    },
}

# =============================================================================
# Prediction Engine
# =============================================================================

def predict_rtx4080s(model_key: str, freq: int) -> dict:
    """Use analytical fit model for RTX 4080S."""
    model = RTX4080S_QWEN_MODEL if model_key == "qwen7b" else RTX4080S_LLAMA_MODEL
    return model.predict(freq)

def predict_v100(model_key: str, freq: int) -> dict:
    """Use full physics model for V100."""
    workload = V100_WORKLOADS[model_key]
    features = derive_model_features(V100_HARDWARE, workload)
    params = build_v100_params(features)
    pred = predict_point(freq, V100_HARDWARE, features, params)
    return {
        "step_time_s": pred.step_time_s,
        "total_time_s": pred.step_time_s * workload.train_iters,
        "power_w": pred.power_w,
        "energy_j": pred.power_w * pred.step_time_s * workload.train_iters,
        "tokens_per_j": pred.tokens_per_j,
    }

def run_prediction(hw_key: str, model_key: str):
    observed = OBSERVED.get((hw_key, model_key), {})
    predictor = predict_rtx4080s if hw_key == "rtx4080s" else predict_v100
    
    results = []
    for freq in sorted(observed.keys()):
        pred = predictor(model_key, freq)
        obs = observed[freq]
        results.append({
            "freq": freq,
            "pred_time": pred["total_time_s"],
            "pred_power": pred["power_w"],
            "pred_energy": pred["energy_j"],
            "pred_tok_j": pred["tokens_per_j"],
            "obs_time": obs["time_s"],
            "obs_power": obs["power_w"],
            "obs_energy": obs["energy_j"],
        })
    
    # Also predict a sparse grid for curve shape
    if hw_key == "rtx4080s":
        grid_freqs = list(range(1005, 2506, 150))
    else:
        grid_freqs = list(range(765, 1531, 75))
    
    grid_results = []
    for freq in grid_freqs:
        if freq not in observed:
            pred = predictor(model_key, freq)
            grid_results.append({
                "freq": freq,
                "pred_time": pred["total_time_s"],
                "pred_power": pred["power_w"],
                "pred_energy": pred["energy_j"],
                "pred_tok_j": pred["tokens_per_j"],
                "obs_time": None,
                "obs_power": None,
                "obs_energy": None,
            })
    
    return results, grid_results


def print_comparison(hw_key: str, model_key: str, results: List[dict], grid: List[dict]):
    hw_label = "RTX 4080S (8×, dual-node, Ethernet)" if hw_key == "rtx4080s" else "V100 (16×, single-node, NVLink)"
    model_label = model_key.upper()
    method = "Analytical Fit" if hw_key == "rtx4080s" else "Full Physics Model"
    
    print(f"\n{'='*100}")
    print(f"  {hw_label} | {model_label} | Method: {method}")
    print(f"{'='*100}")
    
    print(f"  {'Freq':>6} | {'PredTime':>9} | {'ObsTime':>9} | {'TimeErr':>8} | "
          f"{'PredPow':>8} | {'ObsPow':>8} | {'PowErr':>7} | {'Tok/J':>7}")
    print("  " + "-" * 90)
    
    time_errors = []
    power_errors = []
    
    # Print observed points first
    for r in results:
        freq = r["freq"]
        pt, pp, pj = r["pred_time"], r["pred_power"], r["pred_tok_j"]
        ot, op = r["obs_time"], r["obs_power"]
        te = (pt - ot) / ot * 100 if ot else 0
        pe = (pp - op) / op * 100 if op else 0
        time_errors.append(abs(te))
        power_errors.append(abs(pe))
        baseline_marker = " ***" if freq == max(r2["freq"] for r2 in results) else ""
        print(f"  {freq:>6} | {pt:>8.1f}s | {ot:>8.1f}s | {te:>+7.1f}% | "
              f"{pp:>7.1f}W | {op:>7.1f}W | {pe:>+6.1f}% | {pj:>6.2f}{baseline_marker}")
    
    # Print grid points (prediction only)
    for r in grid:
        freq = r["freq"]
        pt, pp, pj = r["pred_time"], r["pred_power"], r["pred_tok_j"]
        print(f"  {freq:>6} | {pt:>8.1f}s | {'—':>9} | {'—':>8} | "
              f"{pp:>7.1f}W | {'—':>8} | {'—':>7} | {pj:>6.2f}")
    
    if time_errors:
        print(f"\n  Time MAPE: {sum(time_errors)/len(time_errors):.2f}%")
        print(f"  Power MAPE: {sum(power_errors)/len(power_errors):.2f}%")


def main():
    configs = [
        ("rtx4080s", "qwen7b"),
        ("rtx4080s", "llama7b"),
        ("v100", "qwen7b"),
        ("v100", "llama7b"),
    ]
    
    print("=" * 100)
    print("  UNIFIED PREDICTOR: RTX 4080S vs V100")
    print("  Same script, best-suited model per hardware")
    print("=" * 100)
    
    all_results = {}
    for hw_key, model_key in configs:
        results, grid = run_prediction(hw_key, model_key)
        print_comparison(hw_key, model_key, results, grid)
        all_results[(hw_key, model_key)] = (results, grid)
    
    # Cross-hardware summary
    print(f"\n{'='*100}")
    print("  CROSS-HARDWARE SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Platform':<28} | {'Model':<10} | {'Method':<20} | {'Best Freq':>9} | {'Energy Save':>12}")
    print("  " + "-" * 92)
    
    for hw_key, model_key in configs:
        results, grid = all_results[(hw_key, model_key)]
        observed = OBSERVED.get((hw_key, model_key), {})
        if not observed:
            continue
        
        predictor = predict_rtx4080s if hw_key == "rtx4080s" else predict_v100
        baseline_freq = max(observed.keys())
        baseline_pred = predictor(model_key, baseline_freq)
        baseline_energy = baseline_pred["energy_j"]
        
        best_energy = float('inf')
        best_freq = None
        search_range = range(1005, 2506, 15) if hw_key == "rtx4080s" else range(765, 1531, 15)
        for freq in search_range:
            pred = predictor(model_key, freq)
            if pred["energy_j"] < best_energy:
                best_energy = pred["energy_j"]
                best_freq = freq
        
        save_pct = (baseline_energy - best_energy) / baseline_energy * 100
        hw_name = "RTX 4080S" if hw_key == "rtx4080s" else "V100 16×"
        method = "Analytical Fit" if hw_key == "rtx4080s" else "Physics Model"
        print(f"  {hw_name:<28} | {model_key.upper():<10} | {method:<20} | {best_freq:>9} MHz | {save_pct:>+11.1f}%")


if __name__ == "__main__":
    main()
