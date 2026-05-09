# Project Status: GPU Frequency Locking for Distributed Training Energy Optimization

## Overview

This project builds a **hardware-agnostic, physics-driven predictor** for GPU static-frequency energy optimization in distributed LLM training. The core deliverable is `analysis.freq_model` — a standalone prediction library that can recommend optimal lock frequencies for any (model, topology, hardware, network) combination with zero empirical data.

---

## 1. Independent Predictor Model (`analysis.freq_model`)

### Architecture

The predictor is organized into six modular layers:

| Module | File | Responsibility |
|--------|------|----------------|
| **Workload** | `workload.py` | Load experiment artifacts (`run.json`) into typed samples; extract workload signatures |
| **Features** | `features.py` | Derive model-communication features from architecture: FLOPs/token, comm_share, cross-node bytes, pipeline bubble fraction |
| **Hardware** | `hardware.py` | `HardwareFeatures` dataclass (GPU specs, supported clocks) + `HardwareFingerprint` (calibrated power/time anchors per platform) |
| **Network** | `network.py` | `NetworkConfig` dataclass (bandwidth, latency, protocol) + dynamic quality adjustments |
| **Cross-Node** | `cross_node.py` | Fit topology-aware penalty model for multi-node overhead (α_pp, α_dp, α_tp) |
| **Model Engine** | `model.py` | Core physics: derive `CalibrationParams` from hardware+workload+network; predict time/power/energy at any frequency |
| **Calibration** | `calibrate.py` | Grid-search fingerprint calibration from observed sweeps; NNLS-based correction layer |
| **Recommend** | `recommend.py` | Build prediction bundles, find sweet spots, generate comparison tables |

### Key Design Principles

1. **Hardware-first derivation**: `CalibrationParams` are derived from hardware specs + workload features + network config. No per-scenario hard-coding.
2. **Fingerprint reuse**: `HardwareFingerprint` is calibrated once per hardware platform (e.g., V100, RTX 4080S) and reused for any model/topology on that platform.
3. **Baseline vs Static dual mode**: Predictor supports both `mode='static'` (fixed clock, no thermal throttle) and `mode='baseline'` (dynamic boost with thermal penalty + multi-card desync).
4. **Cross-topology transfer**: Cross-node penalty terms (α_pp, α_dp, α_tp) enable zero-shot prediction on unseen topologies.

### Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/predict_unified_v3.py` | Zero-shot demo: input hardware+workload+network → full prediction table. No experiment data needed. |
| `scripts/predict_freq_sweet_spot.py` | Artifact-driven prediction: load local `experiments/` → calibrate → recommend sweet spots |
| `scripts/calibrate_fingerprint_demo.py` | Demo fingerprint calibration from hard-coded observations |

### Calibration Accuracy (Current)

| Platform | Time MAPE | Power MAPE | Notes |
|----------|-----------|------------|-------|
| RTX 4080S (Ethernet) | ~0.8% | ~1.4% | 9-point sweep calibration |
| V100 (NVLink) | ~5.4% | ~4.1% | 2-point calibration (LLaMA) |
| V100 (IB) | ~11% | ~18% | Cross-topology transfer; IB fingerprint needs more points |

---

## 2. Experimental Coverage

### V100 DGX-2 (Dual-Node, IB 100Gbps)

**Model**: Qwen2.5-7B-Instruct (28L, hidden=3584, ffn=18944, 28 heads, 4 KV heads)

| Topology | GPUs | GBS | Low-Freq (990–1260) | Refine | High-Freq (1350–1530) | Retest |
|----------|------|-----|---------------------|--------|----------------------|--------|
| TP4PP2DP4 | 32 | 16 | ✅ 5 pts | ✅ 1020/1050/1110/1245 | ✅ 4 pts | ✅ 2 rounds |
| TP2PP4DP4 | 32 | 16 | ✅ 5 pts | — | ✅ 4 pts | ✅ 2 rounds |
| TP2PP2DP8 | 32 | 16 | ✅ 5 pts | — | ✅ 4 pts | ✅ 2 rounds |
| TP4PP2DP4 | 32 | 32 | ✅ 5 pts | — | ✅ 4 pts | ✅ 2 rounds |
| TP4PP2DP2 | 16 | 16 | ✅ 5 pts | — | ✅ 4 pts | ✅ 2 rounds |
| TP4PP2DP1 | 8 | 4 | ✅ 4 pts | — | ✅ 4 pts | ✅ 2 rounds |

**Key Findings**:
- **TP4PP2DP4 (32-card, GBS=16)**: Optimum at **1020–1050 MHz** (~30.9% energy save)
- **TP2PP4DP4 (32-card, GBS=16)**: Optimum at **1080 MHz** (~26.8% energy save)
- **TP2PP2DP8 (32-card, GBS=16)**: Optimum at **1080 MHz** (~29.5% energy save)
- **TP4PP2DP4 (32-card, GBS=32)**: Optimum at **1080 MHz** (~34.1% energy save)
- **1080 MHz anomaly**: In TP4PP2DP4 GBS=16, 1080 MHz is a consistent outlier (~20% worse energy than neighbors). Treated as V100-specific behavioral anomaly.

### V100 Single-Node (NVLink)

| Model | Topology | GPUs | Best Energy Point |
|-------|----------|------|-------------------|
| LLaMA-7B | TP2PP2DP2 | 8 | 1260 MHz (−28.7%) |
| DeepSeek-R1-Qwen-7B | TP2PP2DP2 | 8 | 1080 MHz (−28.7%) |
| Qwen2.5-7B | TP1PP4DP4 | 16 | 1260 MHz (−25.8%) |

### RTX 4080S Ethernet

| Model | Topology | GPUs | Best Energy Point |
|-------|----------|------|-------------------|
| Qwen2.5-7B | TP2PP2DP2 | 8 | 1650 MHz (−27.1%) |
| LLaMA-7B | TP2PP2DP2 | 8 | 1800 MHz (−28.1%) |
| Qwen2.5-1.5B | TP2PP2DP1 | 4 | 1650 MHz (−23.3%) |

---

## 3. Critical Fixes & Operational Lessons

1. **Checkpoint topology lock-in**: Megatron-DeepSpeed checkpoints are converted with fixed TP/PP. Cannot load with different TP/PP. Each topology needs its own checkpoint.
2. **DGX2-2 disk constraint**: 96% full, 35G free. `DISABLE_SAVE_CHECKPOINT=1` required for all runs.
3. **Dual-node desync pattern**: DGX2-2 failure → DGX2-1 hangs in NCCL barrier. Post-failure cleanup must verify zero residual processes on both nodes.
4. **OOM mitigation**: Sweep scripts changed `sleep 10` → `sleep 60` between frequencies to allow GPU memory fragmentation to clear.
5. **1080 MHz anomaly**: Consistent ~20% worse energy than neighbors in TP4PP2DP4 32-card. Likely a V100-specific behavioral anomaly at this frequency; discard as outlier.

---

## 4. Repository Structure

```
analysis/freq_model/          # Independent predictor library
├── __init__.py               # Public API exports
├── model.py                  # Core physics engine (1,064 lines)
├── calibrate.py              # Fingerprint calibration (791 lines)
├── cross_node.py             # Multi-node penalty model (694 lines)
├── features.py               # Workload feature derivation (295 lines)
├── hardware.py               # Hardware specs + fingerprint (201 lines)
├── network.py                # Network config + quality (102 lines)
├── recommend.py              # Sweet-spot recommendation (325 lines)
└── workload.py               # Artifact loading (329 lines)

scripts/
├── predict_unified_v3.py     # Zero-shot predictor demo
├── predict_freq_sweet_spot.py # Artifact-driven predictor
├── run_experiment.sh         # Canonical multi-node launcher
├── run_real_qwen25_7b_*.sh   # Real-model experiment scripts
└── run_all_32card_highfreq_retests.sh  # Sweep orchestrator

memory-bank/                  # Cross-session context
├── activeContext.md          # Current focus & recent results
├── progress.md               # Completed tasks log
├── techContext.md            # Technical constraints & env facts
└── observability.md          # Experiment observations & metrics

docs/
├── FREQ_SCALING_GUIDE.md     # Original comm-scaling guide
└── PROJECT_STATUS.md         # This file
```

---

## 5. Next Steps

1. **Predictor recalibration**: Feed the newly completed 32-card full-curve data into `calibrate_fingerprint_demo.py` to tighten V100 IB fingerprint.
2. **1080 MHz anomaly investigation**: Understand why 1080 MHz is consistently worse than 1020/1050/1155 in TP4PP2DP4.
3. **Cross-topology validation**: Compare predicted vs observed for TP2PP4DP4 and TP2PP2DP8 (currently only TP4PP2DP4 has been used for calibration).
4. **Scale-out to 64+ GPUs**: Extend cross-node penalty model to 4×8 and 8×8 configurations.
