# Project Brief

## Project Name
Megatron-DeepSpeed Lock-Clock Experiments

## Description
A Megatron-DeepSpeed fork used to benchmark large-language-model training behavior on multi-GPU systems, with emphasis on fixed-frequency GPU operation for sustained workloads.

## Vision
Establish a reproducible workflow for comparing baseline GPU behavior against static frequency locking on platforms such as V100 and RTX 4080, with a focus on energy efficiency under realistic LLM training conditions.

## Core Requirements
- Run short and long Megatron-DeepSpeed training experiments reproducibly.
- Track topology, preflight checks, run commands, logs, and power metrics.
- Compare baseline vs static frequency configurations under equivalent training settings.
- Preserve enough metadata to evaluate full-frequency `total_time` / power curves and compare baseline-relative tradeoffs across topologies.

## Goals
- Quantify training energy efficiency using Zeus-based power metrics.
- Compare `samples/Wh` and `tokens/J` across fixed-frequency sweep points.
- Keep experiment execution simple enough to run repeatedly on remote GPU hosts.

## Scope

### In Scope
- Single-node and multi-node Megatron-DeepSpeed launch flows.
- Fixed-frequency control via NVML on supported NVIDIA GPUs.
- Experiment manifests, logs, power tracking, and sweep comparisons.
- Short-run validation experiments for V100.

### Out of Scope
- Quantization experiments for the current V100 frequency-curve investigation.
- Model-quality benchmarking beyond short training stability checks.
- Production checkpoint retention for energy-only sweep runs.

## Key Stakeholders
- Primary user driving V100 and 4080 experiment planning and analysis.
- AI coding agents collaborating through shared run artifacts and memory bank notes.
