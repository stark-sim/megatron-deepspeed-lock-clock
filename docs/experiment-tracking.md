# Experiment Tracking

## Overview

This repo now uses a `run_id`-centric experiment flow for Megatron-DeepSpeed training benchmarks.

The design goal is to make baseline, static clock, dynamic clock, and dry-run experiments comparable across different hardware topologies while keeping one consistent record format.

## What Changed

### 1. Generic launcher

Use `scripts/run_experiment.sh` as the single entrypoint.

It supports:

- `LAUNCHER=deepspeed|torchrun`
- `EXPERIMENT_MODE=baseline|dynamic|dryrun|static`
- `TP`, `PP`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`
- `CUDA_VISIBLE_DEVICES`
- `HOSTFILE` for DeepSpeed multi-node launches
- `STATIC_CLOCK_MHZ`
- `COMM_LOW_FREQ`, `COMM_HIGH_FREQ`, `COMM_MIN_ELEMENTS`
- `VALIDATE_ONLY=1` for preflight-only checks

The old `scripts/pretrain_qwen7b_tp4_*.sh` scripts are now thin wrappers around the generic launcher.

### 2. Topology-first validation

The launcher validates topology before training starts:

- `TP` must fit inside a single node's visible GPUs.
- `WORLD_SIZE % (TP * PP) == 0` must hold.
- `PP > 1` or `NNODES > 1` requires `HOSTFILE`.
- Static clock mode validates the requested frequency on all selected nodes before launch.

This prevents invalid runs such as asking for `TP=32` when no single node has 32 visible GPUs.

### 3. Hostfile snapshot and remote preflight

When multi-node is used, the launcher:

- reads the DeepSpeed hostfile
- snapshots the exact hostfile contents
- selects the first `NNODES` hosts from it
- runs SSH preflight checks on the selected remote nodes

The exact hostfile contents are copied into the run artifacts so later edits to the hostfile do not affect historical experiment interpretation.

### 4. Unified run artifacts

Each run writes to `experiments/<run_id>/`:

- `run.json`: full run metadata snapshot
- `events.jsonl`: interval and checkpoint events
- `notes.md`: manual experiment notes template
- `command.sh`: exact launch command snapshot
- `logs/<run_id>.log`: launcher stdout/stderr
- `topology.json`: resolved topology from environment
- `hostfile_snapshot.json`: hostfile path, raw text, hash, parsed entries, selected hosts
- `preflight.json`: local and remote node preflight results

`experiments/index.jsonl` is appended for cross-run indexing.

### 5. `CUDA_VISIBLE_DEVICES` as the GPU scope

Clock locking, unlocking, and NVML metadata collection now follow `CUDA_VISIBLE_DEVICES` by default instead of blindly operating on all physical GPUs.

`scripts/run_experiment.sh` now also performs an `EXIT`-trap clock reset for `static`, `dynamic`, and `dryrun` modes, so visible GPUs are unlocked even if the in-process frequency manager misses its normal shutdown path.

This matters when:

- running on partial GPUs on a shared node
- comparing 8-GPU vs 16-GPU layouts
- isolating experiments without touching unrelated GPUs on the same host

## Main Files

- `scripts/run_experiment.sh`
- `scripts/experiment_utils.sh`
- `megatron/experiment_tracker.py`
- `megatron/gpu_freq_manager.py`
- `megatron/training.py`
- `megatron/arguments.py`

## Current Design Constraints

- Default launcher preference is `deepspeed`.
- `torchrun` remains supported as a same-level fallback.
- `TP` is node-local only.
- `PP` may span nodes.
- Host selection comes from the DeepSpeed hostfile.
- The hostfile snapshot must be embedded into run artifacts.
- Experiment comparison is organized around `run_id`, not around mutable external files.

## Example Usage

### Single-node validation only

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TP=4
export PP=1
export EXPERIMENT_MODE=baseline
export EXPERIMENT_NAME=qwen7b_tp4_check
export VALIDATE_ONLY=1
bash scripts/run_experiment.sh
```

### Single-node dynamic frequency run

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TP=4
export PP=1
export EXPERIMENT_MODE=dynamic
export EXPERIMENT_NAME=qwen7b_tp4_v100_dynamic
export COMM_LOW_FREQ=1200
export COMM_MIN_ELEMENTS=300000000
bash scripts/run_experiment.sh
```

### Multi-node preflight

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TP=4
export PP=2
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export HOSTFILE=/path/to/hostfile
export VALIDATE_ONLY=1
bash scripts/run_experiment.sh
```

## What Still Needs To Be Done

### Recommended before real machine runs

- run real `VALIDATE_ONLY=1` checks on target machines
- verify DeepSpeed hostfile parsing against your actual hostfile format
- verify SSH connectivity from master node to all selected worker nodes
- verify `deepspeed` launcher arguments on your cluster environment
- verify static clock support on each GPU model actually used
- choose `PRECISION_MODE=fp16` on V100-class hardware if needed

### Recommended next implementation work

- add a summary script for comparing multiple `run_id`s
- add a small parser for extracting throughput/loss summary from logs into structured JSON
- optionally support model presets through env vars instead of hardcoded defaults
- optionally add a dedicated `preflight-only` wrapper script

## Known Caveats

- The code has only been statically validated so far (`bash -n`, `py_compile`).
- Multi-node SSH preflight has not yet been exercised on real hardware in this workspace.
- `megatron/arguments.py` still emits one pre-existing `SyntaxWarning` unrelated to this change.

## Recommended Commit Boundary

This is a good commit boundary.

The current state cleanly covers:

- generic launcher design
- topology validation
- hostfile snapshotting
- remote preflight
- `run_id`-centric experiment artifacts
- wrapper script consolidation

