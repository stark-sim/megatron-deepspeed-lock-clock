#!/usr/bin/env python3
"""
Lightweight NCCL all-reduce benchmark.

This script is intended to be launched under torchrun or deepspeed so it can
measure the effective transport quality of the exact distributed path that will
be used for training. The JSON schema intentionally matches the existing
predictor-side network benchmark loader.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NCCL all-reduce benchmark")
    parser.add_argument("--sizes-mb", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--output", type=str, default="")
    # DeepSpeed injects this argument automatically.
    parser.add_argument("--local_rank", type=int, default=None)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class BenchmarkResult:
    size_mb: int
    median_ms: float
    p95_ms: float
    avg_ms: float
    std_ms: float
    cv: float
    algbw_gbps: float
    busbw_gbps: float


def _benchmark_size(
    *,
    size_mb: int,
    warmup_iters: int,
    iters: int,
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
) -> BenchmarkResult | None:
    decimal_bytes = int(size_mb * 1_000_000)
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = max(1, math.ceil(decimal_bytes / element_size))
    tensor = torch.ones(numel, dtype=dtype, device=device)

    dist.barrier()
    torch.cuda.synchronize(device)

    for _ in range(warmup_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)

    dist.barrier()
    samples_ms: list[float] = []
    timing_tensor = torch.zeros(1, dtype=torch.float64, device=device)

    for _ in range(iters):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Use the slowest rank as the effective iteration time.
        timing_tensor.fill_(elapsed_ms)
        dist.all_reduce(timing_tensor, op=dist.ReduceOp.MAX)
        if dist.get_rank() == 0:
            samples_ms.append(float(timing_tensor.item()))

    dist.barrier()

    if dist.get_rank() != 0:
        return None

    avg_ms = statistics.mean(samples_ms)
    std_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
    avg_s = avg_ms / 1000.0

    # Keep the historical field names/schema used by predictor-side loaders.
    # The numeric convention matches the existing archived benchmark payloads.
    algbw_gbps = (2.0 * decimal_bytes / avg_s) / 1e9 if avg_s > 0.0 else 0.0
    busbw_gbps = algbw_gbps * max(world_size - 1, 1) / max(world_size, 1)

    return BenchmarkResult(
        size_mb=size_mb,
        median_ms=_percentile(samples_ms, 50.0),
        p95_ms=_percentile(samples_ms, 95.0),
        avg_ms=avg_ms,
        std_ms=std_ms,
        cv=(std_ms / avg_ms) if avg_ms > 0.0 else 0.0,
        algbw_gbps=algbw_gbps,
        busbw_gbps=busbw_gbps,
    )


def main() -> int:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    try:
        dtype = _dtype_from_name(args.dtype)
        results: list[BenchmarkResult] = []
        for size_mb in args.sizes_mb:
            result = _benchmark_size(
                size_mb=size_mb,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
                dtype=dtype,
                device=device,
                world_size=world_size,
            )
            if rank == 0 and result is not None:
                results.append(result)
                print(
                    f"{size_mb:>4} MB  avg={result.avg_ms:8.3f} ms  "
                    f"busbw={result.busbw_gbps:8.3f}  cv={result.cv:6.3f}",
                    flush=True,
                )

        if rank == 0:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "hostname": socket.gethostname(),
                "world_size": world_size,
                "local_world_size": local_world_size,
                "master_addr": os.environ.get("MASTER_ADDR", ""),
                "master_port": os.environ.get("MASTER_PORT", ""),
                "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME", ""),
                "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE", ""),
                "busbw_gbps": max((result.busbw_gbps for result in results), default=0.0),
                "nccl_env": {
                    key: value
                    for key in [
                        "NCCL_SOCKET_IFNAME",
                        "NCCL_IB_DISABLE",
                        "NCCL_IB_HCA",
                        "NCCL_DEBUG",
                    ]
                    if (value := os.environ.get(key)) is not None
                },
                "results": [asdict(result) for result in results],
            }
            if args.output:
                output_path = Path(args.output).expanduser()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            else:
                print(json.dumps(payload, indent=2), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
