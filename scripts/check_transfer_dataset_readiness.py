#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TopologyExpectation:
    node_count: int
    gpus_per_node: int
    tp: int
    pp: int
    dp: int


def _parse_topology(text: str) -> TopologyExpectation:
    # format: node_count,gpus_per_node,tp,pp,dp
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 5:
        raise ValueError("topology must be: node_count,gpus_per_node,tp,pp,dp")
    values = [int(part) for part in parts]
    return TopologyExpectation(
        node_count=values[0],
        gpus_per_node=values[1],
        tp=values[2],
        pp=values[3],
        dp=values[4],
    )


def _iter_run_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("run.json"))


def _extract_signature(run_json: dict) -> tuple[int, int, int, int, int] | None:
    topology = ((run_json.get("topology") or {}).get("resolved")) or {}
    parallelism = ((run_json.get("config") or {}).get("parallelism")) or {}
    tp = int(parallelism.get("tensor_model_parallel_size") or topology.get("tp") or 0)
    pp = int(parallelism.get("pipeline_model_parallel_size") or topology.get("pp") or 0)
    world_size = int(topology.get("world_size") or run_json.get("environment", {}).get("WORLD_SIZE") or 0)
    
    # Try to infer node_count and gpus_per_node from various sources
    node_count = int(topology.get("nnodes") or run_json.get("environment", {}).get("NNODES") or 0)
    gpus_per_node = int(topology.get("nproc_per_node") or 0)
    
    # Fallback: infer from experiment_name if it contains topology hints
    if node_count <= 0 or gpus_per_node <= 0:
        exp_name = run_json.get("experiment_name", "")
        # Pattern: dual8 -> 2 nodes, 8 gpus_per_node OR try to infer from world_size
        if exp_name.startswith("dual"):
            # Try to parse number after 'dual' as total gpus per node or similar
            try:
                num_after_dual = int(exp_name[4:6]) if len(exp_name) > 5 else 0
                if num_after_dual > 0 and world_size > 0:
                    # Assume dual{N} means 2 nodes, and gpus_per_node = world_size // 2
                    node_count = 2
                    gpus_per_node = world_size // 2
            except (ValueError, IndexError):
                pass
    
    if tp <= 0 or pp <= 0 or world_size <= 0:
        return None
    if node_count <= 0 or gpus_per_node <= 0:
        # Final fallback: assume even distribution if we have world_size
        if world_size > 0:
            # Try common configurations: 1 node or 2 nodes
            if world_size % 2 == 0:
                node_count = 2
                gpus_per_node = world_size // 2
            else:
                node_count = 1
                gpus_per_node = world_size
    dp = max(world_size // max(tp * pp, 1), 1)
    return node_count, gpus_per_node, tp, pp, dp


def _count_matches(root: Path, expected: TopologyExpectation) -> int:
    expected_sig = (
        expected.node_count,
        expected.gpus_per_node,
        expected.tp,
        expected.pp,
        expected.dp,
    )
    matches = 0
    for run_json_path in _iter_run_json_files(root):
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        signature = _extract_signature(payload)
        if signature == expected_sig:
            matches += 1
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether source/target transfer datasets are present")
    parser.add_argument("--source-root", required=True, help="Directory containing source run folders")
    parser.add_argument("--target-root", required=True, help="Directory containing target run folders")
    parser.add_argument("--source-topology", required=True, help="node_count,gpus_per_node,tp,pp,dp")
    parser.add_argument("--target-topology", required=True, help="node_count,gpus_per_node,tp,pp,dp")
    parser.add_argument("--min-source-runs", type=int, default=3, help="Minimum required source runs")
    parser.add_argument("--min-target-runs", type=int, default=3, help="Minimum required target runs")
    args = parser.parse_args()

    source_expectation = _parse_topology(args.source_topology)
    target_expectation = _parse_topology(args.target_topology)

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)

    source_matches = _count_matches(source_root, source_expectation)
    target_matches = _count_matches(target_root, target_expectation)

    payload = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "source_expectation": source_expectation.__dict__,
        "target_expectation": target_expectation.__dict__,
        "source_matches": source_matches,
        "target_matches": target_matches,
        "ready": source_matches >= args.min_source_runs and target_matches >= args.min_target_runs,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
