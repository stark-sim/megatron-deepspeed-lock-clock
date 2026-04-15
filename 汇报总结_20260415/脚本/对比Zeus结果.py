#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import sys
from typing import Any, Dict, List


def resolve_run_json(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str)
    if path.is_dir():
        candidate = path / "run.json"
        if candidate.exists():
            return candidate
    if path.is_file():
        return path
    raise FileNotFoundError(f"无法从以下路径定位 run.json: {path_str}")


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    zeus = ((payload.get("power_metrics") or {}).get("zeus") or {})
    freq_policy = payload.get("freq_policy") or {}
    topology = ((payload.get("topology") or {}).get("resolved") or {})
    run_id = str(payload.get("run_id") or "")

    mode = freq_policy.get("mode")
    static_clock = freq_policy.get("static_clock_mhz")

    if not mode:
        if "static" in run_id:
            mode = "static"
        elif "baseline" in run_id:
            mode = "baseline"
        else:
            mode = "unknown"

    if not static_clock:
        match = re.search(r"static(\d+)", run_id)
        if match:
            static_clock = match.group(1)

    tp = topology.get("tp", "-")
    pp = topology.get("pp", "-")
    nnodes = topology.get("nnodes", "-")
    gpus_per_node = topology.get("nproc_per_node", "-")
    world_size = topology.get("world_size", "-")

    if tp == "-" or pp == "-":
        topo_match = re.search(r"tp(\d+)pp(\d+)dp(\d+)", run_id)
        if topo_match:
            tp = topo_match.group(1)
            pp = topo_match.group(2)
            if world_size == "-":
                dp = int(topo_match.group(3))
                try:
                    world_size = int(tp) * int(pp) * dp
                except ValueError:
                    world_size = "-"

    return {
        "run_id": run_id,
        "mode": mode,
        "static_clock_mhz": static_clock or "-",
        "tp": tp,
        "pp": pp,
        "nnodes": nnodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
        "time_s": float(zeus.get("total_time_s") or zeus.get("time_s") or 0.0),
        "avg_power_w": float(zeus.get("avg_power_w") or 0.0),
        "energy_j": float(zeus.get("total_energy_j") or zeus.get("energy_j") or 0.0),
        "tokens_per_j": float(zeus.get("total_tokens_per_j") or zeus.get("interval_tokens_per_j") or 0.0),
    }


def pct_delta(value: float, baseline: float) -> str:
    if baseline == 0:
        return "n/a"
    return f"{((value / baseline) - 1.0) * 100:.1f}%"


def format_float(value: float) -> str:
    return f"{value:.1f}"


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    baseline = rows[0]
    lines = []
    lines.append("| 运行ID | 模式 | 频点MHz | TP | PP | 节点数 | 每节点GPU数 | 总时长s | 相对基线时长 | 平均功率W | 相对基线功率 | 总能耗J | 相对基线能耗 | tokens/J |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| {run_id} | {mode} | {static_clock_mhz} | {tp} | {pp} | {nnodes} | {gpus_per_node} | {time_s} | {delta_time} | {avg_power_w} | {delta_power} | {energy_j} | {delta_energy} | {tokens_per_j} |".format(
                run_id=row["run_id"],
                mode=row["mode"],
                static_clock_mhz=row["static_clock_mhz"],
                tp=row["tp"],
                pp=row["pp"],
                nnodes=row["nnodes"],
                gpus_per_node=row["gpus_per_node"],
                time_s=format_float(row["time_s"]),
                delta_time=pct_delta(row["time_s"], baseline["time_s"]) if row is not baseline else "0.0%",
                avg_power_w=format_float(row["avg_power_w"]),
                delta_power=pct_delta(row["avg_power_w"], baseline["avg_power_w"]) if row is not baseline else "0.0%",
                energy_j=format_float(row["energy_j"]),
                delta_energy=pct_delta(row["energy_j"], baseline["energy_j"]) if row is not baseline else "0.0%",
                tokens_per_j=f"{row['tokens_per_j']:.3f}",
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="对多个运行目录中的 Zeus 指标做 Markdown 对比。第一个参数默认为基线运行。")
    parser.add_argument("runs", nargs="+", help="运行目录或 run.json 文件路径，第一个参数会被视为 baseline。")
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        run_json = resolve_run_json(run)
        rows.append(extract_metrics(load_json(run_json)))

    sys.stdout.write(render_markdown(rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
