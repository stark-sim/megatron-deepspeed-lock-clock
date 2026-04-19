#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent
REPO_ROOT = REPORT_DIR.parent
CHART_DIR = REPORT_DIR / "图表"
REPORT_PATH = REPORT_DIR / "09_实验数据图表总览.md"
MPL_CACHE = REPORT_DIR / ".mpl-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.pyplot as plt


IB_SOURCE_RUNS = [
    REPO_ROOT / ".context/ib_formal_rerun_20260410/source/ib_dual8_tp4pp1dp2_formal_990_20260410_20260410_161719_DGX2-1/run.json",
    REPO_ROOT / ".context/ib_formal_rerun_20260410/source/ib_dual8_tp4pp1dp2_formal_1080_20260410_20260410_162533_DGX2-1/run.json",
    REPO_ROOT / ".context/ib_formal_rerun_20260410/source/ib_dual8_tp4pp1dp2_formal_1155_retry_20260410_170335_DGX2-1/run.json",
]

IB_TARGET_RUNS = [
    REPO_ROOT / ".context/ib_formal_rerun_20260410/target_final/ib_dual16_tp4pp1dp4_diag_nozeus_990_20260410_202433_DGX2-1/run.json",
    REPO_ROOT / ".context/ib_formal_rerun_20260410/target_final/ib_dual16_tp4pp1dp4_formal_1080_20260411_110907_DGX2-1/run.json",
    REPO_ROOT / ".context/ib_formal_rerun_20260410/target_final/ib_dual16_tp4pp1dp4_formal_1155_20260411_111702_DGX2-1/run.json",
]

ETH_SOURCE_RUNS = [
    REPO_ROOT / ".context/eth_2x4_curve_eval_20260409/eth_qwen3b_1x4_source_curve_20260409_curated/eth_qwen3b_1x4_source_static1005_20260408_sd-2/run.json",
    REPO_ROOT / ".context/eth_2x4_curve_eval_20260409/eth_qwen3b_1x4_source_curve_20260409_curated/eth_qwen3b_1x4_source_static1200_20260408_sd-2/run.json",
    REPO_ROOT / ".context/eth_2x4_curve_eval_20260409/eth_qwen3b_1x4_source_curve_20260409_curated/eth_qwen3b_1x4_source_static1395_r1_20260409_sd-2/run.json",
]

ETH_TARGET_RUNS = [
    REPO_ROOT / ".context/eth_2x4_curve_eval_20260409/eth_qwen3b_2x4_target_curve_20260408_sd-1/eth_qwen3b_2x4_target_static1005_20260408_sd-1/run.json",
    REPO_ROOT / ".context/eth_2x4_curve_eval_20260409/eth_qwen3b_2x4_target_curve_20260408_sd-1/eth_qwen3b_2x4_target_static1200_20260408_sd-1/run.json",
    REPO_ROOT / ".context/eth_2x4_curve_eval_20260409/eth_qwen3b_2x4_target_curve_20260408_sd-1/eth_qwen3b_2x4_target_static1395_20260408_sd-1/run.json",
]


IB_REPLAY_METRICS = {
    "pair": "2x4 -> 2x8 (IB, topology-fixed, power-fixed)",
    "time_mape": 11.48,
    "power_mape": 3.28,
    "energy_mape": 7.86,
    "alpha_dp": "2.220525e-11 s/byte",
    "artifact": REPO_ROOT
    / ".context/transfer_eval_ib_2x4_to_2x8_rerun_topology_fixed_live_ib_powerfix_20260411/transfer_prediction_report.md",
}

ETH_REPLAY_METRICS = {
    "pair": "1x4 -> 2x4 (Ethernet)",
    "time_mape": 5.16,
    "power_mape": 12.38,
    "energy_mape": 10.42,
    "alpha_dp": "7.321889e-10 s/byte",
    "artifact": REPO_ROOT
    / ".context/transfer_eval_eth_qwen3b_1x4_source_curve_20260409_curated_to_eth_qwen3b_2x4_target_curve_20260408_sd-1/transfer_prediction_report.md",
}


@dataclass(frozen=True)
class RunRecord:
    path: Path
    run_id: str
    created_at: str
    freq_mhz: int
    time_s: float
    avg_power_w: float
    energy_j: float
    total_tokens: int
    train_iters: int
    micro_batch_size: int
    global_batch_size: int
    seq_length: int
    tp: int
    pp: int
    dp: int
    nnodes: int
    nproc_per_node: int
    world_size: int
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    tokenizer_model: str
    data_path: str
    data_impl: str
    split: str
    precision: str
    hostfile_entries: int
    preflight_ok: str


def load_run(path: Path) -> RunRecord:
    raw = json.loads(path.read_text())
    resolved = raw.get("topology", {}).get("resolved", {})
    config = raw["config"]
    model = config["model"]
    training = config["training"]
    data_cfg = config["data"]
    zeus = raw["power_metrics"]["zeus"]
    world_size = int(resolved["world_size"])
    tp = int(resolved["tp"])
    pp = int(resolved["pp"])
    dp = world_size // (tp * pp)
    hostfile = raw.get("hostfile", {})
    entries = hostfile.get("entries") or []
    preflight = raw.get("preflight", {})
    precision = "bf16" if training.get("bf16") else "fp16" if training.get("fp16") else "unknown"
    preflight_ok = preflight.get("ok")
    if preflight_ok is None:
        preflight_text = "null"
    else:
        preflight_text = "true" if preflight_ok else "false"
    return RunRecord(
        path=path,
        run_id=raw["identity"]["run_id"],
        created_at=raw["created_at"],
        freq_mhz=int(raw["freq_policy"]["static_clock_mhz"]),
        time_s=float(zeus["total_time_s"]),
        avg_power_w=float(zeus["avg_power_w"]),
        energy_j=float(zeus["total_energy_j"]),
        total_tokens=int(zeus["total_tokens"]),
        train_iters=int(training["train_iters"]),
        micro_batch_size=int(training["micro_batch_size"]),
        global_batch_size=int(training["global_batch_size"]),
        seq_length=int(model["seq_length"]),
        tp=tp,
        pp=pp,
        dp=dp,
        nnodes=int(resolved["nnodes"]),
        nproc_per_node=int(resolved["nproc_per_node"]),
        world_size=world_size,
        num_layers=int(model["num_layers"]),
        hidden_size=int(model["hidden_size"]),
        ffn_hidden_size=int(model["ffn_hidden_size"]),
        num_attention_heads=int(model["num_attention_heads"]),
        num_key_value_heads=int(model["num_key_value_heads"]),
        tokenizer_model=str(model["tokenizer_model"]),
        data_path=str(data_cfg["data_path"][0]),
        data_impl=str(data_cfg["data_impl"]),
        split=str(data_cfg["split"]),
        precision=precision,
        hostfile_entries=len(entries),
        preflight_ok=preflight_text,
    )


def load_series(paths: list[Path]) -> list[RunRecord]:
    return sorted((load_run(path) for path in paths), key=lambda record: record.freq_mhz)


def rel_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def fmt_seconds(value: float) -> str:
    return f"{value:.2f}"


def fmt_power(value: float) -> str:
    return f"{value:.2f}"


def fmt_energy_kj(value_j: float) -> str:
    return f"{value_j / 1000.0:.2f}"


def fmt_tokens(value: int) -> str:
    return f"{value:,}"


def workload_id(record: RunRecord, *, artifact_hint: str | None = None) -> str:
    tokenizer_name = Path(record.tokenizer_model).name
    hint = f"{artifact_hint}; " if artifact_hint else ""
    return (
        f"{hint}{record.num_layers}L / H{record.hidden_size} / FFN{record.ffn_hidden_size} / "
        f"A{record.num_attention_heads} / KV{record.num_key_value_heads}; tokenizer={tokenizer_name}"
    )


def markdown_table(rows: list[list[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    divider = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join([header, divider, body])


def build_workload_table(left: RunRecord, right: RunRecord, *, left_label: str, right_label: str, artifact_hint: str | None = None) -> str:
    rows = [
        ["字段", left_label, right_label],
        ["Workload 标识", workload_id(left, artifact_hint=artifact_hint), workload_id(right, artifact_hint=artifact_hint)],
        ["节点 x 每节点 GPU", f"{left.nnodes} x {left.nproc_per_node}", f"{right.nnodes} x {right.nproc_per_node}"],
        ["TP / PP / DP / World Size", f"{left.tp} / {left.pp} / {left.dp} / {left.world_size}", f"{right.tp} / {right.pp} / {right.dp} / {right.world_size}"],
        ["Train iters / Seq len", f"{left.train_iters} / {left.seq_length}", f"{right.train_iters} / {right.seq_length}"],
        ["Micro / Global batch", f"{left.micro_batch_size} / {left.global_batch_size}", f"{right.micro_batch_size} / {right.global_batch_size}"],
        ["Precision", left.precision, right.precision],
        ["Dataset", left.data_path, right.data_path],
        ["Data impl / split", f"{left.data_impl} / {left.split}", f"{right.data_impl} / {right.split}"],
        ["Tokenizer", left.tokenizer_model, right.tokenizer_model],
        ["Artifact completeness", f"hostfile={left.hostfile_entries}, preflight={left.preflight_ok}", f"hostfile={right.hostfile_entries}, preflight={right.preflight_ok}"],
        ["Representative artifact", rel_path(left.path), rel_path(right.path)],
    ]
    return markdown_table(rows)


def build_curve_table(source: list[RunRecord], target: list[RunRecord], *, left_label: str, right_label: str) -> str:
    target_by_freq = {record.freq_mhz: record for record in target}
    rows = [
        [
            "Freq (MHz)",
            f"{left_label} Time (s)",
            f"{left_label} Power (W)",
            f"{left_label} Energy (kJ)",
            f"{right_label} Time (s)",
            f"{right_label} Power (W)",
            f"{right_label} Energy (kJ)",
        ]
    ]
    for src in source:
        tgt = target_by_freq[src.freq_mhz]
        rows.append(
            [
                str(src.freq_mhz),
                fmt_seconds(src.time_s),
                fmt_power(src.avg_power_w),
                fmt_energy_kj(src.energy_j),
                fmt_seconds(tgt.time_s),
                fmt_power(tgt.avg_power_w),
                fmt_energy_kj(tgt.energy_j),
            ]
        )
    return markdown_table(rows)


def plot_curves(source: list[RunRecord], target: list[RunRecord], *, title_prefix: str, left_label: str, right_label: str, output: Path) -> None:
    freqs = [record.freq_mhz for record in source]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle(title_prefix, fontsize=14)
    source_values = [
        [record.time_s for record in source],
        [record.avg_power_w for record in source],
        [record.energy_j / 1000.0 for record in source],
    ]
    target_values = [
        [record.time_s for record in target],
        [record.avg_power_w for record in target],
        [record.energy_j / 1000.0 for record in target],
    ]
    titles = ["Runtime (s)", "Avg Power (W)", "Energy (kJ)"]
    colors = ("#34568B", "#A85C2C")
    for axis, title, src_values, tgt_values in zip(axes, titles, source_values, target_values):
        axis.plot(freqs, src_values, marker="o", linewidth=2.0, color=colors[0], label=left_label)
        axis.plot(freqs, tgt_values, marker="s", linewidth=2.0, color=colors[1], label=right_label)
        axis.set_title(title)
        axis.set_xlabel("Frequency (MHz)")
        axis.grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mape_chart(output: Path) -> None:
    labels = ["Time", "Power", "Energy"]
    ib_values = [IB_REPLAY_METRICS["time_mape"], IB_REPLAY_METRICS["power_mape"], IB_REPLAY_METRICS["energy_mape"]]
    eth_values = [ETH_REPLAY_METRICS["time_mape"], ETH_REPLAY_METRICS["power_mape"], ETH_REPLAY_METRICS["energy_mape"]]
    x_positions = [0, 1, 2]
    width = 0.35

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar([x - width / 2 for x in x_positions], ib_values, width=width, label="IB 2x4->2x8", color="#34568B")
    axis.bar([x + width / 2 for x in x_positions], eth_values, width=width, label="Eth 1x4->2x4", color="#A85C2C")
    axis.set_xticks(x_positions, labels)
    axis.set_ylabel("MAPE (%)")
    axis.set_title("Formal Replay MAPE Comparison")
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def replay_table() -> str:
    rows = [
        ["环境", "Pair", "Time MAPE", "Power MAPE", "Energy MAPE", "alpha_dp", "Artifact"],
        [
            "IB",
            IB_REPLAY_METRICS["pair"],
            f'{IB_REPLAY_METRICS["time_mape"]:.2f}%',
            f'{IB_REPLAY_METRICS["power_mape"]:.2f}%',
            f'{IB_REPLAY_METRICS["energy_mape"]:.2f}%',
            str(IB_REPLAY_METRICS["alpha_dp"]),
            rel_path(IB_REPLAY_METRICS["artifact"]),
        ],
        [
            "Ethernet",
            ETH_REPLAY_METRICS["pair"],
            f'{ETH_REPLAY_METRICS["time_mape"]:.2f}%',
            f'{ETH_REPLAY_METRICS["power_mape"]:.2f}%',
            f'{ETH_REPLAY_METRICS["energy_mape"]:.2f}%',
            str(ETH_REPLAY_METRICS["alpha_dp"]),
            rel_path(ETH_REPLAY_METRICS["artifact"]),
        ],
    ]
    return markdown_table(rows)


def pending_table() -> str:
    rows = [
        ["实验项", "当前状态", "为什么不能直接入主图", "建议补跑内容"],
        [
            "V100 `TP=1, PP=4, DP=4` baseline/static headline 案例",
            "仅有 memory-bank preserved baseline summary；本地 2026-03 static run 存在，但 `hostfile/topology/freq_policy` 仍是旧格式或为空",
            "缺 baseline 原始 artifact，且现有 static run 早于 2026-04 launcher / metadata / power-scaling 修复，不能当作当前 workload 级主证据",
            "在最新 launcher 上补 `baseline + 1252/1260/1267 MHz`，保留 `run.json + events.jsonl + command.sh + ds_config.json + hostfile_snapshot.json + preflight.json`",
        ],
        [
            "V100 `TP=2, PP=2, DP=4` baseline/static headline 案例",
            "只有历史 baseline summary；本地 2026-03 static run `freq_policy` 为 `null` 且 `hostfile={}`",
            "虽然 summary 数值好看，但 workload 元数据和基线 provenance 不完整，不能继续拿来做当前图表主线",
            "在最新 launcher 上补 `baseline + 1072/1080/1087/1125 MHz`，并确保 `freq_policy/topology/hostfile/preflight` 全部非空",
        ],
        [
            "IB formal workload 的 baseline 对照",
            "当前只有 `2x4` source 和 `2x8` target 的 static 三频 formal 曲线，没有 baseline",
            "这意味着当前只能做“frequency curve + predictor replay”图，不能做同 workload 下的 baseline vs fixed 对照图",
            "补跑 `2x4 TP=4/PP=1/DP=2` baseline 和 `2x8 TP=4/PP=1/DP=4` baseline，各 20 steps，保持数据集和模型不变",
        ],
        [
            "Ethernet formal workload 的 baseline 对照",
            "当前 `1x4 -> 2x4` 只有 static 三频曲线，没有 baseline；target 是手工 launcher 路径，`preflight` 为空",
            "能支撑 predictor replay，但还不够支撑“固定频率优于 baseline”的最新工作负载图表",
            "补跑 `1x4` baseline 和 `2x4` baseline，并优先让 launcher 写入完整 `hostfile/topology/preflight`",
        ],
        [
            "更多拓扑的 latest-code 覆盖",
            "目前最新、元数据完整的 artifact 主要集中在 IB `2x4 -> 2x8` 与 Ethernet `1x4 -> 2x4`",
            "拓扑覆盖仍偏窄，暂时不能把“跨拓扑稳定有效”写成 workload 级广泛结论",
            "优先补一个 IB 额外拓扑和一个 Ethernet 额外拓扑，建议从历史已出现过的 `TP=2, PP=4, DP=2` 和 `TP=2, PP=2, DP=4` 中选",
        ],
    ]
    return markdown_table(rows)


def build_report(ib_source: list[RunRecord], ib_target: list[RunRecord], eth_source: list[RunRecord], eth_target: list[RunRecord]) -> str:
    lines: list[str] = []
    lines.append("# 实验数据图表总览")
    lines.append("")
    lines.append("## 目的")
    lines.append("")
    lines.append("- 这份文档只整理当前本地 **可直接复核** 的实验图表与 workload 元数据，不继续沿用 PPT 的 headline 写法。")
    lines.append("- 进入主图的前提是：`run.json` 中能读到模型结构、数据集路径、train iters、TP/PP/DP、batch、Zeus 指标，以及至少基本可用的 topology 信息。")
    lines.append("- 对于缺 baseline、缺 workload 元数据、或明显早于 2026-04 当前代码路径的历史实验，这里不会继续凑图，而是单独列入待补实验。")
    lines.append("")
    lines.append("## 先说清楚的边界")
    lines.append("")
    lines.append("- 当前 **不能** 直接把 IB 和 Ethernet 画成一张“环境优劣”图，因为它们的 workload 本身不同。")
    lines.append("- IB formal workload 使用的是 `28L / H3584 / FFN18944 / A28 / KV4`，数据集路径是 `/home/sd/Megatron-DeepSpeed/data/chinese_wiki_megatron_text_document`。")
    lines.append("- Ethernet formal workload 的目录标签是 `eth_qwen3b_*`，`run.json` 记录的是 `36L / H2048 / FFN11008 / A16 / KV2`，数据集路径是 `/home/user/Megatron-DeepSpeed/data/qwen_data_text_document`。")
    lines.append("- 因此本页的主图只做 **同一 workload 内的 source/target / replay 对比**，不做跨 workload 的强行排名。")
    lines.append("")
    lines.append("## A. 当前可直接入图的 latest-code 实验")
    lines.append("")
    lines.append("### A1. IB formal rerun workload")
    lines.append("")
    lines.append("说明：`run.json` 中没有独立的模型名称字段，这里用结构参数和 tokenizer 路径标识 workload。")
    lines.append("")
    lines.append(build_workload_table(ib_source[0], ib_target[0], left_label="IB source `2x4`", right_label="IB target `2x8`", artifact_hint="Qwen2.5-style"))
    lines.append("")
    lines.append("#### IB source/target 曲线图")
    lines.append("")
    lines.append("![IB formal curves](图表/ib_formal_curves.png)")
    lines.append("")
    lines.append(build_curve_table(ib_source, ib_target, left_label="Source", right_label="Target"))
    lines.append("")
    lines.append("观察：")
    lines.append("")
    lines.append("- 在这个 latest-code IB workload 下，`2x4` source 和 `2x8` target 的 runtime 曲线非常接近，说明 DP 扩张后时间没有被明显拉长。")
    lines.append("- target 平均功率几乎接近 source 的 2 倍，这是 `gpus_per_node` 从 `4 -> 8` 的结构性变化，不应再用旧的 per-node power 假设直接外推。")
    lines.append("- 当前 target `990 MHz` 的 artifact 名称虽然带 `diag_nozeus`，但 `run.json.power_metrics.zeus` 实际存在且完整，因此仍可用于 workload 曲线图。")
    lines.append("")
    lines.append("### A2. Ethernet formal workload")
    lines.append("")
    lines.append(build_workload_table(eth_source[0], eth_target[0], left_label="Eth source `1x4`", right_label="Eth target `2x4`", artifact_hint="artifact dir label: eth_qwen3b"))
    lines.append("")
    lines.append("#### Ethernet source/target 曲线图")
    lines.append("")
    lines.append("![Ethernet formal curves](图表/eth_formal_curves.png)")
    lines.append("")
    lines.append(build_curve_table(eth_source, eth_target, left_label="Source", right_label="Target"))
    lines.append("")
    lines.append("观察：")
    lines.append("")
    lines.append("- Ethernet source 与 target 都有完整的 `run.json`、模型结构、数据路径、train-iters 和 Zeus 指标，因此可以支撑 workload 级曲线图。")
    lines.append("- 这组 artifact 的 target `hostfile` 和 `topology` 是存在的，但 `preflight.ok` 为 `null`，说明它来自手工/半手工 launcher 路径，证据等级略低于 IB formal rerun。")
    lines.append("- 尽管如此，这批 run 足以支撑“慢网络 workload 曲线 + replay 精度”的图表展示，但还不足以支撑 baseline vs fixed 的最新对照结论。")
    lines.append("")
    lines.append("### A3. Formal replay 精度对比")
    lines.append("")
    lines.append("![Formal replay MAPE](图表/formal_replay_mape.png)")
    lines.append("")
    lines.append(replay_table())
    lines.append("")
    lines.append("观察：")
    lines.append("")
    lines.append("- 当前最稳的 predictor 精度证据来自 2026-04 的 IB formal rerun 和 Ethernet formal replay。")
    lines.append("- IB 的 power / energy 已经收敛得比较好，剩余主要误差在 runtime。")
    lines.append("- Ethernet 的 time 已经进入可用范围，但 power 侧仍弱于 IB，因此更适合支撑“network-aware predictor 能迁移到慢网络”，而不是支撑“Ethernet 比 IB 更准”的结论。")
    lines.append("")
    lines.append("## B. 当前不应继续画成主图的历史实验")
    lines.append("")
    lines.append("- `TP=1, PP=4, DP=4` 和 `TP=2, PP=2, DP=4` 这两组 V100 双机案例仍然是重要历史线索，但它们的 baseline 目前主要存在于 `memory-bank/observability.md` 的 preserved summary。")
    lines.append("- 本地确实能找到一部分对应的 2026-03 static run，例如：")
    lines.append("  - `.context/dual8_generalization_20260326/tp1pp4dp4/*/run.json`")
    lines.append("  - `.context/dual8_generalization_20260326/tp2pp2dp4/*/run.json`")
    lines.append("- 但这些 run 普遍缺少当前 launcher 时代应有的 `hostfile / topology / freq_policy` 完整信息，且早于 2026-04 的 metadata、continuous alpha scaling 和 power-scaling 修复。")
    lines.append("- 因此，这里不再把它们直接纳入最新 workload 图表，只把它们视为“需要在最新代码上重做”的历史候选。")
    lines.append("")
    lines.append("## C. 需要补跑的实验")
    lines.append("")
    lines.append(pending_table())
    lines.append("")
    lines.append("## D. 推荐的补实验优先级")
    lines.append("")
    lines.append("1. **先补当前 workload 的 baseline**")
    lines.append("   先把 IB formal `2x4 / 2x8` 和 Ethernet formal `1x4 / 2x4` 的 baseline 各补一条。这样马上就能把“predictor 曲线”和“baseline vs fixed”接到同一 workload 上。")
    lines.append("2. **再补历史 headline 两个拓扑的 latest-code 重跑**")
    lines.append("   也就是 V100 的 `TP=1, PP=4, DP=4` 和 `TP=2, PP=2, DP=4`。这一步的目标不是再追更多频点，而是恢复完整 provenance。")
    lines.append("3. **最后补更多 topology 覆盖**")
    lines.append("   至少再增加一个 IB topology 和一个 Ethernet topology，才有资格把“跨拓扑有效”写成 workload 级结论。")
    lines.append("")
    lines.append("## 关联 artifacts")
    lines.append("")
    lines.append("- IB source formal runs:")
    for record in ib_source:
        lines.append(f"  - `{rel_path(record.path)}`")
    lines.append("- IB target formal runs:")
    for record in ib_target:
        lines.append(f"  - `{rel_path(record.path)}`")
    lines.append("- Ethernet source formal runs:")
    for record in eth_source:
        lines.append(f"  - `{rel_path(record.path)}`")
    lines.append("- Ethernet target formal runs:")
    for record in eth_target:
        lines.append(f"  - `{rel_path(record.path)}`")
    lines.append(f"- IB replay report: `{rel_path(IB_REPLAY_METRICS['artifact'])}`")
    lines.append(f"- Ethernet replay report: `{rel_path(ETH_REPLAY_METRICS['artifact'])}`")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- 当前真正适合做 workload 级图表的，是 2026-04 的 IB formal rerun 和 Ethernet formal replay 这两套最新 artifact。")
    lines.append("- 当前 headline 级 baseline/static 案例还没有被 latest-code、metadata-complete 的 artifact 重新托住，因此下一步不应该继续润色 PPT，而应该先补 baseline 和补关键拓扑。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)

    ib_source = load_series(IB_SOURCE_RUNS)
    ib_target = load_series(IB_TARGET_RUNS)
    eth_source = load_series(ETH_SOURCE_RUNS)
    eth_target = load_series(ETH_TARGET_RUNS)

    plot_curves(
        ib_source,
        ib_target,
        title_prefix="IB Formal Rerun Workload: Source vs Target",
        left_label="2x4 source",
        right_label="2x8 target",
        output=CHART_DIR / "ib_formal_curves.png",
    )
    plot_curves(
        eth_source,
        eth_target,
        title_prefix="Ethernet Formal Workload: Source vs Target",
        left_label="1x4 source",
        right_label="2x4 target",
        output=CHART_DIR / "eth_formal_curves.png",
    )
    plot_mape_chart(CHART_DIR / "formal_replay_mape.png")

    REPORT_PATH.write_text(build_report(ib_source, ib_target, eth_source, eth_target))
    print(f"Generated report: {REPORT_PATH}")
    print(f"Generated charts in: {CHART_DIR}")


if __name__ == "__main__":
    main()
