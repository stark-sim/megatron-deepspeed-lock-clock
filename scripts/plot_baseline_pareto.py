#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TradeoffPoint:
    freq_mhz: int
    runtime_ratio_vs_baseline: float
    energy_ratio_vs_baseline: float
    power_ratio_vs_baseline: float
    tokens_per_j_ratio_vs_baseline: float
    time_s: float
    energy_j: float
    avg_power_w: float
    tokens_per_j: float
    pareto_frontier: bool


PAPER_FRONTIER_COLOR = '#c43c39'
PAPER_DOMINATED_COLOR = '#b8bcc6'
ANALYSIS_CMAP = 'viridis'
BASELINE_COLOR = '#111827'
GUIDE_COLOR = '#9ca3af'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot 2D Pareto skyline charts from baseline-relative tradeoff TSV data')
    parser.add_argument('--input', required=True, help='Path to baseline tradeoff TSV file')
    parser.add_argument('--output-dir', required=True, help='Directory for generated chart files')
    parser.add_argument('--title-prefix', default='V100 Static Frequency vs Unfixed Baseline', help='Prefix used in chart titles')
    return parser.parse_args()


def _load_points(path: Path) -> List[TradeoffPoint]:
    with path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        return [
            TradeoffPoint(
                freq_mhz=int(row['freq_mhz']),
                runtime_ratio_vs_baseline=float(row['runtime_ratio_vs_baseline']),
                energy_ratio_vs_baseline=float(row['energy_ratio_vs_baseline']),
                power_ratio_vs_baseline=float(row['power_ratio_vs_baseline']),
                tokens_per_j_ratio_vs_baseline=float(row['tokens_per_j_ratio_vs_baseline']),
                time_s=float(row['time_s']),
                energy_j=float(row['energy_j']),
                avg_power_w=float(row['avg_power_w']),
                tokens_per_j=float(row['tokens_per_j']),
                pareto_frontier=(row['pareto_frontier'].strip().lower() == 'yes'),
            )
            for row in reader
        ]


def _sorted_frontier(points: Iterable[TradeoffPoint]) -> List[TradeoffPoint]:
    return sorted((point for point in points if point.pareto_frontier), key=lambda point: point.runtime_ratio_vs_baseline)


def _setup_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel('Total Runtime / Baseline', fontsize=11)
    ax.set_ylabel('Total Energy / Baseline', fontsize=11)
    ax.axvline(1.0, color=GUIDE_COLOR, linestyle='--', linewidth=1.0, alpha=0.9)
    ax.axhline(1.0, color=GUIDE_COLOR, linestyle='--', linewidth=1.0, alpha=0.9)
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)


def _draw_baseline(ax: plt.Axes) -> None:
    ax.scatter([1.0], [1.0], marker='*', s=260, color=BASELINE_COLOR, edgecolor='white', linewidth=1.0, zorder=5, label='Baseline')
    ax.annotate('Baseline', (1.0, 1.0), xytext=(8, 8), textcoords='offset points', fontsize=10, color=BASELINE_COLOR, weight='bold')


def _draw_skyline(ax: plt.Axes, frontier: List[TradeoffPoint], color: str, linewidth: float = 2.2) -> None:
    if not frontier:
        return
    xs = [point.runtime_ratio_vs_baseline for point in frontier]
    ys = [point.energy_ratio_vs_baseline for point in frontier]
    ax.step(xs, ys, where='post', color=color, linewidth=linewidth, alpha=0.95, label='Pareto skyline', zorder=2)
    ax.plot(xs, ys, color=color, linewidth=0.0, marker='o', markersize=0)


def _pad_limits(values: List[float], baseline: float = 1.0) -> tuple[float, float]:
    lower = min(values + [baseline])
    upper = max(values + [baseline])
    span = max(upper - lower, 0.05)
    pad = span * 0.12
    return lower - pad, upper + pad


def _format_delta(ratio: float) -> str:
    pct = (ratio - 1.0) * 100.0
    if abs(pct) < 0.05:
        return '±0.0%'
    return f'{pct:+.1f}%'


def _plot_paper_style(points: List[TradeoffPoint], output_path: Path, title_prefix: str) -> None:
    frontier = _sorted_frontier(points)
    dominated = [point for point in points if not point.pareto_frontier]

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    _setup_axes(ax, f'{title_prefix} — Paper View')
    _draw_baseline(ax)

    if dominated:
        ax.scatter(
            [point.runtime_ratio_vs_baseline for point in dominated],
            [point.energy_ratio_vs_baseline for point in dominated],
            s=70,
            color=PAPER_DOMINATED_COLOR,
            edgecolor='white',
            linewidth=0.9,
            alpha=0.95,
            label='Dominated points',
            zorder=3,
        )

    ax.scatter(
        [point.runtime_ratio_vs_baseline for point in frontier],
        [point.energy_ratio_vs_baseline for point in frontier],
        s=95,
        color=PAPER_FRONTIER_COLOR,
        edgecolor='white',
        linewidth=1.1,
        label='Pareto points',
        zorder=4,
    )
    _draw_skyline(ax, frontier, PAPER_FRONTIER_COLOR)

    for point in frontier:
        ax.annotate(
            f'{point.freq_mhz} MHz',
            (point.runtime_ratio_vs_baseline, point.energy_ratio_vs_baseline),
            xytext=(6, -12),
            textcoords='offset points',
            fontsize=9,
            color=PAPER_FRONTIER_COLOR,
            weight='bold',
        )

    ax.legend(loc='upper right', frameon=True)
    ax.set_xlim(*_pad_limits([point.runtime_ratio_vs_baseline for point in points]))
    ax.set_ylim(*_pad_limits([point.energy_ratio_vs_baseline for point in points]))
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_analysis_style(points: List[TradeoffPoint], output_path: Path, title_prefix: str) -> None:
    frontier = _sorted_frontier(points)

    fig, ax = plt.subplots(figsize=(10.2, 7.0))
    _setup_axes(ax, f'{title_prefix} — Analysis View')
    _draw_baseline(ax)

    scatter = ax.scatter(
        [point.runtime_ratio_vs_baseline for point in points],
        [point.energy_ratio_vs_baseline for point in points],
        c=[point.tokens_per_j_ratio_vs_baseline for point in points],
        s=[145 if point.pareto_frontier else 95 for point in points],
        cmap=ANALYSIS_CMAP,
        edgecolor='white',
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Tokens/J ratio vs baseline', fontsize=10)

    _draw_skyline(ax, frontier, PAPER_FRONTIER_COLOR, linewidth=2.4)

    for point in points:
        label = (
            f"{point.freq_mhz} MHz\n"
            f"runtime {_format_delta(point.runtime_ratio_vs_baseline)}\n"
            f"energy {_format_delta(point.energy_ratio_vs_baseline)}"
        )
        ax.annotate(
            label,
            (point.runtime_ratio_vs_baseline, point.energy_ratio_vs_baseline),
            xytext=(8, 8 if point.pareto_frontier else -16),
            textcoords='offset points',
            fontsize=8.5,
            color=BASELINE_COLOR if point.pareto_frontier else '#374151',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.72),
        )

    x_min, x_max = _pad_limits([point.runtime_ratio_vs_baseline for point in points])
    y_min, y_max = _pad_limits([point.energy_ratio_vs_baseline for point in points])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.text(x_min + 0.01, y_min + 0.01, 'Better\n(lower-left)', fontsize=10, color=BASELINE_COLOR, weight='bold', va='bottom')
    ax.text(1.005, y_max - 0.01, 'Baseline runtime', fontsize=9, color=GUIDE_COLOR, va='top')
    ax.text(x_max - 0.01, 1.005, 'Baseline energy', fontsize=9, color=GUIDE_COLOR, ha='right')

    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points = _load_points(input_path)
    if not points:
        raise SystemExit(f'No data points found in {input_path}')

    paper_path = output_dir / 'pareto_skyline_paper.png'
    analysis_path = output_dir / 'pareto_skyline_analysis.png'
    _plot_paper_style(points, paper_path, args.title_prefix)
    _plot_analysis_style(points, analysis_path, args.title_prefix)

    print(f'[ParetoPlot] Wrote {paper_path}')
    print(f'[ParetoPlot] Wrote {analysis_path}')


if __name__ == '__main__':
    main()
