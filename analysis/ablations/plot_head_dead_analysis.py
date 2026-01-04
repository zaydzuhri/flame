#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from edd_utils import register_edd_style


register_edd_style()


@dataclass(frozen=True)
class PlotSizes:
    single_col: tuple[float, float]
    double_col: tuple[float, float]
    heatmap: tuple[float, float]


@dataclass(frozen=True)
class AnalysisKeys:
    mode: str
    status_label: str
    rate_key: str
    heads_key: str
    count_key: str
    mostly_count_key: str
    threshold_key: str
    mostly_threshold_key: str
    default_threshold: float
    default_mostly_threshold: float
    metric_label: str
    heatmap_title: str
    heatmap_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot head-dead or head-dormant analysis outputs for ACL-style figures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to head-dead or head-dormant analysis JSON output.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "dead", "dormant"],
        default="auto",
        help="Interpretation mode (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/attention_sink/outputs/figures"),
        help="Directory to write figures.",
    )
    parser.add_argument(
        "--single-col-width",
        type=float,
        default=3.3,
        help="Single-column width in inches (ACL 2-column default).",
    )
    parser.add_argument(
        "--double-col-width",
        type=float,
        default=6.8,
        help="Double-column width in inches (ACL 2-column default).",
    )
    parser.add_argument(
        "--single-col-height",
        type=float,
        default=2.2,
        help="Single-column height in inches.",
    )
    parser.add_argument(
        "--double-col-height",
        type=float,
        default=3.6,
        help="Double-column height in inches.",
    )
    parser.add_argument(
        "--heatmap-height",
        type=float,
        default=4.0,
        help="Heatmap height in inches.",
    )
    parser.add_argument(
        "--head-bin-size",
        type=int,
        default=4,
        help="Bin size for persistent-dead head indices (default: 4).",
    )
    parser.add_argument(
        "--layer-bin-size",
        type=int,
        default=2,
        help="Bin size for persistent-dead layers (default: 2).",
    )
    parser.add_argument(
        "--layer-tick-step",
        type=int,
        default=2,
        help="Tick step for layer axis in persistent-dead plot (default: 2).",
    )
    parser.add_argument(
        "--step-tick-step",
        type=int,
        default=2,
        help="Tick step for training steps in dead-count plot (default: 2).",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Include all plots (default: only dead-head count and persistent-dead map).",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats (default: pdf,png).",
    )
    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="Enable LaTeX text rendering if your environment supports it.",
    )
    return parser.parse_args()


def load_analysis(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_mode(mode: str, data: dict[str, Any]) -> str:
    if mode != "auto":
        return mode
    if "dormant_head_count" in data or "dormant_heads_by_step" in data:
        return "dormant"
    if "dead_head_count" in data or "dead_heads_by_step" in data:
        return "dead"
    if "sink_attention" in data:
        return "dormant"
    if "off_rates" in data:
        return "dead"
    raise ValueError("Unable to infer mode; provide --mode explicitly.")


def analysis_keys(mode: str) -> AnalysisKeys:
    if mode == "dead":
        return AnalysisKeys(
            mode=mode,
            status_label="dead",
            rate_key="off_rates",
            heads_key="dead_heads_by_step",
            count_key="dead_head_count",
            mostly_count_key="mostly_dead_head_count",
            threshold_key="dead_threshold",
            mostly_threshold_key="mostly_dead_threshold",
            default_threshold=0.95,
            default_mostly_threshold=0.75,
            metric_label="Off-rate",
            heatmap_title="Head off-rate heatmap",
            heatmap_name="head_off_rate_heatmap",
        )
    if mode == "dormant":
        return AnalysisKeys(
            mode=mode,
            status_label="dormant",
            rate_key="sink_attention",
            heads_key="dormant_heads_by_step",
            count_key="dormant_head_count",
            mostly_count_key="mostly_dormant_head_count",
            threshold_key="dormant_threshold",
            mostly_threshold_key="mostly_dormant_threshold",
            default_threshold=0.9,
            default_mostly_threshold=0.75,
            metric_label="Sink attention fraction",
            heatmap_title="Sink attention fraction heatmap",
            heatmap_name="head_sink_attention_heatmap",
        )
    raise ValueError(f"Unsupported mode: {mode}")


def sorted_steps(data: dict[str, Any]) -> list[int]:
    steps = [int(step) for step in data.get("steps", [])]
    if not steps:
        raise ValueError("No steps found in analysis JSON.")
    return sorted(steps)


def metric_matrix(
    data: dict[str, Any], steps: list[int], rate_key: str
) -> tuple[np.ndarray, list[int]]:
    rates = data.get(rate_key)
    if rates is None:
        raise ValueError(f"Missing {rate_key} in analysis JSON.")
    matrix = []
    layer_sizes: list[int] = []
    for layer_idx in range(len(rates[str(steps[0])])):
        layer_sizes.append(len(rates[str(steps[0])][layer_idx]))
    for step in steps:
        step_rates = rates[str(step)]
        flattened = [rate for layer in step_rates for rate in layer]
        matrix.append(flattened)
    return np.array(matrix).T, layer_sizes


def heads_by_step(
    data: dict[str, Any], steps: list[int], heads_key: str
) -> dict[int, set[tuple[int, int]]]:
    heads_by_step = data.get(heads_key)
    if heads_by_step is None:
        raise ValueError(f"Missing {heads_key} in analysis JSON.")
    result = {}
    for step in steps:
        pairs = heads_by_step[str(step)]
        result[step] = {(int(layer), int(head)) for layer, head in pairs}
    return result


def persistent_dead_heads(heads_by_step: dict[int, set[tuple[int, int]]]) -> set[tuple[int, int]]:
    steps = list(heads_by_step.keys())
    if not steps:
        return set()
    persistent = heads_by_step[steps[0]].copy()
    for step in steps[1:]:
        persistent &= heads_by_step[step]
    return persistent


def counts_per_step(
    data: dict[str, Any],
    steps: list[int],
    count_key: str,
    mostly_count_key: str,
    heads_by_step_map: dict[int, set[tuple[int, int]]],
) -> tuple[list[int], list[int]]:
    counts = data.get(count_key)
    mostly_counts = data.get(mostly_count_key)
    if counts is None:
        counts = {str(step): len(heads_by_step_map[step]) for step in steps}
    if mostly_counts is None:
        mostly_counts = {str(step): 0 for step in steps}
    return (
        [int(counts[str(step)]) for step in steps],
        [int(mostly_counts[str(step)]) for step in steps],
    )


def layer_dead_counts_at_step(
    heads_by_step: dict[int, set[tuple[int, int]]], step: int
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for layer, _head in heads_by_step[step]:
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def build_persistent_dead_grid(
    persistent_heads: set[tuple[int, int]],
    num_layers: int,
    num_heads: int,
    head_bin_size: int,
    layer_bin_size: int,
) -> tuple[np.ndarray, list[str], list[str], int]:
    if head_bin_size <= 0:
        raise ValueError("--head-bin-size must be positive.")
    if layer_bin_size <= 0:
        raise ValueError("--layer-bin-size must be positive.")
    head_bins = int(np.ceil(num_heads / head_bin_size))
    layer_bins = int(np.ceil(num_layers / layer_bin_size))
    grid = np.zeros((layer_bins, head_bins), dtype=float)
    for layer, head in persistent_heads:
        if 0 <= layer < num_layers and 0 <= head < num_heads:
            layer_idx = layer // layer_bin_size
            head_idx = head // head_bin_size
            grid[layer_idx, head_idx] += 1.0
    head_labels = [
        f"{idx * head_bin_size}-{min((idx + 1) * head_bin_size - 1, num_heads - 1)}"
        for idx in range(head_bins)
    ]
    layer_labels = [
        f"{idx * layer_bin_size}-{min((idx + 1) * layer_bin_size - 1, num_layers - 1)}"
        for idx in range(layer_bins)
    ]
    vmax = max(1, head_bin_size * layer_bin_size)
    return grid, head_labels, layer_labels, vmax


def save_figure(fig: plt.Figure, output_dir: Path, name: str, formats: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{name}.{fmt}", bbox_inches="tight")


def plot_dead_counts(
    steps: list[int],
    dead_counts: list[int],
    mostly_dead_counts: list[int],
    step_tick_step: int,
    sizes: PlotSizes,
    output_dir: Path,
    formats: Iterable[str],
    status_label: str,
    threshold: float,
    mostly_threshold: float,
    output_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=sizes.double_col)
    ax.plot(steps, dead_counts, marker="o", label=f"{status_label} (>{threshold:.2f})")
    ax.plot(
        steps,
        mostly_dead_counts,
        marker="o",
        linestyle="--",
        label=f"mostly {status_label} (>{mostly_threshold:.2f})",
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Head count")
    ax.set_title(f"{status_label.capitalize()} heads over training")
    tick_steps = steps[::step_tick_step] if step_tick_step > 0 else steps
    ax.set_xticks(tick_steps)
    ax.set_xticklabels([f"{step//1000}k" for step in tick_steps])
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir, output_name, formats)
    plt.close(fig)


def plot_off_rate_heatmap(
    matrix: np.ndarray,
    steps: list[int],
    layer_sizes: list[int],
    sizes: PlotSizes,
    output_dir: Path,
    formats: Iterable[str],
    title: str,
    colorbar_label: str,
    output_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=sizes.heatmap)
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Head index (layer-stacked)")
    ax.set_title(title)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(step) for step in steps], rotation=45, ha="right")
    offset = 0
    for size in layer_sizes[:-1]:
        offset += size
        ax.axhline(offset - 0.5, color="white", linewidth=0.5, alpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label=colorbar_label)
    save_figure(fig, output_dir, output_name, formats)
    plt.close(fig)


def plot_persistent_dead_map(
    persistent_heads: set[tuple[int, int]],
    num_layers: int,
    num_heads: int,
    head_bin_size: int,
    layer_bin_size: int,
    layer_tick_step: int,
    sizes: PlotSizes,
    output_dir: Path,
    formats: Iterable[str],
    status_label: str,
    output_name: str,
) -> None:
    grid, head_labels, layer_labels, vmax = build_persistent_dead_grid(
        persistent_heads,
        num_layers=num_layers,
        num_heads=num_heads,
        head_bin_size=head_bin_size,
        layer_bin_size=layer_bin_size,
    )
    fig, ax = plt.subplots(figsize=sizes.double_col)
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xlabel(f"Head index (bin size={head_bin_size})")
    ax.set_ylabel("Layer")
    ax.set_title(f"Persistent {status_label} heads (binned)")
    ax.set_xticks(range(len(head_labels)))
    ax.set_xticklabels(head_labels, rotation=0)
    layer_ticks = (
        list(range(0, len(layer_labels), layer_tick_step))
        if layer_tick_step > 0
        else list(range(len(layer_labels)))
    )
    ax.set_yticks(layer_ticks)
    ax.set_yticklabels([layer_labels[idx] for idx in layer_ticks])
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.02,
        pad=0.02,
        label=f"Persistent {status_label} (count/bin)",
    )
    save_figure(fig, output_dir, output_name, formats)
    plt.close(fig)


def plot_dead_layer_counts(
    counts: dict[int, int],
    num_layers: int,
    sizes: PlotSizes,
    output_dir: Path,
    formats: Iterable[str],
    status_label: str,
    output_name: str,
) -> None:
    layers = list(range(num_layers))
    values = [counts.get(layer, 0) for layer in layers]
    fig, ax = plt.subplots(figsize=sizes.single_col)
    ax.bar(layers, values)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"{status_label.capitalize()} heads")
    ax.set_title(f"{status_label.capitalize()} heads by layer (final step)")
    ax.set_xticks(layers)
    save_figure(fig, output_dir, output_name, formats)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.use_tex:
        plt.rcParams["text.usetex"] = False
    data = load_analysis(args.input)
    mode = detect_mode(args.mode, data)
    keys = analysis_keys(mode)
    steps = sorted_steps(data)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    sizes = PlotSizes(
        single_col=(args.single_col_width, args.single_col_height),
        double_col=(args.double_col_width, args.double_col_height),
        heatmap=(args.double_col_width, args.heatmap_height),
    )

    heads_by_step_map = heads_by_step(data, steps, keys.heads_key)
    dead_counts, mostly_dead_counts = counts_per_step(
        data, steps, keys.count_key, keys.mostly_count_key, heads_by_step_map
    )
    matrix, layer_sizes = metric_matrix(data, steps, keys.rate_key)
    persistent = persistent_dead_heads(heads_by_step_map)

    num_layers = len(layer_sizes)
    num_heads = max(layer_sizes) if layer_sizes else 0
    last_step = steps[-1]
    layer_counts = layer_dead_counts_at_step(heads_by_step_map, last_step)
    threshold = float(data.get(keys.threshold_key, keys.default_threshold))
    mostly_threshold = float(data.get(keys.mostly_threshold_key, keys.default_mostly_threshold))

    plot_dead_counts(
        steps,
        dead_counts,
        mostly_dead_counts,
        args.step_tick_step,
        sizes,
        args.output_dir,
        formats,
        status_label=keys.status_label,
        threshold=threshold,
        mostly_threshold=mostly_threshold,
        output_name=f"{keys.status_label}_head_count",
    )
    plot_persistent_dead_map(
        persistent,
        num_layers=num_layers,
        num_heads=num_heads,
        head_bin_size=args.head_bin_size,
        layer_bin_size=args.layer_bin_size,
        layer_tick_step=args.layer_tick_step,
        sizes=sizes,
        output_dir=args.output_dir,
        formats=formats,
        status_label=keys.status_label,
        output_name=f"persistent_{keys.status_label}_heads",
    )
    if args.plot_all:
        plot_off_rate_heatmap(
            matrix,
            steps,
            layer_sizes,
            sizes,
            args.output_dir,
            formats,
            title=keys.heatmap_title,
            colorbar_label=keys.metric_label,
            output_name=keys.heatmap_name,
        )
        plot_dead_layer_counts(
            layer_counts,
            num_layers,
            sizes,
            args.output_dir,
            formats,
            status_label=keys.status_label,
            output_name=f"{keys.status_label}_heads_by_layer_last_step",
        )


if __name__ == "__main__":
    main()
