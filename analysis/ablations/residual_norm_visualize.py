#!/usr/bin/env python3
"""
Residual Norm Visualization

Generates visualizations from residual_norm_analysis.py output:
1. Layer-wise plot: X=Layer, Y=Mean Residual Norm
2. Position-wise plot: X=Position, Y=Residual Norm (at final layer)
3. Heatmap: Layers x Positions showing norm magnitude
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_layerwise_comparison(
    results_list: list[tuple[str, dict]],
    output_path: Path,
    metric: str = "mean_post_ffn_norm",
    title: str = "Layer-wise Residual Norm",
) -> None:
    """Plot layer-wise residual norms for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in results_list:
        # Get the first step's results (or aggregate if needed)
        steps = data.get("steps", [])
        if not steps:
            continue

        # Use the last step for comparison
        step_key = str(steps[-1])
        step_results = data.get("results_by_step", {}).get(step_key, {})
        norms = step_results.get(metric, [])

        if norms:
            layers = list(range(len(norms)))
            ax.plot(layers, norms, marker="o", label=f"{label} (step {step_key})", linewidth=2)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # Log scale often helps visualize norm growth

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_layerwise_over_training(
    data: dict,
    output_path: Path,
    metric: str = "mean_post_ffn_norm",
    title: str = "Layer-wise Residual Norm Over Training",
) -> None:
    """Plot layer-wise norms for multiple training steps."""
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = data.get("steps", [])
    results_by_step = data.get("results_by_step", {})

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(steps) - 1, 1)) for i in range(len(steps))]

    for step, color in zip(steps, colors):
        step_results = results_by_step.get(str(step), {})
        norms = step_results.get(metric, [])
        if norms:
            layers = list(range(len(norms)))
            ax.plot(layers, norms, marker="o", color=color, label=f"Step {step}", linewidth=1.5, markersize=4)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_positionwise(
    results_list: list[tuple[str, dict]],
    output_path: Path,
    max_positions: int = 1024,
    title: str = "Position-wise Residual Norm (Final Layer)",
) -> None:
    """Plot position-wise residual norms at the final layer."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for label, data in results_list:
        steps = data.get("steps", [])
        if not steps:
            continue

        step_key = str(steps[-1])
        step_results = data.get("results_by_step", {}).get(step_key, {})
        position_norms = step_results.get("position_norms_final_layer", [])

        if position_norms:
            # Limit to max_positions for clarity
            pos_to_plot = position_norms[:max_positions]
            positions = list(range(len(pos_to_plot)))
            ax.plot(positions, pos_to_plot, label=f"{label} (step {step_key})", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Position in Sequence", fontsize=12)
    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_heatmap(
    data: dict,
    output_path: Path,
    metric: str = "mean_post_ffn_norm",
    title: str = "Residual Norm Heatmap (Steps x Layers)",
) -> None:
    """Create heatmap of norms: Steps x Layers."""
    steps = data.get("steps", [])
    results_by_step = data.get("results_by_step", {})

    if not steps:
        print("No steps found, skipping heatmap.")
        return

    # Build matrix: rows=steps, cols=layers
    first_step = str(steps[0])
    num_layers = len(results_by_step.get(first_step, {}).get(metric, []))
    if num_layers == 0:
        print(f"No {metric} data found, skipping heatmap.")
        return

    matrix = np.zeros((len(steps), num_layers))
    for i, step in enumerate(steps):
        norms = results_by_step.get(str(step), {}).get(metric, [])
        if norms:
            matrix[i, :] = norms

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Training Step", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Set y-tick labels to step numbers
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([str(s) for s in steps])

    # Set x-tick labels (sparse if many layers)
    if num_layers <= 30:
        ax.set_xticks(range(num_layers))
    else:
        tick_interval = num_layers // 10
        ax.set_xticks(range(0, num_layers, tick_interval))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("L2 Norm", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_growth_ratio(
    results_list: list[tuple[str, dict]],
    output_path: Path,
    title: str = "Norm Growth Ratio (Layer N / Layer 0)",
) -> None:
    """Plot the ratio of final layer norm to first layer norm."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in results_list:
        steps = data.get("steps", [])
        results_by_step = data.get("results_by_step", {})

        ratios = []
        step_values = []
        for step in steps:
            step_results = results_by_step.get(str(step), {})
            norms = step_results.get("mean_post_ffn_norm", [])
            if len(norms) >= 2 and norms[0] > 0:
                ratio = norms[-1] / norms[0]
                ratios.append(ratio)
                step_values.append(step)

        if ratios:
            ax.plot(step_values, ratios, marker="o", label=label, linewidth=2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Norm Ratio (Last/First Layer)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No growth")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_max_norm_comparison(
    results_list: list[tuple[str, dict]],
    output_path: Path,
    title: str = "Max Residual Norm by Layer",
) -> None:
    """Plot maximum residual norms (worst case) per layer."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in results_list:
        steps = data.get("steps", [])
        if not steps:
            continue

        step_key = str(steps[-1])
        step_results = data.get("results_by_step", {}).get(step_key, {})
        max_norms = step_results.get("max_post_ffn_norm", [])

        if max_norms:
            layers = list(range(len(max_norms)))
            ax.plot(layers, max_norms, marker="s", label=f"{label} (step {step_key})", linewidth=2)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Max L2 Norm", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize residual norm analysis results.")
    parser.add_argument(
        "--vanilla",
        type=Path,
        help="Path to vanilla (softmax) results JSON.",
    )
    parser.add_argument(
        "--softpick",
        type=Path,
        help="Path to softpick results JSON.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Additional input JSON files (can be repeated).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        action="append",
        default=[],
        help="Labels for additional inputs (same order as --input).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/ablations/outputs/residual_norm"),
        help="Output directory for plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_list: list[tuple[str, dict]] = []

    # Load named results
    if args.vanilla and args.vanilla.exists():
        results_list.append(("Vanilla (Softmax)", load_results(args.vanilla)))
    if args.softpick and args.softpick.exists():
        results_list.append(("Softpick", load_results(args.softpick)))

    # Load additional inputs
    for i, input_path in enumerate(args.input):
        if input_path.exists():
            label = args.labels[i] if i < len(args.labels) else input_path.stem
            results_list.append((label, load_results(input_path)))

    if not results_list:
        print("No input files provided or found. Use --vanilla, --softpick, or --input.")
        return

    print(f"Loaded {len(results_list)} result files:")
    for label, _ in results_list:
        print(f"  - {label}")

    # Generate comparison plots
    plot_layerwise_comparison(
        results_list,
        args.output_dir / "residual_layerwise_comparison.png",
        metric="mean_post_ffn_norm",
        title="Layer-wise Mean Residual Norm (Post-FFN)",
    )

    plot_layerwise_comparison(
        results_list,
        args.output_dir / "residual_layerwise_post_attn.png",
        metric="mean_post_attn_norm",
        title="Layer-wise Mean Residual Norm (Post-Attention)",
    )

    plot_positionwise(
        results_list,
        args.output_dir / "residual_positionwise.png",
        title="Position-wise Residual Norm (Final Layer)",
    )

    plot_max_norm_comparison(
        results_list,
        args.output_dir / "residual_max_norm.png",
        title="Max Residual Norm by Layer",
    )

    plot_growth_ratio(
        results_list,
        args.output_dir / "residual_growth_ratio.png",
        title="Residual Norm Growth Ratio Over Training",
    )

    # Generate per-model heatmaps
    for label, data in results_list:
        safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_heatmap(
            data,
            args.output_dir / f"residual_heatmap_{safe_label}.png",
            title=f"Residual Norm Heatmap - {label}",
        )

        plot_layerwise_over_training(
            data,
            args.output_dir / f"residual_training_{safe_label}.png",
            title=f"Layer-wise Norm Over Training - {label}",
        )

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
