#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_metric_key(metric: str) -> str:
    return metric.replace(":", ".")


def extract_metric(record: dict, metric: str) -> float | None:
    metric = normalize_metric_key(metric)
    if metric.startswith("sink_rate."):
        _, eps = metric.split(".", 1)
        return record.get("sink_rate", {}).get(eps)
    if metric.startswith("hidden_activation."):
        _, key = metric.split(".", 1)
        return record.get("hidden_activation", {}).get(key)
    if metric.startswith("sparsity."):
        _, key = metric.split(".", 1)
        return record.get("sparsity", {}).get(key)
    return record.get(metric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot length sweep metrics.")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="One or more length_sweep.jsonl files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing *_length_sweep.jsonl files.",
    )
    parser.add_argument(
        "--metric",
        default="sink_rate.0.2",
        help="Metric key (e.g., sink_rate.0.2, hidden_activation.kurtosis).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (defaults under analysis/attention_sink/outputs/plots).",
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display the plot interactively.",
    )
    return parser.parse_args()


def collect_inputs(inputs: list[Path] | None, input_dir: Path | None) -> list[Path]:
    files: list[Path] = []
    if input_dir is not None:
        files.extend(sorted(input_dir.glob("*_length_sweep.jsonl")))
    if inputs:
        files.extend(inputs)
    return files


def main() -> None:
    args = parse_args()
    inputs = collect_inputs(args.inputs, args.input_dir)
    if not inputs:
        raise SystemExit("No input files provided.")

    series: dict[str, list[tuple[int, float]]] = {}
    for path in inputs:
        for record in iter_jsonl(path):
            length = record.get("sequence_length") or record.get("max_length")
            if length is None:
                continue
            value = extract_metric(record, args.metric)
            if value is None:
                continue
            label = record.get("run_name") or record.get("model") or path.stem
            series.setdefault(label, []).append((int(length), float(value)))

    if not series:
        raise SystemExit("No matching metrics found in inputs.")

    plt.figure(figsize=(8, 5))
    for label, points in series.items():
        points = sorted(points, key=lambda item: item[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Sequence length")
    plt.ylabel(args.metric)
    if args.title:
        plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = args.output
    if output_path is None:
        output_dir = Path("analysis/attention_sink/outputs/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_metric = normalize_metric_key(args.metric).replace(".", "_")
        output_path = output_dir / f"length_sweep_{safe_metric}.png"

    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
