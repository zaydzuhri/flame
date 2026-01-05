#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan multiple dormant thresholds from Guo-style dormant outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to head_dormant_analysis_guo JSON output.",
    )
    parser.add_argument(
        "--thresholds",
        required=True,
        help="Comma-separated dormant thresholds (e.g. 0.95,0.9,0.8).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON output.",
    )
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    if not thresholds:
        raise ValueError("Provide at least one threshold in --thresholds.")
    for threshold in thresholds:
        if threshold <= 0 or threshold > 1:
            raise ValueError("Thresholds must be in (0, 1].")
    return thresholds


def sorted_steps(data: dict[str, Any]) -> list[int]:
    steps = [int(step) for step in data.get("steps", [])]
    if not steps:
        raise ValueError("No steps found in input.")
    return sorted(steps)


def total_heads_from_rates(dormant_rate: dict[str, Any]) -> int:
    first_key = next(iter(dormant_rate))
    layers = dormant_rate[first_key]
    return sum(len(layer) for layer in layers)


def compute_threshold_summary(
    data: dict[str, Any],
    thresholds: Iterable[float],
) -> dict[str, Any]:
    steps = sorted_steps(data)
    dormant_rate = data.get("dormant_rate")
    if dormant_rate is None:
        raise ValueError("Input JSON must contain dormant_rate.")
    total_heads = data.get("lifetime_summary", {}).get(
        "total_heads", total_heads_from_rates(dormant_rate)
    )

    summary: dict[str, Any] = {
        "analysis_type": "dormant_threshold_scan",
        "steps": steps,
        "total_heads": total_heads,
        "thresholds": thresholds,
        "per_threshold": {},
    }

    for threshold in thresholds:
        heads_by_step: dict[int, set[tuple[int, int]]] = {}
        per_step_counts: dict[str, int] = {}
        for step in steps:
            step_rates = dormant_rate[str(step)]
            heads: set[tuple[int, int]] = set()
            for layer_idx, layer_rates in enumerate(step_rates):
                for head_idx, rate in enumerate(layer_rates):
                    if rate > threshold:
                        heads.add((layer_idx, head_idx))
            heads_by_step[step] = heads
            per_step_counts[str(step)] = len(heads)

        union = set.union(*heads_by_step.values()) if heads_by_step else set()
        intersection = (
            set.intersection(*heads_by_step.values()) if heads_by_step else set()
        )

        summary["per_threshold"][str(threshold)] = {
            "dormant_head_count": per_step_counts,
            "union_count": len(union),
            "persistent_count": len(intersection),
        }

    return summary


def render_text(summary: dict[str, Any]) -> str:
    lines = []
    lines.append(f"Steps: {summary['steps']}")
    lines.append(f"Total heads: {summary['total_heads']}")
    for threshold, details in summary["per_threshold"].items():
        lines.append(f"Threshold {threshold}:")
        lines.append(
            f"  union={details['union_count']} persistent={details['persistent_count']}"
        )
        for step, count in details["dormant_head_count"].items():
            lines.append(f"  step {step}: {count}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    summary = compute_threshold_summary(data, thresholds)

    if args.format == "json":
        output = json.dumps(summary, indent=2, sort_keys=True)
    else:
        output = render_text(summary)
    print(output)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
