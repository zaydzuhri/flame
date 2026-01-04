#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize head-dead or head-dormant analysis JSON outputs."
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
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--show-heads",
        choices=["none", "persistent", "new", "revived", "all"],
        default="none",
        help="Which head lists to print (default: none).",
    )
    parser.add_argument(
        "--max-heads",
        type=int,
        default=50,
        help="Max number of heads to list per category (default: 50).",
    )
    return parser.parse_args()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sorted_steps(steps: Iterable[Any]) -> list[int]:
    parsed = [_safe_int(step) for step in steps]
    return sorted(step for step in parsed if step is not None)


def _heads_to_set(heads: Iterable[Iterable[int]]) -> set[tuple[int, int]]:
    return {(int(layer), int(head)) for layer, head in heads}


def _detect_mode(mode: str, data: dict[str, Any]) -> str:
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


def _analysis_keys(mode: str) -> dict[str, Any]:
    if mode == "dead":
        return {
            "status": "dead",
            "rate_key": "off_rates",
            "heads_key": "dead_heads_by_step",
            "mostly_heads_key": "mostly_dead_heads_by_step",
            "threshold_key": "dead_threshold",
            "mostly_threshold_key": "mostly_dead_threshold",
            "default_threshold": 0.95,
            "default_mostly_threshold": 0.75,
        }
    if mode == "dormant":
        return {
            "status": "dormant",
            "rate_key": "sink_attention",
            "heads_key": "dormant_heads_by_step",
            "mostly_heads_key": "mostly_dormant_heads_by_step",
            "threshold_key": "dormant_threshold",
            "mostly_threshold_key": "mostly_dormant_threshold",
            "default_threshold": 0.9,
            "default_mostly_threshold": 0.75,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def _infer_heads_by_step(
    data: dict[str, Any],
    rate_key: str,
    heads_key: str,
    threshold_key: str,
    default_threshold: float,
) -> dict[str, list[list[int]]]:
    if heads_key in data:
        return {
            str(step): [list(pair) for pair in pairs]
            for step, pairs in data[heads_key].items()
        }
    rates = data.get(rate_key)
    if rates is None:
        raise ValueError(f"Missing {heads_key} and {rate_key}; cannot infer heads.")
    threshold = float(data.get(threshold_key, default_threshold))
    heads_by_step: dict[str, list[list[int]]] = {}
    for step, layer_rates in rates.items():
        step_heads: list[list[int]] = []
        for layer_idx, head_rates in enumerate(layer_rates):
            for head_idx, rate in enumerate(head_rates):
                if rate > threshold:
                    step_heads.append([layer_idx, head_idx])
        heads_by_step[str(step)] = step_heads
    return heads_by_step


def _infer_mostly_heads_by_step(
    data: dict[str, Any],
    rate_key: str,
    heads_key: str,
    threshold_key: str,
    default_threshold: float,
) -> dict[str, list[list[int]]]:
    if heads_key in data:
        return {
            str(step): [list(pair) for pair in pairs]
            for step, pairs in data[heads_key].items()
        }
    rates = data.get(rate_key)
    if rates is None:
        raise ValueError(f"Missing {heads_key} and {rate_key}; cannot infer heads.")
    threshold = float(data.get(threshold_key, default_threshold))
    heads_by_step: dict[str, list[list[int]]] = {}
    for step, layer_rates in rates.items():
        step_heads: list[list[int]] = []
        for layer_idx, head_rates in enumerate(layer_rates):
            for head_idx, rate in enumerate(head_rates):
                if rate > threshold:
                    step_heads.append([layer_idx, head_idx])
        heads_by_step[str(step)] = step_heads
    return heads_by_step


def compute_summary(data: dict[str, Any], mode: str) -> dict[str, Any]:
    steps = _sorted_steps(data.get("steps", []))
    if not steps:
        raise ValueError("No steps found in analysis output.")

    keys = _analysis_keys(mode)
    status = keys["status"]
    total_heads = int(data.get("lifetime_summary", {}).get("total_heads", 0))
    heads_by_step = _infer_heads_by_step(
        data,
        keys["rate_key"],
        keys["heads_key"],
        keys["threshold_key"],
        keys["default_threshold"],
    )
    mostly_heads_by_step = _infer_mostly_heads_by_step(
        data,
        keys["rate_key"],
        keys["mostly_heads_key"],
        keys["mostly_threshold_key"],
        keys["default_mostly_threshold"],
    )

    head_sets = {step: _heads_to_set(heads_by_step[str(step)]) for step in steps}
    mostly_sets = {step: _heads_to_set(mostly_heads_by_step[str(step)]) for step in steps}

    persistent_heads = (
        set.intersection(*head_sets.values()) if len(head_sets) > 1 else head_sets[steps[0]]
    )
    head_union = (
        set.union(*head_sets.values()) if len(head_sets) > 1 else head_sets[steps[0]]
    )

    per_step = {}
    prev = None
    for step in steps:
        current = head_sets[step]
        if prev is None:
            new_heads = current
            revived = set()
        else:
            new_heads = current - prev
            revived = prev - current
        per_step[str(step)] = {
            f"{status}_count": len(current),
            f"{status}_fraction": (len(current) / total_heads) if total_heads else None,
            f"new_{status}": sorted(new_heads),
            "revived": sorted(revived),
            f"mostly_{status}_count": len(mostly_sets[step]),
        }
        prev = current

    layer_counts_last = {}
    last_step = steps[-1]
    for layer_idx, head_idx in sorted(head_sets[last_step]):
        layer_counts_last[layer_idx] = layer_counts_last.get(layer_idx, 0) + 1

    return {
        "steps": steps,
        "total_heads": total_heads,
        f"{status}_union_count": len(head_union),
        f"{status}_union_fraction": (len(head_union) / total_heads) if total_heads else None,
        f"persistent_{status}_heads": sorted(persistent_heads),
        f"persistent_{status}_fraction": (len(persistent_heads) / total_heads)
        if total_heads
        else None,
        f"{status}_heads_by_step": {str(step): sorted(head_sets[step]) for step in steps},
        f"mostly_{status}_heads_by_step": {
            str(step): sorted(mostly_sets[step]) for step in steps
        },
        "per_step": per_step,
        f"{status}_layer_counts_last_step": layer_counts_last,
        "mode": mode,
    }


def _format_head_list(heads: list[tuple[int, int]], max_heads: int) -> str:
    if not heads:
        return "[]"
    clipped = heads[:max_heads]
    suffix = "" if len(heads) <= max_heads else f" (+{len(heads) - max_heads} more)"
    return f"{clipped}{suffix}"


def render_text(summary: dict[str, Any], data: dict[str, Any], args: argparse.Namespace) -> str:
    status = summary.get("mode", "dead")
    lines = []
    total_heads = summary["total_heads"]
    lines.append(f"Total heads: {total_heads}")
    lines.append(f"Steps: {summary['steps']}")
    union_key = f"{status}_union_count"
    union_frac_key = f"{status}_union_fraction"
    persistent_key = f"persistent_{status}_heads"
    persistent_frac_key = f"persistent_{status}_fraction"
    lines.append(
        f"{status.capitalize()} union: "
        f"{summary[union_key]} "
        f"({summary[union_frac_key]:.2%})"
        if summary[union_frac_key] is not None
        else f"{status.capitalize()} union: {summary[union_key]}"
    )
    lines.append(
        f"Persistent {status}: "
        f"{len(summary[persistent_key])} "
        f"({summary[persistent_frac_key]:.2%})"
        if summary[persistent_frac_key] is not None
        else f"Persistent {status}: {len(summary[persistent_key])}"
    )

    per_step = summary["per_step"]
    for step in summary["steps"]:
        step_info = per_step[str(step)]
        count_key = f"{status}_count"
        fraction_key = f"{status}_fraction"
        mostly_key = f"mostly_{status}_count"
        new_key = f"new_{status}"
        dead_fraction = step_info[fraction_key]
        if dead_fraction is None:
            line = f"Step {step}: {status}={step_info[count_key]}"
        else:
            line = f"Step {step}: {status}={step_info[count_key]} ({dead_fraction:.2%})"
        line += f", mostly_{status}={step_info[mostly_key]}"
        line += f", new_{status}={len(step_info[new_key])}"
        line += f", revived={len(step_info['revived'])}"
        lines.append(line)

    layer_counts_key = f"{status}_layer_counts_last_step"
    if summary[layer_counts_key]:
        sorted_layers = sorted(
            summary[layer_counts_key].items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_layers = ", ".join(f"L{layer}:{count}" for layer, count in sorted_layers[:10])
        lines.append(f"Top {status} layers at last step: {top_layers}")

    show = args.show_heads
    max_heads = args.max_heads
    if show in ("persistent", "all"):
        heads = summary[persistent_key]
        lines.append(f"Persistent {status} heads: {_format_head_list(heads, max_heads)}")
    if show in ("new", "all"):
        last_step = summary["steps"][-1]
        new_heads = summary["per_step"][str(last_step)][f"new_{status}"]
        lines.append(
            f"Newly {status} at last step: {_format_head_list(new_heads, max_heads)}"
        )
    if show in ("revived", "all"):
        last_step = summary["steps"][-1]
        revived_heads = summary["per_step"][str(last_step)]["revived"]
        lines.append(f"Revived by last step: {_format_head_list(revived_heads, max_heads)}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    mode = _detect_mode(args.mode, data)
    summary = compute_summary(data, mode)

    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(render_text(summary, data, args))


if __name__ == "__main__":
    main()
