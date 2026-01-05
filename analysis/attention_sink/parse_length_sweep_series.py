#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize length_sweep JSONL outputs across checkpoints."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("analysis/attention_sink/outputs_real"),
        help="Directory containing length_sweep JSONL outputs.",
    )
    parser.add_argument(
        "--glob",
        default="vanilla-340M-4096-step-*-truncate-4096_length_sweep.jsonl",
        help="Glob pattern to match length_sweep JSONL files.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Sequence length to select when files contain multiple lengths.",
    )
    parser.add_argument(
        "--sink-eps",
        type=float,
        nargs="+",
        default=None,
        help="Sink eps values to report (default: use keys from file).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the output.",
    )
    return parser.parse_args()


def extract_step(text: str) -> int | None:
    match = re.search(r"step-([0-9]+)", text)
    if match:
        return int(match.group(1))
    return None


def select_record(records: list[dict[str, Any]], sequence_length: int | None) -> dict[str, Any]:
    if not records:
        raise ValueError("No records found in JSONL.")
    if sequence_length is None:
        if len(records) != 1:
            raise ValueError("Multiple records found; use --sequence-length to select one.")
        return records[0]
    for rec in records:
        if rec.get("sequence_length") == sequence_length:
            return rec
    raise ValueError(f"No record found for sequence_length={sequence_length}.")


def format_sink_key(value: float) -> str:
    text = f"{value}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def load_rows(
    input_dir: Path,
    glob: str,
    sequence_length: int | None,
    sink_eps: list[float] | None,
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob(glob)):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        record = select_record(records, sequence_length)
        step = extract_step(path.name) or extract_step(record.get("model", "")) or -1
        sink_rate = record.get("sink_rate", {})
        sparsity = record.get("sparsity", {})
        hidden = record.get("hidden_activation", {})

        if sink_eps is None:
            sink_keys = sorted(
                sink_rate.keys(), key=lambda key: float(key) if key.replace(".", "", 1).isdigit() else key
            )
        else:
            sink_keys = [format_sink_key(value) for value in sink_eps]

        row = {
            "step": step,
            "sequence_length": record.get("sequence_length"),
            "samples_processed": record.get("samples_processed"),
            "sink_rate": {key: sink_rate.get(key) for key in sink_keys},
            "sparsity_at_or_below_eps": sparsity.get("lower_triangle_at_or_below_eps"),
            "sparsity_exact_zero": sparsity.get("lower_triangle_exact_zero"),
            "hidden_kurtosis": hidden.get("kurtosis"),
        }
        rows.append(row)
    rows = sorted(rows, key=lambda row: row["step"])
    return rows


def render_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No rows found."
    sink_keys = list(rows[0]["sink_rate"].keys())
    header = [
        "step",
        "seq_len",
        *[f"sink_rate_{key}" for key in sink_keys],
        "sparsity_eps",
        "sparsity_zero",
        "kurtosis",
        "samples",
    ]
    lines = ["  ".join(header)]
    for row in rows:
        sink_vals = row["sink_rate"]
        values = [
            str(row["step"]),
            str(row["sequence_length"]),
            *[f"{sink_vals.get(key, float('nan')):.4f}" for key in sink_keys],
            f"{row['sparsity_at_or_below_eps']:.4f}",
            f"{row['sparsity_exact_zero']:.4f}",
            f"{row['hidden_kurtosis']:.2f}",
            str(row["samples_processed"]),
        ]
        lines.append("  ".join(values))
    return "\n".join(lines)


def render_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    sink_keys = list(rows[0]["sink_rate"].keys())
    header = [
        "step",
        "sequence_length",
        *[f"sink_rate_{key}" for key in sink_keys],
        "sparsity_at_or_below_eps",
        "sparsity_exact_zero",
        "hidden_kurtosis",
        "samples_processed",
    ]
    lines = [",".join(header)]
    for row in rows:
        sink_vals = row["sink_rate"]
        values = [
            str(row["step"]),
            str(row["sequence_length"]),
            *[str(sink_vals.get(key, "")) for key in sink_keys],
            str(row["sparsity_at_or_below_eps"]),
            str(row["sparsity_exact_zero"]),
            str(row["hidden_kurtosis"]),
            str(row["samples_processed"]),
        ]
        lines.append(",".join(values))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_dir, args.glob, args.sequence_length, args.sink_eps)

    if args.format == "json":
        output = json.dumps(rows, indent=2, sort_keys=True)
    elif args.format == "csv":
        output = render_csv(rows)
    else:
        output = render_text(rows)

    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
