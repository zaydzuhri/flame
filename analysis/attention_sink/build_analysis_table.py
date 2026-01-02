#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SIZE_PATTERN = re.compile(r"(?P<size>\d+(?:\.\d+)?[MB])", re.IGNORECASE)
MODEL_ALIASES = {
    "softmax": "Softmax",
    "vanilla": "Softmax",
    "softpick": "Softpick",
    "gated_attention": "Gated Attention",
    "gpt_oss_sink": "GPT-OSS Sink",
    "rectified": "Rectified",
    "relusoftpick1": "ReluSoftpick1",
    "scaled_softpick": "Scaled Softpick",
    "softpick_plus_one": "Softpick+1",
}
MODEL_PREFIX_TO_LABEL = {
    "vanilla": "Softmax",
    "softmax": "Softmax",
    "softpick": "Softpick",
    "gated_attention": "Gated Attention",
    "gpt_oss_sink": "GPT-OSS Sink",
    "rectified": "Rectified",
    "relusoftpick1": "ReluSoftpick1",
    "scaled_softpick": "Scaled Softpick",
    "softpick_plus_one": "Softpick+1",
}
DEFAULT_MODEL_ORDER = [
    "Softmax",
    "Softpick",
    "Gated Attention",
    "GPT-OSS Sink",
    "Rectified",
    "ReluSoftpick1",
    "Scaled Softpick",
    "Softpick+1",
]


@dataclass(frozen=True)
class SweepRow:
    sequence_length: int
    sink_rate: dict[str, float]
    hidden_activation: dict[str, float]
    sparsity: dict[str, float]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_size(value: str) -> str:
    match = SIZE_PATTERN.search(value)
    if not match:
        raise ValueError(f"Unable to infer size from '{value}'.")
    size = match.group("size")
    return size.replace("b", "B").replace("m", "M")


def infer_size_from_name(name: str) -> str | None:
    match = SIZE_PATTERN.search(name)
    if not match:
        return None
    return match.group("size").replace("b", "B").replace("m", "M")


def size_sort_key(size: str) -> tuple[float, str]:
    match = SIZE_PATTERN.search(size)
    if not match:
        return (float("inf"), size)
    value = float(match.group("size")[:-1])
    unit = match.group("size")[-1].upper()
    multiplier = 1000.0 if unit == "B" else 1.0
    return (value * multiplier, size)


def normalize_model_label(label: str) -> str:
    normalized = MODEL_ALIASES.get(label.strip().lower())
    if normalized is not None:
        return normalized
    cleaned = label.replace("_", " ").strip()
    return cleaned.title() if cleaned else label


def classify_model(name: str) -> str | None:
    lowered = name.lower()
    for prefix, label in MODEL_PREFIX_TO_LABEL.items():
        if lowered.startswith(f"{prefix}-"):
            return label
    return None


def parse_model_file(value: str) -> tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "Expected --model-file to be SIZE:MODEL:FILE (e.g., 340M:Softmax:vanilla.jsonl)."
        )
    size, model, path = parts
    model_label = normalize_model_label(model)
    return normalize_size(size), model_label, Path(path)


def select_row(rows: list[dict], sequence_length: int | None) -> SweepRow:
    if not rows:
        raise ValueError("No rows found in sweep.")
    lengths = []
    for row in rows:
        length = row.get("sequence_length") or row.get("max_length")
        if length is None:
            continue
        lengths.append(int(length))

    if not lengths:
        raise ValueError("No sequence_length entries found in sweep.")

    if sequence_length is None:
        target_length = max(lengths)
    else:
        target_length = sequence_length

    for row in rows:
        length = row.get("sequence_length") or row.get("max_length")
        if length is None:
            continue
        if int(length) == target_length:
            return SweepRow(
                sequence_length=int(length),
                sink_rate=row.get("sink_rate", {}),
                hidden_activation=row.get("hidden_activation", {}),
                sparsity=row.get("sparsity", {}),
            )

    available = ", ".join(str(length) for length in sorted(set(lengths)))
    raise ValueError(
        f"Sequence length {target_length} not found (available: {available})."
    )


def format_value(value: float) -> str:
    return f"{value:.2f}"


def format_percent(value: float) -> str:
    return format_value(value * 100.0)


def format_eps(value: float) -> str:
    return f"{value:g}"


def build_rows(
    sources: dict[tuple[str, str], Path],
    sequence_length: int | None,
) -> dict[tuple[str, str], SweepRow]:
    rows: dict[tuple[str, str], SweepRow] = {}
    for key, path in sources.items():
        records = list(iter_jsonl(path))
        rows[key] = select_row(records, sequence_length)
    return rows


def render_table(
    rows: dict[tuple[str, str], SweepRow],
    sizes: list[str],
    models: list[str],
    sink_eps: list[float],
    caption: str,
    label: str,
    mark_softmax_sparsity: bool,
) -> str:
    sink_header = " & ".join(f"$\\epsilon_s{{=}}{format_eps(eps)}$" for eps in sink_eps)
    sink_cols = "c" * len(sink_eps)
    lines = [
        "\\begin{table}[t]",
        f"\\caption{{{caption}}}",
        "\\small",
        "\\centering",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{ll{sink_cols}}}",
        "\\toprule",
        f"Size & Model & {sink_header} \\\\",
        "\\midrule",
    ]

    for size in sizes:
        available_models = [model for model in models if (size, model) in rows]
        if not available_models:
            continue
        total_models = len(available_models)
        for idx, model in enumerate(available_models):
            row = rows[(size, model)]
            values = [
                format_percent(row.sink_rate[str(eps)]) for eps in sink_eps
            ]
            sink_cells = " & ".join(values)
            if idx == 0:
                lines.append(
                    f"\\multirow{{{total_models}}}{{*}}{{{size}}} & {model} & "
                    f"{sink_cells} \\\\"
                )
            else:
                lines.append(
                    f"                      & {model} & {sink_cells} \\\\"
                )
        if size != sizes[-1]:
            lines.append("\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\vspace{0.5em}",
            "",
            "\\begin{tabular}{llcccc}",
            "\\toprule",
            "Size & Model & Kurt. & Min. & Max. & Spars.\\% \\\\",
            "\\midrule",
        ]
    )

    for size in sizes:
        available_models = [model for model in models if (size, model) in rows]
        if not available_models:
            continue
        total_models = len(available_models)
        for idx, model in enumerate(available_models):
            row = rows[(size, model)]
            sparsity = format_percent(row.sparsity["lower_triangle_exact_zero"])
            if mark_softmax_sparsity and model == "Softmax":
                sparsity += "*"
            cells = (
                f"{format_value(row.hidden_activation['kurtosis'])} & "
                f"{format_value(row.hidden_activation['min'])} & "
                f"{format_value(row.hidden_activation['max'])} & "
                f"{sparsity}"
            )
            if idx == 0:
                lines.append(
                    f"\\multirow{{{total_models}}}{{*}}{{{size}}} & {model} & "
                    f"{cells} \\\\"
                )
            else:
                lines.append(
                    f"                      & {model} & {cells} \\\\"
                )
        if size != sizes[-1]:
            lines.append("\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LaTeX tables from attention sink sweep outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("analysis/attention_sink/outputs_real"),
        help="Directory containing *_sweep.jsonl files.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Sequence length to select (defaults to max available).",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        help="Explicit sizes to include (e.g., 340M 1.8B).",
    )
    parser.add_argument(
        "--sink-eps",
        type=float,
        nargs="+",
        default=[0.2, 0.3],
        help="Sink epsilon values to include.",
    )
    parser.add_argument(
        "--model-file",
        action="append",
        default=[],
        help="Explicit mapping SIZE:MODEL:FILE (MODEL in {Softmax, Softpick}).",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Metrics for analysis of softmax and softpick in terms of sink rate "
            "(in percentage), hidden activation distribution, and attention map "
            "sparsity. *Note that sparsity was calculated by counting exact zeros "
            "in the lower triangle, which means the softmax sparsity only comes "
            "from underflowed values."
        ),
        help="Caption to use for the LaTeX table.",
    )
    parser.add_argument(
        "--label",
        default="tab:analysis-table",
        help="LaTeX label to assign.",
    )
    parser.add_argument(
        "--model-order",
        nargs="+",
        default=None,
        help="Optional model order (e.g., Softmax Softpick Gated\\ Attention).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (defaults to stdout).",
    )
    parser.add_argument(
        "--mark-softmax-sparsity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append '*' to softmax sparsity values.",
    )
    return parser.parse_args()


def collect_sources(
    input_dir: Path,
    model_files: list[str],
) -> dict[tuple[str, str], Path]:
    if model_files:
        sources: dict[tuple[str, str], Path] = {}
        for entry in model_files:
            size, model, path = parse_model_file(entry)
            sources[(size, model)] = path
        return sources

    candidates: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.jsonl")):
        model = classify_model(path.name)
        if model is None:
            continue
        size = infer_size_from_name(path.name)
        if size is None:
            continue
        candidates[(size, model)].append(path)

    sources = {}
    for key, paths in candidates.items():
        if len(paths) > 1:
            names = ", ".join(p.name for p in paths)
            raise ValueError(
                f"Multiple files found for {key}: {names}. "
                "Use --model-file to disambiguate."
            )
        sources[key] = paths[0]
    return sources


def main() -> None:
    args = parse_args()
    sources = collect_sources(args.input_dir, args.model_file)
    if not sources:
        raise SystemExit("No matching sweep files found.")

    if args.sizes is None:
        sizes = sorted({size for size, _ in sources.keys()}, key=size_sort_key)
    else:
        sizes = [normalize_size(size) for size in args.sizes]

    models_found = sorted({model for _, model in sources.keys()})
    if args.model_order:
        model_order = [normalize_model_label(model) for model in args.model_order]
    else:
        model_order = DEFAULT_MODEL_ORDER
    models = [model for model in model_order if model in models_found]
    models.extend(sorted(model for model in models_found if model not in models))

    rows = build_rows(sources, args.sequence_length)
    latex = render_table(
        rows=rows,
        sizes=sizes,
        models=models,
        sink_eps=args.sink_eps,
        caption=args.caption,
        label=args.label,
        mark_softmax_sparsity=args.mark_softmax_sparsity,
    )

    if args.output:
        args.output.write_text(latex, encoding="utf-8")
    else:
        print(latex, end="")


if __name__ == "__main__":
    main()
