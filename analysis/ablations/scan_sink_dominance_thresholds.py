#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable


DEFAULT_THRESHOLDS = [0.1 * idx for idx in range(1, 10)] + [0.95]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Guo-style dormant analysis across sink-dominance thresholds."
    )
    parser.add_argument(
        "--thresholds",
        default="",
        help="Comma-separated sink-dominance thresholds (default: 0.1..0.9 plus 0.95).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/attention_sink/outputs"),
        help="Directory for per-threshold outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="vanilla-340M-head-dormant-guo",
        help="Prefix for output files (default: vanilla-340M-head-dormant-guo).",
    )
    parser.add_argument(
        "--model-template",
        required=True,
        help="Format string for checkpoint paths (use {step}).",
    )
    parser.add_argument(
        "--steps",
        default="",
        help="Comma-separated checkpoint steps (e.g. 10000,20000).",
    )
    parser.add_argument("--step-start", type=int, default=None)
    parser.add_argument("--step-end", type=int, default=None)
    parser.add_argument("--step-stride", type=int, default=10000)
    parser.add_argument(
        "--dataset",
        default="DKYoon/SlimPajama-6B",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset-config",
        default="",
        help="Dataset config name (leave empty if none).",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset order.")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--padding",
        choices=["none", "longest"],
        default="none",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5_000_000,
        help="Stop after processing this many tokens (-1 for all).",
    )
    parser.add_argument(
        "--sink-index",
        type=int,
        default=0,
        help="Key index treated as sink token (default: 0).",
    )
    parser.add_argument("--entropy-eps", type=float, default=1e-12)
    parser.add_argument("--dormant-threshold", type=float, default=0.95)
    parser.add_argument("--mostly-dormant-threshold", type=float, default=0.75)
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=None,
        help="Optional normalized entropy threshold to require for dormancy.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bf16", "float32"],
    )
    parser.add_argument("--attn-impl", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    if not value.strip():
        return list(DEFAULT_THRESHOLDS)
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    for threshold in thresholds:
        if threshold <= 0 or threshold > 1:
            raise ValueError("Thresholds must be in (0, 1].")
    return sorted(set(thresholds))


def format_threshold(threshold: float) -> str:
    value = Decimal(str(threshold)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return str(value).replace(".", "p")


def build_command(args: argparse.Namespace, threshold: float, output_path: Path) -> list[str]:
    script_path = Path(__file__).with_name("head_dormant_analysis_guo.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--model-template",
        args.model_template,
    ]
    if args.steps:
        cmd += ["--steps", args.steps]
    else:
        cmd += [
            "--step-start",
            str(args.step_start),
            "--step-end",
            str(args.step_end),
            "--step-stride",
            str(args.step_stride),
        ]
    cmd += [
        "--attn-impl",
        args.attn_impl,
        "--dataset",
        args.dataset,
        "--dataset-config",
        args.dataset_config,
        "--split",
        args.split,
        "--batch-size",
        str(args.batch_size),
        "--padding",
        args.padding,
        "--max-length",
        str(args.max_length),
        "--max-tokens",
        str(args.max_tokens),
        "--sink-index",
        str(args.sink_index),
        "--entropy-eps",
        str(args.entropy_eps),
        "--dormant-threshold",
        str(args.dormant_threshold),
        "--mostly-dormant-threshold",
        str(args.mostly_dormant_threshold),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--sink-dominance-threshold",
        str(threshold),
        "--output",
        str(output_path),
    ]
    if args.entropy_threshold is not None:
        cmd += ["--entropy-threshold", str(args.entropy_threshold)]
    if args.streaming:
        cmd.append("--streaming")
    if args.shuffle:
        cmd.append("--shuffle")
        cmd += ["--shuffle-seed", str(args.shuffle_seed)]
        cmd += ["--shuffle-buffer", str(args.shuffle_buffer)]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.local_files_only:
        cmd.append("--local-files-only")
    else:
        cmd.append("--no-local-files-only")
    return cmd


def validate_args(args: argparse.Namespace) -> None:
    if args.attn_impl is None:
        raise ValueError("--attn-impl is required (use naive_attn).")
    if args.steps:
        return
    if args.step_start is None or args.step_end is None:
        raise ValueError("Use --steps or --step-start/--step-end.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    thresholds = parse_thresholds(args.thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for threshold in thresholds:
        suffix = format_threshold(threshold)
        output_path = args.output_dir / f"{args.output_prefix}_sink{suffix}.json"
        cmd = build_command(args, threshold, output_path)
        print(f"Running sink_dominance_threshold={threshold} -> {output_path}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
