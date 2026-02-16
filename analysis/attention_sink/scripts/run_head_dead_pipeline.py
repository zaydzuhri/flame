#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def parse_steps(args: argparse.Namespace) -> list[int]:
    if args.steps:
        steps = [int(step.strip()) for step in args.steps.split(",") if step.strip()]
        if not steps:
            raise ValueError("Provide at least one step via --steps.")
        return sorted(set(steps))
    if args.step_start is None or args.step_end is None:
        raise ValueError("Use --steps or --step-start/--step-end.")
    if args.step_stride <= 0:
        raise ValueError("--step-stride must be positive.")
    return list(range(args.step_start, args.step_end + 1, args.step_stride))


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(f"$ {shlex.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def has_hf_weights(output_dir: Path) -> bool:
    weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any((output_dir / name).exists() for name in weight_files)


def is_hf_model_dir_ready(output_dir: Path) -> bool:
    required = (
        output_dir / "config.json",
        output_dir / "tokenizer_config.json",
    )
    return all(path.exists() for path in required) and has_hf_weights(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download DCP checkpoints, convert each step to Hugging Face format, "
            "then run head_dead_analysis in one command."
        )
    )

    parser.add_argument("--repo-id", required=True, help="HF repository with DCP checkpoints.")
    parser.add_argument(
        "--checkpoint-subdir",
        default=None,
        help="Optional repo subdirectory containing step folders.",
    )
    parser.add_argument(
        "--steps",
        default="",
        help="Comma-separated checkpoint steps (e.g. 10000,20000,30000).",
    )
    parser.add_argument("--step-start", type=int, default=None)
    parser.add_argument("--step-end", type=int, default=None)
    parser.add_argument("--step-stride", type=int, default=10000)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Local run directory where checkpoint/step-<n> folders are stored.",
    )
    parser.add_argument(
        "--hf-model-template",
        type=str,
        required=True,
        help="Output HF model template path containing {step}.",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        required=True,
        help="Output JSON path for head_dead_analysis.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help=(
            "Extra allow pattern for checkpoint download. "
            "If omitted, defaults for config/tokenizer files are applied."
        ),
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision.")
    parser.add_argument("--token", default=None, help="Optional HF token override.")

    parser.add_argument(
        "--config",
        required=True,
        help="Config source passed to convert_dcp_to_hf.py --config.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer source passed to convert_dcp_to_hf.py --tokenizer.",
    )
    parser.add_argument(
        "--convert-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local files only in conversion (default: true).",
    )
    parser.add_argument(
        "--analysis-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local files only in head_dead_analysis (default: true).",
    )

    parser.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--padding", choices=["none", "longest"], default="longest")
    parser.add_argument("--max-tokens", type=int, default=5_000_000)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--dead-threshold", type=float, default=0.95)
    parser.add_argument("--mostly-dead-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bf16", "float32"],
    )
    parser.add_argument("--attn-impl", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Always run conversion even if output HF model directory already looks complete.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = parse_steps(args)
    if "{step}" not in args.hf_model_template:
        raise ValueError("--hf-model-template must include '{step}'.")
    if args.padding == "none" and args.batch_size != 1:
        raise ValueError("Use --batch-size 1 when --padding none.")

    repo_root = Path(__file__).resolve().parents[3]
    python_bin = sys.executable
    download_script = repo_root / "analysis/attention_sink/scripts/download_hf_checkpoint.py"
    convert_script = repo_root / "analysis/attention_sink/scripts/convert_dcp_to_hf.py"
    analyze_script = repo_root / "analysis/ablations/head_dead_analysis.py"

    allow_patterns = list(args.allow_pattern) if args.allow_pattern else list(DEFAULT_ALLOW_PATTERNS)
    step_arg = ",".join(str(step) for step in steps)

    for step in steps:
        if not args.skip_download:
            download_cmd = [
                python_bin,
                str(download_script),
                "--repo-id",
                args.repo_id,
                "--step",
                str(step),
                "--output-root",
                str(args.run_root),
            ]
            if args.checkpoint_subdir:
                download_cmd.extend(["--checkpoint-subdir", args.checkpoint_subdir])
            for pattern in allow_patterns:
                download_cmd.extend(["--allow-pattern", pattern])
            if args.revision:
                download_cmd.extend(["--revision", args.revision])
            if args.token:
                download_cmd.extend(["--token", args.token])
            run_command(download_cmd, args.dry_run)

        if not args.skip_convert:
            output_dir = args.hf_model_template.format(step=step)
            output_dir_path = Path(output_dir)
            if not args.force_convert and is_hf_model_dir_ready(output_dir_path):
                print(f"[skip-convert] Step {step}: found existing HF model at {output_dir_path}")
                continue
            convert_cmd = [
                python_bin,
                str(convert_script),
                "--checkpoint-root",
                str(args.run_root),
                "--step",
                str(step),
                "--config",
                args.config,
                "--tokenizer",
                args.tokenizer,
                "--output-dir",
                output_dir,
            ]
            if not args.convert_local_files_only:
                convert_cmd.append("--no-local-files-only")
            run_command(convert_cmd, args.dry_run)

    if args.skip_analysis:
        return

    analyze_cmd = [
        python_bin,
        str(analyze_script),
        "--model-template",
        args.hf_model_template,
        "--steps",
        step_arg,
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--shuffle-seed",
        str(args.shuffle_seed),
        "--shuffle-buffer",
        str(args.shuffle_buffer),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--padding",
        args.padding,
        "--max-tokens",
        str(args.max_tokens),
        "--eps",
        str(args.eps),
        "--dead-threshold",
        str(args.dead_threshold),
        "--mostly-dead-threshold",
        str(args.mostly_dead_threshold),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--output",
        str(args.analysis_output),
    ]
    if args.dataset_config:
        analyze_cmd.extend(["--dataset-config", args.dataset_config])
    if args.streaming:
        analyze_cmd.append("--streaming")
    if args.shuffle:
        analyze_cmd.append("--shuffle")
    if args.attn_impl:
        analyze_cmd.extend(["--attn-impl", args.attn_impl])
    if args.trust_remote_code:
        analyze_cmd.append("--trust-remote-code")
    if not args.analysis_local_files_only:
        analyze_cmd.append("--no-local-files-only")

    run_command(analyze_cmd, args.dry_run)


if __name__ == "__main__":
    main()
