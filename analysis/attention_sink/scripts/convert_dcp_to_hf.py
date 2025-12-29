#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fla  # noqa: F401
from torchtitan.tools.logging import init_logger, logger


def resolve_checkpoint_dir(
    checkpoint_root: Path | None,
    checkpoint_path: Path | None,
    step: int | None,
) -> Path:
    if checkpoint_path is not None:
        return checkpoint_path
    if checkpoint_root is None or step is None:
        raise ValueError("Provide --checkpoint-path or --checkpoint-root with --step.")
    return checkpoint_root / "checkpoint" / f"step-{step}"


@torch.inference_mode()
def save_pretrained(
    checkpoint_dir: Path,
    output_dir: Path,
    config: str,
    tokenizer: str,
    local_files_only: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading the config from {config}")
    config_obj = AutoConfig.from_pretrained(
        config, trust_remote_code=True, local_files_only=local_files_only
    )
    logger.info(f"Saving the config to {output_dir}")
    config_obj.save_pretrained(output_dir)

    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer_obj = AutoTokenizer.from_pretrained(
        tokenizer, trust_remote_code=True, local_files_only=local_files_only
    )
    logger.info(f"Saving the tokenizer to {output_dir}")
    tokenizer_obj.save_pretrained(output_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(str(checkpoint_dir), str(checkpoint_path))

        logger.info(f"Initializing the model from config\n{config_obj}")
        model = AutoModelForCausalLM.from_config(config_obj)
        logger.info("Loading state dict from the checkpoint")

        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model"])

        logger.info(f"Saving the model to {output_dir}")
        model.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Convert DCP format model weights to Hugging Face format."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Training run directory containing checkpoint/step-<n>.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Explicit path to checkpoint/step-<n>.",
    )
    parser.add_argument("--step", type=int, help="Step number under checkpoint root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model repo/path providing config and tokenizer.",
    )
    parser.add_argument("--config", type=str, help="Config repo/path.")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer repo/path.")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid remote downloads in from_pretrained.",
    )
    return parser.parse_args()


def main() -> None:
    init_logger()
    args = parse_args()

    if args.checkpoint_root is not None and args.checkpoint_path is not None:
        raise ValueError("Provide only one of --checkpoint-root or --checkpoint-path.")
    if args.base_model is not None and (args.config or args.tokenizer):
        raise ValueError("Use either --base-model or --config/--tokenizer, not both.")
    if args.base_model is None and (args.config is None or args.tokenizer is None):
        raise ValueError(
            "Provide --base-model or both --config and --tokenizer."
        )

    checkpoint_dir = resolve_checkpoint_dir(
        args.checkpoint_root, args.checkpoint_path, args.step
    )
    config_source = args.base_model or args.config
    tokenizer_source = args.base_model or args.tokenizer
    save_pretrained(
        checkpoint_dir=checkpoint_dir,
        output_dir=args.output_dir,
        config=config_source,
        tokenizer=tokenizer_source,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
