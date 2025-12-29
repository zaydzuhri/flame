#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # ensure custom modeling is registered

DTYPE_MAP = {
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
}


def sanitize_model_name(model_name: str) -> str:
    safe = model_name.strip().replace("/", "_").replace(" ", "_")
    return safe or "model"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    return torch.device(device_name)


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype_name not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Choose from: {', '.join(DTYPE_MAP)}"
        )
    return DTYPE_MAP[dtype_name]


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
    trust_remote_code: bool,
    attn_impl: str | None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    config = AutoConfig.from_pretrained(
        model_name, local_files_only=local_files_only, trust_remote_code=trust_remote_code
    )
    config.output_attentions = False
    config.output_hidden_states = False
    config.return_dict_in_generate = True
    if attn_impl is not None:
        if not hasattr(config, "attn_impl"):
            raise ValueError(
                "Config does not define attn_impl; remove --attn-impl or use a "
                "compatible model config."
            )
        config.attn_impl = attn_impl

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        config=config,
    )
    model.to(device)
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.return_dict_in_generate = True
    return model, tokenizer


def extract_text(sample: dict) -> str | None:
    for key in ("text", "content"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in sample.values():
        if isinstance(value, str) and value.strip():
            return value
    return None


def iter_batches(
    dataset_iterable: Iterable[dict],
    batch_size: int,
    max_samples: int,
) -> Iterable[list[str]]:
    batch: list[str] = []
    samples_seen = 0
    for sample in dataset_iterable:
        if max_samples >= 0 and samples_seen >= max_samples:
            break
        text = extract_text(sample)
        if text is None:
            continue
        batch.append(text)
        samples_seen += 1
        if len(batch) >= batch_size:
            yield list(batch)
            batch.clear()
    if batch:
        yield list(batch)


def build_dataset_iterable(
    dataset_name: str,
    dataset_config: str,
    split: str,
    streaming: bool,
    n_samples: int,
) -> tuple[Iterable[dict], int | None]:
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=streaming,
    )
    if not streaming and n_samples >= 0:
        total_available = len(dataset)
        n_samples = min(n_samples, total_available)
    return dataset, n_samples


def tokenize_batch(
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    padding: str,
) -> dict[str, torch.Tensor]:
    padding_value = padding != "none"
    return tokenizer(
        texts,
        padding=("longest" if padding_value else False),
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def resolve_output_path(
    output: Path | None,
    output_dir: Path | None,
    model_name: str,
    suffix: str,
) -> Path | None:
    if output and output_dir:
        raise ValueError("Use --output or --output-dir, not both.")
    if output_dir is None:
        return output
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_model_name(model_name)}_{suffix}.jsonl"
    return output_dir / filename


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        default="wikitext",
        help="Hugging Face dataset name (default: wikitext).",
    )
    parser.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
        help="Dataset config name (default: wikitext-2-raw-v1).",
    )
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="Number of samples to process (-1 for all samples).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--padding",
        choices=["none", "longest"],
        default="none",
        help="Padding strategy (default: none).",
    )


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (cuda or cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bf16", "float32"],
        help="Model dtype (default: auto).",
    )
    parser.add_argument(
        "--attn-impl",
        default=None,
        help="Override config.attn_impl to force attention outputs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from Hugging Face.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid remote downloads in from_pretrained.",
    )


__all__ = [
    "DTYPE_MAP",
    "add_dataset_args",
    "add_runtime_args",
    "build_dataset_iterable",
    "extract_text",
    "iter_batches",
    "load_model_and_tokenizer",
    "resolve_device",
    "resolve_dtype",
    "resolve_output_path",
    "sanitize_model_name",
    "tokenize_batch",
    "write_results",
]
