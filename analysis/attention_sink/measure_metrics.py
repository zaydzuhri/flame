#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm
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


@dataclass
class MomentAccumulator:
    count: int = 0
    sum1: float = 0.0
    sum2: float = 0.0
    sum3: float = 0.0
    sum4: float = 0.0
    min_val: float | None = None
    max_val: float | None = None

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float64)
        if values.numel() == 0:
            return
        self.count += values.numel()
        self.sum1 += values.sum().item()
        self.sum2 += (values**2).sum().item()
        self.sum3 += (values**3).sum().item()
        self.sum4 += (values**4).sum().item()
        batch_min = values.min().item()
        batch_max = values.max().item()
        if self.min_val is None or batch_min < self.min_val:
            self.min_val = batch_min
        if self.max_val is None or batch_max > self.max_val:
            self.max_val = batch_max

    def kurtosis(self) -> float:
        if self.count == 0:
            return float("nan")
        mean = self.sum1 / self.count
        e2 = self.sum2 / self.count
        e3 = self.sum3 / self.count
        e4 = self.sum4 / self.count
        mu2 = e2 - mean**2
        if mu2 <= 0:
            return float("nan")
        mu4 = e4 - 4 * mean * e3 + 6 * mean**2 * e2 - 3 * mean**4
        return mu4 / (mu2**2)


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


def mean_first_token_attention(
    attention: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    first_token_attention = attention[:, :, :, 0]
    if attention_mask is None:
        return first_token_attention.mean(dim=-1)
    mask = attention_mask[:, None, :].to(dtype=first_token_attention.dtype)
    denom = mask.sum(dim=-1).clamp_min(1)
    return (first_token_attention * mask).sum(dim=-1) / denom


def count_lower_triangle_zeros(
    attention: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sparsity_eps: float,
) -> tuple[int, int, int]:
    batch_size, num_heads, seq_len, _ = attention.shape
    zeros = 0
    zeros_at_or_below_eps = 0
    total = 0
    for batch_idx in range(batch_size):
        length = seq_len
        if attention_mask is not None:
            length = int(attention_mask[batch_idx].sum().item())
        if length <= 0:
            continue
        total += (length * (length + 1) // 2) * num_heads
        for head_idx in range(num_heads):
            head_attn = attention[batch_idx, head_idx, :length, :length]
            zeros += torch.tril(head_attn == 0).sum().item()
            if sparsity_eps > 0:
                zeros_at_or_below_eps += torch.tril(
                    head_attn.abs() <= sparsity_eps
                ).sum().item()
    return zeros, zeros_at_or_below_eps, total


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

def resolve_output_path(
    output: Path | None,
    output_dir: Path | None,
    model_name: str,
) -> Path | None:
    if output and output_dir:
        raise ValueError("Use --output or --output-dir, not both.")
    if output_dir is None:
        return output
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_model_name(model_name)}_metrics.jsonl"
    return output_dir / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure sink rate, hidden activation stats, and lower-triangle sparsity."
        )
    )
    parser.add_argument("--model", required=True, help="HF model name or local path.")
    parser.add_argument(
        "--dataset",
        default="Phando/SlimPajama-val-sampled",
        help="Hugging Face dataset name (default: wikitext).",
    )
    parser.add_argument(
        "--dataset-config",
        default="default",
        help="Dataset config name (default: default).",
    )
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to process (-1 for all samples).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for inference."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--sink-eps",
        type=float,
        nargs="+",
        default=[0.2, 0.3],
        help="Sink rate thresholds (space-separated).",
    )
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
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset mode.",
    )
    parser.add_argument(
        "--sparsity-eps",
        type=float,
        default=1e-2,
        help="Treat values with abs <= eps as zeros for sparsity (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON/JSONL output path for metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write <model>_metrics.jsonl (mutually exclusive with --output).",
    )
    return parser.parse_args()


def compute_metrics(
    model: AutoModelForCausalLM,
    batch_iter: Iterable[dict[str, torch.Tensor]],
    sink_eps: list[float],
    sparsity_eps: float,
    device: torch.device,
    total_batches: int | None = None,
    progress_desc: str = "Batches",
) -> dict:
    sink_hits = {eps: 0 for eps in sink_eps}
    sink_total = 0
    sparsity_zeros = 0
    sparsity_eps_zeros = 0
    sparsity_total = 0
    moments = MomentAccumulator()
    samples_processed = 0

    for encodings in tqdm(
        batch_iter, total=total_batches, desc=progress_desc, unit="batch"
    ):
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        if outputs.attentions is None or not any(
            attn is not None for attn in outputs.attentions
        ):
            raise RuntimeError(
                "Model returned no attentions. Use --attn-impl naive_attn (or another "
                "naive_* impl), or gpt_oss_naive_sink for GPT-OSS sink checkpoints."
            )
        if outputs.hidden_states is None:
            raise RuntimeError("Model returned no hidden states.")

        for layer_attn in outputs.attentions:
            if layer_attn is None:
                raise RuntimeError("Missing attention map for a layer.")
            mean_attention = mean_first_token_attention(layer_attn, attention_mask)
            sink_total += mean_attention.numel()
            for eps in sink_eps:
                sink_hits[eps] += (mean_attention > eps).sum().item()
            layer_zeros, layer_eps_zeros, layer_total = count_lower_triangle_zeros(
                layer_attn, attention_mask, sparsity_eps
            )
            sparsity_zeros += layer_zeros
            sparsity_eps_zeros += layer_eps_zeros
            sparsity_total += layer_total

        for states in outputs.hidden_states:
            moments.update(states)

        samples_processed += input_ids.shape[0]
        del outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sink_rates = {
        str(eps): (sink_hits[eps] / sink_total if sink_total else 0.0)
        for eps in sink_eps
    }
    sparsity = (sparsity_zeros / sparsity_total) if sparsity_total else float("nan")
    sparsity_eps_rate = None
    if sparsity_eps > 0:
        sparsity_eps_rate = (
            (sparsity_eps_zeros / sparsity_total)
            if sparsity_total
            else float("nan")
        )

    return {
        "samples_processed": samples_processed,
        "sink_rate": sink_rates,
        "hidden_activation": {
            "kurtosis": moments.kurtosis(),
            "min": moments.min_val,
            "max": moments.max_val,
        },
        "sparsity": {
            "element_count": sparsity_total,
            "lower_triangle_exact_zero": sparsity,
            "lower_triangle_at_or_below_eps": sparsity_eps_rate,
            "sparsity_eps": sparsity_eps,
            "zero_count": sparsity_zeros,
        },
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        attn_impl=args.attn_impl,
    )

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=args.streaming,
    )
    if not args.streaming and args.n_samples >= 0:
        total_available = len(dataset)
        n_samples = min(args.n_samples, total_available)
        dataset_iterable = dataset.select(range(n_samples))
    else:
        n_samples = args.n_samples
        dataset_iterable = dataset

    total_samples_for_progress = n_samples if n_samples >= 0 else None
    total_batches = None
    if total_samples_for_progress is not None:
        total_batches = math.ceil(total_samples_for_progress / args.batch_size)

    def batch_iter() -> Iterable[dict[str, torch.Tensor]]:
        for batch_texts in iter_batches(
            dataset_iterable, args.batch_size, max_samples=n_samples
        ):
            if not batch_texts:
                continue
            yield tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )

    metrics = compute_metrics(
        model=model,
        batch_iter=batch_iter(),
        sink_eps=args.sink_eps,
        sparsity_eps=args.sparsity_eps,
        device=device,
        total_batches=total_batches,
        progress_desc="Batches",
    )

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "attn_impl": args.attn_impl,
        **metrics,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    output_path = resolve_output_path(args.output, args.output_dir, args.model)
    if output_path is not None:
        write_results(output_path, result)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
