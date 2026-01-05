#!/usr/bin/env python3
"""Measure per-head mean sink attention across checkpoints.

For softmax models, a head is considered "sink-dead" if its mean attention
to the first token exceeds a threshold (e.g., 0.3 or 0.95).

This complements head_dead_analysis.py which measures off-rate for softpick models.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm

try:
    from .utils import (
        extract_text,
        load_model_and_tokenizer,
        resolve_device,
        resolve_dtype,
        resolve_output_path,
        tokenize_batch,
        write_results,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from utils import (  # type: ignore
        extract_text,
        load_model_and_tokenizer,
        resolve_device,
        resolve_dtype,
        resolve_output_path,
        tokenize_batch,
        write_results,
    )


@dataclass(frozen=True)
class LayerSpec:
    layer_idx: int
    num_heads: int
    name: str
    module: torch.nn.Module


def parse_steps(args: argparse.Namespace) -> list[int]:
    if args.steps:
        steps = [int(step.strip()) for step in args.steps.split(",") if step.strip()]
        if not steps:
            raise ValueError("Provide at least one step via --steps.")
        return sorted(set(steps))
    if args.step_start is None or args.step_end is None:
        raise ValueError("Use --steps or --step-start/--step-end to specify checkpoints.")
    if args.step_stride <= 0:
        raise ValueError("--step-stride must be positive.")
    return list(range(args.step_start, args.step_end + 1, args.step_stride))


def build_dataset_iterable(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    streaming: bool,
    shuffle: bool,
    shuffle_seed: int,
    shuffle_buffer: int,
) -> Iterable[dict]:
    if dataset_config:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
        )
    if shuffle:
        if streaming:
            dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
        else:
            dataset = dataset.shuffle(seed=shuffle_seed)
    return dataset


def iter_token_batches(
    dataset_iterable: Iterable[dict],
    tokenizer,
    batch_size: int,
    max_length: int,
    padding: str,
    max_tokens: int,
) -> Iterable[tuple[dict[str, torch.Tensor], int]]:
    batch: list[str] = []
    tokens_seen = 0
    for sample in dataset_iterable:
        text = extract_text(sample)
        if text is None:
            continue
        batch.append(text)
        if len(batch) < batch_size:
            continue
        encodings = tokenize_batch(tokenizer, batch, max_length=max_length, padding=padding)
        batch_tokens = int(encodings["attention_mask"].sum().item())
        if max_tokens >= 0 and tokens_seen >= max_tokens:
            break
        yield encodings, batch_tokens
        tokens_seen += batch_tokens
        batch.clear()
    if batch and (max_tokens < 0 or tokens_seen < max_tokens):
        encodings = tokenize_batch(tokenizer, batch, max_length=max_length, padding=padding)
        batch_tokens = int(encodings["attention_mask"].sum().item())
        yield encodings, batch_tokens


class HeadSinkRateAccumulator:
    """Accumulates per-head mean sink attention across batches."""

    def __init__(self, num_heads_by_layer: Sequence[int]) -> None:
        self.sink_sums = [
            torch.zeros(num_heads, dtype=torch.float64) for num_heads in num_heads_by_layer
        ]
        self.token_counts = [
            torch.zeros(num_heads, dtype=torch.float64) for num_heads in num_heads_by_layer
        ]
        self.attention_mask: torch.Tensor | None = None

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self.attention_mask = attention_mask

    def update_from_attention(
        self, layer_idx: int, attention: torch.Tensor
    ) -> None:
        """Update sink sums from attention tensor.

        Args:
            layer_idx: Index of the layer.
            attention: Attention tensor of shape (batch, heads, seq, seq).
        """
        # attention shape: (batch, heads, seq_q, seq_k)
        # sink attention: attention to first key position
        sink_attention = attention[:, :, :, 0]  # (batch, heads, seq_q)

        if self.attention_mask is not None:
            # Mask out padding tokens
            mask = self.attention_mask[:, None, :].to(dtype=sink_attention.dtype)
            # Sum sink attention for valid tokens only
            sink_sum = (sink_attention * mask).sum(dim=(0, 2))  # (heads,)
            token_count = mask.sum(dim=(0, 2))  # (heads,) - same count per head
        else:
            sink_sum = sink_attention.sum(dim=(0, 2))  # (heads,)
            batch_size, _, seq_len = sink_attention.shape
            token_count = torch.full_like(sink_sum, batch_size * seq_len)

        self.sink_sums[layer_idx] += sink_sum.to(dtype=torch.float64).cpu()
        self.token_counts[layer_idx] += token_count.to(dtype=torch.float64).cpu()

    def mean_sink_rates(self) -> list[list[float]]:
        """Return per-head mean sink attention rates."""
        rates: list[list[float]] = []
        for sink_sum, token_count in zip(self.sink_sums, self.token_counts):
            if token_count.numel() == 0 or token_count.max().item() == 0:
                rates.append([float("nan") for _ in range(token_count.numel())])
                continue
            rates.append((sink_sum / token_count).tolist())
        return rates


def collect_attention_modules(model: torch.nn.Module) -> list[LayerSpec]:
    """Collect Attention modules from the model."""
    from fla.layers.attn import Attention

    modules: list[LayerSpec] = []
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            layer_idx = module.layer_idx if module.layer_idx is not None else -1
            modules.append(
                LayerSpec(
                    layer_idx=layer_idx,
                    num_heads=module.num_heads,
                    name=name,
                    module=module,
                )
            )
    if not modules:
        raise RuntimeError("No Attention modules found.")
    if all(layer.layer_idx >= 0 for layer in modules):
        modules.sort(key=lambda layer: layer.layer_idx)
    else:
        modules = [
            LayerSpec(idx, layer.num_heads, layer.name, layer.module)
            for idx, layer in enumerate(modules)
        ]
    return modules


def compute_sink_death_summary(
    steps: Sequence[int],
    sink_rates_by_step: Sequence[list[list[float]]],
    sink_eps_list: Sequence[float],
) -> dict:
    """Compute death summary for each sink threshold."""
    if not sink_rates_by_step:
        return {}

    num_layers = len(sink_rates_by_step[0])
    num_heads_by_layer = [len(layer) for layer in sink_rates_by_step[0]]
    total_heads = sum(num_heads_by_layer)

    result: dict = {
        "dead_head_count": {},
        "dead_heads_by_step": {},
        "lifetime_summary": {},
    }

    for eps in sink_eps_list:
        eps_key = str(eps)
        dead_counts: dict[str, int] = {}
        dead_heads_by_step: dict[str, list[list[int]]] = {}

        # Track death across steps for lifetime analysis
        death_step = [
            [None for _ in range(num_heads_by_layer[layer_idx])]
            for layer_idx in range(num_layers)
        ]
        revived = [
            [False for _ in range(num_heads_by_layer[layer_idx])]
            for layer_idx in range(num_layers)
        ]

        for step_idx, (step, step_rates) in enumerate(zip(steps, sink_rates_by_step)):
            step_key = str(step)
            dead_count = 0
            dead_heads: list[list[int]] = []

            for layer_idx, layer_rates in enumerate(step_rates):
                for head_idx, rate in enumerate(layer_rates):
                    is_dead = rate > eps
                    if is_dead:
                        dead_count += 1
                        dead_heads.append([layer_idx, head_idx])

                    # Track lifetime
                    was_dead = death_step[layer_idx][head_idx] is not None
                    if is_dead and death_step[layer_idx][head_idx] is None:
                        death_step[layer_idx][head_idx] = step
                    if was_dead and not is_dead:
                        revived[layer_idx][head_idx] = True

            dead_counts[step_key] = dead_count
            dead_heads_by_step[step_key] = dead_heads

        # Compute lifetime summary
        never_dead = 0
        dead_never_revive = 0
        dead_revive = 0
        dead_never_revive_heads: list[list[int]] = []
        dead_revive_heads: list[list[int]] = []

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads_by_layer[layer_idx]):
                if death_step[layer_idx][head_idx] is None:
                    never_dead += 1
                    continue
                if revived[layer_idx][head_idx]:
                    dead_revive += 1
                    dead_revive_heads.append([layer_idx, head_idx])
                else:
                    dead_never_revive += 1
                    dead_never_revive_heads.append([layer_idx, head_idx])

        result["dead_head_count"][eps_key] = dead_counts
        result["dead_heads_by_step"][eps_key] = dead_heads_by_step
        result["lifetime_summary"][eps_key] = {
            "total_heads": total_heads,
            "never_dead": never_dead,
            "dead_never_revive": dead_never_revive,
            "dead_revive": dead_revive,
            "dead_never_revive_heads": dead_never_revive_heads,
            "dead_revive_heads": dead_revive_heads,
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-head mean sink attention across checkpoints."
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
        default="longest",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5_000_000,
        help="Stop after processing this many tokens (-1 for all).",
    )
    parser.add_argument(
        "--sink-eps",
        type=float,
        nargs="+",
        default=[0.3, 0.95],
        help="Sink attention thresholds for 'dead' classification (space-separated).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bf16", "float32"],
    )
    parser.add_argument(
        "--attn-impl",
        default=None,
        help="Attention implementation (must be naive_* for attention outputs).",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.padding == "none" and args.batch_size != 1:
        raise ValueError("Use --batch-size 1 when --padding none to avoid padding.")
    if args.max_tokens == 0:
        raise ValueError("--max-tokens must be non-zero or -1 for all tokens.")
    if args.attn_impl is None:
        raise ValueError(
            "--attn-impl is required (use naive_attn or another naive_* impl "
            "that returns attention weights)."
        )
    if not args.attn_impl.startswith("naive"):
        print(
            f"Warning: --attn-impl '{args.attn_impl}' may not return attention weights. "
            "Use naive_attn or similar."
        )
    for eps in args.sink_eps:
        if eps <= 0 or eps > 1:
            raise ValueError(f"--sink-eps values must be in (0, 1], got {eps}")


def run_checkpoint(
    model_path: str,
    model: torch.nn.Module,
    layers: list[LayerSpec],
    tokenizer,
    dataset_iterable: Iterable[dict],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[list[float]], dict]:
    """Run analysis on a single checkpoint."""
    num_heads_by_layer = [layer.num_heads for layer in layers]
    accumulator = HeadSinkRateAccumulator(num_heads_by_layer)

    tokens_processed = 0
    samples_processed = 0

    for encodings, batch_tokens in tqdm(
        iter_token_batches(
            dataset_iterable,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            padding=args.padding,
            max_tokens=args.max_tokens,
        ),
        desc=f"Tokens {model_path}",
        unit="batch",
    ):
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        accumulator.set_attention_mask(attention_mask)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        if outputs.attentions is None:
            raise RuntimeError(
                "Model returned no attentions. Use --attn-impl naive_attn or similar."
            )

        for layer_idx, layer_attn in enumerate(outputs.attentions):
            if layer_attn is None:
                continue
            accumulator.update_from_attention(layer_idx, layer_attn)

        accumulator.set_attention_mask(None)
        tokens_processed += batch_tokens
        samples_processed += input_ids.shape[0]

        del outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.max_tokens >= 0 and tokens_processed >= args.max_tokens:
            break

    metadata = {
        "samples_processed": samples_processed,
        "tokens_processed": tokens_processed,
    }
    return accumulator.mean_sink_rates(), metadata


def main() -> None:
    args = parse_args()
    validate_args(args)

    steps = parse_steps(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    sink_rates_by_step: list[list[list[float]]] = []
    step_metadata: dict[str, dict] = {}
    layer_ids: list[int] | None = None
    layer_names: list[str] | None = None

    for step in steps:
        model_path = args.model_template.format(step=step)
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            device=device,
            dtype=dtype,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            attn_impl=args.attn_impl,
        )
        layers = collect_attention_modules(model)
        if layer_ids is None:
            layer_ids = [layer.layer_idx for layer in layers]
            layer_names = [layer.name for layer in layers]

        dataset_iterable = build_dataset_iterable(
            args.dataset,
            args.dataset_config or None,
            args.split,
            args.streaming,
            args.shuffle,
            args.shuffle_seed,
            args.shuffle_buffer,
        )

        sink_rates, metadata = run_checkpoint(
            model_path=model_path,
            model=model,
            layers=layers,
            tokenizer=tokenizer,
            dataset_iterable=dataset_iterable,
            device=device,
            args=args,
        )
        sink_rates_by_step.append(sink_rates)
        step_metadata[str(step)] = metadata

        # Free memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = compute_sink_death_summary(
        steps,
        sink_rates_by_step,
        args.sink_eps,
    )

    results = {
        "analysis_type": "head_sink",
        "model_template": args.model_template,
        "steps": steps,
        "layer_ids": layer_ids,
        "layer_names": layer_names,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config or None,
        "split": args.split,
        "streaming": args.streaming,
        "shuffle": args.shuffle,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_buffer": args.shuffle_buffer,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "padding": args.padding,
        "max_tokens": args.max_tokens,
        "sink_eps": args.sink_eps,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "attn_impl": args.attn_impl,
        "step_metadata": step_metadata,
        "mean_sink_rates": {
            str(step): rates for step, rates in zip(steps, sink_rates_by_step)
        },
        **summary,
    }

    print(json.dumps(results, indent=2, sort_keys=True))

    model_label = args.model_template.replace("{step}", "steps")
    output_path = resolve_output_path(
        args.output, args.output_dir, model_label, suffix="head_sink"
    )
    if output_path is not None:
        write_results(output_path, results)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
