#!/usr/bin/env python3
"""Measure per-head off-rate with sink token masked from attention.

This script answers: "Would this head be dead if it couldn't dump attention
on the sink token?" by masking the first key position before softmax and
then measuring head off-rates.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from datasets import load_dataset
from einops import rearrange
from tqdm import tqdm

from fla.ops.attn.naive_abs_softmax import abs_softmax_1, abs_softmax_2
from fla.ops.attn.naive_relusoftpick import relu_softpick_1, relu_softpick_2
from fla.ops.attn.naive_softmax_plus_one import softmax_plus_one
from fla.ops.attn.naive_softpick import softpick
import fla.layers.attn as attn_module

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


# Attention kernel functions (from attention_knockout.py)
ATTN_KERNELS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "naive_attn": lambda scores: torch.softmax(scores, dim=-1),
    "naive_scaled_attn": lambda scores: torch.softmax(scores, dim=-1),
    "naive_rectified_attn": lambda scores: torch.softmax(
        scores.masked_fill(scores < 0, float("-inf")),
        dim=-1,
    ),
    "naive_softpick_attn": lambda scores: softpick(scores, dim=-1),
    "naive_scaled_softpick_attn": lambda scores: softpick(scores, dim=-1),
    "naive_relu_softpick_1_attn": lambda scores: relu_softpick_1(scores, dim=-1),
    "naive_relu_softpick_2_attn": lambda scores: relu_softpick_2(scores, dim=-1),
    "naive_abs_softmax_1_attn": lambda scores: abs_softmax_1(scores, dim=-1),
    "naive_abs_softmax_2_attn": lambda scores: abs_softmax_2(scores, dim=-1),
    "naive_softmax_plus_one_attn": lambda scores: softmax_plus_one(scores, dim=-1),
}

PATCH_TARGETS = {
    "naive_attn",
    "naive_rectified_attn",
    "naive_softpick_attn",
    "naive_relu_softpick_1_attn",
    "naive_relu_softpick_2_attn",
    "naive_abs_softmax_1_attn",
    "naive_abs_softmax_2_attn",
    "naive_softmax_plus_one_attn",
}


class MaskedAttentionPatcher:
    """Context manager that patches attention functions to mask a specific key index."""

    def __init__(self, mask_key_index: int, skip_first_query_on_mask: bool):
        self.mask_key_index = mask_key_index
        self.skip_first_query_on_mask = skip_first_query_on_mask
        self.originals: dict[str, Callable] = {}

    def __enter__(self):
        for name in PATCH_TARGETS:
            if hasattr(attn_module, name):
                self.originals[name] = getattr(attn_module, name)
                setattr(attn_module, name, self._build_masked_function(name))
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, func in self.originals.items():
            setattr(attn_module, name, func)
        self.originals.clear()
        return False

    def _build_masked_function(self, name: str) -> Callable:
        kernel = ATTN_KERNELS.get(name)
        if kernel is None:
            raise ValueError(f"Masked attention does not support {name}.")

        def masked_attn(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            scale: float | None = None,
            cu_seqlens: torch.LongTensor | None = None,
            head_first: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if cu_seqlens is not None:
                raise ValueError("Masked attention does not support cu_seqlens.")
            if not head_first:
                q, k, v = map(lambda x: rearrange(x, "b t h d -> b h t d"), (q, k, v))
            head_dim = q.shape[-1]
            scale_value = scale if scale is not None else head_dim ** -0.5
            q_len = q.shape[-2]
            k_len = k.shape[-2]
            if not (-k_len <= self.mask_key_index < k_len):
                raise ValueError(
                    f"mask_key_index {self.mask_key_index} is out of range for k_len={k_len}."
                )
            scores = torch.matmul(q, k.transpose(2, 3)) * scale_value
            causal_mask = torch.tril(torch.ones(k_len, k_len, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(
                causal_mask[k_len - q_len:k_len, :k_len] == 0, float("-inf")
            )
            if (
                self.skip_first_query_on_mask
                and self.mask_key_index == 0
                and q_len > 0
            ):
                scores[..., 1:, self.mask_key_index] = float("-inf")
            else:
                scores[..., self.mask_key_index] = float("-inf")
            attn = kernel(scores.float()).to(q.dtype)
            if not torch.isfinite(attn).all():
                raise RuntimeError(
                    "Masked attention produced non-finite values; adjust masking or "
                    "skip the first prediction token."
                )
            o = torch.matmul(attn, v)
            if not head_first:
                o = rearrange(o, "b h t d -> b t h d")
            return o, attn

        return masked_attn


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


def compute_off_counts(
    head_output: torch.Tensor,
    attention_mask: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, int]:
    per_token_max = head_output.detach().abs().amax(dim=-1)
    off = per_token_max < eps
    if attention_mask is not None:
        mask = attention_mask.to(dtype=torch.bool)
        off = off & mask[:, :, None]
        valid_tokens = int(mask.sum().item())
    else:
        valid_tokens = head_output.shape[0] * head_output.shape[1]
    off_counts = off.sum(dim=(0, 1)).to(dtype=torch.float64)
    return off_counts, valid_tokens


class HeadOffRateAccumulator:
    def __init__(self, eps: float, num_heads_by_layer: Sequence[int]) -> None:
        self.eps = float(eps)
        self.off_counts = [
            torch.zeros(num_heads, dtype=torch.float64) for num_heads in num_heads_by_layer
        ]
        self.total_counts = [
            torch.zeros(num_heads, dtype=torch.float64) for num_heads in num_heads_by_layer
        ]
        self.attention_mask: torch.Tensor | None = None

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self.attention_mask = attention_mask

    def update_from_o_proj(self, layer_idx: int, num_heads: int, flat_head_output: torch.Tensor) -> None:
        batch_size, seq_len, hidden = flat_head_output.shape
        if hidden % num_heads != 0:
            raise ValueError(
                f"Layer {layer_idx} head output hidden dim {hidden} not divisible by heads {num_heads}."
            )
        head_dim = hidden // num_heads
        head_output = flat_head_output.view(batch_size, seq_len, num_heads, head_dim)
        off_counts, valid_tokens = compute_off_counts(head_output, self.attention_mask, self.eps)
        self.off_counts[layer_idx] += off_counts.cpu()
        if valid_tokens > 0:
            self.total_counts[layer_idx] += float(valid_tokens)

    def off_rates(self) -> list[list[float]]:
        rates: list[list[float]] = []
        for off, total in zip(self.off_counts, self.total_counts):
            if total.numel() == 0 or total.max().item() == 0:
                rates.append([float("nan") for _ in range(total.numel())])
                continue
            rates.append((off / total).tolist())
        return rates


def collect_attention_modules(model: torch.nn.Module) -> list[LayerSpec]:
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
        raise RuntimeError("No Attention modules found to hook.")
    if all(layer.layer_idx >= 0 for layer in modules):
        modules.sort(key=lambda layer: layer.layer_idx)
    else:
        modules = [
            LayerSpec(idx, layer.num_heads, layer.name, layer.module)
            for idx, layer in enumerate(modules)
        ]
    return modules


def register_o_proj_hooks(
    layers: list[LayerSpec],
    accumulator: HeadOffRateAccumulator,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for idx, layer in enumerate(layers):
        def hook(module, inputs, output, layer_idx=idx, num_heads=layer.num_heads):
            if not inputs:
                return
            accumulator.update_from_o_proj(layer_idx, num_heads, inputs[0])

        handle = layer.module.o_proj.register_forward_hook(hook)
        handles.append(handle)
    return handles


def compute_death_summary(
    steps: Sequence[int],
    off_rates_by_step: Sequence[list[list[float]]],
    dead_threshold: float,
    mostly_dead_threshold: float,
) -> dict:
    if not off_rates_by_step:
        return {}
    num_layers = len(off_rates_by_step[0])
    num_heads_by_layer = [len(layer) for layer in off_rates_by_step[0]]
    dead_by_step: list[list[list[bool]]] = []
    mostly_dead_by_step: list[list[list[bool]]] = []
    dead_counts: list[int] = []
    mostly_dead_counts: list[int] = []
    dead_heads_by_step: dict[str, list[list[int]]] = {}
    mostly_dead_heads_by_step: dict[str, list[list[int]]] = {}

    for step, step_rates in zip(steps, off_rates_by_step):
        dead_flags: list[list[bool]] = []
        mostly_flags: list[list[bool]] = []
        step_dead = 0
        step_mostly = 0
        dead_heads: list[list[int]] = []
        mostly_heads: list[list[int]] = []
        for layer_rates in step_rates:
            layer_dead = [rate > dead_threshold for rate in layer_rates]
            layer_mostly = [rate > mostly_dead_threshold for rate in layer_rates]
            step_dead += sum(layer_dead)
            step_mostly += sum(layer_mostly)
            dead_flags.append(layer_dead)
            mostly_flags.append(layer_mostly)
        for layer_idx, layer_dead in enumerate(dead_flags):
            for head_idx, is_dead in enumerate(layer_dead):
                if is_dead:
                    dead_heads.append([layer_idx, head_idx])
        for layer_idx, layer_mostly in enumerate(mostly_flags):
            for head_idx, is_mostly in enumerate(layer_mostly):
                if is_mostly:
                    mostly_heads.append([layer_idx, head_idx])
        dead_by_step.append(dead_flags)
        mostly_dead_by_step.append(mostly_flags)
        dead_counts.append(step_dead)
        mostly_dead_counts.append(step_mostly)
        dead_heads_by_step[str(step)] = dead_heads
        mostly_dead_heads_by_step[str(step)] = mostly_heads

    death_step = [
        [None for _ in range(num_heads_by_layer[layer_idx])] for layer_idx in range(num_layers)
    ]
    revived = [
        [False for _ in range(num_heads_by_layer[layer_idx])] for layer_idx in range(num_layers)
    ]
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads_by_layer[layer_idx]):
            was_dead = False
            for step, dead_flags in zip(steps, dead_by_step):
                is_dead = dead_flags[layer_idx][head_idx]
                if is_dead and death_step[layer_idx][head_idx] is None:
                    death_step[layer_idx][head_idx] = step
                if was_dead and not is_dead:
                    revived[layer_idx][head_idx] = True
                was_dead = is_dead

    total_heads = sum(num_heads_by_layer)
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

    summary = {
        "dead_head_count": {str(step): count for step, count in zip(steps, dead_counts)},
        "mostly_dead_head_count": {
            str(step): count for step, count in zip(steps, mostly_dead_counts)
        },
        "dead_heads_by_step": dead_heads_by_step,
        "mostly_dead_heads_by_step": mostly_dead_heads_by_step,
        "death_step": death_step,
        "revived": revived,
        "dead_never_revive_heads": dead_never_revive_heads,
        "dead_revive_heads": dead_revive_heads,
        "lifetime_summary": {
            "total_heads": total_heads,
            "never_dead": never_dead,
            "dead_never_revive": dead_never_revive,
            "dead_revive": dead_revive,
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-head off-rate with sink token masked from attention."
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
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--dead-threshold", type=float, default=0.95)
    parser.add_argument("--mostly-dead-threshold", type=float, default=0.75)
    parser.add_argument(
        "--mask-key-index",
        type=int,
        default=0,
        help="Key index to mask out of attention (default: 0 for sink token).",
    )
    parser.add_argument(
        "--skip-first-query-on-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When masking key 0, avoid masking the first query position to prevent "
            "all-masked causal rows."
        ),
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.padding == "none" and args.batch_size != 1:
        raise ValueError("Use --batch-size 1 when --padding none to avoid padding.")
    if args.max_tokens == 0:
        raise ValueError("--max-tokens must be non-zero or -1 for all tokens.")
    if args.dead_threshold <= 0 or args.dead_threshold > 1:
        raise ValueError("--dead-threshold must be in (0, 1].")
    if args.mostly_dead_threshold <= 0 or args.mostly_dead_threshold > 1:
        raise ValueError("--mostly-dead-threshold must be in (0, 1].")
    if args.mostly_dead_threshold > args.dead_threshold:
        raise ValueError("--mostly-dead-threshold must be <= --dead-threshold.")
    if args.attn_impl is None:
        raise ValueError(
            "--attn-impl is required for masked attention. Use one of: "
            + ", ".join(sorted(ATTN_KERNELS))
        )
    if args.attn_impl not in ATTN_KERNELS:
        raise ValueError(
            f"Unsupported --attn-impl for masking. Use one of: "
            + ", ".join(sorted(ATTN_KERNELS))
        )


def run_checkpoint(
    model_path: str,
    model: torch.nn.Module,
    layers: list[LayerSpec],
    tokenizer,
    dataset_iterable: Iterable[dict],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[list[float]], dict]:
    num_heads_by_layer = [layer.num_heads for layer in layers]
    accumulator = HeadOffRateAccumulator(args.eps, num_heads_by_layer)
    hooks = register_o_proj_hooks(layers, accumulator)

    tokens_processed = 0
    samples_processed = 0

    with MaskedAttentionPatcher(args.mask_key_index, args.skip_first_query_on_mask):
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
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True,
                )
            accumulator.set_attention_mask(None)
            tokens_processed += batch_tokens
            samples_processed += input_ids.shape[0]
            if args.max_tokens >= 0 and tokens_processed >= args.max_tokens:
                break

    for handle in hooks:
        handle.remove()

    metadata = {
        "samples_processed": samples_processed,
        "tokens_processed": tokens_processed,
    }
    return accumulator.off_rates(), metadata


def main() -> None:
    args = parse_args()
    validate_args(args)

    steps = parse_steps(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    off_rates_by_step: list[list[list[float]]] = []
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

        off_rates, metadata = run_checkpoint(
            model_path=model_path,
            model=model,
            layers=layers,
            tokenizer=tokenizer,
            dataset_iterable=dataset_iterable,
            device=device,
            args=args,
        )
        off_rates_by_step.append(off_rates)
        step_metadata[str(step)] = metadata

    summary = compute_death_summary(
        steps,
        off_rates_by_step,
        dead_threshold=args.dead_threshold,
        mostly_dead_threshold=args.mostly_dead_threshold,
    )

    results = {
        "analysis_type": "head_dead_no_sink",
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
        "eps": args.eps,
        "mask_key_index": args.mask_key_index,
        "skip_first_query_on_mask": args.skip_first_query_on_mask,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "attn_impl": args.attn_impl,
        "step_metadata": step_metadata,
        "off_rates": {str(step): rates for step, rates in zip(steps, off_rates_by_step)},
        **summary,
    }

    print(json.dumps(results, indent=2, sort_keys=True))

    model_label = args.model_template.replace("{step}", "steps")
    output_path = resolve_output_path(
        args.output, args.output_dir, model_label, suffix="head_dead_no_sink"
    )
    if output_path is not None:
        write_results(output_path, results)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
