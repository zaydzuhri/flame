#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


def _sink_index(sink_index: int, k_len: int) -> int:
    if not (-k_len <= sink_index < k_len):
        raise ValueError(f"--sink-index {sink_index} is out of range for k_len={k_len}.")
    return sink_index if sink_index >= 0 else k_len + sink_index


def compute_sink_stats(
    attentions: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sink_index: int,
    entropy_eps: float,
    sink_dominance_threshold: float,
    entropy_threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    k_len = attentions.shape[-1]
    sink_idx = _sink_index(sink_index, k_len)
    sink_attention = attentions[..., sink_idx]  # [B, H, T]
    attn_safe = attentions.float().clamp_min(entropy_eps)
    entropy = -(attn_safe * attn_safe.log()).sum(dim=-1)
    if k_len > 1:
        entropy = entropy / math.log(k_len)
    dormant_mask = sink_attention > sink_dominance_threshold
    if entropy_threshold is not None:
        dormant_mask = dormant_mask & (entropy < entropy_threshold)
    if attention_mask is not None:
        mask = attention_mask.to(dtype=torch.bool)
        sink_attention = sink_attention * mask[:, None, :]
        entropy = entropy * mask[:, None, :]
        dormant_mask = dormant_mask & mask[:, None, :]
        valid_tokens = int(mask.sum().item())
    else:
        valid_tokens = sink_attention.shape[0] * sink_attention.shape[2]
    sink_sum = sink_attention.sum(dim=(0, 2)).to(dtype=torch.float64)
    entropy_sum = entropy.sum(dim=(0, 2)).to(dtype=torch.float64)
    dormant_count = dormant_mask.sum(dim=(0, 2)).to(dtype=torch.float64)
    return sink_sum, entropy_sum, dormant_count, valid_tokens


class SinkAttentionAccumulator:
    def __init__(
        self,
        sink_index: int,
        entropy_eps: float,
        sink_dominance_threshold: float,
        entropy_threshold: float | None,
        num_heads_by_layer: Sequence[int],
    ) -> None:
        self.sink_index = int(sink_index)
        self.entropy_eps = float(entropy_eps)
        self.sink_dominance_threshold = float(sink_dominance_threshold)
        self.entropy_threshold = entropy_threshold
        self.sink_sums = [
            torch.zeros(num_heads, dtype=torch.float64) for num_heads in num_heads_by_layer
        ]
        self.entropy_sums = [
            torch.zeros(num_heads, dtype=torch.float64) for num_heads in num_heads_by_layer
        ]
        self.dormant_counts = [
            torch.zeros(num_heads, dtype=torch.float64) for num_heads in num_heads_by_layer
        ]
        self.total_counts = [0.0 for _ in num_heads_by_layer]
        self.attention_mask: torch.Tensor | None = None

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self.attention_mask = attention_mask

    def update(self, attentions: Sequence[torch.Tensor]) -> None:
        if len(attentions) != len(self.sink_sums):
            raise ValueError(
                f"Expected {len(self.sink_sums)} attention layers, got {len(attentions)}."
            )
        for idx, attn in enumerate(attentions):
            sink_sum, entropy_sum, dormant_count, valid_tokens = compute_sink_stats(
                attn,
                self.attention_mask,
                self.sink_index,
                self.entropy_eps,
                self.sink_dominance_threshold,
                self.entropy_threshold,
            )
            self.sink_sums[idx] += sink_sum.cpu()
            self.entropy_sums[idx] += entropy_sum.cpu()
            self.dormant_counts[idx] += dormant_count.cpu()
            if valid_tokens > 0:
                self.total_counts[idx] += float(valid_tokens)

    def sink_attention_rates(self) -> list[list[float]]:
        rates: list[list[float]] = []
        for sums, total in zip(self.sink_sums, self.total_counts):
            if total <= 0:
                rates.append([float("nan") for _ in range(sums.numel())])
                continue
            rates.append((sums / total).tolist())
        return rates

    def entropy_rates(self) -> list[list[float]]:
        rates: list[list[float]] = []
        for sums, total in zip(self.entropy_sums, self.total_counts):
            if total <= 0:
                rates.append([float("nan") for _ in range(sums.numel())])
                continue
            rates.append((sums / total).tolist())
        return rates

    def dormant_rates(self) -> list[list[float]]:
        rates: list[list[float]] = []
        for counts, total in zip(self.dormant_counts, self.total_counts):
            if total <= 0:
                rates.append([float("nan") for _ in range(counts.numel())])
                continue
            rates.append((counts / total).tolist())
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
        raise RuntimeError("No attention modules found to hook.")
    if all(layer.layer_idx >= 0 for layer in modules):
        modules.sort(key=lambda layer: layer.layer_idx)
    else:
        modules = [
            LayerSpec(idx, layer.num_heads, layer.name, layer.module)
            for idx, layer in enumerate(modules)
        ]
    return modules


def compute_dormant_summary(
    steps: Sequence[int],
    dormant_rates_by_step: Sequence[list[list[float]]],
    dormant_threshold: float,
    mostly_dormant_threshold: float,
) -> dict:
    if not dormant_rates_by_step:
        return {}
    num_layers = len(dormant_rates_by_step[0])
    num_heads_by_layer = [len(layer) for layer in dormant_rates_by_step[0]]
    dormant_by_step: list[list[list[bool]]] = []
    mostly_dormant_by_step: list[list[list[bool]]] = []
    dormant_counts: list[int] = []
    mostly_dormant_counts: list[int] = []
    dormant_heads_by_step: dict[str, list[list[int]]] = {}
    mostly_dormant_heads_by_step: dict[str, list[list[int]]] = {}

    for step_idx, step in enumerate(steps):
        step_rates = dormant_rates_by_step[step_idx]
        dormant_flags: list[list[bool]] = []
        mostly_flags: list[list[bool]] = []
        step_dormant = 0
        step_mostly = 0
        dormant_heads: list[list[int]] = []
        mostly_heads: list[list[int]] = []
        for layer_idx, layer_rates in enumerate(step_rates):
            layer_dormant: list[bool] = []
            layer_mostly: list[bool] = []
            for head_idx, rate in enumerate(layer_rates):
                is_dormant = rate > dormant_threshold
                is_mostly = rate > mostly_dormant_threshold
                layer_dormant.append(is_dormant)
                layer_mostly.append(is_mostly)
                if is_dormant:
                    dormant_heads.append([layer_idx, head_idx])
                if is_mostly:
                    mostly_heads.append([layer_idx, head_idx])
            step_dormant += sum(layer_dormant)
            step_mostly += sum(layer_mostly)
            dormant_flags.append(layer_dormant)
            mostly_flags.append(layer_mostly)
        dormant_by_step.append(dormant_flags)
        mostly_dormant_by_step.append(mostly_flags)
        dormant_counts.append(step_dormant)
        mostly_dormant_counts.append(step_mostly)
        dormant_heads_by_step[str(step)] = dormant_heads
        mostly_dormant_heads_by_step[str(step)] = mostly_heads

    dormant_step = [
        [None for _ in range(num_heads_by_layer[layer_idx])] for layer_idx in range(num_layers)
    ]
    revived = [
        [False for _ in range(num_heads_by_layer[layer_idx])] for layer_idx in range(num_layers)
    ]
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads_by_layer[layer_idx]):
            was_dormant = False
            for step, dormant_flags in zip(steps, dormant_by_step):
                is_dormant = dormant_flags[layer_idx][head_idx]
                if is_dormant and dormant_step[layer_idx][head_idx] is None:
                    dormant_step[layer_idx][head_idx] = step
                if was_dormant and not is_dormant:
                    revived[layer_idx][head_idx] = True
                was_dormant = is_dormant

    total_heads = sum(num_heads_by_layer)
    never_dormant = 0
    dormant_never_revive = 0
    dormant_revive = 0
    dormant_never_revive_heads: list[list[int]] = []
    dormant_revive_heads: list[list[int]] = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads_by_layer[layer_idx]):
            if dormant_step[layer_idx][head_idx] is None:
                never_dormant += 1
                continue
            if revived[layer_idx][head_idx]:
                dormant_revive += 1
                dormant_revive_heads.append([layer_idx, head_idx])
            else:
                dormant_never_revive += 1
                dormant_never_revive_heads.append([layer_idx, head_idx])

    summary = {
        "dormant_head_count": {str(step): count for step, count in zip(steps, dormant_counts)},
        "mostly_dormant_head_count": {
            str(step): count for step, count in zip(steps, mostly_dormant_counts)
        },
        "dormant_heads_by_step": dormant_heads_by_step,
        "mostly_dormant_heads_by_step": mostly_dormant_heads_by_step,
        "dormant_step": dormant_step,
        "revived": revived,
        "dormant_never_revive_heads": dormant_never_revive_heads,
        "dormant_revive_heads": dormant_revive_heads,
        "lifetime_summary": {
            "total_heads": total_heads,
            "never_dormant": never_dormant,
            "dormant_never_revive": dormant_never_revive,
            "dormant_revive": dormant_revive,
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-head sink attention for softmax checkpoints."
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
    parser.add_argument(
        "--sink-dominance-threshold",
        type=float,
        default=0.9,
        help="Per-token sink attention threshold for dormancy checks.",
    )
    parser.add_argument(
        "--dormant-threshold",
        type=float,
        default=0.95,
        help="Per-head dormancy rate threshold (fraction of tokens).",
    )
    parser.add_argument(
        "--mostly-dormant-threshold",
        type=float,
        default=0.75,
        help="Per-head mostly-dormant rate threshold (fraction of tokens).",
    )
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1; naive attention ignores padding.")
    if args.padding != "none":
        raise ValueError("Use --padding none to avoid padded tokens.")
    if args.max_tokens == 0:
        raise ValueError("--max-tokens must be non-zero or -1 for all tokens.")
    if args.sink_dominance_threshold <= 0 or args.sink_dominance_threshold > 1:
        raise ValueError("--sink-dominance-threshold must be in (0, 1].")
    if args.dormant_threshold <= 0 or args.dormant_threshold > 1:
        raise ValueError("--dormant-threshold must be in (0, 1].")
    if args.mostly_dormant_threshold <= 0 or args.mostly_dormant_threshold > 1:
        raise ValueError("--mostly-dormant-threshold must be in (0, 1].")
    if args.mostly_dormant_threshold > args.dormant_threshold:
        raise ValueError("--mostly-dormant-threshold must be <= --dormant-threshold.")
    if args.entropy_threshold is not None:
        if args.entropy_threshold < 0 or args.entropy_threshold > 1:
            raise ValueError("--entropy-threshold must be in [0, 1].")
    if args.attn_impl is None or not args.attn_impl.startswith("naive"):
        raise ValueError("Use --attn-impl naive_attn to enable attention outputs.")


def run_checkpoint(
    model_path: str,
    model: torch.nn.Module,
    layers: list[LayerSpec],
    tokenizer,
    dataset_iterable: Iterable[dict],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], dict]:
    num_heads_by_layer = [layer.num_heads for layer in layers]
    accumulator = SinkAttentionAccumulator(
        sink_index=args.sink_index,
        entropy_eps=args.entropy_eps,
        sink_dominance_threshold=args.sink_dominance_threshold,
        entropy_threshold=args.entropy_threshold,
        num_heads_by_layer=num_heads_by_layer,
    )

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
        if outputs.attentions is None or any(attn is None for attn in outputs.attentions):
            raise RuntimeError(
                "Model returned no attentions. Use --attn-impl naive_attn to enable attention outputs."
            )
        accumulator.update(tuple(attn for attn in outputs.attentions if attn is not None))
        accumulator.set_attention_mask(None)
        tokens_processed += batch_tokens
        samples_processed += input_ids.shape[0]
        if args.max_tokens >= 0 and tokens_processed >= args.max_tokens:
            break

    metadata = {
        "samples_processed": samples_processed,
        "tokens_processed": tokens_processed,
    }
    return (
        accumulator.sink_attention_rates(),
        accumulator.entropy_rates(),
        accumulator.dormant_rates(),
        metadata,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    steps = parse_steps(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    sink_rates_by_step: list[list[list[float]]] = []
    entropy_rates_by_step: list[list[list[float]]] = []
    dormant_rates_by_step: list[list[list[float]]] = []
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

        sink_rates, entropy_rates, dormant_rates, metadata = run_checkpoint(
            model_path=model_path,
            model=model,
            layers=layers,
            tokenizer=tokenizer,
            dataset_iterable=dataset_iterable,
            device=device,
            args=args,
        )
        sink_rates_by_step.append(sink_rates)
        entropy_rates_by_step.append(entropy_rates)
        dormant_rates_by_step.append(dormant_rates)
        step_metadata[str(step)] = metadata

    summary = compute_dormant_summary(
        steps,
        dormant_rates_by_step,
        dormant_threshold=args.dormant_threshold,
        mostly_dormant_threshold=args.mostly_dormant_threshold,
    )

    results = {
        "analysis_type": "head_dormant_guo",
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
        "sink_index": args.sink_index,
        "entropy_eps": args.entropy_eps,
        "entropy_normalized": True,
        "sink_dominance_threshold": args.sink_dominance_threshold,
        "dormant_threshold": args.dormant_threshold,
        "mostly_dormant_threshold": args.mostly_dormant_threshold,
        "entropy_threshold": args.entropy_threshold,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "attn_impl": args.attn_impl,
        "step_metadata": step_metadata,
        "sink_attention": {str(step): rates for step, rates in zip(steps, sink_rates_by_step)},
        "entropy": {str(step): rates for step, rates in zip(steps, entropy_rates_by_step)},
        "dormant_rate": {str(step): rates for step, rates in zip(steps, dormant_rates_by_step)},
        **summary,
    }

    print(json.dumps(results, indent=2, sort_keys=True))

    model_label = args.model_template.replace("{step}", "steps")
    output_path = resolve_output_path(
        args.output, args.output_dir, model_label, suffix="head_dormant"
    )
    if output_path is not None:
        write_results(output_path, results)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
