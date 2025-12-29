#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from einops import rearrange
from tqdm import tqdm

from fla.ops.attn.naive_abs_softmax import abs_softmax_1, abs_softmax_2
from fla.ops.attn.naive_relusoftpick import relu_softpick_1, relu_softpick_2
from fla.ops.attn.naive_softmax_plus_one import softmax_plus_one
from fla.ops.attn.naive_softpick import softpick
import fla.layers.attn as attn_module

from utils import (
    add_dataset_args,
    add_runtime_args,
    build_dataset_iterable,
    iter_batches,
    load_model_and_tokenizer,
    resolve_device,
    resolve_dtype,
    resolve_output_path,
    tokenize_batch,
    write_results,
)


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


@dataclass
class LossAccumulator:
    total_loss: float = 0.0
    total_tokens: int = 0

    def update(self, loss: torch.Tensor, token_count: int) -> None:
        self.total_loss += loss.item()
        self.total_tokens += token_count

    def perplexity(self) -> float:
        if self.total_tokens == 0:
            return float("nan")
        return math.exp(self.total_loss / self.total_tokens)


class MaskedAttentionPatcher:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure PPL with and without masking attention to a sink token."
    )
    parser.add_argument("--model", required=True, help="HF model name or local path.")
    add_dataset_args(parser)
    add_runtime_args(parser)
    parser.add_argument(
        "--mask-key-index",
        type=int,
        default=0,
        help="Key index to mask out of attention (default: 0).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline (unmasked) run.",
    )
    parser.add_argument(
        "--skip-masked",
        action="store_true",
        help="Skip the masked run.",
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
        help="Directory to write <model>_attention_knockout.jsonl.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.padding == "none" and args.batch_size != 1:
        raise ValueError("Use --batch-size 1 when --padding none to avoid padding.")
    if args.attn_impl is None:
        raise ValueError("--attn-impl is required for attention knockout runs.")
    if args.attn_impl not in ATTN_KERNELS:
        raise ValueError(
            "Unsupported --attn-impl for masking. Use one of: "
            + ", ".join(sorted(ATTN_KERNELS))
        )
    if args.streaming and not args.skip_baseline and not args.skip_masked:
        raise ValueError(
            "Streaming datasets cannot be iterated twice; use --skip-baseline or "
            "--skip-masked, or disable --streaming."
        )


def compute_ppl(
    model: torch.nn.Module,
    batch_iter,
    device: torch.device,
    skip_initial_positions: int = 0,
) -> tuple[float, int]:
    loss_acc = LossAccumulator()
    for encodings in tqdm(batch_iter, desc="Batches", unit="batch"):
        input_ids = encodings["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        if skip_initial_positions:
            shift_logits = shift_logits[:, skip_initial_positions:, :].contiguous()
            shift_labels = shift_labels[:, skip_initial_positions:].contiguous()
            if shift_labels.numel() == 0:
                continue
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        loss_acc.update(loss, shift_labels.numel())
    if loss_acc.total_tokens == 0:
        raise RuntimeError(
            "No tokens were processed. Increase --n-samples or ensure the dataset "
            "contains non-empty text entries."
        )
    return loss_acc.perplexity(), loss_acc.total_tokens


def main() -> None:
    args = parse_args()
    validate_args(args)

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

    dataset_iterable, n_samples = build_dataset_iterable(
        args.dataset,
        args.dataset_config,
        args.split,
        args.streaming,
        args.n_samples,
    )

    def batch_iter():
        for batch_texts in iter_batches(
            dataset_iterable, args.batch_size, max_samples=n_samples
        ):
            if not batch_texts:
                continue
            yield tokenize_batch(
                tokenizer, batch_texts, max_length=args.max_length, padding=args.padding
            )

    results: dict[str, float | int | str] = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "attn_impl": args.attn_impl,
        "mask_key_index": args.mask_key_index,
        "skip_first_query_on_mask": args.skip_first_query_on_mask,
    }

    if not args.skip_baseline:
        ppl, token_count = compute_ppl(model, batch_iter(), device)
        results["baseline_ppl"] = ppl
        results["baseline_tokens"] = token_count

    if not args.skip_masked:
        with MaskedAttentionPatcher(
            args.mask_key_index, args.skip_first_query_on_mask
        ):
            skip_initial = 1 if args.mask_key_index == 0 else 0
            ppl, token_count = compute_ppl(
                model, batch_iter(), device, skip_initial_positions=skip_initial
            )
        results["masked_ppl"] = ppl
        results["masked_tokens"] = token_count

    print(json.dumps(results, indent=2, sort_keys=True))
    output_path = resolve_output_path(
        args.output, args.output_dir, args.model, suffix="attention_knockout"
    )
    if output_path is not None:
        write_results(output_path, results)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
