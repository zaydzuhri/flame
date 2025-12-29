#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from tqdm import tqdm

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


@dataclass
class VarianceAccumulator:
    layer_sums: list[float] | None = None
    layer_counts: list[int] | None = None

    def update(self, variances: list[torch.Tensor]) -> None:
        if self.layer_sums is None:
            self.layer_sums = [0.0 for _ in variances]
            self.layer_counts = [0 for _ in variances]
        for idx, var in enumerate(variances):
            self.layer_sums[idx] += var.sum().item()
            self.layer_counts[idx] += var.numel()

    def finalize(self) -> list[float]:
        if self.layer_sums is None or self.layer_counts is None:
            return []
        results = []
        for total, count in zip(self.layer_sums, self.layer_counts):
            results.append(total / count if count else float("nan"))
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure variance of attention logits targeting the sink token."
    )
    parser.add_argument("--model", required=True, help="HF model name or local path.")
    add_dataset_args(parser)
    add_runtime_args(parser)
    parser.add_argument(
        "--mask-key-index",
        type=int,
        default=0,
        help="Key index to analyze (default: 0).",
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
        help="Directory to write <model>_sink_logit_variance.jsonl.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.padding == "none" and args.batch_size != 1:
        raise ValueError("Use --batch-size 1 when --padding none to avoid padding.")


def compute_scores(attn_module, hidden_states: torch.Tensor) -> torch.Tensor:
    q = attn_module.q_proj(hidden_states)
    k = attn_module.k_proj(hidden_states)
    head_dim = attn_module.head_dim
    q = rearrange(q, "b t (h d) -> b t h d", d=head_dim)
    k = rearrange(k, "b t (h d) -> b t h d", d=head_dim)
    if getattr(attn_module, "qk_norm", False):
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)
    q_len = q.shape[1]
    max_seqlen = q_len
    if attn_module.max_position_embeddings is not None:
        max_seqlen = max(max_seqlen, attn_module.max_position_embeddings)
    q, k = attn_module.rotary(
        q, k, seqlen_offset=0, max_seqlen=max_seqlen, cu_seqlens=None
    )
    if hasattr(attn_module, "attn_impl") and "scaled" in attn_module.attn_impl:
        k_len = k.shape[1]
        q = q * attn_module.s.to(q.dtype) * attn_module.logn[k_len - q_len:k_len].to(q.dtype)
    q = rearrange(q, "b t h d -> b h t d")
    k = rearrange(k, "b t h d -> b h t d")
    scores = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
    return scores


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

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Model does not expose model.layers for attention analysis.")

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

    accumulator = VarianceAccumulator()
    samples_processed = 0

    for encodings in tqdm(batch_iter(), desc="Batches", unit="batch"):
        input_ids = encodings["input_ids"].to(device)
        hidden_states = model.model.embeddings(input_ids)
        layer_variances = []
        with torch.inference_mode():
            for layer in model.model.layers:
                normed = layer.attn_norm(hidden_states)
                scores = compute_scores(layer.attn, normed)
                k_len = scores.shape[-1]
                if not (-k_len <= args.mask_key_index < k_len):
                    raise ValueError(
                        f"mask_key_index {args.mask_key_index} is out of range for k_len={k_len}."
                    )
                sink_scores = scores[..., args.mask_key_index]
                layer_variances.append(sink_scores.var(dim=-1, unbiased=False))
                hidden_states = layer(
                    hidden_states,
                    attention_mask=None,
                    output_attentions=False,
                    use_cache=False,
                )[0]
        accumulator.update(layer_variances)
        samples_processed += input_ids.shape[0]

    per_layer_variance = accumulator.finalize()
    results = {
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
        "samples_processed": samples_processed,
        "sink_logit_variance": {
            "per_layer_mean": per_layer_variance,
        },
    }

    print(json.dumps(results, indent=2, sort_keys=True))
    output_path = resolve_output_path(
        args.output, args.output_dir, args.model, suffix="sink_logit_variance"
    )
    if output_path is not None:
        write_results(output_path, results)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
