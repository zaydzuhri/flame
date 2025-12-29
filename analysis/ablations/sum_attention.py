#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
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
class AttentionSumAccumulator:
    layer_sums: list[torch.Tensor] | None = None
    layer_counts: list[int] | None = None

    def update(self, attentions: tuple[torch.Tensor, ...]) -> None:
        if self.layer_sums is None:
            self.layer_sums = [torch.zeros(attn.shape[1], dtype=torch.float64) for attn in attentions]
            self.layer_counts = [0 for _ in attentions]

        for idx, attn in enumerate(attentions):
            attn_sum = attn.sum(dim=-1)  # [B, H, T]
            self.layer_sums[idx] += attn_sum.sum(dim=(0, 2)).double().cpu()
            self.layer_counts[idx] += attn_sum.shape[0] * attn_sum.shape[2]

    def finalize(self) -> tuple[list[list[float]], list[float]]:
        if self.layer_sums is None or self.layer_counts is None:
            return [], []
        per_layer = []
        per_layer_mean = []
        for sums, count in zip(self.layer_sums, self.layer_counts):
            if count == 0:
                per_head = [float("nan") for _ in range(sums.numel())]
                per_layer.append(per_head)
                per_layer_mean.append(float("nan"))
                continue
            per_head = (sums / count).tolist()
            per_layer.append(per_head)
            per_layer_mean.append(float(sum(per_head) / len(per_head)) if per_head else float("nan"))
        return per_layer, per_layer_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure sum(attention) per head to test no-op behavior."
    )
    parser.add_argument("--model", required=True, help="HF model name or local path.")
    add_dataset_args(parser)
    add_runtime_args(parser)
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
        help="Directory to write <model>_sum_attention.jsonl.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.padding == "none" and args.batch_size != 1:
        raise ValueError("Use --batch-size 1 when --padding none to avoid padding.")


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

    accumulator = AttentionSumAccumulator()
    samples_processed = 0

    for encodings in tqdm(batch_iter(), desc="Batches", unit="batch"):
        input_ids = encodings["input_ids"].to(device)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        if outputs.attentions is None or not any(attn is not None for attn in outputs.attentions):
            raise RuntimeError(
                "Model returned no attentions. Use --attn-impl naive_attn (or another "
                "naive_* impl) to enable attention outputs."
            )
        attentions = tuple(attn for attn in outputs.attentions if attn is not None)
        accumulator.update(attentions)
        samples_processed += input_ids.shape[0]

    per_layer, per_layer_mean = accumulator.finalize()
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
        "samples_processed": samples_processed,
        "sum_attention": {
            "per_layer_per_head": per_layer,
            "per_layer_mean": per_layer_mean,
        },
    }

    print(json.dumps(results, indent=2, sort_keys=True))
    output_path = resolve_output_path(
        args.output, args.output_dir, args.model, suffix="sum_attention"
    )
    if output_path is not None:
        write_results(output_path, results)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
