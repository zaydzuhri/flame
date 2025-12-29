#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset

import measure_metrics

DEFAULT_PROMPT = (
    "According to all known laws of aviation, there is no way a bee should be able to"
)
DEFAULT_LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def resolve_sweep_output_path(
    output_dir: Path, model_name: str, run_name: str | None
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = run_name or model_name
    filename = f"{measure_metrics.sanitize_model_name(label)}_length_sweep.jsonl"
    return output_dir / filename


def iter_truncated_batches(
    dataset_iterable: Iterable[dict],
    tokenizer,
    sequence_length: int,
    n_sequences: int,
    batch_size: int,
) -> Iterable[dict[str, torch.Tensor]]:
    batch: list[str] = []
    sequences_emitted = 0
    for sample in dataset_iterable:
        if n_sequences >= 0 and sequences_emitted >= n_sequences:
            break
        text = measure_metrics.extract_text(sample)
        if text is None:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            encodings = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=sequence_length,
                return_tensors="pt",
            )
            sequences_emitted += encodings["input_ids"].shape[0]
            yield encodings
            batch.clear()
    if batch:
        encodings = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=sequence_length,
            return_tensors="pt",
        )
        yield encodings
        sequences_emitted += encodings["input_ids"].shape[0]
    if n_sequences >= 0 and sequences_emitted < n_sequences:
        print(
            f"Warning: only emitted {sequences_emitted} sequences; requested {n_sequences}."
        )


def iter_packed_batches(
    dataset_iterable: Iterable[dict],
    tokenizer,
    sequence_length: int,
    n_sequences: int,
    batch_size: int,
    add_eos: bool,
) -> Iterable[dict[str, torch.Tensor]]:
    buffer: list[int] = []
    batch: list[list[int]] = []
    sequences_emitted = 0
    eos_token_id = tokenizer.eos_token_id if add_eos else None

    for sample in dataset_iterable:
        if n_sequences >= 0 and sequences_emitted >= n_sequences:
            break
        text = measure_metrics.extract_text(sample)
        if text is None:
            continue
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if eos_token_id is not None:
            token_ids.append(eos_token_id)
        buffer.extend(token_ids)

        while len(buffer) >= sequence_length:
            block = buffer[:sequence_length]
            del buffer[:sequence_length]
            batch.append(block)
            sequences_emitted += 1
            if len(batch) >= batch_size:
                input_ids = torch.tensor(batch, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                yield {"input_ids": input_ids, "attention_mask": attention_mask}
                batch.clear()
            if n_sequences >= 0 and sequences_emitted >= n_sequences:
                break

    if batch:
        input_ids = torch.tensor(batch, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        yield {"input_ids": input_ids, "attention_mask": attention_mask}
    if n_sequences >= 0 and sequences_emitted < n_sequences:
        print(
            f"Warning: only emitted {sequences_emitted} sequences; requested {n_sequences}."
        )


def iter_generated_batches(
    model: torch.nn.Module,
    tokenizer,
    sequence_length: int,
    n_sequences: int,
    batch_size: int,
    prompt: str,
    device: torch.device,
    do_sample: bool,
    temperature: float,
    seed: int | None,
) -> Iterable[dict[str, torch.Tensor]]:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    if prompt_len >= sequence_length:
        raise ValueError(
            f"Prompt length {prompt_len} must be smaller than sequence length {sequence_length}."
        )
    max_new_tokens = sequence_length - prompt_len
    if seed is not None:
        torch.manual_seed(seed)

    sequences_emitted = 0
    while n_sequences < 0 or sequences_emitted < n_sequences:
        current_batch = batch_size
        if n_sequences >= 0:
            current_batch = min(batch_size, n_sequences - sequences_emitted)
        input_ids = prompt_ids.expand(current_batch, -1)
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                use_cache=True,
                eos_token_id=None,
                forced_eos_token_id=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        if generated.shape[1] != sequence_length:
            raise RuntimeError(
                f"Expected {sequence_length} tokens, got {generated.shape[1]}."
            )
        attention_mask = torch.ones_like(generated)
        yield {"input_ids": generated, "attention_mask": attention_mask}
        sequences_emitted += current_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep sequence lengths and measure attention sink metrics."
    )
    parser.add_argument("--model", required=True, help="HF model name or local path.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional label to group this sweep in plots.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTHS,
        help="Sequence lengths to sweep.",
    )
    parser.add_argument(
        "--source",
        choices=("dataset", "generate"),
        default="dataset",
        help="Use dataset text or generated tokens as input.",
    )
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
        "--dataset-mode",
        choices=("pack", "truncate"),
        default="pack",
        help="Pack tokens to exact length or truncate/pad each sample.",
    )
    parser.add_argument(
        "--add-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append EOS between packed sequences (pack mode only).",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=64,
        help="Number of sequences per length (-1 for all).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for inference."
    )
    parser.add_argument(
        "--sink-eps",
        type=float,
        nargs="+",
        default=[0.2, 0.3],
        help="Sink rate thresholds (space-separated).",
    )
    parser.add_argument(
        "--sparsity-eps",
        type=float,
        default=0.0,
        help="Treat values with abs <= eps as zeros for sparsity (default: 0).",
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
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text for generation mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation mode.",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample during generation (generation mode only).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/attention_sink/outputs"),
        help="Directory to write sweep JSONL outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sweep output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = measure_metrics.resolve_device(args.device)
    dtype = measure_metrics.resolve_dtype(args.dtype, device)

    model, tokenizer = measure_metrics.load_model_and_tokenizer(
        args.model,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        attn_impl=args.attn_impl,
    )

    output_path = resolve_sweep_output_path(
        args.output_dir, args.model, args.run_name
    )
    if args.overwrite and output_path.exists():
        output_path.unlink()

    lengths = list(dict.fromkeys(args.lengths))
    lengths.sort()

    for sequence_length in lengths:
        if args.source == "dataset":
            dataset = load_dataset(
                args.dataset,
                args.dataset_config,
                split=args.split,
                streaming=args.streaming,
            )
            if args.dataset_mode == "pack":
                batch_iter = iter_packed_batches(
                    dataset,
                    tokenizer,
                    sequence_length,
                    args.n_sequences,
                    args.batch_size,
                    add_eos=args.add_eos,
                )
            else:
                if not args.streaming and args.n_sequences >= 0:
                    total_available = len(dataset)
                    n_samples = min(args.n_sequences, total_available)
                    dataset = dataset.select(range(n_samples))
                batch_iter = iter_truncated_batches(
                    dataset,
                    tokenizer,
                    sequence_length,
                    args.n_sequences,
                    args.batch_size,
                )
        else:
            batch_iter = iter_generated_batches(
                model,
                tokenizer,
                sequence_length,
                args.n_sequences,
                args.batch_size,
                prompt=args.prompt,
                device=device,
                do_sample=args.do_sample,
                temperature=args.temperature,
                seed=args.seed,
            )

        total_batches = None
        if args.n_sequences >= 0:
            total_batches = math.ceil(args.n_sequences / args.batch_size)

        metrics = measure_metrics.compute_metrics(
            model=model,
            batch_iter=batch_iter,
            sink_eps=args.sink_eps,
            sparsity_eps=args.sparsity_eps,
            device=device,
            total_batches=total_batches,
            progress_desc=f"len={sequence_length}",
        )

        result = {
            "run_name": args.run_name or args.model,
            "model": args.model,
            "dataset": args.dataset if args.source == "dataset" else None,
            "dataset_config": args.dataset_config if args.source == "dataset" else None,
            "split": args.split if args.source == "dataset" else None,
            "source": args.source,
            "dataset_mode": args.dataset_mode if args.source == "dataset" else None,
            "sequence_length": sequence_length,
            "batch_size": args.batch_size,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "attn_impl": args.attn_impl,
            **metrics,
        }

        measure_metrics.write_results(output_path, result)
        print(f"Saved {sequence_length} results to {output_path}")


if __name__ == "__main__":
    main()
