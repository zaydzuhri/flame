#!/usr/bin/env python3
"""
Residual State Norm Stability Analysis

Tracks L2 norm of residual states across layers and sequence positions
during inference, comparing Softmax vs Softpick transformers.

Expected behavior:
- Softmax: Linear or superlinear growth in norm with depth
- Softpick: Constant or bounded norm growth
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
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
except ImportError:
    from utils import (
        extract_text,
        load_model_and_tokenizer,
        resolve_device,
        resolve_dtype,
        resolve_output_path,
        tokenize_batch,
        write_results,
    )


@dataclass
class LayerSpec:
    layer_idx: int
    name: str
    block: torch.nn.Module
    attn: torch.nn.Module


@dataclass
class ResidualNormAccumulator:
    """Accumulates residual norm statistics across batches."""

    num_layers: int
    # Per-layer accumulators: sum of norms and count for mean computation
    post_attn_norm_sum: list[torch.Tensor] = field(default_factory=list)
    post_ffn_norm_sum: list[torch.Tensor] = field(default_factory=list)
    post_attn_norm_max: list[torch.Tensor] = field(default_factory=list)
    post_ffn_norm_max: list[torch.Tensor] = field(default_factory=list)
    token_count: list[int] = field(default_factory=list)

    # Per-position accumulators (for final layer only)
    final_layer_position_norms: list[list[float]] = field(default_factory=list)

    # Storage for current batch's intermediate values
    _layer_inputs: dict[int, torch.Tensor] = field(default_factory=dict)
    _attn_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    _attention_mask: torch.Tensor | None = None

    def __post_init__(self):
        for _ in range(self.num_layers):
            self.post_attn_norm_sum.append(torch.tensor(0.0, dtype=torch.float64))
            self.post_ffn_norm_sum.append(torch.tensor(0.0, dtype=torch.float64))
            self.post_attn_norm_max.append(torch.tensor(0.0, dtype=torch.float64))
            self.post_ffn_norm_max.append(torch.tensor(0.0, dtype=torch.float64))
            self.token_count.append(0)

    def set_attention_mask(self, mask: torch.Tensor | None) -> None:
        self._attention_mask = mask

    def clear_batch_state(self) -> None:
        self._layer_inputs.clear()
        self._attn_outputs.clear()

    def store_layer_input(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        self._layer_inputs[layer_idx] = hidden_states.detach()

    def store_attn_output(self, layer_idx: int, attn_out: torch.Tensor) -> None:
        self._attn_outputs[layer_idx] = attn_out.detach()

    def update_post_ffn(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        """Update statistics from post-FFN residual (block output)."""
        # hidden_states: (batch, seq_len, hidden_dim)
        residual = hidden_states.detach().float()

        # Compute L2 norm per position: (batch, seq_len)
        norms = torch.norm(residual, p=2, dim=-1)

        # Apply attention mask if available
        if self._attention_mask is not None:
            mask = self._attention_mask.bool()
            valid_norms = norms[mask]
            valid_count = mask.sum().item()
        else:
            valid_norms = norms.flatten()
            valid_count = norms.numel()

        if valid_count > 0:
            self.post_ffn_norm_sum[layer_idx] += valid_norms.sum().cpu().to(torch.float64)
            self.post_ffn_norm_max[layer_idx] = torch.max(
                self.post_ffn_norm_max[layer_idx],
                valid_norms.max().cpu().to(torch.float64),
            )
            self.token_count[layer_idx] += valid_count

        # For final layer, store per-position norms
        if layer_idx == self.num_layers - 1:
            # Store mean across batch for each position
            if self._attention_mask is not None:
                # Masked mean per position
                mask = self._attention_mask.float()
                masked_norms = norms * mask
                pos_sums = masked_norms.sum(dim=0)
                pos_counts = mask.sum(dim=0).clamp(min=1)
                pos_means = (pos_sums / pos_counts).cpu().tolist()
            else:
                pos_means = norms.mean(dim=0).cpu().tolist()
            self.final_layer_position_norms.append(pos_means)

        # Compute post-attention residual if we have the inputs
        if layer_idx in self._layer_inputs and layer_idx in self._attn_outputs:
            layer_input = self._layer_inputs[layer_idx].float()
            attn_out = self._attn_outputs[layer_idx].float()
            post_attn_residual = layer_input + attn_out
            post_attn_norms = torch.norm(post_attn_residual, p=2, dim=-1)

            if self._attention_mask is not None:
                mask = self._attention_mask.bool()
                valid_post_attn = post_attn_norms[mask]
            else:
                valid_post_attn = post_attn_norms.flatten()

            if valid_count > 0:
                self.post_attn_norm_sum[layer_idx] += valid_post_attn.sum().cpu().to(torch.float64)
                self.post_attn_norm_max[layer_idx] = torch.max(
                    self.post_attn_norm_max[layer_idx],
                    valid_post_attn.max().cpu().to(torch.float64),
                )

    def get_results(self) -> dict:
        """Compute final statistics."""
        mean_post_attn = []
        mean_post_ffn = []
        max_post_attn = []
        max_post_ffn = []

        for i in range(self.num_layers):
            count = self.token_count[i]
            if count > 0:
                mean_post_attn.append(float(self.post_attn_norm_sum[i] / count))
                mean_post_ffn.append(float(self.post_ffn_norm_sum[i] / count))
            else:
                mean_post_attn.append(0.0)
                mean_post_ffn.append(0.0)
            max_post_attn.append(float(self.post_attn_norm_max[i]))
            max_post_ffn.append(float(self.post_ffn_norm_max[i]))

        # Aggregate position-wise norms (mean across all batches)
        if self.final_layer_position_norms:
            max_len = max(len(x) for x in self.final_layer_position_norms)
            aggregated_pos = [0.0] * max_len
            counts_pos = [0] * max_len
            for batch_norms in self.final_layer_position_norms:
                for pos, val in enumerate(batch_norms):
                    aggregated_pos[pos] += val
                    counts_pos[pos] += 1
            position_norms = [
                aggregated_pos[i] / counts_pos[i] if counts_pos[i] > 0 else 0.0
                for i in range(max_len)
            ]
        else:
            position_norms = []

        return {
            "mean_post_attn_norm": mean_post_attn,
            "mean_post_ffn_norm": mean_post_ffn,
            "max_post_attn_norm": max_post_attn,
            "max_post_ffn_norm": max_post_ffn,
            "position_norms_final_layer": position_norms,
            "total_tokens": self.token_count,
        }


def collect_transformer_blocks(model: torch.nn.Module) -> list[LayerSpec]:
    """Find all TransformerBlock modules and their attention submodules."""
    from fla.models.transformer.modeling_transformer import TransformerBlock

    layers: list[LayerSpec] = []
    for name, module in model.named_modules():
        if isinstance(module, TransformerBlock):
            layer_idx = module.layer_idx
            layers.append(
                LayerSpec(
                    layer_idx=layer_idx,
                    name=name,
                    block=module,
                    attn=module.attn,
                )
            )

    if not layers:
        raise RuntimeError("No TransformerBlock modules found.")

    layers.sort(key=lambda x: x.layer_idx)
    return layers


def register_residual_hooks(
    layers: list[LayerSpec],
    accumulator: ResidualNormAccumulator,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register hooks to capture residual norms."""
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer in layers:
        idx = layer.layer_idx

        # Hook to capture block input (layer input for post-attn residual computation)
        def block_pre_hook(module, args, layer_idx=idx):
            # args[0] is hidden_states
            if args:
                accumulator.store_layer_input(layer_idx, args[0])

        # Hook to capture attention output (before residual add)
        def attn_hook(module, args, output, layer_idx=idx):
            # output is (hidden_states, attentions, past_key_values)
            if isinstance(output, tuple) and len(output) > 0:
                accumulator.store_attn_output(layer_idx, output[0])

        # Hook to capture block output (post-FFN residual)
        def block_post_hook(module, args, output, layer_idx=idx):
            # output is (hidden_states, ...) tuple
            if isinstance(output, tuple) and len(output) > 0:
                accumulator.update_post_ffn(layer_idx, output[0])

        h1 = layer.block.register_forward_pre_hook(block_pre_hook)
        h2 = layer.attn.register_forward_hook(attn_hook)
        h3 = layer.block.register_forward_hook(block_post_hook)
        handles.extend([h1, h2, h3])

    return handles


def parse_steps(args: argparse.Namespace) -> list[int]:
    if args.steps:
        steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
        if not steps:
            raise ValueError("Provide at least one step via --steps.")
        return sorted(set(steps))
    if args.step_start is None or args.step_end is None:
        raise ValueError("Use --steps or --step-start/--step-end.")
    if args.step_stride <= 0:
        raise ValueError("--step-stride must be positive.")
    return list(range(args.step_start, args.step_end + 1, args.step_stride))


def build_dataset_iterable(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    streaming: bool,
) -> Iterable[dict]:
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
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


def run_analysis(
    model_path: str,
    model: torch.nn.Module,
    layers: list[LayerSpec],
    tokenizer,
    dataset_iterable: Iterable[dict],
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    num_layers = len(layers)
    accumulator = ResidualNormAccumulator(num_layers=num_layers)
    hooks = register_residual_hooks(layers, accumulator)

    tokens_processed = 0
    batches_processed = 0

    try:
        for encodings, batch_tokens in tqdm(
            iter_token_batches(
                dataset_iterable,
                tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                padding=args.padding,
                max_tokens=args.max_tokens,
            ),
            desc=f"Processing {model_path}",
            unit="batch",
        ):
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            accumulator.set_attention_mask(attention_mask)
            accumulator.clear_batch_state()

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
            batches_processed += 1

            if args.max_tokens >= 0 and tokens_processed >= args.max_tokens:
                break
    finally:
        for handle in hooks:
            handle.remove()

    results = accumulator.get_results()
    results["tokens_processed"] = tokens_processed
    results["batches_processed"] = batches_processed
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual State Norm Stability Analysis: Track L2 norms across layers."
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
    parser.add_argument("--batch-size", type=int, default=16)
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


def main() -> None:
    args = parse_args()
    steps = parse_steps(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    all_results: dict[str, dict] = {}
    layer_names: list[str] | None = None

    for step in steps:
        model_path = args.model_template.format(step=step)
        print(f"\n{'='*60}")
        print(f"Loading checkpoint: {model_path}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(
            model_path,
            device=device,
            dtype=dtype,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            attn_impl=args.attn_impl,
        )
        layers = collect_transformer_blocks(model)
        if layer_names is None:
            layer_names = [layer.name for layer in layers]

        dataset_iterable = build_dataset_iterable(
            args.dataset,
            args.dataset_config or None,
            args.split,
            args.streaming,
        )

        results = run_analysis(
            model_path=model_path,
            model=model,
            layers=layers,
            tokenizer=tokenizer,
            dataset_iterable=dataset_iterable,
            device=device,
            args=args,
        )
        all_results[str(step)] = results

        # Print summary for this step
        print(f"\nStep {step} Summary:")
        print(f"  Tokens processed: {results['tokens_processed']:,}")
        print(f"  Layer-wise mean post-FFN norm: {results['mean_post_ffn_norm'][:5]}...")
        print(f"  Max post-FFN norm by layer: {results['max_post_ffn_norm'][:5]}...")

        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()

    # Compile final output
    output_data = {
        "model_template": args.model_template,
        "steps": steps,
        "layer_names": layer_names,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config or None,
        "split": args.split,
        "streaming": args.streaming,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "padding": args.padding,
        "max_tokens": args.max_tokens,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "attn_impl": args.attn_impl,
        "results_by_step": all_results,
    }

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(json.dumps(output_data, indent=2))

    model_label = args.model_template.replace("{step}", "steps")
    output_path = resolve_output_path(args.output, args.output_dir, model_label, suffix="residual_norm")
    if output_path is not None:
        write_results(output_path, output_data)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
