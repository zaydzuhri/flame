#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from fla.models.dead_head_mask import (
    apply_dead_head_masks_by_layer,
    load_dead_head_masks_by_layer,
)
try:
    from .head_dead_analysis import (
        HeadOffRateAccumulator,
        collect_attention_modules,
        register_o_proj_hooks,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from head_dead_analysis import (  # type: ignore
        HeadOffRateAccumulator,
        collect_attention_modules,
        register_o_proj_hooks,
    )

try:
    from .utils import (
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
except ImportError:  # pragma: no cover - fallback for direct script execution
    from utils import (  # type: ignore
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
class LossAccumulator:
    total_loss: float = 0.0
    total_tokens: int = 0

    def update(self, loss: torch.Tensor, token_count: int) -> None:
        self.total_loss += float(loss.item())
        self.total_tokens += int(token_count)

    def perplexity(self) -> float:
        if self.total_tokens == 0:
            return float("nan")
        return math.exp(self.total_loss / self.total_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline vs dead-head-zeroing perplexity using a dead-head "
            "analysis JSON mask."
        )
    )
    parser.add_argument("--model", required=True, help="HF model name or local path.")
    parser.add_argument(
        "--mask-json",
        type=Path,
        default=None,
        help=(
            "Optional path to head_dead_analysis JSON output. If omitted, auto-mask "
            "derives dead heads from baseline pass."
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "Optional step key in dead_heads_by_step. If omitted, the script "
            "infers from model name/path or uses the only step in mask JSON."
        ),
    )
    parser.add_argument(
        "--mask-mode",
        choices=["dead"],
        default="dead",
        help="Mask mode for model config (default: dead).",
    )
    parser.add_argument(
        "--auto-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically derive dead-head masks from baseline pass when --mask-json "
            "is not provided (default: true)."
        ),
    )
    parser.add_argument(
        "--auto-mask-eps",
        type=float,
        default=1e-6,
        help="Epsilon used to classify per-token head-off during auto-mask derivation.",
    )
    parser.add_argument(
        "--auto-mask-dead-threshold",
        type=float,
        default=0.95,
        help="Off-rate threshold for dead head classification in auto-mask mode.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline run without dead-head masking.",
    )
    parser.add_argument(
        "--skip-masked",
        action="store_true",
        help="Skip masked run with dead-head zeroing enabled.",
    )
    add_dataset_args(parser)
    add_runtime_args(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON/JSONL output path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write <model>_dead_head_zeroing_ppl.jsonl.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.skip_baseline and args.skip_masked:
        raise ValueError("Both --skip-baseline and --skip-masked are set; nothing to run.")
    if args.padding == "none" and args.batch_size != 1:
        raise ValueError("Use --batch-size 1 when --padding none to avoid padding.")
    if args.auto_mask_dead_threshold <= 0 or args.auto_mask_dead_threshold > 1:
        raise ValueError("--auto-mask-dead-threshold must be in (0, 1].")
    if args.auto_mask_eps <= 0:
        raise ValueError("--auto-mask-eps must be positive.")
    if args.mask_json is None and args.step is not None:
        raise ValueError("--step is only valid when --mask-json is provided.")
    if args.mask_json is not None and not args.mask_json.exists():
        raise ValueError(
            f"--mask-json not found: {args.mask_json}. "
            "Omit --mask-json to use default auto-mask derivation."
        )
    if not args.skip_masked and args.mask_json is None and not args.auto_mask:
        raise ValueError(
            "Masked run requires either --mask-json or --auto-mask."
        )


def iter_tokenized_batches(
    dataset_iterable: Iterable[dict],
    tokenizer,
    batch_size: int,
    max_samples: int,
    max_length: int,
    padding: str,
) -> Iterable[dict[str, torch.Tensor]]:
    for batch_texts in iter_batches(
        dataset_iterable,
        batch_size=batch_size,
        max_samples=max_samples,
    ):
        if not batch_texts:
            continue
        yield tokenize_batch(
            tokenizer,
            batch_texts,
            max_length=max_length,
            padding=padding,
        )


def get_dead_head_count(mask_json: Path, step: int) -> int:
    with mask_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    heads_by_step = payload.get("dead_heads_by_step")
    if not isinstance(heads_by_step, dict):
        raise ValueError(
            "Mask JSON does not contain dead_heads_by_step; expected output from head_dead_analysis.py."
        )
    step_key = str(step)
    if step_key not in heads_by_step:
        available = ", ".join(sorted(heads_by_step.keys()))
        raise ValueError(
            f"Step {step} not found in dead_heads_by_step. Available steps: {available or 'none'}."
        )
    entries = heads_by_step[step_key]
    if not isinstance(entries, list):
        raise ValueError(f"dead_heads_by_step[{step_key}] must be a list of [layer, head].")
    return len(entries)


def _available_steps_from_mask(mask_json: Path) -> list[int]:
    with mask_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in ("dead_heads_by_step", "off_rates"):
        obj = payload.get(key)
        if isinstance(obj, dict):
            steps: list[int] = []
            for step_key in obj:
                try:
                    steps.append(int(step_key))
                except (TypeError, ValueError):
                    continue
            return sorted(set(steps))
    raise ValueError("Mask JSON must contain `dead_heads_by_step` or `off_rates`.")


def infer_step_for_model(mask_json: Path, model_ref: str, explicit_step: int | None) -> int:
    available_steps = _available_steps_from_mask(mask_json)
    if not available_steps:
        raise ValueError("No numeric steps found in mask JSON.")
    if explicit_step is not None:
        if explicit_step not in available_steps:
            raise ValueError(
                f"Requested step {explicit_step} not in mask JSON steps: {available_steps}."
            )
        return explicit_step
    if len(available_steps) == 1:
        return available_steps[0]

    # Infer from model string such as "...step-100000" or "...step_100000".
    matched = re.search(r"step[-_](\d+)", model_ref)
    if matched is not None:
        candidate = int(matched.group(1))
        if candidate in available_steps:
            return candidate
    raise ValueError(
        "Could not infer step from model name/path with multiple steps in mask JSON. "
        f"Provide --step explicitly. Available steps: {available_steps}."
    )


def _collect_maskable_attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules: list[torch.nn.Module] = []
    for module in model.modules():
        if (
            hasattr(module, "set_dead_head_mask")
            and hasattr(module, "num_heads")
            and callable(getattr(module, "set_dead_head_mask"))
        ):
            modules.append(module)
    if not modules:
        raise RuntimeError(
            "No attention modules with set_dead_head_mask() found in model."
        )
    if all(
        getattr(module, "layer_idx", -1) is not None
        and getattr(module, "layer_idx", -1) >= 0
        for module in modules
    ):
        modules.sort(key=lambda module: int(getattr(module, "layer_idx")))
    return modules


def masks_from_off_rates(
    off_rates: list[list[float]],
    dead_threshold: float,
) -> list[torch.Tensor]:
    masks: list[torch.Tensor] = []
    for layer_rates in off_rates:
        mask = torch.tensor(
            [float(rate) > dead_threshold for rate in layer_rates],
            dtype=torch.bool,
        )
        masks.append(mask)
    return masks


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    mask_step = None
    mask_source = "none"
    run_baseline = not args.skip_baseline
    run_masked = not args.skip_masked
    if run_masked and args.mask_json is not None:
        mask_step = infer_step_for_model(args.mask_json, args.model, args.step)
        mask_source = "json"
    elif run_masked:
        mask_source = "auto"
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        attn_impl=args.attn_impl,
    )

    masked_dead_head_count = None
    masked_masks_by_layer = None
    zero_masks_by_layer = None
    maskable_modules = None
    if run_masked:
        maskable_modules = _collect_maskable_attention_modules(model)
    if run_masked and args.mask_json is not None:
        masked_dead_head_count = get_dead_head_count(args.mask_json, mask_step)
        num_heads_by_layer = [int(getattr(module, "num_heads")) for module in maskable_modules]
        masked_masks_by_layer = load_dead_head_masks_by_layer(
            mask_path=args.mask_json,
            step=mask_step,
            num_heads_by_layer=num_heads_by_layer,
            mode=args.mask_mode,
        )
        zero_masks_by_layer = [torch.zeros_like(mask) for mask in masked_masks_by_layer]
        total_masked = apply_dead_head_masks_by_layer(model, masked_masks_by_layer)
        if total_masked <= 0:
            raise RuntimeError(
                f"Applied dead-head mask for step {mask_step}, but zero heads were masked."
            )

    dataset_iterable, n_samples = build_dataset_iterable(
        args.dataset,
        args.dataset_config,
        args.split,
        args.streaming,
        args.n_samples,
    )

    baseline_acc = LossAccumulator()
    masked_acc = LossAccumulator()
    cached_batches: list[dict[str, torch.Tensor]] = []
    auto_accumulator = None
    auto_hooks = None
    if run_masked and args.mask_json is None:
        num_heads_by_layer = [int(getattr(module, "num_heads")) for module in maskable_modules]
        auto_accumulator = HeadOffRateAccumulator(args.auto_mask_eps, num_heads_by_layer)
        layers = collect_attention_modules(model)
        if len(layers) != len(maskable_modules):
            raise RuntimeError(
                "Mismatch between attention modules discovered for masking and for hook collection."
            )
        auto_hooks = register_o_proj_hooks(layers, auto_accumulator)

    run_reference_pass = run_baseline or (run_masked and args.mask_json is None)
    for encodings in tqdm(
        iter_tokenized_batches(
            dataset_iterable,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_samples=n_samples,
            max_length=args.max_length,
            padding=args.padding,
        ),
        desc="Batches",
        unit="batch",
    ):
        # Cache a CPU copy so masked run can replay exact same data.
        cached_batches.append(
            {
                "input_ids": encodings["input_ids"].clone(),
                "attention_mask": encodings.get("attention_mask").clone()
                if encodings.get("attention_mask") is not None
                else None,
            }
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if input_ids.shape[1] < 2:
            continue
        token_count = int((input_ids.shape[1] - 1) * input_ids.shape[0])

        if run_reference_pass:
            if run_masked and args.mask_json is not None:
                apply_dead_head_masks_by_layer(model, zero_masks_by_layer)
            if auto_accumulator is not None:
                auto_accumulator.set_attention_mask(attention_mask)
            with torch.inference_mode():
                baseline_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            if auto_accumulator is not None:
                auto_accumulator.set_attention_mask(None)
            if run_baseline:
                shift_logits = baseline_outputs.logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                baseline_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                baseline_acc.update(baseline_loss, token_count)

        if run_masked and args.mask_json is not None:
            apply_dead_head_masks_by_layer(model, masked_masks_by_layer)
            with torch.inference_mode():
                masked_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            shift_logits = masked_outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            masked_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            masked_acc.update(masked_loss, token_count)

    if auto_hooks is not None:
        for handle in auto_hooks:
            handle.remove()

    if run_masked and args.mask_json is None:
        if auto_accumulator is None:
            raise RuntimeError("Auto-mask accumulator was not initialized.")
        masked_masks_by_layer = masks_from_off_rates(
            auto_accumulator.off_rates(),
            dead_threshold=args.auto_mask_dead_threshold,
        )
        zero_masks_by_layer = [torch.zeros_like(mask) for mask in masked_masks_by_layer]
        masked_dead_head_count = sum(int(mask.sum().item()) for mask in masked_masks_by_layer)

        for batch in tqdm(cached_batches, desc="Masked Batches", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"]
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if input_ids.shape[1] < 2:
                continue
            token_count = int((input_ids.shape[1] - 1) * input_ids.shape[0])
            apply_dead_head_masks_by_layer(model, masked_masks_by_layer)
            with torch.inference_mode():
                masked_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            shift_logits = masked_outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            masked_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            masked_acc.update(masked_loss, token_count)

    baseline_ppl = None
    baseline_tokens = None
    if run_baseline:
        if baseline_acc.total_tokens == 0:
            raise RuntimeError(
                "No baseline tokens were processed. Increase --n-samples or check dataset text."
            )
        baseline_ppl = baseline_acc.perplexity()
        baseline_tokens = baseline_acc.total_tokens
        print(f"Baseline PPL: {baseline_ppl:.6f} over {baseline_tokens} tokens")

    masked_ppl = None
    masked_tokens = None
    if run_masked:
        if masked_acc.total_tokens == 0:
            raise RuntimeError(
                "No masked tokens were processed. Increase --n-samples or check dataset text."
            )
        masked_ppl = masked_acc.perplexity()
        masked_tokens = masked_acc.total_tokens
        print(
            f"Masked PPL: {masked_ppl:.6f} over {masked_tokens} tokens "
            f"(dead_heads={masked_dead_head_count})"
        )

    ppl_delta = None
    if baseline_ppl is not None and masked_ppl is not None:
        ppl_delta = masked_ppl - baseline_ppl
        print(f"Delta PPL (masked - baseline): {ppl_delta:+.6f}")

    results = {
        "analysis_type": "dead_head_zeroing_ppl",
        "model": args.model,
        "mask_json": str(args.mask_json) if args.mask_json is not None else None,
        "step": mask_step,
        "mask_mode": args.mask_mode,
        "mask_source": mask_source,
        "auto_mask": args.auto_mask,
        "auto_mask_eps": args.auto_mask_eps,
        "auto_mask_dead_threshold": args.auto_mask_dead_threshold,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "streaming": args.streaming,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "padding": args.padding,
        "attn_impl": args.attn_impl,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "baseline": (
            {"ppl": baseline_ppl, "tokens": baseline_tokens}
            if baseline_ppl is not None
            else None
        ),
        "masked": (
            {
                "ppl": masked_ppl,
                "tokens": masked_tokens,
                "dead_head_count": masked_dead_head_count,
            }
            if masked_ppl is not None
            else None
        ),
        "delta_ppl_masked_minus_baseline": ppl_delta,
    }

    print(json.dumps(results, indent=2, sort_keys=True))
    model_label = args.model.replace("/", "_")
    output_path = resolve_output_path(
        args.output,
        args.output_dir,
        model_label,
        suffix="dead_head_zeroing_ppl",
    )
    if output_path is not None:
        write_results(output_path, results)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
