#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

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
        add_runtime_args,
        build_dataset_iterable,
        extract_text,
        load_model_and_tokenizer,
        resolve_device,
        resolve_dtype,
        resolve_output_path,
        write_results,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from utils import (  # type: ignore
        add_runtime_args,
        build_dataset_iterable,
        extract_text,
        load_model_and_tokenizer,
        resolve_device,
        resolve_dtype,
        resolve_output_path,
        write_results,
    )


@dataclass
class AccuracyAccumulator:
    correct: int = 0
    total: int = 0

    def update(self, is_correct: bool) -> None:
        self.correct += int(is_correct)
        self.total += 1

    def accuracy(self) -> float:
        if self.total == 0:
            return float("nan")
        return float(self.correct) / float(self.total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline vs dead-head-zeroing exact-match accuracy on "
            "LAMBADA-style last-word prediction."
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
        help="Mask mode for dead-head mask loading (default: dead).",
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
    parser.add_argument("--dataset", default="lambada", help="Dataset name (default: lambada).")
    parser.add_argument(
        "--dataset-config",
        default="plain_text",
        help="Dataset config name (default: plain_text).",
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=-1,
        help="Number of non-empty samples to evaluate (-1 for all).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset mode.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum tokenized length for each sample.",
    )
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
        help="Directory to write <model>_dead_head_zeroing_lambada.jsonl.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.skip_baseline and args.skip_masked:
        raise ValueError("Both --skip-baseline and --skip-masked are set; nothing to run.")
    if args.max_length < 2:
        raise ValueError("--max-length must be >= 2.")
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
    matched = re.search(r"step[-_](\d+)", model_ref)
    if matched is not None:
        candidate = int(matched.group(1))
        if candidate in available_steps:
            return candidate
    raise ValueError(
        "Could not infer step from model name/path with multiple steps in mask JSON. "
        f"Provide --step explicitly. Available steps: {available_steps}."
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


def split_prompt_target(text: str) -> tuple[str, str] | None:
    normalized = text.strip()
    if not normalized:
        return None
    parts = normalized.rsplit(maxsplit=1)
    if len(parts) != 2:
        return None
    prompt, target_word = parts
    if not prompt.strip() or not target_word.strip():
        return None
    return prompt, target_word


def prepare_lambada_sample(
    tokenizer,
    text: str,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    split = split_prompt_target(text)
    if split is None:
        return None
    prompt, target_word = split
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(f" {target_word}", add_special_tokens=False)["input_ids"]
    if not prompt_ids or not target_ids:
        return None
    if len(prompt_ids) + len(target_ids) > max_length:
        return None
    full_ids = prompt_ids + target_ids
    if len(full_ids) < 2:
        return None
    input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
    target_tensor = torch.tensor(target_ids, dtype=torch.long)
    return input_ids, target_tensor


def evaluate_last_word_exact_match(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    device: torch.device,
) -> bool:
    model_input = input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(model_input, dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(
            input_ids=model_input,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    logits = outputs.logits[0]
    preds = logits.argmax(dim=-1)
    target_len = int(target_ids.numel())
    pred_target = preds[-target_len:]
    return bool(torch.equal(pred_target, target_ids.to(device=pred_target.device)))


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    run_baseline = not args.skip_baseline
    run_masked = not args.skip_masked
    mask_step = None
    mask_source = "none"
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

    dataset_iterable, sample_limit = build_dataset_iterable(
        args.dataset,
        args.dataset_config,
        args.split,
        args.streaming,
        args.n_samples,
    )

    baseline_acc = AccuracyAccumulator()
    masked_acc = AccuracyAccumulator()
    cached_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    skipped_samples = 0
    seen_nonempty = 0
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

    progress = tqdm(dataset_iterable, desc="Samples", unit="sample")
    for sample in progress:
        if sample_limit is not None and sample_limit >= 0 and seen_nonempty >= sample_limit:
            break
        text = extract_text(sample)
        if text is None:
            continue
        seen_nonempty += 1
        prepared = prepare_lambada_sample(tokenizer, text, args.max_length)
        if prepared is None:
            skipped_samples += 1
            continue
        input_ids_cpu, target_ids_cpu = prepared
        if run_masked and args.mask_json is None:
            cached_samples.append((input_ids_cpu.clone(), target_ids_cpu.clone()))

        if run_baseline or (run_masked and args.mask_json is None):
            if run_masked and args.mask_json is not None:
                apply_dead_head_masks_by_layer(model, zero_masks_by_layer)
            if auto_accumulator is not None:
                attention_mask = torch.ones(
                    (1, int(input_ids_cpu.numel())),
                    dtype=torch.long,
                    device=device,
                )
                auto_accumulator.set_attention_mask(attention_mask)
            is_correct = evaluate_last_word_exact_match(
                model=model,
                input_ids=input_ids_cpu,
                target_ids=target_ids_cpu,
                device=device,
            )
            if auto_accumulator is not None:
                auto_accumulator.set_attention_mask(None)
            if run_baseline:
                baseline_acc.update(is_correct)

        if run_masked and args.mask_json is not None:
            apply_dead_head_masks_by_layer(model, masked_masks_by_layer)
            is_correct = evaluate_last_word_exact_match(
                model=model,
                input_ids=input_ids_cpu,
                target_ids=target_ids_cpu,
                device=device,
            )
            masked_acc.update(is_correct)

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
        masked_dead_head_count = sum(int(mask.sum().item()) for mask in masked_masks_by_layer)
        for input_ids_cpu, target_ids_cpu in tqdm(
            cached_samples,
            desc="Masked Samples",
            unit="sample",
        ):
            apply_dead_head_masks_by_layer(model, masked_masks_by_layer)
            is_correct = evaluate_last_word_exact_match(
                model=model,
                input_ids=input_ids_cpu,
                target_ids=target_ids_cpu,
                device=device,
            )
            masked_acc.update(is_correct)

    baseline_accuracy = None
    if run_baseline:
        if baseline_acc.total == 0:
            raise RuntimeError(
                "No baseline samples were processed. Increase --n-samples or check dataset settings."
            )
        baseline_accuracy = baseline_acc.accuracy()
        print(
            f"Baseline accuracy: {baseline_accuracy:.6f} "
            f"({baseline_acc.correct}/{baseline_acc.total})"
        )

    masked_accuracy = None
    if run_masked:
        if masked_acc.total == 0:
            raise RuntimeError(
                "No masked samples were processed. Increase --n-samples or check dataset settings."
            )
        masked_accuracy = masked_acc.accuracy()
        print(
            f"Masked accuracy: {masked_accuracy:.6f} "
            f"({masked_acc.correct}/{masked_acc.total}, dead_heads={masked_dead_head_count})"
        )

    delta_accuracy = None
    if baseline_accuracy is not None and masked_accuracy is not None:
        delta_accuracy = masked_accuracy - baseline_accuracy
        print(f"Delta accuracy (masked - baseline): {delta_accuracy:+.6f}")

    results = {
        "analysis_type": "dead_head_zeroing_lambada",
        "metric": "exact_match_last_word",
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
        "max_length": args.max_length,
        "attn_impl": args.attn_impl,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "seen_nonempty_samples": seen_nonempty,
        "skipped_samples": skipped_samples,
        "baseline": (
            {
                "accuracy": baseline_accuracy,
                "correct": baseline_acc.correct,
                "total": baseline_acc.total,
            }
            if baseline_accuracy is not None
            else None
        ),
        "masked": (
            {
                "accuracy": masked_accuracy,
                "correct": masked_acc.correct,
                "total": masked_acc.total,
                "dead_head_count": masked_dead_head_count,
            }
            if masked_accuracy is not None
            else None
        ),
        "delta_accuracy_masked_minus_baseline": delta_accuracy,
    }

    print(json.dumps(results, indent=2, sort_keys=True))
    model_label = args.model.replace("/", "_")
    output_path = resolve_output_path(
        args.output,
        args.output_dir,
        model_label,
        suffix="dead_head_zeroing_lambada",
    )
    if output_path is not None:
        write_results(output_path, results)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
