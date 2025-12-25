#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # ensure custom modeling is registered

DEFAULT_PROMPT = (
    "According to all known laws of aviation, there is no way a bee should be able to"
)

DTYPE_MAP = {
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Choose from: {', '.join(DTYPE_MAP)}"
        )
    return DTYPE_MAP[dtype_name]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    return torch.device(device_name)


def validate_device_dtype(device: torch.device, dtype: torch.dtype) -> None:
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"{dtype} on CPU is not supported. Use --dtype float32 or --device cuda."
        )


def sanitize_model_name(model_name: str) -> str:
    safe = model_name.strip().replace("/", "_").replace(" ", "_")
    return safe or "model"


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
    attn_impl: str | None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    config = None
    if attn_impl is not None:
        config = AutoConfig.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        if not hasattr(config, "attn_impl"):
            raise ValueError(
                "Config does not define attn_impl; remove --attn-impl or use a "
                "compatible model config."
            )
        config.attn_impl = attn_impl

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        config=config,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def build_inputs(
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    use_attention_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = None
    if use_attention_mask and hasattr(inputs, "attention_mask"):
        attention_mask = inputs.attention_mask.to(device)
    return input_ids, attention_mask


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    use_cache: bool,
) -> str:
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            use_cache=use_cache,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def run_forward(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    use_cache: bool,
    output_hidden_states: bool,
    output_attentions: bool,
):
    with torch.inference_mode():
        return model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )


def print_hidden_state_stats(hidden_states: Iterable[torch.Tensor]) -> None:
    for idx, states in enumerate(hidden_states):
        print(
            "hidden states",
            idx,
            tuple(states.shape),
            "min:",
            states.min().item(),
            "max:",
            states.max().item(),
            "mean:",
            states.mean().item(),
            "std:",
            states.std().item(),
        )


def print_param_stats(model: AutoModelForCausalLM) -> None:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        print(
            name,
            tuple(param.shape),
            "min:",
            param.min().item(),
            "max:",
            param.max().item(),
            "mean:",
            param.mean().item(),
            "std:",
            param.std().item(),
        )


def plot_hidden_state_histogram(
    hidden_states: Iterable[torch.Tensor],
    output_path: Path | None,
    show: bool,
) -> None:
    if output_path is None and not show:
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for idx, states in enumerate(hidden_states):
        plt.hist(
            states.cpu().detach().numpy().flatten(),
            bins=100,
            alpha=0.5,
            label=f"Layer {idx}",
        )
    plt.title("Histogram of hidden states")
    plt.xlabel("Hidden state value")
    plt.ylabel("Frequency")
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_attention_grid(
    attentions: Iterable[torch.Tensor],
    head_index: int,
    layer_columns: int,
    output_path: Path | None,
    show: bool,
    layer_indices: Iterable[int] | None = None,
) -> None:
    if output_path is None and not show:
        return
    import matplotlib.pyplot as plt

    attentions = list(attentions)
    if not attentions:
        raise ValueError("No attentions available to plot.")
    if head_index < 0 or head_index >= attentions[0].shape[1]:
        raise ValueError(
            f"head_index {head_index} is out of range for {attentions[0].shape[1]} heads."
        )

    num_layers = len(attentions)
    if layer_indices is None:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = list(layer_indices)
        if len(layer_indices) != num_layers:
            raise ValueError("layer_indices length must match attentions length.")
    cols = max(1, layer_columns)
    rows = math.ceil(num_layers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, attn in enumerate(attentions):
        ax = axes_list[idx]
        ax.imshow(attn[0][head_index].cpu().detach().numpy(), cmap="pink")
        ax.set_title(f"Layer {layer_indices[idx]}")
        fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_list[num_layers:]:
        ax.axis("off")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_weight_histogram(
    model: AutoModelForCausalLM,
    output_path: Path | None,
    show: bool,
) -> None:
    if output_path is None and not show:
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        plt.hist(
            param.cpu().detach().numpy().flatten(),
            bins=100,
            alpha=0.5,
            label=name,
        )
    plt.title("Histogram of model weights")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def ensure_plot_output(
    save_dir: Path | None,
    model_name: str,
    suffix: str,
    save_plots: bool,
) -> Path | None:
    if not save_plots:
        return None
    if save_dir is None:
        raise ValueError("save_dir is required when save_plots is enabled.")
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"{sanitize_model_name(model_name)}_{suffix}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference and attention/hidden-state inspection."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or local path (e.g., a save_pretrained output directory).",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text.")
    parser.add_argument(
        "--device", default="cuda", help="Device to run on (cuda or cpu)."
    )
    parser.add_argument(
        "--dtype", default="float16", choices=sorted(DTYPE_MAP), help="Model dtype."
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid remote downloads in from_pretrained.",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default=None,
        help="Override config.attn_impl (e.g., naive_attn) for attention outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=("generate", "forward", "both"),
        default="forward",
        help="Run text generation, forward analysis, or both.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample during generation.",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use KV cache during generation/forward.",
    )
    parser.add_argument(
        "--use-attention-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use tokenizer attention mask instead of None.",
    )
    parser.add_argument(
        "--print-hidden-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print min/max/mean/std for each hidden state layer.",
    )
    parser.add_argument(
        "--print-param-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print min/max/mean/std for each model parameter.",
    )
    parser.add_argument(
        "--plot-hidden-hist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot hidden state histogram.",
    )
    parser.add_argument(
        "--plot-attn-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot attention heatmap grid.",
    )
    parser.add_argument(
        "--plot-weight-hist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot model weight histogram.",
    )
    parser.add_argument(
        "--head-index",
        type=int,
        default=1,
        help="Attention head index for heatmap plots.",
    )
    parser.add_argument(
        "--layer-columns",
        type=int,
        default=6,
        help="Number of columns in the attention grid.",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save plots to disk.",
    )
    parser.add_argument(
        "--show-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display plots interactively.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("analysis/attention_sink/outputs"),
        help="Directory for saved plots.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    validate_device_dtype(device, dtype)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
        attn_impl=args.attn_impl,
    )
    input_ids, attention_mask = build_inputs(
        tokenizer,
        args.prompt,
        device=device,
        use_attention_mask=args.use_attention_mask,
    )

    print("len input ids", len(input_ids[0]))

    if args.mode in ("generate", "both"):
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            use_cache=args.use_cache,
        )
        print(text)

    if args.mode in ("forward", "both"):
        output = run_forward(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=args.use_cache,
            output_hidden_states=True,
            output_attentions=True,
        )

        if output.attentions is None or not any(
            attn is not None for attn in output.attentions
        ):
            raise RuntimeError(
                "Model returned no attentions. Use --attn-impl naive_attn (or another "
                "naive_* impl) to enable attention outputs."
            )

        if args.print_hidden_stats and output.hidden_states is not None:
            print_hidden_state_stats(output.hidden_states)

        if args.print_param_stats:
            print_param_stats(model)

        if (
            args.plot_hidden_hist
            or args.plot_attn_grid
            or args.plot_weight_hist
        ):
            if not args.save_plots and not args.show_plots:
                raise ValueError(
                    "Plotting requested but both --save-plots and --show-plots are off."
                )

        if args.plot_hidden_hist and output.hidden_states is not None:
            output_path = ensure_plot_output(
                args.save_dir, args.model, "hidden_state_hist", args.save_plots
            )
            plot_hidden_state_histogram(
                output.hidden_states, output_path=output_path, show=args.show_plots
            )

        if args.plot_attn_grid:
            if any(attn is None for attn in output.attentions):
                raise RuntimeError(
                    "Missing attentions for one or more layers. Use --attn-impl "
                    "naive_attn (or another naive_* impl) for full attention outputs."
                )
            output_path = ensure_plot_output(
                args.save_dir, args.model, "attn_grid", args.save_plots
            )
            plot_attention_grid(
                output.attentions,
                head_index=args.head_index,
                layer_columns=args.layer_columns,
                output_path=output_path,
                show=args.show_plots,
            )

        if args.plot_weight_hist:
            output_path = ensure_plot_output(
                args.save_dir, args.model, "weight_hist", args.save_plots
            )
            plot_weight_histogram(model, output_path=output_path, show=args.show_plots)


if __name__ == "__main__":
    main()
