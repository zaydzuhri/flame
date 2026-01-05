#!/usr/bin/env python3
"""
Sink Token Redundancy Analysis

Compares attention behavior between:
- Vanilla Softmax: Forces attention sum = 1.0
- GPT-OSS Sink: Explicit sink tokens absorb garbage attention
- Softpick: Natural no-op attention (sum < 1.0) without sink tokens

Key hypothesis: Softpick makes explicit sink tokens redundant by allowing
true "no-op" attention when queries have no relevant keys.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # ensure custom modeling is registered

# Default model paths
DEFAULT_MODELS = {
    "vanilla": "./analysis/attention_sink/hf_models/vanilla-340M-4096-step-10000",
    "softpick": "./analysis/attention_sink/hf_models/softpick-340M-4096-step-10000",
    "softmax_sink": "./analysis/attention_sink/hf_models/gpt_oss_sink-340M-4096-step-10000",
}

# Attention implementation overrides for extracting weights
ATTN_IMPL_MAP = {
    "vanilla": "naive_attn",
    "softpick": "naive_softpick_attn",
    "softmax_sink": "gpt_oss_naive_sink",
}

# Synthetic garbage queries for testing
GARBAGE_QUERIES = {
    "padding_like": "___",
    "bos_first": "The",
    "sentence_boundary": ". The quick brown",
    "whitespace": "   ",
    "newlines": "\n\n",
    "random_symbols": "### @@ %%",
}

# Visualization settings
COLORMAP = "pink"
ZERO_COLOR = "#D0AC93"
BIN_TARGET = 64


@dataclass
class AttentionMetrics:
    """Per-layer attention metrics."""
    entropy: list[list[float]] = field(default_factory=list)  # [layer][head]
    attention_sum: list[list[float]] = field(default_factory=list)  # [layer][head]
    output_magnitude: list[list[float]] = field(default_factory=list)  # [layer][head]


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_name: str
    query_text: str
    num_tokens: int
    metrics: AttentionMetrics


@dataclass
class ModelResults:
    """Aggregated results for a single model."""
    model_name: str
    model_type: str
    num_layers: int
    num_heads: int
    per_query: list[QueryMetrics] = field(default_factory=list)
    aggregated: Optional[AttentionMetrics] = None


def load_model_for_analysis(
    model_path: str,
    model_type: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with attention-weight-returning implementation."""
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.output_attentions = True
    config.output_hidden_states = True

    # Override attention implementation to naive variant
    attn_impl = ATTN_IMPL_MAP.get(model_type)
    if attn_impl and hasattr(config, "attn_impl"):
        config.attn_impl = attn_impl

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def compute_entropy(attention_weights: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute entropy of attention distribution: H = -sum(p * log(p))

    Args:
        attention_weights: Shape [batch, heads, query_len, key_len]

    Returns:
        Entropy per query position: [batch, heads, query_len]
    """
    p = attention_weights.clamp(min=eps)
    log_p = torch.log(p)
    # Zero out log for very small values
    log_p = torch.where(attention_weights > eps, log_p, torch.zeros_like(log_p))
    entropy = -(p * log_p).sum(dim=-1)
    return entropy


def compute_attention_sum(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute sum of attention weights.

    For Softmax: always ~1.0
    For Softpick: can be < 1.0 (key property!)
    For GPT-OSS Sink: < 1.0 when sink absorbs (sink column excluded)

    Returns: [batch, heads, query_len]
    """
    return attention_weights.sum(dim=-1)


def analyze_single_query(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query_name: str,
    query_text: str,
    device: torch.device,
) -> QueryMetrics:
    """Run analysis on a single query."""
    inputs = tokenizer(query_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    num_tokens = input_ids.shape[1]

    with torch.inference_mode():
        outputs = model(
            input_ids,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = outputs.attentions  # tuple of [B, H, T, T] per layer

    layer_entropy = []
    layer_attn_sum = []
    layer_output_mag = []

    for layer_idx, layer_attn in enumerate(attentions):
        if layer_attn is None:
            continue
        # layer_attn: [1, num_heads, seq_len, seq_len]
        entropy = compute_entropy(layer_attn)
        attn_sum = compute_attention_sum(layer_attn)

        # Mean over batch and query positions, keep per-head
        # Shape: [num_heads]
        layer_entropy.append(entropy.mean(dim=(0, 2)).tolist())
        layer_attn_sum.append(attn_sum.mean(dim=(0, 2)).tolist())

        # Output magnitude placeholder (would need hooks for full implementation)
        layer_output_mag.append([0.0] * layer_attn.shape[1])

    return QueryMetrics(
        query_name=query_name,
        query_text=query_text,
        num_tokens=num_tokens,
        metrics=AttentionMetrics(
            entropy=layer_entropy,
            attention_sum=layer_attn_sum,
            output_magnitude=layer_output_mag,
        ),
    )


def aggregate_metrics(per_query: list[QueryMetrics]) -> AttentionMetrics:
    """Aggregate metrics across all queries."""
    if not per_query:
        return AttentionMetrics()

    num_layers = len(per_query[0].metrics.entropy)
    num_heads = len(per_query[0].metrics.entropy[0]) if num_layers > 0 else 0

    agg_entropy = []
    agg_attn_sum = []
    agg_output_mag = []

    for layer_idx in range(num_layers):
        layer_entropy = []
        layer_attn_sum = []
        layer_output_mag = []

        for head_idx in range(num_heads):
            head_entropies = [
                q.metrics.entropy[layer_idx][head_idx]
                for q in per_query
                if layer_idx < len(q.metrics.entropy)
                and head_idx < len(q.metrics.entropy[layer_idx])
            ]
            head_sums = [
                q.metrics.attention_sum[layer_idx][head_idx]
                for q in per_query
                if layer_idx < len(q.metrics.attention_sum)
                and head_idx < len(q.metrics.attention_sum[layer_idx])
            ]
            head_mags = [
                q.metrics.output_magnitude[layer_idx][head_idx]
                for q in per_query
                if layer_idx < len(q.metrics.output_magnitude)
                and head_idx < len(q.metrics.output_magnitude[layer_idx])
            ]

            layer_entropy.append(float(np.mean(head_entropies)) if head_entropies else 0.0)
            layer_attn_sum.append(float(np.mean(head_sums)) if head_sums else 0.0)
            layer_output_mag.append(float(np.mean(head_mags)) if head_mags else 0.0)

        agg_entropy.append(layer_entropy)
        agg_attn_sum.append(layer_attn_sum)
        agg_output_mag.append(layer_output_mag)

    return AttentionMetrics(
        entropy=agg_entropy,
        attention_sum=agg_attn_sum,
        output_magnitude=agg_output_mag,
    )


def analyze_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_type: str,
    queries: dict[str, str],
    device: torch.device,
) -> ModelResults:
    """Run full analysis on a model across all queries."""
    per_query = []
    for query_name, query_text in queries.items():
        metrics = analyze_single_query(model, tokenizer, query_name, query_text, device)
        per_query.append(metrics)
        print(f"    {query_name}: {metrics.num_tokens} tokens")

    aggregated = aggregate_metrics(per_query)

    num_layers = len(aggregated.entropy)
    num_heads = len(aggregated.entropy[0]) if num_layers > 0 else 0

    return ModelResults(
        model_name=model.config._name_or_path,
        model_type=model_type,
        num_layers=num_layers,
        num_heads=num_heads,
        per_query=per_query,
        aggregated=aggregated,
    )


def plot_entropy_comparison(
    results: dict[str, ModelResults],
    output_path: Path,
) -> None:
    """Create bar chart comparing entropy and attention sum across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    model_types = list(results.keys())
    model_colors = {"vanilla": "#1f77b4", "softpick": "#2ca02c", "softmax_sink": "#ff7f0e"}

    # Left: Per-layer mean entropy
    ax1 = axes[0]
    for model_type in model_types:
        res = results[model_type]
        if res.aggregated is None:
            continue
        # Mean entropy across heads for each layer
        layer_means = [np.mean(layer) for layer in res.aggregated.entropy]
        layers = list(range(len(layer_means)))
        ax1.plot(
            layers,
            layer_means,
            marker="o",
            markersize=3,
            label=model_type,
            color=model_colors.get(model_type, None),
        )
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Entropy (nats)")
    ax1.set_title("Attention Entropy by Layer")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Per-layer mean attention sum
    ax2 = axes[1]
    for model_type in model_types:
        res = results[model_type]
        if res.aggregated is None:
            continue
        layer_means = [np.mean(layer) for layer in res.aggregated.attention_sum]
        layers = list(range(len(layer_means)))
        ax2.plot(
            layers,
            layer_means,
            marker="o",
            markersize=3,
            label=model_type,
            color=model_colors.get(model_type, None),
        )
    ax2.axhline(y=1.0, linestyle="--", color="gray", alpha=0.7, label="Full attention")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Attention Sum")
    ax2.set_title("Attention Weight Sum by Layer (< 1.0 = No-Op Possible)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_attention_sum_distribution(
    results: dict[str, ModelResults],
    output_path: Path,
) -> None:
    """Create histogram showing distribution of attention sums."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_colors = {"vanilla": "#1f77b4", "softpick": "#2ca02c", "softmax_sink": "#ff7f0e"}

    for model_type, res in results.items():
        if res.aggregated is None:
            continue
        # Flatten all attention sums
        all_sums = []
        for layer in res.aggregated.attention_sum:
            all_sums.extend(layer)

        ax.hist(
            all_sums,
            bins=50,
            alpha=0.5,
            label=f"{model_type} (mean={np.mean(all_sums):.4f})",
            color=model_colors.get(model_type, None),
        )

    ax.axvline(x=1.0, linestyle="--", color="gray", alpha=0.7, label="Full attention")
    ax.set_xlabel("Attention Sum")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Attention Weight Sums Across All Layers/Heads")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def bin_attention(attn: np.ndarray, bin_size: int) -> np.ndarray:
    """Bin attention matrix to reduce size for visualization."""
    if bin_size <= 1:
        return attn
    size = attn.shape[0] // bin_size * bin_size
    if size <= 0:
        return attn
    attn = attn[:size, :size]
    attn = attn.reshape(size // bin_size, bin_size, size // bin_size, bin_size)
    return attn.mean(axis=(1, 3))


def attention_norm(attn: np.ndarray) -> colors.PowerNorm:
    """Create power norm for attention visualization."""
    vmax = float(np.quantile(attn, 0.995))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(attn)) if np.max(attn) > 0 else 1.0
    return colors.PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax)


def zero_colormap(name: str, zero_color: str) -> colors.Colormap:
    """Create colormap with custom zero color."""
    base = plt.get_cmap(name)
    sampled = base(np.linspace(0, 1, 256))
    sampled[0] = colors.to_rgba(zero_color)
    return colors.ListedColormap(sampled)


def collect_raw_attentions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query_text: str,
    device: torch.device,
) -> list[torch.Tensor]:
    """Collect raw attention tensors for a query."""
    inputs = tokenizer(query_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    return [attn for attn in outputs.attentions if attn is not None]


def plot_attention_heatmaps(
    models: dict[str, tuple[AutoModelForCausalLM, AutoTokenizer]],
    query_text: str,
    layer_head_pairs: list[tuple[int, int]],
    device: torch.device,
    output_path: Path,
) -> None:
    """Create side-by-side attention heatmaps for selected layer/head pairs."""
    model_types = list(models.keys())
    num_pairs = len(layer_head_pairs)
    num_models = len(model_types)

    fig, axes = plt.subplots(
        nrows=num_pairs,
        ncols=num_models,
        figsize=(3.5 * num_models, 3.5 * num_pairs),
    )

    if num_pairs == 1:
        axes = [axes]
    if num_models == 1:
        axes = [[ax] for ax in axes]

    # Collect attentions for each model
    all_attentions = {}
    for model_type, (model, tokenizer) in models.items():
        all_attentions[model_type] = collect_raw_attentions(
            model, tokenizer, query_text, device
        )

    for pair_idx, (layer_idx, head_idx) in enumerate(layer_head_pairs):
        for model_idx, model_type in enumerate(model_types):
            ax = axes[pair_idx][model_idx]
            attns = all_attentions[model_type]

            if layer_idx >= len(attns):
                ax.set_title(f"{model_type}\nL{layer_idx} H{head_idx}\n(N/A)")
                ax.axis("off")
                continue

            raw_attn = attns[layer_idx][0, head_idx].cpu().numpy()
            bin_size = max(1, raw_attn.shape[0] // BIN_TARGET)
            attn = bin_attention(raw_attn, bin_size)

            attn_sum = float(raw_attn.sum(axis=-1).mean())

            if np.max(attn) == 0:
                im = ax.imshow(
                    attn,
                    cmap=zero_colormap(COLORMAP, ZERO_COLOR),
                    interpolation="nearest",
                    norm=colors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0),
                )
            else:
                im = ax.imshow(
                    attn,
                    cmap=COLORMAP,
                    interpolation="nearest",
                    norm=attention_norm(attn),
                )

            ax.set_title(f"{model_type}\nL{layer_idx} H{head_idx}\nsum={attn_sum:.3f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Query: "{query_text[:50]}..."' if len(query_text) > 50 else f'Query: "{query_text}"')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_markdown_report(
    results: dict[str, ModelResults],
    output_path: Path,
) -> None:
    """Generate sink_results.md with interpretation."""
    lines = [
        "# Sink Token Redundancy Analysis Results",
        "",
        "## Summary",
        "",
        "This analysis compares attention behavior between:",
        "1. **Vanilla Softmax**: Standard softmax attention (sum = 1.0)",
        "2. **Softpick**: Sparse attention that can produce sum < 1.0",
        "3. **GPT-OSS Sink**: Softmax with learned sink tokens (sink absorbs garbage)",
        "",
        "## Key Hypothesis",
        "",
        "Softpick makes explicit sink tokens redundant by allowing true 'no-op'",
        "attention when queries have no relevant keys (attention weights sum < 1.0).",
        "",
        "## Results",
        "",
        "### Mean Attention Sum by Model",
        "",
        "| Model | Mean Attention Sum | Interpretation |",
        "|-------|-------------------|----------------|",
    ]

    for model_type, res in results.items():
        if res.aggregated is None:
            continue
        all_sums = []
        for layer in res.aggregated.attention_sum:
            all_sums.extend(layer)
        mean_sum = np.mean(all_sums)

        if model_type == "vanilla":
            interp = "Forced to attend (no no-op)"
        elif model_type == "softpick":
            interp = "Natural no-op capability" if mean_sum < 0.99 else "Similar to softmax"
        else:
            interp = "Sink absorbs garbage attention"

        lines.append(f"| {model_type} | {mean_sum:.4f} | {interp} |")

    lines.extend([
        "",
        "### Mean Attention Entropy by Model",
        "",
        "| Model | Mean Entropy (nats) |",
        "|-------|---------------------|",
    ])

    for model_type, res in results.items():
        if res.aggregated is None:
            continue
        all_entropy = []
        for layer in res.aggregated.entropy:
            all_entropy.extend(layer)
        mean_entropy = np.mean(all_entropy)
        lines.append(f"| {model_type} | {mean_entropy:.4f} |")

    # Per-layer details
    lines.extend([
        "",
        "## Per-Layer Analysis",
        "",
        "### Attention Sum Deficit (1.0 - sum)",
        "",
        "Higher values indicate more 'leaked' attention (true no-op behavior).",
        "",
    ])

    for model_type, res in results.items():
        if res.aggregated is None:
            continue
        lines.append(f"**{model_type}**:")
        layer_deficits = [1.0 - np.mean(layer) for layer in res.aggregated.attention_sum]
        for i, deficit in enumerate(layer_deficits):
            if deficit > 0.001:  # Only show significant deficits
                lines.append(f"- Layer {i}: {deficit:.4f}")
        if all(d <= 0.001 for d in layer_deficits):
            lines.append("- (No significant attention leakage)")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "If Softpick shows attention sums significantly below 1.0 on garbage queries,",
        "it confirms that Softpick naturally achieves 'no-op' behavior that GPT-OSS",
        "requires explicit sink tokens to accomplish.",
        "",
        "---",
        "*Generated by sink_analysis.py*",
    ])

    output_path.write_text("\n".join(lines))
    print(f"  Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sink token redundancy: Softmax+Sink vs Softpick"
    )

    # Model paths
    parser.add_argument(
        "--vanilla-model",
        default=DEFAULT_MODELS["vanilla"],
        help="Path to vanilla Softmax model checkpoint",
    )
    parser.add_argument(
        "--softpick-model",
        default=DEFAULT_MODELS["softpick"],
        help="Path to Softpick model checkpoint",
    )
    parser.add_argument(
        "--softmax-sink-model",
        default=DEFAULT_MODELS["softmax_sink"],
        help="Path to Softmax+Sink model checkpoint",
    )

    # Analysis options
    parser.add_argument(
        "--layer-head-pairs",
        type=str,
        default="11,0;21,0",
        help="Layer,head pairs for heatmap visualization (semicolon-separated)",
    )
    parser.add_argument(
        "--heatmap-query",
        type=str,
        default=". The quick brown fox",
        help="Query text to use for heatmap visualization",
    )
    parser.add_argument(
        "--custom-queries",
        type=str,
        nargs="+",
        default=None,
        help="Additional custom query strings to analyze",
    )

    # Runtime options
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./analysis/attention_sink/outputs/sink_redundancy"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare queries
    queries = dict(GARBAGE_QUERIES)
    if args.custom_queries:
        for i, q in enumerate(args.custom_queries):
            queries[f"custom_{i}"] = q

    # Load models
    print("Loading models...")
    models = {}
    model_paths = {
        "vanilla": args.vanilla_model,
        "softpick": args.softpick_model,
        "softmax_sink": args.softmax_sink_model,
    }

    loaded_models = {}
    for model_type, model_path in model_paths.items():
        if not Path(model_path).exists():
            print(f"  Skipping {model_type}: {model_path} not found")
            continue
        print(f"  Loading {model_type}...")
        model, tokenizer = load_model_for_analysis(model_path, model_type, device, dtype)
        loaded_models[model_type] = (model, tokenizer)
        print(f"    Loaded: {model_path}")

    if not loaded_models:
        print("Error: No models loaded. Check model paths.")
        return

    # Run analysis
    print("\nRunning analysis...")
    results = {}
    for model_type, (model, tokenizer) in loaded_models.items():
        print(f"  Analyzing {model_type}...")
        results[model_type] = analyze_model(model, tokenizer, model_type, queries, device)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_entropy_comparison(results, args.output_dir / "entropy_comparison.png")
    plot_attention_sum_distribution(results, args.output_dir / "attention_sum_comparison.png")

    # Parse layer/head pairs for heatmaps
    layer_head_pairs = []
    for pair in args.layer_head_pairs.split(";"):
        parts = pair.strip().split(",")
        if len(parts) == 2:
            layer_head_pairs.append((int(parts[0]), int(parts[1])))

    if layer_head_pairs:
        plot_attention_heatmaps(
            loaded_models,
            args.heatmap_query,
            layer_head_pairs,
            device,
            args.output_dir / "attention_heatmaps.png",
        )

    # Save raw results as JSON
    json_results = {}
    for model_type, res in results.items():
        json_results[model_type] = {
            "model_name": res.model_name,
            "model_type": res.model_type,
            "num_layers": res.num_layers,
            "num_heads": res.num_heads,
            "aggregated": asdict(res.aggregated) if res.aggregated else None,
            "per_query": [
                {
                    "query_name": q.query_name,
                    "query_text": q.query_text,
                    "num_tokens": q.num_tokens,
                    "metrics": asdict(q.metrics),
                }
                for q in res.per_query
            ],
        }

    json_path = args.output_dir / "sink_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Generate markdown report
    generate_markdown_report(results, args.output_dir / "sink_results.md")

    print(f"\nResults saved to {args.output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_type, res in results.items():
        if res.aggregated is None:
            continue
        all_sums = []
        for layer in res.aggregated.attention_sum:
            all_sums.extend(layer)
        mean_sum = np.mean(all_sums)
        print(f"{model_type:15} Mean Attention Sum: {mean_sum:.4f}")


if __name__ == "__main__":
    main()
