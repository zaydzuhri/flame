import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # ensure custom modeling is registered


SOFTMAX_NAME = "./analysis/attention_sink/hf_models/vanilla-340M-4096-step-10000"
SOFTPICK_NAME = "./analysis/attention_sink/hf_models/softpick-340M-4096-step-10000"
SOFTMAX_ATTN_IMPL = "naive_attn"
SOFTPICK_ATTN_IMPL = "naive_softpick_attn"

INPUT_PROMPTS = [
    "**：1, 5, 25，共**3个**。",
    """ه المقالة عن العاصمة اللبنانية.""",
]

LAYER_HEAD_PAIRS = [(11, 1), (21, 2)]
BIN_TARGET = 64
COLORMAP = "pink"
ZERO_COLOR = "#D0AC93"


def collect_attentions(model_name: str, prompts: list[str], attn_impl: str) -> list:
    config = AutoConfig.from_pretrained(model_name)
    if not hasattr(config, "attn_impl"):
        raise ValueError(
            f"Config for {model_name} does not define attn_impl; use a compatible checkpoint."
        )
    config.attn_impl = attn_impl
    config.output_attentions = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config).cuda().half()
    model.eval()

    attentions_per_prompt = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.cuda()
            output_dict = model.forward(
                input_ids,
                attention_mask=None,
                use_cache=True,
                output_attentions=True,
                output_dict=True,
            )
            if output_dict.attentions is None or not any(
                attn is not None for attn in output_dict.attentions
            ):
                raise RuntimeError(
                    "Model returned no attentions. Use a naive_* attn_impl (e.g., naive_attn "
                    "or naive_softpick_attn) instead of flash/parallel implementations."
                )
            attentions_per_prompt.append(output_dict.attentions)
    return attentions_per_prompt


def bin_attention(attn: np.ndarray, bin_size: int) -> np.ndarray:
    if bin_size <= 1:
        return attn
    size = attn.shape[0] // bin_size * bin_size
    if size <= 0:
        return attn
    attn = attn[:size, :size]
    attn = attn.reshape(size // bin_size, bin_size, size // bin_size, bin_size)
    return attn.mean(axis=(1, 3))


def attention_norm(attn: np.ndarray) -> colors.PowerNorm:
    vmax = float(np.quantile(attn, 0.995))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(attn)) if np.max(attn) > 0 else 1.0
    return colors.PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax)


def zero_colormap(name: str, zero_color: str) -> colors.Colormap:
    base = plt.get_cmap(name)
    sampled = base(np.linspace(0, 1, 256))
    sampled[0] = colors.to_rgba(zero_color)
    return colors.ListedColormap(sampled)




def main() -> None:
    softmax_attentions = collect_attentions(
        SOFTMAX_NAME, INPUT_PROMPTS, SOFTMAX_ATTN_IMPL
    )
    softpick_attentions = collect_attentions(
        SOFTPICK_NAME, INPUT_PROMPTS, SOFTPICK_ATTN_IMPL
    )

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(7.4, 3.8))
        fig.subplots_adjust(wspace=0.45, hspace=0.08, left=0.08)
        fig.text(0.03, 0.73, "Input A", rotation=90, va="center", ha="left")
        fig.text(0.03, 0.29, "Input B", rotation=90, va="center", ha="left")

        for row_idx, _prompt in enumerate(INPUT_PROMPTS):
            for pair_idx, (layer_idx, head_idx) in enumerate(LAYER_HEAD_PAIRS):
                for model_idx, (label, attn_list) in enumerate(
                    [
                        ("Softmax", softmax_attentions),
                        ("Softpick", softpick_attentions),
                    ]
                ):
                    ax = axes[row_idx, pair_idx * 2 + model_idx]
                    raw_attn = (
                        attn_list[row_idx][layer_idx][0][head_idx]
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    bin_size = max(1, raw_attn.shape[0] // BIN_TARGET)
                    attn = bin_attention(raw_attn, bin_size)
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
                    ax.set_title(f"{label}\nL{layer_idx} H{head_idx}", pad=2)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

        output_dir = "testing_chamber/figures"
        os.makedirs(output_dir, exist_ok=True)
        safe_softmax = SOFTMAX_NAME.replace("/", "_").replace(".", "_")
        safe_softpick = SOFTPICK_NAME.replace("/", "_").replace(".", "_")
        base_path = os.path.join(
            output_dir,
            f"attn_compare_{safe_softmax}_vs_{safe_softpick}_L10H4_L21H9",
        )
        fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{base_path}.svg", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
