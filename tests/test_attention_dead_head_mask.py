import torch
import pytest

from fla.layers.attn import Attention


def _identity_linear(linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        rows, cols = linear.weight.shape
        assert rows == cols
        linear.weight.copy_(torch.eye(rows, dtype=linear.weight.dtype))
        if linear.bias is not None:
            linear.bias.zero_()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for rotary/triton attention path")
def test_attention_zeroes_masked_heads_pre_o_proj() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    attn = Attention(
        hidden_size=4,
        num_heads=2,
        num_kv_heads=2,
        qkv_bias=False,
        qk_norm=False,
        layer_idx=0,
        attn_impl="naive_attn",
        dead_head_mask=torch.tensor([True, False]),
    ).to(device)
    _identity_linear(attn.q_proj)
    _identity_linear(attn.k_proj)
    _identity_linear(attn.v_proj)
    _identity_linear(attn.o_proj)

    hidden_states = torch.randn(1, 3, 4, device=device)
    output, _, _ = attn(hidden_states=hidden_states, use_cache=False, return_dict=True)
    # Head 0 occupies dims [0:2], head 1 occupies dims [2:4] after flatten.
    torch.testing.assert_close(output[..., :2], torch.zeros_like(output[..., :2]))
    assert torch.max(output[..., 2:].abs()).item() > 0


def test_attention_set_dead_head_mask_validates_shape() -> None:
    attn = Attention(
        hidden_size=4,
        num_heads=2,
        num_kv_heads=2,
        attn_impl="naive_attn",
    )
    try:
        attn.set_dead_head_mask(torch.tensor([True, False, True]))
    except ValueError as exc:
        assert "does not match num_heads=2" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid dead_head_mask length.")
