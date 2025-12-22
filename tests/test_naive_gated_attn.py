# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.attn import naive_gated_attn


SHAPES = [
    (1, 7, 2, 8),
    (2, 13, 3, 16),
    (1, 31, 4, 32),
]


def _tolerances(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 1e-5, 1e-5


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_naive_gated_attn_shape_and_dtype(shape, dtype: torch.dtype):
    """Test that naive_gated_attn produces correct output shapes for various input shapes and dtypes."""
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bf16 not supported on CPU without appropriate hardware")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, h, d = shape
    torch.manual_seed(0)

    q = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate_score = torch.randn_like(q)

    o, wei = naive_gated_attn(q, k, v, gate_score=gate_score, head_first=False)

    # Check output shape
    assert o.shape == q.shape, f"Expected output shape {q.shape}, got {o.shape}"
    # Check attention weights shape
    assert wei.shape == (b, h, t, t), f"Expected attention weights shape {(b, h, t, t)}, got {wei.shape}"
    # Check output dtype matches input
    assert o.dtype == q.dtype, f"Expected output dtype {q.dtype}, got {o.dtype}"


@pytest.mark.parametrize("head_first", [True, False])
def test_naive_gated_attn_layouts(head_first: bool):
    """Test that naive_gated_attn works with both head_first=True and head_first=False."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, h, d = 2, 13, 3, 16
    torch.manual_seed(42)

    # Create input in head_first format if requested
    if head_first:
        q = torch.randn(b, h, t, d, device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        gate_score = torch.randn_like(q)
    else:
        q = torch.randn(b, t, h, d, device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        gate_score = torch.randn_like(q)

    o, wei = naive_gated_attn(q, k, v, gate_score=gate_score, head_first=head_first)

    # Output should have same shape as input
    assert o.shape == q.shape
    # Attention weights are always [batch, heads, q_len, k_len] regardless of input layout
    assert wei.shape == (b, h, t, t)


def test_naive_gated_attn_causal_masking():
    """Test that naive_gated_attn properly applies causal masking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, h, d = 1, 5, 2, 8
    torch.manual_seed(0)

    q = torch.randn(b, t, h, d, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate_score = torch.randn_like(q)

    o, wei = naive_gated_attn(q, k, v, gate_score=gate_score, head_first=False)

    # Check that attention is causal (upper triangle should be zero)
    # wei shape is [batch, heads, q_len, k_len]
    wei_np = wei[0, 0].cpu().numpy()
    for i in range(t):
        for j in range(i + 1, t):
            assert wei_np[i, j] == 0, f"Attention weight at [{i}, {j}] should be 0 due to causal mask"


def test_naive_gated_attn_gating_effect():
    """Test that gating actually modulates the attention output."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, h, d = 1, 5, 2, 8
    torch.manual_seed(123)

    q = torch.randn(b, t, h, d, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Test with different gate values
    gate_low = torch.full_like(q, -10.0)  # sigmoid(-10) ≈ 0, should zero out output
    gate_high = torch.full_like(q, 10.0)  # sigmoid(10) ≈ 1, should pass through

    o_low, _ = naive_gated_attn(q, k, v, gate_score=gate_low, head_first=False)
    o_high, _ = naive_gated_attn(q, k, v, gate_score=gate_high, head_first=False)

    # Output with low gate should be much smaller than with high gate
    assert torch.abs(o_low).mean() < 0.1 * torch.abs(o_high).mean(), \
        "Gating with low values should significantly reduce output magnitude"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_naive_gated_attn_determinism(dtype: torch.dtype):
    """Test that naive_gated_attn is deterministic."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported on this GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and dtype == torch.bfloat16:
        pytest.skip("bf16 not well supported on CPU")

    b, t, h, d = 1, 17, 2, 16
    torch.manual_seed(1234)

    q = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate_score = torch.randn_like(q)

    o1, _ = naive_gated_attn(q, k, v, gate_score=gate_score, head_first=False)
    o2, _ = naive_gated_attn(q, k, v, gate_score=gate_score, head_first=False)

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(o1, o2, rtol=rtol, atol=atol)


def test_naive_gated_attn_requires_gate():
    """Test that naive_gated_attn requires gate_score argument."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, h, d = 1, 5, 2, 8
    torch.manual_seed(0)

    q = torch.randn(b, t, h, d, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Should raise TypeError if gate_score is not provided
    with pytest.raises(TypeError):
        naive_gated_attn(q, k, v, head_first=False)
