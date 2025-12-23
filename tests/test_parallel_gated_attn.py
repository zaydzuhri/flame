# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.attn import naive_gated_attn, parallel_gated_attn


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
def test_parallel_gated_attn_shape_and_dtype(shape, dtype: torch.dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bf16 not supported on CPU without appropriate hardware")
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("fp16 not supported on CPU without appropriate hardware")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, h, d = shape
    torch.manual_seed(0)

    q = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate_score = torch.randn_like(q)

    o = parallel_gated_attn(q, k, v, gate_score=gate_score, head_first=False)

    assert o.shape == q.shape, f"Expected output shape {q.shape}, got {o.shape}"
    assert o.dtype == q.dtype, f"Expected output dtype {q.dtype}, got {o.dtype}"


def test_parallel_gated_attn_matches_naive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, t, h, d = 2, 13, 3, 16
    torch.manual_seed(42)

    q = torch.randn(b, t, h, d, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate_score = torch.randn_like(q)

    o_parallel = parallel_gated_attn(q, k, v, gate_score=gate_score, head_first=False)
    o_naive, _ = naive_gated_attn(q, k, v, gate_score=gate_score, head_first=False)

    rtol, atol = _tolerances(torch.float32)
    torch.testing.assert_close(o_parallel, o_naive, rtol=rtol, atol=atol)
