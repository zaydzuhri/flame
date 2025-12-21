import pytest
import torch

from fla.ops.attn import (
    parallel_softpick_attn,
    parallel_relu_softpick_1_attn,
    parallel_relu_softpick_2_attn,
    parallel_abs_softmax_1_attn,
    parallel_abs_softmax_2_attn,
)
from fla.ops.attn.naive_softpick import naive_softpick_attn
from fla.ops.attn.naive_relusoftpick import (
    reference_naive_relu_softpick_1_attn,
    reference_naive_relu_softpick_2_attn,
)
from fla.ops.attn.naive_abs_softmax import (
    reference_naive_abs_softmax_1_attn,
    reference_naive_abs_softmax_2_attn,
)


VARIANTS = [
    ("softpick", parallel_softpick_attn, naive_softpick_attn),
    ("relu_softpick_1", parallel_relu_softpick_1_attn, reference_naive_relu_softpick_1_attn),
    ("relu_softpick_2", parallel_relu_softpick_2_attn, reference_naive_relu_softpick_2_attn),
    ("abs_softmax_1", parallel_abs_softmax_1_attn, reference_naive_abs_softmax_1_attn),
    ("abs_softmax_2", parallel_abs_softmax_2_attn, reference_naive_abs_softmax_2_attn),
]

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
    return 1e-3, 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for triton attention")
@pytest.mark.parametrize("variant", VARIANTS, ids=[v[0] for v in VARIANTS])
@pytest.mark.parametrize("shape", SHAPES)
def test_parallel_shape_sweep_matches_reference(variant, shape):
    name, triton_fn, ref_fn = variant
    b, t, h, d = shape
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(b, t, h, d, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = d ** -0.5

    out_triton = triton_fn(q, k, v, scale=scale, head_first=False)
    out_ref, _ = ref_fn(q, k, v, scale=scale, head_first=False)
    torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for triton attention")
@pytest.mark.parametrize("variant", VARIANTS, ids=[v[0] for v in VARIANTS])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_parallel_determinism(variant, dtype: torch.dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported on this GPU")
    name, triton_fn, _ = variant
    torch.manual_seed(1234)
    device = torch.device("cuda")
    b, t, h, d = 1, 17, 2, 16
    q = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = d ** -0.5

    out_a = triton_fn(q, k, v, scale=scale, head_first=False)
    out_b = triton_fn(q, k, v, scale=scale, head_first=False)
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out_a, out_b, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for triton attention")
@pytest.mark.parametrize("variant", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_parallel_memory_less_than_reference(variant):
    name, triton_fn, ref_fn = variant
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, t, h, d = 1, 192, 4, 64
    q = torch.randn(b, t, h, d, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = d ** -0.5

    # Warm-up to avoid compile overhead in measurements
    triton_fn(q, k, v, scale=scale, head_first=False)
    ref_fn(q, k, v, scale=scale, head_first=False)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out_triton = triton_fn(q, k, v, scale=scale, head_first=False)
    torch.cuda.synchronize()
    triton_peak = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out_ref, _ = ref_fn(q, k, v, scale=scale, head_first=False)
    torch.cuda.synchronize()
    ref_peak = torch.cuda.max_memory_allocated()

    assert ref_peak > triton_peak, f"{name}: expected reference memory > triton memory"

    del out_triton, out_ref
