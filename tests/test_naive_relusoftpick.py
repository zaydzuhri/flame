import pytest
import torch
from einops import rearrange

from fla.ops.attn.naive_relusoftpick import (
    reference_naive_relu_softpick_1_attn,
    reference_naive_relu_softpick_2_attn,
)
from fla.ops.attn import (
    parallel_relu_softpick_1_attn,
    parallel_relu_softpick_2_attn,
)

def _cuda_dtypes():
    dtypes = [torch.float16, torch.float32]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


def _tolerances(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 1e-3, 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for triton relu softpick attention")
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("dtype", _cuda_dtypes())
@pytest.mark.parametrize("mode", [1, 2])
def test_parallel_relu_softpick_matches_reference(head_first: bool, dtype: torch.dtype, mode: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, t, h, d = 2, 8, 4, 16
    q = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    if head_first:
        q, k, v = map(lambda x: rearrange(x, "b t h d -> b h t d"), (q, k, v))

    scale = d ** -0.5
    triton_fn = parallel_relu_softpick_2_attn if mode == 2 else parallel_relu_softpick_1_attn
    ref_fn = reference_naive_relu_softpick_2_attn if mode == 2 else reference_naive_relu_softpick_1_attn

    out_triton = triton_fn(q, k, v, scale=scale, head_first=head_first)
    out_ref, _ = ref_fn(q, k, v, scale=scale, head_first=head_first)

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for triton relu softpick attention")
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("dtype", _cuda_dtypes())
@pytest.mark.parametrize("mode", [1, 2])
def test_parallel_relu_softpick_backward_matches_reference(head_first: bool, dtype: torch.dtype, mode: int):
    torch.manual_seed(1)
    device = torch.device("cuda")
    b, t, h, d = 1, 6, 2, 8
    q_base = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k_base = torch.randn_like(q_base)
    v_base = torch.randn_like(q_base)
    if head_first:
        q_base, k_base, v_base = map(
            lambda x: rearrange(x, "b t h d -> b h t d"), (q_base, k_base, v_base)
        )

    def _clone_with_grad(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone().detach().requires_grad_(True)

    triton_fn = parallel_relu_softpick_2_attn if mode == 2 else parallel_relu_softpick_1_attn
    ref_fn = reference_naive_relu_softpick_2_attn if mode == 2 else reference_naive_relu_softpick_1_attn

    q_triton, k_triton, v_triton = map(_clone_with_grad, (q_base, k_base, v_base))
    out_triton = triton_fn(q_triton, k_triton, v_triton, scale=d ** -0.5, head_first=head_first)
    out_triton.sum().backward()
    grads_triton = (q_triton.grad, k_triton.grad, v_triton.grad)

    q_ref, k_ref, v_ref = map(_clone_with_grad, (q_base, k_base, v_base))
    out_ref, _ = ref_fn(q_ref, k_ref, v_ref, scale=d ** -0.5, head_first=head_first)
    out_ref.sum().backward()
    grads_ref = (q_ref.grad, k_ref.grad, v_ref.grad)

    rtol, atol = _tolerances(dtype)
    for grad_triton, grad_ref in zip(grads_triton, grads_ref):
        torch.testing.assert_close(grad_triton, grad_ref, rtol=rtol, atol=atol)
