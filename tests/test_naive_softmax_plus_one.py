import pytest
import torch
from einops import rearrange

from fla.ops.attn import parallel_softmax_plus_one_attn
from fla.ops.attn.naive_softmax_plus_one import reference_naive_softmax_plus_one_attn


def _cuda_dtypes():
    dtypes = [torch.float16, torch.float32]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


def _tolerances(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2, 1e-2
    return 1e-3, 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for triton softmax-plus-one attention")
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("dtype", _cuda_dtypes())
def test_softmax_plus_one_matches_reference(head_first: bool, dtype: torch.dtype):
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, t, h, d = 2, 9, 4, 16
    q = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    if head_first:
        q, k, v = map(lambda x: rearrange(x, "b t h d -> b h t d"), (q, k, v))

    scale = d ** -0.5
    out_triton = parallel_softmax_plus_one_attn(q, k, v, scale=scale, head_first=head_first)
    out_ref, _ = reference_naive_softmax_plus_one_attn(q, k, v, scale=scale, head_first=head_first)

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for triton softmax-plus-one attention")
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("dtype", _cuda_dtypes())
def test_softmax_plus_one_backward_matches_reference(head_first: bool, dtype: torch.dtype):
    torch.manual_seed(1)
    device = torch.device("cuda")
    b, t, h, d = 1, 7, 2, 8
    q_base = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k_base = torch.randn_like(q_base)
    v_base = torch.randn_like(q_base)
    if head_first:
        q_base, k_base, v_base = map(
            lambda x: rearrange(x, "b t h d -> b h t d"), (q_base, k_base, v_base)
        )

    def _clone_with_grad(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone().detach().requires_grad_(True)

    q_triton, k_triton, v_triton = map(_clone_with_grad, (q_base, k_base, v_base))
    out_triton = parallel_softmax_plus_one_attn(
        q_triton, k_triton, v_triton, scale=d ** -0.5, head_first=head_first
    )
    out_triton.sum().backward()
    grads_triton = (q_triton.grad, k_triton.grad, v_triton.grad)

    q_ref, k_ref, v_ref = map(_clone_with_grad, (q_base, k_base, v_base))
    out_ref, _ = reference_naive_softmax_plus_one_attn(
        q_ref, k_ref, v_ref, scale=d ** -0.5, head_first=head_first
    )
    out_ref.sum().backward()
    grads_ref = (q_ref.grad, k_ref.grad, v_ref.grad)

    rtol, atol = _tolerances(dtype)
    for grad_triton, grad_ref in zip(grads_triton, grads_ref):
        torch.testing.assert_close(grad_triton, grad_ref, rtol=rtol, atol=atol)
