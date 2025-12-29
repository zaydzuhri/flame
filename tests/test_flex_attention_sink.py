import pytest
import torch

from fla.ops.attn.gpt_oss_flex_attention_sink import HAS_FLEX_ATTENTION, flex_attention_with_sink

import torch._dynamo
torch._dynamo.config.recompile_limit = 128 # Set to a higher limit

def _cuda_dtypes():
    dtypes = [torch.float16, torch.float32]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in (torch.float16, torch.bfloat16):
        return 2e-2, 2e-2
    return 1e-3, 1e-3


def _clone_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().detach().requires_grad_(True)


class _DummyAttn:
    def __init__(self, sinks: torch.Tensor, scale: float):
        self.sinks = sinks
        self.num_key_value_groups = 1
        self.scaling = scale
        self.sliding_window = None
        self.training = True


def _naive_attention_with_sink(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    attn_logits = torch.matmul(query, key.transpose(2, 3)) * scale
    q_len = query.shape[-2]
    k_len = key.shape[-2]
    causal_mask = torch.tril(
        torch.ones((q_len, k_len), device=attn_logits.device, dtype=torch.bool),
        diagonal=k_len - q_len,
    )
    attn_logits = attn_logits.masked_fill(~causal_mask, float("-inf"))
    sink_logits = sinks.view(1, -1, 1, 1).expand(attn_logits.shape[0], -1, q_len, 1)
    combined_logits = torch.cat([attn_logits, sink_logits], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined_logits, dim=-1)
    probs = probs[..., :-1]
    out = torch.matmul(probs.to(value.dtype), value)
    return out.transpose(1, 2).contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flex attention")
@pytest.mark.skipif(not HAS_FLEX_ATTENTION, reason="flex_attention not available in this build")
@pytest.mark.parametrize("dtype", _cuda_dtypes())
@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 2, 16),
        (2, 64, 4, 32),
        (2, 128, 8, 64),
        (1, 256, 4, 64),
    ],
)
def test_flex_attention_sink_matches_naive(dtype: torch.dtype, shape: tuple[int, int, int, int]):
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, t, h, d = shape
    scale = d ** -0.5

    q = torch.randn(b, h, t, d, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sinks = torch.randn(h, device=device, dtype=dtype)
    attn = _DummyAttn(sinks=sinks, scale=scale)

    with torch.no_grad():
        out_kernel = flex_attention_with_sink(attn, q, k, v, attention_mask=None, scale=scale, compile=True)
        out_ref = _naive_attention_with_sink(q, k, v, sinks, scale=scale)

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out_kernel, out_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flex attention")
@pytest.mark.skipif(not HAS_FLEX_ATTENTION, reason="flex_attention not available in this build")
@pytest.mark.parametrize("dtype", _cuda_dtypes())
@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 2, 16),
        (2, 64, 4, 32),
    ],
)
def test_flex_attention_sink_backward_matches_naive(dtype: torch.dtype, shape: tuple[int, int, int, int]):
    torch.manual_seed(1)
    device = torch.device("cuda")
    b, t, h, d = shape
    scale = d ** -0.5

    q_base = torch.randn(b, h, t, d, device=device, dtype=dtype)
    k_base = torch.randn_like(q_base)
    v_base = torch.randn_like(q_base)
    sinks_base = torch.randn(h, device=device, dtype=dtype)

    q_k, k_k, v_k = map(_clone_with_grad, (q_base, k_base, v_base))
    sinks_k = _clone_with_grad(sinks_base)
    attn_k = _DummyAttn(sinks=sinks_k, scale=scale)
    out_kernel = flex_attention_with_sink(attn_k, q_k, k_k, v_k, attention_mask=None, scale=scale, compile=True)
    out_kernel.sum().backward()
    grads_kernel = (q_k.grad, k_k.grad, v_k.grad, sinks_k.grad)

    q_r, k_r, v_r = map(_clone_with_grad, (q_base, k_base, v_base))
    sinks_r = _clone_with_grad(sinks_base)
    out_ref = _naive_attention_with_sink(q_r, k_r, v_r, sinks_r, scale=scale)
    out_ref.sum().backward()
    grads_ref = (q_r.grad, k_r.grad, v_r.grad, sinks_r.grad)

    rtol, atol = _tolerances(dtype)
    for grad_k, grad_r in zip(grads_kernel, grads_ref):
        torch.testing.assert_close(grad_k, grad_r, rtol=rtol, atol=atol)
