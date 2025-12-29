# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.gpt_oss_sink_attn import GptOssSinkAttention
from fla.ops.attn.gpt_oss_flex_attention_sink import HAS_FLEX_ATTENTION


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
def test_gpt_oss_sink_attention_matches_naive():
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, t, h, d = 2, 32, 4, 16
    hidden_size = h * d
    attn = GptOssSinkAttention(
        hidden_size=hidden_size,
        num_heads=h,
        num_kv_heads=h,
        qkv_bias=False,
        qk_norm=False,
        window_size=None,
        rope_theta=10000.0,
        max_position_embeddings=t,
        layer_idx=0,
        attn_impl="gpt_oss_flex_attention_sink",
        initializer_range=0.006,
    ).to(device)
    hidden_states = torch.randn(b, t, hidden_size, device=device)

    out, _, _ = attn(hidden_states, attention_mask=None)

    q = attn.q_proj(hidden_states)
    k = attn.k_proj(hidden_states)
    v = attn.v_proj(hidden_states)
    q = q.view(b, t, h, d)
    k = k.view(b, t, h, d)
    v = v.view(b, t, h, d)
    q, k = attn.rotary(q, k, seqlen_offset=0, max_seqlen=t, cu_seqlens=None)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    ref = _naive_attention_with_sink(q, k, v, attn.sinks, attn.scaling)
    ref = ref.reshape(b, t, hidden_size)
    ref = attn.o_proj(ref)

    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
