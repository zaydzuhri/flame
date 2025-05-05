from einops import rearrange
from typing import Optional
import torch
from transformers import AttentionInterface
from torch.nn import functional as F

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def softpick(x, dim=-1, eps=1e-8):
    # softpick function: relu(exp(x)-1) / sum(abs(exp(x)-1))
    # numerically stable version
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    r_x_e_1 = F.relu(x_e_1)
    a_x_e_1 = torch.where(x.isfinite(), torch.abs(x_e_1), 0)
    return r_x_e_1 / (torch.sum(a_x_e_1, dim=dim, keepdim=True) + eps) # epsilon is only useful if all inputs are EXACTLY 0. we might not even need it

def naive_softpick_attn(
    module: torch.nn.Module,  # required arg
    query: torch.Tensor,  # required arg
    key: torch.Tensor,  # required arg
    value: torch.Tensor,  # required arg
    attention_mask: Optional[torch.Tensor],  # required arg
    *args,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    **kwargs
) -> torch.Tensor:
    head_dim = query.shape[-1]

    # In transformers, the shape is (batch_size, num_heads, seq_len, head_dim)
    num_query_heads = query.shape[1]
    num_key_valye_heads = key.shape[1]


    if num_query_heads != num_key_valye_heads:
        # MQA or GQA
        key = repeat_kv(key, num_query_heads // num_key_valye_heads)
        value = repeat_kv(value, num_query_heads // num_key_valye_heads)

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    if not head_first:
        query, key, value = map(lambda x: rearrange(x, 'b t h d -> b h t d'), (query, key, value))
    query_len = query.shape[-2]
    key_len = key.shape[-2]
    mask = torch.tril(torch.ones(key_len, key_len, device=query.device))
    wei = torch.matmul(query, key.transpose(2, 3)) # shape: (batch_size, num_heads, query_len, key_len)
    wei = wei * scale
    wei = wei.masked_fill(mask[key_len-query_len:key_len, :key_len] == 0, float('-inf'))
    wei = softpick(wei.float(), dim=-1).to(query.dtype)
    o = torch.matmul(wei, value) # shape: (batch_size, num_heads, q_len, head_dim)
    if not head_first:
        o = rearrange(o, 'b h t d -> b t h d')
    return o, wei

def softpick_attention(*args, **kwargs):
    # print("Using softpick attention") # NOTE: Add print statement here to check whether we actually use softpick or not
    return naive_softpick_attn(*args, **kwargs)

AttentionInterface.register("softpick", softpick_attention)