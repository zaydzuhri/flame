# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .naive import lstm_step_torch
from .triton_lstm import lstm_step_triton


def _use_triton_step(x_t: torch.Tensor, h_t: torch.Tensor, prefer_triton: bool) -> bool:
    # Keep training and BPTT on native PyTorch autograd for correctness.
    if not prefer_triton:
        return False
    if x_t.requires_grad or h_t.requires_grad:
        return False
    if not (x_t.is_cuda and h_t.is_cuda):
        return False
    return True


def lstm_scan(
    hidden_states: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    x_proj: nn.Linear,
    h_proj: nn.Linear,
    attention_mask: Optional[torch.Tensor] = None,
    prefer_triton: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the recurrent LSTM loop over a full sequence.

    Args:
        hidden_states: [batch, seq_len, hidden]
        initial_state: tuple(h0, c0) with shapes [batch, hidden], or None
        x_proj: input-to-gates projection, maps hidden -> 4 * hidden
        h_proj: hidden-to-gates projection, maps hidden -> 4 * hidden
        attention_mask: optional [batch, seq_len] mask with 1 for valid tokens
        prefer_triton: if True, use Triton gate activation path in no-grad inference
    """
    batch_size, seq_len, hidden_size = hidden_states.shape

    x_all = x_proj(hidden_states)
    if initial_state is None:
        h_prev = torch.zeros(batch_size, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
        c_prev = torch.zeros(batch_size, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
    else:
        h_prev = initial_state[0].to(dtype=hidden_states.dtype, device=hidden_states.device)
        c_prev = initial_state[1].to(dtype=hidden_states.dtype, device=hidden_states.device)

    outputs = []
    for t in range(seq_len):
        x_t = x_all[:, t, :]
        h_t = h_proj(h_prev)

        if _use_triton_step(x_t, h_t, prefer_triton):
            activated = lstm_step_triton(x_t.contiguous(), h_t.contiguous())
            if activated is not None:
                i_t, f_t, g_pre, o_t = activated.chunk(4, dim=-1)
                g_t = torch.tanh(g_pre)
                c_next = f_t * c_prev + i_t * g_t
                h_next = o_t * torch.tanh(c_next)
            else:
                h_next, c_next = lstm_step_torch(x_t, h_t, h_prev, c_prev)
        else:
            h_next, c_next = lstm_step_torch(x_t, h_t, h_prev, c_prev)

        if attention_mask is not None:
            mask_t = attention_mask[:, t].to(dtype=h_next.dtype, device=h_next.device).unsqueeze(-1)
            h_next = mask_t * h_next + (1.0 - mask_t) * h_prev
            c_next = mask_t * c_next + (1.0 - mask_t) * c_prev

        outputs.append(h_next)
        h_prev = h_next
        c_prev = c_next

    return torch.stack(outputs, dim=1), h_prev, c_prev
