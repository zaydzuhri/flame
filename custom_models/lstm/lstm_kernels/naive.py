# -*- coding: utf-8 -*-

from typing import Tuple

import torch


def lstm_step_torch(
    x_t: torch.Tensor,
    h_t: torch.Tensor,
    h_prev: torch.Tensor,
    c_prev: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One LSTM recurrent step in pure PyTorch.

    Args:
        x_t: Input projection at step t, shape [batch, 4 * hidden].
        h_t: Hidden projection at step t, shape [batch, 4 * hidden].
        h_prev: Previous hidden state, shape [batch, hidden].
        c_prev: Previous cell state, shape [batch, hidden].
    """
    xi, xf, xg, xo = x_t.chunk(4, dim=-1)
    hi, hf, hg, ho = h_t.chunk(4, dim=-1)

    i_t = torch.sigmoid(xi + hi)
    f_t = torch.sigmoid(xf + hf)
    g_t = torch.tanh(xg + hg)
    o_t = torch.sigmoid(xo + ho)

    c_next = f_t * c_prev + i_t * g_t
    h_next = o_t * torch.tanh(c_next)
    return h_next, c_next
