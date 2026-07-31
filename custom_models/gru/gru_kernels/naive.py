# -*- coding: utf-8 -*-

import torch


def gru_step_torch(x_t: torch.Tensor, h_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    """
    One GRU recurrent step in pure PyTorch.

    Args:
        x_t: Input projection at step t, shape [batch, 3 * hidden].
        h_t: Hidden projection at step t, shape [batch, 3 * hidden].
        h_prev: Previous hidden state, shape [batch, hidden].
    """
    xr, xz, xn = x_t.chunk(3, dim=-1)
    hr, hz, hn = h_t.chunk(3, dim=-1)

    r_t = torch.sigmoid(xr + hr)
    z_t = torch.sigmoid(xz + hz)
    n_t = torch.tanh(xn + r_t * hn)
    return (1.0 - z_t) * n_t + z_t * h_prev
