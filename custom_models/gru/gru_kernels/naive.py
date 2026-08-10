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

    # r_pre = torch.nn.functional.layer_norm(xr + hr, normalized_shape=(xr + hr).shape[-1:], eps=1e-5)
    # z_pre = torch.nn.functional.layer_norm(xz + hz, normalized_shape=(xz + hz).shape[-1:], eps=1e-5)
    # n_pre = torch.nn.functional.layer_norm(xn + hn, normalized_shape=(xn + hn).shape[-1:], eps=1e-5)

    # r_t = torch.sigmoid(r_pre)
    # z_t = torch.sigmoid(z_pre)
    # n_t = torch.tanh(n_pre + r_t * hn)

    # do per component layernorm
    xr_pre = torch.nn.functional.layer_norm(xr, normalized_shape=(xr.shape[-1],), eps=1e-5)
    hr_pre = torch.nn.functional.layer_norm(hr, normalized_shape=(hr.shape[-1],), eps=1e-5)
    xz_pre = torch.nn.functional.layer_norm(xz, normalized_shape=(xz.shape[-1],), eps=1e-5)
    hz_pre = torch.nn.functional.layer_norm(hz, normalized_shape=(hz.shape[-1],), eps=1e-5)
    xn_pre = torch.nn.functional.layer_norm(xn, normalized_shape=(xn.shape[-1],), eps=1e-5)
    hn_pre = torch.nn.functional.layer_norm(hn, normalized_shape=(hn.shape[-1],), eps=1e-5)

    r_t = torch.sigmoid(xr_pre + hr_pre)
    z_t = torch.sigmoid(xz_pre + hz_pre)
    n_t = torch.tanh(xn_pre + r_t * hn_pre)
    return (1.0 - z_t) * n_t + z_t * h_prev
