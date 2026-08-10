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

    # i_t = torch.sigmoid(xi + hi)
    # f_t = torch.sigmoid(xf + hf)
    # g_t = torch.tanh(xg + hg)
    # o_t = torch.sigmoid(xo + ho)

    # c_next = f_t * c_prev + i_t * g_t
    # h_next = o_t * torch.tanh(c_next)

    # do per component layernorm
    xi_pre = torch.nn.functional.layer_norm(xi, normalized_shape=(xi.shape[-1],), eps=1e-5)
    hi_pre = torch.nn.functional.layer_norm(hi, normalized_shape=(hi.shape[-1],), eps=1e-5)
    xf_pre = torch.nn.functional.layer_norm(xf, normalized_shape=(xf.shape[-1],), eps=1e-5)
    hf_pre = torch.nn.functional.layer_norm(hf, normalized_shape=(hf.shape[-1],), eps=1e-5)
    xg_pre = torch.nn.functional.layer_norm(xg, normalized_shape=(xg.shape[-1],), eps=1e-5)
    hg_pre = torch.nn.functional.layer_norm(hg, normalized_shape=(hg.shape[-1],), eps=1e-5)
    xo_pre = torch.nn.functional.layer_norm(xo, normalized_shape=(xo.shape[-1],), eps=1e-5)
    ho_pre = torch.nn.functional.layer_norm(ho, normalized_shape=(ho.shape[-1],), eps=1e-5)

    i_t = torch.sigmoid(xi_pre + hi_pre)
    f_t = torch.sigmoid(xf_pre + hf_pre)
    g_t = torch.tanh(xg_pre + hg_pre)
    o_t = torch.sigmoid(xo_pre + ho_pre)

    c_next = f_t * c_prev + i_t * g_t
    h_next = o_t * torch.tanh(c_next)

    return h_next, c_next
