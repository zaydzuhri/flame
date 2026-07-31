# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def has_triton() -> bool:
    return _HAS_TRITON


if _HAS_TRITON:
    @triton.jit
    def _gru_step_kernel(
        x_ptr,
        h_ptr,
        out_ptr,
        n_elements,
        row_width,
        hidden_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        h = tl.load(h_ptr + offsets, mask=mask, other=0.0)

        # Layout is [r | z | n] on the last dimension for each row.
        idx_in_row = offsets % row_width
        in_n = idx_in_row >= (2 * hidden_size)

        gate = x + h
        # sigmoid for r/z, keep n as pre-activation.
        sig = 1.0 / (1.0 + tl.exp(-gate))

        out_gate = tl.where(in_n, gate, sig)
        tl.store(out_ptr + offsets, out_gate, mask=mask)


def gru_step_triton(x_t: torch.Tensor, h_t: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Optional gate activation kernel.

    Returns:
        Activated [r, z, n_pre] tensor with same shape as x_t/h_t,
        or None if Triton path is unavailable/inapplicable.
    """
    if not _HAS_TRITON:
        return None
    if not (x_t.is_cuda and h_t.is_cuda):
        return None
    if x_t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if not (x_t.is_contiguous() and h_t.is_contiguous()):
        return None
    if x_t.ndim != 2 or h_t.ndim != 2 or x_t.shape != h_t.shape:
        return None
    if x_t.shape[-1] % 3 != 0:
        return None

    out = torch.empty_like(x_t)
    n_elements = x_t.numel()
    hidden_size = x_t.shape[-1] // 3
    row_width = x_t.shape[-1]
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _gru_step_kernel[grid](
        x_t,
        h_t,
        out,
        n_elements,
        row_width,
        hidden_size,
        BLOCK_SIZE=block_size,
    )
    return out
