# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .gru_kernels import gru_scan


class GRUMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        bias: bool = True,
        recurrent_use_triton: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.recurrent_use_triton = recurrent_use_triton

        self.x_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.h_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _get_layer_past(self, past_key_values) -> Optional[torch.Tensor]:
        if past_key_values is None:
            return None
        if not isinstance(past_key_values, (list, tuple)):
            return None
        if self.layer_idx >= len(past_key_values):
            return None
        return past_key_values[self.layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        layer_past = self._get_layer_past(past_key_values)

        recurrent_outputs, recurrent_last = gru_scan(
            hidden_states=hidden_states,
            initial_state=layer_past,
            x_proj=self.x_proj,
            h_proj=self.h_proj,
            attention_mask=attention_mask,
            prefer_triton=self.recurrent_use_triton,
        )
        outputs = self.o_proj(recurrent_outputs)

        attentions = None if not output_attentions else ()
        present = recurrent_last if use_cache else None
        return outputs, attentions, present
