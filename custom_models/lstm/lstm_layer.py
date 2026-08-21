# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

class LSTMMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _get_layer_past(self, past_key_values) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if past_key_values is None:
            return None
        if isinstance(past_key_values, tuple) and len(past_key_values) == 2 and torch.is_tensor(past_key_values[0]):
            return past_key_values
        if not isinstance(past_key_values, (list, tuple)):
            return None
        if self.layer_idx >= len(past_key_values):
            return None
        layer_past = past_key_values[self.layer_idx]
        if not (isinstance(layer_past, (list, tuple)) and len(layer_past) == 2):
            return None
        return layer_past

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        layer_past = self._get_layer_past(past_key_values)

        batch_size = hidden_states.shape[0]
        if layer_past is None:
            initial_state = (
                hidden_states.new_zeros(1, batch_size, self.hidden_size),
                hidden_states.new_zeros(1, batch_size, self.hidden_size),
            )
        else:
            initial_state = tuple(
                state.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(0)
                for state in layer_past
            )

        if attention_mask is None:
            recurrent_outputs, recurrent_state = self.lstm(hidden_states, initial_state)
        else:
            recurrent_outputs = []
            recurrent_state = initial_state
            for token in hidden_states.transpose(0, 1):
                _, next_state = self.lstm(token.unsqueeze(1), recurrent_state)
                mask = attention_mask[:, len(recurrent_outputs)].to(dtype=hidden_states.dtype, device=hidden_states.device)
                recurrent_state = tuple(
                    mask.view(1, batch_size, 1) * next_value
                    + (1 - mask).view(1, batch_size, 1) * current_value
                    for next_value, current_value in zip(next_state, recurrent_state)
                )
                recurrent_outputs.append(recurrent_state[0].squeeze(0))
            recurrent_outputs = torch.stack(recurrent_outputs, dim=1)

        recurrent_h, recurrent_c = (state.squeeze(0) for state in recurrent_state)
        outputs = self.o_proj(recurrent_outputs)

        attentions = None if not output_attentions else ()
        present = (recurrent_h, recurrent_c) if use_cache else None
        return outputs, attentions, present
