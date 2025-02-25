# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= sum(
            i.num_parameters() for i in model.children() if isinstance(i, nn.Embedding)
        )
    return num_params


def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    if hasattr(model_config, "num_heads"):
        num_heads = model_config.num_heads
    elif hasattr(model_config, "num_attention_heads"):
        num_heads = model_config.num_attention_heads
    else:
        num_heads = 1
        logger.warning("num_heads not found in model_config, defaulting to 1. ")

    l, h, q, t = (
        model_config.num_hidden_layers,
        num_heads,
        model_config.hidden_size // num_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token
