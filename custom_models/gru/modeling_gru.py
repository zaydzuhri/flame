# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as GRUMLP
from fla.modules import RMSNorm

from .config_gru import GRUConfig
from .gru_layer import GRUMixer

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


logger = logging.get_logger(__name__)


class GRUBlock(nn.Module):
    def __init__(self, config: GRUConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = GRUMixer(
            hidden_size=config.hidden_size,
            layer_idx=layer_idx,
            bias=config.gru_bias,
            recurrent_use_triton=config.recurrent_use_triton,
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = GRUMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Unpack[Any],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, present = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)

        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attentions,)
        if use_cache:
            outputs += (present,)
        return outputs


class GRUPreTrainedModel(PreTrainedModel):
    config_class = GRUConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GRUBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if rescale_prenorm_residual:
            p = None
            if hasattr(module, "o_proj"):
                p = module.o_proj.weight
            elif hasattr(module, "down_proj"):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class GRUModel(GRUPreTrainedModel):
    def __init__(self, config: GRUConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GRUBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def _normalize_past(self, past_key_values):
        if past_key_values is None:
            return None
        if isinstance(past_key_values, (list, tuple)):
            return past_key_values
        return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            warnings.warn(
                "`GRUModel` does not return attention weights, so `output_attentions` is set to `False`."
            )
            output_attentions = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values = self._normalize_past(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = [] if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_past = None
            if past_key_values is not None and layer_idx < len(past_key_values):
                layer_past = past_key_values[layer_idx]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    layer_past,
                    output_attentions,
                    use_cache,
                    **kwargs,
                )
            else:
                # Pass full cache tuple so each recurrent mixer can fetch by layer index.
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                layer_present = layer_outputs[2 if output_attentions else 1]
                next_cache.append(layer_present)

            if output_attentions:
                all_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        packed_cache = tuple(next_cache) if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, packed_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=packed_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class GRUForCausalLM(GRUPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GRUModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs,
    ):
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and (past_key_values is None or len(past_key_values) == 0):
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "logits_to_keep": logits_to_keep,
            }
        )
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training
        logits = None if fuse_linear_and_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion

            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
