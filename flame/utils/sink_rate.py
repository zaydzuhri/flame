# -*- coding: utf-8 -*-

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from fla.layers.attn import StochasticSoftpickAttention
from torchtitan.tools.logging import logger


def calculate_sink_rate(attention_maps: Iterable[torch.Tensor], epsilon: float = 0.3) -> float:
    """
    Calculate sink rate using the formula:
    sink_rate = (1/(L*H))*sum_L_H(1_((1/T)*sum_T(a_l_h_1_t) > epsilon))
    """
    sink_rate = 0.0
    num_layers_with_maps = 0
    for attention in attention_maps:
        if attention is None:
            continue
        first_token_attention = attention[:, :, :, 0]
        mean_first_token_attention = first_token_attention.mean(dim=-1)
        indicator = (mean_first_token_attention > epsilon).float()
        batch_sink_rates = indicator.mean(dim=1)
        sink_rate += batch_sink_rates.mean().item()
        num_layers_with_maps += 1

    if num_layers_with_maps == 0:
        return 0.0
    return sink_rate / num_layers_with_maps


@dataclass
class SinkMonitorConfig:
    threshold: float
    low_watermark: float
    check_interval: int
    epsilon: float
    eval_batch_size: int
    eval_seq_len: int
    eval_batches: int
    slow_softmax_impl: str
    slow_softpick_impl: str
    strategy: str
    prob_step: float
    min_prob: float
    max_prob: float
    eval_softpick_threshold: float


@contextmanager
def _temporary_attention_overrides(
    model_parts: List[torch.nn.Module],
    slow_softmax_impl: str,
    slow_softpick_impl: str,
    force_softpick: bool,
):
    saved_state: List[Tuple] = []
    for module in _iter_attention_modules(model_parts):
        if isinstance(module, StochasticSoftpickAttention):
            saved_state.append(
                (
                    module,
                    module.attn_impl,
                    module.softpick_attn_impl,
                    module.stochastic_value,
                    module._softpick_manual,
                )
            )
            module._attn_impl = slow_softmax_impl
            module._softpick_attn_impl = slow_softpick_impl
            module._softpick_manual = True
            module.set_stochastic_value(1.0 if force_softpick else 0.0)
        else:
            saved_state.append((module, module.attn_impl))
            module.attn_impl = slow_softmax_impl

    try:
        yield
    finally:
        for saved in saved_state:
            module = saved[0]
            if isinstance(module, StochasticSoftpickAttention):
                _, attn_impl, softpick_impl, stochastic_value, manual_flag = saved
                module._attn_impl = attn_impl
                module._softpick_attn_impl = softpick_impl
                module._softpick_manual = manual_flag
                module.set_stochastic_value(stochastic_value)
            else:
                _, attn_impl = saved
                module.attn_impl = attn_impl


def _iter_attention_modules(model_parts: List[torch.nn.Module]):
    for model in model_parts:
        for module in model.modules():
            if hasattr(module, "attn_impl"):
                yield module


class SinkRateMonitor:
    def __init__(self, config: SinkMonitorConfig):
        self.config = config

    def should_run(self, step: int) -> bool:
        return (
            self.config.threshold is not None
            and self.config.check_interval > 0
            and step % self.config.check_interval == 0
        )

    def evaluate_and_update(
        self,
        model_parts: List[torch.nn.Module],
        batch: Dict[str, torch.Tensor],
    ) -> Optional[Tuple[float, Optional[float], bool]]:
        if not model_parts:
            logger.warning("SinkRateMonitor: no model parts provided for evaluation")
            return None

        eval_batches = list(self._iter_eval_batches(batch))
        if not eval_batches:
            logger.warning("SinkRateMonitor: no evaluation data available from batch")
            return None

        model = model_parts[0]
        was_training = model.training
        model.eval()

        current_prob = self._get_current_probability(model_parts)
        use_softpick_eval = current_prob >= self.config.eval_softpick_threshold
        slow_softmax_impl = (
            self.config.slow_softpick_impl if use_softpick_eval else self.config.slow_softmax_impl
        )

        sink_rates: List[float] = []
        try:
            with _temporary_attention_overrides(
                model_parts,
                slow_softmax_impl=slow_softmax_impl,
                slow_softpick_impl=self.config.slow_softpick_impl,
                force_softpick=use_softpick_eval,
            ):
                for eval_batch in eval_batches:
                    with torch.no_grad():
                        outputs = model(
                            input_ids=eval_batch["input_ids"],
                            attention_mask=eval_batch.get("attention_mask"),
                            output_attentions=True,
                            use_cache=False,
                        )
                    if outputs.attentions is None:
                        continue
                    sink_rates.append(calculate_sink_rate(outputs.attentions, self.config.epsilon))
        finally:
            if was_training:
                model.train()

        if not sink_rates:
            logger.warning("SinkRateMonitor: no attention maps returned during evaluation")
            return None

        sink_rate = sum(sink_rates) / len(sink_rates)
        sink_rate = self._distributed_average(sink_rate, device=eval_batches[0]["input_ids"].device)

        new_prob = self._compute_new_probability(current_prob, sink_rate)
        if new_prob is not None:
            self._apply_probability(model_parts, new_prob)
        return sink_rate, new_prob, use_softpick_eval

    def _compute_new_probability(self, current_prob: float, sink_rate: float) -> Optional[float]:
        if sink_rate >= self.config.threshold:
            if self.config.strategy == "hard_switch":
                return self.config.max_prob
            return min(self.config.max_prob, current_prob + self.config.prob_step)
        if sink_rate <= self.config.low_watermark:
            if self.config.strategy == "hard_switch":
                return self.config.min_prob
            return max(self.config.min_prob, current_prob - self.config.prob_step)
        return None

    def _apply_probability(self, model_parts: List[torch.nn.Module], new_prob: float) -> None:
        for module in _iter_attention_modules(model_parts):
            if isinstance(module, StochasticSoftpickAttention):
                module.set_stochastic_value(new_prob)

    def _distributed_average(self, value: float, device: torch.device) -> float:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor = torch.tensor([value], device=device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG)
            return tensor.item()
        return value

    def _get_current_probability(self, model_parts: List[torch.nn.Module]) -> float:
        for module in _iter_attention_modules(model_parts):
            if isinstance(module, StochasticSoftpickAttention):
                return module.stochastic_value
        return 0.0

    def _iter_eval_batches(self, batch: Dict[str, torch.Tensor]) -> Iterable[Dict[str, torch.Tensor]]:
        input_ids = batch.get("input_ids")
        if input_ids is None:
            return

        attention_mask = batch.get("attention_mask")
        total_sequences = input_ids.size(0)
        max_batches = min(
            self.config.eval_batches,
            (total_sequences + self.config.eval_batch_size - 1) // self.config.eval_batch_size,
        )
        for idx in range(max_batches):
            start = idx * self.config.eval_batch_size
            end = min(start + self.config.eval_batch_size, total_sequences)
            if start >= end:
                break
            ids_slice = input_ids[start:end]
            if ids_slice.size(1) > self.config.eval_seq_len:
                ids_slice = ids_slice[:, : self.config.eval_seq_len]

            eval_batch: Dict[str, torch.Tensor] = {"input_ids": ids_slice}
            if attention_mask is not None:
                mask_slice = attention_mask[start:end]
                if mask_slice.size(1) > self.config.eval_seq_len:
                    mask_slice = mask_slice[:, : self.config.eval_seq_len]
                eval_batch["attention_mask"] = mask_slice
            yield eval_batch
