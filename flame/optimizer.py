# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from functools import partial
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_optimizer_state_dict,
                                                     set_optimizer_state_dict)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR

from torchtitan.config_manager import JobConfig


class OptimizersContainer(Stateful):
    """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages
    and saving/loading optimizer state_dict at checkpoint.
    """

    def __init__(
        self, model_parts: List[nn.Module], optimizer_kwargs: Dict[str, Any], name: str
    ) -> None:
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            if name == "Adam":
                # TODO: make the optimizer options configurable by toml/cmd args
                optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
            elif name == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            else:
                raise NotImplementedError(f"Optimizer {name} not added.")
            self.optimizers.append(optimizer)
        self._validate_length(len(self.model_parts))

    def _validate_length(self, expected_length) -> None:
        assert expected_length == len(
            self.optimizers
        ), "Must pass one optimizer per model part or per param if using OptimizersInBackwardContainer"

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=False),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=False),
        )
        list(map(func, self.model_parts, self.optimizers))


class OptimizersInBackwardContainer(OptimizersContainer):
    """Optimiers in backward to skip .step() and .zero_grad()"""

    def __init__(
        self, model_parts: List[nn.Module], optimizer_kwargs: Dict[str, Any], name: str
    ) -> None:
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            if name == "Adam":
                # TODO: make the optimizer options configurable by toml/cmd args
                optim_dict = {
                    param: torch.optim.Adam([param], **optimizer_kwargs)
                    for param in model.parameters()
                }
            elif name == "AdamW":
                optim_dict = {
                    param: torch.optim.AdamW([param], **optimizer_kwargs)
                    for param in model.parameters()
                }
            else:
                raise NotImplementedError(f"Optimizer {name} not added.")

            def optim_hook(param) -> None:
                optim_dict[param].step()
                optim_dict[param].zero_grad()

            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

            self.optimizers.extend([optim_dict[param] for param in model.parameters()])
        self._validate_length(
            sum(
                len([param for param in model.parameters()])
                for model in self.model_parts
            )
        )

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


# consider split between PP and non-PP
def build_optimizers(
    model_parts: List[nn.Module], job_config: JobConfig
) -> OptimizersContainer:
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    fused = job_config.optimizer.fused
    optimizer_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": fused,
        "foreach": not fused,
    }

    return (
        OptimizersContainer(model_parts, optimizer_kwargs, name)
        if not optim_in_bwd
        else OptimizersInBackwardContainer(model_parts, optimizer_kwargs, name)
    )


def linear_scheduler_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    ratio = max(0., float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return ratio * (1 - min_lr_ratio) + min_lr_ratio


def cosine_scheduler_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_ratio) + min_lr_ratio
    return max(0, factor)


def wsd_scheduler_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    decay_ratio: float = 0.1,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1,
    decay_type: str = "sqrt"
):
    num_stable_steps = num_training_steps * (1 - decay_ratio)
    num_decay_steps = num_training_steps - num_stable_steps - num_warmup_steps
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0
    if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        if decay_type == "linear":
            return min_lr_ratio + (1 - min_lr_ratio) * (1 - progress)
        elif decay_type == "exp":
            return min_lr_ratio ** progress
        elif decay_type == "cosine":
            return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)) * 0.5
        elif decay_type == "square":
            return min_lr_ratio + (1 - min_lr_ratio) * (1 - progress ** 2)
        elif decay_type == "sqrt":
            return min_lr_ratio + (1 - min_lr_ratio) * (1 - math.sqrt(progress))
        else:
            raise ValueError(f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp','square','sqrt']")
    return min_lr_ratio


class SchedulersContainer:
    """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

    def __init__(self, optimizers, lr_lambda) -> None:
        self.schedulers = []
        for optimizer in optimizers:
            self.schedulers.append(LambdaLR(optimizer, lr_lambda=lr_lambda))

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def get_lr_scheduler_state(self) -> Dict[str, Any]:
        state_dict = {}
        if len(self.schedulers) == 1:
            state_dict["lr_scheduler"] = self.schedulers[0]
        else:
            # For now, pipeline-parallel with looped schedules does not support resharding for lr_scheduler.
            # It should only support saving and loading a distributed checkpoint with the same number of pp ranks
            for idx, lr_scheduler in enumerate(self.schedulers):
                state_dict[f"lr_scheduler_{idx}"] = lr_scheduler
        return state_dict


def build_lr_schedulers(optimizers, job_config: JobConfig) -> SchedulersContainer:
    warmup_steps = int(job_config.training.warmup_steps)
    if job_config.optimizer.scheduler == "linear":
        lr_lambda = partial(
            linear_scheduler_lambda,
            num_warmup_steps=warmup_steps,
            num_training_steps=job_config.training.steps,
            min_lr_ratio=job_config.optimizer.min_lr_ratio
        )
    elif job_config.optimizer.scheduler == "cosine":
        lr_lambda = partial(
            cosine_scheduler_lambda,
            num_warmup_steps=warmup_steps,
            num_training_steps=job_config.training.steps,
            min_lr_ratio=job_config.optimizer.min_lr_ratio
        )
    elif job_config.optimizer.scheduler == "wsd":
        lr_lambda = partial(
            wsd_scheduler_lambda,
            num_warmup_steps=warmup_steps,
            num_training_steps=job_config.training.steps,
            min_lr_ratio=job_config.optimizer.min_lr_ratio
        )
    else:
        raise ValueError(f"Scheduler {job_config.optimizer.scheduler} not supported")
    return SchedulersContainer(optimizers, lr_lambda)
