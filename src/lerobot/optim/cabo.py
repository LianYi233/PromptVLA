#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimizer helpers for PI0.5's relative-update CABO controller.

CABO limits action-side parameter motion directly from the deterministic next
AdamW update. It deliberately does not estimate model-output changes: there is
no extra model forward, Jacobian-vector product, or stochastic projection.
"""

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

CABO_VLM_GROUP = "vlm"
CABO_PROMPT_GROUP = "prompt"
CABO_ACTION_EXPERT_GROUP = "action_expert"
CABO_ACTION_PROJECTION_GROUP = "action_projection"
CABO_VLM_PROMPT_GROUP = "vlm_prompt"
CABO_ACTION_PROMPT_GROUP = "action_prompt"
CABO_GROUP_NAME = "name"


@dataclass
class OptimizerStepControl:
    """Policy-provided controls for one optimizer step.

    ``skip_optimizer_step`` is deliberately separate from setting a learning
    rate to zero: AdamW still advances its moments and step counter when its
    learning rate is zero.
    """

    group_scales: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    skip_optimizer_step: bool = False


def unwrap_optimizer(optimizer: Optimizer) -> Optimizer:
    """Return the torch optimizer hidden by wrappers such as AcceleratedOptimizer."""
    unwrapped = optimizer
    seen: set[int] = set()
    while hasattr(unwrapped, "optimizer"):
        if id(unwrapped) in seen:
            break
        seen.add(id(unwrapped))
        nested = unwrapped.optimizer
        if nested is unwrapped:
            break
        unwrapped = nested
    if not isinstance(unwrapped, Optimizer):
        raise TypeError(f"CABO requires a torch Optimizer, got {type(unwrapped).__name__}")
    return unwrapped


def require_adamw(optimizer: Optimizer) -> AdamW:
    """Unwrap and validate the optimizer supported by the CABO controller."""
    unwrapped = unwrap_optimizer(optimizer)
    if not isinstance(unwrapped, AdamW):
        raise TypeError(f"CABO currently supports AdamW only, got {type(unwrapped).__name__}")
    return unwrapped


def get_named_param_group(optimizer: Optimizer, name: str) -> dict[str, Any]:
    """Find exactly one optimizer parameter group by its stable CABO name."""
    unwrapped = unwrap_optimizer(optimizer)
    matches = [group for group in unwrapped.param_groups if group.get(CABO_GROUP_NAME) == name]
    if len(matches) != 1:
        raise ValueError(f"CABO expected one optimizer group named {name!r}, found {len(matches)}")
    return matches[0]


def validate_adamw_param_group(param_group: Mapping[str, Any]) -> None:
    """Reject AdamW modes the native-dtype candidate computation cannot reproduce."""
    if isinstance(param_group.get("lr"), Tensor):
        raise RuntimeError("CABO requires a numeric learning rate; tensor learning rates are not supported")
    unsupported_modes = [
        name
        for name in ("differentiable", "fused", "capturable", "foreach")
        if bool(param_group.get(name, False))
    ]
    if unsupported_modes:
        modes = ", ".join(unsupported_modes)
        raise RuntimeError(f"CABO does not support these AdamW modes: {modes}")


def _next_adamw_moments(
    parameter: nn.Parameter,
    param_group: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[Tensor, Tensor, int, float, float]:
    gradient = parameter.grad
    if gradient is None:
        raise ValueError("A gradient is required to compute an AdamW candidate update")
    if gradient.is_sparse:
        raise RuntimeError("CABO does not support sparse AdamW gradients")
    if torch.is_complex(parameter) or torch.is_complex(gradient):
        raise RuntimeError("CABO does not support complex AdamW parameters")
    validate_adamw_param_group(param_group)

    beta1, beta2 = param_group["betas"]
    grad = -gradient.detach() if bool(param_group.get("maximize", False)) else gradient.detach()
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    next_exp_avg = torch.zeros_like(parameter) if exp_avg is None else exp_avg.detach().clone()
    next_exp_avg_sq = torch.zeros_like(parameter) if exp_avg_sq is None else exp_avg_sq.detach().clone()
    if grad.dtype != next_exp_avg.dtype:
        raise RuntimeError(
            "CABO requires AdamW gradients and optimizer moments to share a dtype, "
            f"got gradient={grad.dtype}, exp_avg={next_exp_avg.dtype}"
        )

    next_exp_avg.lerp_(grad, 1.0 - beta1)
    next_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    raw_step = state.get("step", 0)
    current_step = int(raw_step.item()) if isinstance(raw_step, Tensor) else int(raw_step)
    return next_exp_avg, next_exp_avg_sq, current_step + 1, beta1, beta2


@torch.no_grad()
def adamw_candidate_parameter_delta(
    parameter: nn.Parameter,
    param_group: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    include_weight_decay: bool = True,
) -> Tensor | None:
    """Compute the next AdamW parameter delta without mutating optimizer state.

    ``include_weight_decay=False`` isolates the optimizer's learning component.
    CABO uses that mode so decoupled weight decay is not mistaken for the
    reference group's learning signal when it establishes the action-side budget.
    """
    if parameter.grad is None:
        return None

    next_exp_avg, next_exp_avg_sq, next_step, beta1, beta2 = _next_adamw_moments(
        parameter, param_group, state
    )
    eps = float(param_group["eps"])
    learning_rate = float(param_group["lr"])
    bias_correction1 = 1.0 - beta1**next_step
    bias_correction2 = 1.0 - beta2**next_step

    variance = next_exp_avg_sq
    if bool(param_group.get("amsgrad", False)):
        max_exp_avg_sq = state.get("max_exp_avg_sq")
        previous_max = (
            torch.zeros_like(parameter) if max_exp_avg_sq is None else max_exp_avg_sq.detach().clone()
        )
        variance = torch.maximum(previous_max, variance)

    denominator = variance.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
    candidate = parameter.detach().clone()
    if include_weight_decay:
        candidate.mul_(1.0 - learning_rate * float(param_group["weight_decay"]))
    candidate.addcdiv_(next_exp_avg, denominator, value=-learning_rate / bias_correction1)

    result_dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
    return candidate.to(dtype=result_dtype).sub(parameter.detach().to(dtype=result_dtype))


@torch.no_grad()
def adamw_group_relative_update_moments(
    param_group: Mapping[str, Any],
    optimizer_state: Mapping[nn.Parameter, Mapping[str, Any]],
    *,
    include_weight_decay: bool = False,
) -> tuple[Tensor, Tensor, int]:
    """Return squared candidate-update norm, parameter norm, and active numel.

    Only parameters with gradients participate because AdamW skips parameters
    whose gradient is ``None``. The caller may sum these three values across
    ranks before computing the relative rate.
    """
    parameters = list(param_group["params"])
    if not parameters:
        raise ValueError("CABO optimizer groups must not be empty")

    device = parameters[0].device
    update_norm_sq = torch.zeros((), dtype=torch.float32, device=device)
    parameter_norm_sq = torch.zeros((), dtype=torch.float32, device=device)
    active_numel = 0
    for parameter in parameters:
        delta = adamw_candidate_parameter_delta(
            parameter,
            param_group,
            optimizer_state.get(parameter, {}),
            include_weight_decay=include_weight_decay,
        )
        if delta is None:
            continue
        update_norm_dtype = torch.float64 if delta.dtype == torch.float64 else torch.float32
        parameter_norm_dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
        update_norm = torch.linalg.vector_norm(delta, ord=2, dtype=update_norm_dtype).to(torch.float32)
        parameter_norm = torch.linalg.vector_norm(parameter.detach(), ord=2, dtype=parameter_norm_dtype).to(
            torch.float32
        )
        update_norm_sq.add_(update_norm.square())
        parameter_norm_sq.add_(parameter_norm.square())
        active_numel += parameter.numel()

    return update_norm_sq, parameter_norm_sq, active_numel


def relative_update_rate(update_norm_sq: float, parameter_norm_sq: float) -> float:
    """Convert aggregate squared norms into a dimensionless relative update rate."""
    if update_norm_sq < 0.0 or parameter_norm_sq < 0.0:
        return float("nan")
    if update_norm_sq == 0.0:
        return 0.0
    if parameter_norm_sq == 0.0:
        return float("inf")
    return math.sqrt(update_norm_sq / parameter_norm_sq)


def update_cabo_prompt_relative_update_scales(
    *,
    prompt_rate: float,
    expert_rate: float,
    projection_rate: float,
    prompt_update_ratio: float,
) -> tuple[float, float, dict[str, float]]:
    """Limit each action-side update to the current prompt update divided by a ratio.

    The prompt learning rate stays unchanged. There is no moving average,
    warmup, or reference floor: those could permit an action-side update larger
    than the current prompt update. A zero prompt update therefore gives a zero
    budget to any moving action group. A stationary action group needs no cap.

    These scales assume linear learning-rate scaling. Native parameter rounding
    can violate that assumption, so the caller must verify candidate updates at
    the scaled learning rates before using them as a strict update bound. A
    nonfinite or negative input rate sets ``cabo/update_nonfinite``; the caller
    must skip that optimizer step.
    """
    if not math.isfinite(prompt_update_ratio) or prompt_update_ratio < 1.0:
        raise ValueError("CABO prompt_update_ratio must be finite and at least 1.0")

    finite = all(math.isfinite(rate) and rate >= 0.0 for rate in (prompt_rate, expert_rate, projection_rate))
    limit = prompt_rate / prompt_update_ratio if finite else float("nan")

    def action_scale(rate: float) -> float:
        if not finite or rate == 0.0 or rate <= limit:
            return 1.0
        scale = limit / rate
        # Avoid a one-ULP overshoot even in the linear prediction.
        if scale * rate > limit:
            scale = math.nextafter(scale, 0.0)
        return scale

    expert_scale = action_scale(expert_rate)
    projection_scale = action_scale(projection_rate)
    return (
        expert_scale,
        projection_scale,
        {
            "cabo/prompt_relative_update_rate": prompt_rate,
            "cabo/expert_relative_update_rate": expert_rate,
            "cabo/projection_relative_update_rate": projection_rate,
            "cabo/prompt_update_ratio": prompt_update_ratio,
            "cabo/expert_relative_update_limit": limit,
            "cabo/projection_relative_update_limit": limit,
            "cabo/scaled_expert_relative_update_rate": expert_scale * expert_rate,
            "cabo/scaled_projection_relative_update_rate": projection_scale * projection_rate,
            "cabo/prompt_scale": 1.0,
            "cabo/expert_scale": expert_scale,
            "cabo/projection_scale": projection_scale,
            "cabo/update_nonfinite": float(not finite),
        },
    )


def update_cabo_relative_update_scales(
    controller_group: dict[str, Any],
    *,
    vlm_rate: float,
    expert_rate: float,
    projection_rate: float,
    expert_ratio: float,
    projection_ratio: float,
    ema_decay: float,
    warmup_steps: int,
    vlm_floor_ratio: float,
) -> tuple[float, float, dict[str, float]]:
    """Limit action-side relative AdamW updates using the VLM update as reference."""
    rates = (vlm_rate, expert_rate, projection_rate)
    finite = all(math.isfinite(value) and value >= 0.0 for value in rates)
    previous_expert_scale = float(controller_group.get("cabo_expert_scale", 1.0))
    previous_projection_scale = float(controller_group.get("cabo_projection_scale", 1.0))
    if not finite:
        return (
            previous_expert_scale,
            previous_projection_scale,
            {
                "cabo/vlm_relative_update_rate": vlm_rate,
                "cabo/expert_relative_update_rate": expert_rate,
                "cabo/projection_relative_update_rate": projection_rate,
                "cabo/expert_scale": previous_expert_scale,
                "cabo/projection_scale": previous_projection_scale,
                "cabo/update_nonfinite": 1.0,
            },
        )

    step = int(controller_group.get("cabo_step", 0))
    initialized = bool(controller_group.get("cabo_update_ema_initialized", False))
    vlm_rate_ema = (
        ema_decay * float(controller_group["cabo_vlm_rate_ema"]) + (1.0 - ema_decay) * vlm_rate
        if initialized
        else vlm_rate
    )

    warmup_sum = float(controller_group.get("cabo_vlm_warmup_sum", 0.0))
    warmup_count = int(controller_group.get("cabo_vlm_warmup_count", 0))
    in_warmup = step < warmup_steps
    if in_warmup:
        warmup_sum += vlm_rate
        warmup_count += 1
    warmup_reference = warmup_sum / warmup_count if warmup_count > 0 else vlm_rate_ema
    vlm_floor = vlm_floor_ratio * warmup_reference
    vlm_reference = max(vlm_rate_ema, vlm_floor)

    if in_warmup:
        expert_scale = 1.0
        projection_scale = 1.0
    else:
        expert_scale = 1.0 if expert_rate == 0.0 else min(1.0, expert_ratio * vlm_reference / expert_rate)
        projection_scale = (
            1.0 if projection_rate == 0.0 else min(1.0, projection_ratio * vlm_reference / projection_rate)
        )

    controller_group.update(
        {
            "cabo_step": step + 1,
            "cabo_update_ema_initialized": True,
            "cabo_vlm_rate_ema": vlm_rate_ema,
            "cabo_vlm_warmup_sum": warmup_sum,
            "cabo_vlm_warmup_count": warmup_count,
            "cabo_expert_scale": expert_scale,
            "cabo_projection_scale": projection_scale,
        }
    )
    return (
        expert_scale,
        projection_scale,
        {
            "cabo/vlm_relative_update_rate": vlm_rate,
            "cabo/expert_relative_update_rate": expert_rate,
            "cabo/projection_relative_update_rate": projection_rate,
            "cabo/vlm_relative_update_rate_ema": vlm_rate_ema,
            "cabo/vlm_relative_update_floor": vlm_floor,
            "cabo/vlm_relative_update_reference": vlm_reference,
            "cabo/expert_relative_update_limit": expert_ratio * vlm_reference,
            "cabo/projection_relative_update_limit": projection_ratio * vlm_reference,
            "cabo/scaled_expert_relative_update_rate": expert_scale * expert_rate,
            "cabo/scaled_projection_relative_update_rate": projection_scale * projection_rate,
            "cabo/expert_scale": expert_scale,
            "cabo/projection_scale": projection_scale,
            "cabo/warmup_active": float(in_warmup),
            "cabo/update_nonfinite": 0.0,
        },
    )


@contextmanager
def temporary_optimizer_group_lr_scales(
    optimizer: Optimizer,
    group_scales: Mapping[str, float],
) -> Iterator[dict[str, float]]:
    """Temporarily scale named group learning rates for exactly one optimizer step."""
    if not group_scales:
        yield {}
        return

    unwrapped = unwrap_optimizer(optimizer)
    original_lrs: dict[str, float | Tensor] = {}
    effective_lrs: dict[str, float] = {}
    try:
        for name, scale in group_scales.items():
            if not math.isfinite(scale) or scale < 0.0:
                raise ValueError(
                    f"CABO optimizer group scale must be finite and non-negative, got {scale} for {name!r}"
                )
            group = get_named_param_group(unwrapped, name)
            original_lrs[name] = group["lr"]
            group["lr"] = group["lr"] * scale
            effective_lrs[name] = float(group["lr"])
        yield effective_lrs
    finally:
        for name, learning_rate in original_lrs.items():
            get_named_param_group(unwrapped, name)["lr"] = learning_rate
