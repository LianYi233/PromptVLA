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

import pytest
import torch
from torch import nn

from lerobot.optim.cabo import (
    CABO_ACTION_EXPERT_GROUP,
    CABO_ACTION_PROJECTION_GROUP,
    CABO_GROUP_NAME,
    CABO_PROMPT_GROUP,
    CABO_VLM_GROUP,
    adamw_candidate_parameter_delta,
    adamw_group_relative_update_moments,
    get_named_param_group,
    relative_update_rate,
    temporary_optimizer_group_lr_scales,
    update_cabo_prompt_relative_update_scales,
    update_cabo_relative_update_scales,
    validate_adamw_param_group,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adamw_validation_rejects_tensor_learning_rates(dtype: torch.dtype):
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=torch.tensor(0.001, dtype=dtype), foreach=False)

    with pytest.raises(RuntimeError, match="tensor learning rates are not supported"):
        validate_adamw_param_group(optimizer.param_groups[0])


def test_adamw_candidate_rejects_tensor_learning_rate():
    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter], lr=torch.tensor(0.001), foreach=False)

    with pytest.raises(RuntimeError, match="tensor learning rates are not supported"):
        adamw_candidate_parameter_delta(parameter, optimizer.param_groups[0], {})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("existing_steps", [0, 2])
def test_adamw_candidate_delta_matches_real_optimizer_step(existing_steps: int, dtype: torch.dtype):
    parameter = nn.Parameter(torch.tensor([1.5, -0.75], dtype=dtype))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], CABO_GROUP_NAME: CABO_ACTION_EXPERT_GROUP}],
        lr=3e-3,
        betas=(0.8, 0.95),
        eps=1e-9,
        weight_decay=0.2,
        amsgrad=True,
        foreach=False,
    )

    for index in range(existing_steps):
        parameter.grad = torch.tensor([0.2 + index, -0.4 - index], dtype=dtype)
        optimizer.step()
        optimizer.zero_grad()

    parameter.grad = torch.tensor([0.7, -1.1], dtype=dtype)
    group = get_named_param_group(optimizer, CABO_ACTION_EXPERT_GROUP)
    expected_delta = adamw_candidate_parameter_delta(parameter, group, optimizer.state.get(parameter, {}))
    before = parameter.detach().clone()

    optimizer.step()

    actual_delta = parameter.detach().to(dtype=expected_delta.dtype) - before.to(dtype=expected_delta.dtype)
    torch.testing.assert_close(actual_delta, expected_delta, rtol=0.0, atol=0.0)


def test_candidate_learning_delta_excludes_weight_decay():
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], CABO_GROUP_NAME: CABO_ACTION_EXPERT_GROUP}],
        lr=0.1,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.3,
        foreach=False,
    )
    parameter.grad = torch.tensor([1.0], dtype=torch.float64)
    group = get_named_param_group(optimizer, CABO_ACTION_EXPERT_GROUP)

    full_delta = adamw_candidate_parameter_delta(parameter, group, {})
    learning_delta = adamw_candidate_parameter_delta(
        parameter,
        group,
        {},
        include_weight_decay=False,
    )

    assert learning_delta.item() == pytest.approx(-0.1)
    assert full_delta.item() == pytest.approx(-0.16)


def test_group_relative_update_moments_ignore_parameters_without_gradients():
    active = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
    inactive = nn.Parameter(torch.tensor([100.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [{"params": [active, inactive], CABO_GROUP_NAME: CABO_VLM_GROUP}],
        lr=0.1,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )
    active.grad = torch.tensor([1.0, -1.0], dtype=torch.float64)
    group = get_named_param_group(optimizer, CABO_VLM_GROUP)

    update_norm_sq, parameter_norm_sq, active_numel = adamw_group_relative_update_moments(
        group, optimizer.state
    )

    assert update_norm_sq.item() == pytest.approx(0.02)
    assert parameter_norm_sq.item() == pytest.approx(25.0)
    assert active_numel == 2
    assert relative_update_rate(update_norm_sq.item(), parameter_norm_sq.item()) == pytest.approx(
        (0.02 / 25.0) ** 0.5
    )


def test_relative_update_controller_limits_expert_and_projection_after_warmup():
    controller_group = {}
    common = {
        "vlm_rate": 0.01,
        "expert_rate": 0.05,
        "projection_rate": 0.10,
        "expert_ratio": 2.0,
        "projection_ratio": 5.0,
        "ema_decay": 0.0,
        "warmup_steps": 2,
        "vlm_floor_ratio": 0.1,
    }

    for _ in range(2):
        expert_scale, projection_scale, metrics = update_cabo_relative_update_scales(
            controller_group, **common
        )
        assert expert_scale == pytest.approx(1.0)
        assert projection_scale == pytest.approx(1.0)
        assert metrics["cabo/warmup_active"] == 1.0

    expert_scale, projection_scale, metrics = update_cabo_relative_update_scales(controller_group, **common)

    assert expert_scale == pytest.approx(0.4)
    assert projection_scale == pytest.approx(0.5)
    assert metrics["cabo/scaled_expert_relative_update_rate"] == pytest.approx(0.02)
    assert metrics["cabo/scaled_projection_relative_update_rate"] == pytest.approx(0.05)
    assert metrics["cabo/warmup_active"] == 0.0


def test_relative_update_controller_uses_warmup_floor_when_vlm_becomes_stationary():
    controller_group = {}
    update_cabo_relative_update_scales(
        controller_group,
        vlm_rate=0.1,
        expert_rate=0.1,
        projection_rate=0.1,
        expert_ratio=2.0,
        projection_ratio=5.0,
        ema_decay=0.0,
        warmup_steps=1,
        vlm_floor_ratio=0.1,
    )

    expert_scale, projection_scale, metrics = update_cabo_relative_update_scales(
        controller_group,
        vlm_rate=0.0,
        expert_rate=0.1,
        projection_rate=0.1,
        expert_ratio=2.0,
        projection_ratio=5.0,
        ema_decay=0.0,
        warmup_steps=1,
        vlm_floor_ratio=0.1,
    )

    assert metrics["cabo/vlm_relative_update_floor"] == pytest.approx(0.01)
    assert expert_scale == pytest.approx(0.2)
    assert projection_scale == pytest.approx(0.5)


def test_relative_update_controller_preserves_state_on_nonfinite_rate():
    controller_group = {
        "cabo_step": 3,
        "cabo_expert_scale": 0.25,
        "cabo_projection_scale": 0.5,
    }
    state_before = controller_group.copy()

    expert_scale, projection_scale, metrics = update_cabo_relative_update_scales(
        controller_group,
        vlm_rate=float("inf"),
        expert_rate=0.1,
        projection_rate=0.2,
        expert_ratio=2.0,
        projection_ratio=5.0,
        ema_decay=0.95,
        warmup_steps=100,
        vlm_floor_ratio=0.1,
    )

    assert controller_group == state_before
    assert expert_scale == pytest.approx(0.25)
    assert projection_scale == pytest.approx(0.5)
    assert metrics["cabo/update_nonfinite"] == 1.0


@pytest.mark.parametrize("ratio", [1.0, 2.0, 10.0])
@pytest.mark.parametrize(
    "expert_rate, projection_rate", [(0.05, 0.10), (0.001, 0.10), (0.001, 0.001), (0.0, 0.0)]
)
def test_prompt_controller_caps_both_action_groups_against_current_prompt(
    ratio: float, expert_rate: float, projection_rate: float
):
    prompt_rate = 0.01
    expert_scale, projection_scale, metrics = update_cabo_prompt_relative_update_scales(
        prompt_rate=prompt_rate,
        expert_rate=expert_rate,
        projection_rate=projection_rate,
        prompt_update_ratio=ratio,
    )

    limit = prompt_rate / ratio
    for rate, scale in ((expert_rate, expert_scale), (projection_rate, projection_scale)):
        assert 0.0 <= scale <= 1.0
        assert rate * scale <= limit
        if rate <= limit:
            assert scale == 1.0
        else:
            assert scale == pytest.approx(limit / rate)
    assert metrics["cabo/prompt_relative_update_rate"] == prompt_rate
    assert metrics["cabo/prompt_update_ratio"] == ratio
    assert metrics["cabo/prompt_scale"] == 1.0
    assert metrics["cabo/expert_relative_update_limit"] == limit
    assert metrics["cabo/projection_relative_update_limit"] == limit
    assert metrics["cabo/scaled_expert_relative_update_rate"] == expert_scale * expert_rate
    assert metrics["cabo/scaled_projection_relative_update_rate"] == projection_scale * projection_rate
    assert metrics["cabo/update_nonfinite"] == 0.0


@pytest.mark.parametrize("expert_rate, projection_rate", [(0.1, 0.2), (0.0, 0.2), (0.1, 0.0), (0.0, 0.0)])
def test_prompt_controller_zero_reference_stops_only_moving_action_groups(
    expert_rate: float, projection_rate: float
):
    expert_scale, projection_scale, metrics = update_cabo_prompt_relative_update_scales(
        prompt_rate=0.0,
        expert_rate=expert_rate,
        projection_rate=projection_rate,
        prompt_update_ratio=1.0,
    )

    assert expert_scale == (0.0 if expert_rate else 1.0)
    assert projection_scale == (0.0 if projection_rate else 1.0)
    assert metrics["cabo/scaled_expert_relative_update_rate"] == 0.0
    assert metrics["cabo/scaled_projection_relative_update_rate"] == 0.0
    assert metrics["cabo/update_nonfinite"] == 0.0


def test_prompt_controller_missing_prompt_gradient_has_zero_update_budget():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], CABO_GROUP_NAME: CABO_PROMPT_GROUP}], lr=0.1, foreach=False
    )
    group = get_named_param_group(optimizer, CABO_PROMPT_GROUP)
    update_sq, parameter_sq, active_numel = adamw_group_relative_update_moments(group, optimizer.state)

    expert_scale, projection_scale, metrics = update_cabo_prompt_relative_update_scales(
        prompt_rate=relative_update_rate(update_sq.item(), parameter_sq.item()),
        expert_rate=0.1,
        projection_rate=0.2,
        prompt_update_ratio=1.0,
    )

    assert active_numel == 0
    assert expert_scale == projection_scale == 0.0
    assert metrics["cabo/prompt_relative_update_rate"] == 0.0


@pytest.mark.parametrize("invalid_rate", [float("nan"), float("inf"), -float("inf"), -0.1])
@pytest.mark.parametrize("group", ["prompt_rate", "expert_rate", "projection_rate"])
def test_prompt_controller_flags_invalid_rates_for_caller_to_skip(group: str, invalid_rate: float):
    rates = {"prompt_rate": 0.01, "expert_rate": 0.1, "projection_rate": 0.2}
    rates[group] = invalid_rate

    expert_scale, projection_scale, metrics = update_cabo_prompt_relative_update_scales(
        **rates, prompt_update_ratio=1.0
    )

    assert expert_scale == projection_scale == 1.0
    assert metrics["cabo/update_nonfinite"] == 1.0


@pytest.mark.parametrize("ratio", [0.0, 0.5, -1.0, float("nan"), float("inf"), -float("inf")])
def test_prompt_controller_rejects_invalid_ratio(ratio: float):
    with pytest.raises(ValueError, match="finite and at least 1.0"):
        update_cabo_prompt_relative_update_scales(
            prompt_rate=0.01, expert_rate=0.1, projection_rate=0.2, prompt_update_ratio=ratio
        )


def test_prompt_controller_uses_current_reference_without_history():
    common = {"expert_rate": 0.1, "projection_rate": 0.2, "prompt_update_ratio": 1.0}
    update_cabo_prompt_relative_update_scales(prompt_rate=0.5, **common)

    expert_scale, projection_scale, metrics = update_cabo_prompt_relative_update_scales(
        prompt_rate=0.001, **common
    )

    assert expert_scale == pytest.approx(0.01)
    assert projection_scale == pytest.approx(0.005)
    assert metrics["cabo/expert_relative_update_limit"] == 0.001


def test_prompt_controller_linear_scale_requires_native_dtype_candidate_verification():
    parameter = nn.Parameter(torch.tensor([0.01], dtype=torch.bfloat16))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], CABO_GROUP_NAME: CABO_ACTION_EXPERT_GROUP}],
        lr=0.0001,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )
    group = get_named_param_group(optimizer, CABO_ACTION_EXPERT_GROUP)
    original_delta = adamw_candidate_parameter_delta(parameter, group, {}, include_weight_decay=False)
    expert_rate = abs(original_delta.item() / parameter.item())
    prompt_rate = 0.33 * expert_rate
    expert_scale, _, metrics = update_cabo_prompt_relative_update_scales(
        prompt_rate=prompt_rate,
        expert_rate=expert_rate,
        projection_rate=0.0,
        prompt_update_ratio=1.0,
    )

    with temporary_optimizer_group_lr_scales(optimizer, {CABO_ACTION_EXPERT_GROUP: expert_scale}):
        scaled_delta = adamw_candidate_parameter_delta(parameter, group, {}, include_weight_decay=False)
        scaled_rate = abs(scaled_delta.item() / parameter.item())
        before = parameter.detach().float().clone()
        optimizer.step()

    # BF16 rounding makes the true motion larger than the linear prediction.
    assert metrics["cabo/scaled_expert_relative_update_rate"] <= prompt_rate
    assert scaled_rate > prompt_rate
    torch.testing.assert_close(parameter.detach().float() - before, scaled_delta, rtol=0.0, atol=0.0)


def test_temporary_group_lr_scales_apply_and_restore_both_action_groups():
    vlm_parameter = nn.Parameter(torch.tensor([1.0]))
    expert_parameter = nn.Parameter(torch.tensor([1.0]))
    projection_parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW(
        [
            {"params": [vlm_parameter], CABO_GROUP_NAME: CABO_VLM_GROUP},
            {"params": [expert_parameter], CABO_GROUP_NAME: CABO_ACTION_EXPERT_GROUP},
            {"params": [projection_parameter], CABO_GROUP_NAME: CABO_ACTION_PROJECTION_GROUP},
        ],
        lr=0.1,
    )

    with temporary_optimizer_group_lr_scales(
        optimizer,
        {
            CABO_ACTION_EXPERT_GROUP: 0.25,
            CABO_ACTION_PROJECTION_GROUP: 0.5,
        },
    ) as effective_lrs:
        assert effective_lrs == pytest.approx(
            {
                CABO_ACTION_EXPERT_GROUP: 0.025,
                CABO_ACTION_PROJECTION_GROUP: 0.05,
            }
        )
        assert get_named_param_group(optimizer, CABO_VLM_GROUP)["lr"] == pytest.approx(0.1)

    assert all(group["lr"] == pytest.approx(0.1) for group in optimizer.param_groups)
