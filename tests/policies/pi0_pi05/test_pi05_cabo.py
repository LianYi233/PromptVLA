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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.optim.cabo import (  # noqa: E402
    CABO_ACTION_PROMPT_GROUP,
    CABO_GROUP_NAME,
    CABO_VLM_PROMPT_GROUP,
    temporary_optimizer_group_lr_scales,
)
from lerobot.policies.pi05 import PI05Policy  # noqa: E402


class _FakeAccelerator:
    num_processes = 1

    def reduce(self, value, reduction="mean"):
        raise AssertionError(f"single-process CABO must not reduce moments ({reduction=}, {value=})")


class _ReducingFakeAccelerator:
    num_processes = 2

    def __init__(self):
        self.reduce_shapes = []

    def reduce(self, value, reduction="mean"):
        assert reduction == "sum"
        self.reduce_shapes.append(value.shape)
        return value * self.num_processes


class _TinyCABOPolicy(nn.Module):
    compute_optimizer_step_control = PI05Policy.compute_optimizer_step_control
    validate_optimizer_step_control = PI05Policy.validate_optimizer_step_control

    def __init__(
        self,
        *,
        prompt_update_ratio: float = 2.0,
        dtype=torch.float64,
        training_stage="flow",
        bridge_active=False,
    ):
        super().__init__()
        self.vlm_prompt_weight = nn.Parameter(torch.tensor([10.0], dtype=dtype))
        self.action_prompt_weight = nn.Parameter(torch.tensor([1.0], dtype=dtype))
        self._stage1_bridge_active = bridge_active
        self.config = SimpleNamespace(
            cabo_enabled=True,
            cabo_prompt_update_ratio=prompt_update_ratio,
            training_stage=training_stage,
            # Legacy settings must not relax the current VLM prompt update bound.
            cabo_expert_update_ratio=2.0,
            cabo_projection_update_ratio=5.0,
            cabo_vlm_update_ema_decay=0.99,
            cabo_update_warmup_steps=100,
            cabo_vlm_update_floor_ratio=0.1,
        )

    def _cabo_parameter_groups(self):
        return [self.vlm_prompt_weight], [self.action_prompt_weight]


def _make_optimizer(policy: _TinyCABOPolicy, *, weight_decay: float = 0.0, betas=(0.0, 0.0)):
    return torch.optim.AdamW(
        [
            {
                "params": [policy.vlm_prompt_weight],
                CABO_GROUP_NAME: CABO_VLM_PROMPT_GROUP,
            },
            {
                "params": [policy.action_prompt_weight],
                CABO_GROUP_NAME: CABO_ACTION_PROMPT_GROUP,
            },
        ],
        lr=0.1,
        betas=betas,
        eps=1e-8,
        weight_decay=weight_decay,
        foreach=False,
    )


def _set_unit_gradients(policy: _TinyCABOPolicy) -> None:
    for parameter in policy.parameters():
        parameter.grad = torch.ones_like(parameter)


def test_pi05_cabo_scales_action_prompt_from_relative_adamw_updates_without_touching_gradients():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)
    _set_unit_gradients(policy)
    gradients_before = [parameter.grad.clone() for parameter in policy.parameters()]

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())

    # The first AdamW learning delta is 0.1 for both scalars. Relative rates are
    # VLM prompt=0.01 and action prompt=0.1; the default ratio limits action to 0.005.
    assert control.group_scales == pytest.approx({CABO_ACTION_PROMPT_GROUP: 0.05})
    assert control.metrics["cabo/vlm_prompt_relative_update_rate"] == pytest.approx(0.01)
    assert control.metrics["cabo/action_prompt_relative_update_rate"] == pytest.approx(0.1)
    assert control.metrics["cabo/action_prompt_relative_update_limit"] == pytest.approx(0.005)
    assert control.metrics["cabo/vlm_prompt_scale"] == 1.0
    assert control.metrics["cabo/prompt_update_ratio"] == 2.0
    assert control.metrics["cabo/active"] == 1.0
    assert control.metrics["cabo/action_only_bypass"] == 0.0
    assert not control.skip_optimizer_step
    for parameter, gradient_before in zip(policy.parameters(), gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)


@pytest.mark.parametrize("ratio", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("training_stage,bridge_active", [("flow", False), ("next_action", True)])
def test_pi05_cabo_limits_the_real_action_prompt_step_and_leaves_vlm_prompt_full(
    ratio, dtype, training_stage, bridge_active
):
    policy = _TinyCABOPolicy(
        prompt_update_ratio=ratio,
        dtype=dtype,
        training_stage=training_stage,
        bridge_active=bridge_active,
    )
    optimizer = _make_optimizer(policy)
    reference_prompt = nn.Parameter(policy.vlm_prompt_weight.detach().clone())
    reference_optimizer = torch.optim.AdamW(
        [reference_prompt], lr=0.1, betas=(0.0, 0.0), eps=1e-8, weight_decay=0.0, foreach=False
    )
    reference_prompt.grad = torch.ones_like(reference_prompt)
    reference_optimizer.step()
    _set_unit_gradients(policy)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())
    with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
        optimizer.step()

    torch.testing.assert_close(policy.vlm_prompt_weight, reference_prompt, rtol=0, atol=0)
    rates = [
        (parameter.detach().double() - old.double()).norm().item() / old.double().norm().item()
        for parameter, old in zip(policy.parameters(), before, strict=True)
    ]
    assert rates[0] >= ratio * rates[1] - 1e-8
    assert control.metrics["cabo/scaled_action_prompt_relative_update_rate"] == pytest.approx(rates[1])
    assert all(group["lr"] == 0.1 for group in optimizer.param_groups)


def test_pi05_cabo_leaves_an_already_slower_action_prompt_at_its_configured_lr():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)
    optimizer.param_groups[1]["lr"] = 0.001
    _set_unit_gradients(policy)

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())

    assert control.group_scales == {CABO_ACTION_PROMPT_GROUP: 1.0}
    assert control.metrics["cabo/candidate_verification_passes"] == 0.0
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[1]["lr"] == 0.001


def test_pi05_cabo_rechecks_bfloat16_updates_when_lr_scaling_rounds_above_the_vlm_prompt_budget():
    policy = _TinyCABOPolicy(prompt_update_ratio=1.0)
    policy.action_prompt_weight = nn.Parameter(torch.tensor([0.01], dtype=torch.bfloat16))
    optimizer = _make_optimizer(policy)
    # A full bfloat16 step moves the action prompt by 0.0001220703125. Scaling
    # its LR by 0.33 moves it by 0.00006103515625: half the full delta, not 0.33.
    action_rate = 0.0001220703125 / policy.action_prompt_weight.item()
    optimizer.param_groups[0]["lr"] = action_rate * 0.33 * policy.vlm_prompt_weight.item()
    optimizer.param_groups[1]["lr"] = 0.0001
    _set_unit_gradients(policy)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())
    with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
        optimizer.step()

    assert control.metrics["cabo/candidate_verification_passes"] >= 2.0
    assert control.group_scales[CABO_ACTION_PROMPT_GROUP] < 0.33
    rates = [
        (parameter.detach().double() - old.double()).norm().item() / old.double().norm().item()
        for parameter, old in zip(policy.parameters(), before, strict=True)
    ]
    assert rates[0] > 0.0
    assert rates[0] >= rates[1]


@pytest.mark.parametrize("missing_gradient", [False, True])
@pytest.mark.parametrize("training_stage,bridge_active", [("flow", False), ("next_action", True)])
def test_pi05_cabo_zero_vlm_prompt_update_stops_action_prompt(
    missing_gradient, training_stage, bridge_active
):
    policy = _TinyCABOPolicy(training_stage=training_stage, bridge_active=bridge_active)
    optimizer = _make_optimizer(policy)
    _set_unit_gradients(policy)
    policy.vlm_prompt_weight.grad = None if missing_gradient else torch.zeros_like(policy.vlm_prompt_weight)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())
    with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
        optimizer.step()

    assert control.metrics["cabo/vlm_prompt_relative_update_rate"] == 0.0
    assert control.group_scales == {CABO_ACTION_PROMPT_GROUP: 0.0}
    assert not control.skip_optimizer_step
    for parameter, previous in zip(policy.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, previous, rtol=0, atol=0)


def test_pi05_cabo_action_only_stage_bypasses_vlm_budget_and_updates_action_prompt_in_full():
    policy = _TinyCABOPolicy(training_stage="next_action")
    optimizer = _make_optimizer(policy)
    policy.action_prompt_weight.grad = torch.ones_like(policy.action_prompt_weight)
    vlm_before = policy.vlm_prompt_weight.detach().clone()
    reference_prompt = nn.Parameter(policy.action_prompt_weight.detach().clone())
    reference_optimizer = torch.optim.AdamW(
        [reference_prompt], lr=0.1, betas=(0.0, 0.0), eps=1e-8, weight_decay=0.0, foreach=False
    )
    reference_prompt.grad = torch.ones_like(reference_prompt)
    reference_optimizer.step()
    accelerator = _ReducingFakeAccelerator()

    control = policy.compute_optimizer_step_control({}, optimizer, accelerator)
    with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
        optimizer.step()

    assert control.metrics["cabo/active"] == 0.0
    assert control.metrics["cabo/action_only_bypass"] == 1.0
    assert not control.skip_optimizer_step
    assert not accelerator.reduce_shapes
    torch.testing.assert_close(policy.vlm_prompt_weight, vlm_before, rtol=0, atol=0)
    torch.testing.assert_close(policy.action_prompt_weight, reference_prompt, rtol=0, atol=0)


def test_pi05_cabo_disabled_leaves_optimizer_step_uncontrolled():
    policy = _TinyCABOPolicy()
    policy.config.cabo_enabled = False
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    policy.validate_optimizer_step_control(optimizer)
    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())

    assert not control.group_scales
    assert not control.skip_optimizer_step


def test_pi05_cabo_vlm_prompt_momentum_still_provides_an_update_budget_with_zero_gradient():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy, betas=(0.9, 0.99))
    _set_unit_gradients(policy)
    optimizer.step()
    policy.vlm_prompt_weight.grad.zero_()

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())

    assert control.metrics["cabo/vlm_prompt_relative_update_rate"] > 0.0
    assert 0.0 < control.group_scales[CABO_ACTION_PROMPT_GROUP] < 1.0


def test_pi05_cabo_does_not_reuse_legacy_vlm_ema_or_warmup_budget():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)
    optimizer.param_groups[0].update(cabo_step=100, cabo_vlm_update_ema=100.0)
    _set_unit_gradients(policy)
    first = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())
    policy.vlm_prompt_weight.grad.zero_()

    second = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())

    assert first.group_scales[CABO_ACTION_PROMPT_GROUP] < 1.0
    assert second.group_scales == {CABO_ACTION_PROMPT_GROUP: 0.0}
    assert optimizer.param_groups[0]["cabo_step"] == 100
    assert optimizer.param_groups[0]["cabo_vlm_update_ema"] == 100.0


def test_pi05_cabo_excludes_weight_decay_from_measured_learning_update_rate():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy, weight_decay=0.5)
    for parameter in policy.parameters():
        parameter.grad = torch.zeros_like(parameter)

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())

    assert control.metrics["cabo/vlm_prompt_relative_update_rate"] == 0.0
    assert control.metrics["cabo/action_prompt_relative_update_rate"] == 0.0
    assert control.group_scales[CABO_ACTION_PROMPT_GROUP] == 1.0


def test_pi05_cabo_reduces_candidate_and_verified_update_moments_for_multi_process_training():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)
    accelerator = _ReducingFakeAccelerator()
    _set_unit_gradients(policy)

    control = policy.compute_optimizer_step_control({}, optimizer, accelerator)

    assert accelerator.reduce_shapes[0] == (4,)
    assert accelerator.reduce_shapes[1:]
    assert all(shape == (2,) for shape in accelerator.reduce_shapes[1:])
    single_process_control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())
    assert control.group_scales == single_process_control.group_scales


@pytest.mark.parametrize("group", ["vlm_prompt_weight", "action_prompt_weight"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_pi05_cabo_nonfinite_update_in_any_group_skips_without_mutating_optimizer_state(group, value):
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)
    _set_unit_gradients(policy)
    getattr(policy, group).grad.fill_(value)
    groups_before = [dict(group) for group in optimizer.param_groups]

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())

    assert control.skip_optimizer_step
    assert control.metrics["cabo/update_nonfinite"] == 1.0
    assert control.metrics["optimizer_step/nonfinite_cabo_update"] == 1.0
    assert not optimizer.state
    for group, previous in zip(optimizer.param_groups, groups_before, strict=True):
        for key, value in previous.items():
            if key != "params":
                assert group[key] == value


def test_pi05_cabo_nonfinite_update_from_another_rank_skips_the_step():
    class RemoteNonfiniteAccelerator(_ReducingFakeAccelerator):
        def reduce(self, value, reduction="mean"):
            result = super().reduce(value, reduction=reduction)
            result[0] = float("nan")
            return result

    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)
    _set_unit_gradients(policy)

    control = policy.compute_optimizer_step_control({}, optimizer, RemoteNonfiniteAccelerator())

    assert control.skip_optimizer_step
    assert control.metrics["optimizer_step/nonfinite_cabo_update"] == 1.0


def test_pi05_cabo_optimizer_contract_is_validated_before_training():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)

    policy.validate_optimizer_step_control(optimizer)

    assert all(group["foreach"] is False for group in optimizer.param_groups)
    assert all(group["fused"] is False for group in optimizer.param_groups)


def test_pi05_cabo_optimizer_contract_rejects_wrong_optimizer_or_groups():
    policy = _TinyCABOPolicy()
    with pytest.raises(TypeError, match="AdamW"):
        policy.validate_optimizer_step_control(torch.optim.SGD(policy.parameters(), lr=0.1))

    unnamed_optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="two named"):
        policy.validate_optimizer_step_control(unnamed_optimizer)

    swapped_optimizer = torch.optim.AdamW(
        [
            {"params": [policy.action_prompt_weight], CABO_GROUP_NAME: CABO_VLM_PROMPT_GROUP},
            {"params": [policy.vlm_prompt_weight], CABO_GROUP_NAME: CABO_ACTION_PROMPT_GROUP},
        ],
        lr=0.1,
        foreach=False,
    )
    with pytest.raises(ValueError, match="does not match"):
        policy.validate_optimizer_step_control(swapped_optimizer)
