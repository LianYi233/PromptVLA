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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("datasets")

from lerobot.optim.cabo import OptimizerStepControl  # noqa: E402
from lerobot.scripts.lerobot_train import (  # noqa: E402
    _clip_policy_gradients,
    _normalize_policy_loss_for_distributed_training,
    update_policy,
)


class _PolicyWithCustomGradientClipping(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.clip_calls = 0

    def forward(self, batch):
        return self.weight * batch, {"forward_metric": 2.0}

    @torch.no_grad()
    def clip_gradients(self, accelerator=None) -> dict[str, float]:
        if accelerator is not None:
            accelerator.events.append("custom_clip")
        self.clip_calls += 1
        self.weight.grad.mul_(0.25)
        return {"custom_clip_applied": 1.0}


class _PolicyWithoutCustomGradientClipping(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return self.weight * batch, {}


class _PolicyWithOptimizerStepControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_weight = nn.Parameter(torch.tensor(0.0))
        self.action_weight = nn.Parameter(torch.tensor(0.0))
        self.control_calls = 0

    def forward(self, batch):
        return (self.vlm_weight + self.action_weight) * batch, {}

    def compute_optimizer_step_control(self, batch, optimizer, accelerator):
        _ = batch, optimizer, accelerator
        self.control_calls += 1
        return {"action": 0.25}, {"cabo/action_scale": 0.25}


class _PolicyWithInactiveOptimizerStepControl(_PolicyWithOptimizerStepControl):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(cabo_active=False)


class _PolicyWithSkippingOptimizerStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.update_calls = 0

    def forward(self, batch):
        return self.weight * batch, {}

    def compute_optimizer_step_control(self, batch, optimizer, accelerator):
        _ = batch, optimizer, accelerator
        return OptimizerStepControl(
            metrics={"optimizer_step/skipped": 1.0},
            skip_optimizer_step=True,
        )

    def update(self):
        self.update_calls += 1


class _CountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class _CountingGradScaler:
    def __init__(self):
        self.update_calls = 0

    def update(self):
        self.update_calls += 1


class _FakeAccelerator:
    def __init__(self):
        self.num_processes = 1
        self.unscale_calls = 0
        self.global_clip_calls = 0
        self.no_sync_calls = 0
        self.events = []

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        self.events.append("backward")
        loss.backward()

    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        _ = keep_fp32_wrapper
        return policy

    def unscale_gradients(self, optimizer):
        _ = optimizer
        self.unscale_calls += 1
        self.events.append("unscale")

    def clip_grad_norm_(self, parameters, max_norm):
        self.global_clip_calls += 1
        self.events.append("global_clip")
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def no_sync(self, policy):
        _ = policy
        self.no_sync_calls += 1
        return nullcontext()


class _PolicyWithDistributedLossNormalizer(nn.Module):
    def __init__(self, local_normalizer: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.local_normalizer = local_normalizer

    def get_distributed_loss_normalizer(self, batch):
        _ = batch
        return self.weight.new_tensor(self.local_normalizer)


class _FakeDistributedLossAccelerator:
    num_processes = 2

    def __init__(self, global_normalizer: float):
        self.global_normalizer = global_normalizer
        self.local_normalizers = []

    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        _ = keep_fp32_wrapper
        return policy

    def reduce(self, value, reduction):
        assert reduction == "sum"
        self.local_normalizers.append(value.item())
        return value.new_tensor(self.global_normalizer)


def test_distributed_loss_normalizer_recovers_global_valid_element_mean_gradient():
    rank_0 = _PolicyWithDistributedLossNormalizer(local_normalizer=1)
    rank_1 = _PolicyWithDistributedLossNormalizer(local_normalizer=3)
    accelerator_0 = _FakeDistributedLossAccelerator(global_normalizer=4)
    accelerator_1 = _FakeDistributedLossAccelerator(global_normalizer=4)

    rank_0_loss = _normalize_policy_loss_for_distributed_training(
        rank_0,
        batch={},
        loss=rank_0.weight * 2,
        accelerator=accelerator_0,
    )
    rank_1_loss = _normalize_policy_loss_for_distributed_training(
        rank_1,
        batch={},
        loss=rank_1.weight * 4,
        accelerator=accelerator_1,
    )
    rank_0_loss.backward()
    rank_1_loss.backward()

    ddp_averaged_gradient = (rank_0.weight.grad + rank_1.weight.grad) / 2
    expected_global_gradient = torch.tensor((1 * 2 + 3 * 4) / 4)
    torch.testing.assert_close(ddp_averaged_gradient, expected_global_gradient)
    assert accelerator_0.local_normalizers == [1.0]
    assert accelerator_1.local_normalizers == [3.0]


@pytest.mark.parametrize("global_normalizer", [0.0, 3.0])
def test_distributed_loss_normalizer_keeps_all_padding_rank_graph_connected(global_normalizer):
    policy = _PolicyWithDistributedLossNormalizer(local_normalizer=0)
    accelerator = _FakeDistributedLossAccelerator(global_normalizer=global_normalizer)

    loss = _normalize_policy_loss_for_distributed_training(
        policy,
        batch={},
        loss=policy.weight * 0,
        accelerator=accelerator,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert policy.weight.grad is not None
    assert policy.weight.grad.item() == 0.0


def test_policy_specific_hook_replaces_global_gradient_clipping():
    policy = _PolicyWithCustomGradientClipping()
    policy.weight.grad = torch.tensor(4.0)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    accelerator = _FakeAccelerator()

    grad_norm, metrics = _clip_policy_gradients(
        policy=policy,
        optimizer=optimizer,
        grad_clip_norm=1.0,
        accelerator=accelerator,
    )

    assert grad_norm.item() == pytest.approx(4.0)
    assert policy.weight.grad.item() == pytest.approx(1.0)
    assert metrics == {"custom_clip_applied": 1.0}
    assert policy.clip_calls == 1
    assert accelerator.unscale_calls == 1
    assert accelerator.global_clip_calls == 0
    assert accelerator.events == ["unscale", "custom_clip"]


def test_policy_without_custom_hook_keeps_global_gradient_clipping():
    policy = _PolicyWithoutCustomGradientClipping()
    policy.weight.grad = torch.tensor(4.0)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    accelerator = _FakeAccelerator()

    grad_norm, metrics = _clip_policy_gradients(
        policy=policy,
        optimizer=optimizer,
        grad_clip_norm=1.0,
        accelerator=accelerator,
    )

    assert grad_norm.item() == pytest.approx(4.0)
    assert policy.weight.grad.item() == pytest.approx(1.0)
    assert metrics == {}
    assert accelerator.unscale_calls == 0
    assert accelerator.global_clip_calls == 1


def test_update_policy_calls_custom_hook_and_merges_clipping_metrics():
    policy = _PolicyWithCustomGradientClipping()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
    accelerator = _FakeAccelerator()
    train_metrics = SimpleNamespace()

    train_metrics, output_dict = update_policy(
        train_metrics=train_metrics,
        policy=policy,
        batch=torch.tensor(4.0),
        optimizer=optimizer,
        grad_clip_norm=1.0,
        accelerator=accelerator,
    )

    assert policy.clip_calls == 1
    assert accelerator.events == ["backward", "unscale", "custom_clip"]
    assert accelerator.global_clip_calls == 0
    assert train_metrics.grad_norm == pytest.approx(4.0)
    assert output_dict == {"forward_metric": 2.0, "custom_clip_applied": 1.0}


def test_update_policy_applies_post_preconditioner_group_scale_and_restores_lr():
    policy = _PolicyWithOptimizerStepControl()
    optimizer = torch.optim.SGD(
        [
            {"params": [policy.vlm_weight], "lr": 0.1, "name": "vlm"},
            {"params": [policy.action_weight], "lr": 0.1, "name": "action"},
        ]
    )
    accelerator = _FakeAccelerator()
    train_metrics = SimpleNamespace()

    _, output_dict = update_policy(
        train_metrics=train_metrics,
        policy=policy,
        batch=torch.tensor(1.0),
        optimizer=optimizer,
        grad_clip_norm=0.0,
        accelerator=accelerator,
    )

    assert policy.vlm_weight.item() == pytest.approx(-0.1)
    assert policy.action_weight.item() == pytest.approx(-0.025)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.1)
    assert policy.control_calls == 1
    assert accelerator.no_sync_calls == 1
    assert output_dict == {"cabo/action_scale": 0.25}


def test_update_policy_skips_inactive_cabo_controller():
    policy = _PolicyWithInactiveOptimizerStepControl()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    accelerator = _FakeAccelerator()
    train_metrics = SimpleNamespace()

    _, output_dict = update_policy(
        train_metrics=train_metrics,
        policy=policy,
        batch=torch.tensor(1.0),
        optimizer=optimizer,
        grad_clip_norm=0.0,
        accelerator=accelerator,
    )

    assert policy.vlm_weight.item() == pytest.approx(-0.1)
    assert policy.action_weight.item() == pytest.approx(-0.1)
    assert policy.control_calls == 0
    assert accelerator.no_sync_calls == 0
    assert output_dict == {}


def test_update_policy_structured_control_skips_optimizer_scheduler_and_policy_update():
    policy = _PolicyWithSkippingOptimizerStep()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    policy.weight.grad = torch.tensor(1.0)
    optimizer.step()
    optimizer.zero_grad()
    parameter_before = policy.weight.detach().clone()
    state_before = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in optimizer.state[policy.weight].items()
    }
    scheduler = _CountingScheduler()
    accelerator = _FakeAccelerator()
    accelerator.scaler = _CountingGradScaler()
    train_metrics = SimpleNamespace()

    _, output_dict = update_policy(
        train_metrics=train_metrics,
        policy=policy,
        batch=torch.tensor(2.0),
        optimizer=optimizer,
        grad_clip_norm=0.0,
        accelerator=accelerator,
        lr_scheduler=scheduler,
    )

    torch.testing.assert_close(policy.weight, parameter_before)
    for key, value_before in state_before.items():
        value_after = optimizer.state[policy.weight][key]
        if isinstance(value_before, torch.Tensor):
            torch.testing.assert_close(value_after, value_before)
        else:
            assert value_after == value_before
    assert policy.weight.grad is None
    assert scheduler.step_calls == 0
    assert accelerator.scaler.update_calls == 1
    assert policy.update_calls == 0
    assert output_dict == {"optimizer_step/skipped": 1.0}


def test_update_policy_skips_before_controller_when_training_gradient_is_nonfinite():
    policy = _PolicyWithOptimizerStepControl()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    scheduler = _CountingScheduler()
    accelerator = _FakeAccelerator()
    train_metrics = SimpleNamespace()

    _, output_dict = update_policy(
        train_metrics=train_metrics,
        policy=policy,
        batch=torch.tensor(float("inf")),
        optimizer=optimizer,
        grad_clip_norm=0.0,
        accelerator=accelerator,
        lr_scheduler=scheduler,
    )

    assert policy.control_calls == 0
    assert policy.vlm_weight.item() == pytest.approx(0.0)
    assert policy.action_weight.item() == pytest.approx(0.0)
    assert optimizer.state == {}
    assert scheduler.step_calls == 0
    assert output_dict["optimizer_step/skipped"] == 1.0
    assert output_dict["optimizer_step/nonfinite_gradients"] == 1.0


def test_update_policy_skips_nonfinite_gradient_without_controller_hook():
    policy = _PolicyWithoutCustomGradientClipping()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    scheduler = _CountingScheduler()
    accelerator = _FakeAccelerator()
    train_metrics = SimpleNamespace()

    _, output_dict = update_policy(
        train_metrics=train_metrics,
        policy=policy,
        batch=torch.tensor(float("inf")),
        optimizer=optimizer,
        grad_clip_norm=0.0,
        accelerator=accelerator,
        lr_scheduler=scheduler,
    )

    assert policy.weight.item() == pytest.approx(0.0)
    assert policy.weight.grad is None
    assert optimizer.state == {}
    assert scheduler.step_calls == 0
    assert output_dict["optimizer_step/skipped"] == 1.0
    assert output_dict["optimizer_step/nonfinite_gradients"] == 1.0
