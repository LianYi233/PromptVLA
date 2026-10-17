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

import draccus
import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy  # noqa: E402


def _make_module() -> nn.Module:
    return nn.Linear(2, 2, bias=False)


def _make_policy_with_gradients(
    *,
    action_gradient: float,
    vlm_gradient: float,
    action_head_grad_clip_ratio: float = 10.0,
    clip_action_head_by_vlm: bool = True,
) -> tuple[SimpleNamespace, list[nn.Parameter], list[nn.Parameter]]:
    paligemma_with_expert = SimpleNamespace(
        paligemma=_make_module(),
        gemma_expert=_make_module(),
    )
    action_modules = [
        paligemma_with_expert.gemma_expert,
        _make_module(),
        _make_module(),
        _make_module(),
        _make_module(),
        nn.Embedding(1, 2),
    ]
    model = SimpleNamespace(
        paligemma_with_expert=paligemma_with_expert,
        action_in_proj=action_modules[1],
        action_out_proj=action_modules[2],
        time_mlp_in=action_modules[3],
        time_mlp_out=action_modules[4],
        inpainting_visible_action_embedding=action_modules[5],
        prompt_tokens=nn.Embedding(16, 2),
    )
    policy = SimpleNamespace(
        model=model,
        config=SimpleNamespace(
            clip_action_head_by_vlm=clip_action_head_by_vlm,
            action_head_grad_clip_ratio=action_head_grad_clip_ratio,
            cabo_enabled=False,
            training_stage="flow",
        ),
    )

    action_parameters = [parameter for module in action_modules for parameter in module.parameters()]
    vlm_parameters = list(paligemma_with_expert.paligemma.parameters())
    for parameter in action_parameters:
        parameter.grad = torch.full_like(parameter, action_gradient)
    model.prompt_tokens.weight.grad = torch.full_like(model.prompt_tokens.weight, action_gradient)
    for parameter in vlm_parameters:
        parameter.grad = torch.full_like(parameter, vlm_gradient)

    return policy, action_parameters, vlm_parameters


def _gradient_rms(parameters: list[nn.Parameter]) -> float:
    gradients = torch.cat([parameter.grad.detach().float().flatten() for parameter in parameters])
    return gradients.square().mean().sqrt().item()


def test_pi05_optimizer_and_action_control_defaults():
    config = PI05Config()
    optimizer_config = config.get_optimizer_preset()
    scheduler = config.get_scheduler_preset()
    parameters = [nn.Parameter(torch.zeros(())), nn.Parameter(torch.ones(()))]
    optimizer = optimizer_config.build(parameters)

    assert config.optimizer_grad_clip_norm == 0.0
    assert optimizer_config.grad_clip_norm == 0.0
    assert optimizer_config.lr == pytest.approx(2.5e-5)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert [id(parameter) for parameter in optimizer.param_groups[0]["params"]] == [
        id(parameter) for parameter in parameters
    ]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-5)
    assert optimizer.param_groups[0]["betas"] == pytest.approx((0.9, 0.95))
    assert optimizer.param_groups[0]["eps"] == pytest.approx(1e-8)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    assert scheduler.peak_lr == pytest.approx(2.5e-5)
    assert scheduler.decay_lr == pytest.approx(1e-5)
    assert scheduler.num_warmup_steps == 1_000
    assert scheduler.num_decay_steps == 30_000
    assert not config.clip_action_head_by_vlm
    assert config.action_head_grad_clip_ratio == pytest.approx(10.0)
    assert config.cabo_enabled
    assert config.cabo_active
    assert config.cabo_prompt_update_ratio == pytest.approx(2.0)
    assert config.training_stage == "flow"
    assert config.train_expert_only
    assert config.freeze_vision_encoder
    assert config.num_vlm_prompt_tokens == 16
    assert config.num_prompt_tokens == 16
    assert config.prompt_init_std == pytest.approx(0.02)
    assert config.next_action_masked_steps == 40
    assert config.next_action_full_mask_probability == pytest.approx(0.0)
    assert config.next_action_context_steps == 25
    assert config.next_action_prediction_steps == 25
    assert config.cabo_expert_update_ratio == pytest.approx(2.0)
    assert config.cabo_projection_update_ratio == pytest.approx(5.0)
    assert config.cabo_vlm_update_ema_decay == pytest.approx(0.95)
    assert config.cabo_update_warmup_steps == 100
    assert config.cabo_vlm_update_floor_ratio == pytest.approx(0.1)


def test_pi05_next_action_stage_keeps_cabo_enabled_for_bridge_without_gradient_clipping():
    config = PI05Config(
        training_stage="next_action",
        optimizer_grad_clip_norm=42.0,
        clip_action_head_by_vlm=True,
        cabo_enabled=True,
    )

    assert config.cabo_enabled
    assert config.cabo_active
    assert config.get_optimizer_preset().grad_clip_norm == 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"training_stage": "invalid"}, "training_stage"),
        (
            {"training_stage": "next_action", "next_action_masked_steps": 0},
            "next_action_masked_steps",
        ),
        (
            {
                "training_stage": "next_action",
                "next_action_masked_steps": 51,
                "next_action_full_mask_probability": 0.0,
            },
            "cannot exceed chunk_size",
        ),
        (
            {"next_action_full_mask_probability": 1.1},
            "next_action_full_mask_probability",
        ),
    ],
)
def test_pi05_rejects_invalid_next_action_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PI05Config(**kwargs)


def test_pi05_flow_stage_does_not_require_masked_steps_to_fit_chunk_size():
    config = PI05Config(chunk_size=20, n_action_steps=20)

    assert config.training_stage == "flow"
    assert config.next_action_masked_steps == 40


def test_pi05_full_mask_stage_does_not_require_legacy_mask_count_to_fit_chunk_size():
    config = PI05Config(
        training_stage="next_action",
        chunk_size=20,
        n_action_steps=20,
        next_action_full_mask_probability=1.0,
    )

    assert config.next_action_full_mask_probability == pytest.approx(1.0)
    assert config.next_action_masked_steps == 40


@pytest.mark.parametrize(
    "action_head_grad_clip_ratio",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_pi05_rejects_invalid_action_head_grad_clip_ratio(action_head_grad_clip_ratio: float):
    with pytest.raises(ValueError, match="action_head_grad_clip_ratio"):
        PI05Config(action_head_grad_clip_ratio=action_head_grad_clip_ratio)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cabo_expert_update_ratio", 0.0),
        ("cabo_expert_update_ratio", float("nan")),
        ("cabo_projection_update_ratio", -1.0),
        ("cabo_projection_update_ratio", float("inf")),
    ],
)
def test_pi05_rejects_invalid_cabo_update_ratio(field: str, value: float):
    with pytest.raises(ValueError, match=field):
        PI05Config(**{field: value})


@pytest.mark.parametrize("value", [0.0, 0.99, -1.0, float("nan"), float("inf"), float("-inf")])
def test_pi05_rejects_prompt_cabo_ratios_below_one_or_nonfinite(value: float):
    with pytest.raises(ValueError, match="cabo_prompt_update_ratio"):
        PI05Config(cabo_prompt_update_ratio=value)


@pytest.mark.parametrize("training_stage", ["next_action", "flow"])
@pytest.mark.parametrize("prompt_counts", [(0, 0), (0, 16), (16, 0)])
def test_pi05_cabo_requires_both_prompt_banks(training_stage, prompt_counts):
    with pytest.raises(ValueError, match="CABO requires num_vlm_prompt_tokens > 0 and num_prompt_tokens > 0"):
        PI05Config(
            training_stage=training_stage,
            cabo_enabled=True,
            num_vlm_prompt_tokens=prompt_counts[0],
            num_prompt_tokens=prompt_counts[1],
        )


@pytest.mark.parametrize("training_stage", ["next_action", "flow"])
@pytest.mark.parametrize("prompt_counts", [(0, 0), (0, 16), (16, 0)])
def test_pi05_disabling_cabo_allows_single_bank_and_prompt_free_configuration(training_stage, prompt_counts):
    config = PI05Config(
        training_stage=training_stage,
        cabo_enabled=False,
        num_vlm_prompt_tokens=prompt_counts[0],
        num_prompt_tokens=prompt_counts[1],
    )

    assert not config.cabo_enabled
    assert not config.cabo_active


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0, 10.0])
def test_pi05_accepts_cabo_prompt_update_ratio_at_least_one(ratio):
    config = PI05Config(cabo_prompt_update_ratio=ratio)

    assert config.cabo_active
    assert config.cabo_prompt_update_ratio == ratio


def test_pi05_stage1_prompt_cabo_update_ratio_decodes_from_nested_cli_argument():
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=user/repo",
            "--policy.type=pi05",
            "--policy.training_stage=next_action",
            "--policy.cabo_enabled=true",
            "--policy.cabo_prompt_update_ratio=1.5",
            "--policy.cabo_projection_update_ratio=3.5",
        ],
    )

    assert isinstance(config.policy, PI05Config)
    assert config.policy.cabo_enabled
    assert config.policy.cabo_active
    assert config.policy.cabo_prompt_update_ratio == pytest.approx(1.5)
    assert config.policy.cabo_projection_update_ratio == pytest.approx(3.5)
    assert config.save_freq == 10_000


def test_pi05_next_action_stage_decodes_from_nested_cli_argument():
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=user/repo",
            "--policy.type=pi05",
            "--policy.training_stage=next_action",
        ],
    )

    assert isinstance(config.policy, PI05Config)
    assert config.policy.training_stage == "next_action"
    assert config.policy.cabo_active


def test_pi05_both_prompt_counts_decode_from_nested_cli_arguments():
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=user/repo",
            "--policy.type=pi05",
            "--policy.num_vlm_prompt_tokens=24",
            "--policy.num_prompt_tokens=8",
        ],
    )

    assert isinstance(config.policy, PI05Config)
    assert config.policy.num_vlm_prompt_tokens == 24
    assert config.policy.num_prompt_tokens == 8


@pytest.mark.parametrize("cabo_vlm_update_ema_decay", [-0.1, 1.0, float("nan")])
def test_pi05_rejects_invalid_cabo_ema_decay(cabo_vlm_update_ema_decay: float):
    with pytest.raises(ValueError, match="cabo_vlm_update_ema_decay"):
        PI05Config(cabo_vlm_update_ema_decay=cabo_vlm_update_ema_decay)


def test_pi05_rejects_negative_cabo_warmup():
    with pytest.raises(ValueError, match="cabo_update_warmup_steps"):
        PI05Config(cabo_update_warmup_steps=-1)


@pytest.mark.parametrize("cabo_vlm_update_floor_ratio", [-0.1, 1.1])
def test_pi05_rejects_invalid_cabo_vlm_floor_ratio(cabo_vlm_update_floor_ratio: float):
    with pytest.raises(ValueError, match="cabo_vlm_update_floor_ratio"):
        PI05Config(cabo_vlm_update_floor_ratio=cabo_vlm_update_floor_ratio)


@pytest.mark.parametrize("training_stage", ["next_action", "flow"])
@pytest.mark.parametrize("legacy_freeze_flags", [False, True])
def test_pi05_prompt_cabo_stays_active_and_enforces_freeze_flags(training_stage, legacy_freeze_flags):
    config = PI05Config(
        training_stage=training_stage,
        cabo_enabled=True,
        train_expert_only=legacy_freeze_flags,
        freeze_vision_encoder=legacy_freeze_flags,
    )

    assert config.cabo_enabled
    assert config.cabo_active
    assert config.train_expert_only
    assert config.freeze_vision_encoder


def test_pi05_flow_rejects_vlm_relative_gradient_clipping_with_legacy_unfreeze_flags():
    with pytest.raises(ValueError, match="always freezes the VLM"):
        PI05Config(
            clip_action_head_by_vlm=True,
            train_expert_only=False,
            freeze_vision_encoder=False,
        )


@pytest.mark.parametrize("training_stage", ["next_action", "flow"])
def test_pi05_cabo_requires_policy_preset_for_named_prompt_groups(tmp_path, training_stage):
    policy_config = PI05Config(
        training_stage=training_stage,
        cabo_enabled=True,
        push_to_hub=False,
    )
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy_config,
        output_dir=tmp_path / "new-output",
        use_policy_training_preset=False,
        optimizer=policy_config.get_optimizer_preset(),
        scheduler=policy_config.get_scheduler_preset(),
    )

    with pytest.raises(ValueError, match="CABO requires use_policy_training_preset=True"):
        config.validate()
    assert config.cabo_active

    config.use_policy_training_preset = True
    config.validate()
    assert config.cabo_active


def test_pi05_cabo_disables_relative_gradient_clipping_hook():
    policy, action_parameters, vlm_parameters = _make_policy_with_gradients(
        action_gradient=20.0,
        vlm_gradient=1.0,
    )
    policy.config.cabo_enabled = True
    action_gradients_before = [parameter.grad.clone() for parameter in action_parameters]
    vlm_gradients_before = [parameter.grad.clone() for parameter in vlm_parameters]

    metrics = PI05Policy.clip_gradients(policy)

    assert metrics["action_head_clip_applied"] == 0.0
    assert metrics["cabo/gradient_clip_disabled"] == 1.0
    for parameter, gradient_before in zip(action_parameters, action_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)
    for parameter, gradient_before in zip(vlm_parameters, vlm_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)


def test_pi05_next_action_stage_disables_relative_gradient_clipping_hook():
    policy, action_parameters, vlm_parameters = _make_policy_with_gradients(
        action_gradient=20.0,
        vlm_gradient=1.0,
    )
    policy.config.training_stage = "next_action"
    action_gradients_before = [parameter.grad.clone() for parameter in action_parameters]
    vlm_gradients_before = [parameter.grad.clone() for parameter in vlm_parameters]

    metrics = PI05Policy.clip_gradients(policy)

    assert metrics["action_head_clip_applied"] == 0.0
    assert metrics["next_action/gradient_clip_disabled"] == 1.0
    for parameter, gradient_before in zip(action_parameters, action_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)
    for parameter, gradient_before in zip(vlm_parameters, vlm_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)


def test_pi05_stage2_leaves_action_gradients_unrestricted_by_default():
    policy, action_parameters, vlm_parameters = _make_policy_with_gradients(
        action_gradient=20.0,
        vlm_gradient=1.0,
        clip_action_head_by_vlm=False,
    )
    action_gradients_before = [parameter.grad.clone() for parameter in action_parameters]
    vlm_gradients_before = [parameter.grad.clone() for parameter in vlm_parameters]

    metrics = PI05Policy.clip_gradients(policy)

    assert metrics == {}
    for parameter, gradient_before in zip(action_parameters, action_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)
    for parameter, gradient_before in zip(vlm_parameters, vlm_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)


def test_pi05_leaves_action_gradient_below_ten_times_vlm_rms_unchanged():
    policy, action_parameters, vlm_parameters = _make_policy_with_gradients(
        action_gradient=6.0,
        vlm_gradient=1.0,
    )
    action_gradients_before = [parameter.grad.clone() for parameter in action_parameters]
    vlm_gradients_before = [parameter.grad.clone() for parameter in vlm_parameters]

    metrics = PI05Policy.clip_gradients(policy)

    assert metrics["action_head_clip_applied"] == 0.0
    assert metrics["action_head_grad_rms_before_clip"] == pytest.approx(6.0)
    assert metrics["vlm_grad_rms"] == pytest.approx(1.0)
    assert metrics["action_head_clip_threshold_rms"] == pytest.approx(10.0)
    assert metrics["action_head_grad_rms_after_clip"] == pytest.approx(6.0)
    assert metrics["action_head_clip_scale"] == pytest.approx(1.0)
    for parameter, gradient_before in zip(action_parameters, action_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)
    for parameter, gradient_before in zip(vlm_parameters, vlm_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)


def test_pi05_clips_action_gradient_to_ten_times_vlm_rms_without_modifying_vlm_or_prompt():
    policy, action_parameters, vlm_parameters = _make_policy_with_gradients(
        action_gradient=20.0,
        vlm_gradient=1.0,
    )
    vlm_gradients_before = [parameter.grad.clone() for parameter in vlm_parameters]
    prompt_gradient_before = policy.model.prompt_tokens.weight.grad.clone()

    metrics = PI05Policy.clip_gradients(policy)

    assert metrics["action_head_clip_applied"] == 1.0
    assert metrics["action_head_grad_rms_before_clip"] == pytest.approx(20.0)
    assert metrics["vlm_grad_rms"] == pytest.approx(1.0)
    assert metrics["action_head_clip_threshold_rms"] == pytest.approx(10.0)
    assert _gradient_rms(action_parameters) == pytest.approx(10.0, abs=1e-6)
    assert metrics["action_head_grad_rms_after_clip"] == pytest.approx(10.0, abs=1e-6)
    assert metrics["action_head_clip_scale"] == pytest.approx(0.5, abs=1e-6)
    for parameter, gradient_before in zip(vlm_parameters, vlm_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)
    assert torch.equal(policy.model.prompt_tokens.weight.grad, prompt_gradient_before)
