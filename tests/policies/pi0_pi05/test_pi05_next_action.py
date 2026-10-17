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
from safetensors.torch import save_file
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.optim.cabo import temporary_optimizer_group_lr_scales  # noqa: E402
from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    modeling_pi05,
)
from lerobot.policies.pi05.configuration_pi05 import PI05TrainingObjective  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

_WIDTH = 8
_ACTION_DIM = 3
_MAX_ACTION_DIM = 4
_HORIZON = 50


class _TinySelfAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = _TinySelfAttention(width)


class _TinyExpertBackbone(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(width)])


class _TinyExpert(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.model = _TinyExpertBackbone(width)


class _TinyPaliGemmaWithExpert(nn.Module):
    """Small stand-in that preserves action, expert, and AdaRMS gradient paths."""

    def __init__(self, _paligemma_config, action_expert_config, **_kwargs):
        super().__init__()
        width = action_expert_config.width
        self.paligemma = nn.Linear(width, width, bias=False)
        self.gemma_expert = _TinyExpert(width)
        self.last_call = None

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
    ):
        del past_key_values, use_cache
        if inputs_embeds[1] is None:
            raise AssertionError("The tiny test double only supports action-expert forwards")
        suffix_embs = inputs_embeds[1]
        transformed = self.gemma_expert.model.layers[0].self_attn.q_proj(suffix_embs)
        # The real expert uses the complete bidirectional block and AdaRMS conditioning. Preserve
        # both dependencies so the tests exercise every transferable Stage-1 module.
        suffix_out = transformed + transformed.mean(dim=1, keepdim=True) + adarms_cond[1].unsqueeze(1)
        self.last_call = {
            "attention_mask": attention_mask.detach().clone(),
            "position_ids": position_ids.detach().clone(),
            "suffix_embs": suffix_embs.detach().clone(),
            "adarms_cond": adarms_cond[1].detach().clone(),
        }
        return (None, suffix_out), None


@pytest.fixture(autouse=True)
def _tiny_pi05_modules(monkeypatch):
    monkeypatch.setattr(
        modeling_pi05,
        "get_gemma_config",
        lambda _variant: SimpleNamespace(width=_WIDTH),
    )
    monkeypatch.setattr(
        modeling_pi05,
        "PaliGemmaWithExpertModel",
        _TinyPaliGemmaWithExpert,
    )


def _make_config(stage: str = "next_action", **kwargs) -> PI05Config:
    config_kwargs = {
        "training_stage": stage,
        "dtype": "float32",
        "device": "cpu",
        "max_action_dim": _MAX_ACTION_DIM,
        "max_state_dim": _MAX_ACTION_DIM,
        "output_features": {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,)),
        },
    }
    config_kwargs.update(kwargs)
    return PI05Config(**config_kwargs)


def _make_policy(stage: str = "next_action", **kwargs) -> PI05Policy:
    return PI05Policy(_make_config(stage, **kwargs))


def _save_checkpoint(path, policy: PI05Policy, state_dict=None) -> None:
    path.mkdir()
    source = policy.state_dict() if state_dict is None else state_dict
    tensors = {key: value.detach().cpu().contiguous() for key, value in source.items()}
    save_file(tensors, path / "model.safetensors")


def test_action_only_flow_default_masks_40_valid_actions():
    core = _make_policy().model
    action_is_pad = torch.zeros(3, _HORIZON, dtype=torch.bool)
    action_is_pad[1, 7:] = True
    action_is_pad[2] = True

    inpainting_mask = core.sample_inpainting_mask(action_is_pad)

    assert inpainting_mask.sum(dim=1).tolist() == [40, 7, 0]
    assert ((~inpainting_mask) & (~action_is_pad)).sum(dim=1).tolist() == [10, 0, 0]
    assert not (inpainting_mask & action_is_pad).any()


def test_stage1_training_progress_switches_exactly_after_750_updates():
    policy = _make_policy(next_action_bridge_steps=250)

    policy.set_training_progress(step=749, total_steps=1_000)
    assert not policy._stage1_bridge_active

    policy.set_training_progress(step=750, total_steps=1_000)
    assert policy._stage1_bridge_active

    with pytest.raises(ValueError, match="cannot exceed the Stage-1 training steps"):
        policy.set_training_progress(step=0, total_steps=249)


def test_stage1_observation_flow_override_matches_formal_flow(monkeypatch):
    stage1_core = _make_policy("next_action").model
    flow_core = _make_policy("flow").model
    flow_core.load_state_dict(stage1_core.state_dict())
    actions = torch.randn(2, _HORIZON, _MAX_ACTION_DIM)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.2, 0.8])

    def zero_velocity(_images, _img_masks, _tokens, _masks, x_t, _time):
        return torch.zeros_like(x_t)

    monkeypatch.setattr(stage1_core, "predict_velocity", zero_velocity)
    monkeypatch.setattr(flow_core, "predict_velocity", zero_velocity)

    bridge_losses = stage1_core.forward(
        images=[],
        img_masks=[],
        tokens=torch.zeros(2, 1, dtype=torch.long),
        masks=torch.ones(2, 1, dtype=torch.bool),
        actions=actions,
        noise=noise,
        time=time,
        objective=PI05TrainingObjective.OBSERVATION_FLOW,
    )
    flow_losses = flow_core.forward(
        images=[],
        img_masks=[],
        tokens=torch.zeros(2, 1, dtype=torch.long),
        masks=torch.ones(2, 1, dtype=torch.bool),
        actions=actions,
        noise=noise,
        time=time,
    )

    torch.testing.assert_close(bridge_losses, flow_losses, rtol=0, atol=0)


def test_flow_inpainting_random_mask_selects_exact_valid_steps_only():
    core = _make_policy(
        next_action_masked_steps=4,
        next_action_full_mask_probability=0.0,
    ).model
    action_is_pad = torch.ones(3, _HORIZON, dtype=torch.bool)
    action_is_pad[0] = False
    action_is_pad[1, [1, 7, 13]] = False

    torch.manual_seed(17)
    mask_a = core.sample_inpainting_mask(action_is_pad)
    torch.manual_seed(23)
    mask_b = core.sample_inpainting_mask(action_is_pad)

    assert mask_a.dtype == torch.bool
    assert mask_a.shape == action_is_pad.shape
    assert mask_a.sum(dim=1).tolist() == [4, 3, 0]
    assert mask_b.sum(dim=1).tolist() == [4, 3, 0]
    assert not (mask_a & action_is_pad).any()
    assert not (mask_b & action_is_pad).any()
    assert torch.equal(mask_a[1], ~action_is_pad[1])
    assert not torch.equal(mask_a[0], mask_b[0])


def test_flow_inpainting_can_sample_exact_full_flow_corruption():
    core = _make_policy(
        next_action_masked_steps=4,
        next_action_full_mask_probability=1.0,
    ).model
    action_is_pad = torch.zeros(2, _HORIZON, dtype=torch.bool)
    action_is_pad[1, 9:] = True

    inpainting_mask = core.sample_inpainting_mask(action_is_pad)

    assert torch.equal(inpainting_mask, ~action_is_pad)


def test_action_only_flow_default_noises_only_masked_actions_and_sanitizes_padding(monkeypatch):
    core = _make_policy().model
    actions = torch.linspace(-1.0, 1.0, _HORIZON * _MAX_ACTION_DIM).reshape(1, _HORIZON, _MAX_ACTION_DIM)
    action_is_pad = torch.zeros(1, _HORIZON, dtype=torch.bool)
    action_is_pad[:, -2:] = True
    actions[:, -2:] = torch.nan
    noise = torch.full_like(actions, 0.75)
    time = torch.tensor([0.25])
    captured = {}

    def predict_velocity(x_t, predicted_time, predicted_action_is_pad, predicted_inpainting_mask):
        captured["x_t"] = x_t.detach().clone()
        captured["time"] = predicted_time.detach().clone()
        captured["action_is_pad"] = predicted_action_is_pad.detach().clone()
        captured["inpainting_mask"] = predicted_inpainting_mask.detach().clone()
        return torch.zeros_like(x_t)

    monkeypatch.setattr(core, "predict_inpainting_velocity", predict_velocity)

    losses = core.forward(
        actions=actions,
        noise=noise,
        time=time,
        action_is_pad=action_is_pad,
    )

    valid = ~action_is_pad
    inpainting_mask = captured["inpainting_mask"]
    safe_actions = torch.where(valid.unsqueeze(-1), actions, torch.zeros_like(actions))
    safe_noise = torch.where(valid.unsqueeze(-1), noise, torch.zeros_like(noise))
    noisy_actions = time[:, None, None] * safe_noise + (1 - time[:, None, None]) * safe_actions
    expected_x_t = torch.where(inpainting_mask.unsqueeze(-1), noisy_actions, safe_actions)
    expected_losses = torch.where(
        inpainting_mask.unsqueeze(-1),
        (safe_noise - safe_actions).square(),
        torch.zeros_like(actions),
    )
    assert inpainting_mask.sum().item() == 40
    assert not (inpainting_mask & action_is_pad).any()
    torch.testing.assert_close(captured["x_t"], expected_x_t)
    torch.testing.assert_close(losses, expected_losses)
    assert torch.count_nonzero(captured["x_t"][action_is_pad]) == 0


def test_flow_inpainting_explicit_mask_controls_noising_and_elementwise_loss(monkeypatch):
    core = _make_policy().model
    actions = torch.linspace(-1.0, 1.0, _HORIZON * _MAX_ACTION_DIM).reshape(1, _HORIZON, _MAX_ACTION_DIM)
    action_is_pad = torch.zeros(1, _HORIZON, dtype=torch.bool)
    action_is_pad[:, -2:] = True
    actions[:, -2:] = torch.nan
    inpainting_mask = torch.zeros_like(action_is_pad)
    inpainting_mask[:, [1, 17, 45]] = True
    noise = torch.full_like(actions, 0.75)
    time = torch.tensor([0.25])
    captured = {}

    def predict_velocity(x_t, predicted_time, predicted_action_is_pad, predicted_inpainting_mask):
        captured["x_t"] = x_t.detach().clone()
        captured["time"] = predicted_time.detach().clone()
        captured["action_is_pad"] = predicted_action_is_pad.detach().clone()
        captured["inpainting_mask"] = predicted_inpainting_mask.detach().clone()
        return torch.zeros_like(x_t)

    monkeypatch.setattr(core, "predict_inpainting_velocity", predict_velocity)

    losses = core.forward(
        actions=actions,
        noise=noise,
        time=time,
        action_is_pad=action_is_pad,
        inpainting_mask=inpainting_mask,
    )

    safe_actions = torch.where(action_is_pad.unsqueeze(-1), torch.zeros_like(actions), actions)
    noisy_actions = time[:, None, None] * noise + (1 - time[:, None, None]) * safe_actions
    expected_x_t = torch.where(inpainting_mask.unsqueeze(-1), noisy_actions, safe_actions)
    expected_losses = torch.where(
        inpainting_mask.unsqueeze(-1),
        (noise - safe_actions).square(),
        torch.zeros_like(actions),
    )
    torch.testing.assert_close(captured["x_t"], expected_x_t)
    torch.testing.assert_close(captured["time"], time)
    torch.testing.assert_close(captured["action_is_pad"], action_is_pad)
    torch.testing.assert_close(captured["inpainting_mask"], inpainting_mask)
    torch.testing.assert_close(losses, expected_losses)
    assert torch.isfinite(losses).all()
    assert torch.count_nonzero(losses[~inpainting_mask]) == 0
    # Visible valid actions are exact clean context; padding is sanitized before the expert.
    torch.testing.assert_close(
        captured["x_t"][~inpainting_mask & ~action_is_pad],
        actions[~inpainting_mask & ~action_is_pad],
    )
    assert torch.count_nonzero(captured["x_t"][action_is_pad]) == 0


def test_flow_inpainting_action_block_is_bidirectional_and_padding_is_removed():
    core = _make_policy().model
    x_t = torch.randn(2, _HORIZON, _MAX_ACTION_DIM)
    action_is_pad = torch.zeros(2, _HORIZON, dtype=torch.bool)
    action_is_pad[1, -3:] = True
    x_t[1, -3:] = 0

    inpainting_mask = torch.zeros_like(action_is_pad)
    inpainting_mask[:, ::2] = True
    velocity = core.predict_inpainting_velocity(x_t, torch.tensor([0.2, 0.8]), action_is_pad, inpainting_mask)

    call = core.paligemma_with_expert.last_call
    allowed = call["attention_mask"][:, 0].eq(0)
    num_prompts = core.config.num_prompt_tokens
    assert velocity.shape == x_t.shape
    assert allowed[:, :num_prompts, :num_prompts].all()
    torch.testing.assert_close(
        allowed[:, :num_prompts, num_prompts:], (~action_is_pad)[:, None, :].expand(-1, num_prompts, -1)
    )
    assert allowed[0, num_prompts:, :].all()
    assert allowed[1, num_prompts:-3, :-3].all()
    assert not allowed[1, -3:, :].any()
    assert not allowed[1, :, -3:].any()
    torch.testing.assert_close(
        call["position_ids"], torch.arange(num_prompts + _HORIZON).unsqueeze(0).expand(2, -1)
    )
    expected_embs = core.action_in_proj(x_t)
    visible = ~action_is_pad & ~inpainting_mask
    expected_embs = expected_embs + visible.unsqueeze(-1) * core.inpainting_visible_action_embedding.weight[0]
    torch.testing.assert_close(call["suffix_embs"][:, num_prompts:], expected_embs)
    torch.testing.assert_close(
        call["suffix_embs"][:, :num_prompts], core.prompt_tokens.weight.unsqueeze(0).expand(2, -1, -1)
    )


def test_action_only_full_mask_never_applies_visible_action_role():
    core = _make_policy().model
    x_t = torch.randn(1, _HORIZON, _MAX_ACTION_DIM)
    action_is_pad = torch.zeros(1, _HORIZON, dtype=torch.bool)
    action_is_pad[:, -3:] = True
    x_t[:, -3:] = 0
    inpainting_mask = ~action_is_pad
    with torch.no_grad():
        core.inpainting_visible_action_embedding.weight.fill_(123.0)

    core.predict_inpainting_velocity(x_t, torch.tensor([0.5]), action_is_pad, inpainting_mask)

    torch.testing.assert_close(
        core.paligemma_with_expert.last_call["suffix_embs"][:, core.config.num_prompt_tokens :],
        core.action_in_proj(x_t),
    )


def test_flow_inpainting_policy_samples_mask_and_reduces_only_masked_actual_dimensions(monkeypatch):
    policy = _make_policy()
    action_is_pad = torch.zeros(2, _HORIZON, dtype=torch.bool)
    action_is_pad[1, 2:] = True
    inpainting_mask = torch.zeros_like(action_is_pad)
    inpainting_mask[0, [0, 3]] = True
    inpainting_mask[1, 1] = True
    raw_losses = torch.full((2, _HORIZON, _MAX_ACTION_DIM), float("nan"), requires_grad=True)
    with torch.no_grad():
        raw_losses[0, inpainting_mask[0], :_ACTION_DIM] = 1.0
        raw_losses[1, inpainting_mask[1], :_ACTION_DIM] = 3.0
    captured = {}

    monkeypatch.setattr(
        policy.model,
        "sample_inpainting_mask",
        lambda received_is_pad: inpainting_mask.to(received_is_pad.device),
    )

    def fake_forward(*_args, **kwargs):
        captured["inpainting_mask"] = kwargs["inpainting_mask"].detach().clone()
        return raw_losses

    monkeypatch.setattr(policy.model, "forward", fake_forward)
    batch = {
        ACTION: torch.randn(2, _HORIZON, _ACTION_DIM),
        "action_is_pad": action_is_pad,
    }

    loss, metrics = policy.forward(batch)
    assert torch.equal(batch["pi05_inpainting_mask"], inpainting_mask)
    per_sample, _ = policy.forward(batch, reduction="none")
    loss.backward()

    torch.testing.assert_close(captured["inpainting_mask"], inpainting_mask)
    torch.testing.assert_close(loss, torch.tensor(5 / 3))
    torch.testing.assert_close(per_sample, torch.tensor([1.0, 3.0]))
    assert metrics["flow_inpainting/loss"] == pytest.approx(5 / 3)
    assert metrics["flow_inpainting/masked_valid_count"] == 3
    assert metrics["flow_inpainting/masked_valid_fraction"] == pytest.approx(3 / 52)
    assert metrics["loss_per_dim"] == pytest.approx([5 / 3] * _ACTION_DIM)
    assert raw_losses.grad is not None
    selected_actual = inpainting_mask.unsqueeze(-1).expand_as(raw_losses).clone()
    selected_actual[:, :, _ACTION_DIM:] = False
    assert torch.count_nonzero(raw_losses.grad[~selected_actual]) == 0


def test_flow_inpainting_all_padded_actions_return_graph_connected_zero():
    policy = _make_policy()
    actions = torch.full((2, _HORIZON, _ACTION_DIM), float("nan"))
    action_is_pad = torch.ones(2, _HORIZON, dtype=torch.bool)

    loss, metrics = policy.forward({ACTION: actions, "action_is_pad": action_is_pad})
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert metrics["flow_inpainting/masked_valid_count"] == 0
    assert metrics["flow_inpainting/masked_valid_fraction"] == 0.0
    assert policy.model.prompt_tokens.weight.grad is not None
    assert torch.count_nonzero(policy.model.prompt_tokens.weight.grad) == 0
    assert policy.model.vlm_prompt_tokens.weight.grad is None
    assert policy.model.action_out_proj.weight.grad is None
    assert policy.model.action_in_proj.weight.grad is None
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in policy.parameters()
    )


def test_flow_inpainting_validates_padding_and_explicit_masks():
    core = _make_policy().model
    actions = torch.randn(1, _HORIZON, _MAX_ACTION_DIM)
    action_is_pad = torch.zeros(1, _HORIZON, dtype=torch.bool)
    action_is_pad[0, -1] = True

    with pytest.raises(ValueError, match="action_is_pad is required"):
        core.forward(actions=actions)
    with pytest.raises(ValueError, match="action_is_pad must have shape"):
        core.forward(actions=actions, action_is_pad=torch.zeros(1, 49, dtype=torch.bool))
    with pytest.raises(ValueError, match="inpainting_mask must have shape"):
        core.forward(
            actions=actions,
            action_is_pad=action_is_pad,
            inpainting_mask=torch.zeros(1, 49, dtype=torch.bool),
        )

    invalid_mask = torch.zeros_like(action_is_pad)
    invalid_mask[0, -1] = True
    with pytest.raises(ValueError, match="cannot select padded"):
        core.forward(
            actions=actions,
            action_is_pad=action_is_pad,
            inpainting_mask=invalid_mask,
        )


def test_action_only_flow_updates_only_expert_prompt_with_all_backbones_frozen():
    torch.manual_seed(5)
    policy = _make_policy(cabo_enabled=False)
    core = policy.model
    actions = torch.randn(2, _HORIZON, _MAX_ACTION_DIM)
    action_is_pad = torch.zeros(2, _HORIZON, dtype=torch.bool)
    inpainting_mask = torch.zeros_like(action_is_pad)
    inpainting_mask[:, ::4] = True
    losses = core.forward(
        actions=actions,
        noise=torch.randn_like(actions),
        time=torch.tensor([0.3, 0.7]),
        action_is_pad=action_is_pad,
        inpainting_mask=inpainting_mask,
    )

    losses[inpainting_mask].mean().backward()

    trainable = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    assert trainable == {"model.vlm_prompt_tokens.weight", "model.prompt_tokens.weight"}
    gradient = core.prompt_tokens.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert core.vlm_prompt_tokens.weight.grad is None
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for name, parameter in policy.named_parameters()
        if name not in trainable
    )

    role_parameter = dict(policy.named_parameters())["model.inpainting_visible_action_embedding.weight"]
    assert not role_parameter.requires_grad
    assert role_parameter.grad is None

    vlm_parameters = [
        parameter
        for name, parameter in policy.named_parameters()
        if name.startswith("model.paligemma_with_expert.paligemma.")
    ]
    assert vlm_parameters
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in vlm_parameters)
    assert not hasattr(core, "next_action_query")
    assert not hasattr(core, "next_action_out_proj")
    assert not any("next_action_query" in key or "next_action_out_proj" in key for key in policy.state_dict())

    optimizer_parameters = policy.get_optim_params()
    assert all(isinstance(parameter, nn.Parameter) for parameter in optimizer_parameters)
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter) for parameter in policy.parameters() if parameter.requires_grad
    }


def test_flow_inpainting_forward_supports_gradient_checkpointing():
    policy = _make_policy()
    policy.model.gradient_checkpointing_enabled = True
    policy.train()

    loss, _ = policy.forward(
        {
            ACTION: torch.randn(2, _HORIZON, _ACTION_DIM),
            "action_is_pad": torch.zeros(2, _HORIZON, dtype=torch.bool),
        }
    )
    loss.backward()

    gradient = policy.model.prompt_tokens.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert policy.model.vlm_prompt_tokens.weight.grad is None
    assert all(parameter.grad is None for parameter in policy.parameters() if not parameter.requires_grad)


@pytest.mark.skipif(
    not hasattr(torch, "compile") or not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile is unavailable on this platform",
)
def test_flow_inpainting_forward_supports_torch_compile(monkeypatch):
    core = _make_policy().model

    def simple_predict(x_t, _time, _action_is_pad, _inpainting_mask):
        return core.action_out_proj(core.action_in_proj(x_t))

    monkeypatch.setattr(core, "predict_inpainting_velocity", simple_predict)
    actions = torch.randn(1, _HORIZON, _MAX_ACTION_DIM)
    forward_kwargs = {
        "actions": actions,
        "noise": torch.randn_like(actions),
        "time": torch.tensor([0.4]),
        "action_is_pad": torch.zeros(1, _HORIZON, dtype=torch.bool),
        "inpainting_mask": torch.arange(_HORIZON).unsqueeze(0).remainder(2).eq(0),
    }
    explanation = torch._dynamo.explain(core.forward)(**forward_kwargs)
    compiled_forward = torch.compile(core.forward, backend="eager")

    losses = compiled_forward(**forward_kwargs)

    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    assert losses.shape == (1, _HORIZON, _MAX_ACTION_DIM)


@pytest.mark.skipif(
    not hasattr(torch, "compile") or not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile is unavailable on this platform",
)
def test_compiled_stage1_forward_supports_both_curriculum_objectives(monkeypatch):
    core = _make_policy().model

    def simple_inpainting_velocity(x_t, _time, _action_is_pad, _inpainting_mask):
        return core.action_out_proj(core.action_in_proj(x_t))

    def simple_observation_velocity(_images, _img_masks, _tokens, _masks, x_t, _time):
        return core.action_out_proj(core.action_in_proj(x_t))

    monkeypatch.setattr(core, "predict_inpainting_velocity", simple_inpainting_velocity)
    monkeypatch.setattr(core, "predict_velocity", simple_observation_velocity)
    compiled_forward = torch.compile(core.forward, backend="eager")
    actions = torch.randn(1, _HORIZON, _MAX_ACTION_DIM)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.4])
    action_is_pad = torch.zeros(1, _HORIZON, dtype=torch.bool)
    inpainting_mask = torch.arange(_HORIZON).unsqueeze(0).remainder(2).eq(0)

    inpainting_losses = compiled_forward(
        actions=actions,
        noise=noise,
        time=time,
        action_is_pad=action_is_pad,
        inpainting_mask=inpainting_mask,
        objective=PI05TrainingObjective.ACTION_INPAINTING,
    )
    flow_losses = compiled_forward(
        images=[],
        img_masks=[],
        tokens=torch.zeros(1, 1, dtype=torch.long),
        masks=torch.ones(1, 1, dtype=torch.bool),
        actions=actions,
        noise=noise,
        time=time,
        objective=PI05TrainingObjective.OBSERVATION_FLOW,
    )

    assert inpainting_losses.shape == flow_losses.shape == (1, _HORIZON, _MAX_ACTION_DIM)
    assert torch.isfinite(inpainting_losses).all()
    assert torch.isfinite(flow_losses).all()


def test_flow_masked_loss_uses_only_valid_timesteps_and_actual_action_dims(monkeypatch):
    policy = _make_policy("flow")
    raw_losses = torch.ones(2, _HORIZON, _MAX_ACTION_DIM, requires_grad=True)
    with torch.no_grad():
        raw_losses[1, :2, :_ACTION_DIM] = 2.0
        raw_losses[1, 2:, :_ACTION_DIM] = torch.nan
        raw_losses[:, :, _ACTION_DIM:] = torch.nan
    action_is_pad = torch.zeros(2, _HORIZON, dtype=torch.bool)
    action_is_pad[1, 2:] = True
    batch = {
        ACTION: torch.randn(2, _HORIZON, _ACTION_DIM),
        "action_is_pad": action_is_pad,
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: ([], []))
    monkeypatch.setattr(policy.model, "forward", lambda *_args, **_kwargs: raw_losses)

    loss, metrics = policy.forward(batch)
    per_sample, _ = policy.forward(batch, reduction="none")
    loss.backward()

    expected_loss = torch.tensor((_HORIZON * _ACTION_DIM + 2 * 2 * _ACTION_DIM) / (52 * _ACTION_DIM))
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(per_sample, torch.tensor([1.0, 2.0]))
    assert torch.isfinite(loss)
    assert metrics["valid_action_fraction"] == pytest.approx(52 / 100)
    assert metrics["all_padding_samples"] == 0.0
    assert metrics["loss_per_dim"] == pytest.approx([54 / 52] * _ACTION_DIM)
    assert raw_losses.grad is not None
    assert torch.count_nonzero(raw_losses.grad[1, 2:, :_ACTION_DIM]) == 0
    assert torch.count_nonzero(raw_losses.grad[:, :, _ACTION_DIM:]) == 0


def test_stage1_bridge_uses_full_flow_loss_and_overwrites_inpainting_denominator(monkeypatch):
    policy = _make_policy("next_action", next_action_bridge_steps=250)
    policy.set_training_progress(step=750, total_steps=1_000)
    raw_losses = torch.ones(2, _HORIZON, _MAX_ACTION_DIM, requires_grad=True)
    action_is_pad = torch.zeros(2, _HORIZON, dtype=torch.bool)
    action_is_pad[1, 7:] = True
    stale_inpainting_mask = torch.zeros_like(action_is_pad)
    stale_inpainting_mask[:, :2] = True
    images = [torch.randn(2, 3, 8, 8)]
    image_masks = [torch.ones(2, dtype=torch.bool)]
    batch = {
        ACTION: torch.randn(2, _HORIZON, _ACTION_DIM),
        "action_is_pad": action_is_pad,
        "pi05_inpainting_mask": stale_inpainting_mask,
        "pi05_loss_valid_steps": stale_inpainting_mask,
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    captured = {}
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: (images, image_masks))

    def fake_forward(*args, **kwargs):
        captured["args"] = args
        captured["objective"] = kwargs["objective"]
        return raw_losses

    monkeypatch.setattr(policy.model, "forward", fake_forward)

    pre_forward_normalizer = policy.get_distributed_loss_normalizer(batch)
    loss, metrics = policy.forward(batch)
    normalizer = policy.get_distributed_loss_normalizer(batch)
    loss.backward()

    assert captured["objective"] == PI05TrainingObjective.OBSERVATION_FLOW
    assert captured["args"][0] is images
    assert captured["args"][1] is image_masks
    assert captured["args"][2] is batch[OBS_LANGUAGE_TOKENS]
    assert captured["args"][3] is batch[OBS_LANGUAGE_ATTENTION_MASK]
    assert "pi05_inpainting_mask" not in batch
    assert torch.equal(batch["pi05_loss_valid_steps"], ~action_is_pad)
    assert pre_forward_normalizer.item() == (_HORIZON + 7) * _ACTION_DIM
    assert normalizer.item() == (_HORIZON + 7) * _ACTION_DIM
    assert metrics["stage1/bridge_active"] == 1.0
    assert metrics["stage1/observation_flow_loss"] == pytest.approx(1.0)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    assert raw_losses.grad is not None
    assert torch.count_nonzero(raw_losses.grad[1, 7:]) == 0


def test_stage1_bridge_backpropagates_to_both_prompt_banks(monkeypatch):
    policy = _make_policy("next_action", next_action_bridge_steps=250)
    policy.set_training_progress(step=750, total_steps=1_000)
    policy.train()
    batch_size = 2
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: ([], []))

    def action_path_velocity(_images, _img_masks, _tokens, _masks, x_t, time):
        embeddings = policy.model.action_in_proj(x_t)
        embeddings = embeddings + policy.model.prompt_tokens.weight.mean(dim=0)
        vlm_features = policy.model.paligemma_with_expert.paligemma(policy.model.vlm_prompt_tokens.weight)
        embeddings = embeddings + vlm_features.mean(dim=0)
        expert = policy.model.paligemma_with_expert.gemma_expert
        embeddings = expert.model.layers[0].self_attn.q_proj(embeddings)
        time_features = torch.ones(batch_size, _WIDTH) * time.unsqueeze(1)
        time_features = policy.model.time_mlp_out(
            torch.nn.functional.silu(policy.model.time_mlp_in(time_features))
        )
        return policy.model.action_out_proj(embeddings + time_features.unsqueeze(1))

    monkeypatch.setattr(policy.model, "predict_velocity", action_path_velocity)
    batch = {
        ACTION: torch.randn(batch_size, _HORIZON, _ACTION_DIM),
        "action_is_pad": torch.zeros(batch_size, _HORIZON, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.zeros(batch_size, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 1, dtype=torch.bool),
    }

    loss, _ = policy.forward(batch)
    loss.backward()

    vlm = policy.model.paligemma_with_expert.paligemma
    assert not vlm.training
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in vlm.parameters())
    for parameter_name in (
        "model.prompt_tokens.weight",
        "model.vlm_prompt_tokens.weight",
    ):
        gradient = dict(policy.named_parameters())[parameter_name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    assert all(parameter.grad is None for parameter in policy.parameters() if not parameter.requires_grad)


@pytest.mark.parametrize("cabo_enabled", [False, True])
def test_stage1_objective_switch_preserves_optimizer_and_scheduler_state(monkeypatch, cabo_enabled):
    policy = _make_policy("next_action", next_action_bridge_steps=1, cabo_enabled=cabo_enabled)
    policy.train()
    batch_size = 2
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: ([], []))

    def action_path_velocity(_images, _img_masks, _tokens, _masks, x_t, time):
        embeddings = policy.model.action_in_proj(x_t)
        embeddings = embeddings + policy.model.prompt_tokens.weight.mean(dim=0)
        vlm_features = policy.model.paligemma_with_expert.paligemma(policy.model.vlm_prompt_tokens.weight)
        embeddings = embeddings + vlm_features.mean(dim=0)
        expert = policy.model.paligemma_with_expert.gemma_expert
        embeddings = expert.model.layers[0].self_attn.q_proj(embeddings)
        time_features = torch.ones(batch_size, _WIDTH) * time.unsqueeze(1)
        time_features = policy.model.time_mlp_out(
            torch.nn.functional.silu(policy.model.time_mlp_in(time_features))
        )
        return policy.model.action_out_proj(embeddings + time_features.unsqueeze(1))

    monkeypatch.setattr(policy.model, "predict_velocity", action_path_velocity)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    policy.validate_optimizer_step_control(optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    optimizer_identity = id(optimizer)
    scheduler_identity = id(scheduler)
    initial_scheduler_step = scheduler.last_epoch
    optimizer_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    expected_parameter_ids = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    bridge_metrics = []
    initial_weights = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}

    for step in range(4):
        policy.set_training_progress(step=step, total_steps=4)
        batch = {
            ACTION: torch.randn(batch_size, _HORIZON, _ACTION_DIM),
            "action_is_pad": torch.zeros(batch_size, _HORIZON, dtype=torch.bool),
            OBS_LANGUAGE_TOKENS: torch.zeros(batch_size, 1, dtype=torch.long),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 1, dtype=torch.bool),
        }
        loss, metrics = policy.forward(batch)
        loss.backward()
        control = policy.compute_optimizer_step_control(batch, optimizer, SimpleNamespace(num_processes=1))
        assert not control.skip_optimizer_step
        if cabo_enabled and step == 3:
            assert control.metrics["cabo/active"] == 1.0
            assert control.metrics["cabo/scaled_action_prompt_relative_update_rate"] <= (
                control.metrics["cabo/vlm_prompt_relative_update_rate"]
                / policy.config.cabo_prompt_update_ratio
            )
        else:
            assert not control.group_scales
        with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        bridge_metrics.append(metrics["stage1/bridge_active"])
        if step < 3:
            torch.testing.assert_close(
                policy.model.vlm_prompt_tokens.weight,
                initial_weights["model.vlm_prompt_tokens.weight"],
                rtol=0,
                atol=0,
            )
            assert policy.model.vlm_prompt_tokens.weight not in optimizer.state

    assert bridge_metrics == [0.0, 0.0, 0.0, 1.0]
    assert id(optimizer) == optimizer_identity
    assert id(scheduler) == scheduler_identity
    assert scheduler.last_epoch == initial_scheduler_step + 4
    assert optimizer_parameter_ids == expected_parameter_ids
    assert {id(parameter) for parameter in policy.parameters() if parameter.requires_grad} == (
        expected_parameter_ids
    )
    assert optimizer.state[policy.model.prompt_tokens.weight]["step"].item() == 4
    assert optimizer.state[policy.model.vlm_prompt_tokens.weight]["step"].item() == 1
    assert not torch.equal(policy.model.prompt_tokens.weight, initial_weights["model.prompt_tokens.weight"])
    assert not torch.equal(
        policy.model.vlm_prompt_tokens.weight, initial_weights["model.vlm_prompt_tokens.weight"]
    )
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, initial_weights[name], rtol=0, atol=0)


def test_flow_all_padded_actions_return_graph_connected_zero(monkeypatch):
    policy = _make_policy("flow")
    raw_losses = torch.randn(2, _HORIZON, _MAX_ACTION_DIM, requires_grad=True)
    batch = {
        ACTION: torch.randn(2, _HORIZON, _ACTION_DIM),
        "action_is_pad": torch.ones(2, _HORIZON, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: ([], []))
    monkeypatch.setattr(policy.model, "forward", lambda *_args, **_kwargs: raw_losses)

    loss, metrics = policy.forward(batch)
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert metrics["valid_action_fraction"] == 0.0
    assert metrics["all_padding_samples"] == 2.0
    assert raw_losses.grad is not None
    assert torch.count_nonzero(raw_losses.grad) == 0


def test_flow_without_padding_mask_preserves_unmasked_mean(monkeypatch):
    policy = _make_policy("flow")
    raw_losses = torch.arange(2 * _HORIZON * _MAX_ACTION_DIM, dtype=torch.float32).reshape(
        2, _HORIZON, _MAX_ACTION_DIM
    )
    batch = {
        ACTION: torch.randn(2, _HORIZON, _ACTION_DIM),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: ([], []))
    monkeypatch.setattr(policy.model, "forward", lambda *_args, **_kwargs: raw_losses)

    loss, metrics = policy.forward(batch)

    torch.testing.assert_close(loss, raw_losses[:, :, :_ACTION_DIM].mean())
    assert metrics["valid_action_fraction"] == 1.0


def test_flow_padding_mask_shape_is_validated(monkeypatch):
    policy = _make_policy("flow")
    batch = {
        ACTION: torch.randn(2, _HORIZON, _ACTION_DIM),
        "action_is_pad": torch.zeros(_HORIZON, 2, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    monkeypatch.setattr(policy, "_preprocess_images", lambda _batch: ([], []))
    monkeypatch.setattr(
        policy.model,
        "forward",
        lambda *_args, **_kwargs: torch.zeros(2, _HORIZON, _MAX_ACTION_DIM, requires_grad=True),
    )

    with pytest.raises(ValueError, match="action_is_pad must have shape"):
        policy.forward(batch)


def test_pi05_flow_inpainting_config_keeps_padded_episode_tails():
    assert _make_config("flow").drop_n_last_frames == 0
    inpainting_config = PI05Config(
        training_stage="next_action",
        chunk_size=12,
        n_action_steps=12,
        next_action_masked_steps=5,
    )

    assert inpainting_config.drop_n_last_frames == 0
    assert inpainting_config.next_action_masked_steps == 5


def test_legacy_next_action_split_fields_remain_config_loadable_but_do_not_control_inpainting():
    config = PI05Config(
        training_stage="next_action",
        next_action_context_steps=7,
        next_action_prediction_steps=5,
    )

    assert config.next_action_masked_steps == 40
    assert config.next_action_full_mask_probability == pytest.approx(0.0)
    assert config.drop_n_last_frames == 0


def test_pi05_distributed_loss_normalizer_counts_masked_actual_action_elements():
    batch = {
        ACTION: torch.randn(2, _HORIZON, _ACTION_DIM),
        "action_is_pad": torch.zeros(2, _HORIZON, dtype=torch.bool),
    }
    batch["action_is_pad"][1, 7:] = True

    inpainting_normalizer = _make_policy("next_action").get_distributed_loss_normalizer(batch)
    flow_normalizer = _make_policy("flow").get_distributed_loss_normalizer(batch)

    assert inpainting_normalizer.item() == (40 + 7) * _ACTION_DIM
    assert flow_normalizer.item() == (_HORIZON + 7) * _ACTION_DIM


def test_pi05_distributed_loss_normalizer_uses_actual_full_mask_from_forward():
    policy = _make_policy(
        "next_action",
        next_action_full_mask_probability=1.0,
    )
    batch = {
        ACTION: torch.randn(2, _HORIZON, _ACTION_DIM),
        "action_is_pad": torch.zeros(2, _HORIZON, dtype=torch.bool),
    }
    batch["action_is_pad"][1, 7:] = True

    policy.forward(batch)
    normalizer = policy.get_distributed_loss_normalizer(batch)

    assert torch.equal(batch["pi05_inpainting_mask"], ~batch["action_is_pad"])
    assert normalizer.item() == (_HORIZON + 7) * _ACTION_DIM


@pytest.mark.parametrize(
    "method_name",
    ["select_action", "predict_action_chunk", "predict_action_chunk_with_context"],
)
def test_flow_inpainting_checkpoint_rejects_action_inference_before_preprocessing(method_name):
    policy = _make_policy()

    with pytest.raises(RuntimeError, match="frozen-VLM Stage-1 checkpoint"):
        getattr(policy, method_name)({})


@pytest.mark.parametrize("cabo_enabled", [False, True])
def test_stage1_checkpoint_strictly_preserves_every_weight_when_loaded_for_stage2(tmp_path, cabo_enabled):
    source = _make_policy("next_action", cabo_enabled=cabo_enabled)
    with torch.no_grad():
        source.model.vlm_prompt_tokens.weight.fill_(0.375)
        source.model.prompt_tokens.weight.fill_(0.125)
        source.model.action_in_proj.weight.fill_(0.25)
        source.model.action_in_proj.bias.fill_(-0.5)
        source.model.action_out_proj.weight.fill_(1.25)
        source.model.action_out_proj.bias.fill_(2.0)
        source.model.time_mlp_in.weight.fill_(2.25)
        source.model.time_mlp_out.weight.fill_(3.25)
        source.model.inpainting_visible_action_embedding.weight.fill_(3.75)
        source.model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.fill_(4.25)

    flow_template = _make_policy("flow", cabo_enabled=cabo_enabled)
    assert set(source.state_dict()) == set(flow_template.state_dict())
    assert not hasattr(source.model, "next_action_query")
    assert not hasattr(source.model, "next_action_out_proj")
    checkpoint = tmp_path / "stage1"
    _save_checkpoint(checkpoint, source)

    resumed = PI05Policy.from_pretrained(
        checkpoint,
        config=_make_config("next_action", cabo_enabled=cabo_enabled),
        local_files_only=True,
    )
    flow = PI05Policy.from_pretrained(
        checkpoint,
        config=_make_config("flow", cabo_enabled=cabo_enabled),
        local_files_only=True,
    )

    for key, source_value in source.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], source_value, rtol=0, atol=0)
        torch.testing.assert_close(flow.state_dict()[key], source_value, rtol=0, atol=0)
    torch.testing.assert_close(
        flow.model.action_out_proj.weight,
        source.model.action_out_proj.weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        flow.model.time_mlp_out.weight,
        source.model.time_mlp_out.weight,
        rtol=0,
        atol=0,
    )
    assert flow.config.cabo_active is cabo_enabled
    assert not flow.config.clip_action_head_by_vlm
    optimizer_parameters = flow.get_optim_params()
    if cabo_enabled:
        assert [group["name"] for group in optimizer_parameters] == ["vlm_prompt", "action_prompt"]
        optimizer_parameters = [parameter for group in optimizer_parameters for parameter in group["params"]]
    assert all(isinstance(parameter, nn.Parameter) for parameter in optimizer_parameters)
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter) for parameter in flow.parameters() if parameter.requires_grad
    }
    for name, parameter in flow.named_parameters():
        assert parameter.requires_grad == (
            name in {"model.vlm_prompt_tokens.weight", "model.prompt_tokens.weight"}
        )


def test_legacy_flow_checkpoint_zero_initializes_missing_inpainting_role(tmp_path):
    source = _make_policy("flow")
    state_dict = dict(source.state_dict())
    state_dict.pop("model.inpainting_visible_action_embedding.weight")
    checkpoint = tmp_path / "legacy_flow"
    _save_checkpoint(checkpoint, source, state_dict)

    loaded = PI05Policy.from_pretrained(
        checkpoint,
        config=_make_config("next_action"),
        local_files_only=True,
    )

    assert torch.count_nonzero(loaded.model.inpainting_visible_action_embedding.weight) == 0


def test_legacy_next_action_temporary_head_is_discarded_without_weakening_strict_load(tmp_path):
    source = _make_policy("flow")
    state_dict = dict(source.state_dict())
    state_dict.update(
        {
            "model.next_action_query": torch.zeros(_WIDTH),
            "model.next_action_out_proj.weight": torch.zeros(_MAX_ACTION_DIM, _WIDTH),
            "model.next_action_out_proj.bias": torch.zeros(_MAX_ACTION_DIM),
        }
    )
    checkpoint = tmp_path / "legacy_next_action"
    _save_checkpoint(checkpoint, source, state_dict)

    loaded = PI05Policy.from_pretrained(
        checkpoint,
        config=_make_config("flow"),
        local_files_only=True,
    )

    assert not hasattr(loaded.model, "next_action_query")
    torch.testing.assert_close(loaded.model.action_out_proj.weight, source.model.action_out_proj.weight)


@pytest.mark.parametrize("corruption", ["unknown", "missing_shared", "shape_mismatch"])
def test_checkpoint_loading_remains_strict_across_stages(tmp_path, corruption):
    source = _make_policy()
    state_dict = dict(source.state_dict())
    if corruption == "unknown":
        state_dict["model.unknown_parameter"] = torch.zeros(1)
    elif corruption == "missing_shared":
        state_dict.pop("model.action_out_proj.weight")
    else:
        state_dict["model.time_mlp_out.weight"] = torch.zeros(1, 1)
    checkpoint = tmp_path / corruption
    _save_checkpoint(checkpoint, source, state_dict)

    with pytest.raises(RuntimeError, match="Failed to load PI05 pretrained weights"):
        PI05Policy.from_pretrained(
            checkpoint,
            config=_make_config("flow"),
            local_files_only=True,
        )
