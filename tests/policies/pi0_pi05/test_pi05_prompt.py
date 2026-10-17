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

"""Exercise learned prompts through the real PiGemma attention and cache paths on CPU."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.optim.cabo import temporary_optimizer_group_lr_scales  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

_VLM_WIDTH = 32
_EXPERT_WIDTH = 16
_HORIZON = 4
_ACTION_DIM = 3
_VLM_PROMPTS = 3
_PROMPTS = 2
_VLM_PROMPT_KEY = "model.vlm_prompt_tokens.weight"
_PROMPT_KEY = "model.prompt_tokens.weight"
_PROMPT_KEYS = {_VLM_PROMPT_KEY, _PROMPT_KEY}


@pytest.fixture(autouse=True)
def _restore_matmul_precision():
    precision = torch.get_float32_matmul_precision()
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32


@pytest.fixture(autouse=True)
def _tiny_real_pi05(monkeypatch):
    """Reduce allocation sizes, retaining all production model implementations."""
    original_paligemma = modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma
    original_expert = modeling_pi05.PiGemmaForCausalLM

    def tiny_paligemma(config):
        config._vocab_size = 32
        config.image_token_index = 31
        config.text_config.vocab_size = 32
        config.text_config._attn_implementation = "eager"
        config.vision_config.hidden_size = 16
        config.vision_config.intermediate_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4
        config.vision_config.patch_size = 8
        config.vision_config.projection_dim = 32
        return original_paligemma(config)

    def tiny_expert(config):
        config.vocab_size = 32
        config._attn_implementation = "eager"
        return original_expert(config)

    monkeypatch.setattr(
        modeling_pi05,
        "get_gemma_config",
        lambda variant: modeling_pi05.GemmaConfig(
            width=_VLM_WIDTH if variant == "gemma_2b" else _EXPERT_WIDTH,
            depth=2,
            mlp_dim=64,
            num_heads=8,
            num_kv_heads=1,
            head_dim=4,
        ),
    )
    monkeypatch.setattr(modeling_pi05, "PaliGemmaForConditionalGenerationWithPiGemma", tiny_paligemma)
    monkeypatch.setattr(modeling_pi05, "PiGemmaForCausalLM", tiny_expert)
    torch.manual_seed(37)


def _config(**kwargs):
    options = {
        "device": "cpu",
        "dtype": "float32",
        "chunk_size": _HORIZON,
        "n_action_steps": _HORIZON,
        "max_action_dim": _ACTION_DIM,
        "image_resolution": (16, 16),
        "num_vlm_prompt_tokens": _VLM_PROMPTS,
        "num_prompt_tokens": _PROMPTS,
        "next_action_masked_steps": 2,
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))},
    }
    options.update(kwargs)
    return PI05Config(**options)


def _inputs():
    return {
        "images": [torch.randn(2, 3, 16, 16)],
        "img_masks": [torch.tensor([True, False])],
        "tokens": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "masks": torch.tensor([[True, True, True], [True, True, False]]),
    }


def _actions():
    return torch.randn(2, _HORIZON, _ACTION_DIM)


def _save_checkpoint(path, policy, state_dict=None):
    path.mkdir()
    state_dict = policy.state_dict() if state_dict is None else state_dict
    # Real PaliGemma may tie its language embedding and output weights.
    save_file(
        {key: value.detach().cpu().clone() for key, value in state_dict.items()}, path / "model.safetensors"
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("compile_model", [False, True])
def test_model_precision_applies_before_compile(monkeypatch, dtype, compile_model):
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    compile_calls = []

    def capture_compile(function, *, mode):
        compile_calls.append(
            (
                function.__name__,
                mode,
                torch.get_float32_matmul_precision(),
                torch.backends.cudnn.allow_tf32,
            )
        )
        return function

    monkeypatch.setattr(torch, "compile", capture_compile)
    cfg = _config(dtype=dtype, compile_model=compile_model)
    policy = PI05Policy(cfg)

    expected_precision = "highest" if dtype == "float32" else "high" if compile_model else "medium"
    expected_tf32 = dtype != "float32"
    assert torch.get_float32_matmul_precision() == expected_precision
    assert torch.backends.cuda.matmul.allow_tf32 is expected_tf32
    assert torch.backends.cudnn.allow_tf32 is expected_tf32
    if compile_model:
        assert compile_calls == [
            ("sample_actions", cfg.compile_mode, expected_precision, expected_tf32),
            ("forward", cfg.compile_mode, expected_precision, expected_tf32),
        ]
    else:
        assert not compile_calls

    if dtype == "float32":
        assert all(parameter.dtype == torch.float32 for parameter in policy.parameters())
    else:
        backbone = policy.model.paligemma_with_expert
        assert (
            backbone.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        )
        assert backbone.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    assert policy.model.vlm_prompt_tokens.weight.dtype == torch.float32
    assert policy.model.prompt_tokens.weight.dtype == torch.float32


@pytest.mark.parametrize("field", ["num_prompt_tokens", "num_vlm_prompt_tokens"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_prompt_count_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_prompt_initialization_is_rejected(value):
    with pytest.raises(ValueError, match="prompt_init_std"):
        _config(prompt_init_std=value)


def test_projection_adaptation_rejects_prompt_only_cabo_and_peft():
    with pytest.raises(ValueError, match="train_action_projections"):
        _config(train_action_projections=True)
    with pytest.raises(ValueError, match="train_action_projections"):
        _config(train_action_projections=True, cabo_enabled=False, use_peft=True)
    policy = PI05Policy(_config(train_action_projections=True, cabo_enabled=False))
    with pytest.raises(ValueError, match="full checkpoints"):
        policy.wrap_with_peft()


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_single_vlm_token_updates_through_frozen_expert_without_action_prompt(
    checkpointing, dtype, monkeypatch
):
    policy = PI05Policy(
        _config(
            dtype=dtype,
            vlm_prompt_only=True,
            num_vlm_prompt_tokens=1,
            num_prompt_tokens=0,
            training_stage="flow",
            next_action_pretrain_steps=0,
            next_action_bridge_steps=0,
            cabo_enabled=False,
            gradient_checkpointing=checkpointing,
        )
    )
    policy.train()
    parameters = policy.get_optim_params()
    assert len(parameters) == 1 and parameters[0] is policy.model.vlm_prompt_tokens.weight
    assert {name for name, p in policy.named_parameters() if p.requires_grad} == {_VLM_PROMPT_KEY}
    assert policy.model.prompt_tokens.weight.numel() == 0
    inputs = _inputs()
    monkeypatch.setattr(policy, "_preprocess_images", lambda _: (inputs["images"], inputs["img_masks"]))
    batch = {
        ACTION: _actions(),
        "action_is_pad": torch.zeros(2, _HORIZON, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: inputs["tokens"],
        OBS_LANGUAGE_ATTENTION_MASK: inputs["masks"],
    }
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0)
    loss, _ = policy(batch)
    loss.backward()
    gradient = policy.model.vlm_prompt_tokens.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    for name, parameter in policy.named_parameters():
        if name != _VLM_PROMPT_KEY:
            assert parameter.grad is None, name
    optimizer.step()
    for name, value in policy.state_dict().items():
        if name == _VLM_PROMPT_KEY:
            assert not torch.equal(value, before[name])
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize(
    "conflict",
    [
        {"num_prompt_tokens": 1},
        {"num_vlm_prompt_tokens": 0},
        {"cabo_enabled": True},
        {"next_action_pretrain_steps": 1000},
        {"next_action_bridge_steps": 250},
        {"training_stage": "next_action"},
        {"train_action_projections": True},
        {"use_peft": True},
    ],
)
def test_vlm_only_contract_rejects_other_adaptation_paths(conflict):
    options = {
        "vlm_prompt_only": True,
        "num_vlm_prompt_tokens": 1,
        "num_prompt_tokens": 0,
        "training_stage": "flow",
        "next_action_pretrain_steps": 0,
        "next_action_bridge_steps": 0,
        "cabo_enabled": False,
    }
    options.update(conflict)
    with pytest.raises(ValueError, match="vlm_prompt_only"):
        _config(**options)


def test_zero_prompt_attention_matches_original_pi05_prefix_action_mask():
    pad_masks = torch.tensor([[True, True, False, True, True, True]])
    original = modeling_pi05.make_att_2d_masks(pad_masks, torch.tensor([[0, 0, 0, 1, 0, 0]]))
    actual = modeling_pi05.make_pi05_att_2d_masks(
        pad_masks,
        prefix_length=3,
        num_vlm_prompt_tokens=0,
        num_prompt_tokens=0,
    )
    torch.testing.assert_close(actual, original)


@pytest.mark.parametrize("phase", ["action_only", "bridge", "flow"])
def test_projection_adaptation_updates_only_prompts_and_action_maps(phase, monkeypatch, tmp_path):
    cfg = _config(
        training_stage="flow" if phase == "flow" else "next_action",
        next_action_bridge_steps=1,
        train_action_projections=True,
        cabo_enabled=False,
    )
    policy = PI05Policy(cfg)
    policy.requires_grad_(True)
    policy.eval().train()
    policy.set_training_progress(step=2 if phase == "bridge" else 0, total_steps=3)
    projection_keys = {
        f"model.{module}.{suffix}"
        for module in ("action_in_proj", "action_out_proj")
        for suffix in ("weight", "bias")
    }
    expected = _PROMPT_KEYS | projection_keys
    assert {name for name, p in policy.named_parameters() if p.requires_grad} == expected
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=0.01, weight_decay=0)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in policy.parameters() if p.requires_grad
    }
    inputs = _inputs()
    monkeypatch.setattr(policy, "_preprocess_images", lambda _: (inputs["images"], inputs["img_masks"]))
    batch = {
        ACTION: _actions(),
        "action_is_pad": torch.zeros(2, _HORIZON, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: inputs["tokens"],
        OBS_LANGUAGE_ATTENTION_MASK: inputs["masks"],
    }
    loss, _ = policy(batch)
    loss.backward()
    for name, parameter in policy.named_parameters():
        if name in projection_keys:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        elif name not in expected:
            assert parameter.grad is None, name
    optimizer.step()
    after = policy.state_dict()
    for name in projection_keys:
        assert not torch.equal(before[name], after[name]), name
    for name in before.keys() - expected:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0, msg=name)
    # Full checkpoints preserve the adapted maps exactly, without new tensor keys.
    assert set(before) == set(after)
    directory = tmp_path / "checkpoint"
    _save_checkpoint(directory, policy)
    reloaded = PI05Policy.from_pretrained(directory, config=cfg, strict=True)
    for name in projection_keys:
        torch.testing.assert_close(reloaded.state_dict()[name], after[name], rtol=0, atol=0)


@pytest.mark.parametrize("num_vlm_prompts", [0, _VLM_PROMPTS])
@pytest.mark.parametrize("num_prompts", [0, _PROMPTS])
def test_prompt_and_action_attention_blocks(num_vlm_prompts, num_prompts):
    core = PI05Policy(
        _config(num_prompt_tokens=num_prompts, num_vlm_prompt_tokens=num_vlm_prompts, cabo_enabled=False)
    ).model
    inputs = _inputs()
    prefix, prefix_pad, _ = core.embed_prefix(**inputs)
    actions = _actions()
    suffix, suffix_pad, _, _ = core.embed_suffix(actions, torch.tensor([0.3, 0.7]))

    assert suffix.shape == (2, num_prompts + _HORIZON, _EXPERT_WIDTH)
    torch.testing.assert_close(suffix[:, num_prompts:], core.action_in_proj(actions))
    torch.testing.assert_close(
        suffix[:, :num_prompts], core.prompt_tokens.weight.unsqueeze(0).expand(2, -1, -1)
    )
    allowed = modeling_pi05.make_pi05_att_2d_masks(
        torch.cat([prefix_pad, suffix_pad], dim=1),
        prefix_length=prefix.shape[1],
        num_vlm_prompt_tokens=num_vlm_prompts,
        num_prompt_tokens=num_prompts,
    )
    prefix_len = prefix.shape[1]
    observation_len = prefix_len - num_vlm_prompts
    assert prefix.shape == (2, 7 + num_vlm_prompts, _VLM_WIDTH)
    torch.testing.assert_close(
        prefix[:, observation_len:], core.vlm_prompt_tokens.weight.unsqueeze(0).expand(2, -1, -1)
    )
    assert prefix_pad[:, observation_len:].all()
    assert not allowed[:, :observation_len, observation_len:].any()
    assert allowed[:, observation_len:prefix_len, observation_len:prefix_len].all()
    torch.testing.assert_close(
        allowed[:, :observation_len, :observation_len],
        prefix_pad[:, :observation_len, None] & prefix_pad[:, None, :observation_len],
    )
    torch.testing.assert_close(
        allowed[:, observation_len:prefix_len, :observation_len],
        prefix_pad[:, None, :observation_len].expand(-1, num_vlm_prompts, -1),
    )
    action_start = prefix_len + num_prompts
    assert not allowed[:, :prefix_len, prefix_len:].any()
    assert not allowed[:, prefix_len:action_start, :prefix_len].any()
    assert allowed[:, prefix_len:action_start, prefix_len:].all()
    assert allowed[:, action_start:, prefix_len:].all()
    torch.testing.assert_close(
        allowed[:, action_start:, :prefix_len],
        prefix_pad[:, None, :].expand(-1, _HORIZON, -1),
    )


def test_inpainting_prompts_remain_valid_and_positions_include_them(monkeypatch):
    core = PI05Policy(_config(training_stage="next_action")).model
    original_forward = core.paligemma_with_expert.forward
    captured = {}

    def capture_forward(**kwargs):
        captured.update(kwargs)
        return original_forward(**kwargs)

    monkeypatch.setattr(core.paligemma_with_expert, "forward", capture_forward)
    action_is_pad = torch.tensor([[False, False, False, False], [False, False, True, True]])
    velocity = core.predict_inpainting_velocity(
        _actions(), torch.tensor([0.2, 0.8]), action_is_pad, ~action_is_pad
    )

    assert velocity.shape == (2, _HORIZON, _ACTION_DIM)
    assert torch.isfinite(velocity).all()
    allowed = captured["attention_mask"][:, 0].eq(0)
    assert allowed[:, :_PROMPTS, :_PROMPTS].all()
    torch.testing.assert_close(
        allowed[:, :_PROMPTS, _PROMPTS:], (~action_is_pad)[:, None, :].expand(-1, _PROMPTS, -1)
    )
    assert allowed[0, _PROMPTS:, :].all()
    assert allowed[1, _PROMPTS : _PROMPTS + 2, : _PROMPTS + 2].all()
    assert not allowed[1, -2:, :].any()
    assert not allowed[1, :, -2:].any()
    torch.testing.assert_close(
        captured["position_ids"], torch.arange(_PROMPTS + _HORIZON).unsqueeze(0).expand(2, -1)
    )
    torch.testing.assert_close(
        captured["inputs_embeds"][1][:, :_PROMPTS], core.prompt_tokens.weight.unsqueeze(0).expand(2, -1, -1)
    )


@torch.no_grad()
def test_action_changes_update_expert_prompts_but_not_vlm_prefix():
    core = PI05Policy(_config()).model.eval()
    prefix, prefix_pad, _ = core.embed_prefix(**_inputs())
    time = torch.tensor([0.3, 0.7])

    def forward_hidden_states(actions):
        suffix, suffix_pad, _, condition = core.embed_suffix(actions, time)
        pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        attention = modeling_pi05.make_pi05_att_2d_masks(
            pad,
            prefix_length=prefix.shape[1],
            num_vlm_prompt_tokens=_VLM_PROMPTS,
            num_prompt_tokens=_PROMPTS,
        )
        outputs, _ = core.paligemma_with_expert(
            attention_mask=core._prepare_attention_masks_4d(attention),
            position_ids=pad.cumsum(dim=1) - 1,
            inputs_embeds=[prefix, suffix],
            adarms_cond=[None, condition],
            use_cache=False,
        )
        return outputs

    first_prefix, first_suffix = forward_hidden_states(_actions())
    second_prefix, second_suffix = forward_hidden_states(_actions() * 10)

    torch.testing.assert_close(first_prefix[prefix_pad], second_prefix[prefix_pad], rtol=0, atol=0)
    assert not torch.allclose(first_suffix[:, :_PROMPTS], second_suffix[:, :_PROMPTS])
    assert not torch.allclose(first_suffix[:, _PROMPTS:], second_suffix[:, _PROMPTS:])


@torch.no_grad()
def test_expert_prompts_cannot_read_observations_directly():
    core = PI05Policy(_config()).model.eval()
    # Isolate direct visibility: later action representations can carry VLM information.
    core.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers = 1
    prefix, prefix_pad, _ = core.embed_prefix(**_inputs())
    suffix, suffix_pad, _, condition = core.embed_suffix(_actions(), torch.tensor([0.3, 0.7]))
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    allowed = modeling_pi05.make_pi05_att_2d_masks(
        pad,
        prefix_length=prefix.shape[1],
        num_vlm_prompt_tokens=_VLM_PROMPTS,
        num_prompt_tokens=_PROMPTS,
    )

    def forward_hidden_states(prefix_embs):
        outputs, _ = core.paligemma_with_expert(
            attention_mask=core._prepare_attention_masks_4d(allowed),
            position_ids=pad.cumsum(dim=1) - 1,
            inputs_embeds=[prefix_embs, suffix],
            adarms_cond=[None, condition],
            use_cache=False,
        )
        return outputs

    changed_prefix = prefix.clone()
    changed_prefix[:, :-_VLM_PROMPTS] = torch.randn_like(changed_prefix[:, :-_VLM_PROMPTS])
    first_prefix, first_suffix = forward_hidden_states(prefix)
    second_prefix, second_suffix = forward_hidden_states(changed_prefix)
    torch.testing.assert_close(first_suffix[:, :_PROMPTS], second_suffix[:, :_PROMPTS], rtol=0, atol=0)
    assert not torch.allclose(first_prefix[:, -_VLM_PROMPTS:], second_prefix[:, -_VLM_PROMPTS:])
    assert not torch.allclose(first_suffix[:, _PROMPTS:], second_suffix[:, _PROMPTS:])


@pytest.mark.parametrize("phase", ["action_only", "bridge", "flow"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("cabo_enabled", [False, True])
def test_only_active_prompts_update_across_all_three_phases(phase, checkpointing, cabo_enabled, monkeypatch):
    policy = PI05Policy(
        _config(
            training_stage="flow" if phase == "flow" else "next_action",
            next_action_bridge_steps=1,
            gradient_checkpointing=checkpointing,
            freeze_vision_encoder=False,
            train_expert_only=False,
            cabo_enabled=cabo_enabled,
        )
    )
    # Even an external unfreeze followed by train() must preserve the prompt-only contract.
    policy.requires_grad_(True)
    policy.eval()
    policy.train()
    policy.set_training_progress(step=2 if phase == "bridge" else 0, total_steps=3)
    core = policy.model
    for backbone in (core.paligemma_with_expert.paligemma, core.paligemma_with_expert.gemma_expert):
        assert all(not module.training for module in backbone.modules())
        assert all(not parameter.requires_grad for parameter in backbone.parameters())
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    optimizer_parameters = policy.get_optim_params()
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=0.01, weight_decay=0.0)
    policy.validate_optimizer_step_control(optimizer)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(core.prompt_tokens.weight), id(core.vlm_prompt_tokens.weight)}
    assert {name for name, parameter in policy.named_parameters() if parameter.requires_grad} == _PROMPT_KEYS
    inputs = _inputs()
    monkeypatch.setattr(policy, "_preprocess_images", lambda batch: (inputs["images"], inputs["img_masks"]))
    batch = {ACTION: _actions(), "action_is_pad": torch.zeros(2, _HORIZON, dtype=torch.bool)}
    if phase != "action_only":
        batch[OBS_LANGUAGE_TOKENS] = inputs["tokens"]
        batch[OBS_LANGUAGE_ATTENTION_MASK] = inputs["masks"]
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    if phase != "flow":
        assert metrics["stage1/bridge_active"] == float(phase == "bridge")
    loss.backward()

    active_keys = {_PROMPT_KEY} if phase == "action_only" else _PROMPT_KEYS
    for name, parameter in policy.named_parameters():
        if name not in active_keys:
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
    control = policy.compute_optimizer_step_control(batch, optimizer, SimpleNamespace(num_processes=1))
    assert not control.skip_optimizer_step
    if cabo_enabled and phase != "action_only":
        assert control.metrics["cabo/active"] == 1.0
        assert 0.0 < control.group_scales["action_prompt"] <= 1.0
        assert control.metrics["cabo/vlm_prompt_scale"] == 1.0
    else:
        assert not control.group_scales
        if cabo_enabled:
            assert control.metrics["cabo/action_only_bypass"] == 1.0
    with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
        optimizer.step()
    assert all(group["lr"] == 0.01 for group in optimizer.param_groups)
    for name, value in policy.state_dict().items():
        if name in active_keys:
            assert not torch.equal(value, before[name]), name
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)
    if cabo_enabled and phase != "action_only":
        relative_updates = {
            name: float((policy.state_dict()[name].double() - before[name].double()).norm())
            / float(before[name].double().norm())
            for name in _PROMPT_KEYS
        }
        assert relative_updates[_PROMPT_KEY] <= (
            relative_updates[_VLM_PROMPT_KEY] / policy.config.cabo_prompt_update_ratio * (1.0 + 1e-6)
        )
        assert relative_updates[_PROMPT_KEY] == pytest.approx(
            control.metrics["cabo/scaled_action_prompt_relative_update_rate"], rel=1e-6
        )


@pytest.mark.parametrize("cabo_enabled", [False, True])
def test_cabo_setting_selects_named_dual_prompt_optimizer_groups(cabo_enabled):
    policy = PI05Policy(_config(cabo_enabled=cabo_enabled))
    policy.train()
    core = policy.model
    parameters = policy.get_optim_params()
    assert len(parameters) == 2
    if cabo_enabled:
        assert [group["name"] for group in parameters] == ["vlm_prompt", "action_prompt"]
        assert {id(parameter) for parameter in parameters[0]["params"]} == {id(core.vlm_prompt_tokens.weight)}
        assert {id(parameter) for parameter in parameters[1]["params"]} == {id(core.prompt_tokens.weight)}
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert {id(parameter) for parameter in parameters} == {
        id(core.vlm_prompt_tokens.weight),
        id(core.prompt_tokens.weight),
    }
    policy.validate_optimizer_step_control(optimizer)
    control = policy.compute_optimizer_step_control({}, optimizer, SimpleNamespace(num_processes=1))
    assert not control.skip_optimizer_step
    assert control.group_scales == ({"action_prompt": 1.0} if cabo_enabled else {})


def test_training_requires_at_least_one_prompt_bank():
    policy = PI05Policy(_config(num_prompt_tokens=0, num_vlm_prompt_tokens=0, cabo_enabled=False))
    with pytest.raises(ValueError, match="prompt"):
        policy.get_optim_params()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("num_vlm_prompts", [0, _VLM_PROMPTS])
@pytest.mark.parametrize("num_prompts", [0, _PROMPTS])
@pytest.mark.parametrize("empty_observations", [False, True])
@torch.no_grad()
def test_cached_denoising_matches_full_forward(dtype, num_vlm_prompts, num_prompts, empty_observations):
    core = PI05Policy(
        _config(
            dtype=dtype,
            num_prompt_tokens=num_prompts,
            num_vlm_prompt_tokens=num_vlm_prompts,
            cabo_enabled=False,
        )
    ).model.eval()
    inputs = _inputs()
    if empty_observations:
        inputs["img_masks"][0].fill_(False)
        inputs["masks"].fill_(False)
    actions = _actions()
    time = torch.tensor([0.25, 0.75])
    full_velocity = core.predict_velocity(**inputs, x_t=actions, time=time)
    prefix, prefix_pad, _ = core.embed_prefix(**inputs)
    _, cache = core.paligemma_with_expert(
        attention_mask=core._prepare_attention_masks_4d(
            modeling_pi05.make_pi05_att_2d_masks(
                prefix_pad,
                prefix_length=prefix.shape[1],
                num_vlm_prompt_tokens=num_vlm_prompts,
                num_prompt_tokens=0,
            )
        ),
        position_ids=prefix_pad.cumsum(dim=1) - 1,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )
    prefix_length = cache.get_seq_length()
    cached_velocity = core.denoise_step(prefix_pad, cache, actions, time)
    second_velocity = core.denoise_step(prefix_pad, cache, actions, time)

    assert cached_velocity.shape == actions.shape
    tolerance = 0.02 if dtype == "bfloat16" else 1e-5
    torch.testing.assert_close(cached_velocity, full_velocity, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(second_velocity, cached_velocity, rtol=0, atol=0)
    assert cache.get_seq_length() == prefix_length


@pytest.mark.parametrize("prompt_name", ["prompt_tokens", "vlm_prompt_tokens"])
@torch.no_grad()
def test_same_noise_sampling_changes_with_each_prompt_bank(prompt_name):
    core = PI05Policy(_config()).model.eval()
    inputs = _inputs()
    noise = _actions()
    first_actions, first_context = core.sample_actions(
        **inputs, noise=noise.clone(), num_steps=2, return_prefix_features=True
    )
    prompts = getattr(core, prompt_name)
    prompts.weight.copy_(torch.randn_like(prompts.weight))
    second_actions, second_context = core.sample_actions(
        **inputs, noise=noise.clone(), num_steps=2, return_prefix_features=True
    )

    assert first_actions.shape == noise.shape
    assert torch.isfinite(second_actions).all()
    assert not torch.allclose(first_actions, second_actions)
    if prompt_name == "prompt_tokens":
        torch.testing.assert_close(first_context, second_context, rtol=0, atol=0)
    else:
        assert not torch.allclose(first_context, second_context)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("num_vlm_prompts,num_prompts", [(0, 0), (1, 0), (_VLM_PROMPTS, _PROMPTS)])
@torch.no_grad()
def test_cached_sampling_sdpa_bias_matches_query_dtype(dtype, num_vlm_prompts, num_prompts, monkeypatch):
    """Exercise both cache paths under the stricter CUDA SDPA bias contract on CPU."""
    core = PI05Policy(
        _config(
            dtype=dtype,
            num_vlm_prompt_tokens=num_vlm_prompts,
            num_prompt_tokens=num_prompts,
            cabo_enabled=False,
            attention_implementation="sdpa",
        )
    ).model.eval()
    inputs = _inputs()
    inputs["images"].append(torch.randn_like(inputs["images"][0]))
    inputs["img_masks"].append(torch.tensor([True, True]))
    noise = _actions()
    # Oracle uses the existing full-forward training path and identical Euler steps.
    expected = noise.clone()
    for time in (1.0, 0.5):
        expected = expected - 0.5 * core.predict_velocity(**inputs, x_t=expected, time=torch.full((2,), time))

    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    seen_queries = []

    def strict_sdpa(query, key, value, attn_mask=None, **kwargs):
        if attn_mask is not None and attn_mask.is_floating_point():
            # CPU SDPA accepts fp32 bias with bf16 Q; some CUDA kernels reject it.
            assert attn_mask.dtype == query.dtype, "SDPA attention bias must match query dtype"
            assert torch.isfinite(attn_mask).all(), "PI05 finite padding sentinel must stay finite"
            seen_queries.append(query.shape[-2])
        return original_sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", strict_sdpa)
    actual = core.sample_actions(**inputs, noise=noise, num_steps=2)
    assert torch.isfinite(actual).all()
    # Prefix and action expert both used SDPA; checking just one misses the other crash.
    prefix_length = 2 * 4 + inputs["tokens"].shape[1] + num_vlm_prompts
    assert prefix_length in seen_queries
    assert _HORIZON + num_prompts in seen_queries
    tolerance = 0.03 if dtype == "bfloat16" else 1e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("missing_keys", [{_VLM_PROMPT_KEY}, {_PROMPT_KEY}, _PROMPT_KEYS])
def test_legacy_checkpoint_initializes_only_missing_prompts(tmp_path, missing_keys):
    policy = PI05Policy(_config())
    legacy_state = {key: value for key, value in policy.state_dict().items() if key not in missing_keys}
    checkpoint = tmp_path / "legacy"
    _save_checkpoint(checkpoint, policy, legacy_state)

    loaded = PI05Policy.from_pretrained(checkpoint, config=_config(), strict=True)

    for prompts, shape in [
        (loaded.model.prompt_tokens, (_PROMPTS, _EXPERT_WIDTH)),
        (loaded.model.vlm_prompt_tokens, (_VLM_PROMPTS, _VLM_WIDTH)),
    ]:
        assert prompts.weight.shape == shape
        assert prompts.weight.requires_grad
        assert torch.isfinite(prompts.weight).all()
        assert prompts.weight.count_nonzero() > 0
    for key, value in legacy_state.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


def test_prompt_checkpoint_roundtrips_across_training_stages(tmp_path):
    policy = PI05Policy(_config(training_stage="next_action"))
    with torch.no_grad():
        policy.model.prompt_tokens.weight.copy_(torch.randn_like(policy.model.prompt_tokens.weight))
        policy.model.vlm_prompt_tokens.weight.copy_(torch.randn_like(policy.model.vlm_prompt_tokens.weight))
    checkpoint = tmp_path / "prompts"
    _save_checkpoint(checkpoint, policy)

    loaded = PI05Policy.from_pretrained(checkpoint, config=_config(training_stage="flow"), strict=True)

    for key, value in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert {name for name, parameter in loaded.named_parameters() if parameter.requires_grad} == _PROMPT_KEYS


@pytest.mark.parametrize(
    ("config_field", "parameter_key", "count"),
    [
        ("num_prompt_tokens", _PROMPT_KEY, _PROMPTS),
        ("num_vlm_prompt_tokens", _VLM_PROMPT_KEY, _VLM_PROMPTS),
    ],
)
def test_prompt_checkpoint_shape_mismatch_fails(tmp_path, config_field, parameter_key, count):
    policy = PI05Policy(_config())
    checkpoint = tmp_path / "wrong_prompt_count"
    _save_checkpoint(checkpoint, policy)

    with pytest.raises(RuntimeError, match=f"size mismatch for {parameter_key}"):
        PI05Policy.from_pretrained(checkpoint, config=_config(**{config_field: count + 1}), strict=True)


def test_legacy_checkpoint_still_rejects_missing_action_weights(tmp_path):
    policy = PI05Policy(_config())
    state_dict = {
        key: value
        for key, value in policy.state_dict().items()
        if key not in _PROMPT_KEYS | {"model.action_out_proj.weight"}
    }
    checkpoint = tmp_path / "incomplete"
    _save_checkpoint(checkpoint, policy, state_dict)

    with pytest.raises(RuntimeError, match="model.action_out_proj.weight"):
        PI05Policy.from_pretrained(checkpoint, config=_config(), strict=True)


def test_default_peft_saves_both_prompt_banks_without_backbone_adapters():
    policy = PI05Policy(_config())
    targets = policy._get_default_peft_targets()
    assert targets["target_modules"] == "dummy-target-modules"
    assert targets["modules_to_save"] == ["model.vlm_prompt_tokens", "model.prompt_tokens"]


@pytest.mark.parametrize(
    "existing", [None, ["model.action_out_proj"], ["model.prompt_tokens"], ["model.vlm_prompt_tokens"]]
)
def test_custom_peft_config_preserves_prompt_checkpoint(existing):
    policy = PI05Policy(_config(pretrained_path="test-base"))
    config = SimpleNamespace(modules_to_save=existing)

    policy._validate_peft_config(config)

    assert config.modules_to_save.count("model.prompt_tokens") == 1
    assert config.modules_to_save.count("model.vlm_prompt_tokens") == 1
    for module in existing or []:
        assert module in config.modules_to_save


@pytest.mark.parametrize("custom_targets", [False, True])
def test_real_peft_trains_and_saves_only_both_prompt_banks(tmp_path, custom_targets):
    peft = pytest.importorskip("peft")
    policy = PI05Policy(_config(pretrained_path=tmp_path / "base"))
    # Deliberately broad custom targets also insert adapters into the VLM.
    adapter_config = peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"]) if custom_targets else None
    wrapped = policy.wrap_with_peft(adapter_config)
    # External trainers sometimes unfreeze the entire wrapper before toggling training mode.
    wrapped.requires_grad_(True)
    wrapped.train()
    core = policy.model
    for backbone in (core.paligemma_with_expert.paligemma, core.paligemma_with_expert.gemma_expert):
        assert any("lora_" in name for name, _ in backbone.named_parameters()) == custom_targets
        assert all(not parameter.requires_grad for parameter in backbone.parameters())
        assert all(not module.training for module in backbone.modules())
    active_prompts = {
        "prompt_tokens": core.prompt_tokens.modules_to_save["default"].weight,
        "vlm_prompt_tokens": core.vlm_prompt_tokens.modules_to_save["default"].weight,
    }
    active_ids = {id(parameter) for parameter in active_prompts.values()}
    optimizer_groups = policy.get_optim_params()
    assert [group["name"] for group in optimizer_groups] == ["vlm_prompt", "action_prompt"]
    assert {id(parameter) for group in optimizer_groups for parameter in group["params"]} == active_ids
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    optimizer = torch.optim.AdamW(optimizer_groups, lr=0.01)
    policy.validate_optimizer_step_control(optimizer)

    actions = _actions()
    core(
        **_inputs(),
        actions=actions,
        noise=torch.randn_like(actions),
        time=torch.tensor([0.2, 0.8]),
    ).mean().backward()

    for parameter in policy.parameters():
        if id(parameter) not in active_ids:
            assert not parameter.requires_grad
            assert parameter.grad is None
            continue
        assert parameter.requires_grad
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
    control = policy.compute_optimizer_step_control({}, optimizer, SimpleNamespace(num_processes=1))
    assert not control.skip_optimizer_step
    with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
        optimizer.step()
    for name, parameter in policy.named_parameters():
        if id(parameter) in active_ids:
            assert not torch.equal(parameter, before[name])
        else:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0, msg=name)
    adapter_path = tmp_path / "adapter"
    wrapped.save_pretrained(adapter_path, save_embedding_layers=False)
    weights = load_file(adapter_path / "adapter_model.safetensors")
    for name, active_prompt in active_prompts.items():
        saved_prompts = [value for key, value in weights.items() if key.endswith(f".{name}.weight")]
        assert len(saved_prompts) == 1
        torch.testing.assert_close(saved_prompts[0], active_prompt, rtol=0, atol=0)

    fresh_policy = PI05Policy(_config(pretrained_path=tmp_path / "base"))
    reloaded = peft.PeftModel.from_pretrained(fresh_policy, adapter_path, is_trainable=True)
    fresh_policy.model._freeze_backbones()
    reloaded.train()
    reloaded_active_ids = set()
    for name, trained_prompt in active_prompts.items():
        prompt_module = getattr(fresh_policy.model, name)
        active_prompt = prompt_module.modules_to_save["default"].weight
        reloaded_active_ids.add(id(active_prompt))
        torch.testing.assert_close(active_prompt, trained_prompt, rtol=0, atol=0)
        assert not prompt_module.original_module.weight.requires_grad
    assert {
        id(parameter) for group in fresh_policy.get_optim_params() for parameter in group["params"]
    } == reloaded_active_ids
    assert {id(parameter) for parameter in reloaded.parameters() if parameter.requires_grad} == (
        reloaded_active_ids
    )
    for backbone in (
        fresh_policy.model.paligemma_with_expert.paligemma,
        fresh_policy.model.paligemma_with_expert.gemma_expert,
    ):
        assert all(not module.training for module in backbone.modules())
