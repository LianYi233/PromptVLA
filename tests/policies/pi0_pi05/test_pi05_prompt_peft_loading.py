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

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies import factory  # noqa: E402
from lerobot.policies.pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


class _TinyCore(nn.Module):
    _freeze_backbones = PI05Pytorch._freeze_backbones

    def __init__(self, config):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.vlm_prompt_tokens = nn.Embedding(config.num_vlm_prompt_tokens, 2)
        self.prompt_tokens = nn.Embedding(config.num_prompt_tokens, 2)


class _TinyPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = _TinyCore(config)


class _FakePeftWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.base_policy = policy

    def get_base_model(self):
        return self.base_policy


@pytest.fixture
def mocked_adapter_loader(monkeypatch):
    """Exercise the real factory branch without PEFT, Hub downloads, or a full VLM."""
    state = SimpleNamespace(saved_modules=None, base_loads=0, adapter_loads=0)

    def load_base(**kwargs):
        state.base_loads += 1
        assert kwargs["pretrained_name_or_path"] == "test/base"
        return _TinyPolicy(kwargs["config"])

    def load_adapter(policy, adapter_path, *, config, is_trainable):
        state.adapter_loads += 1
        assert adapter_path == "test/adapter"
        assert is_trainable
        assert config.modules_to_save == state.saved_modules
        # PEFT can re-enable freshly loaded backbone/projection adapters after construction.
        policy.model.train()
        policy.model.requires_grad_(True)
        for parameter in policy.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        return _FakePeftWrapper(policy)

    peft_module = ModuleType("peft")
    peft_module.PeftConfig = SimpleNamespace(
        from_pretrained=lambda _path: SimpleNamespace(
            modules_to_save=state.saved_modules,
            base_model_name_or_path="test/base",
        )
    )
    peft_module.PeftModel = SimpleNamespace(from_pretrained=load_adapter)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setattr(factory, "get_policy_class", lambda _name: SimpleNamespace(from_pretrained=load_base))
    monkeypatch.setattr(
        factory,
        "env_to_policy_features",
        lambda _env: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    monkeypatch.setattr(factory, "validate_visual_features_consistency", lambda *_args: None)
    return state


def _load_policy(num_prompt_tokens=16, num_vlm_prompt_tokens=16, *, cabo_enabled=True):
    config = PI05Config(
        device="cpu",
        use_peft=True,
        pretrained_path="test/adapter",
        num_prompt_tokens=num_prompt_tokens,
        num_vlm_prompt_tokens=num_vlm_prompt_tokens,
        cabo_enabled=cabo_enabled,
    )
    return factory.make_policy(config, env_cfg=SimpleNamespace())


@pytest.mark.parametrize(
    "saved_modules",
    [None, [], ["action_in_proj"], ["other.prompt_tokens"], ["model.prompt_tokens.weight"]],
)
def test_prompt_training_rejects_adapters_without_saved_prompts_before_loading_base(
    mocked_adapter_loader, saved_modules
):
    mocked_adapter_loader.saved_modules = saved_modules

    with pytest.raises(ValueError, match="lacks saved learned prompt tokens"):
        _load_policy()

    assert mocked_adapter_loader.base_loads == 0
    assert mocked_adapter_loader.adapter_loads == 0


@pytest.mark.parametrize("prefix", ["", "model."])
def test_loading_prompt_adapter_immediately_freezes_all_non_prompt_parameters(mocked_adapter_loader, prefix):
    mocked_adapter_loader.saved_modules = [f"{prefix}vlm_prompt_tokens", f"{prefix}prompt_tokens"]

    wrapped = _load_policy()

    core = wrapped.get_base_model().model
    assert mocked_adapter_loader.base_loads == mocked_adapter_loader.adapter_loads == 1
    assert core.prompt_tokens.weight.requires_grad
    assert core.vlm_prompt_tokens.weight.requires_grad
    assert not core.paligemma_with_expert.paligemma.training
    assert not core.paligemma_with_expert.gemma_expert.training
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for name, parameter in core.named_parameters()
        if name not in {"prompt_tokens.weight", "vlm_prompt_tokens.weight"}
    )


@pytest.mark.parametrize(
    ("saved_module", "missing_module"),
    [("model.prompt_tokens", "vlm_prompt_tokens"), ("model.vlm_prompt_tokens", "prompt_tokens")],
)
def test_adapter_must_save_each_enabled_prompt_bank(mocked_adapter_loader, saved_module, missing_module):
    mocked_adapter_loader.saved_modules = [saved_module]

    with pytest.raises(ValueError, match=rf"for model\.{missing_module}\."):
        _load_policy()

    assert mocked_adapter_loader.base_loads == mocked_adapter_loader.adapter_loads == 0


@pytest.mark.parametrize(
    ("num_prompt_tokens", "num_vlm_prompt_tokens", "saved_modules"),
    [(0, 0, None), (16, 0, ["model.prompt_tokens"]), (0, 16, ["model.vlm_prompt_tokens"])],
)
def test_loading_legacy_adapter_allows_explicit_zero_prompt_ablation(
    mocked_adapter_loader, num_prompt_tokens, num_vlm_prompt_tokens, saved_modules
):
    mocked_adapter_loader.saved_modules = saved_modules
    wrapped = _load_policy(
        num_prompt_tokens=num_prompt_tokens,
        num_vlm_prompt_tokens=num_vlm_prompt_tokens,
        cabo_enabled=False,
    )

    core = wrapped.get_base_model().model
    assert mocked_adapter_loader.base_loads == mocked_adapter_loader.adapter_loads == 1
    assert core.prompt_tokens.num_embeddings == num_prompt_tokens
    assert core.vlm_prompt_tokens.num_embeddings == num_vlm_prompt_tokens
    assert not core.paligemma_with_expert.paligemma.training
    assert not core.paligemma_with_expert.gemma_expert.training
    assert all(not parameter.requires_grad for parameter in core.paligemma_with_expert.parameters())
