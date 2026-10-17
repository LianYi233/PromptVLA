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

"""Check optimized training against the dense eager computation on real tiny models."""

from collections import Counter

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05 import PI05Policy, modeling_pi05  # noqa: E402
from tests.policies.pi0_pi05.test_pi05_prompt import (  # noqa: E402
    _PROMPTS,
    _VLM_PROMPTS,
    _actions,
    _config,
    _inputs,
    _restore_matmul_precision,  # noqa: F401 -- Register the shared autouse fixture.
    _tiny_real_pi05,  # noqa: F401 -- Register the shared autouse fixture.
)


def _flow_result(core, inputs, actions, noise, time):
    core.train()
    loss = core(**inputs, actions=actions, noise=noise, time=time)
    assert torch.isfinite(loss).all()
    if loss.requires_grad:
        loss.mean().backward()

    gradients = {}
    for name in ("vlm_prompt_tokens", "prompt_tokens"):
        parameter = getattr(core, name).weight
        if parameter.numel():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.norm() > 0
            gradients[name] = parameter.grad.detach().clone()
        else:
            assert parameter.grad is None
    assert all(parameter.grad is None for parameter in core.parameters() if not parameter.requires_grad)
    return loss.detach(), gradients


def _assert_flow_equivalent(reference, optimized, inputs, dtype):
    optimized.load_state_dict(reference.state_dict(), strict=True)
    actions = _actions()
    noise = torch.randn_like(actions)
    time = torch.tensor([0.3, 0.7])
    expected_loss, expected_gradients = _flow_result(reference, inputs, actions, noise, time)
    actual_loss, actual_gradients = _flow_result(optimized, inputs, actions, noise, time)

    # BF16 changes the rounding order of attention and split GEMMs. Compare every element,
    # including small gradients, rather than merely matching the aggregate gradient norm.
    tolerances = {"rtol": 3e-2, "atol": 2e-3} if dtype == "bfloat16" else {"rtol": 2e-5, "atol": 2e-6}
    torch.testing.assert_close(actual_loss, expected_loss, **tolerances)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in expected_gradients:
        torch.testing.assert_close(actual_gradients[name], expected_gradients[name], **tolerances)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize(
    "prompt_counts", [(_VLM_PROMPTS, _PROMPTS), (0, _PROMPTS), (_VLM_PROMPTS, 0), (0, 0)]
)
@pytest.mark.parametrize(("attention", "separate"), [("sdpa", False), ("eager", True), ("sdpa", True)])
def test_optimized_flow_preserves_loss_and_prompt_gradients(
    dtype, checkpointing, prompt_counts, attention, separate
):
    options = {
        "dtype": dtype,
        "gradient_checkpointing": checkpointing,
        "num_vlm_prompt_tokens": prompt_counts[0],
        "num_prompt_tokens": prompt_counts[1],
        "cabo_enabled": False,
    }
    reference = PI05Policy(
        _config(**options, attention_implementation="eager", separate_frozen_observations=False)
    ).model
    optimized = PI05Policy(
        _config(**options, attention_implementation=attention, separate_frozen_observations=separate)
    ).model
    # The shared fixture supplies an unpadded example and another with a missing camera and
    # padded language token. Both must preserve the same loss and trainable prompt derivatives.
    _assert_flow_equivalent(reference, optimized, _inputs(), dtype)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("num_vlm_prompts", [0, _VLM_PROMPTS])
def test_optimized_flow_handles_sample_with_no_valid_observations(dtype, num_vlm_prompts):
    options = {
        "dtype": dtype,
        "num_vlm_prompt_tokens": num_vlm_prompts,
        "cabo_enabled": False,
    }
    reference = PI05Policy(
        _config(**options, attention_implementation="eager", separate_frozen_observations=False)
    ).model
    optimized = PI05Policy(
        _config(**options, attention_implementation="sdpa", separate_frozen_observations=True)
    ).model
    inputs = _inputs()
    inputs["masks"][1] = False
    assert not inputs["img_masks"][0][1]
    _assert_flow_equivalent(reference, optimized, inputs, dtype)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_optimized_flow_supports_compiled_backward(checkpointing):
    options = {"gradient_checkpointing": checkpointing, "cabo_enabled": False}
    reference = PI05Policy(
        _config(**options, attention_implementation="eager", separate_frozen_observations=False)
    ).model
    optimized = PI05Policy(_config(**options)).model
    # Compile forward and backward graphs on CPU without requiring a GPU or a C++ toolchain.
    optimized.forward = torch.compile(optimized.forward, backend="aot_eager", fullgraph=True)
    _assert_flow_equivalent(reference, optimized, _inputs(), "float32")


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_training_checkpoints_recompute_each_layer_at_most_once(monkeypatch, checkpointing, separate):
    core = PI05Policy(
        _config(
            gradient_checkpointing=checkpointing,
            attention_implementation="sdpa",
            separate_frozen_observations=separate,
        )
    ).model
    calls = Counter()

    def instrument(function_name):
        original = getattr(modeling_pi05, function_name)

        def count_layer_calls(layer_idx, *args, **kwargs):
            calls[function_name, layer_idx] += 1
            return original(layer_idx, *args, **kwargs)

        monkeypatch.setattr(modeling_pi05, function_name, count_layer_calls)

    instrument("compute_layer_complete")
    instrument("compute_layer_frozen_observations")
    actions = _actions()
    _flow_result(core, _inputs(), actions, torch.randn_like(actions), torch.tensor([0.3, 0.7]))
    depth = core.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers
    expected_calls = 2 if checkpointing else 1
    expected = Counter({("compute_layer_complete", layer): expected_calls for layer in range(depth)})
    if separate:
        expected.update({("compute_layer_frozen_observations", layer): 1 for layer in range(depth)})
    assert calls == expected


@pytest.mark.parametrize("checkpointing", [False, True])
def test_observation_operators_skip_autograd_without_detaching_vlm_prompts(checkpointing):
    core = PI05Policy(
        _config(
            gradient_checkpointing=checkpointing,
            attention_implementation="sdpa",
            separate_frozen_observations=True,
        )
    ).model
    inputs = _inputs()
    # The tiny vision tower emits four patches; the language input contributes three tokens.
    observation_length = (core.config.image_resolution[0] // 8) ** 2 + inputs["tokens"].shape[1]
    records = []
    handles = []

    def record_operator(layer, operator):
        def hook(module, args, output):
            records.append((layer, operator, args[0].shape[1], torch.is_grad_enabled(), output.requires_grad))

        return hook

    vlm_layers = core.paligemma_with_expert.paligemma.model.language_model.layers
    for index, layer in enumerate(vlm_layers):
        for name, operator in (("q_proj", layer.self_attn.q_proj), ("mlp_down", layer.mlp.down_proj)):
            handles.append(operator.register_forward_hook(record_operator(index, name)))
    try:
        actions = _actions()
        _, gradients = _flow_result(
            core, inputs, actions, torch.randn_like(actions), torch.tensor([0.3, 0.7])
        )
    finally:
        for handle in handles:
            handle.remove()

    assert set(gradients) == {"vlm_prompt_tokens", "prompt_tokens"}
    for index in range(len(vlm_layers)):
        for name in ("q_proj", "mlp_down"):
            operator_records = [record[2:] for record in records if record[:2] == (index, name)]
            observation_records = [record for record in operator_records if record[0] == observation_length]
            prompt_records = [record for record in operator_records if record[0] == _VLM_PROMPTS]
            assert observation_records, (index, name, operator_records)
            assert len(observation_records) == 1
            assert all(
                not grad_enabled and not requires_grad
                for _, grad_enabled, requires_grad in observation_records
            )
            assert prompt_records, (index, name, operator_records)
            assert all(grad_enabled and requires_grad for _, grad_enabled, requires_grad in prompt_records)
            assert all(length != observation_length + _VLM_PROMPTS for length, _, _ in operator_records)
