"""Exercise NTK paths on the actual PI05 implementation with tiny transformer dimensions."""

import copy

import numpy as np
import pytest
import torch

from lerobot.scripts.analyze_pi05_ntk_stages import analyze_seed, load_policy, make_parser
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.ntk import enable_analysis_gradients, parameter_groups
from tests.policies.pi0_pi05.test_pi05_prompt import (
    PI05Policy,
    _config,
    _inputs,
    _tiny_real_pi05,  # noqa: F401 -- shared fixture retains all real PI05 computations
)


def test_analysis_restores_backbone_and_vision_gradient_paths_without_changing_predictions():
    policy = PI05Policy(_config(separate_frozen_observations=True)).eval()
    inputs = _inputs()
    actions = torch.randn(2, 4, 3)
    time = torch.tensor([0.25, 0.75])
    with torch.no_grad():
        reference = policy.model.predict_velocity(**inputs, x_t=actions, time=time)
    groups = parameter_groups(policy, "both")
    enable_analysis_gradients(policy, groups)
    output = policy.model.predict_velocity(**inputs, x_t=actions, time=time)
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)
    vision = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    vision_parameters = [parameter for parameter in vision.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(output.square().sum(), vision_parameters)
    assert sum(gradient.square().sum().item() for gradient in gradients) > 0
    ids = [id(parameter) for members in groups.values() for _, parameter in members]
    assert len(ids) == len(set(ids))


def test_real_pi05_paired_seeds_and_checkpoint_round_trip(tmp_path, monkeypatch):
    policy = PI05Policy(_config(training_stage="next_action", separate_frozen_observations=True))
    saved = tmp_path / "pretrained_model"
    policy.save_pretrained(saved)
    loaded, groups = load_policy({"name": "priming", "checkpoint": str(saved)}, "both", "cpu")
    assert loaded.model.prompt_tokens.weight.requires_grad
    torch.testing.assert_close(loaded.model.prompt_tokens.weight, policy.model.prompt_tokens.weight)
    inputs = _inputs()
    monkeypatch.setattr(
        loaded,
        "_preprocess_images",
        lambda _batch: (
            [inputs["images"][0][:1]],
            [inputs["img_masks"][0][:1]],
        ),
    )
    samples = [
        {
            "action": torch.randn(1, 4, 3),
            OBS_LANGUAGE_TOKENS: inputs["tokens"][:1],
            OBS_LANGUAGE_ATTENTION_MASK: inputs["masks"][:1],
            "action_is_pad": torch.tensor([[False, False, False, True]]),
        }
        for _ in range(2)
    ]
    args = make_parser().parse_args(
        ["--device=cpu", "--num-samples=2", "--output-probes=2", "--sketch-dim=128"]
    )
    before = {key: value.detach().clone() for key, value in loaded.state_dict().items()}
    first = analyze_seed(loaded, groups, copy.deepcopy(samples), 0, args)
    torch.manual_seed(1234)
    torch.randn(100)  # unrelated RNG activity must not change the experiment
    second = analyze_seed(loaded, groups, copy.deepcopy(samples), 0, args)
    for group in groups:
        assert first[group]["tangent_trace"] > 0
        np.testing.assert_allclose(first[group]["kernel"], second[group]["kernel"], rtol=0, atol=0)
        assert first[group]["effective_rank"] <= 2 + 1e-8
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    # A real prompt change may alter the frozen-backbone NTK without updating backbone weights.
    with torch.no_grad():
        loaded.model.prompt_tokens.weight.add_(0.1 * torch.randn_like(loaded.model.prompt_tokens.weight))
    changed = analyze_seed(loaded, groups, samples, 0, args)
    assert changed["backbone/action"]["tangent_trace"] != pytest.approx(
        first["backbone/action"]["tangent_trace"]
    )
