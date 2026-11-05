"""CPU protocol tests; no checkpoint, simulator, or model-performance claim.

The tiny model below binds the *actual* checked-in PI0.5 chunk and sampling
methods. Only expensive vision/language features and the velocity field are
replaced, so the native multi-step integrator is exercised without GPU weights.
"""

import ast
import copy
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.pi05.action_time import RelativeActionTimeEncoding
from lerobot.scripts import pi05_task7_full_sampling as sampling
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


@lru_cache(maxsize=1)
def _native_methods():
    """Load production methods without importing optional transformer packages."""
    source = Path(__file__).resolve().parents[2] / "src/lerobot/policies/pi05/modeling_pi05.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "make_pi05_att_2d_masks":
            definitions.append(node)
        if isinstance(node, ast.ClassDef) and node.name in ("PI05Pytorch", "PI05Policy"):
            name = "sample_actions" if node.name == "PI05Pytorch" else "predict_action_chunk"
            definitions.append(next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == name))
    assert len(definitions) == 3, "The native PI0.5 sampling API moved; update this integration fixture"
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "ACTION": "action",
        "OBS_LANGUAGE_TOKENS": OBS_LANGUAGE_TOKENS,
        "OBS_LANGUAGE_ATTENTION_MASK": OBS_LANGUAGE_ATTENTION_MASK,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


class _TinyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_time_encoder = RelativeActionTimeEncoding(32, encoding_dim=8, fps=10)
        with torch.no_grad():
            values = torch.arange(256, dtype=torch.float32).reshape(32, 8)
            self.action_time_encoder.weight.copy_((values + 1) / 2560)
        self.rtc_processor = None
        self.paligemma_with_expert = SimpleNamespace(forward=self._prefix_forward)
        self.denoise_calls = []
        self.prefix_calls = []
        self.raise_on_call = None
        self.sample_actions = _native_methods()["sample_actions"].__get__(self)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def sample_noise(self, *args, **kwargs):
        raise AssertionError("The experiment must supply explicit, paired initial noise")

    def predict_velocity(self, *args, **kwargs):
        raise AssertionError("Do not use the one-step recorded-action probe for full sampling")

    def embed_prefix(self, images, img_masks, tokens, masks):
        embeddings = tokens.float().unsqueeze(-1).expand(-1, -1, 4)
        embeddings = embeddings + images[0].float().mean()
        return embeddings, masks, torch.zeros_like(masks)

    def _prepare_attention_masks_4d(self, mask):
        return mask.unsqueeze(1)

    def _prefix_forward(self, **kwargs):
        self.prefix_calls.append({key: kwargs[key].detach().clone() for key in ("attention_mask", "position_ids")})
        embeddings = kwargs["inputs_embeds"][0]
        return [embeddings, None], {"conditioning": embeddings.mean()}

    def denoise_step(self, *, prefix_pad_masks, past_key_values, x_t, timestep):
        self.denoise_calls.append((x_t.detach().clone(), timestep.detach().clone()))
        if self.raise_on_call == len(self.denoise_calls):
            raise RuntimeError("injected sampling failure")
        residual = self.action_time_encoder(x_t) - x_t
        return 0.2 * x_t + 0.4 * residual + 0.01 * past_key_values["conditioning"]


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            action_time_encoding=True,
            action_time_basis="index",
            action_time_injection="input",
            chunk_size=50,
            max_action_dim=32,
            output_features={"action": SimpleNamespace(shape=(7,))},
            # Deliberately shorter than the diagnostic's ten-step prefix.
            n_action_steps=2,
            num_inference_steps=4,
            training_stage="flow",
            num_vlm_prompt_tokens=2,
            num_prompt_tokens=0,
            rtc_config=None,
            compile_model=False,
        )
        self.model = _TinyModel(self.config)
        self.predict_action_chunk = _native_methods()["predict_action_chunk"].__get__(self)
        self.seen_observations = []
        self.eval()

    def _preprocess_images(self, batch):
        assert "action" not in batch and "action_is_pad" not in batch
        assert all(key.startswith("observation.") or key == "task" for key in batch)
        self.seen_observations.append(copy.deepcopy(batch))
        return [batch["observation.images.camera"]], [torch.ones(1, dtype=torch.bool)]

    def _rtc_enabled(self):
        return self.model._rtc_enabled()

    def prepare_action(self, *args, **kwargs):
        raise AssertionError("The sampler must not read future target actions")

    def select_action(self, *args, **kwargs):
        raise AssertionError("The queue API would truncate the full action chunk")


def _observation():
    return {
        "observation.images.camera": torch.full((1, 3, 4, 4), 0.25),
        "observation.state": torch.arange(7, dtype=torch.float32).reshape(1, 7),
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 5, 7]], dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "task": ["pick up the object"],
    }


def test_noise_is_standard_gaussian_reproducible_and_does_not_consume_global_rng():
    before = torch.random.get_rng_state().clone()
    noise = sampling.stable_sampling_noise(3000, 126, (1, 50, 32))
    assert noise.device.type == "cpu" and noise.dtype == torch.float32
    assert noise.shape == (1, 50, 32)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(noise, sampling.stable_sampling_noise(3000, 126, noise.shape), rtol=0, atol=0)
    assert not torch.equal(noise, sampling.stable_sampling_noise(4000, 126, noise.shape))
    assert not torch.equal(noise, sampling.stable_sampling_noise(3000, 127, noise.shape))
    # A broad deterministic check catches accidental one-step interpolation/scaling.
    assert abs(noise.mean().item()) < 0.1
    assert 0.9 < noise.std().item() < 1.1


def test_observation_filter_removes_targets_and_metadata_without_changing_inputs():
    item = {
        **_observation(),
        "action": torch.ones(1, 50, 7),
        "action_is_pad": torch.zeros(1, 50, dtype=torch.bool),
        "pi05_inpainting_mask": torch.ones(1, 50, dtype=torch.bool),
        "next.action": torch.ones(1, 50, 7),
        "episode_index": 9,
        "frame_index": 126,
        "index": 1126,
        "timestamp": 12.6,
    }
    filtered = sampling.observation_only(item)
    assert set(filtered) == set(_observation())
    torch.testing.assert_close(filtered["observation.state"], item["observation.state"], rtol=0, atol=0)
    assert filtered["task"] == item["task"]
    assert "action" in item and "frame_index" in item


def test_all_interventions_use_full_native_schedule_and_identical_pure_noise():
    policy = _TinyPolicy()
    observation = _observation()
    noise = sampling.stable_sampling_noise(3000, 126, (1, 50, 32))
    original_noise = noise.clone()
    weights = {key: value.clone() for key, value in policy.state_dict().items()}
    predictions = {}
    audits = {}
    for mode in ("correct", "zero", "reversed"):
        prediction, audit = sampling._sample_counterfactual(policy, observation, noise, mode)
        predictions[mode] = prediction
        audits[mode] = audit
        assert prediction.shape == (1, 50, 7)
        assert not prediction.requires_grad
        assert audit["flow_times"] == pytest.approx([1.0, 0.75, 0.5, 0.25])
    assert len(policy.model.denoise_calls) == 12
    for offset in (0, 4, 8):
        torch.testing.assert_close(policy.model.denoise_calls[offset][0], original_noise, rtol=0, atol=0)
        observed_times = [call[1].item() for call in policy.model.denoise_calls[offset : offset + 4]]
        assert observed_times == pytest.approx([1.0, 0.75, 0.5, 0.25])
    assert len({audit["initial_noise_sha256"] for audit in audits.values()}) == 1
    assert not torch.equal(predictions["correct"], predictions["zero"])
    assert not torch.equal(predictions["correct"], predictions["reversed"])
    torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)
    for later in policy.model.prefix_calls[1:]:
        for key, tensor in policy.model.prefix_calls[0].items():
            torch.testing.assert_close(tensor, later[key], rtol=0, atol=0)
    for later in policy.seen_observations[1:]:
        for key, value in observation.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, later[key], rtol=0, atol=0)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_future_target_mutation_cannot_change_sampling_inputs_or_prediction():
    policy = _TinyPolicy()
    noise = sampling.stable_sampling_noise(3000, 126, (1, 50, 32))
    item = {**_observation(), "action": torch.zeros(1, 50, 7)}
    first, first_audit = sampling._sample_counterfactual(policy, sampling.observation_only(item), noise, "correct")
    item["action"] = torch.full((1, 50, 7), 1000.0)
    second, second_audit = sampling._sample_counterfactual(policy, sampling.observation_only(item), noise, "correct")
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first_audit["initial_noise_sha256"] == second_audit["initial_noise_sha256"]


def test_batches_preprocess_observations_without_labels_and_score_targets_separately(monkeypatch):
    from torch.utils.data._utils.collate import default_collate

    # The generic collator is enough for this fixture with no language columns.
    collate_module = ModuleType("lerobot.utils.collate")
    collate_module.lerobot_collate_fn = default_collate
    monkeypatch.setitem(sys.modules, collate_module.__name__, collate_module)
    raw = {
        "observation.images.camera": torch.full((3, 4, 4), 128, dtype=torch.uint8),
        "observation.state": torch.arange(7, dtype=torch.float32),
        "task": "pick up the object",
        "index": torch.tensor(0),
        "episode_index": torch.tensor(9),
        # Poison values from a dataset sample must not enter inference preprocessing.
        "action": torch.full((50, 7), 1000.0),
        "action_is_pad": torch.ones(50, dtype=torch.bool),
    }

    class Dataset:
        meta = SimpleNamespace(camera_keys=["observation.images.camera"], has_language_columns=False)

        def __getitem__(self, index):
            assert index == 0
            return copy.deepcopy(raw)

    calls = []

    def processor(batch):
        calls.append(copy.deepcopy(batch))
        result = copy.deepcopy(batch)
        result[OBS_LANGUAGE_TOKENS] = result["observation.state"][:, :3].long()
        result[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, 3, dtype=torch.bool)
        if "action" in result:
            result["action"] = result["action"] / 2
        return result

    first_inputs, first_target = sampling._make_batches(Dataset(), np.zeros((50, 7)), 0, processor)
    second_inputs, second_target = sampling._make_batches(Dataset(), np.full((50, 7), 4), 0, processor)
    assert len(calls) == 4
    for index in (0, 2):
        assert all(key.startswith("observation.") or key == "task" for key in calls[index])
        assert "action" not in calls[index] and "action_is_pad" not in calls[index]
    assert "action" in calls[1] and "action" in calls[3]
    np.testing.assert_array_equal(first_target, np.zeros((50, 7)))
    np.testing.assert_array_equal(second_target, np.full((50, 7), 2))
    assert sampling.observation_sha256(first_inputs) == sampling.observation_sha256(second_inputs)
    assert first_inputs["observation.images.camera"].dtype == torch.float32
    torch.testing.assert_close(first_inputs["observation.images.camera"], torch.full((1, 3, 4, 4), 128 / 255))
    noise = sampling.stable_sampling_noise(3000, 0, (1, 50, 32))
    policy = _TinyPolicy()
    first_prediction, _ = sampling._sample_counterfactual(policy, first_inputs, noise, "correct")
    second_prediction, _ = sampling._sample_counterfactual(policy, second_inputs, noise, "correct")
    torch.testing.assert_close(first_prediction, second_prediction, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["scaled_noise", "one_step"])
def test_sampling_audit_rejects_non_native_initial_state_or_incomplete_schedule(fault):
    policy = _TinyPolicy()
    original_sampler = policy.model.sample_actions

    def wrong_sampler(*args, **kwargs):
        if fault == "scaled_noise":
            kwargs["noise"] = kwargs["noise"] * 0.5
        else:
            kwargs["num_steps"] = 1
        return original_sampler(*args, **kwargs)

    policy.model.sample_actions = wrong_sampler
    before = policy.model.denoise_step
    noise = sampling.stable_sampling_noise(3000, 126, (1, 50, 32))
    with pytest.raises(RuntimeError, match="pure noise|denoising schedule"):
        sampling._sample_counterfactual(policy, _observation(), noise, "correct")
    assert policy.model.denoise_step == before


@pytest.mark.parametrize("mode", ["zero", "reversed"])
def test_sampler_and_time_intervention_restore_after_failure(mode):
    policy = _TinyPolicy()
    encoder = policy.model.action_time_encoder
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    old_step, old_forward, old_features = policy.model.denoise_step, encoder.forward, encoder.features
    policy.model.raise_on_call = 3
    noise = sampling.stable_sampling_noise(3000, 126, (1, 50, 32))
    with pytest.raises(RuntimeError, match="injected sampling failure"):
        sampling._sample_counterfactual(policy, _observation(), noise, mode)
    assert policy.model.denoise_step == old_step
    assert encoder.forward == old_forward
    assert encoder.features == old_features
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    policy.model.raise_on_call = None
    restored, _ = sampling._sample_counterfactual(policy, _observation(), noise, "correct")
    fresh, _ = sampling._sample_counterfactual(_TinyPolicy(), _observation(), noise, "correct")
    torch.testing.assert_close(restored, fresh, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_time_encoding", False),
        ("action_time_basis", "physical"),
        ("action_time_injection", "qk"),
        ("rtc_config", SimpleNamespace(enabled=True)),
    ],
)
def test_unsupported_inference_protocol_is_rejected(field, value):
    policy = _TinyPolicy()
    setattr(policy.config, field, value)
    noise = sampling.stable_sampling_noise(3000, 126, (1, 50, 32))
    with pytest.raises((ValueError, RuntimeError)):
        sampling._sample_counterfactual(policy, _observation(), noise, "correct")
    assert not policy.model.denoise_calls


def test_prefix_arm_gripper_metrics_use_equal_weight_demonstrations():
    conditions = [
        {"episode_index": 9, "global_frame_index": 100, "noise_seed": 3000},
        {"episode_index": 9, "global_frame_index": 100, "noise_seed": 4000},
        {"episode_index": 74, "global_frame_index": 200, "noise_seed": 3000},
    ]
    target = np.zeros((3, 50, 7))
    zero = target.copy()
    zero[0, :10, 6] = 2
    zero[1, :10, 6] = 4
    zero[2, :10, 6] = 1
    zero[:2, 10:, :6] = 3
    zero[2, 10:, :6] = 2
    predictions = {"correct": target.copy(), "zero": zero, "reversed": zero / 2}
    summary = sampling.summarize_predictions(predictions, target, conditions, executed_steps=10)
    assert summary["episode_ids"] == [9, 74]
    assert summary["number_of_conditions"] == 3
    assert summary["number_of_demonstrations"] == 2
    assert summary["regions"] == {"executed_prefix": [0, 10], "full_chunk": [0, 50]}
    prefix = summary["metrics"]["executed_prefix"]
    full = summary["metrics"]["full_chunk"]
    assert prefix["gripper"]["per_demo_mse"]["zero"] == pytest.approx([10, 1])
    assert prefix["gripper"]["mean_mse"]["zero"] == pytest.approx(5.5)
    assert prefix["arm"]["mean_mse"]["zero"] == 0
    assert prefix["all"]["mean_mse"]["zero"] == pytest.approx(5.5 / 7)
    assert full["gripper"]["mean_mse"]["zero"] == pytest.approx(1.1)
    assert full["arm"]["mean_mse"]["zero"] == pytest.approx(5.2)
    assert full["all"]["mean_mse"]["zero"] == pytest.approx((6 * 5.2 + 1.1) / 7)
    assert prefix["gripper"]["mean_paired_deltas"]["reversed_minus_correct"] == pytest.approx(5.5 / 4)
    np.testing.assert_allclose(np.asarray(summary["per_demo_position_mse"]["gripper"]["zero"])[0, :10], 10)


def test_summary_preserves_negative_effects_and_does_not_invent_single_demo_interval():
    conditions = [{"episode_index": 9, "global_frame_index": 100, "noise_seed": 3000}]
    target = np.zeros((1, 50, 7))
    predictions = {"correct": np.ones_like(target), "zero": target.copy(), "reversed": target.copy()}
    summary = sampling.summarize_predictions(predictions, target, conditions)
    for region in ("executed_prefix", "full_chunk"):
        metric = summary["metrics"][region]["all"]
        assert metric["mean_paired_deltas"] == {"zero_minus_correct": -1.0, "reversed_minus_correct": -1.0}
        assert metric["paired_bootstrap_95pct"] == {"zero_minus_correct": None, "reversed_minus_correct": None}
