"""Real tiny PI0.5 inference, plus independent attention and alignment checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import ImageFont
from transformers.models.gemma.modeling_gemma import eager_attention_forward

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.attention_visualization import (
    AttentionVideoConfig,
    PI05AttentionRecorder,
    load_attention_font,
    overlay_attention,
    render_camera_attention_frame,
    selected_attention,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from tests.policies.pi0_pi05.test_pi05_prompt import (
    PI05Policy,
    _config,
    _restore_matmul_precision,  # noqa: F401
    _tiny_real_pi05,  # noqa: F401
)


@pytest.fixture(autouse=True)
def _test_font(monkeypatch):
    # Model/video tests do not require a proprietary font in CI. The production
    # resolver is checked separately and never substitutes this test font.
    font = ImageFont.truetype("DejaVuSerif.ttf", 14)
    for module in (
        "lerobot.policies.pi05.attention_visualization",
        "lerobot.scripts.libero_attention",
        "lerobot.scripts.replot_libero_attention",
        "lerobot.scripts.export_libero_prompt_attention",
    ):
        monkeypatch.setattr(f"{module}.load_attention_font", lambda *_args, **_kwargs: font)


def test_font_resolver_rejects_missing_files_and_other_families():
    with pytest.raises(ValueError, match="Times New Roman"):
        load_attention_font("/missing/times.ttf")
    with pytest.raises(ValueError, match="Times New Roman"):
        load_attention_font(ImageFont.truetype("DejaVuSerif.ttf", 14).path)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_selected_rows_match_eager_with_full_key_normalization(kv_heads):
    torch.manual_seed(8)
    query = torch.randn(1, 4, 6, 8)
    key = torch.randn(1, kv_heads, 13, 8)
    value = torch.randn_like(key)
    mask = torch.zeros(1, 1, 6, 13)
    mask[..., 3] = torch.finfo(torch.float32).min
    module = SimpleNamespace(num_key_value_groups=4 // kv_heads, training=False)
    _, reference = eager_attention_forward(module, query, key, value, mask, 0.25)
    actual = selected_attention(query, key, mask, 0.25, 2, 5)
    torch.testing.assert_close(actual, reference[:, :, 2:5].mean(dim=(1, 2)))
    heads = selected_attention(query, key, mask, 0.25, 2, 5, average_heads=False)
    torch.testing.assert_close(heads, reference[:, :, 2:5].mean(dim=2))
    assert actual[0, 3] == 0
    assert actual[0, :4].sum() < 1  # camera probabilities are not renormalized
    bool_mask = mask == 0
    torch.testing.assert_close(actual, selected_attention(query, key, bool_mask, 0.25, 2, 5))
    mask[..., 2, :] = torch.finfo(torch.float32).min
    with pytest.raises(ValueError, match="fully masked"):
        selected_attention(query, key, mask, 0.25, 2, 5)


def test_overlay_preserves_patch_orientation_and_does_not_invent_zero_hotspots():
    rgb = np.full((16, 16, 3), 128, dtype=np.uint8)
    patches = np.array([[0, 1], [0, 0]], dtype=np.float32)
    overlay = overlay_attention(rgb, patches, alpha=1)
    assert overlay[0, 0, 2] > overlay[0, 0, 0]  # low values have a blue/purple background
    assert min(overlay[0, -1, :2]) > 240 and overlay[0, -1, 2] < 100  # peak stays top-right, bright yellow
    zero = overlay_attention(rgb, patches * 0)
    np.testing.assert_array_equal(zero[0, 0], zero[-1, -1])
    assert zero[0, 0, 2] > zero[0, 0, 0]
    uniform = overlay_attention(rgb, np.ones((2, 2)))
    np.testing.assert_array_equal(uniform[0, 0], uniform[-1, -1])


def test_two_camera_renderer_uses_shared_probability_scale():
    frame = render_camera_attention_frame(
        np.zeros((2, 16, 16, 3), dtype=np.uint8),
        np.stack([np.ones((2, 2)), np.full((2, 2), 0.01)]),
        AttentionVideoConfig(source="vlm_prompt", cameras="0,1", alpha=1),
        ImageFont.truetype("DejaVuSerif.ttf", 14),
        1,
        0,
        0,
    )
    assert frame[100, 400, 0] > 240 and frame[100, 400, 1] > 240
    assert frame[452, 400, 2] > frame[452, 400, 0]  # weaker camera stays blue, not yellow


def make_policy_and_batch(**overrides):
    policy = PI05Policy(
        _config(
            num_inference_steps=3,
            n_action_steps=2,
            input_features={
                "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
                "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            },
            **overrides,
        )
    ).eval()
    batch = {
        "observation.images.image": torch.rand(1, 3, 16, 16),
        "observation.images.image2": torch.rand(1, 3, 16, 16),
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 0]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, False]]),
    }
    return policy, batch


@pytest.mark.parametrize("source", ["action", "vlm_prompt"])
@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_real_pi05_capture_preserves_actions_rng_weights_and_queue_alignment(source, backend):
    policy, batch = make_policy_and_batch()
    composite = policy.model.paligemma_with_expert
    decoders = [composite.paligemma.model.language_model, composite.gemma_expert.model]
    for decoder in decoders:
        decoder.config._attn_implementation = backend
    original_configs = [layer.self_attn.config for decoder in decoders for layer in decoder.layers]
    weights = {key: value.clone() for key, value in policy.state_dict().items()}
    torch.manual_seed(100)
    expected = torch.stack([policy.select_action(batch) for _ in range(5)])
    expected_rng = torch.get_rng_state()
    policy.reset()
    torch.manual_seed(100)
    with PI05AttentionRecorder(
        policy, AttentionVideoConfig(source=source, camera=1, record_routing=True)
    ) as recorder:
        recorded = []
        for step in range(5):
            recorded.append(policy.select_action(batch))
            assert recorder.prediction_step == step - step % 2
            assert recorder.step - recorder.prediction_step == step % 2
        assert recorder.steps == [0, 2, 4]
        assert recorder.counts == ([3] * 3 if source == "action" else [1] * 3)
        assert np.stack(recorder.maps).shape == (3, 2, 2)
        assert all(0 < float(m.sum()) < 1 for m in recorder.maps)
        routing = recorder.routing_arrays()
        np.testing.assert_allclose(routing["routing_head_mass"].sum(axis=-1), 1, atol=2e-6)
        expected_image = (
            (batch["observation.images.image2"][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        )
        np.testing.assert_array_equal(recorder.image, expected_image)
        frame = recorder.render_frames(np.zeros((1, 16, 16, 3), dtype=np.uint8))
        assert frame.shape == (1, 320, 768, 3)
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(frame[0, 24:280, :256], 0)
        recorder.reset_episode()
        assert recorder.latest_map is None and recorder.image is None and recorder.maps == []
    torch.testing.assert_close(torch.stack(recorded), expected, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert not hasattr(policy, "_eval_attention_visualizer")
    assert "embed_prefix" not in policy.model.__dict__
    assert original_configs == [layer.self_attn.config for decoder in decoders for layer in decoder.layers]
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
@pytest.mark.parametrize("source,denoise", [("vlm_prompt", "mean"), ("action", "mean"), ("action", "last")])
def test_two_camera_vlm_only_matches_independent_single_camera_captures(backend, source, denoise):
    policy, batch = make_policy_and_batch(num_prompt_tokens=0, cabo_enabled=False)
    composite = policy.model.paligemma_with_expert
    decoder = composite.gemma_expert.model if source == "action" else composite.paligemma.model.language_model
    decoder.config._attn_implementation = backend
    single_maps = []
    for camera in (0, 1):
        policy.reset()
        torch.manual_seed(83)
        with PI05AttentionRecorder(
            policy, AttentionVideoConfig(source=source, camera=camera, denoise=denoise)
        ) as recorder:
            expected_action = policy.select_action(batch)
            single_maps.append(recorder.latest_map.copy())
    expected_rng = torch.get_rng_state()
    policy.reset()
    torch.manual_seed(83)
    config = AttentionVideoConfig(source=source, cameras="0,1", denoise=denoise, record_routing=True)
    with PI05AttentionRecorder(policy, config) as recorder:
        actual = policy.select_action(batch)
        torch.testing.assert_close(actual, expected_action, atol=0, rtol=0)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        np.testing.assert_allclose(recorder.camera_maps[0], np.stack(single_maps), atol=1e-7)
        routing = recorder.routing_arrays()
        groups = routing["routing_group_names"].tolist()
        masses = routing["routing_head_mass"][0]
        mean_mass = (masses[-1] if denoise == "last" else masses.mean(axis=0)).mean(axis=0)
        np.testing.assert_allclose(mean_mass[:2], recorder.camera_maps[0].sum(axis=(-1, -2)), atol=1e-7)
        np.testing.assert_array_equal(masses[..., groups.index("language_padding")], 0)
        np.testing.assert_array_equal(masses[..., groups.index("action_prompt")], 0)
        np.testing.assert_array_equal(routing["routing_token_ids"][0, 8:11], [1, 2, 0])
        expected_groups = [0] * 4 + [1] * 4 + [2, 2, 3] + [4] * 3
        if source == "action":
            expected_groups += [6, 6, 7, 7]
        np.testing.assert_array_equal(routing["routing_key_group_ids"][0], expected_groups)
        assert recorder.counts == ([policy.config.num_inference_steps] if source == "action" else [1])
        assert recorder.layer == len(decoder.layers) - 1
        if source == "action":
            assert recorder.metadata()["query_tokens"] == "executed action-token prefix"
            assert recorder.metadata()["action_query_count"] == policy.config.n_action_steps
        assert recorder.metadata()["camera_features"] == list(batch)[:2]
        for camera, key in enumerate(list(batch)[:2]):
            expected_image = (batch[key][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            np.testing.assert_array_equal(recorder.camera_images[0][camera], expected_image)
        held_images = recorder.source_images.copy()
        policy.select_action(
            {**batch, "observation.images.image": torch.zeros_like(batch["observation.images.image"])}
        )
        assert recorder.steps == [0]  # second action is queued; keep both source frames
        np.testing.assert_array_equal(recorder.source_images, held_images)
        assert recorder.render_frames(np.zeros((1, 16, 16, 3), dtype=np.uint8)).shape == (1, 784, 640, 3)
    assert "embed_prefix" not in policy.model.__dict__


def test_missing_camera_and_exception_restore_instrumentation():
    policy, batch = make_policy_and_batch()
    del batch["observation.images.image2"]
    with (
        pytest.raises(ValueError, match="missing/padded"),
        PI05AttentionRecorder(policy, AttentionVideoConfig(camera=1)),
    ):
        policy.select_action(batch)
    assert not hasattr(policy, "_eval_attention_visualizer")
    assert "select_action" not in policy.__dict__


def test_routing_accounts_for_unselected_padded_camera():
    policy, batch = make_policy_and_batch()
    del batch["observation.images.image2"]
    with PI05AttentionRecorder(policy, AttentionVideoConfig(camera=0, record_routing=True)) as recorder:
        policy.select_action(batch)
        arrays = recorder.routing_arrays()
        np.testing.assert_array_equal(arrays["routing_head_mass"][..., 1], 0)
        np.testing.assert_array_equal(arrays["routing_visible_key_counts"][..., 1], 0)
        np.testing.assert_allclose(arrays["routing_head_mass"].sum(axis=-1), 1, atol=2e-6)


def test_uniform_attention_group_mass_equals_visible_token_fraction():
    from lerobot.policies.pi05.attention_routing import build_key_groups, summarize_routing

    names, ids = build_key_groups([2, 3], [True, False, True], 2, 0, 4, 2)
    query, key = torch.zeros(1, 2, 4, 8), torch.zeros(1, 1, 14, 8)
    mask = torch.ones(1, 1, 4, 14, dtype=torch.bool)
    mask[..., 6] = False  # middle language token is padding
    heads = selected_attention(query, key, mask, 0.25, 0, 2, average_heads=False)
    mass, counts, token_mass = summarize_routing(heads, mask, 0, 2, ids, names)
    expected_counts = np.array([2, 3, 2, 0, 2, 0, 2, 2])
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(mass, np.tile(expected_counts / 13, (2, 1)), atol=1e-7)
    assert token_mass[6] == 0
    with pytest.raises(ValueError, match="layout"):
        summarize_routing(heads, mask, 0, 2, ids[:-1], names)


def test_peft_wrapped_policy_is_instrumented_without_bypassing_active_adapters():
    peft = pytest.importorskip("peft")
    base, batch = make_policy_and_batch()
    wrapped = peft.get_peft_model(base, peft.LoraConfig(r=2, target_modules=["q_proj"]))
    torch.manual_seed(71)
    expected = wrapped.select_action(batch)
    wrapped.reset()
    torch.manual_seed(71)
    with PI05AttentionRecorder(wrapped, AttentionVideoConfig()) as recorder:
        actual = wrapped.select_action(batch)
        assert wrapped._eval_attention_visualizer is recorder
        assert recorder.policy is base
        assert len(recorder.maps) == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not hasattr(wrapped, "_eval_attention_visualizer")


@pytest.mark.parametrize(
    "kwargs", [{"source": "action_prompt"}, {"alpha": 1.5}, {"vmax": float("nan")}, {"layer": 99}]
)
def test_invalid_settings_fail_before_inference(kwargs):
    policy, _ = make_policy_and_batch()
    with pytest.raises(ValueError), PI05AttentionRecorder(policy, AttentionVideoConfig(**kwargs)):
        pass
    assert not hasattr(policy, "_eval_attention_visualizer")
