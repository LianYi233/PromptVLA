"""Exercise real evaluator/video encoding with a tiny policy and a toy Gym env."""

import json
from pathlib import Path

import av
import gymnasium as gym
import numpy as np
import pytest
import torch

from lerobot.scripts import lerobot_eval
from lerobot.scripts.libero_attention import install_attention_evaluation, require_saved_attention_videos
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from tests.policies.pi0_pi05.test_pi05_attention_visualization import (
    _test_font,  # noqa: F401
    make_policy_and_batch,
)
from tests.policies.pi0_pi05.test_pi05_prompt import (
    _restore_matmul_precision,  # noqa: F401
    _tiny_real_pi05,  # noqa: F401
)


class TinyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    task_description = "move the object"
    task = "test"
    _max_episode_steps = 5

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, (16, 16, 3), dtype=np.uint8) for key in ("image", "image2")}
                ),
            }
        )
        self.action_space = gym.spaces.Box(-10, 10, (3,), dtype=np.float32)

    def _observation(self):
        return {"pixels": {key: self.render() for key in ("image", "image2")}}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        self.succeed = seed % 2 == 0
        return self._observation(), {}

    def step(self, action):
        self.step_index += 1
        return self._observation(), 1.0, self.step_index == 5, False, {"is_success": self.succeed}

    def render(self):
        return np.full((16, 16, 3), self.step_index * 30, dtype=np.uint8)


@pytest.mark.parametrize("two_cameras", [False, True])
def test_actual_eval_saves_success_failure_videos_and_resumable_raw_maps(tmp_path, monkeypatch, two_cameras):
    original_one, original_rollout = lerobot_eval.run_one, lerobot_eval.rollout
    monkeypatch.setattr(lerobot_eval, "run_one", original_one)
    monkeypatch.setattr(lerobot_eval, "rollout", original_rollout)
    monkeypatch.setenv("ATTENTION_SOURCE", "vlm_prompt" if two_cameras else "action")
    monkeypatch.setenv("ATTENTION_CAMERAS", "0,1" if two_cameras else "")
    monkeypatch.setenv("ATTENTION_SNAPSHOT_EVERY", "2" if two_cameras else "0")
    monkeypatch.setenv("ATTENTION_RECORD_ROUTING", "1")
    manifest = install_attention_evaluation(lerobot_eval)
    assert json.loads(json.dumps(manifest)) == manifest  # resume manifest must round-trip
    policy, _ = make_policy_and_batch()
    env = gym.vector.SyncVectorEnv([TinyEnv])

    def preprocess(batch):
        return {
            **batch,
            OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 0]]),
            OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, False]]),
        }

    def identity(value):
        return value

    try:
        result = lerobot_eval.run_one(
            "test_suite",
            0,
            env,
            policy=policy,
            env_preprocessor=identity,
            env_postprocessor=identity,
            preprocessor=preprocess,
            postprocessor=identity,
            n_episodes=2,
            max_episodes_rendered=1,
            videos_dir=tmp_path,
            return_episode_data=False,
            start_seed=0,
        )
    finally:
        env.close()
    metrics = result[2]
    assert metrics["successes"] == [True, False]
    assert len(metrics["video_paths"]) == 2  # save every episode despite incoming limit
    assert "TRUE_attention" in metrics["video_paths"][0]
    assert "FALSE_attention" in metrics["video_paths"][1]
    for path in metrics["video_paths"]:
        video = Path(path)
        with av.open(str(video)) as reader:
            frames = list(reader.decode(video=0))
            assert len(frames) == 5  # preserve existing evaluator's frame count
            assert (frames[0].width, frames[0].height) == ((640, 784) if two_cameras else (768, 320))
        with np.load(video.with_suffix(".attention.npz")) as data:
            np.testing.assert_array_equal(data["input_steps"], [0, 2, 4])
            assert data["maps"].shape == (3, 2, 2)
            np.testing.assert_allclose(data["image_attention_mass"], data["maps"].sum(axis=(1, 2)))
            np.testing.assert_allclose(data["routing_head_mass"].sum(axis=-1), 1, atol=2e-6)
            route_mass = data["routing_head_mass"].mean(axis=(1, 2))
            np.testing.assert_allclose(route_mass[:, 0], data["image_attention_mass"], atol=1e-7)
            if two_cameras:
                assert data["camera_maps"].shape == (3, 2, 2, 2)
                assert data["camera_images"].shape == (3, 2, 16, 16, 3)
                np.testing.assert_array_equal(data["maps"], data["camera_maps"][:, 0])
                np.testing.assert_allclose(
                    data["camera_attention_mass"], data["camera_maps"].sum(axis=(-2, -1))
                )
        meta = json.loads(video.with_suffix(".attention.json").read_text())
        assert meta["camera_feature"] == "observation.images.image"
        assert meta["resolved_layer_zero_based"] == 1
        assert meta["routing_schema"]["version"] == 1
        routing_summary = video.with_suffix(".attention.routing_summary.json")
        summary = json.loads(routing_summary.read_text())
        assert summary["max_probability_closure_error"] < 2e-6
        assert summary["prediction_count"] == 3
        assert summary["display_prediction_mean_mass"]["language_padding"] == 0
        assert video.with_suffix(".attention.routing.csv").stat().st_size > 0
        assert video.with_suffix(".attention.routing_top_tokens.csv").stat().st_size > 0
        routing_summary.unlink()  # resume must rebuild summaries from the NPZ
        if two_cameras:
            figures = video.parent / f"{video.stem}_frames"
            assert sorted(p.name for p in figures.glob("*.png")) == ["step_000000.png", "step_000004.png"]
            assert len((figures / "camera_attention_mass.csv").read_text().splitlines()) == 4
            (figures / "step_000000.png").unlink()  # resume restores figures without inference
    require_saved_attention_videos(metrics)
    for path in metrics["video_paths"]:
        assert Path(path).with_suffix(".attention.routing_summary.json").is_file()
    if two_cameras:
        for path in metrics["video_paths"]:
            video = Path(path)
            assert (video.parent / f"{video.stem}_frames" / "step_000000.png").is_file()
        return  # legacy restyling below is specific to the single-camera layout
    from lerobot.scripts.replot_libero_attention import replot_video

    original = Path(metrics["video_paths"][0])
    restyled = tmp_path / "restyled" / original.name
    assert replot_video(original, restyled) == "5 frames"
    assert (
        original.with_suffix(".attention.npz").read_bytes()
        == restyled.with_suffix(".attention.npz").read_bytes()
    )
    with av.open(str(restyled)) as reader:
        assert len(list(reader.decode(video=0))) == 5
    assert replot_video(original, restyled) == "already done"
    with pytest.raises(ValueError, match="Different output"):
        replot_video(original, restyled, alpha=0.9)
    Path(metrics["video_paths"][0]).with_suffix(".attention.npz").unlink()
    with pytest.raises(RuntimeError, match="Missing saved attention artifact"):
        require_saved_attention_videos(metrics)
    assert not hasattr(policy, "_eval_attention_visualizer")
