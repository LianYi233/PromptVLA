"""Regression coverage for stateless, task/episode/camera-locked LIBERO RGB noise."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.scripts import libero_fixed_noise as noise
from lerobot.scripts import pi05_gripper_event_experiment as experiment


def config(std=0.10, seed=20261102, scope="task8", eval_seed=1000):
    return {
        "protocol": noise.PROTOCOL,
        "kind": "gaussian",
        "std": std,
        "seed": seed,
        "scope": scope,
        "eval_seed": eval_seed,
        "suite": "libero_10",
        "target_task_id": 8,
    }


def observation():
    return {
        "observation.images.image": torch.full((1, 3, 20, 20), 0.5),
        "observation.images.image2": torch.full((1, 3, 20, 20), 0.4),
        "observation.state": torch.tensor([[1.0, 2.0]]),
        "task": ["put both moka pots on the stove"],
    }


def test_repeated_calls_reproduce_exact_pixels_and_leave_observation_untouched():
    original = observation()
    torch.manual_seed(77)
    before_torch = torch.get_rng_state().clone()
    np_before = np.random.get_state()

    a = noise.FixedGaussianImagePreprocessor(lambda x: x, config(), "libero_10", 8, 1001)
    b = noise.FixedGaussianImagePreprocessor(lambda x: x, config(), "libero_10", 8, 1001)
    for step in range(3):
        out_a, out_b = a(original), b(original)
        for camera in ("observation.images.image", "observation.images.image2"):
            assert torch.equal(out_a[camera], out_b[camera]), (step, camera)
            assert not torch.equal(out_a[camera], original[camera])
            assert bool(((out_a[camera] >= 0) & (out_a[camera] <= 1)).all())
        assert torch.equal(out_a["observation.state"], original["observation.state"])
        assert out_a["task"] == original["task"]
    assert torch.equal(original["observation.images.image"], torch.full((1, 3, 20, 20), 0.5))
    assert torch.equal(before_torch, torch.get_rng_state())
    np.testing.assert_equal(np_before, np.random.get_state())


def test_independent_episode_camera_step_seed_and_model_identity():
    values = observation()
    a = noise.FixedGaussianImagePreprocessor(lambda x: x, config(), "libero_10", 8, 1001)
    b = noise.FixedGaussianImagePreprocessor(lambda x: x, config(), "libero_10", 8, 1002)
    c = noise.FixedGaussianImagePreprocessor(lambda x: x, config(seed=6), "libero_10", 8, 1001)
    first = a(values)
    second = a(values)
    assert not torch.equal(first["observation.images.image"], second["observation.images.image"])
    assert not torch.equal(first["observation.images.image"], b(values)["observation.images.image"])
    assert not torch.equal(first["observation.images.image"], c(values)["observation.images.image"])
    assert noise.stable_seed("camera", 7, "libero_10", 8, 1001, 0, "cam") == noise.stable_seed(
        "camera", 7, "libero_10", 8, 1001, 0, "cam"
    )
    assert not torch.equal(first["observation.images.image"], first["observation.images.image2"])


def test_different_observations_use_the_same_camera_noise_pattern_without_clipping():
    first = observation()
    second = observation()
    second["observation.images.image"] = first["observation.images.image"] + 0.05
    proc_a = noise.FixedGaussianImagePreprocessor(lambda x: x, config(std=0.01), "libero_10", 8, 1001)
    proc_b = noise.FixedGaussianImagePreprocessor(lambda x: x, config(std=0.01), "libero_10", 8, 1001)
    x, y = proc_a(first)["observation.images.image"], proc_b(second)["observation.images.image"]
    torch.testing.assert_close(x - first["observation.images.image"],
                               y - second["observation.images.image"], atol=1e-7, rtol=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"std": -0.1}, {"std": float("nan")}, {"std": 2.0},
        {"scope": "plus"}, {"suite": "libero_object"},
        {"kind": "salt_pepper"}, {"seed": -1}, {"target_task_id": 7},
    ],
)
def test_invalid_noise_protocol_rejected(changes):
    with pytest.raises(ValueError):
        noise.validate_config({**config(), **changes})


def test_missing_cameras_is_not_silently_treated_as_clean():
    proc = noise.FixedGaussianImagePreprocessor(lambda x: x, config(), "libero_10", 8, 1000)
    with pytest.raises(RuntimeError, match="No LIBERO camera"):
        proc({"observation.state": torch.ones(1, 8)})


def test_task_scope_is_explicit_and_never_affects_other_suites():
    assert noise.task_is_noisy(config(), "libero_10", 8)
    assert not noise.task_is_noisy(config(), "libero_10", 7)
    assert noise.task_is_noisy(config(scope="all"), "libero_10", 7)
    with pytest.raises(ValueError, match="different benchmark"):
        noise.task_is_noisy(config(), "libero_object", 8)


def test_install_binds_episode_seed_and_reseeds_policy_per_rollout(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config()))
    evaluator = SimpleNamespace()

    def orig_rollout(*, env, seeds, env_preprocessor):
        base = observation()
        image0 = env_preprocessor(base)["observation.images.image"].clone()
        image1 = env_preprocessor(base)["observation.images.image"].clone()
        return image0, image1, torch.randn(3)

    def orig_one(task_group, task_id, env, *, start_seed):
        return evaluator.rollout(
            env=env, seeds=[start_seed], env_preprocessor=lambda value: value
        )

    evaluator.rollout = orig_rollout
    evaluator.run_one = orig_one
    manifest = noise.install(evaluator, path)
    assert manifest["fixed_observation_noise"]["config"] == config()

    a = evaluator.run_one("libero_10", 8, object(), start_seed=1000)
    torch.randn(200)  # Consuming global RNG must not change the next matching rollout.
    b = evaluator.run_one("libero_10", 8, object(), start_seed=1000)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert not torch.equal(a[0], a[1])
    c = evaluator.run_one("libero_10", 8, object(), start_seed=1001)
    assert not torch.equal(a[0], c[0])

    clean = evaluator.run_one("libero_10", 7, object(), start_seed=1000)
    assert torch.equal(clean[0], observation()["observation.images.image"])
    torch.randn(100)
    clean_repeated = evaluator.run_one("libero_10", 7, object(), start_seed=1000)
    assert torch.equal(clean[2], clean_repeated[2])
    with pytest.raises(ValueError):
        evaluator.run_one("libero_object", 8, object(), start_seed=1000)


def test_separate_output_roots_and_same_protocol_across_models(tmp_path):
    def arguments(std, seed=12, scope="task8"):
        return SimpleNamespace(
            output_root=tmp_path, eval_root=None, repeat=1,
            eval_seed=1000, episodes=10,
            noise_std=std, noise_seed=seed, noise_scope=scope,
        )

    record = {"selection": {"task_id": 8, "episode_index": 10}}
    a = experiment.evaluation_root(arguments(0.10), record)
    b = experiment.evaluation_root(arguments(0.10), record)
    assert a == b
    assert a != experiment.evaluation_root(arguments(0.15), record)
    assert a != experiment.evaluation_root(arguments(0.10, seed=13), record)
    assert a != experiment.evaluation_root(arguments(0.10, scope="all"), record)
    clean = experiment.evaluation_root(arguments(0.0), record)
    assert "/fixed-image-noise/" not in str(clean)
    matched = arguments(0.0)
    matched.fixed_eval = True
    assert "/fixed-image-noise/" in str(experiment.evaluation_root(matched, record))
    assert experiment.fixed_noise_config(matched)["std"] == 0
    assert "/fixed-image-noise/" in str(a)
    assert experiment.fixed_noise_config(arguments(0.10)) == {
        **config(std=0.10, seed=12), "eval_seed": 1000
    }


def test_real_resume_wrapper_installs_corruption_before_progress_capture():
    script = (Path(__file__).resolve().parents[2] / "run_eval_libero-full.sh").read_text()
    assert script.index("install_fixed_noise(module,") < script.index("original_rollout = module.rollout")
    assert "**fixed_noise_manifest" in script
    assert "LIBERO_FIXED_NOISE_CONFIG" in script


def test_summary_enforces_identical_protocols_and_checksums(tmp_path):
    args = SimpleNamespace(
        output_root=tmp_path, eval_root=None, repeat=1,
        eval_seed=1000, episodes=2, noise_std=0.10,
        noise_seed=20261102, noise_scope="task8",
    )
    record = {
        "protocol": "pi05-gripper-additive-long-v1",
        "selection": {"task_id": 8, "episode_index": 10},
        "settings": {
            "base_steps": 3000, "extra_steps": 750,
            "steps": 3750, "data_seed": 0,
        },
    }
    protocol = experiment.fixed_noise_config(args)
    protocol_path = experiment.fixed_noise_path(args, record)
    experiment.ensure_noise_config(protocol_path, protocol)
    jobs = experiment.build_jobs(record, protocol_path.parent, args.output_root)

    for job in jobs:
        result_file = experiment.result_path(job, record, args)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps({
            "per_task": [
                {
                    "task_group": "libero_10",
                    "task_id": task_id,
                    "metrics": {"successes": [True, False]},
                }
                for task_id in range(10)
            ]
        }))
        (result_file.parent / "resume_manifest.json").write_text(json.dumps({
            "fixed_observation_noise": {
                "config": protocol,
                "config_sha256": experiment.file_hash(protocol_path),
                "helper_sha256": "same-implementation",
            }
        }))

    summary = experiment.summarize(record, jobs, args)
    assert len(summary["results"]) == 4
    assert summary["fixed_observation_noise"] == protocol
    assert summary["success_rate_scope"] == "task8 corrupted; other nine tasks clean"

    altered_path = experiment.result_path(jobs[-1], record, args).parent / "resume_manifest.json"
    altered = json.loads(altered_path.read_text())
    altered["fixed_observation_noise"]["config"]["seed"] = 1
    altered_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="noise protocol"):
        experiment.summarize(record, jobs, args)
