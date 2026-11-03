"""Task7 physical-perturbation reproducibility and two-stage evaluation guards."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.scripts import pi05_task7_object_shift as shift
from lerobot.scripts import pi05_task7_shift_calibrate as calibrate
from lerobot.scripts import pi05_gripper_event_experiment as runner


def config(magnitude=0.04, seed=20261103, eval_seed=1000):
    return {
        "protocol": shift.PROTOCOL,
        "task_id": 7,
        "suite": "libero_10",
        "shift_m": magnitude,
        "seed": seed,
        "eval_seed": eval_seed,
    }


def poses():
    def pose(x, y):
        return np.asarray([x, y, 0.82, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    return {
        "alphabet_soup_1": pose(0.05, -0.10),
        "cream_cheese_1": pose(-0.15, 0.06),
        "tomato_sauce_1": pose(0.10, -0.20),
        "ketchup_1": pose(-0.20, -0.15),
        "basket_1": pose(0.0, 0.26),
    }


def test_fixed_displacements_are_identical_across_cases_and_change_across_episodes():
    original = poses()
    np_before = np.random.get_state()
    a = shift.choose_translations(original, config(), 1000)
    b = shift.choose_translations(original, config(), 1000)
    c = shift.choose_translations(original, config(), 1001)
    np.testing.assert_equal(np_before, np.random.get_state())
    assert set(a) == set(shift.TARGET_OBJECTS)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
        assert not np.array_equal(a[key][:2], c[key][:2])
        assert np.linalg.norm(a[key][:2] - original[key][:2]) == pytest.approx(0.04)
        np.testing.assert_array_equal(a[key][2:], original[key][2:])
        assert not np.array_equal(a[key], original[key])
    for key in original:
        assert original[key][2] == 0.82


def test_basket_and_distractors_remain_stationary_and_nonoverlapping():
    original = poses()
    out = shift.choose_translations(original, config(0.06), 1008)
    for moved_name in out:
        new_xy = out[moved_name][:2]
        for other_name, other in original.items():
            if other_name == moved_name:
                continue
            other_xy = out.get(other_name, other)[:2]
            assert np.linalg.norm(new_xy - other_xy) >= shift.MIN_OBJECT_SEPARATION_M - 1e-8
    assert set(out) == {"alphabet_soup_1", "cream_cheese_1"}
    assert "basket_1" not in out


class FakeData:
    def __init__(self, objects):
        self.objects = {k: v.copy() for k, v in objects.items()}

    def get_joint_qpos(self, joint):
        return self.objects[joint]

    def set_joint_qpos(self, joint, pose):
        self.objects[joint] = pose.copy()


class FakeSim:
    def __init__(self, objects):
        self.data = FakeData(objects)
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class FakeEnv:
    def __init__(self, objects):
        self.objects_dict = {
            k: SimpleNamespace(joints=[k]) for k in objects
        }


class FakeOffScreen:
    def __init__(self, objects):
        self.sim = FakeSim(objects)
        self.env = FakeEnv(objects)
        self.regenerated = 0

    def get_sim_state(self):
        return self.sim.data.objects

    def regenerate_obs_from_state(self, state):
        self.regenerated += 1
        return {"pixels": {"image": np.zeros((4, 4, 3), dtype=np.uint8)}}


def test_live_mujoco_shift_hook_moves_only_target_free_joints():
    original = poses()
    fake = FakeOffScreen(original)
    wrapper = SimpleNamespace(
        task_id=7, _env=fake, _init_states=[0] * 10
    )
    obs = shift.apply_shift(wrapper, config(), 1000)
    assert "pixels" in obs
    assert fake.sim.forward_calls == 1
    assert fake.regenerated == 1
    for name in shift.TARGET_OBJECTS:
        assert np.linalg.norm(fake.sim.data.objects[name][:2] - original[name][:2]) == pytest.approx(0.04)
    for name in shift.OTHER_OBJECTS:
        np.testing.assert_array_equal(fake.sim.data.objects[name], original[name])


def test_reset_index_independent_of_auto_reset_count():
    assert [shift.init_state_for_episode(1000 + i, config(), 100) for i in range(10)] == list(range(10))
    assert shift.init_state_for_episode(1010, config(), 10) == 0
    with pytest.raises(ValueError):
        shift.init_state_for_episode(999, config(), 10)


@pytest.mark.parametrize("magnitude", (-0.02, 0, float("nan"), 0.16))
def test_invalid_shift_strength_rejected(magnitude):
    with pytest.raises(ValueError):
        shift.validate_config(config(magnitude))


def test_episode_policy_rng_same_across_models_and_resume(tmp_path):
    cfg = tmp_path / "shift.json"
    cfg.write_text(json.dumps(config()))
    evaluator = SimpleNamespace()

    def original_rollout(*, seeds):
        return torch.randn(3)

    def original_one(task_group, task_id, env, *, start_seed):
        return evaluator.rollout(seeds=[start_seed])

    evaluator.run_one, evaluator.rollout = original_one, original_rollout
    identity = shift.install(evaluator, cfg)
    assert identity["task7_object_shift"]["config"] == config()
    a = evaluator.run_one("libero_10", 7, None, start_seed=1001)
    torch.randn(100)
    b = evaluator.run_one("libero_10", 7, None, start_seed=1001)
    assert torch.equal(a, b)
    assert not torch.equal(a, evaluator.run_one("libero_10", 7, None, start_seed=1002))
    with pytest.raises(ValueError):
        evaluator.run_one("libero_10", 8, None, start_seed=1000)


def test_eval_root_and_summary_only_task7(tmp_path):
    args = SimpleNamespace(
        eval_root=tmp_path / "eval", output_root=tmp_path,
        repeat=1, eval_seed=1000, episodes=10,
        only_selected_task=True, fixed_eval=False, noise_std=0,
        shift_m=0.04, shift_seed=20261103,
    )
    record = {"selection": {"task_id": 7, "episode_index": 1}}
    root = runner.evaluation_root(args, record)
    assert "fixed-object-shift/meters-0p040" in str(root)
    assert "task7-only-clean" in str(root)
    assert runner.task7_shift_config(args) == config()
    assert runner.task7_shift_path(args, record) == root / "fixed_object_shift_config.json"
    args.shift_m = 0.06
    assert runner.evaluation_root(args, record) != root
    args.eval_seed = 2000
    assert runner.evaluation_root(args, record) != root


def test_calibration_strength_selection_and_validation(tmp_path):
    assert calibrate._amplitudes("0.06,0.02,0.04") == (0.02, 0.04, 0.06)
    with pytest.raises(ValueError):
        calibrate._amplitudes("0.01,0.16")
    file = tmp_path / "calib.json"
    file.write_text(json.dumps({
        "protocol": calibrate.PROTOCOL,
        "target_success_rate": 50,
        "task_id": 7,
        "episodes": 10,
        "results": [{"shift_m": 0.04, "rate": 50}],
        "selected_shift_m": 0.04,
    }))
    assert calibrate._read_selected(file)["selected_shift_m"] == 0.04


def test_two_gpu_shell_evaluates_uniform_event_then_both_time_cases():
    root = Path(__file__).resolve().parents[2]
    script = (root / "run_eval_libero_gripper_task7_shift_2gpu.sh").read_text()
    assert script.index('run_pair "Phase 1" uniform event') < script.index(
        'run_pair "Phase 2" uniform_time event_time'
    )
    assert "wait \"$p1\"" in script and "wait \"$p2\"" in script
    assert script.index('run_pair "Phase 2"') < script.index("summarize all")
    eval_script = (root / "run_eval_libero_gripper_task7_shift.sh").read_text()
    assert "--only-selected-task --noise-std 0" in eval_script
    env_script = (root / "src/lerobot/envs/libero.py").read_text()
    assert "init_state_for_episode" in env_script
    assert "apply_shift(self, shift_config" in env_script
    full_script = (root / "run_eval_libero-full.sh").read_text()
    assert "**object_shift_manifest" in full_script



def test_stronger_displacement_stays_on_table_and_keeps_vertical_pose():
    original = poses()
    first = shift.choose_translations(original, config(0.12), 1012)
    second = shift.choose_translations(original, config(0.12), 1012)
    for obj in shift.TARGET_OBJECTS:
        np.testing.assert_array_equal(first[obj], second[obj])
        assert np.linalg.norm(first[obj][:2] - original[obj][:2]) == pytest.approx(0.12)
        assert abs(first[obj][0]) <= shift.TABLE_X_LIMIT_M
        assert abs(first[obj][1]) <= shift.TABLE_Y_LIMIT_M
        np.testing.assert_array_equal(first[obj][2:], original[obj][2:])


def test_stronger_displacement_rejects_unreachable_workspace():
    original = poses()
    original["alphabet_soup_1"][:2] = [0.45, 0.0]
    with pytest.raises(RuntimeError, match="No tabletop-safe shift"):
        shift.choose_translations(original, config(0.08), 1001)


def test_strong_pilot_is_separate_from_prior_calibration_file():
    assert calibrate.DEFAULT_CALIBRATION.name == "shift-calibration-strong.json"
    assert calibrate._amplitudes("0.08,0.10,0.12,0.14") == (
        0.08, 0.10, 0.12, 0.14
    )
    assert shift.MAX_SHIFT_M >= 0.14
