"""Task7 dual-seed object shift, isolated from legacy v1 helper and caches.

scene_seed controls ONLY per-object physical displacement; policy_seed
controls ONLY the torch RNG used by the policy's flow sampling. eval_seed
is FIXED across calibration/validation and indexes official init states
0..9. All components are shared by the compared models.
"""

import functools
import hashlib
import inspect
import json
import math
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import torch

from lerobot.scripts import pi05_task7_object_shift as legacy

PROTOCOL = "promptvla-libero-task7-tabletop-shift-dualseed-v1"


def validate_config(config):
    required = {
        "protocol", "task_id", "suite", "shift_m",
        "scene_seed", "policy_seed", "eval_seed",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Dual-seed config has missing or extra fields")
    if (
        config["protocol"] != PROTOCOL
        or config["task_id"] != legacy.TASK_ID
        or config["suite"] != legacy.TASK_SUITE
    ):
        raise ValueError("Dual-seed shift only supports standard LIBERO-long task7")
    magnitude = config["shift_m"]
    if (
        type(magnitude) not in (float, int) or not math.isfinite(magnitude)
        or not 0 < magnitude <= legacy.MAX_SHIFT_M
    ):
        raise ValueError("Invalid dual-seed displacement magnitude")
    for key in ("scene_seed", "policy_seed", "eval_seed"):
        value = config[key]
        if type(value) is not int or not 0 <= value < 2**31:
            raise ValueError(f"{key} must be a nonnegative integer < 2^31")
    return config


def read_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def slot(rollout_seed, config):
    rollout_seed = int(rollout_seed)
    if not config["eval_seed"] <= rollout_seed < config["eval_seed"] + 10:
        raise ValueError("Dual-seed task7 evaluates exactly 10 official init state slots")
    return rollout_seed - config["eval_seed"]


def init_state_for_episode(rollout_seed, config, total):
    if total < 10:
        raise ValueError("Need at least 10 official task7 init states")
    return slot(rollout_seed, config) % total


def scene_rng_seed(config, slot_index, object_name):
    validate_config(config)
    return legacy.stable_seed(
        "shift-v2", config["scene_seed"], config["suite"], config["task_id"],
        slot_index, object_name, config["shift_m"],
    )


def policy_rng_seed(config, slot_index):
    validate_config(config)
    return legacy.stable_seed(
        "policy-v2", config["policy_seed"], config["suite"],
        config["task_id"], slot_index,
    ) % (2**31)


def choose_translations(original_poses, config, rollout_seed):
    validate_config(config)
    names = legacy.TARGET_OBJECTS + legacy.OTHER_OBJECTS
    if set(original_poses) != set(names):
        raise ValueError("Task7 dual-seed displacement needs all five known scene objects")
    positions = {name: np.asarray(pos, dtype=np.float64).copy()
                 for name, pos in original_poses.items()}
    if any(p.shape != (7,) or not np.isfinite(p).all() for p in positions.values()):
        raise ValueError("Invalid task7 object poses")
    frame_slot = slot(rollout_seed, config)
    moved = {}
    for name in legacy.TARGET_OBJECTS:
        rng = np.random.Generator(
            np.random.PCG64(scene_rng_seed(config, frame_slot, name))
        )
        for _ in range(legacy.MAX_ATTEMPTS):
            angle = rng.uniform(-math.pi, math.pi)
            delta = np.array([math.cos(angle), math.sin(angle)]) * config["shift_m"]
            candidate = positions[name][:2] + delta
            neighbors = {
                other: (moved[other][:2] if other in moved else positions[other][:2])
                for other in positions if other != name
            }
            within_table = (
                abs(candidate[0]) <= legacy.TABLE_X_LIMIT_M
                and abs(candidate[1]) <= legacy.TABLE_Y_LIMIT_M
            )
            if within_table and all(
                np.linalg.norm(candidate - p) >= legacy.MIN_OBJECT_SEPARATION_M
                for p in neighbors.values()
            ):
                pose = positions[name].copy()
                pose[:2] = candidate
                moved[name] = pose
                break
        else:
            raise RuntimeError(
                f"No safe dual-seed displacement: scene={config['scene_seed']}, "
                f"slot={frame_slot}, name={name}, magnitude={config['shift_m']}"
            )
    return moved


def apply_shift(wrapper, config, rollout_seed):
    if wrapper.task_id != legacy.TASK_ID or wrapper._env is None:
        raise ValueError("Dual-seed object shift requires live task7 MuJoCo environment")
    all_names = legacy.TARGET_OBJECTS + legacy.OTHER_OBJECTS
    joints, original = {}, {}
    for name in all_names:
        joints[name], original[name] = legacy.object_free_joint(wrapper, name)
    moved = choose_translations(original, config, rollout_seed)
    sim = wrapper._env.sim
    for name, pose in moved.items():
        sim.data.set_joint_qpos(joints[name], pose)
    sim.forward()
    obs = wrapper._env.regenerate_obs_from_state(wrapper._env.get_sim_state())
    report = {
        name: [
            round(float(moved[name][coord] - original[name][coord]), 6)
            for coord in (0, 1)
        ]
        for name in legacy.TARGET_OBJECTS
    }
    print("TASK7_DUALSEED_SHIFT " + json.dumps({
        "scene_seed": config["scene_seed"],
        "policy_seed": config["policy_seed"],
        "init_state": init_state_for_episode(
            rollout_seed, config, len(wrapper._init_states)
        ),
        "slot": slot(rollout_seed, config),
        "shift_m": config["shift_m"],
        "xy_delta": report,
    }, sort_keys=True), flush=True)
    return obs


def install(evaluator, path):
    path = Path(path).resolve(strict=True)
    config = read_config(path)
    old_run_one = evaluator.run_one
    old_rollout = evaluator.rollout
    signature = inspect.signature(old_rollout)
    context = ContextVar("task7_dualseed_context", default=None)

    @functools.wraps(old_run_one)
    def run_one(task_group, task_id, env, **kwargs):
        if (task_group, int(task_id)) != (legacy.TASK_SUITE, legacy.TASK_ID):
            raise ValueError("Dual-seed evaluator can only run standard task7")
        token = context.set((task_group, task_id))
        try:
            return old_run_one(task_group, task_id, env, **kwargs)
        finally:
            context.reset(token)

    @functools.wraps(old_rollout)
    def rollout(*args, **kwargs):
        if context.get() != (legacy.TASK_SUITE, legacy.TASK_ID):
            raise RuntimeError("Dual-seed rollout outside task7")
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        seeds = bound.arguments["seeds"]
        if not isinstance(seeds, (tuple, list)) or len(seeds) != 1:
            raise ValueError("Dual-seed evaluation requires one explicitly seeded episode")
        frame_slot = slot(int(seeds[0]), config)
        torch.manual_seed(policy_rng_seed(config, frame_slot))
        return old_rollout(*bound.args, **bound.kwargs)

    evaluator.run_one = run_one
    evaluator.rollout = rollout
    return {
        "task7_dualseed_shift": {
            "config": config,
            "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "fixed init-state slot; independent XY scene and torch policy seeds",
            "scene_rng_rule": "SHA256(scene_seed, slot, object, shift_m)",
            "policy_rng_rule": "SHA256(policy_seed, slot) independent of scene",
        },
    }
