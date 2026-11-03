"""Deterministic physical tabletop displacement for LIBERO-long task7.

Only the two target objects, alphabet_soup_1 and cream_cheese_1, move by a
small planar translation after loading a saved LIBERO init state and before
the original settlement steps. No image noise, basket movement or model edit.
Every displacement is a pure function of (protocol, seed, suite, task, rollout
seed, object name, shift magnitude), independent of model and GPU.
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

PROTOCOL = "promptvla-libero-task7-tabletop-shift-v1"
TASK_ID = 7
TASK_SUITE = "libero_10"
TARGET_OBJECTS = ("alphabet_soup_1", "cream_cheese_1")
OTHER_OBJECTS = ("tomato_sauce_1", "ketchup_1", "basket_1")
DEFAULT_SHIFT_SEED = 20261103
# Official task7 living-room workspace: 0.70m x 1.60m, at XY=(0, 0).
# Maintain a conservative 5cm edge margin for object centres.
MAX_SHIFT_M = 0.14
TABLE_X_LIMIT_M = 0.30
TABLE_Y_LIMIT_M = 0.75
MIN_OBJECT_SEPARATION_M = 0.07
MAX_ATTEMPTS = 256


def validate_config(config):
    required = {"protocol", "task_id", "suite", "shift_m", "seed", "eval_seed"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Task7 object shift protocol has missing or extra keys")
    if config["protocol"] != PROTOCOL or config["task_id"] != TASK_ID or config["suite"] != TASK_SUITE:
        raise ValueError("Object shift supports only the standard LIBERO-long task7")
    shift = config["shift_m"]
    if type(shift) not in (float, int) or not math.isfinite(shift) or not 0 < shift <= MAX_SHIFT_M:
        raise ValueError(f"shift_m must be in (0,{MAX_SHIFT_M}] meters")
    for name in ("seed", "eval_seed"):
        value = config[name]
        if type(value) is not int or not 0 <= value < 2**31:
            raise ValueError(f"{name} must be an integer in [0,2**31)")
    return config


def read_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def stable_seed(*parts):
    source = json.dumps([PROTOCOL, *parts], ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(source.encode()).digest()[:8], "little")


def init_state_for_episode(rollout_seed, config, total):
    """Episode #0 uses init state #0, independent of automatic reset count."""
    if type(rollout_seed) is not int or rollout_seed < config["eval_seed"]:
        raise ValueError("Task7 rollout seeds must begin at eval_seed")
    if type(total) is not int or total <= 0:
        raise ValueError("Missing official saved task init states")
    return (rollout_seed - config["eval_seed"]) % total


def object_free_joint(wrapper, object_name):
    objects = wrapper._env.env.objects_dict
    if object_name not in objects:
        raise ValueError(f"Object {object_name} absent from task7 BDDL scene")
    joints = objects[object_name].joints
    if not isinstance(joints, (list, tuple)) or len(joints) != 1:
        raise ValueError(f"Object {object_name} must have exactly one movable free joint")
    joint = joints[0]
    pose = np.asarray(wrapper._env.sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"Object {object_name} does not expose a finite 7D free joint")
    return joint, pose


def choose_translations(original_poses, config, rollout_seed):
    """Pick exact-radius XY displacements, avoiding close object centers.

    Reject draws that bring either target within 7 cm of any task object.
    Reject any candidate that leaves the safe tabletop envelope (5 cm
    edge margin) or crowds another object; this lets the calibrated
    displacement increase without silently moving an object off-table.
    """
    validate_config(config)
    required = set(TARGET_OBJECTS + OTHER_OBJECTS)
    if set(original_poses) != required:
        raise ValueError(f"Expected task7 movable objects {sorted(required)}")
    positions = {name: np.asarray(p, dtype=np.float64).copy() for name, p in original_poses.items()}
    if any(p.shape != (7,) or not np.isfinite(p).all() for p in positions.values()):
        raise ValueError("Expected finite 7-D free-joint positions")
    moved = {}
    for name in TARGET_OBJECTS:
        rng = np.random.Generator(
            np.random.PCG64(
                stable_seed("shift", config["seed"], config["suite"],
                            config["task_id"], rollout_seed, name, config["shift_m"])
            )
        )
        for _ in range(MAX_ATTEMPTS):
            angle = rng.uniform(-math.pi, math.pi)
            delta = np.asarray([math.cos(angle), math.sin(angle)]) * config["shift_m"]
            candidate = positions[name][:2] + delta
            other_positions = {
                other: (moved[other][:2] if other in moved else positions[other][:2])
                for other in positions if other != name
            }
            on_table = (
                abs(candidate[0]) <= TABLE_X_LIMIT_M
                and abs(candidate[1]) <= TABLE_Y_LIMIT_M
            )
            if on_table and all(
                np.linalg.norm(candidate - p) >= MIN_OBJECT_SEPARATION_M
                for p in other_positions.values()
            ):
                pose = positions[name].copy()
                pose[:2] = candidate
                moved[name] = pose
                break
        else:
            raise RuntimeError(
                f"No tabletop-safe shift for {name} at {config['shift_m']} m, "
                f"episode seed={rollout_seed}; choose a lower displacement"
            )
    return moved


def apply_shift(wrapper, config, rollout_seed):
    """Apply to live MuJoCo, refresh observations, then let LiberoEnv settle."""
    if wrapper.task_id != TASK_ID:
        raise ValueError("Task7 shift must not be applied to any other task")
    if wrapper._env is None:
        raise RuntimeError("Simulation not initialized")
    targets = TARGET_OBJECTS + OTHER_OBJECTS
    joints = {}
    original_poses = {}
    for name in targets:
        joints[name], original_poses[name] = object_free_joint(wrapper, name)
    moves = choose_translations(original_poses, config, rollout_seed)
    sim = wrapper._env.sim
    for name, pose in moves.items():
        sim.data.set_joint_qpos(joints[name], pose)
    sim.forward()
    # Calling the official wrapper refreshes dynamics, predicates and cameras
    # from the post-shift state. The original reset's noop settlement follows.
    observation = wrapper._env.regenerate_obs_from_state(wrapper._env.get_sim_state())
    report = {
        name: [
            round(float(moves[name][i] - original_poses[name][i]), 6)
            for i in (0, 1)
        ] for name in TARGET_OBJECTS
    }
    print(
        "TASK7_FIXED_SHIFT "
        + json.dumps(
            {"rollout_seed": rollout_seed, "init_state": init_state_for_episode(
                rollout_seed, config, len(wrapper._init_states)
            ), "shift_m": config["shift_m"], "xy_delta": report},
            sort_keys=True,
        ),
        flush=True,
    )
    return observation


def install(evaluator, path):
    """Match policy stochastic noise per episode and record the perturbation."""
    path = Path(path).resolve(strict=True)
    config = read_config(path)
    old_one = evaluator.run_one
    old_rollout = evaluator.rollout
    signature = inspect.signature(old_rollout)
    context = ContextVar("libero_task7_shift_context", default=None)

    @functools.wraps(old_one)
    def run_one(task_group, task_id, env, **kwargs):
        if (task_group, int(task_id)) != (TASK_SUITE, TASK_ID):
            raise ValueError("Task7 object shift must evaluate only libero_10 task7")
        token = context.set((task_group, task_id))
        try:
            return old_one(task_group, task_id, env, **kwargs)
        finally:
            context.reset(token)

    @functools.wraps(old_rollout)
    def rollout(*args, **kwargs):
        if context.get() != (TASK_SUITE, TASK_ID):
            raise RuntimeError("Task7 shift rollout invoked outside selected task")
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        seeds = bound.arguments["seeds"]
        if not isinstance(seeds, (list, tuple)) or len(seeds) != 1:
            raise ValueError("Task7 object shift requires one explicitly seeded episode")
        rollout_seed = int(seeds[0])
        init_state_for_episode(rollout_seed, config, 1)
        # Equal action-flow random stream for the same episode across all
        # four checkpoints, independent of previous rollout lengths/resume.
        policy_seed = stable_seed(
            "policy", config["seed"], config["suite"], config["task_id"],
            rollout_seed,
        ) % (2**31)
        torch.manual_seed(policy_seed)
        return old_rollout(*bound.args, **bound.kwargs)

    evaluator.run_one = run_one
    evaluator.rollout = rollout
    return {
        "task7_object_shift": {
            "config": config,
            "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "LIBERO-long task7: two object free joints, XY only, fixed episode reset",
            "policy_rng_rule": "SHA256-keyed per episode; same among four models",
        }
    }
