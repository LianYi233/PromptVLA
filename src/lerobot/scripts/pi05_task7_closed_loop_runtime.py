"""Opt-in, audited time-branch interventions in the native task7 rollout.

This module does not implement another policy sampler or environment loop.
The ordinary evaluator still resets, observes, calls ``select_action``, runs
its postprocessors, and steps LIBERO. Hooks record those calls and temporarily
intervene only on the added time encoder. In particular, native policy RNG,
RoPE, the 50-action chunk, and the ten-action execution queue are retained.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path


CONFIG_ENV = "LIBERO_TASK7_TIME_INTERVENTION_CONFIG"
PROTOCOL = "pi05-task7-time-intervention-v1"
TRACE_PROTOCOL = "pi05-task7-execution-trace-v1"
MODES = ("correct", "zero", "reversed")
HORIZON = 50
EXECUTED_STEPS = 10
ACTION_DIM = 7
EPISODES = 10
ROOT = Path(__file__).resolve().parents[3]


def validate_config(config):
    fields = {
        "protocol", "intervention", "executed_steps", "horizon", "trace_directory",
        "checkpoint", "scene_seed", "policy_seed", "init_seed", "episodes",
    }
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError("Time intervention config has missing or extra fields")
    if (config["protocol"] != PROTOCOL or config["intervention"] not in MODES
            or type(config["executed_steps"]) is not int or config["executed_steps"] != EXECUTED_STEPS
            or type(config["horizon"]) is not int or config["horizon"] != HORIZON
            or type(config["episodes"]) is not int or config["episodes"] != EPISODES):
        raise ValueError("Time intervention requires Correct/Zero/Reversed, chunk50, execution10, ten episodes")
    for name in ("scene_seed", "policy_seed", "init_seed"):
        value = config[name]
        if type(value) is not int or not 0 <= value < 2**31:
            raise ValueError(f"{name} must be a nonnegative integer < 2^31")
    if config["init_seed"] > 2**31 - EPISODES:
        raise ValueError("The ten rollout seeds must remain below 2^31")
    for name in ("checkpoint", "trace_directory"):
        value = config[name]
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
        if str(Path(value).resolve()) != value:
            raise ValueError(f"{name} must be a resolved absolute path")
    return config


def read_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints():
    return {
        "helper_sha256": _file_sha256(__file__),
        "environment_sha256": _file_sha256(ROOT / "src/lerobot/envs/libero.py"),
        "model_sha256": _file_sha256(ROOT / "src/lerobot/policies/pi05/modeling_pi05.py"),
        "action_time_sha256": _file_sha256(ROOT / "src/lerobot/policies/pi05/action_time.py"),
        "counterfactual_sha256": _file_sha256(ROOT / "src/lerobot/scripts/pi05_task7_time_evidence.py"),
    }


def manifest_for_config(path):
    """Return the inner ``task7_time_intervention`` resume-manifest entry."""
    path = Path(path).resolve(strict=True)
    return {
        "config": read_config(path),
        "config_sha256": _file_sha256(path),
        **source_fingerprints(),
        "sampling": "native PI05 select_action / predict_action_chunk; native per-episode torch RNG",
        "intervention_scope": "added index-based input time branch only; native RoPE retained",
        "trace_protocol": TRACE_PROTOCOL,
    }


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _finite_array(value, shape, name):
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.shape != tuple(shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {tuple(shape)}")
    return array


def _array_hash(value):
    import numpy as np

    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}:".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _contact_pairs(sim):
    """Raw MuJoCo geom contacts, not a grasp/holding classifier."""
    if not hasattr(sim.data, "ncon") or not hasattr(sim.data, "contact"):
        return None
    pairs = set()
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        pairs.add(tuple(sorted((int(contact.geom1), int(contact.geom2)))))
    result = []
    name_lookup = getattr(sim.model, "geom_id2name", None)
    for first, second in sorted(pairs):
        result.append({
            "geom1": first, "geom2": second,
            "geom1_name": name_lookup(first) if callable(name_lookup) else None,
            "geom2_name": name_lookup(second) if callable(name_lookup) else None,
        })
    return result


def environment_state(wrapper, raw_obs):
    """Read actual post-physics state; never integrate normalized commands."""
    import numpy as np
    from lerobot.scripts import pi05_task7_object_shift as objects

    robot = wrapper._env.robots[0]
    controller = robot.controller
    site_id = getattr(robot, "eef_site_id", None)
    # The controller may cache the orientation from before the final physics
    # integration. Prefer MuJoCo's current site matrix for executed geometry.
    if isinstance(site_id, dict) and len(site_id) == 1:
        site_id = next(iter(site_id.values()))
    site_matrices = getattr(wrapper._env.sim.data, "site_xmat", None)
    if isinstance(site_id, (int, np.integer)) and site_matrices is not None:
        rotation = _finite_array(np.asarray(site_matrices[int(site_id)]).reshape(3, 3),
                                 (3, 3), "end-effector site rotation")
        rotation_source = "post-physics simulator site_xmat"
    else:
        rotation = _finite_array(controller.ee_ori_mat, (3, 3), "controller EEF rotation")
        rotation_source = "controller.ee_ori_mat; may be cached before the final physics step"
    simulation = np.asarray(wrapper._env.get_sim_state(), dtype=np.float64)
    if simulation.ndim != 1 or simulation.size == 0 or not np.isfinite(simulation).all():
        raise ValueError("LIBERO must expose a finite flattened simulator state for trace audit")
    states = {}
    for name in objects.TARGET_OBJECTS + objects.OTHER_OBJECTS:
        _, pose = objects.object_free_joint(wrapper, name)
        states[name] = {
            "position": pose[:3].tolist(), "quaternion": pose[3:].tolist(),
            "quaternion_order": "wxyz",
        }
    control_state = {"class": type(controller).__name__, "use_delta": bool(controller.use_delta)}
    for name, shape in (("goal_pos", (3,)), ("goal_ori", (3, 3))):
        value = getattr(controller, name, None)
        if value is not None:
            control_state[name] = _finite_array(value, shape, f"controller.{name}").tolist()
    sim_time = getattr(wrapper._env.sim.data, "time", None)
    if sim_time is not None and not math.isfinite(float(sim_time)):
        raise ValueError("Simulator time is not finite")
    result = {
        "eef_position": _finite_array(raw_obs.get("robot0_eef_pos"), (3,), "EEF position").tolist(),
        "eef_quaternion": _finite_array(raw_obs.get("robot0_eef_quat"), (4,), "EEF quaternion").tolist(),
        "eef_rotation_matrix": rotation.tolist(),
        "eef_rotation_source": rotation_source,
        "gripper_qpos": _finite_array(raw_obs.get("robot0_gripper_qpos"), (2,), "gripper qpos").tolist(),
        "objects": states,
        "simulator_state_sha256": _array_hash(simulation),
        "sim_time": float(sim_time) if sim_time is not None else None,
        "controller_state": control_state,
        "contact_pairs": _contact_pairs(wrapper._env.sim),
    }
    velocity = raw_obs.get("robot0_gripper_qvel")
    if velocity is not None:
        result["gripper_qvel"] = _finite_array(velocity, (2,), "gripper qvel").tolist()
    return result


def arm_environment_trace(wrapper, rollout_seed):
    """Called through vector-env ``call`` before the native seeded reset."""
    if not os.environ.get(CONFIG_ENV):
        raise RuntimeError("Task7 trajectory capture was not enabled")
    config = read_config(os.environ[CONFIG_ENV])
    if (wrapper.task_suite_name, wrapper.task_id) != ("libero_10", 7):
        raise ValueError("Time interventions only support standard LIBERO task7")
    if wrapper.control_mode != "relative" or wrapper.is_libero_plus:
        raise ValueError("Time interventions require standard relative-control LIBERO")
    if type(rollout_seed) is not int or not config["init_seed"] <= rollout_seed < config["init_seed"] + EPISODES:
        raise ValueError("Trace rollout seed is outside the ten declared init states")
    wrapper._task7_execution_trace = {
        "rollout_seed": rollout_seed, "initial_state": None,
        "steps": [], "sealed": False,
    }
    return {"armed": True, "rollout_seed": rollout_seed}


def record_environment_reset(wrapper, raw_obs, seed):
    trace = getattr(wrapper, "_task7_execution_trace", None)
    if trace is None:
        return
    # LiberoEnv.step and some Gymnasium versions automatically reset after
    # termination. Their seed=None reset must not overwrite the terminal trace.
    if seed is None:
        return
    if int(seed) != trace["rollout_seed"] or trace["initial_state"] is not None:
        raise RuntimeError("Unexpected seeded reset while recording a task7 episode")
    trace["initial_state"] = environment_state(wrapper, raw_obs)


def record_environment_step(wrapper, action, raw_obs, *, reward, success, terminated, truncated):
    trace = getattr(wrapper, "_task7_execution_trace", None)
    if trace is None or trace["sealed"]:
        return
    if trace["initial_state"] is None:
        raise RuntimeError("Task7 trace has no seeded, settled initial state")
    if not math.isfinite(float(reward)):
        raise ValueError("Non-finite simulator reward")
    trace["steps"].append({
        "step": len(trace["steps"]),
        "executed_action": _finite_array(action, (ACTION_DIM,), "executed action").tolist(),
        "state_after": environment_state(wrapper, raw_obs),
        "reward": float(reward), "success": bool(success),
        "terminated": bool(terminated), "truncated": bool(truncated),
    })
    trace["sealed"] = bool(terminated or truncated)


def collect_environment_trace(wrapper):
    trace = getattr(wrapper, "_task7_execution_trace", None)
    if trace is None or trace["initial_state"] is None or not trace["steps"]:
        raise RuntimeError("No completed native execution was captured")
    # A timeout is imposed by the native evaluator, not LiberoEnv.step. Sealing
    # here also preserves that last state if a vector-env reset follows later.
    trace["sealed"] = True
    return copy.deepcopy(trace)


@contextmanager
def _replace_method(instance, name, replacement):
    original = getattr(instance, name)
    had_override = name in instance.__dict__
    try:
        setattr(instance, name, replacement)
        yield original
    finally:
        if had_override:
            setattr(instance, name, original)
        else:
            delattr(instance, name)


@contextmanager
def audit_policy(policy, intervention):
    """Retain native randomness and queueing while recording each actual chunk."""
    import numpy as np
    import torch
    from lerobot.scripts.pi05_task7_full_sampling import (
        observation_sha256, tensor_sha256, validate_sampling_config,
    )
    from lerobot.scripts.pi05_task7_time_evidence import time_counterfactual

    if intervention not in MODES:
        raise ValueError("Unknown time intervention")
    validate_sampling_config(policy.config)
    if (policy.config.n_action_steps != EXECUTED_STEPS or policy.config.compile_model
            or policy._rtc_enabled()):
        raise ValueError("Closed-loop intervention requires native eager sampling, RTC off and execution10")
    encoder = getattr(policy.model, "action_time_encoder", None)
    if encoder is None:
        raise ValueError("Every condition requires the same learned Time-Input model")
    original_predict = policy.predict_action_chunk
    original_select = policy.select_action
    original_noise = policy.model.sample_noise
    original_denoise = policy.model.denoise_step
    recorded = {"chunks": [], "normalized_actions": []}
    active = None

    def sampled_noise(*args, **kwargs):
        if active is None or "noise" in active:
            raise RuntimeError("Expected exactly one native Gaussian draw per action chunk")
        result = original_noise(*args, **kwargs)
        if tuple(result.shape) != (1, HORIZON, policy.config.max_action_dim) or not torch.isfinite(result).all():
            raise ValueError("Native initial noise has invalid shape or values")
        active["noise"] = result.detach().clone()
        active["initial_noise_sha256"] = tensor_sha256(result)
        return result

    def denoise(*args, **kwargs):
        if active is None or "noise" not in active:
            raise RuntimeError("Native denoising did not start with a fresh policy noise draw")
        value, time = kwargs.get("x_t"), kwargs.get("timestep")
        if value is None or time is None or tuple(time.shape) != (1,):
            raise RuntimeError("Native denoise_step audit contract changed")
        if not active["flow_times"] and not torch.equal(value, active["noise"]):
            raise RuntimeError("First denoising state differs from the native pure noise draw")
        active["flow_times"].append(float(time.detach().float().cpu().item()))
        return original_denoise(*args, **kwargs)

    def predict(batch, **kwargs):
        nonlocal active
        if active is not None or kwargs:
            raise RuntimeError("Closed-loop chunks must use the unmodified native select_action sampler")
        start_step = len(recorded["normalized_actions"])
        if start_step != len(recorded["chunks"]) * EXECUTED_STEPS:
            raise RuntimeError("Policy replanning cadence is no longer ten executed actions")
        if any(not (key.startswith("observation.") or key == "task") for key in batch):
            raise ValueError("Deployment policy inputs may contain observations and task only")
        active = {"flow_times": []}
        observation_hash = observation_sha256(batch)
        try:
            result = original_predict(batch)
            steps = policy.config.num_inference_steps
            expected = [1.0 - index / steps for index in range(steps)]
            observed = active["flow_times"]
            if (len(observed) != steps or any(not math.isclose(a, b, abs_tol=1e-6)
                                             for a, b in zip(observed, expected, strict=True))):
                raise RuntimeError("Native denoising count or schedule changed")
            if tuple(result.shape) != (1, HORIZON, ACTION_DIM) or not torch.isfinite(result).all():
                raise ValueError("Expected a finite native [1,50,7] action chunk")
            if observation_sha256(batch) != observation_hash:
                raise RuntimeError("Policy sampling modified its observation inputs")
            recorded["chunks"].append({
                "chunk_index": len(recorded["chunks"]), "start_step": start_step,
                "initial_noise_sha256": active["initial_noise_sha256"],
                "observation_sha256": observation_hash,
                "denoising_calls": steps, "flow_times": list(observed),
                "starts_from_native_pure_noise": True,
                "actions": result[0].detach().float().cpu().tolist(),
            })
            return result
        finally:
            active = None

    def select(batch):
        result = original_select(batch)
        if tuple(result.shape) != (1, ACTION_DIM) or not torch.isfinite(result).all():
            raise ValueError("Expected one finite normalized seven-dimensional action")
        value = result[0].detach().float().cpu().tolist()
        index = len(recorded["normalized_actions"])
        chunk_index, chunk_step = divmod(index, EXECUTED_STEPS)
        if len(recorded["chunks"]) != chunk_index + 1:
            raise RuntimeError("Policy action queue and native chunk audit disagree")
        if not np.array_equal(value, recorded["chunks"][chunk_index]["actions"][chunk_step]):
            raise RuntimeError("Selected action differs from the audited native chunk prefix")
        recorded["normalized_actions"].append(value)
        return result

    with _replace_method(policy.model, "sample_noise", sampled_noise), \
            _replace_method(policy.model, "denoise_step", denoise), \
            _replace_method(policy, "predict_action_chunk", predict), \
            _replace_method(policy, "select_action", select), \
            time_counterfactual(encoder, intervention):
        yield recorded


def _validate_state(state, name):
    if not isinstance(state, dict):
        raise ValueError(f"{name}: missing actual simulator state")
    for key, shape in (("eef_position", (3,)), ("eef_quaternion", (4,)),
                       ("eef_rotation_matrix", (3, 3)), ("gripper_qpos", (2,))):
        _finite_array(state.get(key), shape, f"{name}.{key}")
    if "gripper_qvel" in state:
        _finite_array(state["gripper_qvel"], (2,), f"{name}.gripper_qvel")
    if not re.fullmatch("[0-9a-f]{64}", str(state.get("simulator_state_sha256", ""))):
        raise ValueError(f"{name}: missing simulator state fingerprint")
    objects = state.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ValueError(f"{name}: missing actual object poses")
    for key, obj in objects.items():
        _finite_array(obj.get("position"), (3,), f"{name}.{key}.position")
        _finite_array(obj.get("quaternion"), (4,), f"{name}.{key}.quaternion")
        if obj.get("quaternion_order") != "wxyz":
            raise ValueError("Object poses must retain the documented MuJoCo quaternion order")


def validate_trace(trace, config, init_state_id, success=None):
    """Fail closed on incomplete or mispaired trajectory evidence."""
    import numpy as np
    from lerobot.scripts.pi05_task7_object_shift import stable_seed

    validate_config(config)
    if type(init_state_id) is not int or not 0 <= init_state_id < EPISODES:
        raise ValueError("Invalid init-state slot")
    expected = {
        "protocol": TRACE_PROTOCOL, "status": "completed",
        "intervention": config["intervention"], "checkpoint": config["checkpoint"],
        "scene_seed": config["scene_seed"], "policy_seed": config["policy_seed"],
        "init_state_id": init_state_id, "rollout_seed": config["init_seed"] + init_state_id,
        "executed_steps": EXECUTED_STEPS, "horizon": HORIZON,
        "configuration": config,
        "policy_rng_seed": stable_seed("policy-v2", config["policy_seed"], "libero_10", 7,
                                        init_state_id) % (2**31),
    }
    if not isinstance(trace, dict) or any(trace.get(key) != value for key, value in expected.items()):
        raise ValueError("Trace identity/configuration differs from the frozen intervention cell")
    if type(trace.get("success")) is not bool or (success is not None and trace["success"] != success):
        raise ValueError("Trace success differs from the official paired outcome")
    if type(trace.get("policy_rng_seed")) is not int:
        raise ValueError("Missing native policy RNG seed")
    _validate_state(trace.get("initial_state"), "initial_state")
    steps, chunks = trace.get("steps"), trace.get("chunks")
    if not isinstance(steps, list) or not steps or not isinstance(chunks, list):
        raise ValueError("Trace is missing executed steps or sampled chunks")
    if len(chunks) != math.ceil(len(steps) / EXECUTED_STEPS):
        raise ValueError("Trace chunk count differs from the ten-action execution cadence")
    denoising_count = trace.get("num_inference_steps")
    if type(denoising_count) is not int or denoising_count < 1:
        raise ValueError("Trace is missing the checkpoint denoising count")
    for index, chunk in enumerate(chunks):
        if (chunk.get("chunk_index") != index or chunk.get("start_step") != index * EXECUTED_STEPS
                or chunk.get("denoising_calls") != denoising_count
                or chunk.get("starts_from_native_pure_noise") is not True):
            raise ValueError("Invalid native chunk sampling audit")
        _finite_array(chunk.get("actions"), (HORIZON, ACTION_DIM), "chunk actions")
        times = _finite_array(chunk.get("flow_times"), (denoising_count,), "flow schedule")
        if not np.allclose(times, 1 - np.arange(denoising_count) / denoising_count, rtol=0, atol=1e-6):
            raise ValueError("Incorrect native denoising schedule")
        for key in ("initial_noise_sha256", "observation_sha256"):
            if not re.fullmatch("[0-9a-f]{64}", str(chunk.get(key, ""))):
                raise ValueError(f"Missing {key} in chunk audit")
    for index, step in enumerate(steps):
        chunk_index, chunk_step = divmod(index, EXECUTED_STEPS)
        if (step.get("step") != index or step.get("chunk_index") != chunk_index
                or step.get("chunk_step") != chunk_step):
            raise ValueError("Executed trajectory step/chunk indexes are inconsistent")
        _finite_array(step.get("executed_action"), (ACTION_DIM,), "executed action")
        normalized = _finite_array(step.get("normalized_action"), (ACTION_DIM,), "normalized action")
        if not np.array_equal(normalized, chunks[chunk_index]["actions"][chunk_step]):
            raise ValueError("Executed normalized action does not match the corresponding native chunk")
        _validate_state(step.get("state_after"), f"step{index}.state_after")
        if any(type(step.get(key)) is not bool for key in ("success", "terminated", "truncated")):
            raise ValueError("Trace outcome flags must be booleans")
        if index < len(steps) - 1 and (step["terminated"] or step["truncated"]):
            raise ValueError("Trace contains actions after episode termination")
    if not (steps[-1]["terminated"] or steps[-1]["truncated"]):
        raise ValueError("Trace did not reach official termination or the native time limit")
    if any(step["success"] for step in steps) != trace["success"]:
        raise ValueError("Trace task-success flags disagree with the official outcome")
    return trace


def require_saved_traces(config, successes=None):
    validate_config(config)
    if successes is not None and (not isinstance(successes, list) or len(successes) != EPISODES
                                  or any(type(value) is not bool for value in successes)):
        raise ValueError("Trace validation expects ten official boolean outcomes")
    root = Path(config["trace_directory"])
    paths = [root / f"episode-{index:03d}.json" for index in range(EPISODES)]
    if set(root.glob("episode-*.json")) != set(paths):
        raise ValueError(f"Expected exactly ten completed trajectory files in {root}")
    fingerprints = source_fingerprints()
    checkpoint_config = json.loads((Path(config["checkpoint"]) / "config.json").read_text(encoding="utf-8"))
    from lerobot.scripts.pi05_task7_full_sampling import validate_sampling_config

    validate_sampling_config(checkpoint_config)
    for index, path in enumerate(paths):
        trace = validate_trace(json.loads(path.read_text(encoding="utf-8")), config, index,
                               successes[index] if successes is not None else None)
        if trace.get("source_fingerprints") != fingerprints:
            raise ValueError(f"Trajectory was captured by different instrumentation or policy code: {path}")
        if trace["num_inference_steps"] != checkpoint_config["num_inference_steps"]:
            raise ValueError(f"Trajectory denoising count differs from the saved checkpoint: {path}")
    return paths


def _join_trace(result, physical, recorded, config, manifest, policy, slot, dual_config):
    import numpy as np
    from lerobot.scripts import pi05_task7_dualseed_shift as dual

    actions = result["action"].detach().cpu().numpy()
    done = result["done"].detach().cpu().numpy()
    flags = result["success"].detach().cpu().numpy()
    count = len(physical["steps"])
    if (actions.shape != (1, count, ACTION_DIM) or done.shape != (1, count)
            or flags.shape != (1, count) or not bool(done[0, -1])
            or bool(done[0, :-1].any()) or len(recorded["normalized_actions"]) != count):
        raise RuntimeError("Native rollout result and captured execution have different lengths/termination")
    if physical["rollout_seed"] != config["init_seed"] + slot:
        raise RuntimeError("Simulator trace came from a different seeded episode")
    steps = physical["steps"]
    for index, step in enumerate(steps):
        if not np.array_equal(actions[0, index], step["executed_action"]):
            raise RuntimeError("Simulator command differs from the evaluator's executed action")
        if bool(flags[0, index]) != step["success"]:
            raise RuntimeError("Simulator task-success flag differs from the evaluator's official outcome")
        step.update(chunk_index=index // EXECUTED_STEPS, chunk_step=index % EXECUTED_STEPS,
                    normalized_action=recorded["normalized_actions"][index])
    reason = "environment_termination"
    if not steps[-1]["terminated"] and not steps[-1]["truncated"]:
        # The existing native evaluator enforces its _max_episode_steps outside
        # the underlying environment, so mark that final truncation explicitly.
        steps[-1]["truncated"] = True
        reason = "native_evaluator_time_limit"
    trace = {
        "protocol": TRACE_PROTOCOL, "status": "completed",
        "configuration": config, "configuration_sha256": manifest["config_sha256"],
        "source_fingerprints": {key: manifest[key] for key in source_fingerprints()},
        "intervention": config["intervention"], "checkpoint": config["checkpoint"],
        "scene_seed": config["scene_seed"], "policy_seed": config["policy_seed"],
        "init_state_id": slot, "rollout_seed": config["init_seed"] + slot,
        "policy_rng_seed": dual.policy_rng_seed(dual_config, slot),
        "executed_steps": EXECUTED_STEPS, "horizon": HORIZON,
        "num_inference_steps": policy.config.num_inference_steps,
        "success": bool(flags.any()), "termination_reason": reason,
        "initial_state": physical["initial_state"],
        "steps": steps, "chunks": recorded["chunks"],
        "state_conventions": {
            "eef_position": "actual simulator world-frame position, meters",
            "eef_quaternion": "raw robot0_eef_quat; use eef_rotation_matrix for convention-independent rotations",
            "object_quaternion": "MuJoCo free-joint wxyz",
            "executed_action": "seven native commands after policy and environment postprocessing",
            "normalized_action": "seven model outputs before policy postprocessing",
            "gripper_sign": "uninterpreted command; actual finger joint positions are recorded separately",
            "contacts": "raw deduplicated geom pairs; no inferred grasp/holding predicate",
            "simulator_hash": "flattened get_sim_state; controller telemetry recorded separately, not a restorable snapshot",
        },
    }
    return validate_trace(trace, config, slot)


def install(evaluator, path):
    """Install after the dual-seed helper, before the launcher's resume wrapper."""
    from lerobot.scripts import pi05_task7_dualseed_shift as dual

    path = Path(path).resolve(strict=True)
    config = read_config(path)
    dual_path = os.environ.get("LIBERO_TASK7_DUALSEED_SHIFT_CONFIG")
    if not dual_path:
        raise ValueError("Time interventions require the existing dual-seed task7 protocol")
    dual_config = dual.read_config(dual_path)
    for source, target in (("scene_seed", "scene_seed"), ("policy_seed", "policy_seed"),
                           ("init_seed", "eval_seed")):
        if config[source] != dual_config[target]:
            raise ValueError("Time intervention and physical/RNG cell configurations disagree")
    manifest = manifest_for_config(path)
    old_rollout, old_one = evaluator.rollout, evaluator.run_one
    rollout_signature, one_signature = inspect.signature(old_rollout), inspect.signature(old_one)

    @functools.wraps(old_rollout)
    def rollout(*args, **kwargs):
        bound = rollout_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        seeds, env, policy = (bound.arguments[name] for name in ("seeds", "env", "policy"))
        if (not isinstance(seeds, (list, tuple)) or len(seeds) != 1
                or env.num_envs != 1):
            raise ValueError("Time intervention rollout requires one explicitly seeded environment")
        slot = dual.slot(int(seeds[0]), dual_config)
        target = Path(config["trace_directory"]) / f"episode-{slot:03d}.json"
        # Never leave a previous completed trace behind for a newly failed attempt.
        if target.exists():
            validate_trace(json.loads(target.read_text(encoding="utf-8")), config, slot)
            target.unlink()
        armed = list(env.call("task7_start_trace", int(seeds[0])))
        if armed != [{"armed": True, "rollout_seed": int(seeds[0])}]:
            raise RuntimeError("Could not arm the actual LIBERO worker for trajectory capture")
        with audit_policy(policy, config["intervention"]) as recorded:
            result = old_rollout(*bound.args, **bound.kwargs)
        physical = list(env.call("task7_get_trace"))
        if len(physical) != 1:
            raise RuntimeError("Expected exactly one native environment execution trace")
        trace = _join_trace(result, physical[0], recorded, config, manifest, policy, slot, dual_config)
        _atomic_json(target, trace)
        return result

    @functools.wraps(old_one)
    def run_one(*args, **kwargs):
        bound = one_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if ((values["task_group"], values["task_id"]) != ("libero_10", 7)
                or values["n_episodes"] != EPISODES or values["start_seed"] != config["init_seed"]):
            raise ValueError("Time intervention must run exactly the frozen task7 cell")
        result = old_one(*args, **kwargs)
        require_saved_traces(config, successes=result[2]["successes"])
        return result

    evaluator.rollout, evaluator.run_one = rollout, run_one
    return {"task7_time_intervention": manifest}
