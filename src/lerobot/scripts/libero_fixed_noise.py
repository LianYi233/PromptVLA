"""Deterministic evaluation-only Gaussian image perturbation for standard LIBERO-long.

Noise is derived statelessly from (noise seed, suite, task, episode seed, step,
camera) and is independent of model, GPU, task completion order and global RNG.
It is applied after the LIBERO camera flip and before model preprocessing.
No simulator state, language, actions or original checkpoint is modified.
"""

import functools
import hashlib
import inspect
import json
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "promptvla-libero-fixed-image-noise-v1"
NOISE_SCOPES = ("task8", "all")
REQUIRED_KEYS = frozenset(
    {"protocol", "kind", "std", "seed", "scope", "eval_seed", "suite", "target_task_id"}
)


def validate_config(config):
    if not isinstance(config, dict) or set(config) != REQUIRED_KEYS:
        raise ValueError("Fixed image-noise configuration has missing or unknown keys")
    if config["protocol"] != PROTOCOL or config["kind"] != "gaussian":
        raise ValueError("Unsupported fixed-image-noise protocol or corruption kind")
    std = config["std"]
    if type(std) not in (float, int) or not np.isfinite(std) or not 0 <= std <= 1:
        raise ValueError("Gaussian pixel std must be in [0,1] on RGB images normalized to [0,1]")
    if config["scope"] not in NOISE_SCOPES or config["suite"] != "libero_10":
        raise ValueError("Fixed-noise experiment supports standard libero_10 only")
    if type(config["target_task_id"]) is not int or config["target_task_id"] != 8:
        raise ValueError("Fixed-noise task8 scope must target task index 8")
    for key in ("seed", "eval_seed"):
        if type(config[key]) is not int or not 0 <= config[key] < 2**32:
            raise ValueError(f"{key} must be a nonnegative 32-bit integer")
    return config


def read_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def stable_seed(*parts):
    """Use SHA-256, not Python's salted hash() or a global random generator."""
    serial = json.dumps([PROTOCOL, *parts], separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(serial.encode()).digest()[:8], "little")


def task_is_noisy(config, suite, task_id):
    if suite != config["suite"]:
        raise ValueError("Fixed Gaussian corruption must not be run on a different benchmark suite")
    return config["scope"] == "all" or task_id == config["target_task_id"]


class FixedGaussianImagePreprocessor:
    """Add per-image and per-control-step fixed Gaussian noise; leave states unchanged."""

    def __init__(self, upstream, config, suite, task_id, episode_seed):
        self.upstream = upstream
        self.config = validate_config(config)
        self.suite = suite
        self.task_id = int(task_id)
        self.episode_seed = int(episode_seed)
        self.step = 0

    def __call__(self, observation):
        output = self.upstream(observation)
        if not isinstance(output, dict):
            raise TypeError("LIBERO env preprocessor must return a dictionary")
        images = sorted(key for key in output if key.startswith("observation.images."))
        if not images:
            raise RuntimeError("No LIBERO camera tensors found; refusing a silently clean rollout")
        noisy = dict(output)
        for camera in images:
            tensor = output[camera]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise ValueError(f"Expected BCHW camera tensor for {camera}")
            if tensor.shape[0] != 1 or tensor.shape[1] != 3 or not tensor.is_floating_point():
                raise ValueError(f"Expected batch-one float RGB camera tensor for {camera}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite original camera pixels: {camera}")
            # Input comes from preprocess_observation: float32, BCHW, range [0,1].
            if torch.any(tensor < 0) or torch.any(tensor > 1):
                raise ValueError(f"Camera pixels must be normalized to [0,1]: {camera}")
            seed = stable_seed(
                "camera", self.config["seed"], self.suite, self.task_id,
                self.episode_seed, self.step, camera,
            )
            generator = np.random.Generator(np.random.PCG64(seed))
            noise = generator.standard_normal(tuple(tensor.shape), dtype=np.float32)
            noise = torch.from_numpy(noise).to(device=tensor.device, dtype=tensor.dtype)
            noisy[camera] = (tensor + self.config["std"] * noise).clamp(0.0, 1.0)
        self.step += 1
        return noisy


def install(evaluator, path):
    """Wrap standard eval's run_one/rollout before its resume/progress wrappers install.

    The official runner executes tasks serially with eval.batch_size=1.
    The task context tells rollout which suite/task generated this observation;
    the actual episode seed comes from eval_policy's rollout(seeds=[...]).
    """
    config_path = Path(path).resolve(strict=True)
    config = read_config(config_path)
    original_one = evaluator.run_one
    original_rollout = evaluator.rollout
    rollout_signature = inspect.signature(original_rollout)
    task_context = ContextVar("fixed_libero_noise_task", default=None)

    @functools.wraps(original_one)
    def run_one(task_group, task_id, env, **kwargs):
        token = task_context.set((task_group, int(task_id)))
        try:
            return original_one(task_group, task_id, env, **kwargs)
        finally:
            task_context.reset(token)

    @functools.wraps(original_rollout)
    def rollout(*args, **kwargs):
        task = task_context.get()
        if task is None:
            raise RuntimeError("Fixed-noise rollout must execute inside a LIBERO task")
        suite, task_id = task
        noisy_task = task_is_noisy(config, suite, task_id)
        bound = rollout_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        seeds = bound.arguments.get("seeds")
        if not isinstance(seeds, (list, tuple)) or len(seeds) != 1 or type(seeds[0]) is not int:
            raise ValueError("Fixed-noise protocol requires eval.batch_size=1 and explicit episode seed")
        episode_seed = seeds[0]
        if episode_seed < 0:
            raise ValueError("Negative episode seed is not supported")

        # The action flow starts from random noise. Reset the policy-side
        # stochastic stream per episode, independent of task completion order,
        # so skipping completed tasks does not alter the evaluation draws.
        policy_seed = stable_seed(
            "policy", config["eval_seed"], suite, task_id, episode_seed
        ) % (2**31)
        torch.manual_seed(policy_seed)

        if noisy_task and config["std"] > 0:
            bound.arguments["env_preprocessor"] = FixedGaussianImagePreprocessor(
                bound.arguments["env_preprocessor"], config, suite, task_id, episode_seed
            )
        return original_rollout(*bound.args, **bound.kwargs)

    evaluator.run_one = run_one
    evaluator.rollout = rollout
    return {
        "fixed_observation_noise": {
            "config": config,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "noise_generator": "SHA256-keyed NumPy PCG64, per suite/task/episode/frame/camera",
            "policy_rng": "SHA256-keyed torch.manual_seed, per suite/task/episode",
            "injection": "after LIBERO env preprocessing, before policy preprocessing",
        }
    }
