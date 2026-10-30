"""Paired one-action perturbations in standard LIBERO, never on real hardware.

Reconstruct a state by resetting and replaying the SAME action prefix. This also
reconstructs controller/observable caches; copying qpos/qvel alone is insufficient.
Anchors must be action-chunk boundaries of this stateless, single-observation
PI05. Each clean/perturbed/zero-control continuation resets the policy queue and
uses paired inference seeds. Failed replay or zero controls invalidate the pair.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path

import numpy as np

from .core import SCHEMA, arrays_hash, digest, json_hash, load_scores, read_json, write_json
from .analysis import paired_summary


def feasible_delta(action, magnitude, dimensions, seed):
    """Exact L2 norm in normalized controller coordinates, no gripper flip/clipping."""
    action = np.asarray(action, dtype=np.float64)
    dims = np.asarray(dimensions, dtype=int)
    if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-8):
        raise ValueError("Require a finite bounded 7-D executed action")
    if not np.isfinite(magnitude) or magnitude < 0 or magnitude > 1:
        raise ValueError("Require 0 <= magnitude <= 1 normalized command units")
    if not len(dims) or len(set(dims.tolist())) != len(dims) or np.any((dims < 0) | (dims > 5)):
        raise ValueError("Choose distinct continuous action coordinates 0..5, never gripper=6")
    result = np.zeros(7)
    if magnitude == 0:
        return result
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        direction = rng.normal(size=len(dims))
        norm = np.linalg.norm(direction)
        if norm <= 1e-15:
            continue
        result[dims] = magnitude * direction / norm
        candidate = action + result
        if np.all(candidate >= -1) and np.all(candidate <= 1):
            return result
    raise ValueError("Could not draw a bounded exact-norm perturbation; reduce --delta")


def paired_seed(seed, *parts):
    return int(json_hash([seed, *parts])[:8], 16) % (2**31)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nested_hash(value):
    arrays = {}
    def visit(x, prefix):
        if isinstance(x, dict):
            for key, child in sorted(x.items()):
                visit(child, prefix + "/" + key)
        elif isinstance(x, np.ndarray):
            arrays[prefix] = x
        else:
            raise TypeError(f"Unexpected observation component at {prefix}: {type(x)}")
    visit(value, "obs")
    return arrays_hash(arrays)


def trajectories_match(a, b, *, state_atol=1e-8, action_atol=1e-5):
    if a["success"] != b["success"] or a["terminal"] != b["terminal"]:
        return False
    return all(a[key].shape == b[key].shape and np.allclose(a[key], b[key], rtol=0, atol=atol)
               for key, atol in (("actions", action_atol), ("states", state_atol)))


class LiberoRuntime:
    """Use the repository's saved processors and observation conventions."""
    def __init__(self, checkpoint, tokenizer, *, device="cuda", image_size=360, action_steps=10,
                 score_manifest=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoConfig
        from lerobot.envs import make_env_pre_post_processors, preprocess_observation
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.scripts.libero_vlm_only import validate_processors
        checkpoint = Path(checkpoint).resolve()
        raw = read_json(checkpoint / "config.json")
        if raw.get("type") != "pi05" or raw.get("n_obs_steps", 1) != 1 or (
                raw.get("rtc_config") or {}).get("enabled", False):
            raise ValueError("Diagnostic supports only stateless n_obs_steps=1, non-RTC PI05")
        validate_processors(checkpoint, raw)
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device, cfg.compile_model, cfg.gradient_checkpointing = device, False, False
        cfg.tokenizer_name, cfg.n_action_steps = str(tokenizer), action_steps
        if not 1 <= action_steps <= cfg.chunk_size:
            raise ValueError("Invalid execution horizon")
        self.policy = PI05Policy.from_pretrained(checkpoint, config=cfg, local_files_only=True).eval()
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device},
                                    "tokenizer_processor": {"tokenizer_name": str(tokenizer)}})
        env_cfg = LiberoConfig(observation_height=image_size, observation_width=image_size,
                              control_mode="relative")
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg, cfg)
        self.preprocess_observation = preprocess_observation
        self.image_size, self.action_steps, self.device = image_size, action_steps, torch.device(device)
        self.score_manifest = score_manifest
        if score_manifest:
            tower = self.policy.model.paligemma_with_expert.paligemma.model.vision_tower
            h = hashlib.sha256()
            for name, tensor in sorted(tower.state_dict().items()):
                h.update(name.encode())
                h.update(str(tuple(tensor.shape)).encode())
                h.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
            self.vision_hash = h.hexdigest()
            if score_manifest.get("vision_weights_sha256") != self.vision_hash:
                raise ValueError("Scoring cache and checkpoint have different frozen vision weights")

    def reset_policy(self):
        self.policy.reset()
        for processor in (self.pre, self.post, self.env_pre, self.env_post):
            if hasattr(processor, "reset"):
                processor.reset()
        if len(self.policy._action_queue):
            raise RuntimeError("Policy queue did not reset")

    def open(self, suite_name, task_id, episode, reset_seed):
        from lerobot.envs.libero import LiberoEnv, _get_suite
        from libero.libero.envs.bddl_utils import get_problem_info
        seed_everything(reset_seed)
        suite = _get_suite(suite_name)
        if not 0 <= task_id < len(suite.tasks):
            raise ValueError("Task ID outside suite")
        env = LiberoEnv(suite, task_id, suite_name, obs_type="pixels_agent_pos", episode_index=episode,
                        n_envs=1, observation_height=self.image_size, observation_width=self.image_size,
                        control_mode="relative", is_libero_plus=False)
        if not 0 <= episode < len(env._init_states):
            raise ValueError("Episode index exceeds available official initial states; no modulo reuse")
        instruction = get_problem_info(str(env._task_bddl_file))["language_instruction"]
        env.task_description = instruction
        try:
            obs, _ = env.reset(seed=reset_seed)
        except BaseException:
            env.close()
            raise
        self.reset_policy()
        return env, obs

    @staticmethod
    def state(env):
        return np.asarray(env._env.env.sim.get_state().flatten(), dtype=np.float64).copy()

    @staticmethod
    def step(env, action):
        # Deliberately bypass LiberoEnv.step(), which auto-resets after termination.
        raw, _, done, _ = env._env.step(np.asarray(action, dtype=np.float64))
        success = bool(env._env.check_success())
        return env._format_raw_obs(raw), bool(done or success), success

    def batch(self, obs, instruction):
        def stack(x):
            return {k: stack(v) for k, v in x.items()} if isinstance(x, dict) else np.expand_dims(x, 0)
        data = self.preprocess_observation(stack(obs))
        data["task"] = [instruction]
        return self.pre(self.env_pre(data))

    def action(self, obs, instruction, *, features=False):
        import torch
        import torch.nn.functional as F
        batch = self.batch(obs, instruction)
        visual = None
        with torch.inference_mode():
            if features:
                images, masks = self.policy._preprocess_images(batch)
                if len(images) != 2 or not all(bool(mask.all()) for mask in masks):
                    raise ValueError("Scoring requires exactly two real camera inputs")
                tower = self.policy.model.paligemma_with_expert.paligemma.model.vision_tower
                visual = np.stack([F.normalize(tower(pixel_values=image.float()).last_hidden_state.mean(1).float(), dim=-1)
                                   .cpu().numpy()[0] for image in images])
            action = self.policy.select_action(batch)
            action = self.env_post({"action": self.post(action)})["action"].detach().float().cpu().numpy()[0]
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Invalid model action")
        # Explicitly the same controller-range clipping for every branch.
        return np.clip(action, -1, 1).astype(np.float64), visual


def reference_rollout(runtime, suite, task, episode, reset_seed, anchors):
    """Predeclared fixed times, irrespective of score, policy success or risk."""
    env, obs = runtime.open(suite, task, episode, reset_seed)
    actions, states, features, records = [], [], [], {}
    success = terminal = False
    try:
        horizon = int(env._max_episode_steps)
        for step in range(horizon):
            state = runtime.state(env)
            if step in anchors:
                if len(runtime.policy._action_queue):
                    raise ValueError("Anchor is not an empty action-chunk boundary")
                records[step] = {"state": state.copy(), "observation_hash": nested_hash(obs),
                                 "pixels": {k: v.copy() for k, v in obs["pixels"].items()},
                                 "instruction": env.task_description}
            action, feature = runtime.action(obs, env.task_description, features=runtime.score_manifest is not None)
            actions.append(action)
            states.append(state)
            if feature is not None:
                features.append(feature)
            obs, terminal, success = runtime.step(env, action)
            if terminal:
                break
        result = {"actions": np.asarray(actions), "states": np.asarray(states), "anchors": records,
                  "success": bool(success), "terminal": bool(terminal), "horizon": horizon,
                  "instruction": env.task_description,
                  "bddl_sha256": digest(env._task_bddl_file)}
        if features:
            features = np.asarray(features)
            visual_delta = np.zeros((len(actions), 2), dtype=np.float32)
            visual_delta[1:] = np.linalg.norm(np.diff(features, axis=0), axis=-1)
            from lerobot.utils.frameskip_curriculum import ScoreConfig, score_episode
            recipe = runtime.score_manifest
            scored = score_episode(result["actions"], visual_delta, recipe["action_low"], recipe["action_high"],
                                   recipe["gripper_threshold"], episode, ScoreConfig(**recipe["config"]))
            result["importance"] = scored["score"]
            result["visual_delta"] = visual_delta
        return result
    finally:
        env.close()


def continuation(runtime, reference, suite, task, episode, reset_seed, anchor, inference_seed,
                 delta_size, dimensions, delta_seed, *, state_atol=1e-8):
    env, obs = runtime.open(suite, task, episode, reset_seed)
    actions, states = [], []
    success = terminal = False
    try:
        for action in reference["actions"][:anchor]:
            obs, terminal, success = runtime.step(env, action)
            if terminal:
                raise RuntimeError("Prefix terminated before predeclared anchor")
        state = runtime.state(env)
        expected = reference["anchors"][anchor]
        replay_matches = np.allclose(state, expected["state"], rtol=0, atol=state_atol) and (
            nested_hash(obs) == expected["observation_hash"])
        if not replay_matches:
            raise RuntimeError("Replayed simulator state/observations differ; paired comparison invalid")
        runtime.reset_policy()
        seed_everything(inference_seed)
        perturbation = None
        for step in range(anchor, reference["horizon"]):
            states.append(runtime.state(env))
            action, _ = runtime.action(obs, env.task_description)
            if step == anchor:
                perturbation = feasible_delta(action, delta_size, dimensions, delta_seed)
                action = action + perturbation
                if not np.isclose(np.linalg.norm(perturbation), delta_size, rtol=0, atol=1e-10):
                    raise RuntimeError("Realized perturbation norm changed")
            actions.append(action.copy())
            obs, terminal, success = runtime.step(env, action)
            if terminal:
                break
        return {"actions": np.asarray(actions), "states": np.asarray(states), "success": bool(success),
                "terminal": bool(terminal), "replay_matches": bool(replay_matches),
                "perturbation": perturbation.tolist()}
    finally:
        env.close()


def run(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    if args.episodes < 1 or args.trials < 1 or args.seed < 0 or args.state_atol < 0 or args.action_atol < 0:
        raise ValueError("Invalid counts, seed or tolerance")
    anchors = sorted(args.anchor_steps)
    if any(i < 0 or i % args.action_steps for i in anchors):
        raise ValueError("Anchor times must be nonnegative action-chunk boundaries")
    feasible_delta(np.zeros(7), args.delta, args.dims, args.seed)
    tasks = []
    for entry in args.tasks.split(","):
        suite, task = entry.split(":")
        if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
            raise ValueError("This experiment uses standard LIBERO only, not PRO/PLUS or a physical robot")
        tasks.append((suite, int(task)))
    if len(set(tasks)) != len(tasks):
        raise ValueError("Duplicate task")
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if not args.tokenizer.is_dir():
        raise FileNotFoundError(args.tokenizer)
    _, score_manifest = load_scores(args.scores) if args.scores else (None, None)
    model_files = {str(p.relative_to(checkpoint)): digest(p) for p in sorted(checkpoint.glob("*.json"))}
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in read_json(checkpoint / filename)["steps"]:
            state_file = step.get("state_file")
            if state_file:
                if Path(state_file).is_absolute() or ".." in Path(state_file).parts:
                    raise ValueError("Processor state must stay inside its checkpoint")
                model_files[state_file] = digest(checkpoint / state_file)
    from lerobot.envs.libero import _get_suite
    from libero.libero import get_libero_path
    from lerobot.utils.pdet import runtime_info
    task_resources = []
    for suite_name, task_id in tasks:
        suite = _get_suite(suite_name)
        if not 0 <= task_id < len(suite.tasks):
            raise ValueError("Task ID outside suite")
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init = Path(get_libero_path("init_states")) / task.problem_folder / Path(task.init_states_file).name
        task_resources.append({"suite": suite_name, "task_id": task_id, "task_name": task.name,
                               "bddl_sha256": digest(bddl), "init_sha256": digest(init)})
    asset_root = Path(get_libero_path("assets"))
    assets = [[str(p.relative_to(asset_root)), p.stat().st_size, p.stat().st_mtime_ns]
              for p in sorted(asset_root.rglob("*")) if p.is_file()]
    if not assets:
        raise ValueError("No LIBERO assets found")
    weight = checkpoint / "model.safetensors"
    model_files["weights_inventory"] = [str(weight.resolve()), weight.stat().st_size, weight.stat().st_mtime_ns]
    versions = {}
    for name in ("torch", "transformers", "mujoco", "robosuite", "gymnasium", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    from .__main__ import package_hashes
    protocol = {"protocol": SCHEMA, "kind": "libero-prefix-replay-single-action-perturbation",
                "checkpoint": str(checkpoint), "model_files": model_files,
                "tasks": tasks, "episodes": args.episodes, "trials": args.trials,
                "seed": args.seed, "anchors": anchors, "delta": args.delta, "dims": args.dims,
                "action_steps": args.action_steps, "image_size": args.image_size,
                "state_atol": args.state_atol, "action_atol": args.action_atol,
                "score_recipe": score_manifest, "versions": versions, "code_sha256": package_hashes(),
                "native_runtime": runtime_info(), "task_resources": task_resources,
                "asset_inventory_sha256": json_hash(assets),
                "action_protocol": "clip all baseline actions to [-1,1]; bounded exact-norm additive delta once, gripper unchanged",
                "importance_scope": "same scoring formula on baseline POLICY rollouts, NOT demonstration scores",
                "scope": "Sensitivity conditional on selected baseline policy, state and perturbation scale"}
    # Normalize tuples to JSON lists for resume comparisons.
    protocol = json.loads(json.dumps(protocol))
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "protocol.json"
        if marker.exists():
            if read_json(marker) != protocol:
                raise ValueError("Protocol/model/code changed; choose a new output directory")
        else:
            if list(output.glob("pair-*.json")):
                raise ValueError("Unprovenanced existing results")
            write_json(marker, protocol, exclusive=True)
        runtime = LiberoRuntime(checkpoint, args.tokenizer, device=args.device, image_size=args.image_size,
                                action_steps=args.action_steps, score_manifest=score_manifest)
        records, feature_rows, coverage = [], [], []
        for suite, task in tasks:
            for episode in range(args.episodes):
                reset_seed = paired_seed(args.seed, suite, task, episode, "reset")
                ref = reference_rollout(runtime, suite, task, episode, reset_seed, anchors)
                reference_identity = {"suite": suite, "task_id": task, "episode": episode,
                    "trace_sha256": arrays_hash({"actions": ref["actions"], "states": ref["states"]}),
                    "anchors": {str(k): v["observation_hash"] for k, v in ref["anchors"].items()},
                    "bddl_sha256": ref["bddl_sha256"], "success": ref["success"], "horizon": ref["horizon"]}
                reference_path = output / f"reference-{suite}-{task}-{episode}.json"
                if reference_path.exists():
                    if read_json(reference_path) != reference_identity:
                        raise ValueError("Reference rollout changed on resume; refusing to mix state populations")
                else:
                    write_json(reference_path, reference_identity, exclusive=True)
                coverage.append({"suite": suite, "task_id": task, "episode": episode,
                                 "reference_success": ref["success"], "reference_steps": len(ref["actions"]),
                                 "requested_anchors": anchors, "reached_anchors": sorted(ref["anchors"]),
                                 "unreached_anchors": sorted(set(anchors) - set(ref["anchors"]))})
                for anchor in sorted(ref["anchors"]):
                    uid = json_hash([suite, task, episode, reset_seed, anchor,
                                     arrays_hash({"prefix": ref["actions"][:anchor]}), ref["bddl_sha256"]])
                    state_dir = output / "states" / uid
                    state_dir.mkdir(parents=True, exist_ok=True)
                    from PIL import Image
                    pictures = {}
                    for camera, image in ref["anchors"][anchor]["pixels"].items():
                        file = state_dir / f"{camera}.png"
                        # Raw camera orientation retained; metadata explicitly states it.
                        if not file.exists():
                            Image.fromarray(image).save(file)
                        else:
                            with Image.open(file) as saved_image:
                                if not np.array_equal(np.asarray(saved_image), image):
                                    raise ValueError("Saved state image changed; do not edit diagnostic source images")
                        pictures[camera] = str(file)
                    state_info = {"state_uid": uid, "suite": suite, "task_id": task, "episode": episode,
                                  "anchor_step": anchor, "instruction": ref["instruction"], "images": pictures,
                                  "image_orientation": "raw LIBERO camera arrays, before env preprocessing",
                                  "observation_hash": ref["anchors"][anchor]["observation_hash"],
                                  "state_sha256": arrays_hash({"state": ref["anchors"][anchor]["state"]})}
                    if (state_dir / "identity.json").exists():
                        if read_json(state_dir / "identity.json") != state_info:
                            raise ValueError("State changed on resume")
                    else:
                        write_json(state_dir / "identity.json", state_info, exclusive=True)
                        np.savez_compressed(state_dir / "replay.npz", prefix_actions=ref["actions"][:anchor],
                                            state=ref["anchors"][anchor]["state"])
                    feature_rows.append({"state_uid": uid, "semantic_difficulty": "", "action_error": "",
                                         "importance": float(ref["importance"][anchor]) if "importance" in ref else ""})
                    for trial in range(args.trials):
                        path = output / f"pair-{uid}-{trial:03d}.json"
                        if path.exists():
                            row = read_json(path)
                            if (row["state_uid"], row["trial"], row["protocol_id"]) != (uid, trial, json_hash(protocol)):
                                raise ValueError("Wrong saved pair identity")
                            records.append(row)
                            continue
                        inference_seed = paired_seed(args.seed, uid, trial, "inference")
                        delta_seed = paired_seed(args.seed, uid, trial, "delta")
                        common = (runtime, ref, suite, task, episode, reset_seed, anchor, inference_seed)
                        clean = continuation(*common, 0., args.dims, delta_seed, state_atol=args.state_atol)
                        zero = continuation(*common, 0., args.dims, delta_seed, state_atol=args.state_atol)
                        matched = trajectories_match(clean, zero, state_atol=args.state_atol, action_atol=args.action_atol)
                        if not matched:
                            raise RuntimeError("Zero-perturbation replay check failed; no risk pair written")
                        pert = continuation(*common, args.delta, args.dims, delta_seed, state_atol=args.state_atol)
                        if not np.allclose(pert["actions"][0] - clean["actions"][0],
                                           pert["perturbation"], rtol=0, atol=args.action_atol):
                            raise RuntimeError("First paired actions differ beyond the requested perturbation")
                        row = {"protocol_id": json_hash(protocol), "state_uid": uid, "trial": trial,
                               "suite": suite, "task_id": task, "episode": episode, "anchor_step": anchor,
                               "clean_success": clean["success"], "zero_success": zero["success"],
                               "perturbed_success": pert["success"], "replay_matches": True, "zero_matches": True,
                               "inference_seed": inference_seed, "delta_seed": delta_seed, "delta": pert["perturbation"],
                               "clean_steps": len(clean["actions"]), "perturbed_steps": len(pert["actions"])}
                        write_json(path, row, exclusive=True)
                        records.append(row)
                        print(f"{suite}/{task} ep={episode} anchor={anchor} trial={trial}: "
                              f"clean={int(clean['success'])} perturbed={int(pert['success'])}", flush=True)
        summary = paired_summary(records)
        write_json(output / "risk_summary.json", summary)
        write_json(output / "coverage.json", coverage)
        # Never overwrite hand-filled semantic/action measurements on a later resume.
        template = output / "features_template.csv"
        if not template.exists():
            with template.open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["state_uid", "semantic_difficulty", "action_error", "importance"])
                writer.writeheader()
                writer.writerows(feature_rows)
        return {"states": len(summary), "paired_trials": len(records), "output": str(output),
                "all_clean_failed_states": sum(r["all_clean_failed"] for r in summary)}


def main():
    from .__main__ import _csv_ints
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("/home/wyn/models/paligemma-3b-pt-224"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="libero_spatial:0,libero_goal:0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--anchor-steps", type=_csv_ints, default=[20, 60, 100, 140])
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--dims", type=_csv_ints, default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument("--state-atol", type=float, default=1e-8)
    parser.add_argument("--action-atol", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scores", type=Path, help="Reuse exact cached score recipe/frozen encoder; scores POLICY trajectories, not demonstrations")
    args = parser.parse_args()
    if args.action_steps < 1 or args.image_size < 1:
        parser.error("Execution horizon and image size must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
