"""Recorded-action diagnostics, separate from the stochastic flow training objective.

No hardware imports. Physical-unit metrics require Piper absolute joint angles in radians
and the dataset's gripper coordinate. Torch is imported only by runtime helpers.
"""

import json
import math
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np

PIPER_NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]


def as_numpy(value):
    return value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def piper_names(features):
    result = {}
    for key in ("action", "observation.state"):
        feature = features[key]
        names = feature.get("names")
        if feature.get("shape") != [7] or not isinstance(names, list) or sorted(names) != sorted(PIPER_NAMES):
            raise ValueError(f"piper_eval requires seven explicitly named Piper coordinates in {key}")
        result[key] = names
    return result["action"], result["observation.state"]


def action_metrics(predicted, target, valid, hold, names, horizons, moving_threshold_deg=1.0):
    """Score valid time-aligned pairs only; never mix gripper units with joint angles."""
    predicted, target, hold = (np.asarray(x, dtype=np.float64) for x in (predicted, target, hold))
    valid = np.asarray(valid)
    if (
        predicted.ndim != 3
        or predicted.shape != target.shape
        or predicted.shape != hold.shape
        or predicted.shape[-1] != len(names)
        or valid.shape != predicted.shape[:2]
        or valid.dtype != np.bool_
    ):
        raise ValueError("Invalid action or padding-mask shape/dtype")
    if not all(np.isfinite(x[valid]).all() for x in (predicted, target, hold)):
        raise ValueError("Nonfinite predicted actions or valid labels")
    joints = [names.index(name) for name in PIPER_NAMES[:-1]]
    gripper = names.index("gripper.pos")
    metrics = {}
    for horizon in sorted(set(horizons)):
        if not 1 <= horizon <= target.shape[1]:
            raise ValueError("Evaluation horizon exceeds predicted chunk")
        mask = valid[:, :horizon]
        error = (predicted[:, :horizon] - target[:, :horizon])[mask]
        baseline = (hold[:, :horizon] - target[:, :horizon])[mask]
        if not len(error):
            raise ValueError("No valid action pairs")
        joint_error = np.rad2deg(error[:, joints])
        hold_joint = np.rad2deg(baseline[:, joints])
        prefix = f"h{horizon}/"
        scores = {
            "valid_pairs": int(len(error)),
            "joint_mae_deg": float(np.abs(joint_error).mean()),
            "joint_rmse_deg": float(np.sqrt(np.mean(joint_error**2))),
            "joint_p95_deg": float(np.quantile(np.abs(joint_error), 0.95)),
            "gripper_mae": float(np.abs(error[:, gripper]).mean()),
            "gripper_rmse": float(np.sqrt(np.mean(error[:, gripper] ** 2))),
            "hold_joint_mae_deg": float(np.abs(hold_joint).mean()),
            "hold_gripper_mae": float(np.abs(baseline[:, gripper]).mean()),
        }
        for i, name in enumerate(PIPER_NAMES[:-1]):
            scores[f"{name.removesuffix('.pos')}_mae_deg"] = float(np.abs(joint_error[:, i]).mean())
        for key in ("joint_mae_deg", "gripper_mae"):
            denominator = scores[f"hold_{key}"]
            if denominator > 1e-12:
                scores[f"{key}_ratio_to_hold"] = scores[key] / denominator
        # Avoid letting numerous stationary targets dominate the aggregate error.
        moving = np.max(np.abs(hold_joint), axis=1) > moving_threshold_deg
        scores["moving_pairs"] = int(moving.sum())
        if moving.any():
            scores["moving_joint_mae_deg"] = float(np.abs(joint_error[moving]).mean())
        if horizon > 1:
            adjacent = mask[:, 1:] & mask[:, :-1]
            delta_error = (np.diff(predicted[:, :horizon], axis=1) - np.diff(target[:, :horizon], axis=1))[
                adjacent
            ]
            scores["adjacent_pairs"] = int(adjacent.sum())
            if len(delta_error):
                scores["joint_delta_mae_deg_per_step"] = float(
                    np.abs(np.rad2deg(delta_error[:, joints])).mean()
                )
                scores["gripper_delta_mae_per_step"] = float(np.abs(delta_error[:, gripper]).mean())
        metrics.update({prefix + key: value for key, value in scores.items()})
    return metrics


@contextmanager
def evaluation_rng(device, seed):
    """Restore Python/NumPy/CPU/current-device RNG even if decoding or inference fails."""
    import torch

    device = torch.device(device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class PiperActionEvaluator:
    """Cache fixed observations without augmentation; keep labels outside inference inputs."""

    def __init__(self, datasets, policy_config, preprocessor, postprocessor, options, output_dir, device):
        self.config, self.options, self.device = policy_config, options, device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Evaluation never resets or mutates the training processors.
        with evaluation_rng(device, options.seed):
            self.preprocessor, self.postprocessor = deepcopy(preprocessor), deepcopy(postprocessor)
            self.samples = {name: self._cache(dataset) for name, dataset in datasets.items()}
        manifest = {
            "seed": options.seed,
            "chunk_size": policy_config.chunk_size,
            "execution_steps": options.execution_steps,
            "num_inference_steps": policy_config.num_inference_steps,
            "moving_threshold_deg": options.moving_threshold_deg,
            "units": "six absolute joints in radians (reported in degrees); gripper in dataset units",
            "inference": "full observation-conditioned flow at every stage; fixed noise per absolute index",
            "splits": {
                name: [sample["identity"] for sample in samples] for name, samples in self.samples.items()
            },
        }
        path = self.output_dir / "action_eval_samples.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Fixed action evaluation samples/settings changed in this output directory")
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    def _cache(self, dataset):
        import torch

        names, state_names = piper_names(dataset.meta.features)
        state_order = [state_names.index(name) for name in names]
        horizon = self.config.chunk_size
        if len(dataset) == 0:
            raise ValueError("Empty action evaluation split")
        indices = np.linspace(0, len(dataset) - 1, min(self.options.samples, len(dataset)), dtype=int)
        original_transforms = dataset.image_transforms
        samples = []
        dataset.clear_image_transforms()
        try:
            for index in indices:
                item = dataset[int(index)]
                episode_index = int(item["episode_index"].item())
                absolute_index, frame = int(item["index"].item()), int(item["frame_index"].item())
                episode = dataset.meta.episodes[episode_index]
                start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
                if absolute_index != start + frame or not start <= absolute_index < end:
                    raise ValueError("Inconsistent episode/frame/absolute index in fixed sample")
                valid = absolute_index + np.arange(horizon) < end
                padding = item["action_is_pad"]
                target = as_numpy(item["action"]).copy()
                if padding.dtype != torch.bool or target.shape != (horizon, 7):
                    raise ValueError("Invalid action label shape or padding dtype")
                if not np.array_equal(~padding.cpu().numpy(), valid):
                    raise ValueError("Action window does not match the episode boundary")
                state = as_numpy(item["observation.state"])
                if (
                    state.shape != (7,)
                    or not np.isfinite(state).all()
                    or not np.isfinite(target[valid]).all()
                ):
                    raise ValueError("Invalid state or valid action labels")
                if not isinstance(item["task"], str) or not item["task"].strip():
                    raise ValueError("Fixed action evaluation needs a recorded task string")
                observation = {key: deepcopy(item[key]) for key in self.config.input_features}
                observation["task"] = item["task"]
                samples.append(
                    {
                        "identity": {"index": absolute_index, "episode": episode_index, "frame": frame},
                        "observation": observation,
                        "target": target,
                        "valid": valid,
                        "hold": np.repeat(state[state_order][None], horizon, axis=0),
                        "names": names,
                    }
                )
        finally:
            dataset.set_image_transforms(original_transforms)
        return samples

    def evaluate(self, policy, step, autocast_context):
        import torch
        from torch.utils.data import default_collate

        started = time.perf_counter()
        metrics = {}
        was_training = policy.training
        # Inpainting checkpoints contain the same model; evaluate their full conditioned predictor.
        # The stage and training mode are restored even on an inference error.
        configs = {id(config): config for config in (policy.config, policy.model.config)}
        stages = {key: config.training_stage for key, config in configs.items()}
        try:
            with evaluation_rng(self.device, self.options.seed), torch.inference_mode(), autocast_context():
                for config in configs.values():
                    config.training_stage = "flow"
                policy.eval()
                for split, samples in self.samples.items():
                    predictions = []
                    for sample in samples:
                        for processor in (self.preprocessor, self.postprocessor):
                            processor.reset()
                        batch = default_collate([deepcopy(sample["observation"])])
                        for key in self.config.image_features:
                            if batch[key].dtype != torch.uint8:
                                raise ValueError(
                                    "Fixed observations must contain unaugmented uint8 RGB images"
                                )
                            batch[key] = batch[key].float() / 255.0
                        batch = self.preprocessor(batch)
                        if batch.pop("action", None) is not None:
                            raise ValueError("Action labels must never enter action prediction")
                        generator = torch.Generator(device=self.device).manual_seed(
                            (self.options.seed + sample["identity"]["index"]) % (2**63 - 1)
                        )
                        noise = torch.randn(
                            (1, self.config.chunk_size, self.config.max_action_dim),
                            generator=generator,
                            device=self.device,
                        )
                        predicted = policy.predict_action_chunk(batch, noise=noise)
                        if tuple(predicted.shape) != (1, self.config.chunk_size, 7):
                            raise ValueError("Unexpected predicted action shape")
                        physical = torch.stack(
                            [self.postprocessor(predicted[:, h]) for h in range(self.config.chunk_size)],
                            dim=1,
                        )
                        predictions.append(as_numpy(physical[0]))
                    scores = action_metrics(
                        predictions,
                        [sample["target"] for sample in samples],
                        [sample["valid"] for sample in samples],
                        [sample["hold"] for sample in samples],
                        samples[0]["names"],
                        [1, self.options.execution_steps, self.config.chunk_size],
                        self.options.moving_threshold_deg,
                    )
                    scores["samples"] = len(samples)
                    metrics.update({f"action_{split}/{key}": value for key, value in scores.items()})
        finally:
            for key, config in configs.items():
                config.training_stage = stages[key]
            policy.train(was_training)
        metrics["action_eval_seconds"] = time.perf_counter() - started
        record = {"step": step, "training_stage": str(policy.config.training_stage), "metrics": metrics}
        with (self.output_dir / "action_eval.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return metrics


class ActionParameterMonitor:
    """Measure gradients and actual post-optimizer deltas, including CABO and weight decay."""

    def __init__(self, policy):
        self.groups = {
            name: list(getattr(policy.model, name).parameters())
            for name in ("vlm_prompt_tokens", "prompt_tokens", "action_in_proj", "action_out_proj")
        }

    def before_step(self):
        import torch

        self.before = {}
        self.metrics = {}
        for name, parameters in self.groups.items():
            snapshots = [p.detach().float().clone() for p in parameters]
            self.before[name] = snapshots
            count = sum(p.numel() for p in parameters)
            norm_sq = sum(float(torch.sum(value**2).item()) for value in snapshots)
            gradients = [p.grad.detach().float() for p in parameters if p.grad is not None]
            grad_sq = sum(float(torch.sum(value**2).item()) for value in gradients)
            prefix = f"action_update/{name}/"
            self.metrics.update(
                {
                    prefix + "trainable_numel": sum(p.numel() for p in parameters if p.requires_grad),
                    prefix + "grad_numel": sum(value.numel() for value in gradients),
                    prefix + "grad_rms": math.sqrt(grad_sq / max(count, 1)),
                    prefix + "param_rms": math.sqrt(norm_sq / max(count, 1)),
                }
            )

    def after_step(self):
        import torch

        for name, parameters in self.groups.items():
            snapshots = self.before[name]
            count = sum(p.numel() for p in parameters)
            delta_sq = sum(
                float(torch.sum((p.detach().float() - old) ** 2).item())
                for p, old in zip(parameters, snapshots, strict=True)
            )
            norm_sq = sum(float(torch.sum(old**2).item()) for old in snapshots)
            prefix = f"action_update/{name}/"
            self.metrics[prefix + "update_rms"] = math.sqrt(delta_sq / max(count, 1))
            self.metrics[prefix + "relative_update"] = math.sqrt(delta_sq) / max(math.sqrt(norm_sq), 1e-12)
        self.before.clear()
        return self.metrics


def on_main_process(accelerator, callback):
    """Keep DDP ranks synchronized and propagate rank-0 evaluation errors to every rank."""
    import torch

    result, error = None, None
    if accelerator.is_main_process:
        try:
            result = callback()
        except Exception as exc:
            error = exc
    failures = accelerator.reduce(torch.tensor(int(error is not None), device=accelerator.device), "sum")
    if failures.item():
        raise RuntimeError(
            f"Piper action evaluation failed on rank 0: {error or 'see rank-0 traceback'}"
        ) from error
    return result
