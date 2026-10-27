"""Training-time spectra of an output-projected velocity-to-prompt Jacobian.

G[(sample, time, probe), :] = u_probe.T @ d velocity / d prompt / sqrt(R).
Only the output axis is projected; every VLM-prompt parameter is retained.
This is a fixed-probe estimate, NOT the full Jacobian or a loss-gradient rank.
"""

import csv
import hashlib
import json
import logging
import math
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.ntk import spectral_metrics
from lerobot.utils.pi05_trainable import validate_vlm_trainables
from lerobot.utils.piper_training_eval import evaluation_rng


def tangent_spectrum(rows):
    """Report both singular-value entropy (J) and squared-singular-value entropy (NTK)."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or min(rows.shape) < 1 or not np.isfinite(rows).all():
        raise ValueError("Expected a finite nonempty projected Jacobian")
    gram = rows @ rows.T
    spectrum = spectral_metrics(gram, rows.shape[1])
    eigenvalues = np.asarray(spectrum["eigenvalues"])
    # Remove numerical null-space eigenvalues before taking square roots.
    tolerance = np.finfo(np.float64).eps * len(eigenvalues) * eigenvalues.max(initial=0)
    eigenvalues[eigenvalues <= tolerance] = 0
    singular_values = np.sqrt(eigenvalues)
    trace = float(eigenvalues.sum())
    metrics = {
        "jacobian_erank": 0.0,
        "ntk_erank": 0.0,
        "participation_rank": 0.0,
        "stable_rank": 0.0,
        "rank90_energy": 0,
        "rank95_energy": 0,
        "tangent_trace": trace,
        "energy_per_parameter": trace / rows.shape[1],
        "parameter_count": rows.shape[1],
        "row_count": rows.shape[0],
        "rank_ceiling": min(rows.shape),
    }
    if trace > 0:
        p = singular_values[singular_values > 0] / singular_values.sum()
        q = eigenvalues[eigenvalues > 0] / trace
        metrics.update(
            jacobian_erank=float(np.exp(-(p * np.log(p)).sum())),
            ntk_erank=float(np.exp(-(q * np.log(q)).sum())),
            participation_rank=float(1 / (q @ q)),
            stable_rank=float(trace / eigenvalues[0]),
            rank90_energy=int(np.searchsorted(np.cumsum(q), 0.90) + 1),
            rank95_energy=int(np.searchsorted(np.cumsum(q), 0.95) + 1),
        )
    return metrics, gram, eigenvalues, singular_values


def collective_call(accelerator, callback):
    """Every rank calls this; propagate Python failures before the next data collective."""
    result, error = None, None
    try:
        result = callback()
    except Exception as exc:
        error = exc
        logging.exception("Prompt-rank diagnostic failed on rank %s", accelerator.process_index)
    failures = accelerator.reduce(torch.tensor(int(error is not None), device=accelerator.device), "sum")
    if failures.item():
        raise RuntimeError(
            f"Prompt-rank diagnostic failed: {error or 'see another rank traceback'}"
        ) from error
    return result


@contextmanager
def isolated_prompt(policy):
    """Use a temporary leaf outside DDP/optimizer hooks; preserve exact training state.

    A detached FP32 copy avoids accumulating probe gradients in the optimizer's
    parameter or triggering its DDP reducer. Forward embedding casts stay unchanged.
    Module flags are restored individually, including frozen submodules in eval mode.
    """
    original = policy.model.vlm_prompt_tokens.weight
    validate_vlm_trainables(policy)
    parameter_flags = [(p, p.requires_grad) for p in policy.parameters()]
    flags = [(module, module.training) for module in policy.modules()]
    try:
        # Freeze expert leaves during the probe only: no DDP hooks or optimizer
        # gradients are touched. This still measures d velocity / d VLM prompt.
        for parameter, _ in parameter_flags:
            parameter.requires_grad_(False)
        policy.model.vlm_prompt_tokens.weight = torch.nn.Parameter(original.detach().float().clone())
        for module, _ in flags:
            module.training = False
        yield policy.model.vlm_prompt_tokens.weight
    finally:
        policy.model.vlm_prompt_tokens.weight = original
        for parameter, requires_grad in parameter_flags:
            parameter.requires_grad_(requires_grad)
        for module, training in flags:
            module.training = training


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(item) for item in value]
    return deepcopy(value)


def _fingerprint(samples):
    digest = hashlib.sha256()
    for sample in samples:
        for key, value in sorted(sample.items()):
            digest.update(key.encode())
            if isinstance(value, torch.Tensor):
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


class PromptRankMonitor:
    @classmethod
    def from_cache(cls, directory, device):
        """Restore the exact processed observations, noise and projections without a dataset."""
        from lerobot.configs.prompt_rank import PromptRankConfig

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported prompt-rank cache schema")
        cache = torch.load(directory / "fixed_probe.pt", map_location="cpu", weights_only=True)
        result = cls.__new__(cls)
        result.options = PromptRankConfig(**manifest["options"])
        result.options.validate()
        result.device, result.output_dir, result.manifest = torch.device(device), directory, manifest
        result.samples, result.noise, result.probes = (
            cache["samples"],
            cache["noise"],
            cache["output_probes"],
        )
        result.action_dim, result.horizon = manifest["action_dim"], manifest["chunk_size"]
        if "keyframe_masks_sha256" in manifest:
            from lerobot.utils.pdet_order import arrays_hash

            result.keyframe_masks = torch.load(
                directory / "keyframe_masks.pt", map_location="cpu", weights_only=True
            )
            if result.keyframe_masks.dtype != torch.bool or result.keyframe_masks.shape != (
                len(result.samples),
                1,
                result.horizon,
            ):
                raise ValueError("Invalid cached keyframe probe mask dimensions/dtype")
            if (
                arrays_hash({"key": result.keyframe_masks[:, 0, :].numpy()})
                != manifest["keyframe_masks_sha256"]
            ):
                raise ValueError("Cached keyframe probe masks changed")
        if _fingerprint(result.samples) != manifest["sample_sha256"]:
            raise ValueError("Cached observations do not match the recorded hash")
        if len(result.samples) != len(result.noise) or not result.samples:
            raise ValueError("Invalid cached noise/sample count")
        expected = (result.options.output_probes, 1, result.horizon, result.action_dim)
        if tuple(result.probes.shape) != expected or not torch.isfinite(result.probes).all():
            raise ValueError("Invalid cached output projections")
        if any(not torch.isfinite(noise).all() for noise in result.noise):
            raise ValueError("Non-finite cached noise")
        return result

    def __init__(self, dataset, preprocessor, config, options, output_dir, device):
        from lerobot.utils.collate import lerobot_collate_fn

        options.validate()
        self.options, self.device = options, torch.device(device)
        self.output_dir = Path(output_dir) / "prompt_rank"
        count = min(options.samples, len(dataset))
        if count == 0:
            raise ValueError("Cannot build a prompt-rank probe from an empty dataset")
        # One reproducible random frame per dataset stratum; not a claim of task balancing.
        rng = np.random.default_rng(options.seed)
        edges = np.linspace(0, len(dataset), count + 1, dtype=int)
        indices = [int(rng.integers(left, right)) for left, right in zip(edges[:-1], edges[1:], strict=True)]
        original_transforms = dataset.image_transforms
        self.samples, identities = [], []
        with evaluation_rng(device, options.seed):
            processor = deepcopy(preprocessor)
            dataset.clear_image_transforms()
            try:
                for index in indices:
                    item = dataset[index]
                    identity = {"dataset_index": index}
                    for key in ("index", "episode_index", "frame_index", "task_index"):
                        if key in item:
                            identity[key] = int(item[key])
                    identity["task"] = item.get("task", "")
                    identities.append(identity)
                    batch = lerobot_collate_fn([item])
                    for key in dataset.meta.camera_keys:
                        if batch[key].dtype == torch.uint8:
                            batch[key] = batch[key].float() / 255
                    batch = processor(batch)
                    keep = set(config.image_features) | {
                        "action",
                        "action_is_pad",
                        OBS_LANGUAGE_TOKENS,
                        OBS_LANGUAGE_ATTENTION_MASK,
                    }
                    self.samples.append({key: _cpu(value) for key, value in batch.items() if key in keep})
            finally:
                dataset.set_image_transforms(original_transforms)
        self.action_dim = self.samples[0]["action"].shape[-1]
        self.horizon = config.chunk_size
        generator = torch.Generator().manual_seed(options.seed)
        self.probes = torch.randint(
            0, 2, (options.output_probes, 1, self.horizon, self.action_dim), generator=generator
        ).float().mul_(2).sub_(1) / math.sqrt(options.output_probes)
        self.noise = [
            torch.randn((1, self.horizon, config.max_action_dim), generator=generator) for _ in self.samples
        ]
        self.manifest = {
            "schema_version": 1,
            "torch_version": str(torch.__version__),
            "quantity": "stacked output-projected d(flow_velocity)/d(VLM_prompt); no parameter sketch",
            "jacobian_erank": "exp(entropy(sigma/sum(sigma)))",
            "ntk_erank": "exp(entropy(lambda/sum(lambda))); lambda=sigma^2",
            "projection": "shared Rademacher vectors / sqrt(output_probes), masked padded timesteps",
            "forward_precision": config.dtype,
            "parameter_leaf_precision": "float32",
            "options": asdict(options),
            "samples": identities,
            "sample_sha256": _fingerprint(self.samples),
            "dataset_root": str(getattr(dataset, "root", "")),
            "pretrained_path": str(config.pretrained_path),
            "normalization_mapping": {str(k): str(v) for k, v in config.normalization_mapping.items()},
            "chunk_size": self.horizon,
            "action_dim": self.action_dim,
            "num_vlm_prompt_tokens": config.num_vlm_prompt_tokens,
            "projected_rows": count * len(options.timesteps) * options.output_probes,
            "scope": "fixed training observations, teacher-interpolated noisy actions; not rollout Jacobian",
        }

    def save_protocol(self, start_step):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != self.manifest:
            raise ValueError("Prompt-rank probe protocol changed; use a new output directory")
        records = self.output_dir / "metrics.jsonl"
        if records.exists():
            steps = [json.loads(line)["step"] for line in records.read_text().splitlines() if line.strip()]
            if steps and max(steps) > start_step:
                raise ValueError(
                    "Prompt-rank history is newer than the resumed model; use a new output directory"
                )
        path.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n")
        if hasattr(self, "keyframe_masks"):
            torch.save(self.keyframe_masks, self.output_dir / "keyframe_masks.pt")
        probe_path = self.output_dir / "fixed_probe.pt"
        if not probe_path.exists():
            torch.save(
                {"samples": self.samples, "noise": self.noise, "output_probes": self.probes}, probe_path
            )

    def verify_across_ranks(self, accelerator):
        digest = torch.tensor(
            list(bytes.fromhex(self.manifest["sample_sha256"])), device=self.device, dtype=torch.uint8
        )
        digests = accelerator.gather(digest).reshape(accelerator.num_processes, -1)
        if not torch.equal(digests, digests[:1].expand_as(digests)):
            raise ValueError("Processed prompt-rank observations differ between ranks")

    def _probe_forward(self, policy, sample_index, timestep):
        """One native-precision forward; callers own grad/autocast contexts."""
        batch = {key: value.to(self.device) for key, value in self.samples[sample_index].items()}
        actions = policy.prepare_action(batch)
        if actions.shape != self.noise[sample_index].shape:
            raise ValueError("Probe action shape does not match the model")
        noise = self.noise[sample_index].to(self.device)
        time_tensor = torch.tensor([timestep], device=self.device, dtype=torch.float32)
        noisy = timestep * noise + (1 - timestep) * actions
        valid = ~batch.get(
            "action_is_pad", torch.zeros((1, self.horizon), dtype=torch.bool, device=self.device)
        ).bool()
        if not valid.any():
            raise ValueError(f"Probe sample {sample_index} has no valid action targets")
        images, masks = policy._preprocess_images(batch)
        velocity = policy.model.predict_velocity(
            images,
            masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            noisy,
            time_tensor,
        )[..., : self.action_dim]
        target = (noise - actions)[..., : self.action_dim]
        probes = self.probes.to(self.device) * valid[None, :, :, None]
        return velocity, target, probes, valid

    def _rows(self, policy, prompt, sample_index, timestep, autocast_context):
        with torch.enable_grad(), autocast_context():
            velocity, target, probes, valid = self._probe_forward(policy, sample_index, timestep)
            if self.options.fixed_flow_loss:
                squared = (velocity.detach().float() - target.float()).square()
                errors = squared[valid]
                self._last_probe_error = torch.stack(
                    (errors.sum(), torch.tensor(errors.numel(), device=self.device, dtype=torch.float32))
                )
                if hasattr(self, "keyframe_masks"):
                    key = self.keyframe_masks[sample_index].to(self.device) & valid
                    split_errors = []
                    for mask in (key, valid & ~key):
                        values = squared[mask]
                        split_errors.extend(
                            (
                                values.sum(),
                                torch.tensor(values.numel(), device=self.device, dtype=torch.float32),
                            )
                        )
                    self._last_probe_error = torch.cat((self._last_probe_error, torch.stack(split_errors)))
            rows = []
            for index, probe in enumerate(probes):
                (gradient,) = torch.autograd.grad(
                    (velocity.float() * probe).sum(), prompt, retain_graph=index + 1 < len(probes)
                )
                rows.append(gradient.detach().float().flatten())
        rows = torch.stack(rows)
        if not torch.isfinite(rows).all():
            raise FloatingPointError("Non-finite prompt Jacobian")
        return rows

    def measure(self, policy, step, accelerator):
        """Shard samples across DDP ranks, with a small collective after each forward/VJP set."""
        started = time.perf_counter()
        rows = []
        loss_records = []
        with evaluation_rng(self.device, self.options.seed), isolated_prompt(policy) as prompt:
            world = accelerator.num_processes
            for start in range(0, len(self.samples), world):
                sample_index = start + accelerator.process_index
                for timestep in self.options.timesteps:
                    local = collective_call(
                        accelerator,
                        lambda sample_index=sample_index, timestep=timestep: (
                            self._rows(policy, prompt, sample_index, timestep, accelerator.autocast)
                            if sample_index < len(self.samples)
                            else torch.zeros((self.options.output_probes, prompt.numel()), device=self.device)
                        ),
                    )
                    gathered = accelerator.gather(local)
                    if self.options.fixed_flow_loss:
                        error_columns = 6 if hasattr(self, "keyframe_masks") else 2
                        local_error = (
                            self._last_probe_error
                            if sample_index < len(self.samples)
                            else torch.zeros(error_columns, device=self.device)
                        )
                        errors = accelerator.gather(local_error).reshape(world, error_columns)
                        if accelerator.is_main_process:
                            for rank, values in enumerate(errors.cpu().tolist()):
                                total, count = values[:2]
                                if start + rank < len(self.samples):
                                    identity = self.manifest["samples"][start + rank]
                                    loss_records.append(
                                        {
                                            "sample_index": start + rank,
                                            "time": timestep,
                                            "task_index": identity.get("task_index", -1),
                                            "squared_error_sum": total,
                                            "valid_coordinates": int(count),
                                        }
                                    )
                                    if error_columns == 6:
                                        loss_records[-1].update(
                                            key_squared_error_sum=values[2],
                                            key_coordinates=int(values[3]),
                                            nonkey_squared_error_sum=values[4],
                                            nonkey_coordinates=int(values[5]),
                                        )
                    if accelerator.is_main_process:
                        valid_ranks = min(world, len(self.samples) - start)
                        rows.append(gathered[: valid_ranks * self.options.output_probes].cpu().numpy())
                    # No GPU graph or VJP buffer survives between sample/time pairs.
                    del local, gathered

        # Reorder rank-interleaved results to a world-size-independent (sample,time,probe) order.
        def save():
            if not accelerator.is_main_process:
                return None
            blocks, offset = [], 0
            for start in range(0, len(self.samples), world):
                count = min(world, len(self.samples) - start)
                block = np.stack(rows[offset : offset + len(self.options.timesteps)])
                block = block.reshape(len(self.options.timesteps), count, self.options.output_probes, -1)
                blocks.append(block.transpose(1, 0, 2, 3).reshape(-1, prompt.numel()))
                offset += len(self.options.timesteps)
            return self._save_step(
                policy, step, np.concatenate(blocks), time.perf_counter() - started, loss_records
            )

        return collective_call(accelerator, save)

    def _save_step(self, policy, step, rows, elapsed, loss_records=None):
        metrics, gram, eigenvalues, singular_values = tangent_spectrum(rows)
        if loss_records:
            if "key_coordinates" in loss_records[0]:
                for label in ("key", "nonkey"):
                    count = sum(item[f"{label}_coordinates"] for item in loss_records)
                    metrics[f"fixed_training_probe_{label}_coordinates"] = count
                    if count:
                        metrics[f"fixed_training_probe_{label}_mse"] = (
                            sum(item[f"{label}_squared_error_sum"] for item in loss_records) / count
                        )
            metrics["fixed_training_probe_flow_mse"] = sum(
                item["squared_error_sum"] for item in loss_records
            ) / sum(item["valid_coordinates"] for item in loss_records)
            for timestep in self.options.timesteps:
                subset = [item for item in loss_records if item["time"] == timestep]
                metrics[f"fixed_training_probe_flow_mse_t{timestep:g}"] = sum(
                    item["squared_error_sum"] for item in subset
                ) / sum(item["valid_coordinates"] for item in subset)
        metrics["probe_seconds"] = elapsed
        metrics["erank_fraction_of_ceiling"] = metrics["jacobian_erank"] / metrics["rank_ceiling"]
        record = {"step": int(step), **metrics}
        weights = policy.model.vlm_prompt_tokens.weight.detach().float().cpu().numpy()
        path = self.output_dir / f"step_{step:06d}.npz"
        if path.exists():
            with np.load(path) as previous:
                if not np.array_equal(previous["prompt"], weights):
                    raise ValueError(f"Different prompt already recorded at step {step}")
            return metrics  # Do not duplicate a measurement when resuming at an existing step.
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary, gram=gram, eigenvalues=eigenvalues, singular_values=singular_values, prompt=weights
        )
        temporary.replace(path)
        if loss_records:
            with (self.output_dir / "fixed_training_probe_loss.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "scope": "fixed training observations; NOT held-out validation",
                            "samples": loss_records,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
        with (self.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        csv_path = self.output_dir / "metrics.csv"
        exists = csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)
        return metrics
