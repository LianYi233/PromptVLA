"""Key-action weighting without resampling, new flow draws, or global RNG changes.

R2 is a within-chunk placebo: circularly shift the event mask ONLY over valid
target timesteps. It exactly matches K2's count/denominator in every sample.
It tests action-time placement conditional on the same observations; it is not
a control for choosing different input images. A boundary can split a window.
"""

import json
from pathlib import Path

import numpy as np
import torch

from lerobot.utils.pdet_order import META_COLUMNS, arrays_hash

PROTOCOL = "promptvla-key-action-loss-v1"
MODES = ("B", "K2", "R2", "Kearly", "Klate")


def episode_ends(metadata):
    episodes = np.asarray(metadata["episode_index"])
    edges = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1, len(episodes)]
    if len(np.unique(episodes[edges[:-1]])) != len(edges) - 1:
        raise ValueError("Episodes must be contiguous")
    result = np.empty(len(episodes), dtype=np.int64)
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if not np.array_equal(metadata["frame_index"][left:right], np.arange(right - left)):
            raise ValueError("Non-contiguous episode frame numbering")
        result[left:right] = right
    return result


def chunk_masks(arrays, indices, horizon=50):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= len(arrays["event_mask"])):
        raise ValueError("Invalid keyframe sample index")
    ends = arrays["episode_end"][indices]
    targets = indices[:, None] + np.arange(horizon)[None, :]
    valid = targets < ends[:, None]
    # Padded targets never inherit the final event mask, even if the dataset repeats its action.
    mask = arrays["event_mask"][np.minimum(targets, ends[:, None] - 1)] & valid
    return mask, valid


def shifted_masks(mask, valid, shifts):
    lengths = valid.sum(axis=1)
    if np.any(lengths == 0) or not np.array_equal(
        valid, np.arange(valid.shape[1])[None, :] < lengths[:, None]
    ):
        raise ValueError("Expected a nonempty contiguous valid action prefix")
    positions = (np.arange(mask.shape[1])[None, :] - np.asarray(shifts)[:, None]) % lengths[:, None]
    return np.take_along_axis(mask, positions, axis=1) & valid


def normalized_key_mass(weights, key_counts, valid_counts):
    return weights * key_counts / np.maximum(valid_counts + (weights - 1) * key_counts, 1)


def calibrated_curve(direction, key_counts, valid_counts):
    """Match K2's SUM of global-batch normalized key coefficients, not mean raw weight."""
    if direction not in ("early", "late"):
        raise ValueError("Unknown curriculum direction")
    base = np.linspace(1.5, 0.5, len(key_counts))
    if direction == "late":
        base = base[::-1]
    target = normalized_key_mass(2.0, key_counts, valid_counts).sum()
    low, high = 0.0, 4.0
    for _ in range(80):
        middle = (low + high) / 2
        mass = normalized_key_mass(1 + middle * base, key_counts, valid_counts).sum()
        if mass < target:
            low = middle
        else:
            high = middle
    curve = (1 + (low + high) / 2 * base).astype(np.float32)
    if not np.isclose(
        normalized_key_mass(curve.astype(float), key_counts, valid_counts).sum(), target, rtol=1e-7, atol=1e-6
    ):
        raise ValueError("Could not calibrate curriculum budget")
    return curve


def build_archive(scores, score_manifest, plan, *, random_seed=2026, horizon=50):
    info = plan["manifest"]
    if info["within_seed"] != 0 or "curriculum" in info:
        raise ValueError("Use the ORIGINAL D0/W0 orderbind plan, not a FrameSkip plan")
    if info["dataset_metadata_sha256"] != score_manifest["dataset_metadata_sha256"]:
        raise ValueError("Scores and old W0 plan refer to different data")
    config = score_manifest["config"]
    if (config["event_before"], config["event_after"], config["debounce"]) != (10, 2, 2):
        raise ValueError("This experiment requires the recorded [-10,+2], debounce=2 event windows")
    event = np.asarray(scores["event_neighborhood"])
    if not np.isin(event, [0, 1]).all():
        raise ValueError("Expected a binary event-neighborhood mask")
    arrays = {key: np.array(scores[key], copy=True) for key in META_COLUMNS}
    if arrays_hash(arrays) != info["dataset_metadata_sha256"]:
        raise ValueError("Event-mask metadata differs from the old plan")
    arrays.update(event_mask=event.astype(bool), episode_end=episode_ends(scores))
    count = len(plan["order"])
    # A private generator; this is not any P/D/E/T stream. Save all draws once.
    rng = np.random.default_rng(random_seed)
    arrays["random_shifts"] = rng.integers(1, 2**31, size=count, dtype=np.int64)
    global_batch = info["batch_size"] * info["world_size"]
    key_counts, valid_counts, overlap = [], [], 0
    per_task = {int(t): [0, 0] for t in np.unique(scores["task_index"])}
    for start in range(0, count, global_batch):
        ids = plan["order"][start : start + global_batch]
        indices = plan["index"][ids]
        mask, valid = chunk_masks(arrays, indices, horizon)
        random_mask = shifted_masks(mask, valid, arrays["random_shifts"][ids])
        assert np.array_equal(mask.sum(axis=1), random_mask.sum(axis=1))
        key_counts.append(mask.sum())
        valid_counts.append(valid.sum())
        overlap += int((mask & random_mask).sum())
        for task in np.unique(plan["task_index"][ids]):
            selected = plan["task_index"][ids] == task
            per_task[int(task)][0] += int(mask[selected].sum())
            per_task[int(task)][1] += int(valid[selected].sum())
    arrays["key_counts"] = np.asarray(key_counts, dtype=np.int64)
    arrays["valid_counts"] = np.asarray(valid_counts, dtype=np.int64)
    if not 0 < arrays["key_counts"].sum() < arrays["valid_counts"].sum():
        raise ValueError("No contrast: event mask covers none or all valid action targets")
    for mode in MODES:
        if mode in ("Kearly", "Klate"):
            weights = calibrated_curve(mode.removeprefix("K"), arrays["key_counts"], arrays["valid_counts"])
        else:
            weights = np.full(info["steps"], 1 if mode == "B" else 2, dtype=np.float32)
        arrays[f"weight_{mode}"] = weights
    manifest = {
        "protocol": PROTOCOL,
        "order_plan": info,
        "horizon": horizon,
        "random_seed": random_seed,
        "score_recipe": score_manifest,
        "random_control": "within-valid-chunk circular shift; exact per-sample key count; possible overlap",
        "random_overlap_fraction": overlap / int(arrays["key_counts"].sum()),
        "task_target_counts": per_task,
        "budget": "sum over updates of global-batch normalized key-action coefficients",
        "scope": "original input order and flow bindings; action-target weighting only",
        "arrays_sha256": arrays_hash(arrays),
    }
    return arrays, json.loads(json.dumps(manifest))


def save_archive(path, arrays, manifest):
    path = Path(path)
    if path.exists():
        _, previous = load_archive(path)
        if previous != manifest:
            raise ValueError(f"Different keyframe archive already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays, manifest=json.dumps(manifest, sort_keys=True))


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        arrays = {key: archive[key] for key in archive.files if key != "manifest"}
    if manifest.get("protocol") != PROTOCOL or arrays_hash(arrays) != manifest.get("arrays_sha256"):
        raise ValueError("Invalid/corrupted keyframe archive")
    if arrays["event_mask"].dtype != bool or not np.array_equal(episode_ends(arrays), arrays["episode_end"]):
        raise ValueError("Invalid event mask or episode boundaries")
    steps = manifest["order_plan"]["steps"]
    length = len(arrays["index"])
    if any(arrays[k].shape != (length,) for k in (*META_COLUMNS, "event_mask", "episode_end")):
        raise ValueError("Invalid event metadata dimensions")
    if arrays_hash({k: arrays[k] for k in META_COLUMNS}) != manifest["order_plan"]["dataset_metadata_sha256"]:
        raise ValueError("Event metadata hash differs from original plan")
    info = manifest["order_plan"]
    if arrays["random_shifts"].shape != (steps * info["world_size"] * info["batch_size"],):
        raise ValueError("Invalid random mask bank dimensions")
    if (
        arrays["key_counts"].shape != (steps,)
        or arrays["valid_counts"].shape != (steps,)
        or np.any(arrays["key_counts"] < 0)
        or np.any(arrays["key_counts"] > arrays["valid_counts"])
        or np.any(arrays["valid_counts"] <= 0)
    ):
        raise ValueError("Invalid key/valid counts")
    for mode in MODES:
        w = arrays[f"weight_{mode}"]
        if w.shape != (steps,) or not np.isfinite(w).all() or np.any(w < 1):
            raise ValueError("Invalid keyframe weight schedule")
    if not np.all(arrays["weight_B"] == 1) or not all(
        np.all(arrays[f"weight_{m}"] == 2) for m in ("K2", "R2")
    ):
        raise ValueError("B/K2/R2 weights were changed")
    for mode, sign in (("Kearly", -1), ("Klate", 1)):
        if np.any(sign * np.diff(arrays[f"weight_{mode}"]) < 0):
            raise ValueError("Curriculum has the wrong direction")
        reference = normalized_key_mass(2, arrays["key_counts"], arrays["valid_counts"]).sum()
        actual = normalized_key_mass(
            arrays[f"weight_{mode}"].astype(float), arrays["key_counts"], arrays["valid_counts"]
        ).sum()
        if not np.isclose(reference, actual, rtol=1e-7, atol=1e-6):
            raise ValueError("Curriculum normalized budgets differ")
    return arrays, manifest


def validate_order(plan, manifest):
    if plan["manifest"] != manifest["order_plan"] or plan["manifest"]["within_seed"] != 0:
        raise ValueError("Original W0 plan/occurrence IDs changed; do not regenerate or relabel")
    if "curriculum" in plan["manifest"]:
        raise ValueError("A FrameSkip sampling plan is not the original W0 baseline")


def loss_splits(losses, valid, key):
    """Detached sums/counts for aggregation; never confuse weighted training loss with probe MSE."""
    if key.shape != valid.shape or key.dtype != torch.bool:
        raise ValueError("Expected boolean key-action mask matching valid timesteps")
    values = losses.detach().float()
    result = {}
    for label, mask in (("key", valid & key), ("nonkey", valid & ~key)):
        selected = mask.unsqueeze(-1).expand_as(values)
        total = torch.where(selected, values, 0).sum().item()
        count = int(selected.sum().item())
        result[f"keyframe/{label}_squared_error_sum"] = total
        result[f"keyframe/{label}_coordinates"] = count
        if count:
            result[f"keyframe/{label}_mse"] = total / count
    return result


def weighted_action_loss(losses, valid, weights):
    if weights.shape != valid.shape or weights.requires_grad:
        raise ValueError("Action loss weights must be fixed (batch,horizon) values")
    weights = weights.to(device=losses.device, dtype=torch.float32)
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Action loss weights must be finite and positive")
    weights = torch.where(valid, weights, 0)
    masked = torch.where(valid.unsqueeze(-1), losses, 0) * weights.unsqueeze(-1)
    dimensions = losses.shape[-1]
    scalar = masked.sum() / (weights.sum() * dimensions).clamp_min(1)
    per_sample = masked.sum((1, 2)) / (weights.sum(1) * dimensions).clamp_min(1)
    per_dim = masked.sum((0, 1)) / weights.sum().clamp_min(1)
    return scalar, per_sample, per_dim


class KeyframeLossRuntime:
    def __init__(self, cfg, pdet_streams, accelerator):
        self.mode = cfg.keyframe_loss.mode
        self.arrays, self.manifest = load_archive(cfg.keyframe_loss.archive)
        self.plan = pdet_streams.plan
        validate_order(self.plan, self.manifest)
        if arrays_hash(pdet_streams.frames) != self.manifest["order_plan"]["dataset_metadata_sha256"]:
            raise ValueError("Keyframe metadata differs from the training dataset")
        self.rank = accelerator.process_index
        self.output = Path(cfg.output_dir) / "keyframe_loss"
        self.output.mkdir(parents=True, exist_ok=True)
        self.log_freq = cfg.log_freq
        # Copy the small archive into the run so evaluation/audits do not depend on a temporary path.
        if accelerator.is_main_process:
            save_archive(self.output / "archive.npz", self.arrays, self.manifest)
        record = {"mode": self.mode, "archive": self.manifest, "rank": self.rank}
        path = self.output / f"manifest-rank{self.rank}.json"
        if path.exists():
            raise FileExistsError(f"Keyframe runtime already exists; do not overwrite: {path}")
        path.write_text(json.dumps(record, indent=2) + "\n")

    def inputs(self, batch, step, indices, occurrences):
        from lerobot.utils.pdet import tensor_hash

        ids = torch.as_tensor(occurrences).cpu().numpy().astype(np.int64)
        indices = torch.as_tensor(indices).cpu().numpy().astype(np.int64)
        info = self.plan["manifest"]
        start = (step * info["world_size"] + self.rank) * info["batch_size"]
        if not np.array_equal(ids, self.plan["order"][start : start + info["batch_size"]]):
            raise ValueError("Keyframe runtime received different W0 occurrences")
        if not np.array_equal(indices, self.plan["index"][ids]):
            raise ValueError("Keyframe runtime received different frame identities")
        key, valid = chunk_masks(self.arrays, indices, self.manifest["horizon"])
        if "action_is_pad" not in batch or not np.array_equal(
            ~batch["action_is_pad"].bool().cpu().numpy(), valid
        ):
            raise ValueError("Actual action padding/timestamps differ from the original contiguous chunk")
        selected = shifted_masks(key, valid, self.arrays["random_shifts"][ids]) if self.mode == "R2" else key
        strength = float(self.arrays[f"weight_{self.mode}"][step])
        weights = 1 + np.float32(strength - 1) * selected.astype(np.float32)
        device = batch["action"].device
        output = {"action_key_mask": torch.as_tensor(key, device=device)}
        if self.mode != "B":
            output["action_loss_weights"] = torch.as_tensor(weights, device=device)
        record = {
            "step": step,
            "rank": self.rank,
            "mode": self.mode,
            "occurrence_ids": ids.tolist(),
            "sample_ids": indices.tolist(),
            "key_mask_sha256": tensor_hash(torch.from_numpy(key)),
            "weight_sha256": tensor_hash(torch.from_numpy(weights)),
            "valid_targets": int(valid.sum()),
            "key_targets": int(key.sum()),
            "weighted_key_targets": int(selected.sum()),
            "strength": strength,
            "denominator_targets": float(np.where(valid, weights, 0).sum()),
            "scope": "unweighted diagnostics use TRUE event mask also in R2",
        }
        with (self.output / f"inputs-rank{self.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return output

    def record_metrics(self, step, metrics):
        if (self.log_freq > 0 and step % self.log_freq == 0) or step == self.plan["manifest"]["steps"]:
            row = {"step": step, **{k: v for k, v in (metrics or {}).items() if k.startswith("keyframe/")}}
            with (self.output / f"training-rank{self.rank}.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")

    def attach_probe_masks(self, monitor):
        indices = [item["index"] for item in monitor.manifest["samples"]]
        key, valid = chunk_masks(self.arrays, indices, monitor.horizon)
        for i, sample in enumerate(monitor.samples):
            if not np.array_equal(~sample["action_is_pad"].bool().numpy().reshape(-1), valid[i]):
                raise ValueError("Probe action padding differs from the event timeline")
        monitor.keyframe_masks = torch.from_numpy(key[:, None, :])
        monitor.manifest["keyframe_masks_sha256"] = arrays_hash({"key": key})
        monitor.manifest["keyframe_archive_sha256"] = self.manifest["arrays_sha256"]
