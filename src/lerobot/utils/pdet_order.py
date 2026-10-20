"""Finite, fixed-task-slot experiments. No images, model loading or global RNG draws.

An occurrence ID identifies one use of a frame. W changes only the permutation of
occurrences within task slots; the pool, task sequence and occurrence-bound E/T
remain fixed. This is deliberately NOT a replay of legacy step/rank PDET draws.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PLAN_PROTOCOL = "pdet-fixed-task-slots-v1"
FLOW_PROTOCOL = "pdet-occurrence-flow-v1"
SLOT_FLOW_PROTOCOL = "pdet-slot-flow-v1"
OCCURRENCE_KEY = "_pdet_occurrence_id"
META_COLUMNS = ("index", "task_index", "episode_index", "frame_index", "phase_bin", "gripper_change")


def arrays_hash(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def read_frame_metadata(root):
    """Read scalar identities/actions from local parquet; never decode videos/download data."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((Path(root) / "data").rglob("*.parquet"))
    if not paths:
        raise ValueError(f"No local data parquet files under {root}/data")
    columns = ["index", "task_index", "episode_index", "frame_index", "action"]
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])
    order = np.argsort(np.asarray(table["index"].to_pylist(), dtype=np.int64))
    result = {key: np.asarray(table[key].to_pylist(), dtype=np.int64)[order] for key in columns[:-1]}
    if not np.array_equal(result["index"], np.arange(len(order))):
        raise ValueError("Fixed-task plans require a complete dataset with unique contiguous global indices")
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[order]
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("LIBERO coverage diagnostics require finite 7-D actions (last coordinate: gripper)")
    phase = np.zeros(len(order), dtype=np.int64)
    change = np.zeros(len(order), dtype=np.int64)
    # Require intact chronological episodes. Phase is a temporal quartile, NOT a skill label.
    edges = np.r_[0, np.flatnonzero(np.diff(result["episode_index"])) + 1, len(order)]
    if len(np.unique(result["episode_index"][edges[:-1]])) != len(edges) - 1:
        raise ValueError("Episodes must occupy contiguous ranges in the dataset")
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        positions = np.arange(left, right)
        episode = result["episode_index"][left]
        if not np.array_equal(result["frame_index"][positions], np.arange(len(positions))):
            raise ValueError(f"Non-contiguous frame numbering in episode {episode}")
        phase[positions] = np.minimum(3, np.arange(len(positions)) * 4 // len(positions))
        change[positions[1:]] = np.abs(np.diff(actions[positions, -1])) > 1e-6
    result.update(phase_bin=phase, gripper_change=change)
    return result


def create_plan(metadata, *, steps, batch_size, world_size, data_seed, within_seed):
    """Pick a finite pool once, then permute its occurrences only within each task."""
    for value in (steps, batch_size, world_size):
        if type(value) is not int or value < 1:
            raise ValueError("steps, batch_size and world_size must be positive integers")
    for value in (data_seed, within_seed):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError("Plan seeds must be integers in [0, 2**32)")
    length = len(metadata["index"])
    if not length:
        raise ValueError("Empty frame metadata")
    count = steps * batch_size * world_size
    pieces = []
    remaining, epoch = count, 0
    while remaining:
        seed = int(np.random.SeedSequence([data_seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        piece = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).numpy()[:remaining]
        pieces.append(piece)
        remaining -= len(piece)
        epoch += 1
    indices = np.concatenate(pieces)
    arrays = {key: np.asarray(metadata[key], dtype=np.int64)[indices] for key in META_COLUMNS}
    permutation = np.arange(count, dtype=np.int64)
    for task in np.unique(arrays["task_index"]):
        slots = np.flatnonzero(arrays["task_index"] == task)
        rng = np.random.default_rng(np.random.SeedSequence([9173, within_seed, int(task)]))
        permutation[slots] = rng.permutation(slots)
    manifest = {
        "protocol": PLAN_PROTOCOL,
        "flow_protocol": FLOW_PROTOCOL,
        "steps": steps,
        "batch_size": batch_size,
        "world_size": world_size,
        "data_seed": data_seed,
        "within_seed": within_seed,
        "dataset_frames": length,
        "dataset_metadata_sha256": arrays_hash(metadata),
        "instance_pool_sha256": arrays_hash(arrays),
        "task_schedule_sha256": arrays_hash({"task": arrays["task_index"]}),
        "scope": "New finite pool protocol; random W0/W1, not easy/hard curricula; no epoch-tail padding",
    }
    arrays["order"] = permutation
    manifest["arrays_sha256"] = arrays_hash(arrays)
    return {**arrays, "manifest": manifest}


def save_plan(path, plan):
    """Idempotent only for identical plans; never silently replace an experiment."""
    path = Path(path)
    if path.exists():
        if load_plan(path)["manifest"] != plan["manifest"]:
            raise ValueError(f"Different order plan already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **{**plan, "manifest": json.dumps(plan["manifest"], sort_keys=True)})


def load_plan(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in META_COLUMNS + ("order",)}
        manifest = json.loads(str(archive["manifest"]))
    if manifest.get("protocol") != PLAN_PROTOCOL or manifest.get("flow_protocol") != FLOW_PROTOCOL:
        raise ValueError("Unsupported order/flow plan protocol")
    count = manifest["steps"] * manifest["batch_size"] * manifest["world_size"]
    if any(value.dtype != np.int64 or value.shape != (count,) for value in result.values()):
        raise ValueError("Invalid order plan dimensions/dtypes")
    if not np.array_equal(np.sort(result["order"]), np.arange(count)):
        raise ValueError("Order must be a permutation of the SAME occurrence multiset")
    if not np.array_equal(result["task_index"][result["order"]], result["task_index"]):
        raise ValueError("Order changed a task slot")
    if arrays_hash(result) != manifest["arrays_sha256"]:
        raise ValueError("Order plan checksum mismatch")
    pool = {key: result[key] for key in META_COLUMNS}
    if arrays_hash(pool) != manifest["instance_pool_sha256"]:
        raise ValueError("Instance pool checksum mismatch")
    if arrays_hash({"task": result["task_index"]}) != manifest["task_schedule_sha256"]:
        raise ValueError("Task schedule checksum mismatch")
    if np.any(result["index"] < 0) or np.any(result["index"] >= manifest["dataset_frames"]):
        raise ValueError("Plan references invalid dataset indices")
    return {**result, "manifest": manifest}


def validate_plan_dataset(plan, metadata, *, steps, batch_size, world_size):
    manifest = plan["manifest"]
    if (steps, batch_size, world_size) != (manifest["steps"], manifest["batch_size"], manifest["world_size"]):
        raise ValueError("Plan requires the original steps, per-device batch size and GPU count")
    if arrays_hash(metadata) != manifest["dataset_metadata_sha256"]:
        raise ValueError("Dataset identities/action coverage metadata differ from order plan")
    for key in META_COLUMNS:
        if not np.array_equal(plan[key], metadata[key][plan["index"]]):
            raise ValueError(f"Plan {key} does not match the source dataset")


class PlannedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        index, occurrence = item
        result = dict(self.dataset[int(index)])
        if int(result["index"]) != index:
            raise ValueError("Dataset index mapping changed since order plan construction")
        result[OCCURRENCE_KEY] = int(occurrence)
        return result


class PlannedSampler:
    def __init__(self, plan, start_step=0):
        self.plan = plan
        manifest = plan["manifest"]
        if not 0 <= start_step <= manifest["steps"]:
            raise ValueError("Invalid order plan resume step")
        self.offset = start_step * manifest["batch_size"] * manifest["world_size"]

    def __len__(self):
        return len(self.plan["order"]) - self.offset

    def __iter__(self):
        for occurrence in self.plan["order"][self.offset :]:
            yield int(self.plan["index"][occurrence]), int(occurrence)
