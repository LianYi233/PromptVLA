"""CPU-only protocol construction and validation for controlled VLA curricula.

A score is an importance proxy, not measured difficulty or control risk. Held-out
means excluded from THIS fine-tuning occurrence pool, not unseen in pretraining.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "pi05-curriculum-study-v1"
META = ("index", "task_index", "episode_index", "frame_index", "phase_bin", "gripper_change")
ORDERS = ("H2E", "E2H", "RND")
PLAN_PROTOCOL = "pdet-fixed-task-slots-v1"
FLOW_PROTOCOL = "pdet-occurrence-flow-v1"


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def json_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def arrays_hash(arrays: dict[str, np.ndarray]) -> str:
    """Byte-for-byte compatible with lerobot.utils.pdet_order.arrays_hash."""
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        h.update(key.encode())
        h.update(str((value.dtype.str, value.shape)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def write_json(path: str | Path, value: Any, *, exclusive: bool = False) -> None:
    """Atomic publication; exclusive writes never replace an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temp, path)  # atomic no-replace on the same filesystem
        else:
            os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_archive(path: str | Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        info = json.loads(str(archive["manifest"]))
        arrays = {key: archive[key].copy() for key in archive.files if key != "manifest"}
    if arrays_hash(arrays) != info["arrays_sha256"]:
        raise ValueError(f"Array checksum mismatch: {path}")
    return arrays, info


def load_plan(path: str | Path) -> dict:
    arrays, info = load_archive(path)
    if info.get("protocol") != PLAN_PROTOCOL or info.get("flow_protocol") != FLOW_PROTOCOL:
        raise ValueError("Unsupported PDET plan protocol")
    n = info["steps"] * info["batch_size"] * info["world_size"]
    if set(arrays) != set(META) | {"order"}:
        raise ValueError("Unexpected plan columns")
    if any(a.dtype != np.int64 or a.shape != (n,) for a in arrays.values()):
        raise ValueError("Invalid plan shape/dtype")
    if not np.array_equal(np.sort(arrays["order"]), np.arange(n)):
        raise ValueError("Order is not a permutation of the occurrence pool")
    if not np.array_equal(arrays["task_index"][arrays["order"]], arrays["task_index"]):
        raise ValueError("Task slots changed")
    if arrays_hash({key: arrays[key] for key in META}) != info["instance_pool_sha256"]:
        raise ValueError("Occurrence-pool checksum mismatch")
    if arrays_hash({"task": arrays["task_index"]}) != info["task_schedule_sha256"]:
        raise ValueError("Task schedule checksum mismatch")
    if np.any(arrays["index"] < 0) or np.any(arrays["index"] >= info["dataset_frames"]):
        raise ValueError("Out-of-range dataset index")
    return {**arrays, "manifest": info}


def save_plan(path: str | Path, plan: dict) -> None:
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **{key: plan[key] for key in (*META, "order")},
                            manifest=json.dumps(plan["manifest"], sort_keys=True))
    load_plan(path)


def load_scores(path: str | Path) -> tuple[dict, dict]:
    arrays, info = load_archive(path)
    n = len(arrays["index"])
    if not np.array_equal(arrays["index"], np.arange(n)):
        raise ValueError("Scores require contiguous global frame indices")
    if any(key not in arrays or arrays[key].shape != (n,) for key in META):
        raise ValueError("Missing score identities")
    if arrays_hash({key: arrays[key] for key in META}) != info["dataset_metadata_sha256"]:
        raise ValueError("Score metadata checksum mismatch")
    if arrays["score"].shape != (n,) or not np.isfinite(arrays["score"]).all():
        raise ValueError("Invalid importance scores")
    return arrays, info


def positive_int(value: int, name: str, *, zero: bool = False) -> int:
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _cycles(values: np.ndarray, count: int, seed: list[int]) -> np.ndarray:
    if len(values) == 0 or count < 1:
        raise ValueError("Empty sampling group or budget")
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    pieces = []
    while count:
        piece = rng.permutation(values)[:count]
        pieces.append(piece)
        count -= len(piece)
    return np.concatenate(pieces)


def make_family(base: dict, scores: dict, score_info: dict, *, steps: int = 6000,
                easy_percent: int = 30, holdout_per_task: int = 2,
                probe_per_group: int = 2, seed: int = 0, warmup_steps: int = 200,
                independent_labels: dict[int, str] | None = None) -> tuple[dict, dict]:
    """Reserve whole episodes, then pair all sample uses and random-flow keys.

    All orders share an identical random warmup PREFIX, and permute only the
    remaining occurrences within fixed task slots. No sample/time/noise re-draw
    is caused by changing order or LR. Low/high are defined within task using
    TRAINING candidate scores, with index tie-breaking recorded explicitly.
    """
    for name, val in (("steps", steps), ("holdout_per_task", holdout_per_task),
                      ("probe_per_group", probe_per_group)):
        positive_int(val, name)
    positive_int(seed, "seed", zero=True)
    positive_int(warmup_steps, "warmup_steps", zero=True)
    if not 1 <= easy_percent <= 99 or type(easy_percent) is not int or warmup_steps >= steps:
        raise ValueError("Invalid exposure ratio or warmup budget")
    info = base["manifest"]
    if score_info["dataset_metadata_sha256"] != info["dataset_metadata_sha256"]:
        raise ValueError("Scores and baseline refer to different datasets")
    for key in META:
        if not np.array_equal(base[key], scores[key][base["index"]]):
            raise ValueError(f"Baseline metadata differs: {key}")
    if independent_labels is not None and (not independent_labels or
            set(independent_labels.values()) != {"critical", "routine"}):
        raise ValueError("Independent labels must contain both critical and routine")
    # Mixed-task episodes would make per-task holdout selection ambiguous.
    episode_tasks: dict[int, int] = {}
    for ep, task in zip(scores["episode_index"], scores["task_index"], strict=True):
        ep, task = int(ep), int(task)
        if episode_tasks.setdefault(ep, task) != task:
            raise ValueError(f"Episode {ep} contains multiple tasks")
    tasks = np.unique(base["task_index"])
    heldout: list[int] = []
    for task in tasks:
        eps = np.unique(base["episode_index"][base["task_index"] == task])
        if len(eps) <= holdout_per_task:
            raise ValueError(f"Task {task}: not enough episodes for train/holdout")
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(task), 3107]))
        heldout.extend(map(int, rng.permutation(eps)[:holdout_per_task]))
    heldout = sorted(heldout)
    test_mask = np.isin(scores["episode_index"], heldout)
    batch = info["batch_size"] * info["world_size"]
    size, warm = steps * batch, warmup_steps * batch
    task_slots = np.resize(base["task_index"], size)
    tasks, counts = np.unique(task_slots, return_counts=True)
    quotas = counts * easy_percent // 100
    target = (size * easy_percent + 50) // 100
    remainder = int(target - quotas.sum())
    quotas[np.argsort(-(counts * easy_percent % 100), kind="stable")[:remainder]] += 1
    indices = np.empty(size, dtype=np.int64)
    is_low = np.zeros(len(scores["index"]), dtype=bool)
    samples, boundaries, budgets = [], [], []
    for task, count, quota in zip(tasks, counts, quotas, strict=True):
        task = int(task)
        slots = np.flatnonzero(task_slots == task)
        candidates = base["index"][base["task_index"] == task]
        candidates = candidates[~test_mask[candidates]]
        unique = np.unique(candidates)
        if len(unique) < 2 or np.ptp(scores["score"][unique]) == 0:
            raise ValueError(f"Task {task}: insufficient training score variation")
        ranked = unique[np.lexsort((unique, scores["score"][unique]))]
        split = len(ranked) // 2
        is_low[ranked[:split]] = True
        edge = int(ranked[split - 1])
        edge_score = float(scores["score"][edge])
        low_candidates = candidates[is_low[candidates]]
        high_candidates = candidates[~is_low[candidates]]
        chosen = np.concatenate((
            _cycles(low_candidates, int(quota), [seed, task, 101]),
            _cycles(high_candidates, int(count - quota), [seed, task, 102])))
        rng = np.random.default_rng(np.random.SeedSequence([seed, task, 103]))
        indices[slots] = rng.permutation(chosen)
        boundaries.append({"task_index": task, "last_low_score": edge_score,
                           "last_low_index": edge,
                           "boundary_tied": bool(edge_score == scores["score"][ranked[split]])})
        budgets.append({"task_index": task, "low_uses": int(quota),
                        "high_uses": int(count - quota)})
        test = np.flatnonzero(test_mask & (scores["task_index"] == task))
        if independent_labels is None:
            low = ((scores["score"][test] < edge_score) |
                   ((scores["score"][test] == edge_score) & (test <= edge)))
            groups = {"low_score": test[low], "high_score": test[~low]}
        else:
            groups = {label: np.array([i for i in test if independent_labels.get(int(i)) == label],
                                     dtype=np.int64)
                      for label in ("routine", "critical")}
        for gi, (label, choices) in enumerate(groups.items()):
            if len(choices) < probe_per_group:
                raise ValueError(f"Task {task}: only {len(choices)} held-out {label} frames; "
                                 "increase --holdout-per-task or supply more independent labels")
            rng = np.random.default_rng(np.random.SeedSequence([seed, task, gi, 104]))
            for index in sorted(map(int, rng.choice(choices, probe_per_group, replace=False))):
                samples.append({"sample_index": len(samples), "dataset_index": index,
                                **{key: int(scores[key][index]) for key in META},
                                "group": label, "importance": float(scores["score"][index])})
    arrays = {key: np.asarray(scores[key][indices], dtype=np.int64) for key in META}
    common = {"protocol": PLAN_PROTOCOL, "flow_protocol": FLOW_PROTOCOL, "steps": steps,
              "batch_size": info["batch_size"], "world_size": info["world_size"],
              "data_seed": info["data_seed"], "within_seed": 0,
              "dataset_frames": info["dataset_frames"],
              "dataset_metadata_sha256": info["dataset_metadata_sha256"],
              "instance_pool_sha256": arrays_hash(arrays),
              "task_schedule_sha256": arrays_hash({"task": task_slots})}
    family = {}
    for direction in ORDERS:
        order = np.arange(size, dtype=np.int64)
        for task in tasks:
            slots = np.flatnonzero((task_slots == task) & (np.arange(size) >= warm))
            if direction == "RND":
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(task), 105]))
                order[slots] = rng.permutation(slots)
            else:
                sign = -1 if direction == "H2E" else 1
                group = (~is_low[indices[slots]]).astype(np.int64)
                order[slots] = slots[np.lexsort((slots, sign * scores["score"][indices[slots]],
                                                sign * group))]
        values = {**arrays, "order": order}
        family[direction] = {**values, "manifest": {
            **common, "arrays_sha256": arrays_hash(values),
            "study": {"protocol": SCHEMA, "order": direction, "seed": seed,
                      "easy_percent": easy_percent, "common_warmup_steps": warmup_steps,
                      "heldout_episodes": heldout, "budgets": budgets,
                      "score_arrays_sha256": score_info["arrays_sha256"]},
            "scope": "Same fine-tuning pool, task slots and occurrence keys; shared random warmup"}}
    probe = {"protocol": SCHEMA, "dataset_metadata_sha256": info["dataset_metadata_sha256"],
             "pool_sha256": common["instance_pool_sha256"],
             "score_arrays_sha256": score_info["arrays_sha256"], "seed": seed + 2026,
             "timesteps": [0.2, 0.5, 0.8], "noise_repeats": 2,
             "heldout_episodes": heldout, "boundaries": boundaries, "samples": samples,
             "group_definition": "independent_labels" if independent_labels is not None else "score_proxy",
             "scope": "Episodes excluded from these fine-tuning optimizer updates; not certified "
                      "unseen in source pretraining or preprocessing statistics"}
    probe["probe_id"] = json_hash(probe)
    validate_family(family, probe)
    return family, probe


def validate_family(family: dict, probe: dict) -> None:
    if set(family) != set(ORDERS):
        raise ValueError("Expected H2E/E2H/RND")
    if probe["probe_id"] != json_hash({k: v for k, v in probe.items() if k != "probe_id"}):
        raise ValueError("Probe checksum mismatch")
    ref = family["RND"]
    batch = ref["manifest"]["batch_size"] * ref["manifest"]["world_size"]
    warm = ref["manifest"]["study"]["common_warmup_steps"] * batch
    for label, plan in family.items():
        if plan["manifest"]["study"]["order"] != label:
            raise ValueError("Order label mismatch")
        for key in META:
            if not np.array_equal(plan[key], ref[key]):
                raise ValueError("Changed immutable occurrence pool")
        if not np.array_equal(plan["order"][:warm], ref["order"][:warm]):
            raise ValueError("Warmup differs")
        if np.intersect1d(plan["episode_index"], probe["heldout_episodes"]).size:
            raise ValueError("Probe episode leaked into training")
        if plan["manifest"]["instance_pool_sha256"] != probe["pool_sha256"]:
            raise ValueError("Probe belongs to another training pool")
        if not np.array_equal(plan["task_index"][plan["order"]], ref["task_index"]):
            raise ValueError("Changed task slots")
    if len({p["manifest"]["arrays_sha256"] for p in family.values()}) != 3:
        raise ValueError("Insufficient post-warmup budget to produce three distinct orders")
    for row in probe["samples"]:
        if row["episode_index"] not in probe["heldout_episodes"]:
            raise ValueError("Probe sample is not held out")


def read_labels(path: str | Path) -> dict[int, str]:
    result = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            index = int(row["index"])
            if index in result or row["group"] not in {"critical", "routine"}:
                raise ValueError("Duplicate index or invalid independent group label")
            result[index] = row["group"]
    return result


def lr_value(step: int, *, steps: int, warmup: int, peak: float, end: float) -> float:
    """Matches the repository's cosine_decay_with_warmup with decay_steps=steps.

    step is the zero-based optimizer update index, not a post-scheduler log index.
    """
    if warmup and step < warmup:
        return peak * (1 / (warmup + 1) + (1 - 1 / (warmup + 1)) * step / warmup)
    return end + (peak - end) * .5 * (1 + math.cos(math.pi * min(step, steps) / steps))
