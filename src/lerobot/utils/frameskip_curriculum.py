"""FrameSkip-inspired importance and finite, occurrence-bound curricula.

This is an adaptation, not a reproduction: frozen PI05 SigLIP features and a
local gripper-event prior replace DINOv2 and annotated task progress. Scores
select chunk STARTS; the dataset must still return original contiguous actions.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.utils.pdet_order import META_COLUMNS, arrays_hash, create_plan, load_plan, save_plan

PROTOCOL = "promptvla-frameskip-curriculum-v1"
SCHEDULES = ("U", "R", "S", "C")


@dataclass(frozen=True)
class ScoreConfig:
    retention: float = 0.2
    action_weight: float = 0.6
    visual_weight: float = 0.2
    event_weight: float = 0.2
    future_window: int = 3
    future_variance_weight: float = 0.1
    denominator_floor: float = 0.05
    event_before: int = 10
    event_after: int = 2
    debounce: int = 2
    random_seed: int = 2026

    def validate(self):
        if not 0 < self.retention < 1 or self.denominator_floor <= 0:
            raise ValueError("Invalid retention or visual denominator floor")
        weights = (self.action_weight, self.visual_weight, self.event_weight)
        if not np.isfinite(weights).all() or min(weights) < 0 or not np.isclose(sum(weights), 1):
            raise ValueError("Score weights must be nonnegative and sum to one")
        if min(self.future_window, self.debounce) < 1 or min(self.event_before, self.event_after) < 0:
            raise ValueError("Invalid scoring windows")


def robust_unit(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite score component")
    lo, hi = np.quantile(values, [0.01, 0.99])
    if hi - lo < 1e-12:
        # Sparse real events must not disappear because both quantiles are zero.
        lo, hi = float(values.min()), float(values.max())
    return np.zeros_like(values) if hi - lo < 1e-12 else (np.clip(values, lo, hi) - lo) / (hi - lo)


def gripper_events(values, threshold, debounce):
    """Mark the first frame of a new state only after a persistent transition."""
    states = np.asarray(values) > threshold
    events = np.zeros(len(states), dtype=bool)
    current = bool(states[0])
    for i in range(1, len(states)):
        if (
            states[i] != current
            and i + debounce <= len(states)
            and np.all(states[i : i + debounce] == states[i])
        ):
            events[i] = True
            current = bool(states[i])
    return events


def score_episode(
    actions, visual_delta, action_low, action_high, threshold, episode_id, config=ScoreConfig()
):
    config.validate()
    actions = np.asarray(actions, dtype=np.float64)
    visual_delta = np.asarray(visual_delta, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 1:
        raise ValueError("Expected one chronological episode of 7-D actions")
    if (
        visual_delta.shape != (len(actions), 2)
        or not np.isfinite(visual_delta).all()
        or np.any(visual_delta < 0)
    ):
        raise ValueError("Expected two finite nonnegative camera feature deltas per frame")
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite actions")
    low, high = np.asarray(action_low), np.asarray(action_high)
    if low.shape != (6,) or high.shape != (6,) or not np.isfinite([low, high]).all() or np.any(high < low):
        raise ValueError("Expected six valid training-data action quantiles")
    span = high - low
    span = np.where(span > 1e-6, span, 1.0)
    normalized = 2 * (actions[:, :6] - low) / span - 1
    delta = np.r_[0.0, np.linalg.norm(np.diff(normalized, axis=0), axis=1)]
    variance = np.zeros(len(actions))
    for i in range(len(actions) - 1):
        future = normalized[i + 1 : i + 1 + config.future_window]
        variance[i] = np.var(future, axis=0).mean() if len(future) > 1 else 0
    avi = delta + config.future_variance_weight * variance
    vac = visual_delta / np.maximum(delta[:, None], config.denominator_floor)
    # Cap each camera's ratio within the episode before combining views.
    vac = np.minimum(vac, np.quantile(vac, 0.95, axis=0))
    visual = robust_unit(vac.mean(axis=1))
    events = gripper_events(actions[:, -1], threshold, config.debounce)
    neighborhood = np.zeros(len(actions), dtype=bool)
    for i in np.flatnonzero(events):
        neighborhood[max(0, i - config.event_before) : i + config.event_after + 1] = True
    action = robust_unit(avi)
    score = config.action_weight * action + config.visual_weight * visual + config.event_weight * neighborhood
    protected = events.copy()
    protected[[0, -1]] = True
    quota = max(int(np.ceil(len(actions) * config.retention)), int(protected.sum()))
    available = np.flatnonzero(~protected)
    rng = np.random.default_rng(np.random.SeedSequence([config.random_seed, int(episode_id)]))
    # Deterministic random tie-breaking avoids selecting all early frames in flat episodes.
    selected = protected.copy()
    ranked = available[np.lexsort((rng.random(len(available)), -score[available]))]
    selected[ranked[: quota - protected.sum()]] = True
    random_selected = protected.copy()
    random_selected[rng.permutation(available)[: quota - protected.sum()]] = True
    return {
        "score": score,
        "action_score": action,
        "visual_score": visual,
        "action_delta": delta,
        "action_variance": variance,
        "avi": avi,
        "visual_delta_0": visual_delta[:, 0],
        "visual_delta_1": visual_delta[:, 1],
        "vac_0": vac[:, 0],
        "vac_1": vac[:, 1],
        "event": events,
        "event_neighborhood": neighborhood,
        "protected": protected,
        "selected": selected,
        "random_selected": random_selected,
    }


def save_scores(path, metadata, scores, provenance):
    arrays = {**metadata, **scores}
    manifest = {
        "protocol": PROTOCOL,
        "dataset_metadata_sha256": arrays_hash(metadata),
        "arrays_sha256": arrays_hash(arrays),
        **provenance,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _, old = load_scores(path)
        if old != manifest:
            raise ValueError(f"Different scoring archive exists: {path}")
        return
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays, manifest=json.dumps(manifest, sort_keys=True))


def load_scores(path):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        arrays = {key: archive[key] for key in archive.files if key != "manifest"}
    if manifest.get("protocol") != PROTOCOL or arrays_hash(arrays) != manifest["arrays_sha256"]:
        raise ValueError("Invalid/corrupted score archive")
    length = len(arrays["index"])
    if any(value.shape != (length,) for value in arrays.values()) or not np.isfinite(arrays["score"]).all():
        raise ValueError("Invalid scoring dimensions/values")
    if arrays_hash({key: arrays[key] for key in META_COLUMNS}) != manifest["dataset_metadata_sha256"]:
        raise ValueError("Score dataset identity mismatch")
    for key in ("protected", "selected", "random_selected"):
        if arrays[key].dtype != bool:
            raise ValueError(f"Expected boolean selection: {key}")
    for episode in np.unique(arrays["episode_index"]):
        positions = arrays["episode_index"] == episode
        a, b, protected = (arrays[key][positions] for key in ("selected", "random_selected", "protected"))
        if a.sum() != b.sum() or not np.all(a[protected]) or not np.all(b[protected]):
            raise ValueError("Random/scored selection must share episode counts and protected frames")
    return arrays, manifest


def dependent_round(probabilities, rng):
    """Fixed-size randomized selection with the specified marginal probabilities."""
    p = np.asarray(probabilities, dtype=np.float64).copy()
    target = int(round(p.sum()))
    if not np.isclose(p.sum(), target) or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must sum to an integer")
    active = list(rng.permutation(np.flatnonzero((p > 1e-10) & (p < 1 - 1e-10))))
    while len(active) > 1:
        a, b = active.pop(), active.pop()
        total = p[a] + p[b]
        if total < 1:
            p[a], p[b] = (total, 0) if rng.random() < p[a] / total else (0, total)
        else:
            chance = (1 - p[b]) / (2 - total) if total < 2 else 1
            p[a], p[b] = (1, total - 1) if rng.random() < chance else (total - 1, 1)
        active.extend(i for i in (a, b) if 1e-10 < p[i] < 1 - 1e-10)
    result = p > 0.5
    if int(result.sum()) != target:
        raise ValueError("Dependent rounding did not preserve the exact count")
    return result


def source_mask(plan):
    """Canonical pool: within each task, selected-source uses precede full-pool uses."""
    info = plan["manifest"]["curriculum"]
    mask = np.zeros(len(plan["order"]), dtype=bool)
    warm = info["warmup_occurrences"]
    for task, count in info["key_draws_per_task"].items():
        positions = np.flatnonzero(plan["task_index"][warm:] == int(task)) + warm
        mask[positions[:count]] = True
    return mask


def make_plans(
    metadata,
    scores,
    score_manifest,
    *,
    steps=3000,
    batch_size=32,
    world_size=2,
    warmup_steps=500,
    data_seed=0,
    sampling_seed=2026,
):
    if not 0 < warmup_steps < steps:
        raise ValueError("Warmup must be positive and shorter than training")
    if arrays_hash(metadata) != score_manifest["dataset_metadata_sha256"]:
        raise ValueError("Scores and dataset metadata differ")
    base = create_plan(
        metadata,
        steps=steps,
        batch_size=batch_size,
        world_size=world_size,
        data_seed=data_seed,
        within_seed=0,
    )
    count = len(base["order"])
    warm = warmup_steps * batch_size * world_size
    if (count - warm) % 2:
        raise ValueError("Post-warmup count must be even for exactly 50% selected-source uses")
    tasks = base["task_index"]
    task_ids, task_counts = np.unique(tasks[warm:], return_counts=True)
    key_counts = task_counts // 2
    # Allocate odd-task remainders once to preserve EXACTLY half of the global pool.
    odd = np.flatnonzero(task_counts % 2)
    rng = np.random.default_rng(sampling_seed)
    key_counts[rng.permutation(odd)[: (count - warm) // 2 - key_counts.sum()]] += 1
    budgets = dict(zip(map(int, task_ids), map(int, key_counts), strict=True))
    plans = {}
    for schedule in SCHEDULES:
        indices = base["index"][base["order"]].copy()
        for task in task_ids:
            positions = np.flatnonzero(tasks[warm:] == task) + warm
            full = np.flatnonzero(metadata["task_index"] == task)
            rng = np.random.default_rng(np.random.SeedSequence([sampling_seed, int(task), 731]))
            # Common full-pool draws in R/S/C; U uses only full-pool draws.
            full_draws = rng.choice(full, len(positions), replace=True)
            indices[positions] = full_draws
            k = budgets[int(task)] if schedule != "U" else 0
            if k:
                field = "random_selected" if schedule == "R" else "selected"
                selected = full[scores[field][full]]
                if not len(selected):
                    raise ValueError(f"Empty selected pool for task {task}")
                indices[positions[:k]] = rng.choice(selected, k, replace=True)
        arrays = {key: np.asarray(metadata[key][indices], dtype=np.int64) for key in META_COLUMNS}
        order = np.arange(count, dtype=np.int64)
        for task in task_ids:
            positions = np.flatnonzero(tasks[warm:] == task) + warm
            rng = np.random.default_rng(np.random.SeedSequence([sampling_seed, int(task), 932]))
            if schedule != "C":
                order[positions] = rng.permutation(positions)
                continue
            k = budgets[int(task)]
            raw = 0.2 + 0.6 * (positions - warm) / max(1, count - warm - 1)
            # Small per-task intercept correction gives the shared EXACT task budget.
            lo, hi = -1.0, 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if np.clip(raw + mid, 0, 1).sum() < k:
                    lo = mid
                else:
                    hi = mid
            selected_slots = dependent_round(np.clip(raw + (lo + hi) / 2, 0, 1), rng)
            order[positions[selected_slots]] = rng.permutation(positions[:k])
            order[positions[~selected_slots]] = rng.permutation(positions[k:])
        info = {
            "protocol": PROTOCOL,
            "schedule": schedule,
            "warmup_steps": warmup_steps,
            "warmup_occurrences": warm,
            "sampling_seed": sampling_seed,
            "key_draws_per_task": {str(t): budgets[t] if schedule != "U" else 0 for t in budgets},
            "score_arrays_sha256": score_manifest["arrays_sha256"],
            "score_recipe": score_manifest,
            "scope": "Chunk starts only; S/C share immutable sample uses, only order differs",
        }
        manifest = {
            **base["manifest"],
            "curriculum": info,
            "instance_pool_sha256": arrays_hash(arrays),
            "scope": "FrameSkip-inspired finite curriculum; full warmup; task slots fixed",
        }
        arrays["order"] = order
        manifest["arrays_sha256"] = arrays_hash(arrays)
        plans[schedule] = {**arrays, "manifest": manifest}
    validate_family(plans)
    return plans


def validate_family(plans):
    if set(plans) != set(SCHEDULES):
        raise ValueError("Expected exactly U/R/S/C plans")
    reference = plans["S"]
    warm = reference["manifest"]["curriculum"]["warmup_occurrences"]
    for label, plan in plans.items():
        if plan["manifest"]["curriculum"]["schedule"] != label:
            raise ValueError("Schedule label mismatch")
        for key in ("task_schedule_sha256", "dataset_metadata_sha256", "steps", "batch_size", "world_size"):
            if plan["manifest"][key] != reference["manifest"][key]:
                raise ValueError(f"Schedule family changed {key}")
        for key in META_COLUMNS:
            if not np.array_equal(plan[key][plan["order"][:warm]], reference[key][reference["order"][:warm]]):
                raise ValueError("Warmup inputs differ")
        if not np.array_equal(plan["order"][:warm], reference["order"][:warm]):
            raise ValueError("Warmup occurrence IDs differ")
        if (
            plan["manifest"]["curriculum"]["score_recipe"]
            != reference["manifest"]["curriculum"]["score_recipe"]
        ):
            raise ValueError("Score recipes differ")
    for key in META_COLUMNS:
        if not np.array_equal(plans["S"][key], plans["C"][key]):
            raise ValueError("S/C changed the sample-use multiset")
    if not np.array_equal(source_mask(plans["S"]), source_mask(plans["C"])):
        raise ValueError("S/C source identities differ")
    if np.array_equal(plans["S"]["order"], plans["C"]["order"]):
        raise ValueError("S/C ordering intervention had no effect")


def prepare_plans(scores_path, output, **kwargs):
    scores, manifest = load_scores(scores_path)
    metadata = {key: scores[key] for key in META_COLUMNS}
    plans = make_plans(metadata, scores, manifest, **kwargs)
    output = Path(output)
    for label, plan in plans.items():
        save_plan(output / f"{label}.npz", plan)
    validate_family({label: load_plan(output / f"{label}.npz") for label in SCHEDULES})
    return plans
