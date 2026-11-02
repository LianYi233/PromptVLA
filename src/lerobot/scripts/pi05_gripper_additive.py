"""Compute-matched additive gripper-closure sampling for one LIBERO-long demonstration.

Every condition retains the EXACT original uniform base-plan sample sequence.
We insert extra training updates into matching slots: uniform extras for the
control; closure-neighborhood extras for the event treatment. No existing
sample is removed or replaced. Action targets remain contiguous 50-step chunks.
"""

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from lerobot.scripts.frameskip_scores import read_actions
from lerobot.scripts.pi05_gripper_event import (
    CASES,
    _inventory,
    case_config,
    read_tasks,
    resolve_task,
    select_episode,
)
from lerobot.utils.pdet_order import (
    FLOW_PROTOCOL,
    META_COLUMNS,
    PLAN_PROTOCOL,
    arrays_hash,
    load_plan,
    read_frame_metadata,
    save_plan,
    validate_plan_dataset,
)

PROTOCOL = "pi05-gripper-additive-long-v1"
DEFAULT_DIR = Path("/data1/wyn/prompt-work/gripper-event-additive-task8-r1")
DEFAULT_BASE_PLAN = Path("/data1/wyn/prompt-work/gripper-event-long-task8-r1/uniform.npz")


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def additive_slots(base_steps: int, extra_steps: int) -> np.ndarray:
    """False=original uniform batch; True=new extra batch, evenly interleaved."""
    if type(base_steps) is not int or base_steps < 1:
        raise ValueError("base_steps must be a positive integer")
    if type(extra_steps) is not int or extra_steps < 1:
        raise ValueError("extra_steps must be a positive integer")
    schedule = []
    added = 0
    for base_idx in range(base_steps):
        schedule.append(False)
        until = (base_idx + 1) * extra_steps // base_steps
        while added < until:
            schedule.append(True)
            added += 1
    result = np.asarray(schedule, dtype=np.bool_)
    if len(result) != base_steps + extra_steps or result.sum() != extra_steps:
        raise AssertionError("Additive schedule lost a training step")
    return result


def _validate_source_plan(base, metadata, selection, candidates, settings):
    """Require the existing r1 uniform plan, not a resampled approximation."""
    manifest = base["manifest"]
    old = manifest.get("gripper_event", {})
    if old.get("protocol") != "pi05-gripper-event-long-v1" or old.get("sampling") != "uniform":
        raise ValueError("--base-plan must point to a uniform.npz from the original gripper experiment")
    expected = (settings["base_steps"], settings["batch_size"], settings["world_size"])
    actual = (manifest["steps"], manifest["batch_size"], manifest["world_size"])
    if actual != expected:
        raise ValueError(f"Original uniform plan has budget {actual}; expected {expected}")
    if manifest.get("data_seed") != settings["data_seed"]:
        raise ValueError("Original uniform plan uses a different data seed")
    source_selection = old.get("selection", {})
    for field in ("task_id", "task_name", "episode_index", "chunk_size", "closure_frames"):
        if source_selection.get(field) != selection[field]:
            raise ValueError(f"Original uniform plan differs in {field}; preserve the same demonstration")
    validate_plan_dataset(
        base, metadata, steps=settings["base_steps"],
        batch_size=settings["batch_size"], world_size=settings["world_size"],
    )
    sample_indices = base["index"][base["order"]]
    if not np.isin(sample_indices, candidates).all():
        raise ValueError("Original uniform plan contains a start outside the selected episode")
    return sample_indices


def make_additive_plan(
    metadata,
    base_indices,
    candidates,
    event_mask,
    selection,
    *,
    base_steps,
    extra_steps,
    batch_size,
    world_size,
    data_seed,
    sampling,
    base_sha256,
):
    """Keep base-plan occurrences exactly; append only additive exposures."""
    if sampling not in ("uniform", "event"):
        raise ValueError("sampling must be uniform or event")
    slots = additive_slots(base_steps, extra_steps)
    per_step = batch_size * world_size
    base_indices = np.asarray(base_indices, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    event_mask = np.asarray(event_mask, dtype=bool)
    if base_indices.shape != (base_steps * per_step,):
        raise ValueError("Original uniform stream size does not match the training budget")
    if not len(candidates) or event_mask.shape != candidates.shape or not event_mask.any():
        raise ValueError("Need original episode candidates and at least one event-neighborhood start")
    if not np.isin(base_indices, candidates).all():
        raise ValueError("A base occurrence is outside the permitted episode")
    if type(data_seed) is not int or not 0 <= data_seed < 2**32:
        raise ValueError("Invalid data seed")

    # Common random numbers for extra slots. Only the eligible extra pool
    # differs; uniform/time and event/time share their exact respective file.
    uniforms = np.random.default_rng(
        np.random.SeedSequence([84521, data_seed])
    ).random(extra_steps * per_step)
    pool = candidates if sampling == "uniform" else candidates[event_mask]
    extra_indices = pool[np.minimum((uniforms * len(pool)).astype(np.int64), len(pool) - 1)]

    chosen = np.empty((base_steps + extra_steps, per_step), dtype=np.int64)
    chosen[~slots] = base_indices.reshape(base_steps, per_step)
    chosen[slots] = extra_indices.reshape(extra_steps, per_step)
    chosen = chosen.reshape(-1)

    arrays = {key: np.asarray(metadata[key][chosen], dtype=np.int64) for key in META_COLUMNS}
    arrays["order"] = np.arange(len(chosen), dtype=np.int64)
    manifest = {
        "protocol": PLAN_PROTOCOL,
        "flow_protocol": FLOW_PROTOCOL,
        "steps": base_steps + extra_steps,
        "batch_size": batch_size,
        "world_size": world_size,
        "data_seed": data_seed,
        "within_seed": 0,
        "dataset_frames": len(metadata["index"]),
        "dataset_metadata_sha256": arrays_hash(metadata),
        "instance_pool_sha256": arrays_hash({k: arrays[k] for k in META_COLUMNS}),
        "task_schedule_sha256": arrays_hash({"task": arrays["task_index"]}),
        "arrays_sha256": arrays_hash(arrays),
        "additive": {
            "protocol": PROTOCOL,
            "sampling": sampling,
            "selection": selection,
            "base_steps": base_steps,
            "extra_steps": extra_steps,
            "original_uniform_plan_sha256": base_sha256,
            "original_uniform_sequence_sha256": arrays_hash({"index": base_indices}),
            "additional_sequence_sha256": arrays_hash({"index": extra_indices}),
            "slot_schedule_sha256": arrays_hash({"slot_is_extra": slots.astype(np.int64)}),
            "additional_pool_size": int(len(pool)),
        },
        "scope": (
            "Original uniform occurrences retained in-order; "
            "extra full contiguous chunks in separate interleaved updates"
        ),
    }
    return {**arrays, "manifest": manifest}, slots


def _build(dataset_root, base_plan_path, settings):
    dataset_root = Path(dataset_root).resolve(strict=True)
    base_plan_path = Path(base_plan_path).resolve(strict=True)
    info = json.loads((dataset_root / "meta/info.json").read_text())
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset info must declare a finite positive FPS")

    metadata = read_frame_metadata(dataset_root)
    actions = read_actions(dataset_root)
    task = resolve_task(read_tasks(dataset_root), settings["task_id"])
    selection, candidates, _, event_mask, contains_closure = select_episode(
        metadata,
        actions,
        task,
        episode_index=settings["episode_index"],
        chunk_size=settings["chunk_size"],
        window_before=settings["window_before"],
        window_after=settings["window_after"],
        event_weight=3.0,  # Event definition only; extras use the eligible window directly.
        closed_direction=settings["closed_direction"],
        gripper_threshold=settings["gripper_threshold"],
        min_state_frames=settings["min_state_frames"],
    )
    base = load_plan(base_plan_path)
    base_indices = _validate_source_plan(base, metadata, selection, candidates, settings)
    base_sha = _sha256_file(base_plan_path)
    slots = additive_slots(settings["base_steps"], settings["extra_steps"])

    source_record = {
        "path": str(base_plan_path),
        "sha256": base_sha,
        "original_uniform_sequence_sha256": arrays_hash({"index": base_indices}),
        "original_steps": settings["base_steps"],
    }
    meta_hashes = {
        str(path.relative_to(dataset_root)): _sha256_file(path)
        for path in sorted((dataset_root / "meta").rglob("*")) if path.is_file()
    }
    record = {
        "protocol": PROTOCOL,
        "dataset_root": str(dataset_root),
        "fps": float(fps),
        "settings": settings,
        "selection": selection,
        "original_uniform_plan": source_record,
        "dataset_fingerprints": {
            "dataset_metadata_sha256": arrays_hash(metadata),
            "actions_sha256": arrays_hash({"action": actions}),
            "meta_sha256": meta_hashes,
            "file_inventory": _inventory(dataset_root),
        },
        "plans": {},
        "sampling_summary": {},
    }

    plans, counts = {}, {}
    for sampling in ("uniform", "event"):
        plan, actual_slots = make_additive_plan(
            metadata, base_indices, candidates, event_mask, selection,
            base_steps=settings["base_steps"],
            extra_steps=settings["extra_steps"],
            batch_size=settings["batch_size"],
            world_size=settings["world_size"],
            data_seed=settings["data_seed"],
            sampling=sampling, base_sha256=base_sha,
        )
        if not np.array_equal(slots, actual_slots):
            raise AssertionError("Different cases must share the same update schedule")
        plan_indices = plan["index"].reshape(len(slots), -1)
        if not np.array_equal(plan_indices[~slots].reshape(-1), base_indices):
            raise AssertionError("A base draw was removed, replaced, or reordered")
        extras = plan_indices[slots].reshape(-1)
        if sampling == "event" and not np.isin(extras, candidates[event_mask]).all():
            raise AssertionError("Event extras escaped the gripper-closure neighborhood")
        candidates_pos = np.searchsorted(candidates, plan["index"])
        extra_pos = np.searchsorted(candidates, extras)
        counts[sampling] = np.bincount(extra_pos, minlength=len(candidates))
        record["plans"][sampling] = plan["manifest"]
        record["sampling_summary"][sampling] = {
            "base_steps_unchanged": settings["base_steps"],
            "additional_steps": settings["extra_steps"],
            "total_steps": settings["steps"],
            "base_draws_unchanged": int(len(base_indices)),
            "extra_draws": int(len(extras)),
            "total_draws": int(len(plan["index"])),
            "base_near_closure_fraction": float(
                np.isin(base_indices, candidates[event_mask]).mean()
            ),
            "extra_near_closure_fraction": float(event_mask[extra_pos].mean()),
            "total_near_closure_fraction": float(event_mask[candidates_pos].mean()),
            "total_closure_in_chunk_fraction": float(contains_closure[candidates_pos].mean()),
            "additional_unique_starts": int(len(np.unique(extras))),
        }
        plans[sampling] = plan

    base_counts = np.bincount(np.searchsorted(candidates, base_indices), minlength=len(candidates))
    rows = [
        [
            int(global_index),
            int(metadata["frame_index"][global_index]),
            int(event_mask[i]),
            int(contains_closure[i]),
            int(base_counts[i]),
            int(counts["uniform"][i]),
            int(counts["event"][i]),
            int(base_counts[i] + counts["uniform"][i]),
            int(base_counts[i] + counts["event"][i]),
        ]
        for i, global_index in enumerate(candidates)
    ]
    return record, plans, rows


def prepare(
    dataset_root,
    output,
    *,
    base_plan=DEFAULT_BASE_PLAN,
    task_id=8,
    episode_index=10,
    base_steps=3000,
    extra_steps=750,
    batch_size=32,
    world_size=2,
    data_seed=0,
    chunk_size=50,
    window_before=5,
    window_after=5,
    closed_direction="increasing",
    gripper_threshold=None,
    min_state_frames=2,
):
    for name, value in (("base_steps", base_steps), ("extra_steps", extra_steps),
                        ("batch_size", batch_size), ("world_size", world_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    output = Path(output).resolve()
    dataset_root = Path(dataset_root).resolve()
    base_plan = Path(base_plan).resolve()
    settings = {
        "task_id": task_id,
        "episode_index": episode_index,
        "base_steps": base_steps,
        "extra_steps": extra_steps,
        "steps": base_steps + extra_steps,
        "batch_size": batch_size,
        "world_size": world_size,
        "data_seed": data_seed,
        "chunk_size": chunk_size,
        "window_before": window_before,
        "window_after": window_after,
        "closed_direction": closed_direction,
        "gripper_threshold": gripper_threshold,
        "min_state_frames": min_state_frames,
    }
    if output.exists():
        current = check(output)
        if (current["settings"] != settings or
                current["dataset_root"] != str(dataset_root) or
                current["original_uniform_plan"]["path"] != str(base_plan)):
            raise ValueError("Existing additive preparation differs; choose a new output directory")
        return current
    record, plans, rows = _build(dataset_root, base_plan, settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gripper-additive-", dir=output.parent) as scratch:
        temporary = Path(scratch)
        for sampling, plan in plans.items():
            save_plan(temporary / f"{sampling}.npz", plan)
        with (temporary / "anchors.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "global_index", "frame_index", "near_closure", "closure_in_chunk",
                "original_uniform_draws", "uniform_extra_draws", "event_extra_draws",
                "uniform_total_draws", "event_total_draws",
            ])
            writer.writerows(rows)
        record["anchors_csv_sha256"] = _sha256_file(temporary / "anchors.csv")
        (temporary / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        if output.exists():
            raise FileExistsError("Concurrent preparation created this directory")
        temporary.rename(output)
    return record


def check(directory):
    directory = Path(directory).resolve()
    record = json.loads((directory / "manifest.json").read_text())
    if record.get("protocol") != PROTOCOL:
        raise ValueError("Not an additive gripper-event preparation")
    expected, plans, _ = _build(
        record["dataset_root"], record["original_uniform_plan"]["path"], record["settings"]
    )
    if {k: v for k, v in record.items() if k != "anchors_csv_sha256"} != expected:
        raise ValueError("Source dataset, original uniform plan, or additive recipe changed")
    if _sha256_file(directory / "anchors.csv") != record["anchors_csv_sha256"]:
        raise ValueError("Sampling diagnostics changed")
    for sampling, expected_plan in plans.items():
        actual = load_plan(directory / f"{sampling}.npz")
        if actual["manifest"] != expected_plan["manifest"] or not np.array_equal(
            actual["index"], expected_plan["index"]
        ):
            raise ValueError(f"Prepared {sampling} sampling plan differs from recorded recipe")
    return record


def prefix(directory_or_record, case, repeat=1):
    case_config(case)
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be a positive run label")
    record = directory_or_record if isinstance(directory_or_record, dict) else check(directory_or_record)
    selection, settings = record["selection"], record["settings"]
    return (
        f"pi05-gripper-additive-r{repeat}-{case}-task{selection['task_id']}"
        f"-ep{selection['episode_index']}-base{settings['base_steps']}"
        f"-extra{settings['extra_steps']}-steps{settings['steps']}-aexpert"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="keep uniform base; add compute-matched extra starts")
    prep.add_argument("--dataset-root", type=Path, default=Path("/home/wyn/datasets/libero"))
    prep.add_argument("--output", type=Path, default=DEFAULT_DIR)
    prep.add_argument("--base-plan", type=Path, default=DEFAULT_BASE_PLAN)
    for name, default in (
        ("task-id", 8), ("base-steps", 3000), ("extra-steps", 750),
        ("batch-size", 32), ("world-size", 2), ("data-seed", 0),
        ("chunk-size", 50), ("window-before", 5), ("window-after", 5),
        ("min-state-frames", 2),
    ):
        prep.add_argument(f"--{name}", type=int, default=default)
    prep.add_argument("--episode-index", type=int, default=10)
    prep.add_argument("--closed-direction", choices=("increasing", "decreasing"), default="increasing")
    prep.add_argument("--gripper-threshold", type=float)
    checker = sub.add_parser("check")
    checker.add_argument("directory", type=Path)
    namer = sub.add_parser("prefix")
    namer.add_argument("directory", type=Path)
    namer.add_argument("case", choices=CASES)
    namer.add_argument("--repeat", type=int, default=1)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prefix":
        print(prefix(args["directory"], args["case"], args["repeat"]))
    else:
        record = prepare(**args) if command == "prepare" else check(args["directory"])
        print(json.dumps({
            "status": "prepared" if command == "prepare" else "checked",
            "fps": record["fps"],
            "selection": record["selection"],
            "settings": record["settings"],
            "original_uniform_plan": record["original_uniform_plan"],
            "sampling_summary": record["sampling_summary"],
        }, indent=2))


if __name__ == "__main__":
    main()
