"""Prepare matched, single-demonstration LIBERO-long gripper-event sampling plans.

Only observation/action-chunk STARTS are resampled. Targets remain the original
contiguous chunk; neither event annotations nor episode time enter the policy.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np

from lerobot.scripts.frameskip_scores import read_actions
from lerobot.utils.pdet_order import (
    FLOW_PROTOCOL,
    META_COLUMNS,
    PLAN_PROTOCOL,
    arrays_hash,
    load_plan,
    read_frame_metadata,
    save_plan,
)

PROTOCOL = "pi05-gripper-event-long-v1"
CASES = ("uniform", "event", "uniform_time", "event_time")
DEFAULT_DIRECTORY = Path("/data1/wyn/prompt-work/gripper-event-long-r1")

# Official task_order_index=0. Match language exactly, never assume that a
# combined LeRobot dataset's task_index is a LIBERO-long suite task ID.
# Source: Lifelong-Robot-Learning/LIBERO,
# libero/libero/benchmark/libero_suite_task_map.py, blob
# 08144b4dd01d91fb0ca40e2c1d93ccaa85025fbc.
LONG_TASKS = (
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
)


def canonical_language(value):
    value = re.sub(r"^(?:LIVING_ROOM|KITCHEN|STUDY)_SCENE\d+_", "", str(value), flags=re.IGNORECASE)
    return " ".join(value.replace("_", " ").lower().strip().split())


def case_config(case):
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose one of {CASES}")
    return {"sampling": case.split("_", 1)[0], "time_encoding": case.endswith("_time")}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tasks(root):
    """Read only local task metadata; do not import the simulator or download data."""
    root = Path(root)
    path = root / "meta/tasks.parquet"
    if path.is_file():
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        names = [key for key in ("task", "__index_level_0__", "task_name") if key in table.column_names]
        if len(names) != 1 or "task_index" not in table.column_names:
            raise ValueError("Expected task names and task_index in meta/tasks.parquet")
        pairs = zip(table["task_index"].to_pylist(), table[names[0]].to_pylist(), strict=True)
    else:
        path = root / "meta/tasks.jsonl"
        if not path.is_file():
            raise FileNotFoundError("Need meta/tasks.parquet or meta/tasks.jsonl to identify LIBERO-long")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        pairs = ((row["task_index"], row["task"]) for row in rows)
    tasks = {}
    for task_index, name in pairs:
        if not isinstance(task_index, int) or task_index < 0 or not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid task metadata")
        if task_index in tasks:
            raise ValueError("Duplicate task_index in task metadata")
        tasks[task_index] = name
    if not tasks:
        raise ValueError("Empty task metadata")
    return tasks


def resolve_task(tasks, task_id=0):
    if type(task_id) is not int or not 0 <= task_id < len(LONG_TASKS):
        raise ValueError("task_id must be a LIBERO-long suite ID in [0, 9]")
    language = canonical_language(LONG_TASKS[task_id])
    matches = [index for index, value in tasks.items() if canonical_language(value) == language]
    if len(matches) != 1:
        raise ValueError(
            f"LIBERO-long task {task_id} requires one exact task-language match; got {matches}: {language}"
        )
    return {
        "suite": "libero_10",
        "task_id": task_id,
        "task_index": matches[0],
        "task_name": LONG_TASKS[task_id],
        "task_language": language,
        "dataset_task_name": tasks[matches[0]],
        "task_order_index": 0,
    }


def closure_events(values, *, closed_direction="increasing", threshold=None, min_state_frames=2):
    """Return first frame of each stable open -> closed COMMAND transition.

    A sustained closed segment contributes one event. Initial closed state and
    short chatter do not contribute events. Commands do not prove object contact.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Gripper commands must be a nonempty finite vector")
    if closed_direction not in ("increasing", "decreasing"):
        raise ValueError("closed_direction must be increasing or decreasing")
    if type(min_state_frames) is not int or min_state_frames < 1:
        raise ValueError("min_state_frames must be a positive integer")
    if threshold is None:
        unique = np.unique(values)
        if np.all(np.isin(unique, [0, 1])):
            threshold = 0.5
        elif np.all(np.isin(unique, [-1, 1])):
            threshold = 0.0
        else:
            raise ValueError("Nonbinary gripper commands require an explicit --gripper-threshold")
    if not np.isfinite(threshold):
        raise ValueError("Gripper threshold must be finite")
    closed = values >= threshold if closed_direction == "increasing" else values <= threshold
    edges = np.r_[0, np.flatnonzero(closed[1:] != closed[:-1]) + 1, len(values)]
    stable_state = None
    events = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if right - left < min_state_frames:
            continue
        state = bool(closed[left])
        if stable_state is False and state:
            events.append(int(left))
        stable_state = state
    return np.asarray(events, dtype=np.int64), float(threshold)


def select_episode(
    metadata,
    actions,
    task,
    *,
    episode_index=None,
    chunk_size=50,
    window_before=5,
    window_after=5,
    event_weight=3.0,
    closed_direction="increasing",
    gripper_threshold=None,
    min_state_frames=2,
):
    for name, value, minimum in (
        ("chunk_size", chunk_size, 1),
        ("window_before", window_before, 0),
        ("window_after", window_after, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {name}")
    if not math.isfinite(event_weight) or event_weight <= 1:
        raise ValueError("event_weight must be finite and > 1")
    if episode_index is not None and (type(episode_index) is not int or episode_index < 0):
        raise ValueError("episode_index must be a nonnegative integer")
    candidates = np.flatnonzero(metadata["task_index"] == task["task_index"])
    episodes = sorted(np.unique(metadata["episode_index"][candidates]).tolist())
    if episode_index is not None:
        if episode_index not in episodes:
            raise ValueError("Requested episode does not belong to the selected LIBERO-long task")
        episodes = [episode_index]
    reasons = []
    for episode in episodes:
        positions = np.flatnonzero(metadata["episode_index"] == episode)
        if (
            not np.all(metadata["task_index"][positions] == task["task_index"])
            or not np.array_equal(metadata["frame_index"][positions], np.arange(len(positions)))
            or not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1))
        ):
            raise ValueError(f"Episode {episode} must have one task and contiguous chronological frames")
        events, threshold = closure_events(
            actions[positions, -1],
            closed_direction=closed_direction,
            threshold=gripper_threshold,
            min_state_frames=min_state_frames,
        )
        starts = positions[: max(0, len(positions) - chunk_size + 1)]
        frames = metadata["frame_index"][starts]
        event_anchor = np.zeros(len(starts), dtype=bool)
        contains_closure = np.zeros(len(starts), dtype=bool)
        for event in events:
            event_anchor |= (frames >= event - window_before) & (frames <= event + window_after)
            contains_closure |= (frames <= event) & (frames + chunk_size > event)
        if not len(events) or not len(starts) or not event_anchor.any() or event_anchor.all():
            reasons.append(
                f"episode {episode}: {len(events)} closures, {len(starts)} full chunks, {event_anchor.sum()} near-event starts"
            )
            continue
        weights = np.where(event_anchor, float(event_weight), 1.0)
        selection = {
            **task,
            "episode_index": int(episode),
            "episode_length": len(positions),
            "episode_first_index": int(positions[0]),
            "episode_last_index": int(positions[-1]),
            "chunk_size": chunk_size,
            "num_candidate_anchors": len(starts),
            "candidate_first_index": int(starts[0]),
            "candidate_last_index": int(starts[-1]),
            "closure_frames": events.tolist(),
            "closure_global_indices": positions[events].tolist(),
            "gripper_values": np.unique(actions[positions, -1]).astype(float).tolist(),
            "gripper_threshold": threshold,
            "closed_direction": closed_direction,
            "anchor_event_fraction": float(event_anchor.mean()),
            "selection_rule": "explicit_episode"
            if episode_index is not None
            else "first_eligible_episode_in_dataset_order",
        }
        return selection, starts, weights, event_anchor, contains_closure
    raise ValueError(
        "No eligible episode: need stable open-to-closed events and both near-event and ordinary complete chunks. "
        "Check gripper direction/threshold, task, episode or window. " + "; ".join(reasons[:8])
    )


def make_plan(metadata, indices, weights, selection, *, steps, batch_size, world_size, data_seed, sampling):
    for name, value in (("steps", steps), ("batch_size", batch_size), ("world_size", world_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(data_seed) is not int or not 0 <= data_seed < 2**32:
        raise ValueError("data_seed must be an integer in [0, 2**32)")
    if sampling not in ("uniform", "event"):
        raise ValueError("sampling must be uniform or event")
    indices, weights = np.asarray(indices), np.asarray(weights, dtype=np.float64)
    if (
        indices.ndim != 1
        or not len(indices)
        or weights.shape != indices.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("Need positive finite weights for every candidate anchor")
    probabilities = np.ones(len(indices), dtype=np.float64) if sampling == "uniform" else weights.copy()
    probabilities /= probabilities.sum()
    # Common uniform random numbers couple the two distributions without changing
    # the global RNG. Time-on/off use the identical saved file for each sampler.
    uniforms = np.random.default_rng(np.random.SeedSequence([42073, data_seed])).random(
        steps * batch_size * world_size
    )
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    chosen = indices[np.searchsorted(cdf, uniforms, side="right")]
    arrays = {key: np.asarray(metadata[key][chosen], dtype=np.int64) for key in META_COLUMNS}
    manifest = {
        "protocol": PLAN_PROTOCOL,
        "flow_protocol": FLOW_PROTOCOL,
        "steps": steps,
        "batch_size": batch_size,
        "world_size": world_size,
        "data_seed": data_seed,
        "within_seed": 0,
        "dataset_frames": len(metadata["index"]),
        "dataset_metadata_sha256": arrays_hash(metadata),
        "instance_pool_sha256": arrays_hash(arrays),
        "task_schedule_sha256": arrays_hash({"task": arrays["task_index"]}),
        "gripper_event": {
            "protocol": PROTOCOL,
            "sampling": sampling,
            "selection": selection,
            "candidate_probability_sha256": arrays_hash({"index": indices, "probability": probabilities}),
        },
        "scope": "One LIBERO-long demonstration; weighted chunk STARTS with replacement; contiguous full targets",
    }
    arrays["order"] = np.arange(len(chosen), dtype=np.int64)
    manifest["arrays_sha256"] = arrays_hash(arrays)
    return {**arrays, "manifest": manifest}, probabilities


def _inventory(root):
    root = Path(root)
    return [
        [str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns]
        for directory in ("meta", "data", "videos")
        for path in sorted((root / directory).rglob("*"))
        if path.is_file()
    ]


def _build(dataset_root, settings):
    dataset_root = Path(dataset_root).resolve()
    info = json.loads((dataset_root / "meta/info.json").read_text())
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset meta/info.json must declare a finite positive FPS")
    metadata, actions = read_frame_metadata(dataset_root), read_actions(dataset_root)
    tasks = read_tasks(dataset_root)
    task = resolve_task(tasks, settings["task_id"])
    selection, indices, weights, event_anchor, contains_closure = select_episode(
        metadata,
        actions,
        task,
        **{
            key: settings[key]
            for key in (
                "episode_index",
                "chunk_size",
                "window_before",
                "window_after",
                "event_weight",
                "closed_direction",
                "gripper_threshold",
                "min_state_frames",
            )
        },
    )
    if "total_frames" in info and info["total_frames"] != len(metadata["index"]):
        raise ValueError("Dataset info total_frames differs from the complete parquet table")
    meta_hashes = {
        str(path.relative_to(dataset_root)): _digest(path)
        for path in sorted((dataset_root / "meta").rglob("*"))
        if path.is_file()
    }
    record = {
        "protocol": PROTOCOL,
        "dataset_root": str(dataset_root),
        "fps": float(fps),
        "settings": settings,
        "selection": selection,
        "dataset_fingerprints": {
            "dataset_metadata_sha256": arrays_hash(metadata),
            "actions_sha256": arrays_hash({"action": actions}),
            "meta_sha256": meta_hashes,
            "file_inventory": _inventory(dataset_root),
            "scope": "Exact metadata/action hashes; data/video file size and mtime inventory (not video content hashes)",
        },
        "plans": {},
        "sampling_summary": {},
    }
    plans, probabilities, counts = {}, {}, {}
    for sampling in ("uniform", "event"):
        plan, probability = make_plan(
            metadata,
            indices,
            weights,
            selection,
            sampling=sampling,
            **{key: settings[key] for key in ("steps", "batch_size", "world_size", "data_seed")},
        )
        positions = np.searchsorted(indices, plan["index"])
        plans[sampling], probabilities[sampling] = plan, probability
        counts[sampling] = np.bincount(positions, minlength=len(indices))
        record["plans"][sampling] = plan["manifest"]
        record["sampling_summary"][sampling] = {
            "event_anchor_probability": float(probability[event_anchor].sum()),
            "closure_in_chunk_probability": float(probability[contains_closure].sum()),
            "realized_event_anchor_fraction": float(event_anchor[positions].mean()),
            "realized_closure_in_chunk_fraction": float(contains_closure[positions].mean()),
            "draws": len(positions),
            "unique_sampled_anchors": len(np.unique(positions)),
        }
    rows = [
        [
            int(index),
            int(metadata["frame_index"][index]),
            int(event_anchor[i]),
            int(contains_closure[i]),
            float(weights[i]),
            float(probabilities["uniform"][i]),
            float(probabilities["event"][i]),
            int(counts["uniform"][i]),
            int(counts["event"][i]),
        ]
        for i, index in enumerate(indices)
    ]
    return record, plans, rows


def prepare(
    dataset_root,
    output,
    *,
    task_id=0,
    episode_index=None,
    steps=3000,
    batch_size=32,
    world_size=2,
    data_seed=0,
    chunk_size=50,
    window_before=5,
    window_after=5,
    event_weight=3.0,
    closed_direction="increasing",
    gripper_threshold=None,
    min_state_frames=2,
):
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    settings = {
        "task_id": task_id,
        "episode_index": episode_index,
        "steps": steps,
        "batch_size": batch_size,
        "world_size": world_size,
        "data_seed": data_seed,
        "chunk_size": chunk_size,
        "window_before": window_before,
        "window_after": window_after,
        "event_weight": event_weight,
        "closed_direction": closed_direction,
        "gripper_threshold": gripper_threshold,
        "min_state_frames": min_state_frames,
    }
    if output.exists():
        record = check(output)
        if record["settings"] != settings or record["dataset_root"] != str(dataset_root):
            raise ValueError(
                "Existing experiment has different inputs/settings; choose a new output directory"
            )
        return record
    record, plans, rows = _build(dataset_root, settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gripper-long-", dir=output.parent) as temporary:
        temp = Path(temporary)
        for sampling, plan in plans.items():
            save_plan(temp / f"{sampling}.npz", plan)
        with (temp / "anchors.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "global_index",
                    "frame_index",
                    "near_closure",
                    "closure_in_chunk",
                    "event_weight",
                    "uniform_probability",
                    "event_probability",
                    "uniform_draws",
                    "event_draws",
                ]
            )
            writer.writerows(rows)
        record["anchors_csv_sha256"] = _digest(temp / "anchors.csv")
        (temp / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        if output.exists():
            raise FileExistsError("Preparation raced; rerun to validate the completed experiment")
        temp.rename(output)
    return record


def check(directory):
    directory = Path(directory).resolve()
    record = json.loads((directory / "manifest.json").read_text())
    if record.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported gripper-event experiment protocol")
    expected, plans, _ = _build(record["dataset_root"], record["settings"])
    if {key: value for key, value in record.items() if key != "anchors_csv_sha256"} != expected:
        raise ValueError(
            "Dataset, selection, settings or recorded sampling summary changed after preparation"
        )
    if _digest(directory / "anchors.csv") != record["anchors_csv_sha256"]:
        raise ValueError("Anchor diagnostics changed after preparation")
    for sampling, expected_plan in plans.items():
        actual = load_plan(directory / f"{sampling}.npz")
        if actual["manifest"] != expected_plan["manifest"]:
            raise ValueError("Prepared plan differs from its declared deterministic sampling recipe")
    return record


def prefix(directory_or_record, case, repeat=1):
    case_config(case)
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be a positive integer (a name label; seeds are separate controls)")
    record = directory_or_record if isinstance(directory_or_record, dict) else check(directory_or_record)
    selection = record["selection"]
    return (
        f"pi05-gripper-long-r{repeat}-{case}-task{selection['task_id']}-ep{selection['episode_index']}"
        f"-steps{record['settings']['steps']}-aexpert"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="prepare one demonstration and two finite sampling plans")
    prep.add_argument("--dataset-root", type=Path, default=Path("/home/wyn/datasets/libero"))
    prep.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY)
    for name, default in (
        ("task-id", 0),
        ("steps", 3000),
        ("batch-size", 32),
        ("world-size", 2),
        ("data-seed", 0),
        ("chunk-size", 50),
        ("window-before", 5),
        ("window-after", 5),
        ("min-state-frames", 2),
    ):
        prep.add_argument(f"--{name}", type=int, default=default)
    prep.add_argument("--episode-index", type=int)
    prep.add_argument("--event-weight", type=float, default=3.0)
    prep.add_argument("--closed-direction", choices=("increasing", "decreasing"), default="increasing")
    prep.add_argument("--gripper-threshold", type=float)
    checker = sub.add_parser("check", help="recompute source fingerprints, selection and deterministic plans")
    checker.add_argument("directory", type=Path)
    namer = sub.add_parser("prefix", help="print a reproducible run-name prefix")
    namer.add_argument("directory", type=Path)
    namer.add_argument("case", choices=CASES)
    namer.add_argument("--repeat", type=int, default=1)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prefix":
        print(prefix(args["directory"], args["case"], args["repeat"]))
    else:
        record = prepare(**args) if command == "prepare" else check(args["directory"])
        print(
            json.dumps(
                {
                    "status": "prepared" if command == "prepare" else "checked",
                    "directory": str(
                        Path(args["output"] if command == "prepare" else args["directory"]).resolve()
                    ),
                    **{key: record[key] for key in ("fps", "settings", "selection", "sampling_summary")},
                    "manifest": "manifest.json contains the complete source fingerprints and plan identities",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
