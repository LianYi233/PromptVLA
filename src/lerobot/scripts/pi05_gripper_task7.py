"""Prepare a task7-only, one-demonstration, 1000-update additive 2x2 experiment.

The first eligible task7 episode is selected once and recorded; all four
conditions reuse that exact episode. The original 800 Uniform updates are
shared, and a compute-matched 200 added updates are sampled uniformly or
from a ±5-frame gripper-closure neighborhood. Targets stay contiguous.
"""

import argparse
import json
from pathlib import Path

from lerobot.scripts import pi05_gripper_additive as additive
from lerobot.scripts import pi05_gripper_event as base

TASK_ID = 7
BASE_STEPS = 800
EXTRA_STEPS = 200
TOTAL_STEPS = BASE_STEPS + EXTRA_STEPS
DEFAULT_DIR = Path("/data1/wyn/prompt-work/gripper-event-task7-1k")
DEFAULT_DATASET = Path("/home/wyn/datasets/libero")


def _verify(original, prepared, work_dir):
    if (
        original["selection"]["task_id"] != TASK_ID
        or prepared["selection"]["task_id"] != TASK_ID
        or prepared["selection"]["episode_index"] != original["selection"]["episode_index"]
        or original["settings"]["steps"] != BASE_STEPS
        or prepared["settings"]["base_steps"] != BASE_STEPS
        or prepared["settings"]["extra_steps"] != EXTRA_STEPS
        or prepared["settings"]["steps"] != TOTAL_STEPS
        or prepared["original_uniform_plan"]["path"] != str(
            (work_dir / "base" / "uniform.npz").resolve()
        )
        or original["selection"]["task_index"] != prepared["selection"]["task_index"]
    ):
        raise ValueError("Task7 preparation does not match the one-episode 800+200 recipe")
    if original["dataset_root"] != prepared["dataset_root"]:
        raise ValueError("Task7 plans refer to different datasets")
    if original["settings"]["batch_size"] != prepared["settings"]["batch_size"]:
        raise ValueError("Task7 plans use different per-device batch sizes")
    if original["settings"]["world_size"] != prepared["settings"]["world_size"]:
        raise ValueError("Task7 plans use different GPU counts")
    if original["settings"]["data_seed"] != prepared["settings"]["data_seed"]:
        raise ValueError("Task7 plans use different sampling seeds")
    return {
        "task_id": TASK_ID,
        "task": prepared["selection"]["task_name"],
        "episode_index": prepared["selection"]["episode_index"],
        "episode_length": prepared["selection"]["episode_length"],
        "closure_frames": prepared["selection"]["closure_frames"],
        "base_steps": BASE_STEPS,
        "extra_steps": EXTRA_STEPS,
        "total_steps": TOTAL_STEPS,
        "batch_size": prepared["settings"]["batch_size"],
        "world_size": prepared["settings"]["world_size"],
        "additive_directory": str(work_dir / "additive"),
        "sampling_summary": prepared["sampling_summary"],
    }


def prepare(
    *,
    work_dir=DEFAULT_DIR,
    dataset_root=DEFAULT_DATASET,
    episode_index=None,
    batch_size=32,
    world_size=2,
    data_seed=0,
    window_before=5,
    window_after=5,
    closed_direction="increasing",
    gripper_threshold=None,
    min_state_frames=2,
):
    work_dir = Path(work_dir).resolve()
    dataset_root = Path(dataset_root).resolve()
    common = {
        "task_id": TASK_ID,
        "episode_index": episode_index,
        "batch_size": batch_size,
        "world_size": world_size,
        "data_seed": data_seed,
        "chunk_size": 50,
        "window_before": window_before,
        "window_after": window_after,
        "closed_direction": closed_direction,
        "gripper_threshold": gripper_threshold,
        "min_state_frames": min_state_frames,
    }
    # Old pi05_gripper_event.make_plan is used ONLY to generate a fresh task7
    # uniform base stream. Its unused weighted plan is never a training input.
    original = base.prepare(
        dataset_root, work_dir / "base",
        steps=BASE_STEPS, event_weight=3.0, **common
    )
    chosen_episode = int(original["selection"]["episode_index"])
    prepared = additive.prepare(
        dataset_root,
        work_dir / "additive",
        base_plan=work_dir / "base" / "uniform.npz",
        base_steps=BASE_STEPS,
        extra_steps=EXTRA_STEPS,
        **{**common, "episode_index": chosen_episode},
    )
    return _verify(original, prepared, work_dir)


def check(work_dir=DEFAULT_DIR):
    work_dir = Path(work_dir).resolve()
    original = base.check(work_dir / "base")
    prepared = additive.check(work_dir / "additive")
    return _verify(original, prepared, work_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check"))
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--window-before", type=int, default=5)
    parser.add_argument("--window-after", type=int, default=5)
    parser.add_argument("--closed-direction", choices=("increasing", "decreasing"), default="increasing")
    parser.add_argument("--gripper-threshold", type=float, default=None)
    parser.add_argument("--min-state-frames", type=int, default=2)
    args = parser.parse_args()
    try:
        if args.command == "check":
            summary = check(args.work_dir)
        else:
            summary = prepare(
                work_dir=args.work_dir, dataset_root=args.dataset_root,
                episode_index=args.episode_index, batch_size=args.batch_size,
                world_size=args.world_size, data_seed=args.data_seed,
                window_before=args.window_before, window_after=args.window_after,
                closed_direction=args.closed_direction,
                gripper_threshold=args.gripper_threshold,
                min_state_frames=args.min_state_frames,
            )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Task7 1k preparation failed: {error}\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
