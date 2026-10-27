"""Two observation/chunk-start samples, balanced exposure, opposite block orders."""

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path

import numpy as np

from lerobot.scripts import pi05_source_compare as sources
from lerobot.scripts.libero_vlm_only import assert_same_processors, validate_checkpoint
from lerobot.utils.frameskip_curriculum import load_scores
from lerobot.utils.pdet_order import META_COLUMNS, arrays_hash, fixed_probe_indices, load_plan, save_plan

PROTOCOL = "pi05-two-frame-order-v1"
ROOT = Path("/data1/wyn/prompt-work")
DIRECTIONS = ("E2H", "H2E")
NAME = re.compile(
    r"pi05-twoframe-r([1-9][0-9]*)-(E2H|H2E)-e([0-9]+)-h([0-9]+)-steps([0-9]+)"
    r"-aexpert-base-vlm2-P1-D0-E1-T1"
)


def validate_steps(steps):
    if type(steps) is not int or steps < 2 or steps % 2:
        raise ValueError("Two-frame training requires a positive EVEN number of steps")


def select_pair(
    baseline, scores, *, chunk_size=50, task_index=0, episode_index=None, easy_index=None, hard_index=None
):
    """Same task/episode, complete non-overlapping action chunks, 20/80 score-rank targets."""
    if (easy_index is None) != (hard_index is None):
        raise ValueError("Specify both easy_index and hard_index, or neither")
    if type(task_index) is not int or task_index < 0 or chunk_size < 1:
        raise ValueError("Invalid task index or chunk size")
    pool = np.unique(baseline["index"])
    pool = pool[scores["task_index"][pool] == task_index]
    if not len(pool):
        raise ValueError(f"Task {task_index} has no candidates in the original D0 pool")
    episodes = sorted(set(scores["episode_index"][pool].tolist()))
    if episode_index is not None:
        episodes = [episode_index]
    for episode in episodes:
        episode_frames = np.flatnonzero(scores["episode_index"] == episode)
        if not len(episode_frames):
            continue
        candidates = pool[scores["episode_index"][pool] == episode]
        candidates = candidates[scores["frame_index"][candidates] + chunk_size <= len(episode_frames)]
        if len(candidates) < 2:
            continue
        ranked = candidates[np.lexsort((candidates, scores["score"][candidates]))]
        chosen = None
        if easy_index is not None:
            if easy_index not in candidates or hard_index not in candidates:
                continue
            chosen = int(easy_index), int(hard_index)
        else:
            # Minimize distance to the predeclared quantile ranks subject to no
            # target overlap and a strict score difference. No model outcomes used.
            best = None
            rank_positions = np.arange(len(ranked))
            for i, easy in enumerate(ranked):
                valid = (scores["score"][ranked] > scores["score"][easy]) & (
                    np.abs(scores["frame_index"][ranked] - scores["frame_index"][easy]) >= chunk_size
                )
                positions = rank_positions[valid]
                if not len(positions):
                    continue
                j = min(positions, key=lambda j: (abs(j - 0.8 * (len(ranked) - 1)), int(ranked[j])))
                cost = abs(i - 0.2 * (len(ranked) - 1)) + abs(j - 0.8 * (len(ranked) - 1))
                candidate = float(cost), int(easy), int(ranked[j])
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                chosen = best[1:]
        if chosen is None:
            continue
        easy, hard = chosen
        if (
            scores["score"][easy] >= scores["score"][hard]
            or abs(int(scores["frame_index"][easy]) - int(scores["frame_index"][hard])) < chunk_size
        ):
            raise ValueError("Need distinct easy < hard scores and non-overlapping action chunks")
        return {
            "task_index": int(task_index),
            "episode_index": int(episode),
            "chunk_size": int(chunk_size),
            "rule": "manual"
            if easy_index is not None
            else "first_eligible_episode_nearest_score_ranks_20_80",
            **{
                label: {
                    **{key: int(scores[key][index]) for key in META_COLUMNS},
                    "score": float(scores["score"][index]),
                    "score_rank_fraction": float(np.flatnonzero(ranked == index)[0] / (len(ranked) - 1)),
                }
                for label, index in (("easy", easy), ("hard", hard))
            },
        }
    raise ValueError(
        "No eligible two-frame pair: require same task/episode, complete separated chunks and distinct scores; choose another task/episode"
    )


def make_plan(baseline, scores, selection, *, steps=6000, direction="E2H"):
    validate_steps(steps)
    if direction not in DIRECTIONS:
        raise ValueError("Direction must be E2H or H2E")
    info = baseline["manifest"]
    size = steps * info["world_size"] * info["batch_size"]
    indices = np.repeat([selection[label]["index"] for label in ("easy", "hard")], size // 2)
    arrays = {key: np.asarray(scores[key][indices], dtype=np.int64) for key in META_COLUMNS}
    pool_hash = arrays_hash(arrays)
    order = np.arange(size, dtype=np.int64)
    if direction == "H2E":
        order = np.concatenate((order[size // 2 :], order[: size // 2]))
    arrays["order"] = order
    manifest = {
        **info,
        "steps": steps,
        "instance_pool_sha256": pool_hash,
        "task_schedule_sha256": arrays_hash({"task": arrays["task_index"]}),
        "arrays_sha256": arrays_hash(arrays),
        "two_frame": {
            "protocol": PROTOCOL,
            "direction": direction,
            "selection": selection,
            "probe_indices": [selection[label]["index"] for label in ("easy", "hard")],
            "switch_step": steps // 2,
            "uses_per_frame": size // 2,
        },
        "scope": "Exactly two observation starts; equal exposure, opposite block orders, same occurrence-to-frame and E/T pairs",
    }
    return {**arrays, "manifest": manifest}


def prepare(
    source_pair,
    scores_path,
    output,
    *,
    steps=6000,
    task_index=0,
    episode_index=None,
    easy_index=None,
    hard_index=None,
):
    from lerobot.scripts.frameskip_scores import check_inputs

    validate_steps(steps)
    source_pair, scores_path, output = [Path(p).resolve() for p in (source_pair, scores_path, output)]
    requested = {
        "steps": steps,
        "task_index": task_index,
        "episode_index": episode_index,
        "easy_index": easy_index,
        "hard_index": hard_index,
    }
    if output.exists():
        record = check(output)
        if (record["source_pair"], record["scores"], record["requested"]) != (
            str(source_pair),
            str(scores_path),
            requested,
        ):
            raise ValueError("Existing pair has different inputs/settings; choose another PAIR_DIR")
        return {
            "prepared": "reused",
            "selection": record["selection"],
            "steps": steps,
            "directory": str(output),
        }
    source = sources.check(source_pair)
    check_inputs(scores_path, Path(source["dataset_root"]), Path(source["config_source"]))
    scores, score_manifest = load_scores(scores_path)
    baseline = load_plan(source_pair / "D0-W0.npz")
    if score_manifest["dataset_metadata_sha256"] != baseline["manifest"]["dataset_metadata_sha256"]:
        raise ValueError("Score dataset differs from original D0 pool")
    if (
        baseline["manifest"]["batch_size"],
        baseline["manifest"]["world_size"],
        baseline["manifest"]["data_seed"],
    ) != (32, 2, 0):
        raise ValueError("Expected original D0 pool from 2 GPUs x batch32")
    chunk_size = sources.read_json(source_pair / "libero/config.json")["chunk_size"]
    selection = select_pair(
        baseline, scores, chunk_size=chunk_size, **{k: v for k, v in requested.items() if k != "steps"}
    )
    plans = {
        direction: make_plan(baseline, scores, selection, steps=steps, direction=direction)
        for direction in DIRECTIONS
    }
    record = {
        "protocol": PROTOCOL,
        "requested": requested,
        "selection": selection,
        "source_pair": str(source_pair),
        "source_pair_sha256": sources.digest(source_pair / "pair.json"),
        "scores": str(scores_path),
        "scores_sha256": sources.digest(scores_path),
        "score_recipe": score_manifest,
        "plans": {key: plan["manifest"] for key, plan in plans.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".twoframe-", dir=output.parent) as temporary:
        temp = Path(temporary)
        for direction, plan in plans.items():
            save_plan(temp / f"{direction}.npz", plan)
        (temp / "pair.json").write_text(json.dumps(record, indent=2) + "\n")
        if output.exists():
            raise FileExistsError("Pair preparation raced; rerun to check/reuse the completed pair")
        temp.rename(output)
    return {
        "prepared": "passed",
        "selection": selection,
        "steps": steps,
        "directory": str(output),
        "distinct_training_starts": 2,
    }


def check(directory):
    directory = Path(directory).resolve()
    record = sources.read_json(directory / "pair.json")
    if record.get("protocol") != PROTOCOL:
        raise ValueError("Unknown two-frame protocol")
    source_pair = Path(record["source_pair"])
    if (
        sources.digest(source_pair / "pair.json") != record["source_pair_sha256"]
        or sources.digest(record["scores"]) != record["scores_sha256"]
    ):
        raise ValueError("Two-frame preparation inputs changed")
    sources.check(source_pair)
    scores, score_manifest = load_scores(record["scores"])
    baseline = load_plan(source_pair / "D0-W0.npz")
    requested = dict(record["requested"])
    steps = requested.pop("steps")
    if (
        score_manifest != record["score_recipe"]
        or score_manifest["dataset_metadata_sha256"] != baseline["manifest"]["dataset_metadata_sha256"]
    ):
        raise ValueError("Scores or dataset identity changed")
    chunk_size = sources.read_json(source_pair / "libero/config.json")["chunk_size"]
    selection = select_pair(baseline, scores, chunk_size=chunk_size, **requested)
    if selection != record["selection"]:
        raise ValueError("Selected samples changed")
    for direction in DIRECTIONS:
        expected = make_plan(baseline, scores, selection, steps=steps, direction=direction)
        actual = load_plan(directory / f"{direction}.npz")
        if actual["manifest"] != expected["manifest"] or actual["manifest"] != record["plans"][direction]:
            raise ValueError("Plan differs from the declared two frames, count or block order")
        fixed_probe_indices(actual)
    return record


def prefix(record, direction, repeat=1):
    if direction not in DIRECTIONS or type(repeat) is not int or repeat < 1:
        raise ValueError("Invalid direction or repeat")
    selection = record["selection"]
    return f"pi05-twoframe-r{repeat}-{direction}-e{selection['easy']['index']}-h{selection['hard']['index']}-steps{record['requested']['steps']}-aexpert"


def check_name(run, factors, tokens, within_seed):
    match = NAME.fullmatch(Path(run).name)
    if match is None or (
        within_seed,
        tokens,
        factors["prompt_seed"],
        factors["data_seed"],
        factors["noise_seed"],
        factors["time_seed"],
        factors.get("flow_binding"),
    ) != (0, 2, 1, 0, 1, 1, "occurrence"):
        raise ValueError("Expected two-frame/aexpert/VLM2/P1/D0/E1/T1/Bocc name and config")
    repeat, direction, easy, hard, steps = match.groups()
    info = load_plan(factors["order_plan"])["manifest"]
    pair = info["two_frame"]
    if (direction, int(easy), int(hard), int(steps)) != (
        pair["direction"],
        *pair["probe_indices"],
        info["steps"],
    ):
        raise ValueError("Run name disagrees with the two-frame plan")
    return int(repeat)


def check_checkpoint(directory, run):
    record = check(directory)
    run = Path(run).resolve()
    steps = record["requested"]["steps"]
    checkpoint = run / f"checkpoints/{steps:06d}/pretrained_model"
    config = sources.read_json(checkpoint / "train_config.json")
    policy, factors = config["policy"], config["pdet"]
    check_name(run, factors, policy["num_vlm_prompt_tokens"], 0)
    direction = NAME.fullmatch(run.name)[2]
    if (
        (config["seed"], config["steps"], config["batch_size"]) != (0, steps, 32)
        or not factors["enabled"]
        or Path(factors["order_plan"]).resolve() != Path(directory).resolve() / f"{direction}.npz"
    ):
        raise ValueError("Checkpoint changed two-frame budget, seed or plan")
    if not policy.get("train_action_expert") or policy.get("vlm_prompt_only") is not False:
        raise ValueError("Expected VLM prompt plus full action expert")
    if (config.get("keyframe_loss") or {}).get("mode", "off") != "off" or config.get(
        "sample_weighting"
    ) is not None:
        raise ValueError("Two-frame experiment must use unweighted flow loss")
    validate_checkpoint(
        checkpoint, 2, source=Path(record["source_pair"]) / "libero", kind="base", train_action_expert=True
    )
    return {"checkpoint": "passed", "run": str(run), "direction": direction}


def read_losses(run, expected_indices, steps):
    directory = Path(run) / "prompt_rank"
    manifest = sources.read_json(directory / "manifest.json")
    if (
        manifest.get("fixed_dataset_indices") != expected_indices
        or [x["dataset_index"] for x in manifest["samples"]] != expected_indices
    ):
        raise ValueError("Fixed probes did not use the exact easy/hard training samples")
    records = [
        json.loads(line)
        for line in (directory / "fixed_training_probe_loss.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not {0, steps // 2, steps} <= {row["step"] for row in records}:
        raise ValueError("Missing initial, switching-point or final two-frame losses")
    rows = []
    for record in records:
        if {s["sample_index"] for s in record["samples"]} != {0, 1}:
            raise ValueError("Probe losses include samples outside the fixed pair")
        row = {"step": record["step"]}
        for i, label in enumerate(("easy", "hard")):
            samples = [s for s in record["samples"] if s["sample_index"] == i]
            if sorted(s["time"] for s in samples) != sorted(manifest["options"]["timesteps"]) or any(
                s["valid_coordinates"] <= 0 for s in samples
            ):
                raise ValueError("Incomplete per-frame probe losses")
            row[f"{label}_mse"] = sum(s["squared_error_sum"] for s in samples) / sum(
                s["valid_coordinates"] for s in samples
            )
            if not np.isfinite(row[f"{label}_mse"]) or row[f"{label}_mse"] < 0:
                raise ValueError("Invalid fixed-probe error")
        row["balanced_mse"] = (row["easy_mse"] + row["hard_mse"]) / 2
        rows.append(row)
    return manifest, rows


def audit(directory, runs):
    from lerobot.scripts.pdet_order_binding import inspect_run

    record = check(directory)
    reports, first, first_probe, seen = [], None, None, set()
    steps = record["requested"]["steps"]
    indices = [record["selection"][label]["index"] for label in ("easy", "hard")]
    for run in runs:
        checkpoint = check_checkpoint(directory, run)
        direction = checkpoint["direction"]
        if direction in seen:
            raise ValueError("Duplicate direction; audit one E2H/H2E pair at a time")
        seen.add(direction)
        invariant, report = inspect_run(run, name_checker=check_name)
        invariant.pop("experiment")
        probe, losses = read_losses(run, indices, steps)
        if first is None:
            first, first_probe = invariant, probe
        elif first != invariant or first_probe != probe or reports[0]["bank_sha256"] != report["bank_sha256"]:
            raise ValueError(
                "Pair differs beyond order: recipe/source/prompt/sample-use pool/code/noise bank/probes"
            )
        if reports:
            assert_same_processors(
                Path(reports[0]["run"]) / f"checkpoints/{steps:06d}/pretrained_model",
                Path(run) / f"checkpoints/{steps:06d}/pretrained_model",
            )
        with (Path(run) / "two_frame_losses.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "easy_mse", "hard_mse", "balanced_mse"])
            writer.writeheader()
            writer.writerows(losses)
        reports.append(
            {
                **report,
                "direction": direction,
                "checkpoints_of_loss": [row for row in losses if row["step"] in (0, steps // 2, steps)],
            }
        )
    if not reports:
        raise ValueError("No runs")
    return {
        "audit": "paired_order_only_passed" if seen == set(DIRECTIONS) else "single_run_passed",
        "distinct_training_starts": 2,
        "runs": reports,
        "scope": "Same two observed chunks with equal use counts; paired audit requires identical frame/occurrence/noise/probe/recipe/code, only order changes. Fixed losses are training diagnostics, not held-out validation or a general curriculum claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-pair", type=Path, default=ROOT / "source-pair-r1")
    prep.add_argument("--scores", type=Path, default=ROOT / "frameskip-r1/scores.npz")
    prep.add_argument("--output", type=Path, default=ROOT / "twoframe-r1")
    prep.add_argument("--steps", type=int, default=6000)
    prep.add_argument("--task-index", type=int, default=0)
    prep.add_argument("--episode-index", type=int)
    prep.add_argument("--easy-index", type=int)
    prep.add_argument("--hard-index", type=int)
    sub.add_parser("check").add_argument("directory", type=Path)
    name = sub.add_parser("prefix")
    name.add_argument("directory", type=Path)
    name.add_argument("direction", choices=DIRECTIONS)
    name.add_argument("--repeat", type=int, default=1)
    name.add_argument("--steps", type=int, default=6000)
    checkpoint = sub.add_parser("check-checkpoint")
    checkpoint.add_argument("directory", type=Path)
    checkpoint.add_argument("run", type=Path)
    aud = sub.add_parser("audit")
    aud.add_argument("directory", type=Path)
    aud.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.source_pair,
            args.scores,
            args.output,
            steps=args.steps,
            task_index=args.task_index,
            episode_index=args.episode_index,
            easy_index=args.easy_index,
            hard_index=args.hard_index,
        )
    elif args.command == "audit":
        result = audit(args.directory, args.runs)
    elif args.command == "check-checkpoint":
        result = check_checkpoint(args.directory, args.run)
    else:
        record = check(args.directory)
        if args.command == "prefix":
            if record["requested"]["steps"] != args.steps:
                raise ValueError("FLOW_STEPS differs from prepared two-frame plan")
            print(prefix(record, args.direction, args.repeat))
            return
        result = {"check": "passed", "selection": record["selection"], "steps": record["requested"]["steps"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
