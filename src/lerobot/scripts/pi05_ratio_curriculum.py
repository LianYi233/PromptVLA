"""Task-local score ordering with an explicit easy/hard exposure budget.

Unlike the order-only H2E control, this protocol resamples the original D0 pool.
Every repeated use receives a new occurrence ID; E/T follow these new uses.
Preparation is CPU-only and never changes process-global RNG state.
"""

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path

import numpy as np

from lerobot.scripts import pi05_hard_to_easy as original, pi05_source_compare as sources
from lerobot.scripts.libero_vlm_only import assert_same_processors, validate_checkpoint
from lerobot.utils.frameskip_curriculum import load_scores
from lerobot.utils.pdet_order import META_COLUMNS, arrays_hash, load_plan, save_plan

PROTOCOL = "pi05-ratio-curriculum-v1"
DIRECTIONS = {"easy_to_hard": "E2H", "hard_to_easy": "H2E"}
NAME = re.compile(
    r"pi05-curriculum-(E2H|H2E)-e([0-9]+)-h([0-9]+)-S([0-9]+)-aexpert-r([1-9][0-9]*)"
    r"-libero-Bocc-base-vlm2-P1-D0-E1-T1"
)


def options(direction, easy_percent, sample_seed):
    if direction not in DIRECTIONS:
        raise ValueError("direction must be easy_to_hard or hard_to_easy")
    if type(easy_percent) is not int or not 1 <= easy_percent <= 99:
        raise ValueError("easy_percent must be an integer in [1, 99]")
    if type(sample_seed) is not int or not 0 <= sample_seed < 2**32:
        raise ValueError("sample_seed must be an integer in [0, 2**32)")
    return {"direction": direction, "easy_percent": easy_percent, "sample_seed": sample_seed}


def label(direction, easy_percent, sample_seed):
    options(direction, easy_percent, sample_seed)
    return f"{DIRECTIONS[direction]}-e{easy_percent}-h{100 - easy_percent}-S{sample_seed}"


def partition(baseline, scores, score_manifest):
    """Lower half of UNIQUE candidate frames in each task; tied scores use frame index."""
    info = baseline["manifest"]
    if any(info.get(key) for key in ("curriculum", "difficulty_order", "difficulty_sampling")) or (
        info["within_seed"] != 0
    ):
        raise ValueError("Use the original W0 baseline plan")
    if arrays_hash({key: scores[key] for key in META_COLUMNS}) != info["dataset_metadata_sha256"] or (
        score_manifest["dataset_metadata_sha256"] != info["dataset_metadata_sha256"]
    ):
        raise ValueError("Scores and baseline have different dataset metadata")
    if not np.array_equal(scores["index"], np.arange(info["dataset_frames"])):
        raise ValueError("Scores must cover all dataset frames in index order")
    for key in META_COLUMNS:
        if not np.array_equal(baseline[key], scores[key][baseline["index"]]):
            raise ValueError(f"Baseline sample pool differs from score metadata: {key}")
    values = np.asarray(scores["score"], dtype=np.float64)
    if values.shape != scores["index"].shape or not np.isfinite(values).all():
        raise ValueError("Invalid difficulty scores")
    easy = np.zeros(len(values), dtype=bool)
    for task in np.unique(baseline["task_index"]):
        frames = np.unique(baseline["index"][baseline["task_index"] == task])
        if len(frames) < 2 or np.ptp(values[frames]) == 0:
            raise ValueError(f"Task {task} has insufficient distinct scores for easy/hard groups")
        ranked = frames[np.lexsort((frames, values[frames]))]
        easy[ranked[: len(ranked) // 2]] = True
    return easy


def _draw(candidates, count, seed):
    """Shuffle cycles: use every candidate before repeating, truncate the last cycle."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    pieces = []
    while count:
        piece = rng.permutation(candidates)[:count]
        pieces.append(piece)
        count -= len(piece)
    return np.concatenate(pieces)


def make_plan(baseline, scores, score_manifest, *, direction, easy_percent, sample_seed=0):
    settings = options(direction, easy_percent, sample_seed)
    easy = partition(baseline, scores, score_manifest)
    tasks, counts = np.unique(baseline["task_index"], return_counts=True)
    # Largest remainder allocation gives the requested GLOBAL number of sample uses,
    # with less than one use of rounding error per task. Ties use ascending task ID.
    quotas = counts * easy_percent // 100
    target = (len(baseline["order"]) * easy_percent + 50) // 100
    remainder = int(target - quotas.sum())
    quotas[np.argsort(-(counts * easy_percent % 100), kind="stable")[:remainder]] += 1
    if np.any(quotas == 0) or np.any(quotas == counts):
        raise ValueError("Budget too small for this ratio: every task must retain both groups")
    parents = np.empty_like(baseline["order"])
    summary = []
    for task, count, quota in zip(tasks, counts, quotas, strict=True):
        slots = np.flatnonzero(baseline["task_index"] == task)
        uses = baseline["order"][slots]
        groups = easy[baseline["index"][uses]]
        selected = np.concatenate(
            [
                _draw(uses[groups], int(quota), [72819, sample_seed, int(task), 0]),
                _draw(uses[~groups], int(count - quota), [72819, sample_seed, int(task), 1]),
            ]
        )
        # Canonical occurrence IDs are independent of curriculum direction.
        rng = np.random.default_rng(np.random.SeedSequence([72819, sample_seed, int(task), 2]))
        parents[slots] = rng.permutation(selected)
        easy_frames = np.unique(baseline["index"][uses[groups]])
        hard_frames = np.unique(baseline["index"][uses[~groups]])
        summary.append(
            {
                "task": int(task),
                "uses": int(count),
                "easy_uses": int(quota),
                "hard_uses": int(count - quota),
                "candidate_easy_frames": len(easy_frames),
                "candidate_hard_frames": len(hard_frames),
                "selected_unique_frames": len(np.unique(baseline["index"][selected])),
                "repeated_parent_uses": int(count - len(np.unique(selected))),
                "boundary_score_tied": bool(
                    scores["score"][easy_frames].max() == scores["score"][hard_frames].min()
                ),
            }
        )
    arrays = {key: baseline[key][parents].copy() for key in META_COLUMNS}
    order = np.arange(len(parents), dtype=np.int64)
    values = scores["score"][arrays["index"]]
    for task in tasks:
        slots = np.flatnonzero(arrays["task_index"] == task)
        # At a tied boundary, all easy-group uses precede hard-group uses in E2H,
        # and vice versa in H2E. Remaining ties use immutable new occurrence ID.
        group = (~easy[arrays["index"][slots]]).astype(np.int64)
        sign = 1 if direction == "easy_to_hard" else -1
        order[slots] = slots[np.lexsort((slots, sign * values[slots], sign * group))]
    manifest = {
        **baseline["manifest"],
        "difficulty_sampling": {
            "protocol": PROTOCOL,
            **settings,
            "partition": "per_task_lower_half_unique_candidate_frames_score_then_frame_index",
            "sampling": "independent_group_shuffle_cycles",
            "score_arrays_sha256": score_manifest["arrays_sha256"],
            "parent_plan_arrays_sha256": baseline["manifest"]["arrays_sha256"],
            "parent_use_assignment_sha256": arrays_hash({"parents": parents}),
            "easy_uses": int(quotas.sum()),
            "hard_uses": int(len(parents) - quotas.sum()),
            "tasks": summary,
        },
        "instance_pool_sha256": arrays_hash(arrays),
        "scope": "Resampled D0 pool, original task slots, NEW occurrence IDs; E/T follow new uses",
    }
    arrays["order"] = order
    manifest["arrays_sha256"] = arrays_hash(arrays)
    return {**arrays, "manifest": manifest}


def prepare(pair, scores_path, reference, output, *, direction, easy_percent, sample_seed=0):
    from lerobot.scripts.frameskip_scores import check_inputs

    settings = options(direction, easy_percent, sample_seed)
    pair, scores_path, reference, output = [Path(p).resolve() for p in (pair, scores_path, reference, output)]
    if output.exists():
        record = check(output)
        if (record["pair"], record["scores"], record["reference_run"], record["settings"]) != (
            str(pair),
            str(scores_path),
            str(reference),
            settings,
        ):
            raise ValueError("Existing curriculum uses different settings/inputs; choose a new ORDER_DIR")
        return {"prepared": "reused", "directory": str(output), **settings}
    source = original.reference_inputs(pair, reference)
    check_inputs(scores_path, Path(source["dataset_root"]), Path(source["config_source"]))
    scores, score_manifest = load_scores(scores_path)
    baseline = load_plan(pair / "D0-W0.npz")
    plan = make_plan(baseline, scores, score_manifest, **settings)
    record = {
        "protocol": PROTOCOL,
        "settings": settings,
        "pair": str(pair),
        "pair_sha256": sources.digest(pair / "pair.json"),
        "scores": str(scores_path),
        "scores_sha256": sources.digest(scores_path),
        "score_recipe": score_manifest,
        "reference_run": str(reference),
        "reference_manifest_sha256": sources.digest(reference / "pdet/manifest-rank0.json"),
        "plan": plan["manifest"],
    }
    indices = plan["index"][plan["order"]]
    easy = partition(baseline, scores, score_manifest)[indices]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ratio-curriculum-", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        save_plan(temp / "plan.npz", plan)
        (temp / "curriculum.json").write_text(json.dumps(record, indent=2) + "\n")
        with (temp / "schedule.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["update", "easy_uses", "hard_uses", "mean_score", "min_score", "max_score"])
            batch = plan["manifest"]["batch_size"] * plan["manifest"]["world_size"]
            for step, start in enumerate(range(0, len(indices), batch), 1):
                values = scores["score"][indices[start : start + batch]]
                n_easy = int(easy[start : start + batch].sum())
                writer.writerow(
                    [step, n_easy, len(values) - n_easy, values.mean(), values.min(), values.max()]
                )
        if output.exists():
            raise FileExistsError(output)
        temp.rename(output)
    return {
        "prepared": "passed",
        "directory": str(output),
        **settings,
        "easy_uses": int(easy.sum()),
        "hard_uses": int((~easy).sum()),
        "new_scoring": False,
    }


def check(directory):
    directory = Path(directory).resolve()
    record = sources.read_json(directory / "curriculum.json")
    if record.get("protocol") != PROTOCOL:
        raise ValueError("Unknown ratio-curriculum protocol")
    pair, reference = Path(record["pair"]), Path(record["reference_run"])
    for path, key in (
        (pair / "pair.json", "pair_sha256"),
        (Path(record["scores"]), "scores_sha256"),
        (reference / "pdet/manifest-rank0.json", "reference_manifest_sha256"),
    ):
        if sources.digest(path) != record[key]:
            raise ValueError(f"Prepared curriculum input changed: {path}")
    original.reference_inputs(pair, reference)
    scores, score_manifest = load_scores(record["scores"])
    if score_manifest != record["score_recipe"]:
        raise ValueError("Score recipe changed")
    expected = make_plan(load_plan(pair / "D0-W0.npz"), scores, score_manifest, **record["settings"])
    actual = load_plan(directory / "plan.npz")
    if actual["manifest"] != expected["manifest"] or actual["manifest"] != record["plan"]:
        raise ValueError("Plan does not match the declared ratio, pool and order")
    return record


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
        raise ValueError("Expected ratio-curriculum/aexpert/VLM2/P1/D0/E1/T1/occurrence name and config")
    direction, easy, hard, seed, repeat = match.groups()
    settings = load_plan(factors["order_plan"])["manifest"]["difficulty_sampling"]
    if f"{direction}-e{easy}-h{hard}-S{seed}" != label(
        settings["direction"], settings["easy_percent"], settings["sample_seed"]
    ):
        raise ValueError("Run name disagrees with the saved ratio, direction or sample seed")
    return int(repeat)


def check_checkpoint(directory, run):
    directory, run = Path(directory).resolve(), Path(run).resolve()
    record = check(directory)
    checkpoint = run / "checkpoints/003000/pretrained_model"
    config = sources.read_json(checkpoint / "train_config.json")
    factors, policy = config["pdet"], config["policy"]
    check_name(run, factors, policy["num_vlm_prompt_tokens"], 0)
    if (config["seed"], config["steps"], config["batch_size"]) != (0, 3000, 32) or (
        not factors["enabled"] or Path(factors["order_plan"]).resolve() != directory / "plan.npz"
    ):
        raise ValueError("Checkpoint changed the budget, seed or curriculum plan")
    weighting = config.get("keyframe_loss") or {}
    if (
        weighting.get("mode", "off") != "off"
        or weighting.get("enabled", False)
        or (config.get("sample_weighting") is not None)
    ):
        raise ValueError("Ratio curriculum must not add loss weighting")
    if not policy.get("train_action_expert", False) or policy.get("vlm_prompt_only") is not False:
        raise ValueError("This experiment requires VLM prompt plus full action expert training")
    validate_checkpoint(
        checkpoint, 2, source=Path(record["pair"]) / "libero", kind="base", train_action_expert=True
    )
    return {"checkpoint": "passed", "run": str(run), **record["settings"]}


def audit(directory, run):
    from lerobot.scripts.pdet_order_binding import inspect_run

    record = check(directory)
    check_checkpoint(directory, run)
    reference = Path(record["reference_run"])
    old, old_report = inspect_run(reference, name_checker=sources.check_name)
    new, new_report = inspect_run(run, name_checker=check_name)
    code, pools = {}, {}
    for name, invariant, expert in (("reference", old, False), ("curriculum", new, True)):
        code[name] = {key: invariant.pop(key) for key in ("git_commit", "source_tree_sha256")}
        pools[name] = invariant.pop(
            "pool"
        )  # Explicitly changed by resampling, not silently treated as fixed.
        invariant.pop("experiment")
        policy = invariant["recipe"]["policy"]
        if policy.get("train_action_expert", False) != expert or policy.get("vlm_prompt_only") != (
            not expert
        ):
            raise ValueError("Recorded recipe disagrees with trainable scope")
        policy.pop("train_action_expert", None)
        policy["vlm_prompt_only"] = True
    if old != new:
        changed = [key for key in old.keys() | new.keys() if old.get(key) != new.get(key)]
        raise ValueError(f"Unexpected changes beyond resampling, order and action-expert training: {changed}")
    if old_report["bank_sha256"] != new_report["bank_sha256"]:
        raise ValueError("Realized indexed noise/time bank changed")
    assert_same_processors(
        reference / "checkpoints/003000/pretrained_model", Path(run) / "checkpoints/003000/pretrained_model"
    )
    return {
        "audit": "inputs_and_recipe_passed",
        "settings": record["settings"],
        "exposure": record["plan"]["difficulty_sampling"],
        "reference": old_report,
        "curriculum": new_report,
        "interventions": [
            "easy_hard_resampling",
            record["settings"]["direction"],
            "full_action_expert_training",
        ],
        "sample_pools": pools,
        "code_revision": {
            **code,
            "status": "same_recorded_code"
            if code["reference"] == code["curriculum"]
            else "different_not_certified",
        },
        "scope": "All realized uses: planned ratio/order/task slots, same indexed E1/T1 bank and P1/recipe except declared interventions. Resampling changes sample-to-noise pairing; not historical code equivalence or numerical determinism.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--pair", type=Path, default=original.DEFAULT_ROOT / "source-pair-r1")
    prep.add_argument("--scores", type=Path, default=original.DEFAULT_ROOT / "frameskip-r1/scores.npz")
    prep.add_argument("--reference-run", type=Path, default=original.DEFAULT_REFERENCE)
    prep.add_argument("--output", type=Path, required=True)
    for cmd in (prep, sub.add_parser("label")):
        cmd.add_argument("--direction", choices=DIRECTIONS, required=True)
        cmd.add_argument("--easy-percent", type=int, required=True)
        cmd.add_argument("--sample-seed", type=int, default=0)
    sub.add_parser("check").add_argument("directory", type=Path)
    for command in ("check-checkpoint", "audit"):
        cmd = sub.add_parser(command)
        cmd.add_argument("directory", type=Path)
        cmd.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.command == "label":
        print(label(args.direction, args.easy_percent, args.sample_seed))
        return
    if args.command == "prepare":
        result = prepare(
            args.pair,
            args.scores,
            args.reference_run,
            args.output,
            direction=args.direction,
            easy_percent=args.easy_percent,
            sample_seed=args.sample_seed,
        )
    elif args.command == "check":
        record = check(args.directory)
        result = {"preflight": "passed", **record["settings"]}
    elif args.command == "audit":
        result = audit(args.directory, args.run)
    else:
        result = check_checkpoint(args.directory, args.run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
