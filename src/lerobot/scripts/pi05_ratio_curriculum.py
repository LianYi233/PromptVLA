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
DIRECTIONS = {"easy_to_hard": "E2H", "hard_to_easy": "H2E", "random": "RND"}
SCOPES = {"joint": "aexpert", "expert_only": "frozenp-aexpert", "prompt_only": "promptonly"}
NAME = re.compile(
    r"pi05-curriculum-(E2H|H2E|RND)-e([0-9]+)-h([0-9]+)-S([0-9]+)(-steps[1-9][0-9]*)?"
    r"-(aexpert|frozenp-aexpert|promptonly)-r([1-9][0-9]*)"
    r"-libero-Bocc-base-vlm2-P1-D0-E1-T1"
)


def options(direction, easy_percent, sample_seed):
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {tuple(DIRECTIONS)}")
    if type(easy_percent) is not int or not 1 <= easy_percent <= 99:
        raise ValueError("easy_percent must be an integer in [1, 99]")
    if type(sample_seed) is not int or not 0 <= sample_seed < 2**32:
        raise ValueError("sample_seed must be an integer in [0, 2**32)")
    return {"direction": direction, "easy_percent": easy_percent, "sample_seed": sample_seed}


def validate_steps(steps):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    return steps


def label(direction, easy_percent, sample_seed, steps=3000):
    options(direction, easy_percent, sample_seed)
    validate_steps(steps)
    suffix = "" if steps == 3000 else f"-steps{steps}"
    return f"{DIRECTIONS[direction]}-e{easy_percent}-h{100 - easy_percent}-S{sample_seed}{suffix}"


def scope_flags(scope):
    if scope not in SCOPES:
        raise ValueError(f"Unknown training scope: {scope}")
    return {
        "train_action_expert": scope != "prompt_only",
        "vlm_prompt_only": scope == "prompt_only",
        "freeze_vlm_prompt": scope == "expert_only",
    }


def scope_from_run(run):
    match = NAME.fullmatch(Path(run).name)
    if match is None:
        raise ValueError(f"Invalid ratio curriculum run name: {run}")
    return next(scope for scope, tag in SCOPES.items() if tag == match[6])


def run_name(direction, easy_percent, sample_seed=0, steps=6000, scope="joint", repeat=2):
    scope_flags(scope)
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be a positive integer")
    return (
        f"pi05-curriculum-{label(direction, easy_percent, sample_seed, steps)}"
        f"-{SCOPES[scope]}-r{repeat}-libero-Bocc-base-vlm2-P1-D0-E1-T1"
    )


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


def make_plan(baseline, scores, score_manifest, *, direction, easy_percent, sample_seed=0, steps=None):
    settings = options(direction, easy_percent, sample_seed)
    easy = partition(baseline, scores, score_manifest)
    info = baseline["manifest"]
    steps = validate_steps(info["steps"] if steps is None else steps)
    # Extend task slots, NOT the already-sorted curriculum. Sort the entire new
    # budget once so an E2H/H2E course never restarts halfway through training.
    size = steps * info["batch_size"] * info["world_size"]
    task_slots = np.resize(baseline["task_index"], size)
    tasks, counts = np.unique(task_slots, return_counts=True)
    # Largest remainder allocation gives the requested GLOBAL number of sample uses,
    # with less than one use of rounding error per task. Ties use ascending task ID.
    quotas = counts * easy_percent // 100
    target = (size * easy_percent + 50) // 100
    remainder = int(target - quotas.sum())
    quotas[np.argsort(-(counts * easy_percent % 100), kind="stable")[:remainder]] += 1
    if np.any(quotas == 0) or np.any(quotas == counts):
        raise ValueError("Budget too small for this ratio: every task must retain both groups")
    parents = np.empty(size, dtype=np.int64)
    summary = []
    for task, count, quota in zip(tasks, counts, quotas, strict=True):
        slots = np.flatnonzero(task_slots == task)
        uses = baseline["order"][baseline["task_index"] == task]
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
        if direction == "random":
            # Independent local generator: permute the SAME canonical occurrence
            # pool, without changing task slots, frame counts or per-ID E/T.
            rng = np.random.default_rng(np.random.SeedSequence([72820, sample_seed, int(task)]))
            order[slots] = rng.permutation(slots)
            continue
        # At a tied boundary, all easy-group uses precede hard-group uses in E2H,
        # and vice versa in H2E. Remaining ties use immutable new occurrence ID.
        group = (~easy[arrays["index"][slots]]).astype(np.int64)
        sign = 1 if direction == "easy_to_hard" else -1
        order[slots] = slots[np.lexsort((slots, sign * values[slots], sign * group))]
    manifest = {
        **baseline["manifest"],
        "steps": steps,
        "task_schedule_sha256": arrays_hash({"task": task_slots}),
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
    if steps != info["steps"]:
        manifest["difficulty_sampling"]["budget"] = {
            "reference_steps": info["steps"],
            "steps": steps,
            "task_slots": "cycle_original_task_schedule_then_truncate",
            "ordering": "single_course_over_full_budget",
        }
        manifest["scope"] = (
            "Resampled original D0 candidates, cycled task slots, NEW occurrence IDs; E/T follow new uses"
        )
    arrays["order"] = order
    manifest["arrays_sha256"] = arrays_hash(arrays)
    return {**arrays, "manifest": manifest}


def prepare(pair, scores_path, reference, output, *, direction, easy_percent, sample_seed=0, steps=6000):
    from lerobot.scripts.frameskip_scores import check_inputs

    settings = options(direction, easy_percent, sample_seed)
    validate_steps(steps)
    pair, scores_path, reference, output = [Path(p).resolve() for p in (pair, scores_path, reference, output)]
    if output.exists():
        record = check(output)
        if (
            record["pair"],
            record["scores"],
            record["reference_run"],
            record["settings"],
            record["plan"]["steps"],
        ) != (
            str(pair),
            str(scores_path),
            str(reference),
            settings,
            steps,
        ):
            raise ValueError("Existing curriculum uses different settings/inputs; choose a new ORDER_DIR")
        return {"prepared": "reused", "directory": str(output), "steps": steps, **settings}
    source = original.reference_inputs(pair, reference)
    check_inputs(scores_path, Path(source["dataset_root"]), Path(source["config_source"]))
    scores, score_manifest = load_scores(scores_path)
    baseline = load_plan(pair / "D0-W0.npz")
    plan = make_plan(baseline, scores, score_manifest, **settings, steps=steps)
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
        "steps": steps,
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
    expected = make_plan(
        load_plan(pair / "D0-W0.npz"),
        scores,
        score_manifest,
        **record["settings"],
        steps=record["plan"]["steps"],
    )
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
    direction, easy, hard, seed, budget_tag, _scope, repeat = match.groups()
    info = load_plan(factors["order_plan"])["manifest"]
    settings = info["difficulty_sampling"]
    if f"{direction}-e{easy}-h{hard}-S{seed}{budget_tag or ''}" != label(
        settings["direction"], settings["easy_percent"], settings["sample_seed"], info["steps"]
    ):
        raise ValueError("Run name disagrees with the saved ratio, direction, sample seed or steps")
    return int(repeat)


def check_checkpoint(directory, run):
    directory, run = Path(directory).resolve(), Path(run).resolve()
    record = check(directory)
    steps = record["plan"]["steps"]
    checkpoint = run / f"checkpoints/{steps:06d}/pretrained_model"
    config = sources.read_json(checkpoint / "train_config.json")
    factors, policy = config["pdet"], config["policy"]
    check_name(run, factors, policy["num_vlm_prompt_tokens"], 0)
    if (config["seed"], config["steps"], config["batch_size"]) != (0, steps, 32) or (
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
    scope = scope_from_run(run)
    expected = scope_flags(scope)
    saved_policy = sources.read_json(checkpoint / "config.json")
    for key, value in expected.items():
        if policy.get(key, False) != value or saved_policy.get(key, False) != value:
            raise ValueError(f"Checkpoint training scope disagrees with {scope}: {key}")
    validate_checkpoint(
        checkpoint,
        2,
        source=Path(record["pair"]) / "libero",
        kind="base",
        train_action_expert=expected["train_action_expert"],
    )
    if expected["freeze_vlm_prompt"]:
        from safetensors import safe_open

        from lerobot.utils.prompt_initialization import read_initial_prompt

        initial, _ = read_initial_prompt(Path(record["pair"]) / "P1.safetensors", 2)
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
            final = weights.get_tensor("model.vlm_prompt_tokens.weight").float().numpy()
        if not np.array_equal(initial, final):
            raise ValueError("Frozen P1 prompt changed during expert-only training")
    return {"checkpoint": "passed", "run": str(run), "steps": steps, "scope": scope, **record["settings"]}


def audit(directory, run):
    from lerobot.scripts.curriculum_audit_utils import differences, normalize_path_aliases
    from lerobot.scripts.pdet_order_binding import inspect_run

    record = check(directory)
    check_checkpoint(directory, run)
    reference = Path(record["reference_run"])
    ref_manifest = sources.read_json(reference / "pdet/manifest-rank0.json")
    reference_steps, steps = ref_manifest["total_steps"], record["plan"]["steps"]
    shared_count = min(reference_steps, steps) * record["plan"]["batch_size"] * record["plan"]["world_size"]
    old, old_report = inspect_run(reference, name_checker=sources.check_name, bank_prefix_count=shared_count)
    new, new_report = inspect_run(run, name_checker=check_name, bank_prefix_count=shared_count)
    code, pools, tasks = {}, {}, {}
    scope = scope_from_run(run)
    for name, invariant, train_scope, budget in (
        ("reference", old, "prompt_only", reference_steps),
        ("curriculum", new, scope, steps),
    ):
        code[name] = {key: invariant.pop(key) for key in ("git_commit", "source_tree_sha256")}
        pools[name] = invariant.pop(
            "pool"
        )  # Explicitly changed by resampling, not silently treated as fixed.
        invariant.pop("experiment")
        # Plan/check/inspect_run have verified the complete task schedule, including
        # its extension. Only normalize the explicitly declared budget differences.
        tasks[name] = invariant.pop("tasks")
        if invariant.pop("total_steps") != budget or invariant["recipe"].pop("steps") != budget:
            raise ValueError("Recorded step budget differs from the plan")
        if invariant["factors"].pop("audit_steps") != budget:
            raise ValueError("Expected all-step audit at exactly the configured training budget")
        if "save_freq" in invariant["recipe"] and invariant["recipe"].pop("save_freq") != budget:
            raise ValueError("Unexpected checkpoint save frequency")
        policy = invariant["recipe"]["policy"]
        for key, value in scope_flags(train_scope).items():
            if policy.get(key, False) != value:
                raise ValueError(f"Recorded recipe disagrees with trainable scope: {key}")
        policy.pop("train_action_expert", None)
        policy.pop("freeze_vlm_prompt", None)
        policy["vlm_prompt_only"] = True
    aliases = normalize_path_aliases(old, new)
    if old != new:
        raise ValueError(
            "Unexpected changes beyond resampling, order and action-expert training: "
            + json.dumps(differences(old, new), ensure_ascii=False)
        )
    if old_report["bank_prefix_sha256"] != new_report["bank_prefix_sha256"]:
        raise ValueError("Realized indexed noise/time bank changed on shared keys")
    assert_same_processors(
        reference / f"checkpoints/{reference_steps:06d}/pretrained_model",
        Path(run) / f"checkpoints/{steps:06d}/pretrained_model",
    )
    return {
        "audit": "inputs_and_recipe_passed",
        "settings": record["settings"],
        "steps": steps,
        "reference_steps": reference_steps,
        "path_aliases": aliases,
        "exposure": record["plan"]["difficulty_sampling"],
        "reference": old_report,
        "curriculum": new_report,
        "interventions": [
            "easy_hard_resampling",
            record["settings"]["direction"],
            f"training_scope:{scope}",
        ]
        + (["training_budget"] if steps != reference_steps else []),
        "sample_pools": pools,
        "task_schedules": tasks,
        "bank_comparison": {
            "shared_keys": shared_count,
            "new_keys_without_historical_reference": max(0, new_report["occurrences"] - shared_count),
        },
        "code_revision": {
            **code,
            "status": "same_recorded_code"
            if code["reference"] == code["curriculum"]
            else "different_not_certified",
        },
        "scope": "All realized uses: planned budget/ratio/order/task slots and P1/recipe except declared interventions. Shared E1/T1 keys match reference; extended bank keys have trace coverage but no historical counterpart. Resampling changes sample-to-noise pairing; not historical code equivalence or numerical determinism.",
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
        cmd.add_argument("--steps", type=int, default=6000)
    sub.add_parser("check").add_argument("directory", type=Path)
    for command in ("check-checkpoint", "audit"):
        cmd = sub.add_parser(command)
        cmd.add_argument("directory", type=Path)
        cmd.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.command == "label":
        print(label(args.direction, args.easy_percent, args.sample_seed, args.steps))
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
            steps=args.steps,
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
