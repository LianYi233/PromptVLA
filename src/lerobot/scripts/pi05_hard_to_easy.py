"""Order-only hard-to-easy curriculum on the completed LIBERO source-pair baseline.

Difficulty is the existing FrameSkip-inspired importance proxy, not measured model
error. Each task keeps its original slots. Immutable sample uses carry their E/T.
No model construction, video decoding, new scores or random draws during preparation.
"""

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path

import numpy as np

from lerobot.scripts import pi05_source_compare as source_compare
from lerobot.scripts.libero_vlm_only import assert_same_processors, validate_checkpoint
from lerobot.utils.frameskip_curriculum import load_scores
from lerobot.utils.pdet_order import META_COLUMNS, arrays_hash, load_plan, save_plan

PROTOCOL = "pi05-hard-to-easy-order-v1"
NAME = re.compile(r"pi05-hard2easy-(?:aexpert-)?r([1-9][0-9]*)-libero-Bocc-base-vlm2-P1-D0-E1-T1")
DEFAULT_ROOT = Path("/data1/wyn/prompt-work")
DEFAULT_REFERENCE = Path(
    "/data1/wyn/chkpt/2601-lerobot/PromptVLA/pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1"
)


def reorder(plan, scores, score_manifest):
    """Stable descending sort within each task; ties retain the original W0 order."""
    info = plan["manifest"]
    if info.get("curriculum") or info.get("difficulty_order") or info["within_seed"] != 0:
        raise ValueError("Use the original W0 baseline plan, not a previously changed curriculum")
    metadata = {key: scores[key] for key in META_COLUMNS}
    if arrays_hash(metadata) != info["dataset_metadata_sha256"] or (
        score_manifest["dataset_metadata_sha256"] != info["dataset_metadata_sha256"]
    ):
        raise ValueError("Scores and original plan refer to different dataset metadata")
    if not np.array_equal(scores["index"], np.arange(info["dataset_frames"])):
        raise ValueError("Scores must cover all original dataset indices in index order")
    for key in META_COLUMNS:
        if not np.array_equal(plan[key], scores[key][plan["index"]]):
            raise ValueError(f"Original sample pool differs from score metadata: {key}")
    difficulty = np.asarray(scores["score"], dtype=np.float64)[plan["index"]]
    if difficulty.shape != plan["order"].shape or not np.isfinite(difficulty).all():
        raise ValueError("Invalid difficulty scores")
    order = plan["order"].copy()
    for task in np.unique(plan["task_index"]):
        slots = np.flatnonzero(plan["task_index"] == task)
        uses = plan["order"][slots]
        order[slots] = uses[np.argsort(-difficulty[uses], kind="stable")]
    if np.array_equal(order, plan["order"]):
        raise ValueError("Scores did not change the baseline order; no order intervention to train")
    arrays = {key: plan[key].copy() for key in META_COLUMNS}
    arrays["order"] = order
    manifest = {
        **info,
        "difficulty_order": {
            "protocol": PROTOCOL,
            "direction": "hard_to_easy",
            "score_key": "score",
            "score_arrays_sha256": score_manifest["arrays_sha256"],
            "parent_plan_arrays_sha256": info["arrays_sha256"],
            "tie_break": "original_W0_order",
        },
        "scope": "Descending proxy score within fixed task slots; original W0 pool and occurrence IDs unchanged",
        "arrays_sha256": arrays_hash(arrays),
    }
    return {**arrays, "manifest": manifest}


def reference_inputs(pair, reference):
    record = source_compare.check(pair)
    result = source_compare.check_checkpoint(pair, reference)
    if result["weight_source"] != "libero":
        raise ValueError("The reference must be the completed pi05_libero_base source run")
    plan_info = load_plan(pair / "D0-W0.npz")["manifest"]
    if tuple(plan_info[key] for key in ("steps", "batch_size", "world_size", "data_seed")) != (
        3000,
        32,
        2,
        0,
    ):
        raise ValueError("Expected original D0 plan: 3000 steps, 2 GPUs x batch32")
    manifest = source_compare.read_json(reference / "pdet/manifest-rank0.json")
    if manifest["order_plan"] != plan_info or (manifest["prompt_sha256"] != record["prompt_sha256"]):
        raise ValueError("Reference did not use the exact prepared W0 plan/P1 tensor")
    if Path(manifest["source_pretrained_path"]).resolve() != (pair / "libero").resolve():
        raise ValueError("Reference source differs from the prepared LIBERO view")
    source_compare.check_name(reference, manifest["factors"], manifest["tokens"], 0)
    return record


def prepare(pair, scores_path, reference, output):
    from lerobot.scripts.frameskip_scores import check_inputs

    pair, scores_path, reference, output = [Path(p).resolve() for p in (pair, scores_path, reference, output)]
    if output.exists():
        existing = check(output)
        if (existing["pair"], existing["scores"], existing["reference_run"]) != tuple(
            map(str, (pair, scores_path, reference))
        ):
            raise ValueError("Existing curriculum uses different inputs; choose a new ORDER_DIR")
        return {"prepared": "reused", "directory": str(output)}
    source = reference_inputs(pair, reference)
    check_inputs(scores_path, Path(source["dataset_root"]), Path(source["config_source"]))
    scores, score_manifest = load_scores(scores_path)
    baseline = load_plan(pair / "D0-W0.npz")
    plan = reorder(baseline, scores, score_manifest)
    record = {
        "protocol": PROTOCOL,
        "pair": str(pair),
        "pair_sha256": source_compare.digest(pair / "pair.json"),
        "scores": str(scores_path),
        "scores_sha256": source_compare.digest(scores_path),
        "score_recipe": score_manifest,
        "reference_run": str(reference),
        "reference_manifest_sha256": source_compare.digest(reference / "pdet/manifest-rank0.json"),
        "plan": plan["manifest"],
        "scope": "Order only: all original uses, task slots, P1 and occurrence-bound E1/T1 retained",
    }
    difficulty = scores["score"][plan["index"][plan["order"]]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hard2easy-", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        save_plan(temp / "H2E.npz", plan)
        (temp / "curriculum.json").write_text(json.dumps(record, indent=2) + "\n")
        # An interpretable schedule, without another GPU scoring pass.
        with (temp / "schedule.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["update", "mean_score", "min_score", "max_score"])
            batch = plan["manifest"]["batch_size"] * plan["manifest"]["world_size"]
            for step, start in enumerate(range(0, len(difficulty), batch), 1):
                values = difficulty[start : start + batch]
                writer.writerow([step, float(values.mean()), float(values.min()), float(values.max())])
        if output.exists():
            raise FileExistsError(output)
        temp.rename(output)
    return {
        "prepared": "passed",
        "directory": str(output),
        "occurrences": len(plan["order"]),
        "direction": "hard_to_easy_within_each_task",
        "new_scoring": False,
    }


def check(directory):
    directory = Path(directory).resolve()
    record = source_compare.read_json(directory / "curriculum.json")
    if record.get("protocol") != PROTOCOL:
        raise ValueError("Unknown difficulty-order protocol")
    pair, reference = Path(record["pair"]), Path(record["reference_run"])
    for path, key in (
        (pair / "pair.json", "pair_sha256"),
        (Path(record["scores"]), "scores_sha256"),
        (reference / "pdet/manifest-rank0.json", "reference_manifest_sha256"),
    ):
        if source_compare.digest(path) != record[key]:
            raise ValueError(f"Prepared curriculum input changed: {path}")
    reference_inputs(pair, reference)
    scores, score_manifest = load_scores(record["scores"])
    if score_manifest != record["score_recipe"]:
        raise ValueError("Score recipe changed")
    expected = reorder(load_plan(pair / "D0-W0.npz"), scores, score_manifest)
    actual = load_plan(directory / "H2E.npz")
    if actual["manifest"] != expected["manifest"] or actual["manifest"] != record["plan"]:
        raise ValueError("Plan is not the exact descending order of the baseline sample uses")
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
        raise ValueError("Expected H2E/libero/VLM2/P1/D0/E1/T1/occurrence name and configuration")
    return int(match[1])


def check_checkpoint(directory, run):
    directory, run = Path(directory).resolve(), Path(run).resolve()
    record = check(directory)
    checkpoint = run / "checkpoints/003000/pretrained_model"
    config = source_compare.read_json(checkpoint / "train_config.json")
    factors = config["pdet"]
    check_name(run, factors, config["policy"]["num_vlm_prompt_tokens"], 0)
    if (config["seed"], config["steps"], config["batch_size"]) != (0, 3000, 32) or (
        not factors["enabled"] or Path(factors["order_plan"]).resolve() != directory / "H2E.npz"
    ):
        raise ValueError("Checkpoint changed the training budget, global seed or difficulty plan")
    if (config.get("keyframe_loss") or {}).get("enabled", False):
        raise ValueError("Order-only experiment must not add keyframe loss weighting")
    full_expert = run.name.startswith("pi05-hard2easy-aexpert-")
    if config["policy"].get("train_action_expert", False) != full_expert or config["policy"].get(
        "vlm_prompt_only"
    ) != (not full_expert):
        raise ValueError("Run name disagrees with saved action-expert training scope")
    validate_checkpoint(
        checkpoint, 2, source=Path(record["pair"]) / "libero", kind="base", train_action_expert=full_expert
    )
    return {"checkpoint": "passed", "run": str(run), "train_action_expert": full_expert}


def audit(directory, run):
    from lerobot.scripts.pdet_order_binding import inspect_run

    record = check(directory)
    reference = Path(record["reference_run"])
    check_checkpoint(directory, run)
    old, old_report = inspect_run(reference, name_checker=source_compare.check_name)
    new, new_report = inspect_run(run, name_checker=check_name)
    code = {}
    for label, invariant in (("reference", old), ("hard_to_easy", new)):
        code[label] = {key: invariant.pop(key) for key in ("git_commit", "source_tree_sha256")}
        invariant.pop("experiment")  # Repetition IDs only name outputs.
    # The optional expert run intentionally changes TWO factors versus the old
    # prompt-only/random-order reference. Normalize only these verified flags.
    full_expert = Path(run).name.startswith("pi05-hard2easy-aexpert-")
    for invariant, expected_expert in ((old, False), (new, full_expert)):
        policy = invariant["recipe"]["policy"]
        if policy.get("train_action_expert", False) != expected_expert or (
            policy.get("vlm_prompt_only") != (not expected_expert)
        ):
            raise ValueError("Recorded recipe disagrees with intended trainable scope")
        policy.pop("train_action_expert", None)
        policy["vlm_prompt_only"] = True
    if old != new:
        changed = [key for key in old.keys() | new.keys() if old.get(key) != new.get(key)]
        raise ValueError(
            f"Unexpected recipe/input changes beyond order and declared trainable scope: {changed}"
        )
    if old_report["bank_sha256"] != new_report["bank_sha256"]:
        raise ValueError("Realized per-occurrence noise/time bank changed")
    if old_report["order_sha256"] == new_report["order_sha256"]:
        raise ValueError("No realized order change")
    assert_same_processors(
        reference / "checkpoints/003000/pretrained_model", Path(run) / "checkpoints/003000/pretrained_model"
    )
    return {
        "audit": "inputs_and_recipe_passed",
        "reference": old_report,
        "hard_to_easy": new_report,
        "interventions": ["hard_to_easy_order"] + (["full_action_expert_training"] if full_expert else []),
        "code_revision": {
            **code,
            "status": "same_recorded_code"
            if code["reference"] == code["hard_to_easy"]
            else "different_not_certified",
        },
        "scope": "All steps: identical task slots, sample-use pool, P1, E1/T1 bank; recipe differs only in declared interventions. Historical code equivalence/numerical determinism are not certified.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--pair", type=Path, default=DEFAULT_ROOT / "source-pair-r1")
    prep.add_argument("--scores", type=Path, default=DEFAULT_ROOT / "frameskip-r1/scores.npz")
    prep.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    prep.add_argument("--output", type=Path, default=DEFAULT_ROOT / "hard2easy-r1")
    sub.add_parser("check").add_argument("directory", type=Path)
    for command in ("check-checkpoint", "audit"):
        cmd = sub.add_parser(command)
        cmd.add_argument("directory", type=Path)
        cmd.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.pair, args.scores, args.reference_run, args.output)
    elif args.command == "check":
        check(args.directory)
        result = {"preflight": "passed", "directory": str(args.directory)}
    elif args.command == "audit":
        result = audit(args.directory, args.run)
    else:
        result = check_checkpoint(args.directory, args.run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
