"""Prepare/audit FrameSkip-inspired curricula and realized occurrence-bound training."""

import argparse
import csv
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np

from lerobot.scripts.pdet_followup import read_run
from lerobot.utils.frameskip_curriculum import (
    PROTOCOL,
    SCHEDULES,
    load_scores,
    prepare_plans,
    source_mask,
    validate_family,
)
from lerobot.utils.pdet_order import FLOW_PROTOCOL, META_COLUMNS, load_plan

NAME = re.compile(r"pi05-fskip-r([1-9][0-9]*)-([URSC])-Bocc-base-vlm2-P1-D0-E([01])-T([01])")
SHARED = (
    "global_seed",
    "world_size",
    "per_device_batch_size",
    "tokens",
    "dtype",
    "prompt_sha256",
    "source_pretrained_path",
    "dataset_root",
    "dataset_metadata_sha256",
    "versions",
    "cuda",
    "recipe",
    "git_commit",
    "source_tree_sha256",
    "total_steps",
    "flow_key_namespace",
)


def check_plan(plan, schedule, *, production=True):
    info = plan["manifest"]
    curriculum = info.get("curriculum", {})
    if curriculum.get("protocol") != PROTOCOL or curriculum.get("schedule") != schedule:
        raise ValueError("Missing curriculum protocol or schedule label mismatch")
    warm = curriculum["warmup_steps"] * info["batch_size"] * info["world_size"]
    if curriculum["warmup_occurrences"] != warm or not 0 < warm < len(plan["order"]):
        raise ValueError("Invalid warmup length")
    expected = 0 if schedule == "U" else (len(plan["order"]) - warm) // 2
    if int(source_mask(plan).sum()) != expected:
        raise ValueError("Selected-source budget differs from the fixed 50% design")
    if production and (
        info["steps"],
        info["batch_size"],
        info["world_size"],
        info["data_seed"],
        curriculum["warmup_steps"],
    ) != (3000, 32, 2, 0, 500):
        raise ValueError(
            "This training recipe requires 3000 steps, 2 GPUs x batch32, D0 and 500 warmup steps"
        )


def audit_plans(directory, scores_path, *, production=True):
    plans = {label: load_plan(Path(directory) / f"{label}.npz") for label in SCHEDULES}
    validate_family(plans)
    scores, score_manifest = load_scores(scores_path)
    report = []
    for label, plan in plans.items():
        check_plan(plan, label, production=production)
        info = plan["manifest"]
        if info["curriculum"]["score_recipe"] != score_manifest:
            raise ValueError("Score archive differs from the recorded scoring recipe")
        for key in META_COLUMNS:
            if not np.array_equal(plan[key], scores[key][plan["index"]]):
                raise ValueError(f"Plan contains incorrect dataset identity: {key}")
        origin = source_mask(plan)
        field = "random_selected" if label == "R" else "selected"
        if not np.all(scores[field][plan["index"][origin]]):
            raise ValueError("A selected-source occurrence is outside its declared pool")
        global_batch = info["batch_size"] * info["world_size"]
        warm = info["curriculum"]["warmup_steps"]
        # Split at warmup as well as regular windows, including in small CPU tests.
        cuts = sorted({0, warm, info["steps"], *range(250, info["steps"], 250)})
        for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
            ids = plan["order"][start * global_batch : stop * global_batch]
            frames = plan["index"][ids]
            values = scores["score"][frames]
            report.append(
                {
                    "schedule": label,
                    "start_step": start,
                    "end_step": stop,
                    "uses": len(ids),
                    "selected_source_fraction": float(origin[ids].mean()),
                    "actual_scored_key_fraction": float(scores["selected"][frames].mean()),
                    "actual_own_key_fraction": float(scores[field][frames].mean()),
                    "score_mean": float(values.mean()),
                    "score_q10": float(np.quantile(values, 0.1)),
                    "score_q50": float(np.quantile(values, 0.5)),
                    "score_q90": float(np.quantile(values, 0.9)),
                    "unique_frames": len(np.unique(frames)),
                    "repeated_uses": len(frames) - len(np.unique(frames)),
                    "unique_episodes": len(np.unique(plan["episode_index"][ids])),
                    "cumulative_unique_frames": len(
                        np.unique(plan["index"][plan["order"][: stop * global_batch]])
                    ),
                }
            )
    with tempfile.NamedTemporaryFile(
        mode="w", dir=directory, suffix=".csv", delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    Path(handle.name).replace(Path(directory) / "schedule_report.csv")
    return {
        "audit": "plans_passed",
        "warmup": "identical uses and IDs",
        "S_C": "same task slots, sample-use multiset and IDs; different order",
        "report": str(Path(directory) / "schedule_report.csv"),
    }


def check_checkpoint(run, step=3000):
    run = Path(run)
    match = NAME.fullmatch(run.name)
    if match is None:
        raise ValueError("Unexpected FrameSkip run name")
    _, schedule, e, t = match.groups()
    config = json.loads(
        (run / "checkpoints" / f"{step:06d}" / "pretrained_model/train_config.json").read_text()
    )
    factors = config["pdet"]
    if not factors["enabled"] or factors["flow_binding"] != "occurrence":
        raise ValueError("Expected enabled occurrence-bound PDET")
    if (factors["prompt_seed"], factors["data_seed"], factors["noise_seed"], factors["time_seed"]) != (
        1,
        0,
        int(e),
        int(t),
    ) or e != t:
        raise ValueError("Run name and P/D/E/T mismatch")
    if config["seed"] != 0 or config["steps"] != step or config["policy"]["num_vlm_prompt_tokens"] != 2:
        raise ValueError("Unexpected training seed/steps/tokens")
    plan = load_plan(factors["order_plan"])
    check_plan(plan, schedule)
    return {"checkpoint": "name_and_config_match", "run": str(run)}


def audit_runs(runs, *, allow_partial=False, production=True):
    reference, cells, reports, banks, warmups, pools, orders = None, set(), [], {}, {}, {}, {}
    expected_shapes = None
    for run in runs:
        match = NAME.fullmatch(Path(run).name)
        if match is None:
            raise ValueError(f"Unexpected FrameSkip run name: {run}")
        experiment, schedule, e, t = match.groups()
        manifest, rows = read_run(run)
        factors = manifest["factors"]
        if (factors["prompt_seed"], factors["data_seed"], factors["noise_seed"], factors["time_seed"]) != (
            1,
            0,
            int(e),
            int(t),
        ) or e != t:
            raise ValueError("Run name and P/D/E/T mismatch")
        if (
            manifest["protocol"] != FLOW_PROTOCOL
            or factors["flow_binding"] != "occurrence"
            or manifest["flow_key_namespace"] != FLOW_PROTOCOL
        ):
            raise ValueError("Expected occurrence-bound CPU noise/time bank")
        if (
            factors["audit_steps"] < manifest["total_steps"]
            or not manifest.get("git_commit")
            or not manifest.get("source_tree_sha256")
        ):
            raise ValueError("Complete traces and code provenance are required")
        for path in (Path(run) / "pdet").glob("manifest-rank*.json"):
            other = json.loads(path.read_text())
            if any(other.get(key) != manifest.get(key) for key in SHARED):
                raise ValueError("Different runtime/source across ranks")
        plan = load_plan(factors["order_plan"])
        check_plan(plan, schedule, production=production)
        info = plan["manifest"]
        if info != manifest["order_plan"]:
            raise ValueError("Recorded plan was modified")
        if info["dataset_metadata_sha256"] != manifest["dataset_metadata_sha256"]:
            raise ValueError("Dataset identity differs from plan")
        if (info["steps"], info["batch_size"], info["world_size"]) != (
            manifest["total_steps"],
            manifest["per_device_batch_size"],
            manifest["world_size"],
        ):
            raise ValueError("Plan and runtime batch/world size/steps differ")
        invariant = {key: manifest[key] for key in SHARED}
        invariant.update(
            experiment=experiment,
            task_slots=info["task_schedule_sha256"],
            score_recipe=info["curriculum"]["score_recipe"],
        )
        invariant["factors"] = {
            k: v for k, v in factors.items() if k not in ("order_plan", "noise_seed", "time_seed")
        }
        if reference is None:
            reference = invariant
        elif reference != invariant:
            raise ValueError("Changed recipe/source/P/D/scoring/task slots across runs")
        cell = schedule, e
        if cell in cells:
            raise ValueError(f"Duplicate experimental cell: {cell}")
        cells.add(cell)
        count = len(plan["order"])
        bank = [None] * count
        warm_rows = []
        batch = info["batch_size"]
        warm = info["curriculum"]["warmup_occurrences"]
        for row in rows:
            start = (row["step"] * info["world_size"] + row["rank"]) * batch
            ids = row["occurrence_ids"]
            if (
                len(ids) != batch
                or ids != plan["order"][start : start + batch].tolist()
                or row.get("flow_key_ids") != ids
            ):
                raise ValueError("Realized occurrence order/noise binding differs from plan")
            if row["noise_shape"][0] != batch or row["time_shape"] != [batch]:
                raise ValueError("Invalid realized noise/time shape")
            shapes = (row["noise_shape"], row["time_shape"])
            if expected_shapes is None:
                expected_shapes = shapes
            if shapes != expected_shapes:
                raise ValueError("Flow shapes changed across steps/ranks/runs")
            for key in META_COLUMNS:
                if row["sample_ids" if key == "index" else key] != plan[key][ids].tolist():
                    raise ValueError(f"Realized {key} differs from plan")
            for factor in ("noise", "time"):
                if len(row[f"occurrence_{factor}_sha256"]) != batch:
                    raise ValueError("Missing per-use flow hashes")
            for i, occurrence in enumerate(ids):
                hashes = [row[f"occurrence_{f}_sha256"][i] for f in ("noise", "time")]
                if bank[occurrence] is not None or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
                    raise ValueError("Repeated occurrence or invalid flow hash")
                bank[occurrence] = bytes.fromhex("".join(hashes))
                if start + i < warm:
                    warm_rows.append((row["sample_ids"][i], occurrence, *hashes))
        if any(value is None for value in bank):
            raise ValueError("Incomplete realized bank")
        digest = hashlib.sha256(b"".join(bank)).hexdigest()
        warm_digest = hashlib.sha256(json.dumps(warm_rows).encode()).hexdigest()
        if e in banks and banks[e] != digest:
            raise ValueError("Same E/T label produced different realized noise/time bank")
        if e in warmups and warmups[e] != warm_digest:
            raise ValueError("Warmup data/noise differs across schedules")
        banks[e], warmups[e] = digest, warm_digest
        pool = info["instance_pool_sha256"]
        if schedule in pools and pools[schedule] != pool:
            raise ValueError("Sample pool changed across noise banks")
        order_hash = hashlib.sha256(plan["order"].tobytes()).hexdigest()
        if schedule in orders and orders[schedule] != order_hash:
            raise ValueError("Order changed across noise banks")
        pools[schedule], orders[schedule] = pool, order_hash
        reports.append(
            {
                "run": str(run),
                "schedule": schedule,
                "bank": int(e),
                "uses": count,
                "bank_sha256": digest,
                "pool_sha256": pool,
                "order_sha256": order_hash,
            }
        )
    if not reports:
        raise ValueError("No runs supplied")
    if "S" in pools and "C" in pools and (pools["S"] != pools["C"] or orders["S"] == orders["C"]):
        raise ValueError("S/C must have identical sample uses and different orders")
    complete = cells == {(s, str(b)) for s in SCHEDULES for b in (0, 1)}
    if not complete and not allow_partial:
        raise ValueError("Expected all eight schedule x bank cells; use --allow-partial for a subset")
    if len(banks) == 2 and len(set(banks.values())) != 2:
        raise ValueError("Different bank seeds produced identical draws")
    return {
        "audit": "passed" if complete else "partial_inputs_passed",
        "complete": complete,
        "scope": "All realized uses: fixed task slots/shared warmup; S/C matched data and occurrence noise; not numerical determinism",
        "runs": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--scores", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit-plans")
    audit.add_argument("directory", type=Path)
    audit.add_argument("--scores", type=Path, required=True)
    inputs = sub.add_parser("check-inputs")
    inputs.add_argument("--scores", type=Path, required=True)
    inputs.add_argument("--dataset-root", type=Path, required=True)
    inputs.add_argument("--source", type=Path, required=True)
    runs = sub.add_parser("audit-runs")
    runs.add_argument("runs", type=Path, nargs="+")
    runs.add_argument("--allow-partial", action="store_true")
    check = sub.add_parser("check-checkpoint")
    check.add_argument("run", type=Path)
    check.add_argument("--step", type=int, default=3000)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_plans(args.scores, args.output)
        result = audit_plans(args.output, args.scores)
    elif args.command == "audit-plans":
        result = audit_plans(args.directory, args.scores)
    elif args.command == "audit-runs":
        result = audit_runs(args.runs, allow_partial=args.allow_partial)
    elif args.command == "check-inputs":
        from lerobot.scripts.frameskip_scores import check_inputs

        result = check_inputs(args.scores, args.dataset_root, args.source)
    else:
        result = check_checkpoint(args.run, args.step)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
