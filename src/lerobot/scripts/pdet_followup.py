"""Prepare fixed-task plans, audit realized order experiments and summarize training traces."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from lerobot.utils.pdet_order import (
    FLOW_PROTOCOL,
    create_plan,
    load_plan,
    read_frame_metadata,
    save_plan,
)


def prepare_plans(root, output, seeds, steps=3000, batch_size=32, world_size=2, data_seed=0):
    metadata = read_frame_metadata(root)
    result = []
    for seed in seeds:
        plan = create_plan(
            metadata,
            steps=steps,
            batch_size=batch_size,
            world_size=world_size,
            data_seed=data_seed,
            within_seed=seed,
        )
        path = Path(output) / f"D{data_seed}-W{seed}.npz"
        save_plan(path, plan)
        result.append({"path": str(path), **plan["manifest"]})
    return result


def read_run(run):
    directory = Path(run) / "pdet"
    files = sorted(directory.glob("manifest-rank*.json"))
    if not files:
        raise ValueError(f"No PDET manifests: {run}")
    manifests, rows = {}, {}
    for path in files:
        rank = int(path.stem.removeprefix("manifest-rank"))
        manifests[rank] = json.loads(path.read_text())
        for line in (directory / f"inputs-rank{rank}.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = row["step"], row["rank"]
            if row["rank"] != rank or (key in rows and rows[key] != row):
                raise ValueError(f"Inconsistent duplicate trace in {run}: {key}")
            rows[key] = row
    manifest = manifests[0]
    if set(manifests) != set(range(manifest["world_size"])):
        raise ValueError(f"Missing rank manifests: {run}")
    for rank, other in manifests.items():
        for key in (
            "protocol",
            "factors",
            "global_seed",
            "world_size",
            "per_device_batch_size",
            "tokens",
            "prompt_sha256",
            "dataset_metadata_sha256",
            "order_plan",
        ):
            if other.get(key) != manifest.get(key):
                raise ValueError(f"Inconsistent {key} across ranks in {run}")
        expected = set(range(min(manifest["factors"]["audit_steps"], manifest["total_steps"])))
        if {step for step, r in rows if r == rank} != expected:
            raise ValueError(f"Incomplete trace in {run}, rank {rank}")
    return manifest, [rows[key] for key in sorted(rows)]


def audit_order(runs):
    if len(runs) < 2:
        raise ValueError("Order audit needs at least two runs")
    reference = None
    reports = []
    for run in runs:
        manifest, rows = read_run(run)
        if manifest["protocol"] != FLOW_PROTOCOL:
            raise ValueError("Do not mix legacy step/rank PDET with occurrence-keyed order experiments")
        if manifest["factors"]["audit_steps"] < manifest["total_steps"]:
            raise ValueError("Order audit requires the complete realized training trace")
        plan = load_plan(manifest["factors"]["order_plan"])
        if plan["manifest"] != manifest["order_plan"]:
            raise ValueError("Recorded plan has changed on disk")
        slots, occurrences, realized_order = [], {}, []
        batch_size = manifest["per_device_batch_size"]
        for row in rows:
            ids = row["occurrence_ids"]
            start = (row["step"] * manifest["world_size"] + row["rank"]) * batch_size
            if len(ids) != batch_size or ids != plan["order"][start : start + batch_size].tolist():
                raise ValueError("Realized order differs from planned DDP slots")
            if row["task_index"] != plan["task_index"][start : start + batch_size].tolist():
                raise ValueError("Realized task schedule differs from plan")
            slots.extend(row["task_index"])
            realized_order.extend(ids)
            for i, occurrence in enumerate(ids):
                if occurrence in occurrences:
                    raise ValueError("An occurrence was trained twice")
                if row["sample_ids"][i] != int(plan["index"][occurrence]):
                    raise ValueError("Realized frame differs from occurrence identity")
                occurrences[occurrence] = (
                    row["sample_ids"][i],
                    row["occurrence_noise_sha256"][i],
                    row["occurrence_time_sha256"][i],
                )
        if len(occurrences) != len(plan["order"]):
            raise ValueError("Missing realized training occurrences")
        invariant = {
            key: manifest[key]
            for key in (
                "protocol",
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
            )
        }
        invariant["factors"] = {
            key: value for key, value in manifest["factors"].items() if key != "order_plan"
        }
        invariant["pool"] = plan["manifest"]["instance_pool_sha256"]
        if reference is None:
            reference = invariant, slots, occurrences, realized_order
        else:
            if invariant != reference[0] or slots != reference[1] or occurrences != reference[2]:
                raise ValueError(
                    "Order isolation failed: recipe, task slots, pool, P or occurrence-bound E/T changed"
                )
            if realized_order == reference[3]:
                raise ValueError("Order intervention had no effect")
        reports.append(
            {"run": str(run), "occurrences": len(occurrences), "within_seed": plan["manifest"]["within_seed"]}
        )
    return {
        "audit": "passed",
        "scope": "All realized task slots and occurrence-bound E/T; not numerical determinism",
        "runs": reports,
    }


def compare_inputs(original, repeated):
    """Compare original/repeated stochastic inputs without claiming recipe/code equivalence."""
    a, rows_a = read_run(original)
    b, rows_b = read_run(repeated)
    for key in (
        "protocol",
        "global_seed",
        "world_size",
        "per_device_batch_size",
        "tokens",
        "dtype",
        "prompt_sha256",
        "source_pretrained_path",
        "dataset_root",
        "versions",
        "cuda",
    ):
        if a[key] != b[key]:
            raise ValueError(f"Repeat changed {key}")
    for factor in ("prompt_seed", "data_seed", "noise_seed", "time_seed"):
        if a["factors"][factor] != b["factors"][factor]:
            raise ValueError(f"Repeat changed {factor}")
    lookup = {(row["step"], row["rank"]): row for row in rows_b}
    common = []
    for row in rows_a:
        other = lookup.get((row["step"], row["rank"]))
        if other is not None:
            for key in ("sample_ids_sha256", "noise_sha256", "time_sha256", "noise_shape", "time_shape"):
                if row[key] != other[key]:
                    raise ValueError(f"Repeat changed {key} at step={row['step']} rank={row['rank']}")
            common.append((row["step"], row["rank"]))
    if not common:
        raise ValueError("No overlapping recorded inputs")
    return {
        "audit": "matching_recorded_inputs",
        "original": str(original),
        "repeat": str(repeated),
        "matched_step_rank_pairs": len(common),
        "matched_steps": len({step for step, _ in common}),
        "recorded_commits": [a["git_commit"], b["git_commit"]],
        "scope": "Common recorded input prefix only; NOT recipe/code equivalence or numerical determinism",
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(run):
    run = Path(run)
    manifest, rows = read_run(run)
    output = run / "pdet_analysis"
    output.mkdir(exist_ok=True)
    cutoffs = [step for step in (50, 100, 250, 500, 1000, 2000, 3000) if step <= manifest["total_steps"]]
    coverage = []
    previous = 0
    for stop in cutoffs:
        for label, start in (("cumulative", 0), ("window", previous)):
            selected = [row for row in rows if start <= row["step"] < stop]
            if len(selected) != (stop - start) * manifest["world_size"]:
                raise ValueError(f"Coverage needs full traces through step {stop}")
            ids = [value for row in selected for value in row["sample_ids"]]
            tasks = Counter(value for row in selected for value in row["task_index"])
            phases = Counter(value for row in selected for value in row["phase_bin"])
            episodes = {value for row in selected for value in row["episode_index"]}
            event_count = sum(value for row in selected for value in row["gripper_change"])
            coverage.append(
                {
                    "scope": label,
                    "start_step": start,
                    "end_step": stop,
                    "samples": len(ids),
                    "unique_frames": len(set(ids)),
                    "repeated_frame_uses": len(ids) - len(set(ids)),
                    "unique_episodes": len(episodes),
                    "unique_tasks": len(tasks),
                    "gripper_change_samples": event_count,
                    **{f"task_{key}": value for key, value in sorted(tasks.items())},
                    **{f"temporal_quartile_{key}": value for key, value in sorted(phases.items())},
                }
            )
        previous = stop
    write_csv(output / "coverage.csv", coverage)
    curve = []
    diagnostics = run / "training_diagnostics.jsonl"
    if diagnostics.exists():
        for line in diagnostics.read_text().splitlines():
            row = json.loads(line)
            curve.append(
                {
                    "step": row["step"],
                    **row["train"],
                    **{
                        key: value["mean"]
                        for key, value in row["diagnostics_rank0"].items()
                        if key.startswith("prompt_update/")
                    },
                }
            )
    write_csv(output / "training_curve.csv", curve)
    rank = run / "prompt_rank/metrics.jsonl"
    if rank.exists():
        write_csv(
            output / "rank_and_fixed_probe_loss.csv",
            [json.loads(line) for line in rank.read_text().splitlines()],
        )
    report = {
        "run": str(run),
        "protocol": manifest["protocol"],
        "output": str(output),
        "coverage": "Temporal quartiles and gripper-coordinate changes, NOT annotated contact/grasp phases",
        "probe_loss": "Fixed training observations; not held-out validation or rollout success",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--within-seeds", type=int, nargs="+", default=[0, 1])
    prepare.add_argument("--data-seed", type=int, default=0)
    prepare.add_argument("--steps", type=int, default=3000)
    prepare.add_argument("--batch-size", type=int, default=32)
    prepare.add_argument("--world-size", type=int, default=2)
    sub.add_parser("audit-order").add_argument("runs", type=Path, nargs="+")
    sub.add_parser("summarize").add_argument("runs", type=Path, nargs="+")
    compare = sub.add_parser("compare-inputs")
    compare.add_argument("original", type=Path)
    compare.add_argument("repeat", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_plans(
            args.dataset_root,
            args.output,
            args.within_seeds,
            args.steps,
            args.batch_size,
            args.world_size,
            args.data_seed,
        )
    elif args.command == "audit-order":
        result = audit_order(args.runs)
    elif args.command == "compare-inputs":
        result = compare_inputs(args.original, args.repeat)
    else:
        result = [summarize(run) for run in args.runs]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
