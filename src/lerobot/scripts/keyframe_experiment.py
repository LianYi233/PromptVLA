"""Prepare/audit key-action loss weighting on the original W0-Bocc finite plan (CPU only)."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from lerobot.scripts.pdet_order_binding import inspect_run
from lerobot.utils.keyframe_loss import (
    MODES,
    build_archive,
    chunk_masks,
    load_archive,
    normalized_key_mass,
    save_archive,
    shifted_masks,
    validate_order,
)
from lerobot.utils.pdet import tensor_hash
from lerobot.utils.pdet_order import load_plan, save_plan

NAME = re.compile(r"pi05-key-r([1-9][0-9]*)-(B|K2|R2|Kearly|Klate)-W0-Bocc-base-vlm2-P1-D0-E([01])-T([01])")
STAGES = {"baseline": ("B",), "weight": ("B", "K2", "R2"), "curriculum": MODES}


def check_name(run, factors, tokens, within_seed):
    match = NAME.fullmatch(Path(run).name)
    if match is None:
        raise ValueError(f"Not a keyframe experiment name: {run}")
    experiment, _, e, t = match.groups()
    if (
        within_seed,
        tokens,
        factors["prompt_seed"],
        factors["data_seed"],
        factors["noise_seed"],
        factors["time_seed"],
        factors.get("flow_binding"),
    ) != (0, 2, 1, 0, int(e), int(t), "occurrence") or e != t:
        raise ValueError("Run name does not match original W0/P1/D0/paired-bank configuration")
    return int(experiment)


def validate_prepared(directory, *, production=True):
    directory = Path(directory)
    arrays, manifest = load_archive(directory / "archive.npz")
    plan = load_plan(directory / "D0-W0.npz")
    validate_order(plan, manifest)
    info = plan["manifest"]
    if production and (
        info["data_seed"],
        info["steps"],
        info["batch_size"],
        info["world_size"],
        manifest["horizon"],
    ) != (0, 3000, 32, 2, 50):
        raise ValueError("Expected old D0/W0 plan: 3000 steps, 2 GPUs x batch32, chunk50")
    # Recompute the budget from actual ordered action targets, not trusted summary counts.
    batch = info["batch_size"] * info["world_size"]
    counts = []
    for start in range(0, len(plan["order"]), batch):
        key, valid = chunk_masks(
            arrays, plan["index"][plan["order"][start : start + batch]], manifest["horizon"]
        )
        counts.append((int(key.sum()), int(valid.sum())))
    if not np.array_equal(np.asarray(counts).T, np.stack((arrays["key_counts"], arrays["valid_counts"]))):
        raise ValueError("Saved schedule counts differ from actual W0 action chunks")
    return arrays, manifest, plan


def validate_current_inputs(manifest, root, source):
    from lerobot.scripts.frameskip_scores import file_inventory, read_actions, source_inventory
    from lerobot.utils.pdet_order import arrays_hash, read_frame_metadata

    recipe = manifest["score_recipe"]
    if str(root.resolve()) != recipe["dataset_root"] or str(source.resolve()) != recipe["source"]:
        raise ValueError("Keyframe experiment must use the scoring dataset/source")
    values = {
        "dataset_metadata_sha256": arrays_hash(read_frame_metadata(root)),
        "actions_sha256": arrays_hash({"action": read_actions(root)}),
        "stats_sha256": hashlib.sha256((root / "meta/stats.json").read_bytes()).hexdigest(),
        "source_inventory": source_inventory(source),
        "video_inventory_sha256": hashlib.sha256(
            json.dumps(file_inventory(root, sorted((root / "videos").rglob("*")))).encode()
        ).hexdigest(),
    }
    for key, value in json.loads(json.dumps(values)).items():
        if recipe[key] != value:
            raise ValueError(f"Dataset/source changed after scoring: {key}")


def prepare(scores_path, old_plan, output, dataset_root, source, random_seed=2026):
    from lerobot.scripts.frameskip_scores import check_inputs
    from lerobot.utils.frameskip_curriculum import load_scores

    if not old_plan.is_file():
        raise FileNotFoundError(
            f"Original orderbind W0 plan is required; do not replace with U.npz: {old_plan}"
        )
    check_inputs(scores_path, dataset_root, source)
    scores, score_manifest = load_scores(scores_path)
    plan = load_plan(old_plan)
    arrays, manifest = build_archive(scores, score_manifest, plan, random_seed=random_seed)
    output.mkdir(parents=True, exist_ok=True)
    save_plan(output / "D0-W0.npz", plan)
    save_archive(output / "archive.npz", arrays, manifest)
    validate_prepared(output)
    summary = {
        "prepared": "passed",
        "directory": str(output),
        "original_plan": str(old_plan),
        "source": str(source),
        "dataset_root": str(dataset_root),
        "event_window": "gripper switches: [-10,+2] frames, debounce=2; existing recorded mask",
        "event_frame_fraction": float(arrays["event_mask"].mean()),
        "key_target_fraction_in_W0": float(arrays["key_counts"].sum() / arrays["valid_counts"].sum()),
        "random_overlap_fraction": manifest["random_overlap_fraction"],
        "task_target_counts_key_valid": manifest["task_target_counts"],
        "schedules": {},
        "next": "Review existing event previews; run the two B baselines before K2/R2",
    }
    with (output / "schedule.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "update",
                "key_targets",
                "valid_targets",
                *[f"weight_{m}" for m in MODES],
                *[f"normalized_key_mass_{m}" for m in MODES],
            ]
        )
        for step, (key, valid) in enumerate(zip(arrays["key_counts"], arrays["valid_counts"], strict=True)):
            weights = [float(arrays[f"weight_{m}"][step]) for m in MODES]
            writer.writerow(
                [
                    step + 1,
                    int(key),
                    int(valid),
                    *weights,
                    *[float(normalized_key_mass(w, key, valid)) for w in weights],
                ]
            )
    for mode in MODES:
        weights = arrays[f"weight_{mode}"].astype(float)
        summary["schedules"][mode] = {
            "first": float(weights[0]),
            "last": float(weights[-1]),
            "sum_normalized_selected_mass": float(
                normalized_key_mass(weights, arrays["key_counts"], arrays["valid_counts"]).sum()
            ),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def inspect_weights(run, mode):
    arrays, manifest = load_archive(run / "keyframe_loss/archive.npz")
    info = manifest["order_plan"]
    pdet = json.loads((run / "pdet/manifest-rank0.json").read_text())
    plan = load_plan(pdet["factors"]["order_plan"])
    validate_order(plan, manifest)
    settings = pdet["recipe"].get("keyframe_loss", {})
    if settings.get("mode") != mode:
        raise ValueError("Saved training recipe does not match keyframe mode/name")
    bank = np.random.default_rng(manifest["random_seed"]).integers(
        1, 2**31, len(plan["order"]), dtype=np.int64
    )
    if not np.array_equal(bank, arrays["random_shifts"]):
        raise ValueError("Random control mask bank was changed")
    totals = np.zeros((info["steps"], 2), dtype=np.int64)
    expected_ranks = {f"manifest-rank{rank}.json" for rank in range(info["world_size"])}
    if {p.name for p in (run / "keyframe_loss").glob("manifest-rank*.json")} != expected_ranks:
        raise ValueError("Missing or extra keyframe rank manifests")
    for rank in range(info["world_size"]):
        metadata = json.loads((run / f"keyframe_loss/manifest-rank{rank}.json").read_text())
        if metadata != {"mode": mode, "archive": manifest, "rank": rank}:
            raise ValueError("Keyframe archive/mode differs across ranks")
        rows = [
            json.loads(line)
            for line in (run / f"keyframe_loss/inputs-rank{rank}.jsonl").read_text().splitlines()
        ]
        if [row["step"] for row in rows] != list(range(info["steps"])):
            raise ValueError("Incomplete/duplicate/out-of-order keyframe trace")
        for step, row in enumerate(rows):
            left = (step * info["world_size"] + rank) * info["batch_size"]
            ids = plan["order"][left : left + info["batch_size"]]
            indices = plan["index"][ids]
            key, valid = chunk_masks(arrays, indices, manifest["horizon"])
            selected = shifted_masks(key, valid, bank[ids]) if mode == "R2" else key
            strength = float(arrays[f"weight_{mode}"][step])
            weights = 1 + np.float32(strength - 1) * selected.astype(np.float32)
            expected = {
                "step": step,
                "rank": rank,
                "mode": mode,
                "occurrence_ids": ids.tolist(),
                "sample_ids": indices.tolist(),
                "key_mask_sha256": tensor_hash(torch.from_numpy(key)),
                "weight_sha256": tensor_hash(torch.from_numpy(weights)),
                "valid_targets": int(valid.sum()),
                "key_targets": int(key.sum()),
                "weighted_key_targets": int(selected.sum()),
                "strength": strength,
                "denominator_targets": float(np.where(valid, weights, 0).sum()),
            }
            for name, value in expected.items():
                if row.get(name) != value:
                    raise ValueError(f"Keyframe trace mismatch: {name}, {run.name}, rank{rank}, step{step}")
            totals[step] += [int(key.sum()), int(valid.sum())]
    if not np.array_equal(totals.T, np.stack((arrays["key_counts"], arrays["valid_counts"]))):
        raise ValueError("Realized keyframe schedule budget differs from saved counts")
    return manifest


def audit_runs(runs, stage="weight", allow_partial=False):
    reference, reports, cells, banks = None, [], set(), {}
    for run in map(Path, runs):
        invariant, report = inspect_run(run, name_checker=check_name)
        mode = NAME.fullmatch(run.name).group(2)
        archive = inspect_weights(run, mode)
        invariant["recipe"] = dict(invariant["recipe"])
        invariant["recipe"].pop("keyframe_loss", None)  # Only intended treatment differs.
        invariant.update(keyframe_archive=archive, order_sha256=report["order_sha256"])
        if reference is None:
            reference = invariant
        elif invariant != reference:
            raise ValueError("Changed source/recipe/initialization/order/keyframe archive across cells")
        bank = report["noise_seed"]
        if bank in banks and banks[bank] != report["bank_sha256"]:
            raise ValueError("Realized epsilon/time BANK differs between treatments")
        banks[bank] = report["bank_sha256"]
        cell = (mode, bank)
        if cell in cells:
            raise ValueError(f"Duplicate experimental cell: {cell}")
        cells.add(cell)
        reports.append({"mode": mode, **report})
    if not reports:
        raise ValueError("No runs selected")
    if len(banks) == 2 and len(set(banks.values())) != 2:
        raise ValueError("E0/T0 and E1/T1 did not change the recorded bank")
    expected = {(mode, bank) for mode in STAGES[stage] for bank in (0, 1)}
    complete = cells == expected
    if not allow_partial and not complete:
        raise ValueError(
            f"Incomplete {stage} comparison; missing {sorted(expected - cells)}, extra {sorted(cells - expected)}"
        )
    return {
        "audit": "passed" if complete else "partial_inputs_passed",
        "complete": complete,
        "stage": stage,
        "code_revision": reference["git_commit"],
        "runs": reports,
        "scope": "All realized steps: unchanged sample/flow pairs, source/recipe, masks, weighted denominators and normalized schedule budgets; not numerical determinism or historical code equivalence",
    }


def check_checkpoint(run, step=3000):
    config = json.loads(
        (run / "checkpoints" / f"{step:06d}" / "pretrained_model/train_config.json").read_text()
    )
    arrays, archive = load_archive(run / "keyframe_loss/archive.npz")
    check_name(
        run, config["pdet"], config["policy"]["num_vlm_prompt_tokens"], archive["order_plan"]["within_seed"]
    )
    mode = NAME.fullmatch(run.name).group(2)
    if (
        config["keyframe_loss"]["mode"] != mode
        or not config["pdet"]["enabled"]
        or config["seed"] != 0
        or config["steps"] != step
        or len(arrays["weight_B"]) != step
    ):
        raise ValueError("Checkpoint training configuration differs from keyframe experiment")
    return {"checkpoint": "name_and_config_match", "run": str(run)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument(
        "--scores", type=Path, default=Path("/data1/wyn/prompt-work/frameskip-r1/scores.npz")
    )
    prepare_parser.add_argument(
        "--old-plan", type=Path, default=Path("/data1/wyn/prompt-work/orderbind-r1-occ/D0-W0.npz")
    )
    prepare_parser.add_argument("--output", type=Path, default=Path("/data1/wyn/prompt-work/keyframe-r1"))
    prepare_parser.add_argument("--dataset-root", type=Path, default=Path("/home/wyn/datasets/libero"))
    prepare_parser.add_argument(
        "--source", type=Path, default=Path("/home/wyn/models/lerobot/pi05_libero_base")
    )
    prepare_parser.add_argument("--random-seed", type=int, default=2026)
    check = sub.add_parser("check")
    check.add_argument("directory", type=Path)
    check.add_argument("--dataset-root", type=Path)
    check.add_argument("--source", type=Path)
    audit = sub.add_parser("audit")
    audit.add_argument("runs", type=Path, nargs="+")
    audit.add_argument("--stage", choices=STAGES, default="weight")
    audit.add_argument("--allow-partial", action="store_true")
    checkpoint = sub.add_parser("check-checkpoint")
    checkpoint.add_argument("run", type=Path)
    checkpoint.add_argument("--step", type=int, default=3000)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.scores, args.old_plan, args.output, args.dataset_root, args.source, args.random_seed
        )
    elif args.command == "check":
        _, archive, _ = validate_prepared(args.directory)
        if (args.dataset_root is None) != (args.source is None):
            parser.error("--dataset-root and --source must be supplied together")
        if args.dataset_root is not None:
            validate_current_inputs(archive, args.dataset_root, args.source)
        result = {"preflight": "passed", "archive_sha256": archive["arrays_sha256"]}
    elif args.command == "check-checkpoint":
        result = check_checkpoint(args.run, args.step)
    else:
        result = audit_runs(args.runs, args.stage, args.allow_partial)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
