"""Lightweight PDET initialization preflight and cross-run random-input audit."""

import argparse
import json
from itertools import combinations
from pathlib import Path

from lerobot.utils.prompt_initialization import read_initial_prompt


def check_initializations(directory):
    hashes = []
    for seed in (0, 1):
        _, metadata = read_initial_prompt(Path(directory) / f"pi05-base-vlm2-init-seed{seed}.safetensors", 2)
        if metadata["source_training_seed"] != seed or metadata.get("source_prompt_init_path"):
            raise ValueError(f"P{seed} provenance must refer to an original, unswapped seed-{seed} prompt")
        hashes.append(metadata["prompt_sha256"])
    if hashes[0] == hashes[1]:
        raise ValueError("P0 and P1 have identical tensors; this cannot test initialization sensitivity")
    return {f"P{seed}": digest for seed, digest in enumerate(hashes)}


def audit_runs(runs):
    records = []
    for run in runs:
        directory = Path(run) / "pdet"
        manifests = sorted(directory.glob("manifest-rank*.json"))
        if not manifests:
            raise ValueError(f"No PDET manifests: {run}")
        for path in manifests:
            manifest = json.loads(path.read_text())
            rank = int(path.stem.removeprefix("manifest-rank"))
            inputs = [
                json.loads(line) for line in (directory / f"inputs-rank{rank}.jsonl").read_text().splitlines()
            ]
            by_step = {}
            for row in inputs:
                if row["rank"] != rank or (row["step"] in by_step and by_step[row["step"]] != row):
                    raise ValueError(f"Inconsistent repeated audit step/rank in {run}")
                by_step[row["step"]] = row
            if set(by_step) != set(range(manifest["factors"]["audit_steps"])):
                raise ValueError(f"Incomplete initial-step audit: {path}")
            records.append((str(run), rank, manifest, by_step))
        if {int(p.stem.removeprefix("manifest-rank")) for p in manifests} != set(
            range(manifest["world_size"])
        ):
            raise ValueError(f"Missing rank manifests: {run}")
    comparisons = 0
    for (run_a, rank_a, a, rows_a), (run_b, rank_b, b, rows_b) in combinations(records, 2):
        if rank_a != rank_b:
            continue
        for key in (
            "protocol",
            "global_seed",
            "world_size",
            "per_device_batch_size",
            "tokens",
            "dtype",
            "source_pretrained_path",
            "dataset_root",
            "versions",
            "git_commit",
            "cuda",
            "recipe",
        ):
            if a[key] != b[key]:
                raise ValueError(f"Different {key}: {run_a} vs {run_b}")
        if (a["factors"]["prompt_seed"] == b["factors"]["prompt_seed"]) != (
            a["prompt_sha256"] == b["prompt_sha256"]
        ):
            raise ValueError(f"P labels/tensors disagree: {run_a} vs {run_b}")
        for factor, key in (
            ("data_seed", "sample_ids_sha256"),
            ("noise_seed", "noise_sha256"),
            ("time_seed", "time_sha256"),
        ):
            steps = sorted(set(rows_a) & set(rows_b))
            equal = all(rows_a[step][key] == rows_b[step][key] for step in steps)
            if equal != (a["factors"][factor] == b["factors"][factor]):
                raise ValueError(
                    f"Random factor isolation failed for {factor}: {run_a} vs {run_b}, rank {rank_a}"
                )
        comparisons += 1
    if comparisons == 0:
        raise ValueError("Need at least two runs for a cross-run factor audit")
    return {
        "audit": "passed",
        "run_pairs_per_rank": comparisons,
        "scope": "Initial audited inputs only; not a proof of complete numerical determinism",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-inits").add_argument("directory", type=Path)
    sub.add_parser("audit").add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    result = check_initializations(args.directory) if args.command == "check-inits" else audit_runs(args.runs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
