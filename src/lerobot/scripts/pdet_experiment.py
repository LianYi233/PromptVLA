"""Lightweight PDET initialization preflight and cross-run random-input audit."""

import argparse
import json
import re
import subprocess
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


def _non_training_path(path):
    """Conservatively permit documentation, tests and standalone evaluation/audit entry points."""
    return (
        path in {"README.md", "README_LEROBOT.md", "AGENTS.md", "AGENT_GUIDE.md"}
        or path.startswith(("docs/", "tests/"))
        or (path.startswith("examples/analysis/") and path.endswith(".md"))
        or ("/" not in path and path.startswith("run_eval_libero") and path.endswith(".sh"))
        or path == "src/lerobot/scripts/pdet_experiment.py"
    )


def audit_code_revisions(records, repo_root=None, *, inputs_only=False):
    commits = sorted({row[2].get("git_commit") for row in records}, key=str)
    report = {
        "commits": commits,
        "scope": "Recorded Git trees only; historical uncommitted changes were not captured",
    }
    if inputs_only:
        return {**report, "status": "not_verified"}
    if any(not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40,64}", sha) for sha in commits):
        raise ValueError(
            f"Missing/invalid recorded git_commit: {commits}. "
            "Use --inputs-only for input diagnostics without claiming code equivalence."
        )
    if len(commits) == 1:
        return {**report, "status": "same_recorded_commit"}
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    comparisons = []
    for other in commits[1:]:
        try:
            # Commits are validated hex identifiers; never fetch or modify the checkout.
            result = subprocess.run(
                [
                    "git", "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                    "--name-only", "-z", f"{commits[0]}^{{commit}}", f"{other}^{{commit}}", "--",
                ],
                cwd=repo_root, capture_output=True, check=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(
                f"Cannot inspect recorded Git revisions {commits[0]} vs {other} locally. "
                "Fetch the recorded history, or use --inputs-only for input diagnostics only."
            ) from error
        paths = [p.decode("utf-8", errors="surrogateescape") for p in result.stdout.split(b"\0") if p]
        protected = [path for path in paths if not _non_training_path(path)]
        if protected:
            raise ValueError(
                f"Different git_commit with training-relevant or unclassified changes: "
                f"{commits[0]} vs {other}: {protected}. "
                "Review these changes; --inputs-only checks random inputs without approving code equivalence."
            )
        comparisons.append({"from": commits[0], "to": other, "changed_paths": paths})
    return {**report, "status": "non_training_changes_only", "comparisons": comparisons}


def audit_runs(runs, *, repo_root=None, inputs_only=False):
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
    # Finish configuration/input checks first. Different commit IDs alone do not
    # establish a training-code change, e.g. adding an evaluation launcher mid-sweep.
    code_revision = audit_code_revisions(records, repo_root, inputs_only=inputs_only)
    return {
        "audit": "inputs_only_passed" if inputs_only else "passed",
        "random_inputs": "passed",
        "code_revision": code_revision,
        "run_pairs_per_rank": comparisons,
        "scope": "Initial audited inputs only; not a proof of complete numerical determinism",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-inits").add_argument("directory", type=Path)
    audit = sub.add_parser("audit")
    audit.add_argument("runs", nargs="+", type=Path)
    audit.add_argument(
        "--inputs-only", action="store_true",
        help="Check configuration/random inputs only; report code revision as NOT verified",
    )
    args = parser.parse_args()
    result = (
        check_initializations(args.directory)
        if args.command == "check-inits"
        else audit_runs(args.runs, inputs_only=args.inputs_only)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
