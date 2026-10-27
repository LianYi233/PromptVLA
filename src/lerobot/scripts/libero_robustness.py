"""Isolated PRO/PLUS evaluation: explicit fork, task language, coverage and resume identity."""

import argparse
import functools
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PRO_DIMENSIONS = ("object", "swap", "lan", "task", "env")
PLUS_COUNTS = dict(zip(SUITES, (2402, 2518, 2591, 2519), strict=True))
HORIZONS = dict(zip(SUITES, (220, 280, 300, 520), strict=True))
PROTOCOL = "promptvla-libero-robustness-v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def selected_suites(kind, mode, dimensions="all"):
    if kind not in ("pro", "plus") or mode not in ("all", *SUITES):
        raise ValueError("Choose pro/plus and all or a standard base suite")
    bases = SUITES if mode == "all" else (mode,)
    if kind == "plus":
        if dimensions != "all":
            raise ValueError("PRO_DIMENSIONS applies only to PRO")
        return list(bases)
    dims = PRO_DIMENSIONS if dimensions == "all" else tuple(dimensions.split(","))
    if not dims or len(set(dims)) != len(dims) or any(d not in PRO_DIMENSIONS for d in dims):
        raise ValueError("PRO_DIMENSIONS must be all or comma-separated object,swap,lan,task,env")
    return [f"{base}_{dim}" for base in bases for dim in dims]


def activate_fork(root, config_dir):
    """Bind the namespace in this process, without pip uninstall or changing ~/.libero."""
    root, config_dir = Path(root).resolve(), Path(config_dir).resolve()
    package = root / "libero/libero"
    for path in (
        package / "__init__.py",
        package / "benchmark/__init__.py",
        package / "assets/scenes/libero_tabletop_base_style.xml",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark code/assets missing: {path}; see README_libero_robustness.md")
    config = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "assets": str(package / "assets"),
        "datasets": str(root / "datasets"),
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "config.yaml"
    # JSON is YAML-compatible; all processes use the exact same immutable config.
    with path.open("a+") as f:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        old = f.read()
        if old and json.loads(old) != config:
            raise ValueError("Benchmark config changed; choose a new output directory")
        if not old:
            f.write(json.dumps(config, indent=2) + "\n")
    loaded = sys.modules.get("libero.libero")
    if loaded is not None and Path(loaded.__file__).resolve() != package / "__init__.py":
        raise RuntimeError("Another LIBERO fork is already imported; start a fresh evaluator process")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    if loaded is None:
        namespace = types.ModuleType("libero")
        namespace.__path__ = [str(root / "libero")]
        namespace.__spec__ = importlib.machinery.ModuleSpec("libero", loader=None, is_package=True)
        namespace.__spec__.submodule_search_locations = namespace.__path__
        sys.modules["libero"] = namespace
    importlib.import_module("libero.libero")
    return package


def task_files(task, package, kind):
    bddl = package / "bddl_files" / task.problem_folder / task.bddl_file
    # PLUS encodes camera/robot/noise in a virtual BDDL filename; its renderer
    # reads the base BDDL but MUST still receive the original virtual filename.
    if kind == "plus" and "_view_" in bddl.name and "_initstate_" in bddl.name:
        bddl = bddl.with_name(bddl.name.split("_view_", 1)[0] + ".bddl")
    filename = Path(task.init_states_file)
    if kind == "plus" and ("_add_" in filename.name or "_level" in filename.name):
        initial = package / "init_files/libero_newobj" / task.problem_folder / filename.name
    elif kind == "plus":
        stem = re.sub(r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+", "", filename.stem)
        initial = package / "init_files" / task.problem_folder / (stem + filename.suffix)
    else:
        initial = package / "init_files" / task.problem_folder / filename.name
    for path in (bddl, initial):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing benchmark task resource: {path}")
    return bddl, initial


def build_manifest(root, config_dir, kind, mode, dimensions, episodes, seed):
    if episodes < 1 or seed < 0 or (kind == "plus" and episodes != 1):
        raise ValueError("PLUS requires 1 episode/variant; PRO episodes must be positive; seed >= 0")
    suites = selected_suites(kind, mode, dimensions)
    package = activate_fork(root, config_dir)
    from libero.libero import benchmark
    from libero.libero.envs.bddl_utils import get_problem_info

    classification = {}
    if kind == "plus":
        labels = json.loads((package / "benchmark/task_classification.json").read_text())
        classification = {s: {x["name"]: x["category"] for x in labels[s]} for s in suites}
    files, tasks = {}, []
    for suite_name in suites:
        if suite_name not in benchmark.get_benchmark_dict():
            raise ValueError(f"Wrong/incomplete {kind} fork: suite {suite_name} is not registered")
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        expected = PLUS_COUNTS[suite_name] if kind == "plus" else 10
        if len(suite.tasks) != expected:
            raise ValueError(f"Wrong task coverage for {suite_name}: {len(suite.tasks)} != {expected}")
        for task_id, task in enumerate(suite.tasks):
            bddl, initial = task_files(task, package, kind)
            for path in (bddl, initial):
                key = str(path.relative_to(package))
                if key not in files:
                    files[key] = digest(path)
            instruction = get_problem_info(str(bddl))["language_instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Empty/invalid BDDL instruction: {bddl}")
            category = (
                classification[suite_name].get(task.name) if kind == "plus" else suite_name.rsplit("_", 1)[1]
            )
            if not category:
                raise ValueError(f"Missing PLUS category for {suite_name}/{task.name}")
            tasks.append(
                {
                    "suite": suite_name,
                    "task_id": task_id,
                    "name": task.name,
                    "instruction": instruction,
                    "category": category,
                }
            )
    source = [[str(p.relative_to(package)), digest(p)] for p in sorted(package.rglob("*.py"))]
    assets = [
        [str(p.relative_to(package)), p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted((package / "assets").rglob("*"))
        if p.is_file()
    ]
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    versions = {}
    for name in ("torch", "numpy", "mujoco", "robosuite", "transformers", "scikit-image", "Wand"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "protocol": PROTOCOL,
        "benchmark": kind,
        "root": str(Path(root).resolve()),
        "config_dir": str(Path(config_dir).resolve()),
        "mode": mode,
        "dimensions": dimensions,
        "episodes_per_task": episodes,
        "seed": seed,
        "suites": suites,
        "tasks": tasks,
        "expected_episodes": len(tasks) * episodes,
        "image_size": 224,
        "horizons": HORIZONS,
        "instruction_source": "BDDL language_instruction",
        "task_seed_rule": "sha256(seed,suite,task_name) first32bits modulo 2**31; reset before each task",
        "videos": False,
        "git_commit": result.stdout.strip() if result.returncode == 0 else None,
        "source_sha256": json_hash(source),
        "task_files_sha256": json_hash(files),
        "assets_inventory_sha256": json_hash(assets),
        "versions": versions,
        "helper_sha256": digest(__file__),
    }


def prepare(root, directory, kind, mode, dimensions, episodes, seed):
    directory = Path(directory).resolve()
    manifest = build_manifest(root, directory / "config", kind, mode, dimensions, episodes, seed)
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Benchmark inputs/protocol changed; choose a new BASE_OUTPUT")
    else:
        # Protocol directories are unique per benchmark/suite/seed/episode count.
        with path.open("x") as f:
            json.dump(manifest, f, indent=2)
    return manifest


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unknown robustness protocol")
    return manifest


def install(evaluator, path):
    manifest = read_manifest(path)
    current = build_manifest(
        manifest["root"],
        manifest["config_dir"],
        manifest["benchmark"],
        manifest["mode"],
        manifest["dimensions"],
        manifest["episodes_per_task"],
        manifest["seed"],
    )
    if current != manifest:
        raise ValueError("Benchmark changed since preflight")
    from lerobot.envs import libero as env_module
    from lerobot.utils.random_utils import set_seed

    if manifest["benchmark"] == "plus":
        import numpy as np

        # Upstream fog uses the removed NumPy 1.x alias; float64 is identical.
        if "float_" not in vars(np):
            np.float_ = np.float64
    by_suite = defaultdict(dict)
    for row in manifest["tasks"]:
        by_suite[row["suite"]][row["task_id"]] = row
    original_suite = env_module._get_suite

    def get_suite(name):
        suite = original_suite(name)
        rows = by_suite[name]
        for i, task in enumerate(suite.tasks):
            if task.name != rows[i]["name"]:
                raise ValueError("Benchmark task order changed")
        # PRO language/task BDDLs may retain the original filename. Reading the
        # instruction from filenames would silently evaluate the wrong command.
        suite.tasks = [task._replace(language=rows[i]["instruction"]) for i, task in enumerate(suite.tasks)]
        return suite

    env_module._get_suite = get_suite
    for name in by_suite:
        base = name if manifest["benchmark"] == "plus" else name.rsplit("_", 1)[0]
        env_module.TASK_SUITE_MAX_STEPS[name] = HORIZONS[base]
    original_one = evaluator.run_one

    @functools.wraps(original_one)
    def run_one(task_group, task_id, env, **kwargs):
        row = by_suite[task_group][task_id]
        seed = int(json_hash([manifest["seed"], task_group, row["name"]])[:8], 16) % (2**31)
        set_seed(seed)
        kwargs.update(max_episodes_rendered=0, videos_dir=None, start_seed=seed)
        return original_one(task_group, task_id, env, **kwargs)

    evaluator.run_one = run_one
    return {
        "robustness": {
            "manifest_sha256": json_hash(manifest),
            "benchmark": manifest["benchmark"],
            "expected_episodes": manifest["expected_episodes"],
        }
    }


def summarize(manifest_path, result_path):
    manifest = read_manifest(manifest_path)
    data = json.loads(Path(result_path).read_text())
    rows = data.get("per_task") or data.get("Aggregated Metrics for per_task")
    if not isinstance(rows, list):
        raise ValueError("No per-task evaluation results")
    expected = {(x["suite"], x["task_id"]): x for x in manifest["tasks"]}
    seen, suites, categories, overall = set(), defaultdict(list), defaultdict(list), []
    for row in rows:
        key = row["task_group"], row["task_id"]
        if key not in expected or key in seen:
            raise ValueError("Unexpected/duplicate result task")
        seen.add(key)
        values = row.get("metrics", row)["successes"]
        if len(values) != manifest["episodes_per_task"] or any(type(v) is not bool for v in values):
            raise ValueError("Incomplete/nonboolean success records")
        suites[key[0]].extend(values)
        categories[expected[key]["category"]].extend(values)
        overall.extend(values)
    if seen != set(expected):
        raise ValueError(f"Incomplete evaluation: {len(seen)}/{len(expected)} tasks")

    def stats(values):
        return {
            "successes": sum(values),
            "episodes": len(values),
            "success_rate": 100 * sum(values) / len(values),
        }

    report = {
        "benchmark": manifest["benchmark"],
        "complete": True,
        "full_benchmark": set(manifest["suites"]) == set(selected_suites(manifest["benchmark"], "all")),
        "scope": {"suites": manifest["suites"], "dimensions": manifest["dimensions"]},
        "per_suite": {k: stats(v) for k, v in suites.items()},
        "per_category": {k: stats(v) for k, v in categories.items()},
        "overall": stats(overall),
    }
    Path(result_path).with_name("robustness_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for label in ("per_suite", "per_category"):
        print(f"===== {label} =====")
        for name, row in report[label].items():
            print(f"{name}: {row['success_rate']:.2f}% ({row['successes']}/{row['episodes']})")
    row = report["overall"]
    scope = "full benchmark" if report["full_benchmark"] else "selected scope only"
    print(f"overall ({scope}): {row['success_rate']:.2f}% ({row['successes']}/{row['episodes']})")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--directory", type=Path, required=True)
    prep.add_argument("--benchmark", choices=("pro", "plus"), required=True)
    prep.add_argument("--mode", choices=("all", *SUITES), default="all")
    prep.add_argument("--dimensions", default="all")
    prep.add_argument("--episodes", type=int, default=1)
    prep.add_argument("--seed", type=int, default=1000)
    sub.add_parser("tasks").add_argument("manifest", type=Path)
    ckpt = sub.add_parser("checkpoint")
    ckpt.add_argument("directory", type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("manifest", type=Path)
    summary.add_argument("results", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        record = prepare(
            args.root, args.directory, args.benchmark, args.mode, args.dimensions, args.episodes, args.seed
        )
        print(
            f"{args.benchmark}: {len(record['tasks'])} tasks, {record['expected_episodes']} episodes; preflight passed"
        )
    elif args.command == "tasks":
        print(",".join(read_manifest(args.manifest)["suites"]))
    elif args.command == "checkpoint":
        from lerobot.scripts.libero_vlm_only import validate_checkpoint

        validate_checkpoint(args.directory, 2, train_action_expert=True)
        print("Checkpoint: trained VLM2 + full action expert; saved processors verified")
    else:
        summarize(args.manifest, args.results)


if __name__ == "__main__":
    main()
