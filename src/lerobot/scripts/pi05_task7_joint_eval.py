"""Evaluate BOTH task7 temporal mechanism studies, then print ONE terminal report.

Experiment I: four 1000-step RoPE/QK/V/Input policies, actual LIBERO-long
task7 rollouts with 12cm fixed object displacement (success %, 10 episodes
per seed). Default independent eval seeds: 1000,3000,4000,5000.

Experiment II: four separate 1000-step RoPE/Index/Global-Dt/Physical
policies, true 1x/2x (optional unseen 3x) future recorded target cadence;
OFFLINE normalized flow-velocity MSE. No altered-rate robot execution.

The two tables have DIFFERENT metrics and must NEVER be pooled together.
Children reuse the existing two-GPU paired launchers and their resumable
result files. Cached summaries are revalidated against underlying data.
"""

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

from lerobot.scripts import pi05_task7_timepath_experiment as timepath
from lerobot.scripts import pi05_task7_physical_time_experiment as physical
from lerobot.scripts import pi05_task7_cadence as cadence

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path("/data1/wyn/chkpt/2601-lerobot/PromptVLA")
CASES_I = timepath.CASES
CASES_II = physical.CASES
PROTOCOL = "pi05-task7-joint-eightway-report-v1"


def parse_seeds(text):
    try:
        raw = [int(x.strip()) for x in text.split(",")]
    except ValueError as exc:
        raise ValueError("Timepath seeds must be comma-separated nonnegative integers") from exc
    if not raw or any(seed < 0 for seed in raw) or len(raw) != len(set(raw)):
        raise ValueError("Timepath seeds must be nonempty, nonnegative, and distinct")
    return raw


def selected(args, target):
    return args.only in ("all", target)


def source_records(args):
    prior = timepath.check_plan(args.timepath_directory) if selected(args, "timepath") else None
    current = cadence.check(args.cadence_directory) if selected(args, "cadence") else None
    if current is not None:
        current["prepared_root"] = str(args.cadence_directory.resolve())
    return prior, current


def group_jobs(args, prior, current):
    j1 = (
        timepath.build_jobs(prior, args.timepath_directory, args.output_root, "all", args.repeat)
        if prior is not None else []
    )
    j2 = (
        physical.enrich_jobs(current, args.output_root, "all", args.repeat)
        if current is not None else []
    )
    if j1 and set(job["case"] for job in j1) != set(CASES_I):
        raise ValueError("Missing experiment-I model definition")
    if j2 and set(job["case"] for job in j2) != set(CASES_II):
        raise ValueError("Missing experiment-II model definition")
    return j1, j2


def require_checkpoints(jobs):
    missing = [str(job["checkpoint"]) for job in jobs
               if not (job["checkpoint"] / "model.safetensors").is_file()
               or not (job["checkpoint"] / "config.json").is_file()]
    if missing:
        raise FileNotFoundError(
            "Some 1000-step models are not trained. Finish training before unified eval:\n"
            + "\n".join(missing)
        )


def timepath_args(args, seed):
    return SimpleNamespace(
        eval_root=args.timepath_eval_root,
        eval_seed=seed,
        shift_seed=args.shift_seed,
        shift_m=args.shift_m,
        episodes=args.episodes,
        repeat=args.repeat,
    )


def physical_args(args):
    return SimpleNamespace(
        eval_root=args.cadence_eval_root,
        eval_seed=args.cadence_seed,
        flow_time=args.flow_time,
        max_eval_starts=args.max_eval_starts,
        dataset_repo_id=args.dataset_repo_id,
    )


def timepath_summary_path(args, record, seed):
    return timepath.evaluation_root(timepath_args(args, seed), record) / "summary-all.json"


def physical_summary_path(args):
    return (args.cadence_eval_root
            / f"seed{args.cadence_seed}-t{args.flow_time:.2f}"
            / "summary-all.json")


def pair_command(args, target, *, seed=None):
    common = ["--repeat", str(args.repeat), "--output-root", str(args.output_root)]
    if target == "timepath":
        if seed is None:
            raise ValueError("Timepath requires an eval seed")
        return [
            "bash", str(ROOT / "run_eval_libero_task7_timepath_2gpu.sh"),
            *common, "--directory", str(args.timepath_directory),
            "--eval-root", str(args.timepath_eval_root),
            "--log-root", str(args.log_root),
            "--tokenizer", str(args.tokenizer),
            "--eval-seed", str(seed),
            "--episodes", str(args.episodes),
            "--shift-m", str(args.shift_m),
            "--shift-seed", str(args.shift_seed),
        ]
    if target == "cadence":
        return [
            "bash", str(ROOT / "run_eval_libero_task7_physical_time_2gpu.sh"),
            *common, "--directory", str(args.cadence_directory),
            "--eval-root", str(args.cadence_eval_root),
            "--log-root", str(args.log_root),
            "--tokenizer", str(args.tokenizer),
            "--dataset-repo-id", str(args.dataset_repo_id),
            "--eval-seed", str(args.cadence_seed),
            "--flow-time", str(args.flow_time),
            "--max-eval-starts", str(args.max_eval_starts),
        ]
    raise ValueError("Unknown mechanism experiment")


def run_child(command, args):
    env = os.environ.copy()
    env.pop("DRY_RUN", None)
    env["GPU_IDS"] = args.gpus
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH','')}"
    print("$ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def verify_timepath_summary(args, record, jobs, seed):
    path = timepath_summary_path(args, record, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing timepath summary for seed {seed}: {path}")
    # Rebuild from raw eval_info.json + physics manifests, not from stale
    # appended shell logs. Enforces identical displacement helper/seed.
    return timepath.summarize(record, jobs, timepath_args(args, seed))


def verify_cadence_summary(args, prepared, jobs):
    path = physical_summary_path(args)
    if not path.exists():
        raise FileNotFoundError(f"Missing physical-time summary: {path}")
    # Verifies every case's exact checkpoint fingerprint, evaluation manifest,
    # matching target semantics and held-out cadence sample provenance.
    return physical.summarize(prepared, jobs, physical_args(args))


def _check_timepath_rates(rows, expected_episodes):
    if not rows:
        return
    for seed, entry in rows.items():
        if set(entry["results"]) != set(CASES_I):
            raise ValueError(f"Seed {seed} is missing an experiment-I case")
        if entry.get("episodes") != expected_episodes:
            raise ValueError(f"Seed {seed} used the wrong number of episodes")
        for case in CASES_I:
            row = entry["results"][case]
            n, yes = int(row["overall_episodes"]), int(row["overall_successes"])
            if n != expected_episodes or not 0 <= yes <= n:
                raise ValueError(f"Seed {seed} {case}: invalid task7 success counts")
            rate = 100 * yes / n
            if abs(float(row["selected_task_rate"]) - rate) > 1e-5:
                raise ValueError(f"Seed {seed} {case}: rate/count disagreement")


def _check_cadence_paired(results):
    if set(results) != set(CASES_II):
        raise ValueError("Experiment II missing an arm")
    strides = set(results["rope"])
    if not {"1", "2"}.issubset(strides):
        raise ValueError("Experiment II requires both trained cadences 1x and 2x")
    for case in CASES_II:
        if set(results[case]) != strides:
            raise ValueError("Experiment-II arms were tested on different cadences")
    for stride in sorted(strides, key=int):
        reference = results["rope"][stride]
        n = int(reference["n"])
        starts = reference["start_indices"]
        if n < 1 or len(starts) != n:
            raise ValueError("Missing held-out starts in offline test")
        for case in CASES_II:
            row = results[case][stride]
            if (
                int(row["n"]) != n or row["start_indices"] != starts
                or len(row["sample_mse"]) != n
                or not all(isinstance(x, (int, float)) and 0 <= x < float("inf")
                           for x in row["sample_mse"])
            ):
                raise ValueError(f"Cadence {stride} {case} has unpaired/malformed targets")
            mse = sum(row["sample_mse"]) / n
            if abs(float(row["mean_mse"]) - mse) > 1e-6 * max(1, mse):
                raise ValueError(f"Cadence {stride} {case} MSE mismatch")


def _table(headers, rows):
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    def fmt(row):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))
    return "\n".join([fmt(headers), "  ".join("-" * w for w in widths), *map(fmt, rows)])


def _relative_change(value, reference):
    if reference == 0:
        return "n/a"
    return f"{(value / reference - 1) * 100:+.1f}%"


def print_report(timepath_rows, cadence_row, args):
    print("\n" + "=" * 84)
    print("TASK7 JOINT RESULTS · TWO DISTINCT EXPERIMENTS · 1000 UPDATES / ARM")
    print("=" * 84)

    if timepath_rows:
        _check_timepath_rates(timepath_rows, args.episodes)
        print("\n[Experiment I] Q/K/V/Input injection · LIBERO-long task7 · 12 cm object shift")
        print(f"  Per eval seed: {args.episodes} paired episodes; "
              f"shift_seed={args.shift_seed}; metric=success rate (%)")
        rows = []
        pooled = {case: [0, 0] for case in CASES_I}
        for seed, entry in timepath_rows.items():
            cells = [str(seed)]
            for case in CASES_I:
                item = entry["results"][case]
                n, hits = int(item["overall_episodes"]), int(item["overall_successes"])
                pooled[case][0] += hits
                pooled[case][1] += n
                cells.append(f"{hits}/{n} ({100*hits/n:.1f}%)")
            rows.append(cells)
        cells = ["POOLED"]
        for case in CASES_I:
            hits, n = pooled[case]
            cells.append(f"{hits}/{n} ({100*hits/n:.1f}%)")
        rows.append(cells)
        print(_table(["Eval seed", "RoPE", "Time-QK", "Time-V", "Time-Input"], rows))
        qk = pooled["time_qk"][0] / pooled["time_qk"][1]
        vin = pooled["time_v"][0] / pooled["time_v"][1]
        inp = pooled["time_input"][0] / pooled["time_input"][1]
        rope = pooled["rope"][0] / pooled["rope"][1]
        print(f"  Time-Input − RoPE: {100*(inp-rope):+.2f} pp; "
              f"Time-Input − Time-QK: {100*(inp-qk):+.2f} pp; "
              f"Time-V − RoPE: {100*(vin-rope):+.2f} pp")
        print("  Note: eval seeds reuse the demonstration's official initial-state IDs; "
              "seed 2000 (shift calibration) excluded by default.")

    if cadence_row:
        results = cadence_row["results"]
        _check_cadence_paired(results)
        print("\n[Experiment II] Index vs physical time · OFFLINE flow-velocity MSE (↓ better)")
        print(f"  True future-action spacing = stride / fps; "
              f"fixed flow timestep t={args.flow_time}; seed={args.cadence_seed}")
        rows = []
        for stride in sorted(results["rope"], key=int):
            info = results["rope"][stride]
            n = int(info["n"])
            row = [f"{stride}x", str(n)]
            for case in CASES_II:
                row.append(f"{results[case][stride]['mean_mse']:.6f}")
            rows.append(row)
        print(_table(
            ["Cadence", "Starts", "RoPE", "Index", "Global-Δt", "Physical"], rows,
        ))
        for stride in sorted(results["rope"], key=int):
            metrics = {case: float(results[case][stride]["mean_mse"]) for case in CASES_II}
            print(f"  {stride}x Physical vs Index: "
                  f"{_relative_change(metrics['physical'], metrics['index'])} MSE; "
                  f"vs Global-Δt: "
                  f"{_relative_change(metrics['physical'], metrics['global_dt'])} MSE")
        print("  Note: offline anchors share ONE demonstration; future-action windows may overlap.")
        print("  No 2x/3x commands executed in a robot/simulator; these are NOT success rates.")

    print("\n" + "=" * 84, flush=True)


def write_report(timepath_rows, cadence_row, args):
    report = {
        "protocol": PROTOCOL,
        "settings": {
            "timepath_seeds": args.timepath_seeds if timepath_rows else [],
            "cadence_seed": args.cadence_seed if cadence_row else None,
            "episodes": args.episodes if timepath_rows else None,
            "shift_m": args.shift_m if timepath_rows else None,
            "shift_seed": args.shift_seed if timepath_rows else None,
            "flow_time": args.flow_time if cadence_row else None,
            "max_eval_starts": args.max_eval_starts if cadence_row else None,
            "repeat": args.repeat,
        },
        "timepath": list(timepath_rows.values()) if timepath_rows else None,
        "cadence": cadence_row,
        "warning": (
            "Experiment I success rates and Experiment II offline flow-MSE "
            "are incomparable metrics and must not be pooled."
        ),
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    config_hash = hashlib.sha256(
        json.dumps(report["settings"], sort_keys=True).encode()
    ).hexdigest()[:12]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    root = args.report_dir / f"eightway-{config_hash}"
    (root.with_suffix(".json")).write_text(payload)
    with root.with_name(root.name + "-timepath.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "case", "successes", "episodes", "rate_percent"])
        for seed, item in timepath_rows.items():
            for case in CASES_I:
                row = item["results"][case]
                writer.writerow([seed, case, row["overall_successes"],
                                 row["overall_episodes"], row["selected_task_rate"]])
    with root.with_name(root.name + "-cadence.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "cadence_stride", "n_starts", "mean_normalized_flow_mse"])
        if cadence_row:
            for case in CASES_II:
                for stride, row in cadence_row["results"][case].items():
                    writer.writerow([case, stride, row["n"], row["mean_mse"]])
    print("\nCombined JSON: " + str(root.with_suffix(".json")))
    print("Combined CSVs: " + str(root.with_name(root.name + "-timepath.csv"))
          + ", " + str(root.with_name(root.name + "-cadence.csv")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "report"), default="run",
                        help="run paired evals or only print/verify existing results")
    parser.add_argument("--only", choices=("all", "timepath", "cadence"), default="all")
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--timepath-seeds", default="1000,3000,4000,5000")
    parser.add_argument("--cadence-seed", type=int, default=3000)
    parser.add_argument("--shift-m", type=float, default=0.12)
    parser.add_argument("--shift-seed", type=int, default=20261103)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--flow-time", type=float, default=0.5)
    parser.add_argument("--max-eval-starts", type=int, default=32)
    parser.add_argument("--dataset-repo-id", default=os.environ.get("DATASET_REPO_ID", "libero"))
    parser.add_argument("--tokenizer", type=Path, default=Path(os.environ.get(
        "TOKENIZER_PATH", "/home/wyn/models/paligemma-3b-pt-224"
    )))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get(
        "OUTPUT_ROOT", os.environ.get("CKPT_ROOT", DEFAULT_OUTPUT)
    )))
    parser.add_argument("--log-root", type=Path, default=Path(os.environ.get(
        "LOG_ROOT", "/data1/wyn/chkpt/2601-lerobot/logs"
    )))
    parser.add_argument("--timepath-directory", type=Path, default=timepath.DEFAULT_PLAN)
    parser.add_argument("--cadence-directory", type=Path, default=cadence.DEFAULT_DIR)
    parser.add_argument("--timepath-eval-root", type=Path, default=Path(os.environ.get(
        "TIMEPATH_EVAL_ROOT", timepath.DEFAULT_EVAL
    )))
    parser.add_argument("--cadence-eval-root", type=Path, default=Path(os.environ.get(
        "CADENCE_EVAL_ROOT", physical.DEFAULT_EVAL
    )))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()
    if args.report_dir is None:
        args.report_dir = args.output_root / "eval" / "task7-eightway-summary"
    try:
        args.timepath_seeds = parse_seeds(args.timepath_seeds)
        gpus = args.gpus.split(",")
        if len(gpus) != 2 or len(set(gpus)) != 2 or not all(x.isdecimal() for x in gpus):
            raise ValueError("Specify exactly two distinct GPU indices: --gpus 0,1")
        if (
            args.repeat < 1 or args.episodes != 10
            or args.cadence_seed < 0 or args.shift_seed < 0
            or not 0 <= args.shift_m <= 0.14
            or not 0 < args.flow_time < 1 or args.max_eval_starts < 1
            or (args.dry_run and args.command == "report")
        ):
            raise ValueError("Invalid evaluation settings (task7 requires exactly 10 episodes/seed)")
        if args.dry_run:
            print("DRY RUN: planned two-GPU evaluations; no data loaded or GPUs used.")
            if selected(args, "timepath"):
                for seed in args.timepath_seeds:
                    print("$ " + shlex.join(pair_command(args, "timepath", seed=seed)))
            if selected(args, "cadence"):
                print("$ " + shlex.join(pair_command(args, "cadence")))
            return

        prior, current = source_records(args)
        j1, j2 = group_jobs(args, prior, current)
        require_checkpoints([*j1, *j2])
        timepath_rows = {}
        cadence_row = None
        if j1:
            for i, seed in enumerate(args.timepath_seeds, 1):
                print(f"\n=== Experiment I · eval seed {seed} "
                      f"({i}/{len(args.timepath_seeds)}) ===", flush=True)
                summary_path = timepath_summary_path(args, prior, seed)
                if args.command == "run" and not summary_path.is_file():
                    run_child(pair_command(args, "timepath", seed=seed), args)
                elif args.command == "run":
                    print(f"Found previous summary; validating raw results: {summary_path}",
                          flush=True)
                timepath_rows[seed] = verify_timepath_summary(args, prior, j1, seed)
        if j2:
            print("\n=== Experiment II · offline physical-time evaluation ===", flush=True)
            summary_path = physical_summary_path(args)
            if args.command == "run" and not summary_path.is_file():
                run_child(pair_command(args, "cadence"), args)
            elif args.command == "run":
                print(f"Found previous summary; validating raw results: {summary_path}",
                      flush=True)
            cadence_row = verify_cadence_summary(args, current, j2)
        print_report(timepath_rows, cadence_row, args)
        write_report(timepath_rows, cadence_row, args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Unified eightway task7 evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
