"""Four sequential rounds, with two independent two-GPU curricula per round."""

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from lerobot.scripts import pi05_ratio_curriculum as curriculum

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "examples/training/train_pi05_ratio_curriculum.sh"
# round, column, cell, direction, easy percentage, training scope
CELLS = (
    (1, 0, "B", "easy_to_hard", 30, "joint"),
    (1, 1, "C", "random", 30, "joint"),
    (2, 0, "D", "hard_to_easy", 50, "joint"),
    (2, 1, "F", "random", 50, "joint"),
    (3, 0, "E", "easy_to_hard", 50, "joint"),
    (3, 1, "A", "hard_to_easy", 30, "joint"),
    (4, 0, "G", "hard_to_easy", 30, "expert_only"),
    (4, 1, "H", "hard_to_easy", 30, "prompt_only"),
)
CONTROLLED = (
    "PRETRAINED_PATH",
    "BASE_PRETRAINED_PATH",
    "RUN_NAME",
    "RUN_PREFIX",
    "VLM_PROMPT_TOKENS",
    "PROMPT_INIT_PATH",
    "PROMPT_EXPORT_PATH",
    "PROMPT_EXPORT_REFERENCE",
    "PDET_CASES",
    "TRAIN_ACTION_EXPERT",
    "FREEZE_VLM_PROMPT",
    "TRAIN_SCOPE",
    "CURRICULUM_DIRECTION",
    "EASY_PERCENT",
    "SAMPLE_SEED",
    "ORDER_DIR",
    "GPU_IDS",
    "NUM_PROCESSES",
    "MAIN_PROCESS_PORT",
)


def positive(value, name):
    if not str(value).isdigit() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def build_jobs(env, extra):
    if any(not arg.startswith("--wandb.") or "=" not in arg for arg in extra):
        raise ValueError("Only --wandb.*=... CLI overrides are allowed")
    repeat = positive(env.get("REPEAT_ID", "2"), "REPEAT_ID")
    steps = positive(env.get("FLOW_STEPS", "6000"), "FLOW_STEPS")
    pairs = [env.get("GPU_IDS_A", "0,1"), env.get("GPU_IDS_B", "2,3")]
    gpu_ids = [part.split(",") for part in pairs]
    if any(len(pair) != 2 or any(not gpu.isdigit() for gpu in pair) for pair in gpu_ids):
        raise ValueError("GPU_IDS_A and GPU_IDS_B must each select exactly two GPU indices")
    if len({int(gpu) for pair in gpu_ids for gpu in pair}) != 4:
        raise ValueError("The two GPU pairs must be disjoint")
    ports = [positive(env.get("PORT_A", "29588"), "PORT_A"), positive(env.get("PORT_B", "29589"), "PORT_B")]
    if len(set(ports)) != 2 or max(ports) > 65535:
        raise ValueError("PORT_A/PORT_B must be distinct ports <= 65535")
    roots = {
        "OUTPUT_ROOT": env.get("OUTPUT_ROOT", "/data1/wyn/chkpt/2601-lerobot/PromptVLA"),
        "LOG_ROOT": env.get("LOG_ROOT", "/data1/wyn/chkpt/2601-lerobot/logs"),
        "SOURCE_PAIR_DIR": env.get("SOURCE_PAIR_DIR", "/data1/wyn/prompt-work/source-pair-r1"),
        "SCORES_PATH": env.get("SCORES_PATH", "/data1/wyn/prompt-work/frameskip-r1/scores.npz"),
        "DATASET_ROOT": env.get("DATASET_ROOT", "/home/wyn/datasets/libero"),
        "TOKENIZER_PATH": env.get("TOKENIZER_PATH", "/home/wyn/models/paligemma-3b-pt-224"),
    }
    roots = {key: str(Path(value).resolve()) for key, value in roots.items()}
    roots["REFERENCE_RUN"] = str(
        Path(
            env.get(
                "REFERENCE_RUN", f"{roots['OUTPUT_ROOT']}/pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1"
            )
        ).resolve()
    )
    plans = Path(env.get("CURRICULUM_PLAN_ROOT", "/data1/wyn/prompt-work/ratio-curriculum")).resolve()
    jobs = []
    for round_id, lane, cell, direction, easy, scope in CELLS:
        name = curriculum.run_name(direction, easy, steps=steps, scope=scope, repeat=repeat)
        settings = {
            **roots,
            "FLOW_STEPS": str(steps),
            "REPEAT_ID": str(repeat),
            "BATCH_SIZE": "32",
            "NUM_PROCESSES": "2",
            "SAVE_ONLY_FINAL": "true",
            "DTYPE": "bfloat16",
            "CURRICULUM_DIRECTION": direction,
            "EASY_PERCENT": str(easy),
            "SAMPLE_SEED": "0",
            "TRAIN_SCOPE": scope,
            "GPU_IDS": pairs[lane],
            "MAIN_PROCESS_PORT": str(ports[lane]),
            "ORDER_DIR": str(plans / curriculum.label(direction, easy, 0, steps)),
        }
        jobs.append(
            {
                "round": round_id,
                "cell": cell,
                "name": name,
                "env": settings,
                "run": str(Path(roots["OUTPUT_ROOT"]) / name),
                "command": ["bash", str(TRAIN), *extra],
            }
        )
    return jobs


def child_env(job, *, dry_run=False):
    env = {key: value for key, value in os.environ.items() if key not in CONTROLLED}
    env.update(job["env"], DRY_RUN=str(dry_run).lower())
    # Nested bash/python commands must use the same active training interpreter.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def checkpoint_complete(job):
    run = Path(job["run"])
    step = int(job["env"]["FLOW_STEPS"])
    target = run / "checkpoints" / f"{step:06d}"
    last = run / "checkpoints/last"
    if not last.is_symlink() or last.resolve() != target.resolve() or not target.is_dir():
        raise ValueError(f"Incomplete existing run; not overwritten or resumed automatically: {run}")
    checkpoints = {p.name for p in (run / "checkpoints").iterdir() if p.name.isdigit()}
    if checkpoints != {f"{step:06d}"}:
        raise ValueError(f"Expected only the final model checkpoint, found {checkpoints}: {run}")
    curriculum.check_checkpoint(job["env"]["ORDER_DIR"], run)
    report = curriculum.audit(job["env"]["ORDER_DIR"], run)
    atomic_json(run / "curriculum_audit.json", report)
    return report


def check_current_runtime(job, current):
    from lerobot.scripts.curriculum_audit_utils import compare_runtime

    recorded = json.loads((Path(job["run"]) / "pdet/manifest-rank0.json").read_text())
    report = compare_runtime(recorded, current)
    atomic_json(Path(job["run"]) / "curriculum_runtime_audit.json", report)
    return report


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def audit_suite(jobs):
    """Check actual all-step traces, not just seeds or planned manifests."""
    from lerobot.scripts.curriculum_audit_utils import compare_runtime, normalize_path_aliases
    from lerobot.scripts.pdet_order_binding import inspect_run

    reference, banks, pools, orders, reports = None, set(), {}, {}, []
    first_runtime, code_reports, path_aliases = None, [], []
    for job in jobs:
        curriculum.check_checkpoint(job["env"]["ORDER_DIR"], job["run"])
        invariant, report = inspect_run(job["run"], name_checker=curriculum.check_name)
        runtime = {key: invariant[key] for key in ("git_commit", "source_tree_sha256", "versions", "cuda")}
        if first_runtime is None:
            first_runtime = runtime
        code_reports.append({"cell": job["cell"], **compare_runtime(first_runtime, runtime)})
        for key in ("git_commit", "source_tree_sha256"):
            invariant.pop(key)
        scope = job["env"]["TRAIN_SCOPE"]
        policy = invariant["recipe"]["policy"]
        for key, value in curriculum.scope_flags(scope).items():
            if policy.get(key, False) != value:
                raise ValueError(f"Wrong realized trainable scope: {job['name']} / {key}")
            policy.pop(key, None)
        percent = job["env"]["EASY_PERCENT"]
        pool = invariant.pop("pool")
        if percent in pools and pools[percent] != pool:
            raise ValueError("Sample/occurrence pool differs within an exposure ratio")
        pools[percent] = pool
        order_key = percent, job["env"]["CURRICULUM_DIRECTION"]
        if order_key in orders and orders[order_key] != report["order_sha256"]:
            raise ValueError("Module controls changed sample order")
        orders[order_key] = report["order_sha256"]
        if reference is None:
            reference = invariant
        else:
            path_aliases.extend(
                {"cell": job["cell"], **row} for row in normalize_path_aliases(reference, invariant)
            )
        if reference != invariant:
            changed = [
                key for key in reference.keys() | invariant.keys() if reference.get(key) != invariant.get(key)
            ]
            raise ValueError(f"Unexpected source/recipe/task/code changes across runs: {changed}")
        banks.add(report["bank_sha256"])
        reports.append({"cell": job["cell"], "scope": scope, **report})
    if len(banks) != 1:
        raise ValueError("Realized per-key noise/time bank differs across ablations")
    for percent in pools:
        signatures = [order for (ratio, _), order in orders.items() if ratio == percent]
        if len(set(signatures)) != 3:
            raise ValueError(f"Expected three distinct orders at ratio {percent}")
    return {
        "audit": "passed",
        "runs": reports,
        "code_revision": code_reports,
        "path_aliases": path_aliases,
        "scope": "All realized steps: same task slots/recipe/source and per-key E1/T1; same recorded code or verified audit-only edits; same occurrence pools within each ratio; identical H2E30 inputs across trainable scopes. Not numerical determinism or historical code equivalence.",
    }


def run_round(jobs, statuses, state_path, log, *, poll_seconds=1):
    """Round barrier; a failed child stops both lanes and all following rounds."""
    active = []
    try:
        for job in jobs:
            if statuses[job["cell"]]["state"] == "complete":
                log(f"SKIP verified completed cell {job['cell']}: {job['name']}")
                continue
            output = Path(state_path).parent / f"{job['cell']}.launch.log"
            stream = output.open("a", buffering=1)
            try:
                process = subprocess.Popen(
                    job["command"],
                    cwd=ROOT,
                    env=child_env(job),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except BaseException:
                stream.close()
                raise
            active.append((job, process, stream))
            statuses[job["cell"]] = {"state": "running", "pid": process.pid, "log": str(output)}
            atomic_json(state_path, statuses)
            log(f"START {job['cell']} GPUs={job['env']['GPU_IDS']}: {job['name']} (log: {output})")
        while active:
            for job, process, stream in list(active):
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    statuses[job["cell"]].update(state="failed", exit_code=code)
                    atomic_json(state_path, statuses)
                    raise RuntimeError(f"Cell {job['cell']} failed (exit={code}); inspect {stream.name}")
                stream.close()
                active.remove((job, process, stream))
                statuses[job["cell"]]["state"] = "verifying"
                atomic_json(state_path, statuses)
                try:
                    checkpoint_complete(job)
                except Exception:
                    statuses[job["cell"]]["state"] = "verification_failed"
                    atomic_json(state_path, statuses)
                    raise
                statuses[job["cell"]]["state"] = "complete"
                atomic_json(state_path, statuses)
                log(f"DONE {job['cell']}: final checkpoint and all-step audit passed")
            if active:
                time.sleep(poll_seconds)
    finally:
        # Accelerate launches children; terminate the complete isolated process
        # group, even if its shell leader has already exited with an error.
        for _job, process, _stream in active:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while any(p.poll() is None for _, p, _ in active) and time.monotonic() < deadline:
            time.sleep(0.1)
        for job, process, stream in active:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            stream.close()
            if statuses[job["cell"]]["state"] == "running":
                statuses[job["cell"]]["state"] = "interrupted"
        atomic_json(state_path, statuses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "audit"), nargs="?", default="run")
    args, extra = parser.parse_known_args()
    jobs = build_jobs(os.environ, extra)
    dry = os.environ.get("DRY_RUN", "false")
    if dry not in ("true", "false"):
        raise ValueError("DRY_RUN must be true or false")
    log_root = Path(jobs[0]["env"]["LOG_ROOT"])
    repeat, steps = jobs[0]["env"]["REPEAT_ID"], jobs[0]["env"]["FLOW_STEPS"]
    directory = log_root / f"curriculum-ablation-r{repeat}-steps{steps}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another suite runner owns {directory}") from error
        with (directory / "runner.log").open("a", buffering=1) as runner_log:

            def log(message):
                line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
                print(line, flush=True)
                runner_log.write(line + "\n")

            if args.command == "audit":
                report = audit_suite(jobs)
                atomic_json(directory / "audit.json", report)
                log(f"Suite audit passed: {directory / 'audit.json'}")
                return
            statuses = {job["cell"]: {"state": "pending"} for job in jobs}
            state_path = directory / "status.json"
            from lerobot.utils.pdet import runtime_info

            runtime = runtime_info()
            log("Checking ALL 8 runs before any GPU training; each starts from pi05_libero_base.")
            # Sequential CPU preparation avoids races on the shared H2E30 plan.
            for job in jobs:
                if Path(job["run"]).exists():
                    checkpoint_complete(job)
                    check_current_runtime(job, runtime)
                    statuses[job["cell"]]["state"] = "complete"
                else:
                    preflight = directory / f"{job['cell']}.preflight.log"
                    with preflight.open("w") as stream:
                        result = subprocess.run(
                            job["command"],
                            cwd=ROOT,
                            env=child_env(job, dry_run=True),
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                        )
                    if result.returncode:
                        raise RuntimeError(f"Preflight failed for {job['cell']}; inspect {preflight}")
                log(
                    f"READY round {job['round']} cell {job['cell']} GPUs={job['env']['GPU_IDS']}: {job['name']}"
                )
            atomic_json(directory / "jobs.json", jobs)
            atomic_json(state_path, statuses)
            if dry == "true":
                log(f"Dry run passed; no training launched. Plans and commands: {directory}")
                return

            def interrupted(signum, _frame):
                raise KeyboardInterrupt(f"Received signal {signum}")

            previous = signal.signal(signal.SIGTERM, interrupted)
            try:
                for round_id in range(1, 5):
                    if runtime_info() != runtime:
                        raise RuntimeError("Training code/environment changed while the suite was running")
                    log(f"ROUND {round_id}/4")
                    run_round([job for job in jobs if job["round"] == round_id], statuses, state_path, log)
                report = audit_suite(jobs)
                atomic_json(directory / "audit.json", report)
                log("All 8 runs completed; each retains only its final model; cross-run audit passed.")
            finally:
                signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
