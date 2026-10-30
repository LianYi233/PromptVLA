"""Evaluate the A-H curriculum ablation with one sequential queue per GPU.

Usage: bash run_eval_libero_curriculum_ablation_pro.sh [run|status|summary]
Defaults: GPU_IDS=0,1,2,3 CELLS="A B C D E F G H" REPEAT_ID=2,
CHECKPOINT_STEP=6000 EVAL_SEED=1000 EPISODES_PER_TASK=10,
PRO_DIMENSIONS=object,swap,lan,task. DRY_RUN=true preflights all selected models.
"""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lerobot.scripts import libero_robustness as robust

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "run_eval_libero_robustness.sh"
CELLS = {
    "A": ("H2E", 30, "joint", "aexpert"),
    "B": ("E2H", 30, "joint", "aexpert"),
    "C": ("RND", 30, "joint", "aexpert"),
    "D": ("H2E", 50, "joint", "aexpert"),
    "E": ("E2H", 50, "joint", "aexpert"),
    "F": ("RND", 50, "joint", "aexpert"),
    "G": ("H2E", 30, "expert_only", "frozenp-aexpert"),
    "H": ("H2E", 30, "prompt_only", "promptonly"),
}
CLEAR = (
    "PRETRAINED_PATH",
    "BASE_PRETRAINED_PATH",
    "RUN_NAME",
    "RUN_PREFIX",
    "VLM_PROMPT_TOKENS",
    "BASE_CKPT",
    "BASE_MODEL_PATH",
    "PDET_CASES",
    "ORDER_DIR",
    "TRAIN_SCOPE",
    "TRAIN_ACTION_EXPERT",
    "FREEZE_VLM_PROMPT",
    "LIBERO_CONFIG_PATH",
    "LIBERO_ROBUSTNESS_MANIFEST",
    "LIBERO_PROGRESS_FD",
    "LIBERO_PROGRESS_LABEL",
    "LEROBOT_EVAL_BASE",
    "LEROBOT_EVAL_ATTENTION",
    "LEROBOT_EVAL_FINETUNED_PROMPT",
    "CUDA_VISIBLE_DEVICES",
    "MUJOCO_EGL_DEVICE_ID",
)


def settings(env):
    chosen = env.get("CELLS", "A B C D E F G H").replace(",", " ").split()
    gpus = env.get("GPU_IDS", "0,1,2,3").split(",")
    if not chosen or len(set(chosen)) != len(chosen) or any(c not in CELLS for c in chosen):
        raise ValueError("CELLS must contain unique letters from A-H")
    if not gpus or any(not x.isdigit() for x in gpus) or len({int(x) for x in gpus}) != len(gpus):
        raise ValueError("GPU_IDS must contain unique numeric GPU indices")
    values = {}
    for key, default in (
        ("REPEAT_ID", 2),
        ("CHECKPOINT_STEP", 6000),
        ("EVAL_SEED", 1000),
        ("EPISODES_PER_TASK", 10),
    ):
        value = env.get(key, str(default))
        if not value.isdigit() or int(value) < (0 if key == "EVAL_SEED" else 1):
            raise ValueError(f"Invalid {key}")
        values[key] = str(int(value))
    for key in ("PROGRESS", "DRY_RUN"):
        values[key] = env.get(key, "true" if key == "PROGRESS" else "false")
        if values[key] not in ("true", "false"):
            raise ValueError(f"Invalid {key}")
    dims = env.get("PRO_DIMENSIONS", "object,swap,lan,task")
    robust.selected_suites("pro", "all", dims)
    ckpt = Path(env.get("CKPT_ROOT", "/data1/wyn/chkpt/2601-lerobot/PromptVLA")).resolve()
    repeat = values["REPEAT_ID"]
    output = Path(env.get("BASE_OUTPUT", str(ckpt / f"eval/curriculum-ablation-r{repeat}"))).resolve()
    logs = Path(
        env.get("EVAL_LOG_ROOT", f"/data1/wyn/chkpt/2601-lerobot/logs/eval/curriculum-ablation-r{repeat}")
    ).resolve()
    protocol = f"seed{values['EVAL_SEED']}-ep{values['EPISODES_PER_TASK']}-dims{dims.replace(',', '-')}"
    batch = logs / f"pro-batch-{protocol}"
    base = {k: v for k, v in env.items() if k not in CLEAR}
    base.update(
        values,
        PRO_DIMENSIONS=dims,
        CKPT_ROOT=str(ckpt),
        BASE_OUTPUT=str(output),
        EVAL_LOG_ROOT=str(logs),
        BENCHMARK_ROOT=env.get("BENCHMARK_ROOT", "/home/wyn/VLAA/LIBERO-PRO"),
        TOKENIZER_PATH=env.get("TOKENIZER_PATH", "/home/wyn/models/paligemma-3b-pt-224"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    jobs = []
    for index, cell in enumerate(chosen):
        direction, easy, scope, tag = CELLS[cell]
        name = (
            f"pi05-curriculum-{direction}-e{easy}-h{100 - easy}-S0-steps{values['CHECKPOINT_STEP']}"
            f"-{tag}-r{repeat}-libero-Bocc-base-vlm2-P1-D0-E1-T1"
        )
        step = f"{int(values['CHECKPOINT_STEP']):06d}"
        result_root = output / "pro" / protocol
        jobs.append(
            {
                "cell": cell,
                "name": name,
                "gpu": gpus[index % len(gpus)],
                "env": base | {"RUN_NAME": name, "TRAIN_SCOPE": scope, "GPU_ID": gpus[index % len(gpus)]},
                "checkpoint": ckpt / name / f"checkpoints/{step}/pretrained_model",
                "manifest": result_root / "protocol-all/manifest.json",
                "result": result_root / f"libero060-all-{name}-{step}-pro-libero-all4-resume/eval_info.json",
            }
        )
    return jobs, batch


def read_result(job):
    path = job["result"]
    # A summary is accepted only for this model and the requested evaluation recipe.
    resume = json.loads(path.with_name("resume_manifest.json").read_text())
    args = dict(x.split("=", 1) for x in resume["arguments"] if x.startswith("--") and "=" in x)
    if Path(args["--policy.path"]).resolve() != job["checkpoint"].resolve():
        raise ValueError("Result belongs to a different checkpoint")
    expected_args = {
        "--seed": job["env"]["EVAL_SEED"],
        "--eval.n_episodes": job["env"]["EPISODES_PER_TASK"],
        "--policy.n_action_steps": "10",
        "--eval.batch_size": "1",
        "--env.control_mode": "relative",
        "--policy.use_amp": "false",
        "--policy.compile_model": "false",
        "--env.observation_height": "224",
        "--env.observation_width": "224",
    }
    if any(args.get(k) != v for k, v in expected_args.items()):
        raise ValueError("Evaluation settings differ")
    manifest = robust.read_manifest(job["manifest"])
    if (
        manifest["benchmark"] != "pro"
        or manifest["mode"] != "all"
        or manifest["seed"] != int(job["env"]["EVAL_SEED"])
        or manifest["episodes_per_task"] != int(job["env"]["EPISODES_PER_TASK"])
        or manifest["dimensions"] != job["env"]["PRO_DIMENSIONS"]
        or manifest["suites"] != robust.selected_suites("pro", "all", job["env"]["PRO_DIMENSIONS"])
        or len(manifest["tasks"]) != 10 * len(manifest["suites"])
    ):
        raise ValueError("PRO protocol coverage/settings differ")
    if args.get("--env.task") != ",".join(manifest["suites"]):
        raise ValueError("Result evaluated different suites")
    if resume.get("robustness", {}).get("manifest_sha256") != robust.json_hash(manifest):
        raise ValueError("Evaluation manifest identity differs")
    # Capture the child process, not process-global stdout (four worker threads run concurrently).
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.libero_robustness",
            "summarize",
            str(job["manifest"]),
            str(path),
        ],
        env=os.environ | {"PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return json.loads(path.with_name("robustness_summary.json").read_text())


def summary(jobs, batch):
    lines = [
        "| Cell | Long | Goal | Object | Spatial | Overall | Episodes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    categories = ["| Cell | object | swap | lan | task | env |", "|---|---:|---:|---:|---:|---:|"]
    reports, errors = {}, {}
    for job in jobs:
        cell = job["cell"]
        try:
            report = read_result(job)
            reports[cell] = report
            rates = []
            for base in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
                rows = [row for suite, row in report["per_suite"].items() if suite.rsplit("_", 1)[0] == base]
                rates.append(
                    f"{100 * sum(r['successes'] for r in rows) / sum(r['episodes'] for r in rows):.2f}"
                )
            overall = report["overall"]
            lines.append(
                f"| {cell} | "
                + " | ".join(rates)
                + f" | {overall['success_rate']:.2f} | {overall['successes']}/{overall['episodes']} |"
            )
            cats = report["per_category"]
            categories.append(
                f"| {cell} | "
                + " | ".join(
                    f"{cats[d]['success_rate']:.2f}" if d in cats else "—" for d in robust.PRO_DIMENSIONS
                )
                + " |"
            )
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            errors[cell] = f"{type(error).__name__}: {error}"
            lines.append(f"| {cell} | — | — | — | — | 未完成/未通过校验 | — |")
    text = "\n".join(
        [
            "Success rates (%); selected PRO dimensions: " + jobs[0]["env"]["PRO_DIMENSIONS"],
            *lines,
            "",
            *categories,
        ]
    )
    print(text, flush=True)
    for cell, error in errors.items():
        print(f"[{cell}] {error}", flush=True)
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "summary.txt").write_text(text + "\n")
    (batch / "summary.json").write_text(
        json.dumps(
            {"reports": reports, "errors": errors, "sources": {j["cell"]: str(j["result"]) for j in jobs}},
            indent=2,
        )
        + "\n"
    )
    return not errors


def run(jobs, batch):
    batch.mkdir(parents=True, exist_ok=True)
    with (batch / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Batch logs: {batch}", flush=True)
        # Serialize preflight: create the shared immutable benchmark manifest before workers start.
        for job in jobs:
            cell = job["cell"]
            print(f"[{cell}] preflight GPU={job['gpu']} {job['name']}", flush=True)
            with (batch / f"{cell}.preflight.log").open("w") as stream:
                result = subprocess.run(
                    ["bash", str(ENTRY), "pro", "all"],
                    cwd=ROOT,
                    env=job["env"] | {"DRY_RUN": "true"},
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(f"{cell} preflight failed; see {batch / (cell + '.preflight.log')}")
        if jobs[0]["env"]["DRY_RUN"] == "true":
            print(
                "All selected checkpoints/resources passed preflight; no GPU evaluation started.", flush=True
            )
            return True
        status, mutex = {}, threading.Lock()

        def record(cell, value):
            with mutex:
                status[cell] = value
                temp = batch / "status.json.tmp"
                temp.write_text(json.dumps(status, indent=2) + "\n")
                temp.replace(batch / "status.json")

        def worker(queue):
            for job in queue:
                cell = job["cell"]
                record(cell, {"state": "running", "gpu": job["gpu"], "run": job["name"]})
                print(f"[{cell}] START GPU={job['gpu']}", flush=True)
                try:
                    with (
                        (batch / f"{cell}.result.txt").open("a") as out,
                        (batch / f"{cell}.progress.log").open("a") as err,
                    ):
                        result = subprocess.run(
                            ["bash", str(ENTRY), "pro", "all"],
                            cwd=ROOT,
                            env=job["env"] | {"DRY_RUN": "false"},
                            stdout=out,
                            stderr=err,
                        )
                        code = result.returncode
                        err.write(f"\n[{cell}] evaluation exit={code}\n")
                    if code == 0:
                        read_result(job)
                    record(
                        cell,
                        {
                            "state": "completed" if code == 0 else "failed",
                            "exit_code": code,
                            "gpu": job["gpu"],
                            "run": job["name"],
                        },
                    )
                    print(f"[{cell}] evaluation exit={code}", flush=True)
                except Exception as error:
                    record(
                        cell, {"state": "failed", "error": str(error), "gpu": job["gpu"], "run": job["name"]}
                    )
                    print(f"[{cell}] failed: {error}", flush=True)

        queues = [[j for j in jobs if j["gpu"] == gpu] for gpu in dict.fromkeys(j["gpu"] for j in jobs)]
        for job in jobs:
            record(job["cell"], {"state": "pending", "gpu": job["gpu"], "run": job["name"]})
        with ThreadPoolExecutor(max_workers=len(queues)) as pool:
            list(pool.map(worker, queues))
        complete = summary(jobs, batch)
        return complete and all(s["state"] == "completed" for s in status.values())


def status(jobs, batch):
    print(f"Logs: {batch}")
    path = batch / "status.json"
    states = json.loads(path.read_text()) if path.exists() else {}
    for job in jobs:
        cell = job["cell"]
        state = states.get(cell, {}).get("state", "pending/preflight")
        print(f"[{cell}] GPU={job['gpu']} {state}")
        path = batch / f"{cell}.progress.log"
        if state == "running" and path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 4096))
                lines = [
                    s.strip()
                    for s in re.split(r"[\r\n]", stream.read().decode(errors="replace"))
                    if s.strip()
                ]
            print("  " + (lines[-1] if lines else "loading model/environments ..."))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("run", "status", "summary"), nargs="?", default="run")
    args = parser.parse_args()
    jobs, batch = settings(dict(os.environ))
    if args.command == "summary":
        summary(jobs, batch)
    elif args.command == "status":
        status(jobs, batch)
    else:
        raise SystemExit(0 if run(jobs, batch) else 1)


if __name__ == "__main__":
    main()
