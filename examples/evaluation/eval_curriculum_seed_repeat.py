"""Standard LIBERO four-suite evaluation for the six D/E/F seed-repeat models.

Usage: bash run_eval_libero_curriculum_seed_repeat.sh [run|status|summary]
Default GPU_IDS=0,1,2,3 JOBS_PER_GPU=2 runs all six concurrently; use 1 for four queues.
LIBERO_OUTPUT_ROOT and LIBERO_EVAL_LOG_ROOT are separate from the PRO launcher's roots.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from examples.evaluation import eval_curriculum_seed_repeat_pro as seeds

queue = seeds.baseline
ROOT = queue.ROOT
ENTRY = ROOT / "run_eval_libero_curriculum_seed_repeat.sh"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def settings(env):
    # Reuse exact trained run names and seed validation, never the PRO result locations.
    template_env = dict(env, PRO_DIMENSIONS="object,swap,lan,task")
    jobs, _ = seeds.settings(template_env)
    suffix = f"curriculum-def-seeds-{env.get('SEED_SET_1', '2')}-{env.get('SEED_SET_2', '3')}"
    ckpt = Path(jobs[0]["env"]["CKPT_ROOT"])
    output = Path(env.get("LIBERO_OUTPUT_ROOT", str(ckpt / "eval" / suffix / "libero"))).resolve()
    logs = Path(
        env.get("LIBERO_EVAL_LOG_ROOT", f"/data1/wyn/chkpt/2601-lerobot/logs/eval/{suffix}/libero")
    ).resolve()
    protocol = f"seed{jobs[0]['env']['EVAL_SEED']}-ep{jobs[0]['env']['EPISODES_PER_TASK']}"
    batch = logs / f"libero-batch-{protocol}"
    concurrent = env.get("JOBS_PER_GPU", "2")
    if concurrent not in ("1", "2"):
        raise ValueError("JOBS_PER_GPU must be 1 or 2")
    for job in jobs:
        child = job["env"]
        for key in ("LIBERO_BENCHMARK", "LIBERO_ROBUSTNESS_MANIFEST", "BENCHMARK_ROOT", "PRO_DIMENSIONS"):
            child.pop(key, None)
        child.update(
            LIBERO_OUTPUT_ROOT=str(output),
            LIBERO_EVAL_LOG_ROOT=str(logs),
            BASE_OUTPUT=str(output / protocol),
            EVAL_LOG_ROOT=str(batch / "full"),
            LIBERO_CONFIG_PATH=env.get("STANDARD_LIBERO_CONFIG_PATH", "/home/wyn/.libero"),
            LEROBOT_EVAL_FINETUNED_PROMPT="1",
            TQDM_DISABLE="1",
            CELLS=job["cell"],
            GPU_IDS=job["gpu"],
            JOBS_PER_GPU=concurrent,
        )
        job.pop("manifest", None)
        job["result"] = (
            output / protocol / f"libero060-all-{job['name']}-006000-libero-all4-resume/eval_info.json"
        )
    return jobs, batch, int(concurrent)


def read_result(job):
    path = job["result"]
    manifest = json.loads(path.with_name("resume_manifest.json").read_text())
    args = dict(x.split("=", 1) for x in manifest["arguments"] if x.startswith("--") and "=" in x)
    if Path(args["--policy.path"]).resolve() != job["checkpoint"].resolve():
        raise ValueError("Result belongs to a different checkpoint")
    expected_args = {
        "--env.type": "libero",
        "--env.task": ",".join(SUITES),
        "--seed": job["env"]["EVAL_SEED"],
        "--eval.n_episodes": job["env"]["EPISODES_PER_TASK"],
        "--env.control_mode": "relative",
        "--env.max_parallel_tasks": "1",
        "--eval.batch_size": "1",
        "--policy.n_action_steps": "10",
        "--policy.use_amp": "false",
        "--policy.compile_model": "false",
        "--policy.gradient_checkpointing": "false",
    }
    if any(args.get(k) != v for k, v in expected_args.items()) or manifest.get("robustness"):
        raise ValueError("Expected standard LIBERO evaluation settings, not PRO/PLUS")
    tokenizer = manifest.get("finetuned_prompt_evaluation", {}).get("tokenizer_path")
    if not tokenizer or Path(tokenizer).resolve() != Path(job["env"]["TOKENIZER_PATH"]).resolve():
        raise ValueError("Missing/mismatched saved-processor tokenizer override")
    # The historical standard entry uses these env defaults; do not accidentally accept PRO's 224px.
    if any(args.get(k, "360") != "360" for k in ("--env.observation_height", "--env.observation_width")):
        raise ValueError("Standard LIBERO camera resolution differs")
    data = json.loads(path.read_text())
    rows = data.get("per_task") or data.get("Aggregated Metrics for per_task")
    if not isinstance(rows, list):
        raise ValueError("Missing per-task evaluation results")
    expected = {(suite, task) for suite in SUITES for task in range(10)}
    seen, by_suite = set(), {suite: [] for suite in SUITES}
    episodes = int(job["env"]["EPISODES_PER_TASK"])
    for row in rows:
        key = row["task_group"], row["task_id"]
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected/duplicate task: {key}")
        values = row.get("metrics", row).get("successes")
        if (
            not isinstance(values, list)
            or len(values) != episodes
            or any(type(x) is not bool for x in values)
        ):
            raise ValueError(f"Incomplete/nonboolean episode outcomes: {key}")
        seen.add(key)
        by_suite[key[0]].extend(values)
    if seen != expected:
        raise ValueError(f"Incomplete evaluation: {len(seen)}/40 tasks")

    def stats(values):
        return {
            "successes": sum(values),
            "episodes": len(values),
            "success_rate": 100 * sum(values) / len(values),
        }

    return {
        "benchmark": "libero",
        "complete": True,
        "per_suite": {suite: stats(values) for suite, values in by_suite.items()},
        "overall": stats([x for values in by_suite.values() for x in values]),
    }


def summary(jobs, batch):
    reports, errors = {}, {}
    lines = [
        "Standard LIBERO success rates (%); eval seed=" + jobs[0]["env"]["EVAL_SEED"],
        "| Cell | Long | Goal | Object | Spatial | Overall | Episodes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for job in jobs:
        cell = job["cell"]
        try:
            report = read_result(job)
            reports[cell] = report
            rates = [f"{report['per_suite'][s]['success_rate']:.2f}" for s in reversed(SUITES)]
            total = report["overall"]
            lines.append(
                f"| {cell} | "
                + " | ".join(rates)
                + f" | {total['success_rate']:.2f} | {total['successes']}/{total['episodes']} |"
            )
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            errors[cell] = f"{type(error).__name__}: {error}"
            lines.append(f"| {cell} | — | — | — | — | 未完成/未通过校验 | — |")
    text = "\n".join(lines) + "\n"
    print(text, end="", flush=True)
    for cell, error in errors.items():
        print(f"[{cell}] {error}", flush=True)
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "summary.txt").write_text(text)
    (batch / "summary.json").write_text(
        json.dumps(
            {
                "reports": reports,
                "errors": errors,
                "sources": {j["cell"]: str(j["result"]) for j in jobs},
            },
            indent=2,
        )
        + "\n"
    )
    return not errors


def single(job, batch):
    # Lazy import: status and result summary do not load model/checkpoint dependencies.
    from lerobot.scripts.libero_robustness_checkpoint import check

    seeds.check_training_identity([job])
    check(job["checkpoint"], "joint")
    if not Path(job["env"]["TOKENIZER_PATH"]).is_dir():
        raise FileNotFoundError("Missing local TOKENIZER_PATH")
    command = ["bash", str(ROOT / "run_eval_libero-full.sh"), "vlm_only", "all", str(job["training_seed"])]
    if job["env"]["DRY_RUN"] == "true":
        subprocess.run(command, cwd=ROOT, env=job["env"], check=True)
        return
    directory = batch / "full"
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"{job['cell']}.launch.log"
    child = dict(job["env"])
    fd = None
    try:
        if child["PROGRESS"] == "true":
            # Keep a dedicated descriptor outside run_eval_libero-full.sh's stdout/stderr pipeline.
            fd = os.dup(sys.stderr.fileno())
            child.update(LIBERO_PROGRESS_FD=str(fd), LIBERO_PROGRESS_LABEL=f"LIBERO/{job['cell']}")
        with log.open("a") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=child,
                stdout=stream,
                stderr=subprocess.STDOUT,
                pass_fds=() if fd is None else (fd,),
            )
    finally:
        if fd is not None:
            os.close(fd)
    if result.returncode:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 16384))
            tail = stream.read().decode(errors="replace").replace("\r", "\n").splitlines()[-60:]
        print("\n".join(tail), file=sys.stderr)
        raise RuntimeError(f"{job['cell']} evaluation failed (exit={result.returncode}); full log: {log}")
    report = read_result(job)
    print(f"===== {job['name']} | checkpoint 006000 =====")
    for suite in sorted(SUITES):
        stats = report["per_suite"][suite]
        print(f"{suite}: {stats['success_rate']:.2f}% ({stats['successes']}/{stats['episodes']})")
    total = report["overall"]
    print(f"overall: {total['success_rate']:.2f}% ({total['successes']}/{total['episodes']})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "summary", "single"), nargs="?", default="run")
    args = parser.parse_args()
    jobs, batch, concurrent = settings(dict(os.environ))
    if args.command == "status":
        queue.status(jobs, batch)
    elif args.command == "summary":
        raise SystemExit(0 if summary(jobs, batch) else 1)
    elif args.command == "single":
        if len(jobs) != 1:
            raise ValueError("Internal single command requires exactly one CELLS label")
        single(jobs[0], batch)
    else:
        raise SystemExit(
            0
            if queue.run(
                jobs,
                batch,
                entry=ENTRY,
                arguments=("single",),
                reader=read_result,
                reporter=summary,
                jobs_per_gpu=concurrent,
            )
            else 1
        )


if __name__ == "__main__":
    main()
