"""Matched D/E/F seed repeats; two GPU pairs, three automatic rounds, final models only.

D1/E1/F1 and D2/E2/F2 are block labels, not literal seeds. Defaults use new
P/D/E/T/global/resampling seeds 2 and 3; diagnostics keep their fixed seed 2026.
"""

import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from lerobot.scripts import (
    pi05_curriculum_ablation as runner,
    pi05_ratio_curriculum as curriculum,
    pi05_source_compare as sources,
)
from lerobot.scripts.curriculum_audit_utils import differences, normalize_path_aliases
from lerobot.utils.pdet_order import META_COLUMNS, arrays_hash, create_plan, load_plan, save_plan

PROTOCOL = "pi05-def-seed-repeat-v1"
DIRECTIONS = {"D": "hard_to_easy", "E": "easy_to_hard", "F": "random"}
TRAIN = runner.ROOT / "examples/training/train_pi05_vlm_only.sh"


def build_jobs(env, extra):
    # Reuse the established GPU/port/path validation and native hyperparameters.
    settings = dict(env, FLOW_STEPS="6000", REPEAT_ID="2")
    settings.setdefault("PORT_A", "29592")
    settings.setdefault("PORT_B", "29593")
    templates = {job["cell"]: job for job in runner.build_jobs(settings, extra)}
    seeds = [int(env.get(f"SEED_SET_{i}", default)) for i, default in ((1, "2"), (2, "3"))]
    if len(set(seeds)) != 2 or any(not 2 <= seed < 2**32 for seed in seeds):
        raise ValueError("Choose two distinct NEW seeds >= 2 and < 2**32")
    suffix = f"curriculum-def-seeds-{seeds[0]}-{seeds[1]}"
    work = Path(env.get("SEED_WORK_DIR", f"/data1/wyn/prompt-work/{suffix}")).resolve()
    jobs = []
    for round_id, (letter, direction) in enumerate(DIRECTIONS.items(), 1):
        for block, seed in enumerate(seeds, 1):
            cell = f"{letter}{block}"
            name = (
                f"pi05-curriculum-{cell}-{curriculum.DIRECTIONS[direction]}-e50-h50-S{seed}"
                f"-steps6000-aexpert-libero-Bocc-vlm2-P{seed}-D{seed}-E{seed}-T{seed}"
            )
            parent = templates["D" if block == 1 else "F"]["env"]
            child = {
                **parent,
                "CURRICULUM_DIRECTION": direction,
                "SAMPLE_SEED": str(seed),
                "EASY_PERCENT": "50",
                "DATASET_REPO_ID": "libero",
                "RUN_NAME": name,
                "PRETRAINED_PATH": str(Path(parent["SOURCE_PAIR_DIR"]) / "libero"),
                "VLM_PROMPT_TOKENS": "2",
                "TRAIN_ACTION_EXPERT": "true",
                "FREEZE_VLM_PROMPT": "false",
                "SAVE_ONLY_FINAL": "true",
                "SAVE_FREQ": "6000",
                "ORDER_DIR": str(work / cell),
                "PROMPT_INIT_PATH": str(
                    work / "inits" / f"pi05-libero-base-vlm2-init-seed{seed}.safetensors"
                ),
            }
            plan = str(work / cell / "plan.npz")
            command = [
                "bash",
                str(TRAIN),
                "base",
                str(seed),
                "--policy.compile_model=false",
                "--pdet.enabled=true",
                f"--pdet.prompt_seed={seed}",
                f"--pdet.data_seed={seed}",
                f"--pdet.noise_seed={seed}",
                f"--pdet.time_seed={seed}",
                f"--pdet.order_plan={plan}",
                "--pdet.flow_binding=occurrence",
                "--pdet.audit_steps=6000",
                "--pdet.diagnostics=true",
                "--prompt_rank.freq=250",
                "--prompt_rank.samples=16",
                "--prompt_rank.output_probes=4",
                "--prompt_rank.seed=2026",
                "--prompt_rank.timesteps=[0.2,0.5,0.8]",
                "--prompt_rank.extra_steps=[50,100]",
                "--prompt_rank.fixed_flow_loss=true",
                "--log_freq=10",
                *extra,
            ]
            jobs.append(
                {
                    "round": round_id,
                    "cell": cell,
                    "block": block,
                    "seed": seed,
                    "name": name,
                    "env": child,
                    "work": str(work),
                    "run": str(Path(child["OUTPUT_ROOT"]) / name),
                    "command": command,
                }
            )
    directory = Path(env.get("SEED_LOG_DIR", str(Path(jobs[0]["env"]["LOG_ROOT"]) / suffix))).resolve()
    return jobs, directory


def plans_for_seed(scores, score_manifest, seed, *, parent_steps=3000, steps=6000, batch=32, world=2):
    """Same finite-pool procedure as D/E/F, with only its random seed changed."""
    baseline = create_plan(
        {key: scores[key] for key in META_COLUMNS},
        steps=parent_steps,
        batch_size=batch,
        world_size=world,
        data_seed=seed,
        within_seed=0,
    )
    plans = {}
    for letter, direction in DIRECTIONS.items():
        plan = curriculum.make_plan(
            baseline,
            scores,
            score_manifest,
            direction=direction,
            easy_percent=50,
            sample_seed=seed,
            steps=steps,
        )
        # The inherited algorithm's human-readable scope refers to historical D0.
        plan["manifest"]["scope"] = (
            f"Resampled D{seed}/W0 candidates, 50/50 exposure, cycled task slots; "
            "immutable new occurrence IDs carry E/T across orders"
        )
        plans[letter] = plan
    pools = {p["manifest"]["instance_pool_sha256"] for p in plans.values()}
    tasks = {p["manifest"]["task_schedule_sha256"] for p in plans.values()}
    orders = {arrays_hash({"order": p["order"]}) for p in plans.values()}
    if len(pools) != 1 or len(tasks) != 1 or len(orders) != 3:
        raise ValueError("Expected one shared pool/task schedule and three distinct orders per seed")
    return plans


def prepare(jobs):
    from lerobot.scripts.frameskip_scores import check_inputs
    from lerobot.utils.frameskip_curriculum import load_scores

    env = jobs[0]["env"]
    pair = Path(env["SOURCE_PAIR_DIR"])
    source = curriculum.original.reference_inputs(pair, Path(env["REFERENCE_RUN"]))
    if not Path(source["dataset_root"]).samefile(env["DATASET_ROOT"]):
        raise ValueError("Dataset differs from historical D/E/F")
    check_inputs(Path(env["SCORES_PATH"]), Path(source["dataset_root"]), Path(source["config_source"]))
    scores, score_manifest = load_scores(env["SCORES_PATH"])
    work = Path(jobs[0]["work"])
    work.mkdir(parents=True, exist_ok=True)
    record = {
        "protocol": PROTOCOL,
        "pair": str(pair),
        "pair_sha256": sources.digest(pair / "pair.json"),
        "scores": env["SCORES_PATH"],
        "scores_sha256": sources.digest(env["SCORES_PATH"]),
        "seeds": [job["seed"] for job in jobs if job["round"] == 1],
        "parent_budget": {"steps": 3000, "batch_size": 32, "world_size": 2},
        "training_budget": {"steps": 6000, "batch_size": 32, "world_size": 2},
        "plans": {},
    }
    for block in (1, 2):
        selected = [job for job in jobs if job["block"] == block]
        plans = plans_for_seed(scores, score_manifest, selected[0]["seed"])
        for job in selected:
            plan = plans[job["cell"][0]]
            path = Path(job["env"]["ORDER_DIR"]) / "plan.npz"
            save_plan(path, plan)  # Idempotent only when every array/manifest still agrees.
            record["plans"][job["cell"]] = plan["manifest"]
    path = work / "plans.json"
    if path.exists() and sources.read_json(path) != record:
        raise ValueError(f"Prepared inputs changed; existing plans are protected: {path}")
    runner.atomic_json(path, record)
    return record


def validate_init(job):
    from lerobot.utils.prompt_initialization import read_initial_prompt

    path = Path(job["env"]["PROMPT_INIT_PATH"])
    _, metadata = read_initial_prompt(path, 2)
    if (
        metadata["source_training_seed"] != job["seed"]
        or metadata.get("source_prompt_init_path")
        or metadata.get("source_kind") != "regenerated_training_initialization"
    ):
        raise ValueError(f"Prompt is not a fresh native initialization for seed {job['seed']}: {path}")
    if Path(metadata["source_pretrained_path"]).resolve() != Path(job["env"]["PRETRAINED_PATH"]).resolve():
        raise ValueError("Initialization was constructed from a different pretrained model")
    return metadata


def validate_recipe(cfg, job):
    """Catch unintended changes before accepting an expensive completed run."""
    policy = cfg["policy"]
    expected = {
        **curriculum.scope_flags("joint"),
        "dtype": "bfloat16",
        "compile_model": False,
        "gradient_checkpointing": True,
        "optimizer_lr": 2.5e-5,
        "optimizer_grad_clip_norm": 0.0,
        "scheduler_warmup_steps": 1000,
        "scheduler_decay_steps": 30000,
        "scheduler_decay_lr": 1e-5,
        "ntk_save_stage_snapshots": False,
    }
    for key, value in expected.items():
        if policy.get(key, False) != value:
            raise ValueError(f"Changed fixed D/E/F policy setting: {key}")
    rank = cfg["prompt_rank"]
    for key, value in {
        "freq": 250,
        "samples": 16,
        "output_probes": 4,
        "seed": 2026,
        "timesteps": [0.2, 0.5, 0.8],
        "extra_steps": [50, 100],
        "fixed_flow_loss": True,
    }.items():
        if rank.get(key) != value:
            raise ValueError(f"Changed fixed diagnostics: {key}")
    if cfg["save_steps"] != [] or not cfg["save_checkpoint"] or cfg["env_eval_freq"] != 0:
        raise ValueError("Expected final-only checkpoint saving and no training-time evaluation")
    for value, expected_path in (
        (cfg["dataset"]["root"], job["env"]["DATASET_ROOT"]),
        (policy["tokenizer_name"], job["env"]["TOKENIZER_PATH"]),
    ):
        if Path(value).resolve() != Path(expected_path).resolve():
            raise ValueError("Changed dataset/tokenizer")


def ensure_init(job, directory, *, dry=False):
    path = Path(job["env"]["PROMPT_INIT_PATH"])
    if path.exists():
        validate_init(job)
        return
    if dry:
        return  # Real export requires the GPUs; dry-run must never construct the model.
    export = copy.deepcopy(job)
    export["env"].pop("PROMPT_INIT_PATH")
    export["env"].update(
        PROMPT_EXPORT_PATH=str(path),
        TRAIN_ACTION_EXPERT="false",
        RUN_NAME=f"pi05-def-init-vlm2-seed{job['seed']}",
        OUTPUT_ROOT=str(Path(job["work"]) / "initialization-runs"),
    )
    command = [
        "bash",
        str(TRAIN),
        "base",
        str(job["seed"]),
        "--policy.compile_model=false",
        "--wandb.enable=false",
    ]
    log = directory / f"P{job['seed']}.export.log"
    with log.open("a") as stream:
        result = subprocess.run(
            command,
            cwd=runner.ROOT,
            env=runner.child_env(export),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if result.returncode:
        raise RuntimeError(f"Initialization export failed (exit={result.returncode}); inspect {log}")
    validate_init(job)


def check_checkpoint(job):
    from lerobot.scripts.libero_vlm_only import validate_checkpoint
    from lerobot.scripts.pdet_order_binding import inspect_run

    run = Path(job["run"])
    target = run / "checkpoints/006000"
    last = run / "checkpoints/last"
    if not target.is_dir() or not last.is_symlink() or last.resolve() != target.resolve():
        raise ValueError(f"Incomplete run; never overwritten automatically: {run}")
    if {p.name for p in (run / "checkpoints").iterdir() if p.name.isdigit()} != {"006000"}:
        raise ValueError(f"Expected only final 006000: {run}")
    checkpoint = target / "pretrained_model"
    cfg = sources.read_json(checkpoint / "train_config.json")
    validate_recipe(cfg, job)
    policy, factors = cfg["policy"], cfg["pdet"]
    expected = {f"{key}_seed": job["seed"] for key in ("prompt", "data", "noise", "time")}
    expected.update(enabled=True, flow_binding="occurrence", audit_steps=6000, diagnostics=True)
    if any(factors.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Checkpoint P/D/E/T/binding disagrees with {job['cell']}")
    if (cfg["seed"], cfg["steps"], cfg["batch_size"], cfg["save_freq"]) != (job["seed"], 6000, 32, 6000):
        raise ValueError("Changed global seed, steps, batch size or save frequency")
    if Path(factors["order_plan"]).resolve() != Path(job["env"]["ORDER_DIR"]) / "plan.npz":
        raise ValueError("Checkpoint used the wrong order plan")
    if Path(cfg["vlm_prompt_init_path"]).resolve() != Path(job["env"]["PROMPT_INIT_PATH"]):
        raise ValueError("Checkpoint used the wrong prompt initialization")
    saved_policy = sources.read_json(checkpoint / "config.json")
    for key, value in curriculum.scope_flags("joint").items():
        if saved_policy.get(key, False) != value:
            raise ValueError(f"Wrong training scope: {key}")
    if (
        policy["compile_model"]
        or cfg.get("sample_weighting") is not None
        or (
            (cfg.get("keyframe_loss") or {}).get("mode", "off") != "off"
            or (cfg.get("keyframe_loss") or {}).get("enabled", False)
        )
    ):
        raise ValueError("Unexpected compile or loss weighting")
    validate_checkpoint(
        checkpoint, 2, source=job["env"]["PRETRAINED_PATH"], kind="base", train_action_expert=True
    )
    plan = load_plan(factors["order_plan"])
    prepared = sources.read_json(Path(job["work"]) / "plans.json")["plans"][job["cell"]]
    if plan["manifest"] != prepared:
        raise ValueError("Prepared plan changed")

    def name_checker(actual, values, tokens, within_seed):
        if Path(actual).name != job["name"] or tokens != 2 or within_seed != 0:
            raise ValueError("Unexpected cell name/tokens/within-task protocol")
        if any(values.get(key) != value for key, value in expected.items()):
            raise ValueError("Trace seeds disagree with checkpoint")
        return job["block"]

    invariant, report = inspect_run(run, name_checker=name_checker)
    if invariant["prompt_sha256"] != validate_init(job)["prompt_sha256"]:
        raise ValueError("Realized initial prompt differs from the exported tensor")
    if (
        invariant["global_seed"],
        invariant["total_steps"],
        invariant["per_device_batch_size"],
        invariant["world_size"],
    ) != (job["seed"], 6000, 32, 2):
        raise ValueError("Trace seed/budget differs from checkpoint")
    runner.atomic_json(run / "seed_repeat_audit.json", {"audit": "passed", "cell": job["cell"], **report})
    return invariant, report


def audit_suite(jobs):
    invariants, reports = {}, []
    for job in jobs:
        invariant, report = check_checkpoint(job)
        invariants[job["cell"]] = invariant
        reports.append({"cell": job["cell"], "seed": job["seed"], **report})
    for block in (1, 2):
        cells = [f"{letter}{block}" for letter in DIRECTIONS]
        values = [invariants[cell] for cell in cells]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Changed recipe/source/P/pool/task schedule within seed block {block}")
        rows = [r for r in reports if r["cell"] in cells]
        if len({r["bank_sha256"] for r in rows}) != 1 or len({r["order_sha256"] for r in rows}) != 3:
            raise ValueError("Within a triplet require identical realized E/T bank and distinct orders")
    left, right = [copy.deepcopy(invariants[f"D{block}"]) for block in (1, 2)]
    for value in (left, right):
        for key in ("global_seed", "prompt_sha256", "experiment", "pool", "tasks"):
            value.pop(key)
        for key in ("prompt_seed", "data_seed"):
            value["factors"].pop(key)
        value["recipe"].pop("seed")
    normalize_path_aliases(left, right)
    if left != right:
        raise ValueError(f"Non-seed recipe differences across blocks: {differences(left, right)}")
    for key in ("prompt_sha256", "pool", "tasks"):
        if invariants["D1"][key] == invariants["D2"][key]:
            raise ValueError(f"New seeds did not change {key}")
    if reports[0]["bank_sha256"] == reports[1]["bank_sha256"]:
        raise ValueError("New E/T seeds did not change the realized bank")
    return {
        "audit": "passed",
        "protocol": PROTOCOL,
        "runs": reports,
        "scope": "Full realized traces: only order differs within each triplet; fresh P/D/E/T/global/S across triplets. Not numerical determinism or historical code equivalence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "audit", "list"), default="run")
    args, extra = parser.parse_known_args()
    jobs, directory = build_jobs(os.environ, extra)
    if args.command == "list":
        print(json.dumps(jobs, indent=2))
        return
    dry = os.environ.get("DRY_RUN", "false")
    if dry not in ("true", "false"):
        raise ValueError("DRY_RUN must be true or false")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another runner owns {directory}") from error
        with (directory / "runner.log").open("a", buffering=1) as stream:

            def log(message):
                line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
                print(line, flush=True)
                stream.write(line + "\n")

            prepare(jobs)
            runner.atomic_json(directory / "jobs.json", jobs)
            if args.command == "audit":
                runner.atomic_json(directory / "audit.json", audit_suite(jobs))
                log(f"All six passed: {directory / 'audit.json'}")
                return
            from lerobot.utils.pdet import runtime_info

            runtime = runtime_info()
            for job in jobs:
                log(f"PLAN {job['cell']} GPUs={job['env']['GPU_IDS']} seed={job['seed']}: {job['name']}")
            # Reject partial old runs BEFORE any new model construction/training.
            statuses = {job["cell"]: {"state": "pending"} for job in jobs}
            for job in jobs:
                if Path(job["run"]).exists():
                    invariant, _ = check_checkpoint(job)
                    if any(invariant[key] != runtime[key] for key in runtime):
                        raise ValueError("Existing run used different code/environment; not silently mixed")
                    statuses[job["cell"]]["state"] = "complete"
            state_path = directory / "status.json"
            runner.atomic_json(state_path, statuses)
            if dry == "true":
                log(f"CPU plans validated; no GPU export/training. Commands: {directory / 'jobs.json'}")
                return
            for job in jobs[:2]:
                log(f"Validate/export fresh native P{job['seed']} (zero updates)")
                ensure_init(job, directory)
            for job in jobs:
                if statuses[job["cell"]]["state"] == "complete":
                    continue
                preflight = directory / f"{job['cell']}.preflight.log"
                with preflight.open("w") as output:
                    result = subprocess.run(
                        job["command"],
                        cwd=runner.ROOT,
                        env=runner.child_env(job, dry_run=True),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode:
                    raise RuntimeError(f"Preflight failed: {preflight}")

            def interrupted(signum, _frame):
                raise KeyboardInterrupt(f"Received signal {signum}")

            previous = signal.signal(signal.SIGTERM, interrupted)
            try:
                for round_id in (1, 2, 3):
                    if runtime_info() != runtime:
                        raise RuntimeError(
                            "Code/environment changed while suite was running; stop to avoid mixing"
                        )
                    log(f"ROUND {round_id}/3")
                    runner.run_round(
                        [job for job in jobs if job["round"] == round_id],
                        statuses,
                        state_path,
                        log,
                        checkpoint_validator=check_checkpoint,
                    )
                runner.atomic_json(directory / "audit.json", audit_suite(jobs))
                log(f"DONE all six; final checkpoints verified. Audit: {directory / 'audit.json'}")
            finally:
                signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
