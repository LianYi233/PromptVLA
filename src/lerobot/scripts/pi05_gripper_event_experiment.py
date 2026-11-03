"""Matched 2 x 2 gripper-event sampling / relative action-time experiment.

Run preparation with ``pi05_gripper_event prepare`` first. Training uses one full
LIBERO-long demonstration; evaluation always covers the ten ``libero_10`` tasks.
No GPU training or simulation takes place in the prepare/check/dry-run commands.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

CASES = ("uniform", "event", "uniform_time", "event_time")
PROTOCOL = "pi05-gripper-event-experiment-v1"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = "/data1/wyn/chkpt/2601-lerobot/PromptVLA"


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_record(path, record):
    """Reuse identical provenance, never replace a different experiment record."""
    path = Path(path)
    if path.exists():
        if read_json(path) != record:
            raise ValueError(f"Different experiment already recorded at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def source_identity(source):
    source = Path(source).resolve(strict=True)
    weight = (source / "model.safetensors").resolve(strict=True)
    stat = weight.stat()
    return {
        "path": str(source),
        "config_sha256": file_hash(source / "config.json"),
        "weights_path": str(weight),
        "weights_size": stat.st_size,
        "weights_mtime_ns": stat.st_mtime_ns,
    }


def preparation_module(record):
    """Route saved preparation protocols without changing legacy r1/r2 runs."""
    if record.get("protocol") == "pi05-gripper-additive-long-v1":
        from lerobot.scripts import pi05_gripper_additive as module
    elif record.get("protocol") == "pi05-gripper-event-long-v1":
        from lerobot.scripts import pi05_gripper_event as module
    else:
        raise ValueError("Unrecognized gripper experiment preparation protocol")
    return module


def build_jobs(record, directory, output_root, selection="all", repeat=1):
    module = preparation_module(record)
    case_config, prefix = module.case_config, module.prefix

    if selection not in (*CASES, "all") or type(repeat) is not int or repeat < 1:
        raise ValueError("Choose a known case and a positive repeat identifier")
    settings = record["settings"]
    jobs = []
    for case in CASES if selection == "all" else (selection,):
        factors = case_config(case)
        name = prefix(record, case, repeat=repeat) + (f"-base-vlm2-P1-D{settings['data_seed']}-E1-T1")
        run = Path(output_root).resolve() / name
        jobs.append(
            {
                "case": case,
                **factors,
                "name": name,
                "run": run,
                "plan": Path(directory).resolve() / f"{factors['sampling']}.npz",
                "checkpoint": run / "checkpoints" / f"{settings['steps']:06d}" / "pretrained_model",
            }
        )
    return jobs


def training_command(job, record, args, prompt):
    settings = record["settings"]
    env = os.environ.copy()
    # These inherited controls belong to other experiments, not this fixed recipe.
    for key in ("PROMPT_EXPORT_PATH", "PROMPT_EXPORT_REFERENCE", "RUN_PREFIX", "PDET_CASES"):
        env.pop(key, None)
    env.update(
        PRETRAINED_PATH=str(args.source.resolve()),
        TOKENIZER_PATH=str(args.tokenizer.resolve()),
        DATASET_ROOT=record["dataset_root"],
        DATASET_REPO_ID=args.dataset_repo_id,
        OUTPUT_ROOT=str(args.output_root.resolve()),
        LOG_ROOT=str(args.log_root.resolve()),
        PROMPT_INIT_PATH=str(prompt.resolve()),
        VLM_PROMPT_TOKENS="2",
        TRAIN_ACTION_EXPERT="true",
        FREEZE_VLM_PROMPT="false",
        GPU_IDS=args.gpus,
        NUM_PROCESSES=str(settings["world_size"]),
        MAIN_PROCESS_PORT=str(args.port),
        BATCH_SIZE=str(settings["batch_size"]),
        FLOW_STEPS=str(settings["steps"]),
        SAVE_ONLY_FINAL="true",
        DTYPE="bfloat16",
        RUN_NAME=job["name"],
        DRY_RUN="true" if args.dry_run else "false",
    )
    command = [
        "bash",
        str(ROOT / "examples/training/train_pi05_vlm_only.sh"),
        "base",
        "0",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=true",
        f"--policy.action_time_encoding={str(job['time_encoding']).lower()}",
        "--policy.action_time_encoding_dim=32",
        f"--policy.action_time_encoding_fps={record['fps']}",
        "--pdet.enabled=true",
        "--pdet.prompt_seed=1",
        f"--pdet.data_seed={settings['data_seed']}",
        "--pdet.noise_seed=1",
        "--pdet.time_seed=1",
        "--pdet.flow_binding=occurrence",
        f"--pdet.order_plan={job['plan']}",
        f"--pdet.audit_steps={settings['steps']}",
        "--pdet.diagnostics=true",
        '--keyframe_loss.mode="off"',
        "--prompt_rank.freq=0",
        f"--num_workers={args.num_workers}",
        "--log_freq=10",
        f"--wandb.enable={str(args.wandb_mode != 'disabled').lower()}",
        f"--wandb.mode={'online' if args.wandb_mode == 'disabled' else args.wandb_mode}",
        "--wandb.project=prompt-learning",
    ]
    return command, env


def train(record, jobs, args):
    from lerobot.scripts.libero_vlm_only import validate_source
    from lerobot.utils.prompt_init_names import resolve_libero_prompt
    from lerobot.utils.prompt_initialization import read_initial_prompt

    gpu_ids = args.gpus.split(",")
    if (
        any(not value.isdecimal() for value in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
        or len(gpu_ids) != record["settings"]["world_size"]
    ):
        raise ValueError("GPU_IDS count must match the prepared world_size, without duplicates")
    if not args.tokenizer.is_dir() or not 1 <= args.port <= 65535 or args.num_workers < 0:
        raise ValueError("Check tokenizer directory, port, and num_workers")
    validate_source(args.source, "base", record["dataset_root"])
    source = source_identity(args.source)
    source_config = read_json(args.source / "config.json")
    if source_config.get("chunk_size") != record["selection"]["chunk_size"]:
        raise ValueError("Source model chunk_size must match prepared contiguous chunks")
    prompt = args.prompt or resolve_libero_prompt(args.prompt_dir, 1, 2)
    _, prompt_meta = read_initial_prompt(prompt, 2)
    if prompt_meta["source_training_seed"] != 1 or prompt_meta.get("source_prompt_init_path"):
        raise ValueError("Use an original unswapped seed-1, two-token prompt initialization")
    prompt_digest = file_hash(prompt)
    commands = []
    for job in jobs:
        if job["run"].exists():
            raise FileExistsError(
                f"Output already exists; select unfinished cases or a new repeat: {job['run']}"
            )
        command, env = training_command(job, record, args, prompt)
        subprocess.run(command, env={**env, "DRY_RUN": "true"}, check=True, cwd=ROOT)
        commands.append((job, command, env))
    print(f"All {len(jobs)} training preflights passed. Scope: one episode, LIBERO-long only.", flush=True)
    if args.dry_run:
        return
    for job, command, env in commands:
        if source_identity(args.source) != source or file_hash(prompt) != prompt_digest:
            raise ValueError("Source or imported prompt changed between experimental conditions")
        descriptor = {
            "protocol": (
                "pi05-gripper-additive-experiment-v1"
                if record["protocol"] == "pi05-gripper-additive-long-v1"
                else PROTOCOL
            ),
            "case": job["case"],
            "name": job["name"],
            "preparation": record,
            "source": source,
            "prompt_file_sha256": prompt_digest,
            "prompt_sha256": prompt_meta["prompt_sha256"],
            "command": command,
            "plan_sha256": file_hash(job["plan"]),
            "time_encoding": job["time_encoding"],
        }
        save_record(args.directory / "runs" / f"{job['name']}.json", descriptor)
        print(f"Starting {job['name']}", flush=True)
        try:
            subprocess.run(command, env=env, check=True, cwd=ROOT)
        finally:
            if job["run"].is_dir():
                save_record(job["run"] / "gripper_event_experiment.json", descriptor)
        audit(record, [job])
    if len(jobs) > 1:
        audit(record, jobs)


def audit(record, jobs):
    """Verify actual sampled inputs, full budgets, and shared prompt/flow draws."""
    import numpy as np

    from lerobot.utils.pdet_order import load_plan

    settings = record["settings"]
    streams, descriptors, manifests = {}, {}, {}
    for job in jobs:
        config = read_json(job["checkpoint"] / "config.json")
        training = read_json(job["checkpoint"] / "train_config.json")
        expected = {
            "train_action_expert": True,
            "freeze_vlm_prompt": False,
            "num_vlm_prompt_tokens": 2,
            "num_prompt_tokens": 0,
            "action_time_encoding": job["time_encoding"],
            "action_time_encoding_dim": 32,
            "action_time_encoding_fps": record["fps"],
            "chunk_size": record["selection"]["chunk_size"],
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Checkpoint {job['case']} changed {key}; expected {value}")
        if (training["steps"], training["batch_size"], training["seed"]) != (
            settings["steps"],
            settings["batch_size"],
            0,
        ):
            raise ValueError("Checkpoint training budget/seed differs from preparation")
        descriptor = read_json(job["run"] / "gripper_event_experiment.json")
        if descriptor["preparation"] != record or descriptor["case"] != job["case"]:
            raise ValueError("Run provenance does not match the selected episode and condition")
        if descriptor["plan_sha256"] != file_hash(job["plan"]):
            raise ValueError("Prepared sampling plan changed after training")
        descriptors[job["case"]] = descriptor
        plan = load_plan(job["plan"])
        rows = []
        for rank in range(settings["world_size"]):
            manifest = read_json(job["run"] / "pdet" / f"manifest-rank{rank}.json")
            if (
                manifest["order_plan"] != plan["manifest"]
                or manifest["prompt_sha256"] != descriptor["prompt_sha256"]
                or manifest["world_size"] != settings["world_size"]
                or manifest["per_device_batch_size"] != settings["batch_size"]
                or manifest["total_steps"] != settings["steps"]
            ):
                raise ValueError("Actual PDET training inputs differ from the prepared experiment")
            manifests[(job["case"], rank)] = manifest
            with (job["run"] / "pdet" / f"inputs-rank{rank}.jsonl").open() as stream:
                rank_rows = [json.loads(line) for line in stream if line.strip()]
            if len(rank_rows) != settings["steps"]:
                raise ValueError(f"Incomplete/duplicate input audit for {job['case']} rank {rank}")
            for step, row in enumerate(rank_rows):
                start = (step * settings["world_size"] + rank) * settings["batch_size"]
                occurrences = plan["order"][start : start + settings["batch_size"]]
                if (
                    row["step"] != step
                    or row["rank"] != rank
                    or not np.array_equal(row["occurrence_ids"], occurrences)
                    or not np.array_equal(row["sample_ids"], plan["index"][occurrences])
                ):
                    raise ValueError("Observed samples do not follow the prepared plan")
                rows.append((row["sample_ids_sha256"], row["noise_sha256"], row["time_sha256"]))
        streams[job["case"]] = rows
    first = jobs[0]["case"]
    for case in streams:
        for key in ("source", "prompt_file_sha256", "prompt_sha256"):
            if descriptors[first][key] != descriptors[case][key]:
                raise ValueError(f"Experimental conditions differ in {key}")
        for key in ("source_tree_sha256", "versions", "cuda"):
            if manifests[(first, 0)][key] != manifests[(case, 0)][key]:
                raise ValueError(f"Experimental conditions used different code/environment: {key}")
        if [row[1:] for row in streams[first]] != [row[1:] for row in streams[case]]:
            raise ValueError("Noise/flow-time draws differ across conditions")
    for sampling in ("uniform", "event"):
        if (
            sampling in streams
            and f"{sampling}_time" in streams
            and streams[sampling] != streams[f"{sampling}_time"]
        ):
            raise ValueError("Time on/off conditions did not receive identical training inputs")
    result = {"audit": "passed", "cases": list(streams), "steps": settings["steps"]}
    print(json.dumps(result), flush=True)
    return result


def fixed_noise_config(args):
    """A single immutable corruption protocol shared across all evaluation cases."""
    from lerobot.scripts.libero_fixed_noise import PROTOCOL, validate_config

    std = getattr(args, "noise_std", 0.0)
    if type(std) not in (int, float) or not math.isfinite(std) or not 0 <= std <= 1:
        raise ValueError("--noise-std must be finite and in [0,1]")
    # Legacy direct evaluation is unchanged at std=0. Additive launchers
    # enable the fixed protocol even at zero std: this gives a clean control
    # with the same deterministic per-episode policy RNG as noisy evaluation.
    matched_clean = getattr(args, "fixed_eval", False)
    if std == 0 and not matched_clean:
        return None
    return validate_config(
        {
            "protocol": PROTOCOL,
            "kind": "gaussian",
            "std": float(std),
            "seed": getattr(args, "noise_seed", 20261102),
            "scope": getattr(args, "noise_scope", "task8"),
            "eval_seed": args.eval_seed,
            "suite": "libero_10",
            "target_task_id": 8,
        }
    )


def fixed_noise_path(args, record):
    return evaluation_root(args, record) / "fixed_image_noise_config.json"


def ensure_noise_config(path, config):
    """Serialize shared protocol creation under a filesystem lock (four GPUs)."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ".fixed_image_noise.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        save_record(path, config)


def evaluation_root(args, record):
    # Do not inherit BASE_OUTPUT from unrelated curriculum experiments. The
    # additive experiment gets its own default root; --eval-root can override.
    default_name = (
        "gripper-additive"
        if record.get("protocol") == "pi05-gripper-additive-long-v1"
        else "gripper-event"
    )
    base = args.eval_root or args.output_root / "eval" / default_name
    noise = fixed_noise_config(args)
    if noise:
        fingerprint = hashlib.sha256(json.dumps(noise, sort_keys=True).encode()).hexdigest()[:12]
        sigma_label = str(noise["std"]).replace(".", "p")
        base = (
            Path(base) / "fixed-image-noise"
            / f"scope-{noise['scope']}-sigma-{sigma_label}-seed{noise['seed']}-{fingerprint}"
        )
    selection = record["selection"]
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:12]
    experiment_name = f"task{selection['task_id']}-ep{selection['episode_index']}-{identity}"
    return base / experiment_name / f"r{args.repeat}-seed{args.eval_seed}-n{args.episodes}"


def result_path(job, record, args):
    return (
        evaluation_root(args, record)
        / (f"libero060-all-{job['name']}-{record['settings']['steps']:06d}-libero_10-resume")
        / "eval_info.json"
    )


def evaluate(record, jobs, args):
    # The common evaluator supports PLUS/PRO via environment overrides. This
    # experiment explicitly requires the standard ten long-horizon tasks.
    if os.environ.get("LIBERO_ROBUSTNESS_MANIFEST") or os.environ.get("LIBERO_BENCHMARK"):
        raise ValueError("Unset LIBERO_ROBUSTNESS_MANIFEST/LIBERO_BENCHMARK for standard LIBERO-long")
    audit(record, jobs)
    noise_spec = fixed_noise_config(args)
    noise_path = fixed_noise_path(args, record) if noise_spec else None
    if noise_spec and not args.dry_run:
        ensure_noise_config(noise_path, noise_spec)
    if noise_spec:
        print(
            f"Fixed Gaussian image noise: scope={noise_spec['scope']} "
            f"std={noise_spec['std']} seed={noise_spec['seed']} "
            f"eval_seed={noise_spec['eval_seed']} config={noise_path}",
            flush=True,
        )
    print(
        f"Evaluation settings: standard libero_10; episodes_per_task={args.episodes}; "
        f"eval_seed={args.eval_seed}; noise_std={getattr(args, 'noise_std', 0)}; "
        f"output={evaluation_root(args, record)}",
        flush=True,
    )
    commands = []
    for job in jobs:
        env = os.environ.copy()
        # Do not mix result/log roots or a noise manifest from another project.
        for stale in (
            "LIBERO_FIXED_NOISE_CONFIG", "BASE_OUTPUT", "EVAL_LOG_ROOT",
            "LIBERO_PROGRESS_FD", "LIBERO_PROGRESS_LABEL",
        ):
            env.pop(stale, None)
        if noise_path:
            env["LIBERO_FIXED_NOISE_CONFIG"] = str(noise_path)
        for key in ("BASE_CKPT", "BASE_MODEL_PATH", "RUN_PREFIX", "PDET_CASES"):
            env.pop(key, None)
        descriptor = read_json(job["run"] / "gripper_event_experiment.json")
        env.update(
            PRETRAINED_PATH=descriptor["source"]["path"],
            RUN_NAME=job["name"],
            TOKENIZER_PATH=str(args.tokenizer.resolve()),
            CKPT_ROOT=str(args.output_root.resolve()),
            BASE_OUTPUT=str(evaluation_root(args, record).resolve()),
            LOG_ROOT=str(args.log_root.resolve()),
            EVAL_LOG_ROOT=str(
                (args.log_root / "eval" / (
                    "gripper-additive"
                    if record.get("protocol") == "pi05-gripper-additive-long-v1"
                    else "gripper-event"
                )).resolve()
            ),
            TRAIN_ACTION_EXPERT="true",
            VLM_PROMPT_TOKENS="2",
            CHECKPOINT_STEP=str(record["settings"]["steps"]),
            EVAL_SEED=str(args.eval_seed),
            EPISODES_PER_TASK=str(args.episodes),
            GPU_ID=str(args.gpu),
            PROGRESS="true",
            DRY_RUN="true" if args.dry_run else "false",
        )
        command = ["bash", str(ROOT / "run_eval_libero_vlm_only.sh"), "base", "trained", "libero_10", "0"]
        subprocess.run(command, env={**env, "DRY_RUN": "true"}, check=True, cwd=ROOT)
        commands.append((job, command, env))
    if args.dry_run:
        return
    for job, command, env in commands:
        print(f"Evaluating {job['case']}: {result_path(job, record, args)}", flush=True)
        subprocess.run(command, env=env, check=True, cwd=ROOT)
    summarize(record, jobs, args)


def read_eval_result(path, selected_task, episodes):
    info = read_json(path)
    tasks = info.get("per_task", info.get("Aggregated Metrics for per_task", []))
    rows = {}
    for task in tasks:
        task_id = task["task_id"]
        if task["task_group"] != "libero_10" or type(task_id) is not int or task_id in rows:
            raise ValueError("Evaluation contains another suite or duplicate/invalid task IDs")
        values = task.get("metrics", task).get("successes", [])
        if len(values) != episodes or any(
            type(value) not in (bool, int) or value not in (0, 1) for value in values
        ):
            raise ValueError("Evaluation has incomplete/invalid per-episode success data")
        rows[task_id] = {
            "successes": sum(values),
            "episodes": len(values),
            "rate": 100 * sum(values) / len(values),
        }
    if set(rows) != set(range(10)):
        raise ValueError("Expected all ten LIBERO-long tasks; partial results are not a completed evaluation")
    successes = sum(row["successes"] for row in rows.values())
    return {
        "overall_successes": successes,
        "overall_episodes": episodes * 10,
        "overall_rate": 100 * successes / (episodes * 10),
        "selected_task_rate": rows[selected_task]["rate"],
        "per_task": rows,
    }


def summarize(record, jobs, args):
    noise_spec = fixed_noise_config(args)
    results = {}
    corruption_identities = set()
    for job in jobs:
        path = result_path(job, record, args)
        if noise_spec:
            metadata = read_json(path.parent / "resume_manifest.json")
            applied = metadata.get("fixed_observation_noise", {})
            expected_file = fixed_noise_path(args, record)
            if (
                applied.get("config") != noise_spec
                or applied.get("config_sha256") != file_hash(expected_file)
            ):
                raise ValueError("Evaluated image-noise protocol differs from requested fixed settings")
            corruption_identities.add(
                (applied.get("config_sha256"), applied.get("helper_sha256"))
            )
        results[job["case"]] = {
            **read_eval_result(path, record["selection"]["task_id"], args.episodes),
            "eval_info": str(path),
            "checkpoint": str(job["checkpoint"]),
        }
    if len(corruption_identities) > 1:
        raise ValueError("Models were evaluated with different noise protocols or implementations")
    contrasts = {}
    if set(results) == set(CASES):
        for metric in ("selected_task_rate", "overall_rate"):
            u, e, ut, et = (results[case][metric] for case in CASES)
            contrasts[metric] = {
                "event_minus_uniform_pp": e - u,
                "time_under_uniform_pp": ut - u,
                "time_under_event_pp": et - e,
                "interaction_pp": (et - e) - (ut - u),
            }
    summary = {
        "protocol": PROTOCOL,
        "selection": record["selection"],
        "eval_seed": args.eval_seed,
        "episodes_per_task": args.episodes,
        "fixed_observation_noise": noise_spec,
        "success_rate_scope": (
            "all ten tasks clean; deterministic per-episode policy RNG"
            if noise_spec and noise_spec["std"] == 0
            else "task8 corrupted; other nine tasks clean"
            if noise_spec and noise_spec["scope"] == "task8"
            else "all ten tasks corrupted" if noise_spec else "all ten tasks clean"
        ),
        "results": results,
        "contrasts": contrasts,
        "scope": "One-demonstration adaptation diagnostic; not a full-dataset or unseen-task generalization claim.",
    }
    destination = evaluation_root(args, record)
    destination.mkdir(parents=True, exist_ok=True)
    suffix = "all" if len(jobs) == 4 else jobs[0]["case"]
    target = destination / f"summary-{suffix}.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    with (destination / f"summary-{suffix}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "selected_task_rate", "overall_rate", *[f"task_{i}" for i in range(10)]])
        for case, result in results.items():
            writer.writerow(
                [
                    case,
                    result["selected_task_rate"],
                    result["overall_rate"],
                    *[result["per_task"][i]["rate"] for i in range(10)],
                ]
            )
    print("case              selected task    LIBERO-long overall")
    for case, result in results.items():
        print(f"{case:18s} {result['selected_task_rate']:8.2f}%        {result['overall_rate']:8.2f}%")
    print(f"Results: {target}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval", "audit", "summarize"))
    parser.add_argument("case", nargs="?", default="all", choices=(*CASES, "all"))
    parser.add_argument(
        "--directory",
        type=Path,
        default=os.environ.get("GRIPPER_EVENT_DIR", "/data1/wyn/prompt-work/gripper-event-long-task8-r1"),
    )
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("REPEAT_ID", "1")))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=os.environ.get("OUTPUT_ROOT", os.environ.get("CKPT_ROOT", DEFAULT_OUTPUT)),
    )
    parser.add_argument(
        "--log-root", type=Path, default=os.environ.get("LOG_ROOT", "/data1/wyn/chkpt/2601-lerobot/logs")
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=os.environ.get(
            "PRETRAINED_PATH",
            os.environ.get("BASE_PRETRAINED_PATH", "/home/wyn/models/lerobot/pi05_libero_base"),
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=os.environ.get("TOKENIZER_PATH", "/home/wyn/models/paligemma-3b-pt-224"),
    )
    parser.add_argument("--prompt", type=Path, default=os.environ.get("PROMPT_INIT_PATH"))
    parser.add_argument(
        "--prompt-dir", type=Path, default=os.environ.get("PROMPT_INIT_DIR", "/data1/wyn/prompt-work")
    )
    parser.add_argument("--dataset-repo-id", default=os.environ.get("DATASET_REPO_ID", "libero"))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GPU_ID", "0")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MAIN_PROCESS_PORT", "29589")))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("GRIPPER_EXPERIMENT_DEFAULT_WANDB_MODE", "online"),
    )
    parser.add_argument("--eval-root", type=Path, default=os.environ.get("GRIPPER_EVAL_ROOT"))
    parser.add_argument("--eval-seed", type=int, default=int(os.environ.get("EVAL_SEED", "1000")))
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get(
            "GRIPPER_EVAL_DEFAULT_EPISODES",
            os.environ.get("EPISODES_PER_TASK", "10"),
        )),
    )
    parser.add_argument(
        "--fixed-eval",
        action="store_true",
        default=os.environ.get("GRIPPER_FIXED_EVAL", "false").lower() == "true",
        help="Use matched per-episode policy seeding even if --noise-std=0",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=float(os.environ.get("GRIPPER_FIXED_NOISE_STD", "0")),
        help="Gaussian observation RGB std in [0,1]; zero keeps the clean benchmark",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=int(os.environ.get("GRIPPER_FIXED_NOISE_SEED", "20261102")),
    )
    parser.add_argument(
        "--noise-scope",
        choices=("task8", "all"),
        default=os.environ.get("GRIPPER_FIXED_NOISE_SCOPE", "task8"),
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "false") == "true"
    )
    args = parser.parse_args()
    try:
        if args.repeat < 1 or args.episodes < 1 or args.eval_seed < 0 or args.gpu < 0:
            raise ValueError("Invalid repeat, episodes, evaluation seed, or GPU")
        record = preparation_module(read_json(args.directory / "manifest.json")).check(args.directory)
        # Legacy direct calls may have relied on BASE_OUTPUT as a default.
        # Additive evaluations never inherit that unrelated global variable.
        if (
            args.eval_root is None
            and record.get("protocol") == "pi05-gripper-event-long-v1"
            and os.environ.get("BASE_OUTPUT")
        ):
            args.eval_root = Path(os.environ["BASE_OUTPUT"])
        jobs = build_jobs(record, args.directory, args.output_root, args.case, args.repeat)
        if args.command == "train":
            train(record, jobs, args)
        elif args.command == "eval":
            evaluate(record, jobs, args)
        elif args.command == "audit":
            audit(record, jobs)
        else:
            if args.dry_run:
                raise ValueError("summarize writes JSON/CSV reports; omit --dry-run")
            audit(record, jobs)
            summarize(record, jobs, args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Gripper-event experiment failed: {error}\n")


if __name__ == "__main__":
    main()
