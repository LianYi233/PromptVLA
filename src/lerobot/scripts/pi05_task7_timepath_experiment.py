"""Matched one-demo LIBERO-long task7 temporal-injection experiment (1000 steps).

Cases: rope (existing RoPE only), time_qk (first expert layer Q/K only),
time_v (first expert layer V only), time_input (action input embeddings).
Every case uses the EXACT same existing uniform 1000-update plan and
matching seed-1 prompt, flow noise and source policy. No Event resampling.

Time-QK/V have ONE zero-initialized 1024x32 time projection shared by the
selected first-layer native projections. Gradient-checkpoint replay and
cached denoising see the same stateless hooks. RoPE itself is unchanged.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from lerobot.scripts import pi05_gripper_event as preparation
from lerobot.scripts import pi05_gripper_event_experiment as previous
from lerobot.scripts.pi05_task7_object_shift import (
    PROTOCOL as SHIFT_PROTOCOL,
    validate_config as validate_shift,
)

PROTOCOL = "pi05-task7-timepath-qkv-v1"
CASES = ("rope", "time_qk", "time_v", "time_input")
MODES = {"rope": (False, "input"), "time_qk": (True, "qk"),
         "time_v": (True, "v"), "time_input": (True, "input")}
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = Path("/data1/wyn/prompt-work/gripper-event-task7-1200/base")
DEFAULT_OUTPUT = Path("/data1/wyn/chkpt/2601-lerobot/PromptVLA")
DEFAULT_EVAL = DEFAULT_OUTPUT / "eval" / "task7-timepath-qkv"


def check_plan(directory):
    record = preparation.check(directory)
    if (
        record["selection"]["task_id"] != 7
        or record["settings"]["steps"] != 1000
        or record["settings"]["batch_size"] != 32
        or record["settings"]["world_size"] != 2
        or record["selection"]["chunk_size"] != 50
    ):
        raise ValueError("Expected task7 one-demo uniform 1000-step, batch32x2, chunk50 plan")
    path = Path(directory) / "uniform.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing 1000-step Uniform plan: {path}")
    return record


def build_jobs(record, directory, output_root, selection="all", repeat=1):
    if selection not in (*CASES, "all") or type(repeat) is not int or repeat < 1:
        raise ValueError("Expected one of the four timepath cases and repeat>=1")
    selection_meta = record["selection"]
    jobs = []
    for case in CASES if selection == "all" else (selection,):
        enabled, injection = MODES[case]
        name = (
            f"pi05-task7-timepath-r{repeat}-{case}-task7-ep{selection_meta['episode_index']}"
            "-steps1000-aexpert-base-vlm2-P1-D0-E1-T1"
        )
        run = Path(output_root).resolve() / name
        jobs.append({
            "case": case,
            "time_encoding": enabled,
            "time_injection": injection,
            "name": name,
            "run": run,
            "plan": Path(directory).resolve() / "uniform.npz",
            "checkpoint": run / "checkpoints" / "001000" / "pretrained_model",
        })
    return jobs


def training_command(job, record, args, prompt):
    command, env = previous.training_command(job, record, args, prompt)
    command.append(f"--policy.action_time_injection={job['time_injection']}")
    return command, env


def _descriptor(job, record, source, prompt_digest, prompt_meta, command):
    return {
        "protocol": PROTOCOL,
        "case": job["case"],
        "name": job["name"],
        "preparation": record,
        "source": source,
        "prompt_file_sha256": prompt_digest,
        "prompt_sha256": prompt_meta["prompt_sha256"],
        "command": command,
        "plan_sha256": previous.file_hash(job["plan"]),
        "time_encoding": job["time_encoding"],
        "action_time_injection": job["time_injection"],
    }


def train(record, jobs, args):
    from lerobot.scripts.libero_vlm_only import validate_source
    from lerobot.utils.prompt_init_names import resolve_libero_prompt
    from lerobot.utils.prompt_initialization import read_initial_prompt

    gpu_ids = args.gpus.split(",")
    if (
        len(gpu_ids) != 2
        or len(set(gpu_ids)) != 2
        or not all(s.isdecimal() for s in gpu_ids)
    ):
        raise ValueError("This study requires two distinct DDP GPU indices")
    if not args.tokenizer.is_dir() or not 1 <= args.port <= 65535 or args.num_workers < 0:
        raise ValueError("Tokenizer, port or workers are invalid")
    validate_source(args.source, "base", record["dataset_root"])
    source = previous.source_identity(args.source)
    config = previous.read_json(args.source / "config.json")
    if config.get("chunk_size") != 50:
        raise ValueError("Source policy must predict contiguous 50-step action chunks")
    prompt = args.prompt or resolve_libero_prompt(args.prompt_dir, 1, 2)
    _, prompt_meta = read_initial_prompt(prompt, 2)
    if prompt_meta["source_training_seed"] != 1 or prompt_meta.get("source_prompt_init_path"):
        raise ValueError("Require the exact unswapped seed-1 two-token initial prompt")
    prompt_digest = previous.file_hash(prompt)

    commands = []
    for job in jobs:
        if job["run"].exists():
            raise FileExistsError(
                f"Run exists: {job['run']}; train unfinished cases individually or use a new repeat"
            )
        cmd, env = training_command(job, record, args, prompt)
        subprocess.run(cmd, env={**env, "DRY_RUN": "true"}, check=True, cwd=ROOT)
        commands.append((job, cmd, env))
    print(f"Timepath training preflights passed: {len(commands)} cases, 1000 updates each", flush=True)
    if args.dry_run:
        return
    for job, cmd, env in commands:
        if previous.source_identity(args.source) != source or previous.file_hash(prompt) != prompt_digest:
            raise ValueError("Starting checkpoint or prompt initialization changed between cases")
        descriptor = _descriptor(job, record, source, prompt_digest, prompt_meta, cmd)
        previous.save_record(args.directory / "timepath_runs" / f"{job['name']}.json", descriptor)
        print(f"Training timepath={job['case']} checkpoint=001000", flush=True)
        try:
            subprocess.run(cmd, env=env, check=True, cwd=ROOT)
        finally:
            if job["run"].is_dir():
                previous.save_record(job["run"] / "gripper_event_experiment.json", descriptor)
        audit(record, [job])
    if len(jobs) > 1:
        audit(record, jobs)


def audit(record, jobs):
    """Check actual PDET streams and checkpoint mode for every case."""
    from lerobot.utils.pdet_order import load_plan

    previous.audit(record, jobs)
    planned = load_plan(jobs[0]["plan"])
    if planned["manifest"]["steps"] != 1000:
        raise ValueError("Incorrect source Uniform plan budget")
    reference = None
    for job in jobs:
        config = previous.read_json(job["checkpoint"] / "config.json")
        if config.get("action_time_injection", "input") != job["time_injection"]:
            raise ValueError(f"Checkpoint {job['case']} has wrong temporal injection path")
        descriptor = previous.read_json(job["run"] / "gripper_event_experiment.json")
        if descriptor.get("action_time_injection") != job["time_injection"]:
            raise ValueError("Run provenance does not match temporal path")
        if descriptor.get("protocol") != PROTOCOL:
            raise ValueError("Not a task7 timepath checkpoint")
        if job["time_encoding"]:
            from safetensors import safe_open

            with safe_open(job["checkpoint"] / "model.safetensors", framework="pt", device="cpu") as f:
                if "model.action_time_encoder.weight" not in f.keys():
                    raise ValueError(f"Missing learned temporal tensor for {job['case']}")
                shape = tuple(f.get_slice("model.action_time_encoder.weight").get_shape())
                if shape != (1024, 32):
                    raise ValueError(f"Unexpected time projection shape: {shape}")
        rank_lines = []
        for rank in range(record["settings"]["world_size"]):
            path = job["run"] / "pdet" / f"inputs-rank{rank}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != 1000:
                raise ValueError("Incomplete 1000-step sampling audit")
            rank_lines.extend(
                (row["sample_ids_sha256"], row["noise_sha256"], row["time_sha256"])
                for row in rows
            )
        if reference is None:
            reference = rank_lines
        elif reference != rank_lines:
            raise ValueError("Four cases received different samples, flow noise or time draws")
    print(f"Timepath audit passed: {[job['case'] for job in jobs]}", flush=True)


def shift_config(args):
    scene_seed = getattr(args, "scene_seed", None)
    policy_seed = getattr(args, "policy_seed", None)
    if (scene_seed is None) != (policy_seed is None):
        raise ValueError("Supply both --scene-seed and --policy-seed, or neither")
    if scene_seed is not None:
        from lerobot.scripts.pi05_task7_dualseed_shift import (
            PROTOCOL as DUAL_PROTOCOL, validate_config as validate_dual,
        )
        return validate_dual({
            "protocol": DUAL_PROTOCOL, "task_id": 7, "suite": "libero_10",
            "shift_m": float(args.shift_m), "scene_seed": scene_seed,
            "policy_seed": policy_seed, "eval_seed": args.eval_seed,
        })
    return validate_shift({
        "protocol": SHIFT_PROTOCOL, "task_id": 7, "suite": "libero_10",
        "shift_m": float(args.shift_m),
        "seed": args.shift_seed, "eval_seed": args.eval_seed,
    })


def evaluation_root(args, record):
    cfg = shift_config(args)
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:12]
    shift_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    if "scene_seed" in cfg:
        return (
            args.eval_root / "fixed-object-shift-dualseed"
            / f"meters-{cfg['shift_m']:.3f}-scene{cfg['scene_seed']}"
            / f"policy{cfg['policy_seed']}-{shift_hash}"
            / f"task7-ep{record['selection']['episode_index']}-{identity}"
            / f"r{args.repeat}-init{args.eval_seed}-n{args.episodes}-task7-only"
        )
    return (
        args.eval_root / "fixed-object-shift"
        / f"meters-{cfg['shift_m']:.3f}-seed{cfg['seed']}-{shift_hash}"
        / f"task7-ep{record['selection']['episode_index']}-{identity}"
        / f"r{args.repeat}-seed{args.eval_seed}-n{args.episodes}-task7-only"
    )


def result_path(job, record, args):
    return (
        evaluation_root(args, record)
        / f"libero060-all-{job['name']}-001000-libero_10-task7-resume"
        / "eval_info.json"
    )


def evaluation_config_path(args, record):
    return evaluation_root(args, record) / "fixed_object_shift_config.json"


def evaluate(record, jobs, args):
    if os.environ.get("LIBERO_ROBUSTNESS_MANIFEST") or os.environ.get("LIBERO_BENCHMARK"):
        raise ValueError("Disable PLUS/PRO overrides for the standard task7 study")
    if args.episodes != 10:
        raise ValueError("This focused matched test requires exactly 10 episodes")
    audit(record, jobs)
    cfg = shift_config(args)
    path = evaluation_config_path(args, record)
    if not args.dry_run:
        previous.ensure_noise_config(path, cfg)
    commands = []
    for job in jobs:
        descriptor = previous.read_json(job["run"] / "gripper_event_experiment.json")
        env = os.environ.copy()
        for key in (
            "LIBERO_FIXED_NOISE_CONFIG", "LIBERO_TASK7_OBJECT_SHIFT_CONFIG",
            "LIBERO_TASK7_DUALSEED_SHIFT_CONFIG",
            "LIBERO_SELECTED_TASK_ID", "LIBERO_ROBUSTNESS_MANIFEST", "LIBERO_BENCHMARK",
            "BASE_OUTPUT", "EVAL_LOG_ROOT", "BASE_CKPT", "BASE_MODEL_PATH",
            "RUN_PREFIX", "PDET_CASES",
        ):
            env.pop(key, None)
        env.update(
            PRETRAINED_PATH=descriptor["source"]["path"],
            TOKENIZER_PATH=str(args.tokenizer.resolve()),
            CKPT_ROOT=str(args.output_root.resolve()),
            RUN_NAME=job["name"],
            BASE_OUTPUT=str(evaluation_root(args, record).resolve()),
            EVAL_LOG_ROOT=str((args.log_root / "eval" / "task7-timepath-qkv").resolve()),
            LOG_ROOT=str(args.log_root.resolve()),
            TRAIN_ACTION_EXPERT="true", VLM_PROMPT_TOKENS="2",
            CHECKPOINT_STEP="1000", LIBERO_SELECTED_TASK_ID="7",
            EVAL_SEED=str(args.eval_seed), EPISODES_PER_TASK=str(args.episodes),
            GPU_ID=str(args.gpu), DRY_RUN="true" if args.dry_run else "false",
            PROGRESS="true",
        )
        env[
            "LIBERO_TASK7_DUALSEED_SHIFT_CONFIG"
            if "scene_seed" in cfg else "LIBERO_TASK7_OBJECT_SHIFT_CONFIG"
        ] = str(path.resolve())
        cmd = ["bash", str(ROOT / "run_eval_libero_vlm_only.sh"), "base", "trained", "libero_10", "0"]
        subprocess.run(cmd, env={**env, "DRY_RUN": "true"}, cwd=ROOT, check=True)
        commands.append((job, cmd, env))
    if args.dry_run:
        print("Matched timepath dry-run complete; no GPU evaluation launched", flush=True)
        return
    for job, cmd, env in commands:
        print(f"Evaluating {job['case']}: {result_path(job, record, args)}", flush=True)
        subprocess.run(cmd, env=env, cwd=ROOT, check=True)
    summarize(record, jobs, args)


def summarize(record, jobs, args):
    if args.episodes != 10:
        raise ValueError("Summary requires 10 episodes per task")
    cfg = shift_config(args)
    config_path = evaluation_config_path(args, record)
    if not config_path.is_file():
        raise FileNotFoundError("Missing saved fixed-object-shift configuration")
    sha = previous.file_hash(config_path)
    results = {}
    helper_sha = set()
    for job in jobs:
        info_path = result_path(job, record, args)
        manifest = previous.read_json(info_path.parent / "resume_manifest.json")
        shifted = manifest.get(
            "task7_dualseed_shift" if "scene_seed" in cfg else "task7_object_shift", {}
        )
        if shifted.get("config") != cfg or shifted.get("config_sha256") != sha:
            raise ValueError(f"{job['case']} was evaluated with another shift config")
        if not shifted.get("helper_sha256"):
            raise ValueError("Missing physical shift code fingerprint")
        helper_sha.add(shifted["helper_sha256"])
        if manifest.get("fixed_observation_noise"):
            raise ValueError("Image noise was enabled in the position-shift study")
        result = previous.read_eval_result(info_path, 7, args.episodes, only_selected=True)
        results[job["case"]] = {
            **result,
            "checkpoint": str(job["checkpoint"]),
            "eval_info": str(info_path),
        }
    if len(helper_sha) != 1:
        raise ValueError("Different object-shift implementations among model cases")
    contrasts = {}
    if all(k in results for k in CASES):
        success = {k: results[k]["selected_task_rate"] for k in CASES}
        contrasts = {
            "qk_minus_rope_pp": success["time_qk"] - success["rope"],
            "v_minus_rope_pp": success["time_v"] - success["rope"],
            "input_minus_rope_pp": success["time_input"] - success["rope"],
            "input_minus_qk_pp": success["time_input"] - success["time_qk"],
        }
    summary = {
        "protocol": PROTOCOL,
        "scope": "single-demo task7; 1000-step 4-way temporal-path study; fixed shift",
        "training_episode_index": record["selection"]["episode_index"],
        "eval_seed": args.eval_seed,
        "shift_config": cfg,
        "episodes": args.episodes,
        "results": results,
        "contrasts": contrasts,
    }
    root = evaluation_root(args, record)
    root.mkdir(parents=True, exist_ok=True)
    suffix = "all" if len(jobs) == 4 else jobs[0]["case"]
    target = root / f"summary-{suffix}.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (root / f"summary-{suffix}.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["case", "successes", "episodes", "rate"])
        for case, value in results.items():
            writer.writerow([case, value["overall_successes"], value["overall_episodes"],
                             value["selected_task_rate"]])
    print("case              task7 successes    task7 success rate")
    for case, value in results.items():
        print(f"{case:18s}{value['overall_successes']:>4}/10           "
              f"{value['selected_task_rate']:>6.2f}%", flush=True)
    print(f"Results: {target}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "audit", "eval", "summarize"))
    parser.add_argument("case", nargs="?", default="all", choices=(*CASES, "all"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get(
        "OUTPUT_ROOT", os.environ.get("CKPT_ROOT", DEFAULT_OUTPUT)
    )))
    parser.add_argument("--eval-root", type=Path, default=Path(os.environ.get(
        "TIMEPATH_EVAL_ROOT", DEFAULT_EVAL
    )))
    parser.add_argument("--log-root", type=Path, default=Path(os.environ.get(
        "LOG_ROOT", "/data1/wyn/chkpt/2601-lerobot/logs"
    )))
    parser.add_argument("--source", type=Path, default=Path(os.environ.get(
        "PRETRAINED_PATH", os.environ.get("BASE_PRETRAINED_PATH", "/home/wyn/models/lerobot/pi05_libero_base")
    )))
    parser.add_argument("--tokenizer", type=Path, default=Path(os.environ.get(
        "TOKENIZER_PATH", "/home/wyn/models/paligemma-3b-pt-224"
    )))
    parser.add_argument("--prompt", type=Path, default=os.environ.get("PROMPT_INIT_PATH"))
    parser.add_argument("--prompt-dir", type=Path, default=Path(os.environ.get(
        "PROMPT_INIT_DIR", "/data1/wyn/prompt-work"
    )))
    parser.add_argument("--dataset-repo-id", default=os.environ.get("DATASET_REPO_ID", "libero"))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GPU_ID", "0")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MAIN_PROCESS_PORT", "29589")))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--eval-seed", type=int, default=3000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--shift-m", type=float, default=0.12)
    parser.add_argument("--shift-seed", type=int, default=20261103)
    parser.add_argument("--scene-seed", type=int, default=None)
    parser.add_argument("--policy-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "false") == "true")
    args = parser.parse_args()
    try:
        if args.repeat < 1 or args.eval_seed < 0 or args.gpu < 0 or args.episodes < 1:
            raise ValueError("Invalid repeat, evaluation seed, GPU, or episode count")
        record = check_plan(args.directory)
        jobs = build_jobs(record, args.directory, args.output_root, args.case, args.repeat)
        if args.command == "train":
            train(record, jobs, args)
        elif args.command == "audit":
            audit(record, jobs)
        elif args.command == "eval":
            evaluate(record, jobs, args)
        else:
            if args.dry_run:
                raise ValueError("Summaries write files; omit --dry-run")
            audit(record, jobs)
            summarize(record, jobs, args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Task7 timepath experiment failed: {error}\n")


if __name__ == "__main__":
    main()
