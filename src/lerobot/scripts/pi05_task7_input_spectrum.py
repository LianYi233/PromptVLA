"""Spectrally-bounded & content-calibrated PI0.5 action-input time residual.

CONTROL: reuse EXISTING task7 1000-step 'rope' and 'time_input' checkpoints;
there is no reason to retrain these. Train only three NEW models at 1000
updates, using the exact ORIGINAL one-demonstration Uniform PDET order
and flow noise bank:
    lowfreq      0.1..4 Hz, no gain
    rms_gate     0.1..10 Hz, per-token relative-RMS gain
    lowfreq_rms  0.1..4 Hz, per-token relative-RMS gain

The 2x2 factorial among time-on arms uses identical 32x1024 extra trainable
parameters. All arms retain the underlying pretrained RoPE. No extra
episode time, action labels, gripper oracle, or future event time are used.

Evaluation is matched task7 physics-shift rollouts; independent training
replications/multiple tasks will be necessary before publication claims.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot.scripts import pi05_gripper_event_experiment as previous
from lerobot.scripts import pi05_task7_timepath_experiment as original

PROTOCOL = "pi05-task7-input-spectral-rms-v1"
CASES = ("lowfreq", "rms_gate", "lowfreq_rms")
CONTROLS = ("rope", "time_input")
ALL_CASES = (*CONTROLS, *CASES)
PARAMETERS = {
    "lowfreq": (0.1, 4.0, "none"),
    "rms_gate": (0.1, 10.0, "relative_rms"),
    "lowfreq_rms": (0.1, 4.0, "relative_rms"),
}
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = original.DEFAULT_OUTPUT
DEFAULT_EVAL = DEFAULT_OUTPUT / "eval" / "task7-time-input-spectral"


def check_preparation(directory):
    record = original.check_plan(directory)
    if abs(record["fps"] - 10.0) > 1e-6:
        raise ValueError(
            "This fixed 0.1–4.0 Hz frequency ablation is calibrated for the 10-FPS "
            f"task7 episode, but the saved plan has fps={record['fps']}"
        )
    return record


def training_jobs(record, args, case="all"):
    if case not in (*CASES, "all") or type(args.repeat) is not int or args.repeat < 1:
        raise ValueError("Unknown optimized time-input case or repeat")
    selected = CASES if case == "all" else (case,)
    output = []
    for name in selected:
        lo, hi, gain = PARAMETERS[name]
        model_name = (
            f"pi05-task7-timeinput-spectral-r{args.repeat}-{name}"
            f"-task7-ep{record['selection']['episode_index']}"
            "-steps1000-aexpert-base-vlm2-P1-D0-E1-T1"
        )
        run = Path(args.output_root).resolve() / model_name
        output.append({
            "case": name,
            "name": model_name,
            "run": run,
            "plan": Path(args.directory).resolve() / "uniform.npz",
            "checkpoint": run / "checkpoints" / "001000" / "pretrained_model",
            "time_encoding": True,
            "time_injection": "input",
            "low_hz": lo,
            "high_hz": hi,
            "gain_mode": gain,
        })
    return output


def baseline_jobs(record, args):
    return [
        original.build_jobs(record, args.directory, args.output_root, selection=case,
                            repeat=args.repeat)[0]
        for case in CONTROLS
    ]


def training_command(job, record, args, prompt):
    cmd, env = original.training_command(job, record, args, prompt)
    cmd.extend([
        "--policy.action_time_basis=index",
        f"--policy.action_time_min_frequency_hz={job['low_hz']}",
        f"--policy.action_time_max_frequency_hz={job['high_hz']}",
        f"--policy.action_time_gain_mode={job['gain_mode']}",
    ])
    # Never leak experiment-II cadence resampling into this fixed-contiguous
    # action-chunk experiment, even from the user's current shell.
    env.pop("PI05_CADENCE_PLAN", None)
    for stale in (
        "LIBERO_FIXED_NOISE_CONFIG", "LIBERO_TASK7_OBJECT_SHIFT_CONFIG",
        "LIBERO_SELECTED_TASK_ID", "LIBERO_ROBUSTNESS_MANIFEST", "LIBERO_BENCHMARK",
    ):
        env.pop(stale, None)
    return cmd, env


def source_and_prompt(args, record):
    from lerobot.scripts.libero_vlm_only import validate_source
    from lerobot.utils.prompt_init_names import resolve_libero_prompt
    from lerobot.utils.prompt_initialization import read_initial_prompt

    gpu_ids = args.gpus.split(",")
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(s.isdecimal() for s in gpu_ids):
        raise ValueError("Training requires 2 distinct GPU indices for fixed batch32x2 PDET")
    if not args.tokenizer.is_dir() or args.num_workers < 0 or not 1 <= args.port <= 65535:
        raise ValueError("Tokenizer directory, num_workers or port is invalid")
    validate_source(args.source, "base", record["dataset_root"])
    source = previous.source_identity(args.source)
    config = previous.read_json(args.source / "config.json")
    if config.get("chunk_size") != 50:
        raise ValueError("Need 50-step PI0.5 source")
    prompt = args.prompt or resolve_libero_prompt(args.prompt_dir, 1, 2)
    _, meta = read_initial_prompt(prompt, 2)
    if meta["source_training_seed"] != 1 or meta.get("source_prompt_init_path"):
        raise ValueError("Need the original seed1 two-token prompt")
    return source, Path(prompt).resolve(), meta


def descriptor(job, record, source, prompt, prompt_meta, command):
    result = original._descriptor(
        job, record, source, previous.file_hash(prompt), prompt_meta, command
    )
    result.update({
        "protocol": PROTOCOL,
        "frequency_low_hz": job["low_hz"],
        "frequency_high_hz": job["high_hz"],
        "gain_mode": job["gain_mode"],
        "source_training_code": "separate branch; identical PDET sample/noise stream",
    })
    return result


def train(record, jobs, args):
    source, prompt, prompt_meta = source_and_prompt(args, record)
    prompt_digest = previous.file_hash(prompt)
    commands = []
    for job in jobs:
        if job["run"].exists():
            raise FileExistsError(
                f"Existing run directory {job['run']}; use another --repeat "
                "or select only unfinished case"
            )
        cmd, env = training_command(job, record, args, prompt)
        subprocess.run(cmd, env={**env, "DRY_RUN": "true"}, cwd=ROOT, check=True)
        commands.append((job, cmd, env))
    print(f"Preflight: {len(jobs)} NEW cases × 1000 updates; old control weights preserved")
    if args.dry_run:
        return
    for job, cmd, env in commands:
        if previous.source_identity(args.source) != source:
            raise ValueError("Source weights or config changed between runs")
        if previous.file_hash(prompt) != prompt_digest:
            raise ValueError("Imported prompt changed between runs")
        info = descriptor(job, record, source, prompt, prompt_meta, cmd)
        previous.save_record(
            Path(args.directory) / "input_spectral_runs" / f"{job['name']}.json", info
        )
        print(f"\nTraining {job['case']} (1000 updates): {job['run']}", flush=True)
        try:
            subprocess.run(cmd, env=env, cwd=ROOT, check=True)
        finally:
            if job["run"].is_dir():
                previous.save_record(job["run"] / "gripper_event_experiment.json", info)
        audit_new(record, [job])
    audit_new(record, jobs)


def _rows(run, steps=1000):
    rows = []
    for rank in (0, 1):
        logfile = run / "pdet" / f"inputs-rank{rank}.jsonl"
        rank_rows = [json.loads(line) for line in logfile.read_text().splitlines() if line.strip()]
        if len(rank_rows) != steps:
            raise ValueError(f"Expected {steps} PDET records per rank in {run}")
        rows.extend(
            (entry["sample_ids_sha256"], entry["noise_sha256"], entry["time_sha256"])
            for entry in rank_rows
        )
    return rows


def audit_new(record, jobs):
    if not jobs:
        return
    # Previous audit checks full 1000-step DDP provenance, no change of
    # source checkpoint/prompt, and all per-rank actual data/noise/flow draws.
    previous.audit(record, jobs)
    from safetensors import safe_open

    for job in jobs:
        config = previous.read_json(job["checkpoint"] / "config.json")
        spec = {
            "action_time_injection": "input",
            "action_time_basis": "index",
            "action_time_min_frequency_hz": job["low_hz"],
            "action_time_max_frequency_hz": job["high_hz"],
            "action_time_gain_mode": job["gain_mode"],
        }
        for key, value in spec.items():
            if config.get(key) != value:
                raise ValueError(f"{job['case']} checkpoint has unexpected {key}")
        info = previous.read_json(job["run"] / "gripper_event_experiment.json")
        if (
            info["protocol"] != PROTOCOL
            or info["frequency_low_hz"] != job["low_hz"]
            or info["frequency_high_hz"] != job["high_hz"]
            or info["gain_mode"] != job["gain_mode"]
        ):
            raise ValueError("Spectral/RMS run provenance mismatch")
        with safe_open(job["checkpoint"] / "model.safetensors", framework="pt", device="cpu") as f:
            key = "model.action_time_encoder.weight"
            if key not in f.keys():
                raise ValueError(f"{job['case']} is missing temporal residual weights")
            tensor = f.get_tensor(key)
            if tensor.shape != (1024, 32) or not torch.isfinite(tensor).all():
                raise ValueError(f"{job['case']}: time encoder is malformed/nonfinite")
            if torch.count_nonzero(tensor) == 0:
                raise ValueError(f"{job['case']}: time residual did not learn")
    print(f"Optimized training audit passed: {[x['case'] for x in jobs]}", flush=True)


def audit_controls(record, baselines):
    if len(baselines) != 2 or [x["case"] for x in baselines] != list(CONTROLS):
        raise ValueError("Expected old RoPE and Time-Input controls")
    original.audit(record, baselines)
    print("Original controls audited independently; no checkpoint was retrained")


def audit_cross_runs(record, baselines, new_jobs):
    audit_controls(record, baselines)
    audit_new(record, new_jobs)
    ref = _rows(baselines[0]["run"], record["settings"]["steps"])
    for job in [*baselines[1:], *new_jobs]:
        if _rows(job["run"], record["settings"]["steps"]) != ref:
            raise ValueError(f"Not matched to original PDET plan/noise/time: {job['case']}")
        a = previous.read_json(job["run"] / "gripper_event_experiment.json")
        b = previous.read_json(baselines[0]["run"] / "gripper_event_experiment.json")
        for key in ("source", "prompt_file_sha256", "prompt_sha256", "plan_sha256"):
            if a[key] != b[key]:
                raise ValueError(f"{job['case']} used different {key}")
    print("All five controls/optimized runs share the exact saved Uniform and Flow streams")


def _eval_args(args, seed, *, prior=False):
    return SimpleNamespace(
        eval_root=(getattr(args, "control_eval_root", original.DEFAULT_EVAL)
                   if prior else args.eval_root),
        eval_seed=seed, shift_m=args.shift_m, shift_seed=args.shift_seed,
        episodes=10, repeat=args.repeat,
    )


def _result(job, record, args, seed, *, prior=False):
    return original.result_path(job, record, _eval_args(args, seed, prior=prior))


def run_eval_job(job, record, args, seed, gpu, *, prior=False):
    evaluated = _result(job, record, args, seed, prior=prior)
    if evaluated.is_file():
        print(f"Existing {job['case']} seed={seed}: {evaluated}", flush=True)
        return
    logs = Path(args.log_root) / "eval" / "task7-time-input-spectral"
    logs.mkdir(parents=True, exist_ok=True)
    logfile = logs / f"r{args.repeat}-seed{seed}-{job['case']}-gpu{gpu}.log"
    cmd = (
        [
            "python", "-u", "-m", "lerobot.scripts.pi05_task7_timepath_experiment",
            "eval", job["case"], "--directory", str(args.directory),
            "--repeat", str(args.repeat), "--output-root", str(args.output_root),
            "--tokenizer", str(args.tokenizer), "--eval-seed", str(seed),
            "--eval-root", str(args.control_eval_root), "--episodes", "10",
            "--shift-m", str(args.shift_m), "--shift-seed", str(args.shift_seed),
            "--gpu", str(gpu),
        ]
        if prior
        else [
            "python", "-u", "-m", "lerobot.scripts.pi05_task7_input_spectrum",
            "eval_one", job["case"], "--directory", str(args.directory),
            "--repeat", str(args.repeat), "--output-root", str(args.output_root),
            "--tokenizer", str(args.tokenizer), "--eval-seed", str(seed),
            "--eval-root", str(args.eval_root), "--episodes", "10",
            "--shift-m", str(args.shift_m), "--shift-seed", str(args.shift_seed),
            "--gpu", str(gpu),
        ]
    )
    env = os.environ.copy()
    # Keep all physical CUDA indices visible: the existing LIBERO evaluator
    # chooses torch.device('cuda:<GPU_ID>') itself. Masking visibility to a
    # single GPU would incorrectly make --gpu 1 refer to a nonexistent cuda:1.
    env.update(GPU_ID=str(gpu), DRY_RUN="false")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("PI05_CADENCE_PLAN", None)
    print(f"GPU {gpu}: {job['case']} seed {seed}  log={logfile}", flush=True)
    with logfile.open("w") as out:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Eval failed ({job['case']}, seed={seed}), see {logfile}")
    if not evaluated.is_file():
        raise FileNotFoundError(f"Evaluator returned success but did not write {evaluated}")


def eval_one(record, job, args, seed):
    from lerobot.scripts.pi05_gripper_event_experiment import ensure_noise_config

    # The original suite is task7 only, with no image noise. Keep the
    # original object-shift helper and simulator seed logic unchanged.
    spec = original.shift_config(_eval_args(args, seed))
    config_path = original.evaluation_config_path(_eval_args(args, seed), record)
    ensure_noise_config(config_path, spec)
    desc = previous.read_json(job["run"] / "gripper_event_experiment.json")
    if desc["protocol"] != PROTOCOL or desc["case"] != job["case"]:
        raise ValueError("Not a checkpoint from the optimized study")
    env = os.environ.copy()
    for key in (
        "LIBERO_FIXED_NOISE_CONFIG", "LIBERO_TASK7_OBJECT_SHIFT_CONFIG",
        "LIBERO_SELECTED_TASK_ID", "LIBERO_BENCHMARK", "LIBERO_ROBUSTNESS_MANIFEST",
        "BASE_OUTPUT", "EVAL_LOG_ROOT", "BASE_CKPT", "BASE_MODEL_PATH",
        "RUN_PREFIX", "PDET_CASES", "PI05_CADENCE_PLAN",
    ):
        env.pop(key, None)
    env.update(
        PRETRAINED_PATH=desc["source"]["path"],
        TOKENIZER_PATH=str(args.tokenizer.resolve()),
        CKPT_ROOT=str(args.output_root.resolve()),
        RUN_NAME=job["name"],
        BASE_OUTPUT=str(original.evaluation_root(_eval_args(args, seed), record).resolve()),
        EVAL_LOG_ROOT=str((Path(args.log_root) / "eval" / "task7-time-input-spectral" / f"seed{seed}").resolve()),
        LOG_ROOT=str(args.log_root.resolve()),
        TRAIN_ACTION_EXPERT="true", VLM_PROMPT_TOKENS="2",
        CHECKPOINT_STEP="1000", LIBERO_SELECTED_TASK_ID="7",
        LIBERO_TASK7_OBJECT_SHIFT_CONFIG=str(config_path.resolve()),
        EVAL_SEED=str(seed), EPISODES_PER_TASK="10",
        GPU_ID=str(args.gpu), DRY_RUN="true" if args.dry_run else "false",
        PROGRESS="true",
    )
    cmd = ["bash", str(ROOT / "run_eval_libero_vlm_only.sh"),
           "base", "trained", "libero_10", "0"]
    subprocess.run(cmd, cwd=ROOT, env={**env, "DRY_RUN": "true"}, check=True)
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _checked_result(job, record, args, seed, *, prior=False):
    root = _eval_args(args, seed, prior=prior)
    cfg = original.shift_config(root)
    config_path = original.evaluation_config_path(root, record)
    manifest = previous.read_json(_result(job, record, args, seed, prior=prior).parent / "resume_manifest.json")
    shifted = manifest.get("task7_object_shift", {})
    # The underlying evaluator's own resume manifest records all checkpoint
    # file sizes/mtime_ns. Verify that an old eval is still for these exact
    # weights; never silently reuse results for a changed model.
    ckpt = Path(job["checkpoint"])
    current_model_files = [
        [str(p.relative_to(ckpt)), p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(ckpt.rglob("*")) if p.is_file()
    ]
    if not current_model_files or manifest.get("model_files") != current_model_files:
        raise ValueError(f"{job['case']} seed={seed}: model checkpoint differs from cached evaluation")
    if (
        shifted.get("config") != cfg
        or shifted.get("config_sha256") != previous.file_hash(config_path)
        or not shifted.get("helper_sha256")
        or manifest.get("fixed_observation_noise")
    ):
        raise ValueError(f"{job['case']} seed={seed} has a mismatched physics/noise manifest")
    raw = previous.read_eval_result(
        _result(job, record, args, seed, prior=prior), 7, 10, only_selected=True
    )
    return raw, shifted["helper_sha256"]


def summarize_seed(record, args, seed, baselines, new_jobs, *, print_now=True):
    results, helpers = {}, set()
    for job in baselines:
        values, helper = _checked_result(job, record, args, seed, prior=True)
        helpers.add(helper)
        results[job["case"]] = values
    for job in new_jobs:
        values, helper = _checked_result(job, record, args, seed, prior=False)
        helpers.add(helper)
        results[job["case"]] = values
    if len(helpers) != 1 or set(results) != set(ALL_CASES):
        raise ValueError("Models differ in physics implementation or incomplete task7 coverage")
    output = {
        "protocol": PROTOCOL,
        "seed": seed,
        "episodes": 10,
        "shift_m": args.shift_m,
        "shift_seed": args.shift_seed,
        "results": results,
        "contrasts_pp": {
            name + "_vs_original_time": (
                results[name]["selected_task_rate"] - results["time_input"]["selected_task_rate"]
            )
            for name in CASES
        },
        "independent_training_demonstrations": False,
        "note": "One demo adaptation; eval seeds reuse official init-state indices; 12cm was calibrated",
    }
    directory = original.evaluation_root(_eval_args(args, seed), record)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary-five.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if print_now:
        print(f"Task7 12cm shift | seed {seed} | 10 episodes per model")
        for case in ALL_CASES:
            x=results[case]
            print(f"  {case:<15} {x['overall_successes']}/10 "
                  f"{x['selected_task_rate']:.2f}%")
    return output


def print_pooled(summaries):
    print("\n" + "="*78)
    print("TASK7 TEMPORAL INPUT OPTIMIZATION | Original + 3 new | 1000 steps per model")
    print("Seed        RoPE    Input(original)  LowFreq  RMS-Gate  LowFreq+RMS")
    successes = {case:0 for case in ALL_CASES}
    for seed, summary in summaries.items():
        values=[int(summary["results"][k]["overall_successes"]) for k in ALL_CASES]
        for case,v in zip(ALL_CASES,values,strict=True):
            successes[case]+=v
        print(f"{seed:<10}  " + "    ".join(f"{v}/10" for v in values))
    total = len(summaries)*10
    print(f"POOLED      " + "    ".join(
        f"{successes[c]}/{total} ({100*successes[c]/total:.1f}%)" for c in ALL_CASES
    ))
    for case in CASES:
        d=100*(successes[case]-successes["time_input"])/total
        print(f"  {case} − original Time-Input = {d:+.1f} pp")
    print("ONE-DEMO caveat: success seeds reuse official init-state IDs; "
          "no independent training seeds", flush=True)
    return successes, total


def write_pooled(args, summaries, successes, n):
    dest = args.eval_root / "pooled"
    dest.mkdir(parents=True, exist_ok=True)
    cfg = {
        "protocol": PROTOCOL, "seeds": list(summaries),
        "shift_m": args.shift_m, "shift_seed": args.shift_seed,
        "repeat": args.repeat, "total_episodes_per_case": n,
    }
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:10]
    target=dest / f"summary-{digest}.json"
    target.write_text(json.dumps({
        "settings":cfg,"by_seed":summaries,
        "pooled_successes":successes,
        "pooled_success_rates":{k:100*v/n for k,v in successes.items()},
    },indent=2,sort_keys=True) + "\n")
    with (dest / f"summary-{digest}.csv").open("w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["case", "successes", "episodes", "success_rate_pct"])
        for k in ALL_CASES:
            w.writerow([k,successes[k],n,100*successes[k]/n])
    print(f"Saved pooled result: {target}")


def parse_seeds(raw):
    try:
        seeds=[int(x) for x in raw.split(",")]
    except ValueError as err:
        raise ValueError("Expected comma-separated eval seeds") from err
    if not seeds or len(set(seeds)) != len(seeds) or any(x < 0 for x in seeds):
        raise ValueError("Eval seeds must be distinct nonnegative integers")
    return seeds


def _run_pair(records, args, seed):
    if not records:
        return
    gpu_ids=[int(i) for i in args.gpus.split(",")]
    from itertools import zip_longest

    for pairs in zip_longest(*(iter(records),)*2):
        work = [
            (pair[0], int(gpu_ids[i]), pair[1])
            for i, pair in enumerate(pairs) if pair is not None
        ]
        with ThreadPoolExecutor(max_workers=len(work)) as pool:
            futures = [
                pool.submit(run_eval_job, job, args.record, args, seed, gpu, prior=old)
                for job,gpu,old in work
            ]
            for task in futures:
                task.result()


def run(args, record, new_jobs, old_jobs):
    args.record=record
    audit_cross_runs(record, old_jobs, new_jobs)
    work=[(job, True) for job in old_jobs] + [(job,False) for job in new_jobs]
    summaries = {}
    for seed in args.seeds:
        print(f"\nEvaluating 5 matched policies at seed {seed}", flush=True)
        _run_pair(work, args, seed)
        summaries[seed]=summarize_seed(record,args,seed,old_jobs,new_jobs)
    pooled,n=print_pooled(summaries)
    write_pooled(args,summaries,pooled,n)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("train","audit","run","eval_one","report"))
    parser.add_argument("case",nargs="?",default="all",choices=(*CASES,"all"))
    parser.add_argument("--directory",type=Path,default=original.DEFAULT_PLAN)
    parser.add_argument("--output-root",type=Path,default=Path(os.environ.get(
        "OUTPUT_ROOT",os.environ.get("CKPT_ROOT",DEFAULT_OUTPUT)
    )))
    parser.add_argument("--eval-root",type=Path,default=Path(os.environ.get(
        "SPECTRAL_EVAL_ROOT",DEFAULT_EVAL
    )))
    parser.add_argument("--control-eval-root",type=Path,default=Path(os.environ.get(
        "TIMEPATH_EVAL_ROOT",original.DEFAULT_EVAL
    )))
    parser.add_argument("--log-root",type=Path,default=Path(os.environ.get(
        "LOG_ROOT","/data1/wyn/chkpt/2601-lerobot/logs"
    )))
    parser.add_argument("--source",type=Path,default=Path(os.environ.get(
        "PRETRAINED_PATH","/home/wyn/models/lerobot/pi05_libero_base"
    )))
    parser.add_argument("--tokenizer",type=Path,default=Path(os.environ.get(
        "TOKENIZER_PATH","/home/wyn/models/paligemma-3b-pt-224"
    )))
    parser.add_argument("--prompt",type=Path,default=os.environ.get("PROMPT_INIT_PATH"))
    parser.add_argument("--prompt-dir",type=Path,default=Path(os.environ.get(
        "PROMPT_INIT_DIR","/data1/wyn/prompt-work"
    )))
    parser.add_argument("--dataset-repo-id",default=os.environ.get("DATASET_REPO_ID","libero"))
    parser.add_argument("--gpus",default=os.environ.get("GPU_IDS","0,1"))
    parser.add_argument("--gpu",type=int,default=int(os.environ.get("GPU_ID","0")))
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--port",type=int,default=int(os.environ.get("MAIN_PROCESS_PORT","29589")))
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--wandb-mode",choices=("online","offline","disabled"),default="offline")
    parser.add_argument("--seeds",default="6000,7000")
    parser.add_argument("--eval-seed",type=int,default=3000)
    parser.add_argument("--shift-m",type=float,default=0.12)
    parser.add_argument("--shift-seed",type=int,default=20261103)
    parser.add_argument("--episodes",type=int,default=10)
    parser.add_argument("--dry-run",action="store_true",
                        default=os.environ.get("DRY_RUN","false")=="true")
    args=parser.parse_args()
    try:
        args.seeds=parse_seeds(args.seeds)
        record=check_preparation(args.directory)
        selected=training_jobs(record,args,args.case)
        old_jobs=baseline_jobs(record,args)
        gpus = args.gpus.split(",")
        if (
            args.repeat < 1 or args.episodes != 10
            or len(gpus) != 2 or gpus[0] == gpus[1]
            or not all(value.isdecimal() for value in gpus)
            or any(seed < 0 for seed in args.seeds)
            or args.shift_m < 0 or args.shift_m > 0.14
        ):
            raise ValueError("Expected 10 episodes, two distinct GPUs and shift_m in [0, 0.14]")
        if args.command=="train":
            train(record,selected,args)
        elif args.command=="audit":
            if args.case=="all":
                audit_cross_runs(record,old_jobs,selected)
            else:
                audit_new(record,selected)
        elif args.command=="eval_one":
            if len(selected)!=1:
                raise ValueError("eval_one must select one optimized model")
            audit_new(record,selected)
            eval_one(record,selected[0],args,args.eval_seed)
        elif args.dry_run:
            for seed in args.seeds:
                for job in [*old_jobs,*selected]:
                    print(f"DRY_RUN seed={seed} case={job['case']} checkpoint={job['checkpoint']}")
            print("No GPUs launched; run without DRY_RUN for actual rollouts.")
        elif args.command=="run":
            if args.case!="all":
                raise ValueError("Use run all to compare every intervention")
            run(args,record,selected,old_jobs)
        else:
            if args.case!="all":
                raise ValueError("Use report all for the five-way comparison")
            audit_cross_runs(record,old_jobs,selected)
            summaries={
                seed:summarize_seed(record,args,seed,old_jobs,selected,print_now=False)
                for seed in args.seeds
            }
            pooled,n=print_pooled(summaries)
            write_pooled(args,summaries,pooled,n)
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as err:
        parser.exit(1,f"Task7 input spectral experiment failed: {err}\n")


if __name__=="__main__":
    main()
