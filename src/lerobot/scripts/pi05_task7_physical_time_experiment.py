"""Experiment II: token index vs physical time on matched variable-cadence task7 labels.

Four 1000-update, 2-GPU DDP cases with identical one-demo (start,stride)
examples: original RoPE; RoPE+index φ(j/fps); RoPE+global Δt φ(Δt);
RoPE+physical φ(jΔt). All use zero-initialized 32x1024 projections except
the original RoPE arm, which has no extra projection.

Testing is OFFLINE against the same recorded future instantaneous actions,
never a robot rollout at altered control frequencies. No closed-loop success
rate is claimed by these variable-interval test results.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

from lerobot.scripts import pi05_task7_cadence as cadence
from lerobot.scripts import pi05_gripper_event_experiment as legacy

PROTOCOL = "pi05-task7-physical-time-experiment-v1"
CASES = ("rope", "index", "global_dt", "physical")
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path("/data1/wyn/chkpt/2601-lerobot/PromptVLA")
DEFAULT_EVAL = DEFAULT_OUTPUT / "eval" / "task7-physical-time"


def jobs(prepared, root, selection="all", repeat=1):
    if selection not in (*CASES, "all") or type(repeat) is not int or repeat < 1:
        raise ValueError("Unknown case or repeat identifier")
    root = Path(root).resolve()
    return [
        {
            "case": case,
            "name": f"pi05-task7-physicaltime-r{repeat}-{case}-ep{prepared['episode_index']}"
                    "-steps1000-aexpert-base-vlm2-P1-D0-E1-T1",
            "time_encoding": case != "rope",
            "time_injection": "input",
            "time_basis": "index" if case == "rope" else case,
            "plan": (Path(prepared["prepared_root"]) / "uniform.npz").resolve(),
        }
        for case in CASES if selection == "all" or case == selection
    ]


def enrich_jobs(prepared, root, selection="all", repeat=1):
    out = []
    for job in jobs(prepared, root, selection, repeat):
        run = Path(root).resolve() / job["name"]
        out.append({
            **job, "run": run,
            "checkpoint": run / "checkpoints" / "001000" / "pretrained_model",
        })
    return out


def recipe_record(prepared):
    return {
        "dataset_root": prepared["dataset_root"],
        "fps": prepared["fps"],
        "settings": {
            "steps": prepared["steps"],
            "batch_size": prepared["batch_size"],
            "world_size": prepared["world_size"],
            "data_seed": prepared["data_seed"],
        },
        "selection": {
            "task_id": prepared["task_id"],
            "episode_index": prepared["episode_index"],
            "chunk_size": prepared["horizon"],
        },
    }


def training_command(job, prepared, args, prompt):
    command, env = legacy.training_command(job, recipe_record(prepared), args, prompt)
    command.extend([
        "--policy.action_time_injection=input",
        f"--policy.action_time_basis={job['time_basis']}",
    ])
    # Unlike the earlier task7 study, labels come from true future timestamps
    # selected by occurrence-specific strides, loaded in the native trainer.
    env["PI05_CADENCE_PLAN"] = str(Path(prepared["prepared_root"]).resolve())
    return command, env


def train(prepared, job_list, args):
    from lerobot.scripts.libero_vlm_only import validate_source
    from lerobot.utils.prompt_init_names import resolve_libero_prompt
    from lerobot.utils.prompt_initialization import read_initial_prompt

    gpu_ids = args.gpus.split(",")
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(x.isdecimal() for x in gpu_ids):
        raise ValueError("Expected two distinct DDP GPU indices")
    validate_source(args.source, "base", prepared["dataset_root"])
    if not args.tokenizer.is_dir() or args.num_workers < 0:
        raise ValueError("Tokenizer or data-loader worker settings invalid")
    source = legacy.source_identity(args.source)
    if legacy.read_json(args.source / "config.json").get("chunk_size") != 50:
        raise ValueError("This experiment requires the pretrained 50-action PI0.5")
    prompt = args.prompt or resolve_libero_prompt(args.prompt_dir, 1, 2)
    _, info = read_initial_prompt(prompt, 2)
    if info["source_training_seed"] != 1 or info.get("source_prompt_init_path"):
        raise ValueError("Expected unswapped seed-1 two-token initial prompt")
    prompt_digest = legacy.file_hash(prompt)
    commands = []
    for job in job_list:
        if job["run"].exists():
            raise FileExistsError(f"Already exists: {job['run']}")
        command, env = training_command(job, prepared, args, prompt)
        subprocess.run(command, env={**env, "DRY_RUN": "true"}, check=True, cwd=ROOT)
        commands.append((job, command, env))
    if args.dry_run:
        print(f"Dry-run validated {len(commands)} variable-cadence training cases", flush=True)
        return
    for job, command, env in commands:
        if legacy.source_identity(args.source) != source or legacy.file_hash(prompt) != prompt_digest:
            raise ValueError("Starting checkpoint or prompt initialization changed")
        descriptor = {
            "protocol": PROTOCOL,
            "case": job["case"],
            "basis": job["time_basis"],
            "time_encoding": job["time_encoding"],
            "name": job["name"],
            "source": source,
            "prompt_file_sha256": prompt_digest,
            "prompt_sha256": info["prompt_sha256"],
            "cadence_manifest_sha256": legacy.file_hash(
                Path(prepared["prepared_root"]) / "manifest.json"
            ),
            "prepared": prepared,
            "command": command,
        }
        legacy.save_record(Path(prepared["prepared_root"]) / "runs" / f"{job['name']}.json", descriptor)
        print(f"Starting {job['case']} at 1000 steps; cadence=1x/2x", flush=True)
        try:
            subprocess.run(command, env=env, check=True, cwd=ROOT)
        finally:
            if job["run"].is_dir():
                legacy.save_record(job["run"] / "cadence_experiment.json", descriptor)
        audit(prepared, [job])
    if len(job_list) > 1:
        audit(prepared, job_list)


def audit(prepared, job_list):
    from safetensors import safe_open

    from lerobot.utils.pdet_order import load_plan

    plan = load_plan(Path(prepared["prepared_root"]) / "uniform.npz")
    stream_reference = None
    pdet_protocol_reference = None
    descriptors = []
    for job in job_list:
        config = legacy.read_json(job["checkpoint"] / "config.json")
        train_config = legacy.read_json(job["checkpoint"] / "train_config.json")
        expected = {
            "action_time_encoding": job["time_encoding"],
            "action_time_basis": job["time_basis"],
            "action_time_injection": "input",
            "train_action_expert": True,
            "num_vlm_prompt_tokens": 2,
            "num_prompt_tokens": 0,
            "action_time_encoding_dim": 32,
            "action_time_encoding_fps": prepared["fps"],
            "chunk_size": 50,
        }
        for key, value in expected.items():
            if config.get(key, "index" if key == "action_time_basis" else None) != value:
                raise ValueError(f"{job['case']} checkpoint mismatches {key}")
        if (train_config["steps"], train_config["batch_size"], train_config["seed"]) != (1000,32,0):
            raise ValueError("Checkpoint has unexpected training budget or seed")
        descriptor = legacy.read_json(job["run"] / "cadence_experiment.json")
        if descriptor["protocol"] != PROTOCOL or descriptor["case"] != job["case"]:
            raise ValueError("Training provenance case mismatch")
        if descriptor["cadence_manifest_sha256"] != legacy.file_hash(
            Path(prepared["prepared_root"]) / "manifest.json"
        ):
            raise ValueError("Cadence provenance changed after training")
        descriptors.append(descriptor)
        with safe_open(job["checkpoint"] / "model.safetensors", framework="pt", device="cpu") as f:
            if job["time_encoding"]:
                shape = f.get_slice("model.action_time_encoder.weight").get_shape()
                if tuple(shape) != (1024,32):
                    raise ValueError("Time parameter not 1024x32")
            elif "model.action_time_encoder.weight" in f.keys():
                raise ValueError("RoPE control unexpectedly has time residual parameters")
        rows = []
        ranks = []
        for rank in (0,1):
            manifest = legacy.read_json(job["run"] / "pdet" / f"manifest-rank{rank}.json")
            if (
                manifest["total_steps"] != 1000
                or manifest["world_size"] != 2
                or manifest["per_device_batch_size"] != 32
                or manifest["order_plan"] != plan["manifest"]
            ):
                raise ValueError("Invalid PDET plan metadata")
            ranks.append(manifest)
            lines = (job["run"] / "pdet" / f"inputs-rank{rank}.jsonl").read_text().splitlines()
            if len(lines) != 1000:
                raise ValueError("Incomplete PDET inputs")
            for step, text_line in enumerate(lines):
                row = json.loads(text_line)
                first = (step * 2 + rank) * 32
                occurrences = plan["order"][first:first+32].tolist()
                if (
                    row["step"] != step or row["rank"] != rank
                    or row["occurrence_ids"] != occurrences
                    or row["sample_ids"] != plan["index"][occurrences].tolist()
                ):
                    raise ValueError("PDET deviated from immutable cadence sample occurrences")
                rows.append((
                    row["sample_ids_sha256"], row["noise_sha256"], row["time_sha256"],
                ))
        signatures = [
            (rank["source_tree_sha256"], rank["versions"], rank["cuda"])
            for rank in ranks
        ]
        if stream_reference is None:
            stream_reference = rows
            pdet_protocol_reference = signatures
        elif rows != stream_reference or signatures != pdet_protocol_reference:
            raise ValueError("Four cases saw different observations/flow draws or source environment")
    if descriptors:
        first = descriptors[0]
        for descriptor in descriptors[1:]:
            if descriptor["source"] != first["source"] or descriptor["prompt_file_sha256"] != first["prompt_file_sha256"]:
                raise ValueError("Different source weights or prompt initialization")
    print(f"Variable-cadence audit passed: {[j['case'] for j in job_list]}", flush=True)


def result_path(job, args):
    return args.eval_root / f"seed{args.eval_seed}-t{args.flow_time:.2f}" / f"{job['name']}.json"


def evaluation_manifest(prepared, args):
    return {
        "protocol": "pi05-cadence-offline-eval-v1",
        "cadence_preparation_sha256": legacy.file_hash(
            Path(prepared["prepared_root"]) / "manifest.json"
        ),
        "flow_time": args.flow_time,
        "seed": args.eval_seed,
        "max_eval_starts": args.max_eval_starts,
        "dataset_repo_id": args.dataset_repo_id,
        "episode": prepared["episode_index"],
        "target_semantics": prepared["target_semantics"],
        "metric": "normalised flow-velocity MSE (same x_t and noise per target pair)",
    }


def eval_one(prepared, job_list, args):
    if len(job_list) != 1:
        raise ValueError("Eval command tests one checkpoint at a time; use paired GPU shell for all")
    audit(prepared, job_list)
    job = job_list[0]
    from lerobot.scripts.pi05_task7_cadence_eval import evaluate_checkpoint
    output = result_path(job, args)
    if args.dry_run:
        print(json.dumps({"case": job["case"], "checkpoint": str(job["checkpoint"]),
                          "result": str(output), "manifest": evaluation_manifest(prepared, args)}))
        return
    evaluate_checkpoint(
        prepared, job["checkpoint"], output, args,
        expected_case=job["case"], eval_manifest=evaluation_manifest(prepared, args)
    )


def summarize(prepared, job_list, args):
    manifest = evaluation_manifest(prepared, args)
    from lerobot.scripts.pi05_task7_cadence_eval import _check_saved_checkpoint

    results = {}
    for job in job_list:
        path = result_path(job, args)
        row = legacy.read_json(path)
        if row.get("eval_manifest") != manifest or row.get("case") != job["case"]:
            raise ValueError("Mixed evaluation protocol or checkpoint identity")
        if row["checkpoint"] != str(job["checkpoint"]):
            raise ValueError("Model checkpoint differs from evaluated model")
        if row["checkpoint_fingerprint"] != _check_saved_checkpoint(
            job["checkpoint"], job["case"]
        ):
            raise ValueError("Model weights/config changed after offline evaluation")
        results[job["case"]] = row["per_stride"]
    root = args.eval_root / f"seed{args.eval_seed}-t{args.flow_time:.2f}"
    root.mkdir(parents=True, exist_ok=True)
    suffix = "all" if len(job_list)==4 else job_list[0]["case"]
    report = root / f"summary-{suffix}.json"
    output = {"protocol": PROTOCOL, "eval_manifest": manifest, "results": results}
    report.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    with (root / f"summary-{suffix}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "stride", "n", "normalised_flow_velocity_mse"])
        for case, sub in results.items():
            for stride, row in sub.items():
                writer.writerow([case, stride, row["n"], row["mean_mse"]])
    for case, sub in results.items():
        print(case, {stride: round(row["mean_mse"],6) for stride,row in sub.items()})
    print("Saved:", report, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "audit", "eval", "summarize"))
    parser.add_argument("case", nargs="?", default="all", choices=(*CASES, "all"))
    parser.add_argument("--directory", type=Path, default=cadence.DEFAULT_DIR)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", DEFAULT_OUTPUT)))
    parser.add_argument("--eval-root", type=Path, default=Path(os.environ.get("CADENCE_EVAL_ROOT", DEFAULT_EVAL)))
    parser.add_argument("--log-root", type=Path, default=Path(os.environ.get(
        "LOG_ROOT", "/data1/wyn/chkpt/2601-lerobot/logs"
    )))
    parser.add_argument("--source", type=Path, default=Path(os.environ.get(
        "PRETRAINED_PATH", "/home/wyn/models/lerobot/pi05_libero_base",
    )))
    parser.add_argument("--tokenizer", type=Path, default=Path(os.environ.get(
        "TOKENIZER_PATH", "/home/wyn/models/paligemma-3b-pt-224",
    )))
    parser.add_argument("--prompt", type=Path, default=os.environ.get("PROMPT_INIT_PATH"))
    parser.add_argument("--prompt-dir", type=Path, default=Path(os.environ.get(
        "PROMPT_INIT_DIR", "/data1/wyn/prompt-work"
    )))
    parser.add_argument("--dataset-repo-id", default=os.environ.get("DATASET_REPO_ID", "libero"))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GPU_ID","0")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MAIN_PROCESS_PORT","29589")))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-mode", choices=("offline","online","disabled"), default="offline")
    parser.add_argument("--eval-seed", type=int, default=3000)
    parser.add_argument("--flow-time", type=float, default=0.5)
    parser.add_argument("--max-eval-starts", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "false")=="true")
    args = parser.parse_args()
    try:
        if not 0 < args.flow_time < 1 or args.eval_seed < 0 or args.max_eval_starts < 1:
            raise ValueError("Invalid fixed Flow Matching time or evaluation seed")
        prepared = cadence.check(args.directory)
        prepared["prepared_root"] = str(args.directory.resolve())
        selected = enrich_jobs(prepared, args.output_root, args.case, args.repeat)
        if args.command == "train":
            train(prepared, selected, args)
        elif args.command == "audit":
            audit(prepared, selected)
        elif args.command == "eval":
            eval_one(prepared, selected, args)
        else:
            if args.dry_run:
                raise ValueError("No summary in dry-run mode")
            summarize(prepared, selected, args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as err:
        parser.exit(1, f"Physical-time experiment failed: {err}\n")


if __name__ == "__main__":
    main()
