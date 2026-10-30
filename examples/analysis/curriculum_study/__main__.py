"""Prepare/run/audit/plot controlled curricula and paired LIBERO diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .core import (ORDERS, SCHEMA, digest, json_hash, load_plan, load_scores, lr_value,
                   make_family, read_json, read_labels, save_plan, validate_family, write_json)

ROOT = Path(__file__).resolve().parents[3]


def _csv_ints(value):
    result = [int(x) for x in value.split(",") if x.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Provide a nonempty comma-separated list of distinct integers")
    return result


def package_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def training_args(spec, source, dataset, tokenizer, prompt):
    flags = {
        "policy.path": str(source), "policy.tokenizer_name": str(tokenizer),
        "policy.device": "cuda", "policy.dtype": "bfloat16", "policy.use_amp": "false",
        "policy.compile_model": "false", "policy.gradient_checkpointing": "true",
        "policy.vlm_prompt_only": "false", "policy.num_vlm_prompt_tokens": 2,
        "policy.num_prompt_tokens": 0, "policy.train_action_expert": "true",
        "policy.freeze_vlm_prompt": "false", "policy.training_stage": "flow",
        "policy.next_action_pretrain_steps": 0, "policy.next_action_bridge_steps": 0,
        "policy.cabo_enabled": "false", "policy.train_action_projections": "false", "policy.use_peft": "false",
        "policy.freeze_vision_encoder": "true", "policy.train_expert_only": "true",
        "policy.normalization_mapping": '{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"QUANTILES"}',
        "preserve_pretrained_normalization": "false", "policy.ntk_save_stage_snapshots": "false",
        "policy.push_to_hub": "false", "policy.optimizer_lr": spec["peak_lr"],
        "policy.optimizer_grad_clip_norm": 0.0, "policy.scheduler_warmup_steps": spec["warmup_steps"],
        "policy.scheduler_decay_steps": spec["steps"], "policy.scheduler_decay_lr": spec["end_lr"],
        "dataset.repo_id": "libero", "dataset.root": str(dataset), "dataset.video_backend": "pyav",
        "output_dir": spec["run"], "job_name": Path(spec["run"]).name,
        "steps": spec["steps"], "batch_size": spec["batch_size"], "seed": spec["seed"],
        "env_eval_freq": 0, "eval_steps": 0, "save_checkpoint": "true", "save_freq": spec["steps"],
        "save_steps": json.dumps(spec["checkpoints"], separators=(",", ":")),
        "vlm_prompt_init_path": str(prompt), "pdet.enabled": "true", "pdet.prompt_seed": 1,
        "pdet.data_seed": spec["data_seed"], "pdet.noise_seed": spec["flow_seed"],
        "pdet.time_seed": spec["flow_seed"], "pdet.order_plan": spec["plan"],
        "pdet.flow_binding": "occurrence", "pdet.audit_steps": spec["steps"], "pdet.diagnostics": "false",
        "prompt_rank.freq": 0, "log_freq": 10, "wandb.enable": str(spec["wandb"]).lower(),
        "wandb.project": spec["wandb_project"], "wandb.mode": "online", "wandb.disable_artifact": "true",
    }
    return [f"--{key}={value}" for key, value in flags.items()]


def prepare(args):
    pair, scores_path, output, dataset, tokenizer = [Path(x).resolve() for x in (
        args.source_pair, args.scores, args.output, args.dataset_root, args.tokenizer)]
    if output.exists():
        raise FileExistsError(f"Study directory exists; use a new --output: {output}")
    source, prompt = pair / "libero", pair / "P1.safetensors"
    required = [pair / "pair.json", pair / "D0-W0.npz", source / "config.json",
                source / "model.safetensors", prompt, dataset / "meta/stats.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not tokenizer.is_dir():
        raise FileNotFoundError(tokenizer)
    pair_info = read_json(pair / "pair.json")
    if Path(pair_info["dataset_root"]).resolve() != dataset:
        raise ValueError("Dataset root differs from the source-pair protocol")
    source_cfg = read_json(source / "config.json")
    if source_cfg.get("paligemma_variant") != "gemma_2b" or source_cfg.get("action_expert_variant") != "gemma_300m":
        raise ValueError("Expected the existing PI05 gemma_2b/gemma_300m source")
    if not 0 <= args.seed < 2**32 - 2027:
        raise ValueError("Seed outside supported range")
    if not (0 < args.decay_lr <= args.peak_lr) or not np.isfinite([args.decay_lr, args.peak_lr]).all():
        raise ValueError("Require finite 0 < decay_lr <= peak_lr")
    if args.probe_frequency < 1:
        raise ValueError("Probe frequency must be positive")
    base = load_plan(pair / "D0-W0.npz")
    scores, score_info = load_scores(scores_path)
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, args.steps // 30)
    family, probe = make_family(base, scores, score_info, steps=args.steps, easy_percent=args.easy_percent,
                               holdout_per_task=args.holdout_per_task, probe_per_group=args.probe_per_group,
                               seed=args.seed, warmup_steps=warmup,
                               independent_labels=read_labels(args.labels) if args.labels else None)
    checkpoints = sorted(set(args.checkpoints or [max(1, args.steps//6), max(1, args.steps//2), args.steps]))
    if any(step < 1 or step > args.steps for step in checkpoints):
        raise ValueError("Checkpoint steps must be within the full training budget")
    checkpoints = sorted(set(checkpoints + [args.steps]))
    input_paths = [pair / "pair.json", source / "config.json", prompt, scores_path, dataset / "meta/stats.json"]
    if args.labels:
        input_paths.append(Path(args.labels).resolve())
    base_hashes = {str(path): digest(path) for path in input_paths}
    weights = source / "model.safetensors"
    weight_inventory = {"path": str(weights), "resolved": str(weights.resolve()),
                        "size": weights.stat().st_size, "mtime_ns": weights.stat().st_mtime_ns}
    code = package_hashes()
    specs = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".curriculum-study-", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        write_json(temp / "probe.json", probe, exclusive=True)
        for order, plan in family.items():
            save_plan(temp / f"{order}.npz", plan)
            used_indices = plan["index"][plan["order"]]
            with (temp / f"{order}-schedule.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["step", "mean_score", "min_score", "max_score", "warmup"])
                width = plan["manifest"]["batch_size"] * plan["manifest"]["world_size"]
                for step, start in enumerate(range(0, len(used_indices), width)):
                    v = scores["score"][used_indices[start:start+width]]
                    writer.writerow([step+1, v.mean(), v.min(), v.max(), step < warmup])
        for mode in ("cosine", "constant"):
            for order in ORDERS:
                name = f"{order}-{mode}"
                run = output / "runs" / f"pi05-study-{name}-e{args.easy_percent}-seed{args.seed}-steps{args.steps}"
                spec = {"protocol": SCHEMA, "order": order, "lr_mode": mode, "steps": args.steps,
                        "peak_lr": args.peak_lr, "end_lr": args.decay_lr if mode == "cosine" else args.peak_lr,
                        "warmup_steps": warmup, "seed": args.seed, "flow_seed": args.seed + 1,
                        "data_seed": base["manifest"]["data_seed"],
                        "batch_size": base["manifest"]["batch_size"], "world_size": base["manifest"]["world_size"],
                        "run": str(run), "plan": str(output / f"{order}.npz"), "probe": str(output / "probe.json"),
                        "checkpoints": checkpoints, "probe_frequency": args.probe_frequency,
                        "probe_steps": sorted(set([min(50, args.steps), min(100, args.steps), warmup, *checkpoints])),
                        "source": str(source), "tokenizer": str(tokenizer), "weight_inventory": weight_inventory,
                        "wandb": not args.no_wandb, "wandb_project": args.wandb_project, "code_sha256": code,
                        "input_sha256": {**base_hashes, str(output / f"{order}.npz"): digest(temp / f"{order}.npz"),
                                         str(output / "probe.json"): digest(temp / "probe.json")}}
                spec["training_args"] = training_args(spec, source, dataset, tokenizer, prompt)
                spec["spec_id"] = json_hash(spec)
                write_json(temp / "specs" / f"{name}.json", spec, exclusive=True)
                specs[name] = str(output / "specs" / f"{name}.json")
        study = {"protocol": SCHEMA, "specs": specs, "probe_id": probe["probe_id"],
                 "steps": args.steps, "seed": args.seed, "easy_percent": args.easy_percent,
                 "heldout_episodes": probe["heldout_episodes"], "probe_samples": len(probe["samples"]),
                 "common_warmup_steps": warmup, "group_definition": probe["group_definition"],
                 "scope": probe["scope"], "code_sha256": code}
        write_json(temp / "study.json", study, exclusive=True)
        if output.exists():
            raise FileExistsError(output)
        temp.rename(output)
    return study


def get_specs(directory):
    directory = Path(directory).resolve()
    study = read_json(directory / "study.json")
    if study.get("protocol") != SCHEMA:
        raise ValueError("Unknown study")
    return study, {name: Path(path) for name, path in study["specs"].items()}


def launch(args):
    from .train import check_spec
    study, paths = get_specs(args.study)
    selected = list(paths) if args.all else [f"{args.order}-{args.lr}"]
    gpus = args.gpus.split(",")
    if not gpus or any(not x.isdigit() for x in gpus) or len(set(map(int, gpus))) != len(gpus):
        raise ValueError("GPU IDs must be distinct integers")
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid port")
    commands = []
    for name in selected:
        spec, _ = check_spec(paths[name])
        if len(gpus) != spec["world_size"]:
            raise ValueError(f"Plan requires exactly {spec['world_size']} GPUs, not {len(gpus)}")
        if Path(spec["run"]).exists():
            raise FileExistsError(f"Output exists; not resumed/overwritten: {spec['run']}")
        command = [sys.executable, "-m", "accelerate.commands.launch", "--num_machines=1",
                   f"--num_processes={len(gpus)}", "--mixed_precision=bf16",
                   f"--main_process_port={args.port}", "--dynamo_backend=no"]
        if len(gpus) > 1:
            command.append("--multi_gpu")
        command += [str(Path(__file__).with_name("train.py")), "--spec", str(paths[name])]
        commands.append((name, command))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for name, command in commands:
        print(name + ": " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    return {"launched": [] if args.dry_run else selected, "dry_run": args.dry_run}


def audit(args):
    """Full actual PDET traces, learning rates, heldout IDs, processors, probe inputs."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from lerobot.scripts.pdet_order_binding import inspect_run
    from lerobot.scripts.libero_vlm_only import validate_checkpoint, assert_same_processors
    from .analysis import read_new_run
    from .train import check_spec
    _, paths = get_specs(args.study)
    first, first_probe, first_checkpoint, bank = None, None, None, None
    reports = []
    family = {order: load_plan(Path(args.study) / f"{order}.npz") for order in ORDERS}
    validate_family(family, read_json(Path(args.study) / "probe.json"))
    for name, path in paths.items():
        spec, _ = check_spec(path)
        run = Path(spec["run"])
        checkpoint = run / f"checkpoints/{spec['steps']:06d}/pretrained_model"
        validate_checkpoint(checkpoint, 2, source=Path(spec["source"]), kind="base", train_action_expert=True)
        def check_name(_run, factors, tokens, within_seed):
            expected = (1, spec["data_seed"], spec["flow_seed"], spec["flow_seed"], "occurrence", 2, 0)
            actual = (factors["prompt_seed"], factors["data_seed"], factors["noise_seed"],
                      factors["time_seed"], factors["flow_binding"], tokens, within_seed)
            if actual != expected:
                raise ValueError("Unexpected PDET factors")
            return spec["seed"]
        invariant, report = inspect_run(run, name_checker=check_name)
        if bank is None:
            bank = report["bank_sha256"]
        if report["bank_sha256"] != bank:
            raise ValueError("Realized noise/time bank differs")
        # Only the explicitly crossed LR endpoint may differ, not other optimizer knobs.
        recipe = invariant["recipe"]
        if recipe["policy"].pop("scheduler_decay_lr") != spec["end_lr"]:
            raise ValueError("Policy LR endpoint differs from spec")
        if "scheduler" in recipe and recipe["scheduler"] is not None:
            if recipe["scheduler"].pop("decay_lr") != spec["end_lr"]:
                raise ValueError("Scheduler endpoint differs from spec")
        if first is None:
            first = invariant
        elif invariant != first:
            raise ValueError("Unexpected non-LR configuration/source/pool difference across the factorial")
        manifest, rows = read_new_run(run)
        if first_probe is None:
            first_probe, first_checkpoint = manifest, checkpoint
        elif manifest != first_probe:
            raise ValueError("Probe protocol/preprocessed tensors/noise differs")
        assert_same_processors(first_checkpoint, checkpoint)
        if rows[-1]["step"] != spec["steps"]:
            raise ValueError("Final probe missing")
        recorded = [json.loads(line) for line in (run / "curriculum_probe/lr.jsonl").read_text().splitlines()]
        if [r["step"] for r in recorded] != list(range(1, spec["steps"] + 1)):
            raise ValueError("Incomplete/duplicate per-update LR trace")
        for row in recorded:
            expected_lr = lr_value(row["step"] - 1, steps=spec["steps"], warmup=spec["warmup_steps"],
                                   peak=spec["peak_lr"], end=spec["end_lr"])
            if row["optimizer_step_skipped"] or not np.allclose(row["used_lr"], expected_lr, rtol=1e-7, atol=1e-14):
                raise ValueError("Actual LR/update budget is not the declared protocol")
        actual_saved = {int(p.name) for p in (run / "checkpoints").iterdir() if p.name.isdigit()}
        if actual_saved != set(spec["checkpoints"]):
            raise ValueError(f"Wrong checkpoint milestones for {name}: {actual_saved}")
        reports.append({"name": name, **report})
    result = {"audit": "passed", "runs": reports,
              "scope": "All six realized input/noise/LR traces; same pool and probe episodes, "
                       "same source/recipe/code except declared LR endpoint. Not bitwise GPU determinism."}
    write_json(Path(args.study) / "audit.json", result)
    return result


def evaluate(args):
    from .train import check_spec
    _, paths = get_specs(args.study)
    spec, _ = check_spec(paths[f"{args.order}-{args.lr}"])
    step = args.step or spec["steps"]
    if step not in spec["checkpoints"]:
        raise ValueError(f"Step {step} was not a requested full checkpoint")
    run = Path(spec["run"])
    checkpoint = run / f"checkpoints/{step:06d}/pretrained_model"
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from lerobot.scripts.libero_vlm_only import validate_checkpoint
    validate_checkpoint(checkpoint, 2, source=Path(spec["source"]), kind="base", train_action_expert=True)
    remove = {"BASE_MODEL_PATH", "BASE_CKPT", "VLM_PROMPT_TOKENS", "PRETRAINED_PATH", "LEROBOT_EVAL_BASE",
              "LIBERO_ROBUSTNESS_MANIFEST", "LIBERO_BENCHMARK", "LEROBOT_EVAL_ATTENTION"}
    env = {k: v for k, v in os.environ.items() if k not in remove}
    env.update(RUN_NAME=run.name, CKPT_ROOT=str(run.parent), CHECKPOINT_STEP=str(step),
               GPU_ID=str(args.gpu), EVAL_SEED=str(args.eval_seed), EPISODES_PER_TASK=str(args.episodes),
               TOKENIZER_PATH=spec["tokenizer"], LEROBOT_EVAL_FINETUNED_PROMPT="1",
               BASE_OUTPUT=str(Path(args.study).resolve() / "eval" / f"seed{args.eval_seed}-ep{args.episodes}"),
               DRY_RUN=str(args.dry_run).lower())
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = ["bash", str(ROOT / "run_eval_libero-full.sh"), "vlm_only", args.suite, str(spec["seed"])]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return {"evaluated": not args.dry_run, "checkpoint": str(checkpoint)}


def named_runs(values):
    result = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or name in result:
            raise ValueError("Use distinct LABEL=/absolute/run/path entries")
        result[name] = Path(path).resolve()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare", help="Reserve episodes and create all 3 x 2 immutable protocols")
    prep.add_argument("--source-pair", type=Path, default=Path("/data1/wyn/prompt-work/source-pair-r1"))
    prep.add_argument("--scores", type=Path, default=Path("/data1/wyn/prompt-work/frameskip-r1/scores.npz"))
    prep.add_argument("--dataset-root", type=Path, default=Path("/home/wyn/datasets/libero"))
    prep.add_argument("--tokenizer", type=Path, default=Path("/home/wyn/models/paligemma-3b-pt-224"))
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--steps", type=int, default=6000)
    prep.add_argument("--easy-percent", type=int, default=30)
    prep.add_argument("--holdout-per-task", type=int, default=2)
    prep.add_argument("--probe-per-group", type=int, default=2)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--warmup-steps", type=int)
    prep.add_argument("--peak-lr", type=float, default=2.5e-5)
    prep.add_argument("--decay-lr", type=float, default=1e-5)
    prep.add_argument("--probe-frequency", type=int, default=250)
    prep.add_argument("--checkpoints", type=_csv_ints)
    prep.add_argument("--labels", type=Path, help="Independent index,group CSV (routine/critical); never inferred")
    prep.add_argument("--no-wandb", action="store_true")
    prep.add_argument("--wandb-project", default="curriculum-study")
    for command in ("run", "eval"):
        sub = subs.add_parser(command)
        sub.add_argument("--study", type=Path, required=True)
        sub.add_argument("--order", choices=ORDERS, default="H2E")
        sub.add_argument("--lr", choices=("cosine", "constant"), default="cosine")
        sub.add_argument("--dry-run", action="store_true")
        if command == "run":
            sub.add_argument("--gpus", default="0,1")
            sub.add_argument("--port", type=int, default=29620)
            sub.add_argument("--all", action="store_true", help="Run six jobs sequentially on the selected GPU pair")
        else:
            sub.add_argument("--step", type=int)
            sub.add_argument("--gpu", type=int, default=0)
            sub.add_argument("--episodes", type=int, default=10)
            sub.add_argument("--eval-seed", type=int, default=1000)
            sub.add_argument("--suite", choices=("all", "no10", "libero_spatial", "libero_object", "libero_goal", "libero_10"), default="all")
    subs.add_parser("audit").add_argument("--study", type=Path, required=True)
    for command in ("legacy", "plot"):
        sub = subs.add_parser(command)
        if command == "legacy":
            sources = sub.add_mutually_exclusive_group(required=True)
            sources.add_argument("--runs", nargs="+", help="LABEL=/path/to/run ...")
            sources.add_argument("--jobs", type=Path, help="Existing A-H suite jobs.json (read only)")
            sub.add_argument("--cells", nargs="+", default=["A", "B", "C"])
        else:
            sub.add_argument("--runs", nargs="+", required=True, help="LABEL=/path/to/run ...")
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--weighting", choices=("task", "micro"), default="task")
        if command == "legacy":
            sub.add_argument("--scores", type=Path, required=True)
            sub.add_argument("--baseline-plan", type=Path, required=True)
            sub.add_argument("--allow-unverified", action="store_true")
    risk = subs.add_parser("risk-plot")
    risk.add_argument("--results", type=Path, required=True)
    risk.add_argument("--features", type=Path, required=True)
    risk.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "run":
        result = launch(args)
    elif args.command == "audit":
        result = audit(args)
    elif args.command == "eval":
        result = evaluate(args)
    else:
        from . import analysis
        if args.command == "legacy":
            if args.jobs:
                jobs = {job["cell"]: job for job in read_json(args.jobs)}
                if len(args.cells) != len(set(args.cells)) or any(cell not in jobs for cell in args.cells):
                    raise ValueError("Select distinct cells present in jobs.json")
                runs = {cell: Path(jobs[cell]["run"]).resolve() for cell in args.cells}
            else:
                runs = named_runs(args.runs)
            trajectories, meta = analysis.legacy_runs(runs, args.scores, args.baseline_plan,
                        weighting=args.weighting, allow_unverified=args.allow_unverified)
            analysis.export_trajectories(trajectories, args.output, meta)
        elif args.command == "plot":
            analysis.plot_new(named_runs(args.runs), args.output, weighting=args.weighting)
        else:
            analysis.risk_plots(read_json(args.results), args.features, args.output)
        result = {"output": str(args.output)}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
