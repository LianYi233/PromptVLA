"""Paired native task7 rollouts with one Time-Input checkpoint and three interventions.

Correct, Zero and Reversed share the original scene/policy RNG rules, initial
states, processors and execution cadence. Each arm closes the loop on its own
new observations. Old uninstrumented success caches are never imported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


PROTOCOL = "pi05-task7-closed-loop-v1"
COMPARISON_PROTOCOL = "pi05-task7-closed-loop-comparison-v1"
INTERVENTION_PROTOCOL = "pi05-task7-time-intervention-v1"
MODES = ("correct", "zero", "reversed")
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path("/data1/wyn/chkpt/2601-lerobot/PromptVLA")
DEFAULT_PLAN = Path("/data1/wyn/prompt-work/gripper-event-task7-1200/base")
TIMEPATH_MODULE = "lerobot.scripts.pi05_task7_timepath_experiment"


def integer_list(value, label):
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{label} must contain distinct nonnegative integers") from exc
    if not result or len(set(result)) != len(result) or any(not 0 <= x < 2**31 for x in result):
        raise ValueError(f"{label} must contain distinct nonnegative integers < 2^31")
    return result


def comma(values):
    return ",".join(str(value) for value in values)


def evaluation_options(args, mode, scene, policy):
    if mode not in MODES:
        raise ValueError("Unknown time intervention")
    return SimpleNamespace(
        eval_root=args.output / "evaluations" / mode,
        control_eval_root=args.output / "evaluations" / mode,
        optimized_eval_root=args.output / "evaluations" / mode,
        eval_seed=args.init_seed, init_seed=args.init_seed,
        shift_m=args.shift_m, shift_seed=20261103,
        scene_seed=scene, policy_seed=policy, repeat=args.repeat, episodes=10,
    )


def intervention_config(args, job, mode, scene, policy, result_path):
    return {
        "protocol": INTERVENTION_PROTOCOL, "intervention": mode,
        "executed_steps": 10, "horizon": 50, "episodes": 10,
        "checkpoint": str(Path(job["checkpoint"]).resolve()),
        "scene_seed": int(scene), "policy_seed": int(policy), "init_seed": args.init_seed,
        "trace_directory": str((result_path.parent / "trajectories").resolve()),
    }


def cell_command(args, mode, scene, policy, gpu):
    return [
        sys.executable, "-u", "-m", TIMEPATH_MODULE, "eval", "time_input",
        "--directory", str(args.directory), "--output-root", str(args.output_root),
        "--repeat", str(args.repeat),
        "--eval-root", str(args.output / "evaluations" / mode),
        "--tokenizer", str(args.tokenizer), "--eval-seed", str(args.init_seed),
        "--episodes", "10", "--shift-m", str(args.shift_m),
        "--scene-seed", str(scene), "--policy-seed", str(policy), "--gpu", str(gpu),
        "--log-root", str(args.output / "logs" / mode / f"scene{scene}-policy{policy}"),
    ]


def cell_environment(config_path, gpu):
    from lerobot.scripts.pi05_task7_train_eval import process_environment

    env = process_environment()
    for key in (
        "LIBERO_TASK7_TIME_INTERVENTION_CONFIG", "LEROBOT_EVAL_ATTENTION",
        "LIBERO_PROGRESS_FD", "BASE_CKPT", "BASE_MODEL_PATH", "RUN_NAME", "BASE_OUTPUT",
        "LIBERO_SELECTED_TASK_ID", "LEROBOT_EVAL_BASE", "LEROBOT_EVAL_FINETUNED_PROMPT",
    ):
        env.pop(key, None)
    env["LIBERO_TASK7_TIME_INTERVENTION_CONFIG"] = str(config_path)
    env["GPU_ID"] = str(gpu)
    # The existing shell launchers invoke `python`; use this interpreter's environment.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def freeze_json(path, value):
    from lerobot.scripts.pi05_task7_train_eval import atomic_json

    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen closed-loop settings changed: {path}; use a separate --output")
    else:
        atomic_json(path, value)


def recover_empty_failed_attempt(output, settings, specs):
    """Archive a failed, evidence-free attempt after a source-only fix.

    Must run under the existing output lock. Completed or partially recorded
    experiments never cross implementation fingerprints, even if the enclosing
    state did not yet register a completed cell. Unknown evaluation files also
    prevent recovery because their lack of episode evidence cannot be established.
    """
    import fcntl
    import tempfile
    from contextlib import ExitStack

    from lerobot.scripts.pi05_task7_train_eval import atomic_json, utc_now

    protocol_path = output / "closed_loop_protocol.json"
    if not protocol_path.exists():
        return None

    def reject(reason):
        raise ValueError(
            f"Frozen closed-loop settings changed: {protocol_path}; {reason}; use a separate --output"
        )

    try:
        previous = json.loads(protocol_path.read_text())
    except (OSError, ValueError) as exc:
        reject(f"cannot verify the prior frozen protocol ({exc})")
    if previous == settings:
        return None
    if (not isinstance(previous, dict)
            or {key: value for key, value in previous.items() if key != "source_sha256"}
            != {key: value for key, value in settings.items() if key != "source_sha256"}
            or not isinstance(previous.get("source_sha256"), dict)
            or not previous["source_sha256"]):
        reject("automatic recovery requires identical non-code settings")
    state_path = output / "closed_loop_state.json"
    try:
        previous_state = json.loads(state_path.read_text())
    except (OSError, ValueError) as exc:
        reject(f"cannot verify the prior failed state ({exc})")
    if (not isinstance(previous_state, dict)
            or previous_state.get("status") != "failed"
            or previous_state.get("protocol") != PROTOCOL
            or previous_state.get("completed_cells") != []
            or previous_state.get("settings") != previous):
        reject("automatic recovery requires a matching failed state with zero completed cells")
    comparison_path = output / "comparison-time-interventions.json"
    if comparison_path.exists() or comparison_path.is_symlink():
        reject("an existing comparison contains evaluation outcomes")

    configs = {spec[key]: spec[value] for spec in specs
               for key, value in (("shift_path", "shift_config"), ("config_path", "config"))}
    cell_directories = {spec["result"].parent for spec in specs}
    metadata = set(configs) | {directory / name for directory in cell_directories
                              for name in ("resume_manifest.json", "resume_summary.json")}
    locks = {directory / ".resume.lock" for directory in cell_directories}
    # The native timepath launcher writes the object-shift config through
    # pi05_gripper_event_experiment.ensure_noise_config. That shared writer
    # retains this historical lock name even when image noise is disabled.
    # Its directory is the shared shift-config directory, not result.parent.
    locks.update(spec["shift_path"].parent / ".fixed_image_noise.lock" for spec in specs)
    artifacts = {protocol_path, state_path}

    def check_empty_summary(path):
        try:
            value = json.loads(path.read_text())
            suites = value["per_suite"]
            rows = [value, *suites.values()]
            empty = isinstance(suites, dict) and all(
                type(row.get("saved_episodes")) is int and row["saved_episodes"] == 0
                and type(row.get("successes")) is int and row["successes"] == 0
                and row.get("pc_success_saved") is None for row in rows
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            empty = False
        if not empty:
            reject(f"saved episodes or unverifiable progress exist in {path}")

    with ExitStack() as held_locks:
        # Do not move or replace lock files: another process may still have the
        # original inode open. A previous orphaned native worker must finish first.
        for path in sorted(locks):
            if path.exists():
                if path.is_symlink() or not path.is_file() or path.resolve() != path:
                    reject(f"cannot verify native lock {path}")
                stream = held_locks.enter_context(path.open("a"))
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    reject(f"a native evaluator is still using {path.parent}")

        evaluations = output / "evaluations"
        if evaluations.is_symlink() or (evaluations.exists() and not evaluations.is_dir()):
            reject(f"cannot inspect the evaluation directory: {evaluations}")
        for path in sorted(evaluations.rglob("*")):
            if path.is_symlink():
                reject(f"cannot establish absence of episode evidence through {path}")
            if path.is_dir():
                continue
            if not path.is_file() or path not in metadata | locks:
                reject(f"episode/task outputs, traces, or unknown evaluation artifacts exist: {path}")
            if path in configs:
                try:
                    matches = json.loads(path.read_text()) == configs[path]
                except (OSError, ValueError):
                    matches = False
                if not matches:
                    reject(f"a per-cell configuration changed: {path}")
            elif path.name == "resume_summary.json":
                check_empty_summary(path)
            if path not in locks:
                artifacts.add(path)

        # Only the named cells' generated logs belong to this attempt. Personal
        # notes or other files in the output directory stay exactly where they are.
        for spec in specs:
            identity = f"scene{spec['scene_seed']}-policy{spec['policy_seed']}"
            outer_log = output / "logs" / f"{spec['mode']}-{identity}.log"
            if outer_log.exists():
                artifacts.add(outer_log)
            native_logs = output / "logs" / spec["mode"] / identity / "eval" / "task7-timepath-qkv"
            artifacts.update(native_logs.glob("eval_*.log"))
        if any(path.is_symlink() or not path.is_file() or path.resolve() != path for path in artifacts):
            reject("a failed-attempt artifact is not a regular file inside the output directory")

        archive_root = output / "empty_failed_attempts"
        if archive_root.is_symlink():
            reject("the failed-attempt archive directory is a symlink")
        archive_root.mkdir(exist_ok=True)
        archive = Path(tempfile.mkdtemp(prefix="attempt-", dir=archive_root))
        # Move protocol/state last. If archival fails, roll back every move so
        # frozen settings cannot disappear while their old native caches remain.
        ordered = sorted(artifacts - {protocol_path, state_path}) + [protocol_path, state_path]
        moved = []
        try:
            for source in ordered:
                target = archive / source.relative_to(output)
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
                moved.append((source, target))
            atomic_json(archive / "recovery.json", {
                "archived_at": utc_now(),
                "reason": "Source-only change after a failed attempt with no saved episode/task/trace evidence.",
                "files": [str(path.relative_to(output)) for path in ordered],
                "next_source_sha256": settings["source_sha256"],
            })
        except BaseException:
            for source, target in reversed(moved):
                target.rename(source)
            raise
    print(f"Archived empty failed attempt: {archive}; restarting with the updated execution source.", flush=True)
    return archive


def tokenizer_fingerprint(directory):
    """Match the native launcher's processor provenance, including in-place changes."""
    directory = Path(directory)
    files = [[str(path.relative_to(directory)), path.stat().st_size, path.stat().st_mtime_ns]
             for path in sorted(directory.rglob("*")) if path.is_file()]
    if not directory.is_dir() or not files:
        raise FileNotFoundError(f"Missing tokenizer files: {directory}")
    return files


def source_hashes():
    from lerobot.scripts.pi05_task7_train_eval import file_sha256

    paths = (
        "src/lerobot/scripts/pi05_task7_closed_loop_runtime.py",
        "src/lerobot/scripts/pi05_task7_dualseed_shift.py",
        "src/lerobot/scripts/pi05_task7_object_shift.py",
        "src/lerobot/scripts/pi05_task7_time_evidence.py",
        "src/lerobot/scripts/pi05_task7_full_sampling.py",
        "src/lerobot/scripts/lerobot_eval.py",
        "src/lerobot/scripts/libero_base_eval.py",
        "src/lerobot/scripts/libero_finetuned_prompt.py",
        "src/lerobot/envs/libero.py",
        "src/lerobot/policies/pi05/modeling_pi05.py",
        "src/lerobot/policies/pi05/action_time.py",
        "src/lerobot/policies/pi05/processor_pi05.py",
        "run_eval_libero-full.sh", "run_eval_libero_vlm_only.sh",
    )
    return {name: file_sha256(ROOT / name) for name in paths}


class Backend:
    """Use the existing audited training provenance and native evaluator unchanged."""

    def __init__(self):
        from lerobot.scripts import pi05_task7_dualseed_eval as dual
        from lerobot.scripts import pi05_task7_timepath_experiment as timepath
        from lerobot.scripts import pi05_task7_closed_loop_runtime as runtime

        self.dual, self.timepath, self.runtime = dual, timepath, runtime

    def prepare(self, args):
        from lerobot.scripts.pi05_task7_full_sampling import validate_sampling_config

        record = self.timepath.check_plan(args.directory)
        job = self.timepath.build_jobs(
            record, args.directory, args.output_root, selection="time_input", repeat=args.repeat
        )[0]
        checkpoint = Path(job["checkpoint"]).resolve(strict=True)
        if args.time_input_checkpoint is not None and args.time_input_checkpoint != checkpoint:
            raise ValueError(
                "--time-input-checkpoint must match the audited --directory/--output-root/--repeat "
                f"Time-Input final checkpoint: {checkpoint}"
            )
        if not args.tokenizer.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {args.tokenizer}")
        self.timepath.audit(record, [job])
        config = json.loads((checkpoint / "config.json").read_text())
        validate_sampling_config(config)
        if not math.isclose(float(config.get("action_time_encoding_fps", 0)), float(record["fps"])):
            raise ValueError("Checkpoint action-time FPS differs from training plan")
        return record, job, config

    def fingerprint(self, checkpoint):
        return self.dual.fingerprint(checkpoint)

    def spec(self, args, record, job, mode, scene, policy):
        opts = evaluation_options(args, mode, scene, policy)
        result = self.timepath.result_path(job, record, opts)
        return {
            "mode": mode, "scene_seed": scene, "policy_seed": policy,
            "result": result, "shift_path": self.timepath.evaluation_config_path(opts, record),
            "shift_config": self.timepath.shift_config(opts),
            "config_path": result.parent / "time_intervention_config.json",
            "config": intervention_config(args, job, mode, scene, policy, result),
        }

    def verify(self, args, record, job, spec):
        from lerobot.scripts.pi05_task7_train_eval import file_sha256

        opts = evaluation_options(args, spec["mode"], spec["scene_seed"], spec["policy_seed"])
        row = self.dual.verified_result(record, opts, job, spec["scene_seed"], spec["policy_seed"])
        manifest = json.loads((spec["result"].parent / "resume_manifest.json").read_text())
        if manifest.get("task7_time_intervention") != self.runtime.manifest_for_config(spec["config_path"]):
            raise ValueError(f"Time intervention or trajectory implementation changed: {spec['result']}")
        if manifest.get("evaluator_sha256") != file_sha256(ROOT / "src/lerobot/scripts/lerobot_eval.py"):
            raise ValueError("Cached native evaluator differs from the current checkout")
        expected = {
            "--policy.path": str(Path(job["checkpoint"]).resolve()),
            "--env.type": "libero", "--env.task": "libero_10", "--env.task_ids": "[7]",
            "--env.control_mode": "relative", "--eval.batch_size": "1", "--eval.n_episodes": "10",
            "--policy.n_action_steps": "10", "--policy.compile_model": "false",
            "--policy.gradient_checkpointing": "false", "--policy.use_amp": "false",
            "--seed": str(args.init_seed),
        }
        arguments = dict(item.split("=", 1) for item in manifest.get("arguments", []) if "=" in item)
        if any(arguments.get(key) != value for key, value in expected.items()):
            raise ValueError("Cached evaluator arguments changed the native execution protocol")
        processors = manifest.get("finetuned_prompt_evaluation", {})
        if (processors.get("tokenizer_path") != str(args.tokenizer)
                or processors.get("tokenizer_files") != tokenizer_fingerprint(args.tokenizer)):
            raise ValueError("Cached evaluation uses a different tokenizer or changed tokenizer files")
        traces = self.runtime.require_saved_traces(spec["config"], successes=row["per_episode"])
        return {
            "successes": row["per_episode"], "traces": [str(path.resolve()) for path in traces],
            "eval_info": str(spec["result"].resolve()),
            "resume_manifest": str((spec["result"].parent / "resume_manifest.json").resolve()),
            "checkpoint": str(Path(job["checkpoint"]).resolve()),
        }


def settings_for(args, job, config, fingerprint):
    from lerobot.scripts.pi05_task7_train_eval import file_sha256

    return {
        "protocol": PROTOCOL, "checkpoint": str(Path(job["checkpoint"]).resolve()),
        "checkpoint_files": fingerprint, "repeat": args.repeat,
        "preparation_sha256": file_sha256(args.directory / "manifest.json"),
        "uniform_plan_sha256": file_sha256(args.directory / "uniform.npz"),
        "tokenizer": str(args.tokenizer), "tokenizer_files": tokenizer_fingerprint(args.tokenizer),
        "source_sha256": source_hashes(),
        "scene_seeds": list(args.scene_seeds), "policy_seeds": list(args.test_policy_seeds),
        "fixed_init_states": list(range(10)), "init_seed": args.init_seed, "shift_m": args.shift_m,
        "modes": list(MODES), "horizon": 50, "executed_steps": 10,
        "num_inference_steps": config["num_inference_steps"], "control_mode": "relative",
        "primary_contrast": "correct_vs_zero", "secondary_contrast": "correct_vs_reversed",
        "policy_noise": "Original dual-seed SHA256(policy_seed, init slot), native sampler RNG sequence.",
        "selection": "Frozen historical task7 stress preset; no success-based recalibration or selection.",
        "scope": "Same checkpoint. Repeat is a run namespace, not an independent training seed.",
    }


def show_figure_outputs(args):
    print(
        f"Closed-loop figures after all cells complete: {args.output / 'fig08_closed_loop_success.png'} "
        f"and {args.output / 'fig09_execution_trajectories.png'}.", flush=True,
    )
    print(
        "Offline seven-dimensional figures (fig06/fig07) use run_analyze_task7_seven_dim.sh "
        "with saved full_sampling_evidence.json/full_sampling_arrays.npz; no GPU inference is needed.",
        flush=True,
    )
    print("For the standard sibling full_sampling cache:", flush=True)
    print(shlex.join([
        "bash", str(ROOT / "run_analyze_task7_seven_dim.sh"),
        "--input", str(args.output.parent / "full_sampling"),
        "--output", str(args.output.parent / "seven_dim"),
    ]), flush=True)


def show_plan(args):
    cells = len(args.scene_seeds) * len(args.test_policy_seeds)
    print("DRY RUN: no files are read/written; no models, training or MuJoCo are started.")
    print(f"Same Time-Input checkpoint; Correct / Zero / Reversed; {cells * 10} episodes per arm, {cells * 30} total.")
    print("Native 50-step chunks; execute 10 and reobserve. Same initial state/RNG, later observations may differ.")
    print("A new trajectory cache is required for Correct too; old success-only caches are not reused.")
    show_figure_outputs(args)
    for index, (mode, scene, policy) in enumerate(
        (mode, scene, policy) for mode in MODES for scene in args.scene_seeds for policy in args.test_policy_seeds
    ):
        print(f"[{mode} scene={scene} policy={policy}] per-cell LIBERO_TASK7_TIME_INTERVENTION_CONFIG")
        print(shlex.join(cell_command(args, mode, scene, policy, args.gpu_ids[index % len(args.gpu_ids)])))
    print(f"Paired results and execution figures: {args.output}")


def run(args, *, backend=None, command_runner=None, reporter=None):
    if args.dry_run:
        show_plan(args)
        return None
    from lerobot.scripts.pi05_task7_train_eval import atomic_json, output_lock, run_logged, utc_now

    show_figure_outputs(args)
    backend = Backend() if backend is None else backend
    command_runner = run_logged if command_runner is None else command_runner
    record, job, config = backend.prepare(args)
    fingerprint = backend.fingerprint(job["checkpoint"])
    settings = settings_for(args, job, config, fingerprint)
    with output_lock(args.output):
        specs = [backend.spec(args, record, job, mode, scene, policy)
                 for mode in MODES for scene in args.scene_seeds for policy in args.test_policy_seeds]
        archive = recover_empty_failed_attempt(args.output, settings, specs)
        freeze_json(args.output / "closed_loop_protocol.json", settings)
        state_path = args.output / "closed_loop_state.json"
        state = {"protocol": PROTOCOL, "status": "running", "settings": settings,
                 "started_at": utc_now(), "completed_cells": []}
        if archive is not None:
            state["recovered_empty_attempt"] = str(archive)
        atomic_json(state_path, state)
        try:
            # Validate/freeze all configs before starting workers. Each intervention has its own cache.
            for spec in specs:
                freeze_json(spec["shift_path"], spec["shift_config"])
                freeze_json(spec["config_path"], spec["config"])

            def evaluate(spec, gpu):
                if spec["result"].is_file():
                    print(f"Verify cached {spec['mode']} scene={spec['scene_seed']} policy={spec['policy_seed']}", flush=True)
                else:
                    command = cell_command(args, spec["mode"], spec["scene_seed"], spec["policy_seed"], gpu)
                    log = args.output / "logs" / f"{spec['mode']}-scene{spec['scene_seed']}-policy{spec['policy_seed']}.log"
                    command_runner(command, log, cell_environment(spec["config_path"], gpu))
                return backend.verify(args, record, job, spec)

            for start in range(0, len(specs), len(args.gpu_ids)):
                batch = specs[start:start + len(args.gpu_ids)]
                with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    futures = [pool.submit(evaluate, spec, args.gpu_ids[index]) for index, spec in enumerate(batch)]
                    for spec, future in zip(batch, futures, strict=True):
                        future.result()
                        state["completed_cells"].append({
                            "mode": spec["mode"], "scene_seed": spec["scene_seed"],
                            "policy_seed": spec["policy_seed"], "eval_info": str(spec["result"]),
                        })
                        atomic_json(state_path, state)
            # Re-read every cache and its complete traces before producing a comparison.
            rows = {(spec["mode"], spec["scene_seed"], spec["policy_seed"]): backend.verify(args, record, job, spec)
                    for spec in specs}
            if backend.fingerprint(job["checkpoint"]) != fingerprint:
                raise ValueError("Checkpoint changed while evaluating the three arms")
            if source_hashes() != settings["source_sha256"]:
                raise ValueError("Execution source changed while evaluating the three arms")
            if tokenizer_fingerprint(args.tokenizer) != settings["tokenizer_files"]:
                raise ValueError("Tokenizer files changed while evaluating the three arms")
            comparison = {
                "protocol": COMPARISON_PROTOCOL, "status": "completed", "settings": settings,
                "cells": [{"scene_seed": scene, "policy_seed": policy,
                           "results": {mode: rows[(mode, scene, policy)] for mode in MODES}}
                          for scene in args.scene_seeds for policy in args.test_policy_seeds],
            }
            comparison_path = args.output / "comparison-time-interventions.json"
            atomic_json(comparison_path, comparison)
            if reporter is None:
                from lerobot.scripts.pi05_task7_closed_loop_report import run_report
                reporter = run_report
            reporter(comparison_path, args.output)
            state.update(status="completed", completed_at=utc_now(), comparison_json=str(comparison_path))
            atomic_json(state_path, state)
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=utc_now())
            atomic_json(state_path, state)
            raise
    print(f"Closed-loop experiment completed: {state_path}", flush=True)
    return state


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", os.environ.get("CKPT_ROOT", str(DEFAULT_OUTPUT)))))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--time-input-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path(os.environ.get("TOKENIZER_PATH", "/home/wyn/models/paligemma-3b-pt-224")))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1"))
    parser.add_argument("--scene-seeds", default="20261103,20261104")
    parser.add_argument("--test-policy-seeds", default="1001,1002,1003")
    parser.add_argument("--init-seed", type=int, default=3000)
    parser.add_argument("--shift-m", type=float, default=0.12)
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "false").lower() == "true")
    return parser


def parse_args(argv=None):
    args = make_parser().parse_args(argv)
    args.gpu_ids = integer_list(args.gpus, "--gpus")
    args.scene_seeds = integer_list(args.scene_seeds, "--scene-seeds")
    args.test_policy_seeds = integer_list(args.test_policy_seeds, "--test-policy-seeds")
    if args.repeat < 1 or not 0 <= args.init_seed < 2**31 - 10 or not math.isfinite(args.shift_m) or not 0 < args.shift_m <= 0.14:
        raise ValueError("Invalid repeat, initial seed or shift magnitude")
    for key in ("directory", "output_root", "tokenizer"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.time_input_checkpoint is not None:
        args.time_input_checkpoint = args.time_input_checkpoint.expanduser().resolve()
    args.output = (args.output or args.output_root / "analysis" / "task7-train-eval" / f"r{args.repeat}" / "closed_loop").expanduser().resolve()
    return args


def main(argv=None):
    parser = make_parser()
    try:
        run(parse_args(argv))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Task7 closed-loop evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
