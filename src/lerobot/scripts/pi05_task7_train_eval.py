"""Train task7 models, evaluate each, then run paired time-alignment diagnostics.

The existing two-GPU, batch32-per-GPU, 1000-update training recipe is reused
without changing its samples, prompt, flow-noise, or flow-time streams.
The evaluation preset is written before training and never selected by the
new model's success rate. Repeat selects a run namespace, not a new training
seed. No calibration lock is created, edited, or relabelled here.

Heavy dependencies are imported only after --dry-run has returned. A dry run
requires neither checkpoints nor a saved plan and does not create directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL = "pi05-task7-train-eval-v1"
REPORT_PROTOCOL = "pi05-task7-fixed-protocol-v1"
CASES = ("rope", "time_input")
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = Path("/data1/wyn/prompt-work/gripper-event-task7-1200/base")
DEFAULT_OUTPUT = Path("/data1/wyn/chkpt/2601-lerobot/PromptVLA")
DEFAULT_SCENES = "20261103,20261104"
DEFAULT_POLICY_SEEDS = "1001,1002,1003"
DEFAULT_FLOW_TIMES = "0.2,0.5,0.8"
DEFAULT_NOISE_SEEDS = "3000,4000,5000"
TIMEPATH_MODULE = "lerobot.scripts.pi05_task7_timepath_experiment"
CONSTANT_MODULE = "lerobot.scripts.pi05_task7_constant_action"
EVIDENCE_MODULE = "lerobot.scripts.pi05_task7_time_evidence"
FULL_SAMPLING_MODULE = "lerobot.scripts.pi05_task7_full_sampling"
# Match the frozen closed-loop entry point in run_eval_libero-full.sh.
EXECUTED_ACTION_STEPS = 10


def integer_list(value: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated integers") from exc
    if (
        not values
        or len(set(values)) != len(values)
        or any(number < 0 or number >= 2**31 for number in values)
    ):
        raise ValueError(f"{label} must contain distinct integers in [0, 2**31)")
    return values


def flow_times(value: str) -> tuple[float, ...]:
    try:
        times = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("--flow-times must contain comma-separated numbers") from exc
    if (
        not times
        or len(set(times)) != len(times)
        or any(not math.isfinite(t) or not 0 < t < 1 for t in times)
    ):
        raise ValueError("--flow-times must contain distinct finite values in (0, 1)")
    return times


def comma(values) -> str:
    return ",".join(str(value) for value in values)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def output_lock(directory: Path):
    """Fail on concurrent pipelines sharing an output; process exit releases it."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".pipeline.lock").open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another task7 pipeline is using {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "LIBERO_TASK7_OBJECT_SHIFT_CONFIG",
        "LIBERO_TASK7_DUALSEED_SHIFT_CONFIG",
        "LIBERO_ROBUSTNESS_MANIFEST",
        "LIBERO_BENCHMARK",
        "LIBERO_FIXED_NOISE_CONFIG",
        "PI05_CADENCE_PLAN",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["DRY_RUN"] = "false"
    return env


def run_logged(command: list[str], log: Path, env: dict[str, str]) -> None:
    """Stream output while retaining a complete log; propagate any failed child."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {shlex.join(command)}\nLog: {log}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{utc_now()}] {shlex.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("Child stdout pipe was not created")
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            returncode = process.wait()
        except BaseException:
            # DDP training has descendants; stop the process group as well.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def training_command(args, case: str) -> list[str]:
    command = [
        sys.executable, "-u", "-m", TIMEPATH_MODULE, "train", case,
        "--directory", str(args.directory),
        "--output-root", str(args.output_root),
        "--repeat", str(args.repeat),
        "--source", str(args.source),
        "--tokenizer", str(args.tokenizer),
        "--prompt-dir", str(args.prompt_dir),
        "--dataset-repo-id", args.dataset_repo_id,
        "--gpus", comma(args.gpu_ids),
        "--port", str(args.port),
        "--num-workers", str(args.num_workers),
        "--wandb-mode", args.wandb_mode,
        "--log-root", str(args.log_root),
    ]
    if args.prompt is not None:
        command.extend(["--prompt", str(args.prompt)])
    return command


def evaluation_preview_command(args, case: str, scene: int, policy: int, gpu: int) -> list[str]:
    """Mirror the original dual-seed launcher without loading an experiment plan."""
    return [
        sys.executable, "-u", "-m", TIMEPATH_MODULE, "eval", case,
        "--directory", str(args.directory),
        "--output-root", str(args.output_root),
        "--repeat", str(args.repeat),
        "--eval-root", str(args.control_eval_root),
        "--tokenizer", str(args.tokenizer),
        "--eval-seed", str(args.init_seed),
        "--episodes", "10",
        "--shift-m", str(args.shift_m),
        "--scene-seed", str(scene),
        "--policy-seed", str(policy),
        "--gpu", str(gpu),
        "--log-root", str(args.log_root),
    ]


def constant_command(args, checkpoint: Path | None = None) -> list[str]:
    command = [sys.executable, "-u", "-m", CONSTANT_MODULE]
    if checkpoint is None:
        command.extend([
            "--from-task7-plan",
            "--directory", str(args.directory),
            "--output-root", str(args.output_root),
            "--repeat", str(args.repeat),
        ])
    else:
        command.extend(["--checkpoint", str(checkpoint)])
    command.extend(["--output", str(args.output / "constant_action")])
    return command


def evidence_command(args, comparison: Path, checkpoints: dict[str, Path] | None = None) -> list[str]:
    command = [
        sys.executable, "-u", "-m", EVIDENCE_MODULE,
        "--mode", "all",
        "--comparison-json", str(comparison),
        "--directory", str(args.directory),
        "--output-root", str(args.output_root),
        "--repeat", str(args.repeat),
        "--tokenizer", str(args.tokenizer),
        "--dataset-repo-id", args.dataset_repo_id,
        "--gpu", str(args.probe_gpu),
        "--flow-times", comma(args.flow_times),
        "--noise-seeds", comma(args.noise_seeds),
        "--probe-demos", str(args.probe_demos),
        "--windows-per-demo", str(args.windows_per_demo),
        "--output", str(args.output / "evidence"),
    ]
    if checkpoints is not None:
        command.extend([
            "--rope-checkpoint", str(checkpoints["rope"]),
            "--time-input-checkpoint", str(checkpoints["time_input"]),
        ])
    return command


def full_sampling_command(args, checkpoint: Path | None = None) -> list[str]:
    """Use the native sampler and one Time-Input model for all three conditions."""
    command = [
        sys.executable, "-u", "-m", FULL_SAMPLING_MODULE,
        "--directory", str(args.directory),
        "--output-root", str(args.output_root),
        "--repeat", str(args.repeat),
        "--tokenizer", str(args.tokenizer),
        "--dataset-repo-id", args.dataset_repo_id,
        "--gpu", str(args.probe_gpu),
        "--noise-seeds", comma(args.noise_seeds),
        "--probe-demos", str(args.probe_demos),
        "--windows-per-demo", str(args.windows_per_demo),
        "--executed-steps", str(EXECUTED_ACTION_STEPS),
        "--output", str(args.output / "full_sampling"),
    ]
    if checkpoint is not None:
        command.extend(["--time-input-checkpoint", str(checkpoint)])
    return command


def show_plan(args) -> None:
    print("DRY RUN: no files are read or written; no GPU, model, or saved plan is required.")
    print(
        "Recipe: 1000 updates, 2 DDP GPUs, batch32/GPU, contiguous action chunk50; "
        f"cases={comma(CASES)}, repeat={args.repeat} (run namespace, not a training seed)."
    )
    print(
        f"Fixed evaluation: scenes={comma(args.scene_seeds)}, "
        f"policy seeds={comma(args.test_policy_seeds)}, init IDs=0..9, shift={args.shift_m}m. "
        "No new calibration or success-rate filtering."
    )
    cells = [(scene, policy) for scene in args.scene_seeds for policy in args.test_policy_seeds]
    for case in CASES:
        if args.skip_training:
            print(f"\n[CHECK/AUDIT EXISTING {case}] Missing or incomplete checkpoint is an error.")
        else:
            suffix = " (reuse only if a complete audited checkpoint exists)" if args.reuse_existing else ""
            print(f"\n[TRAIN {case}]{suffix}")
            print(shlex.join(training_command(args, case)))
        print(f"[AUDIT {case}] Require the final 001000 checkpoint and original training records.")
        print(f"[EVALUATE {case} IMMEDIATELY] Two cells at a time, one per GPU.")
        for index, (scene, policy) in enumerate(cells):
            print(shlex.join(evaluation_preview_command(
                args, case, scene, policy, args.gpu_ids[index % len(args.gpu_ids)]
            )))
    print("\n[PAIRED TRAINING AUDIT] Check both models used the same original inputs.")
    comparison = args.output / "comparison-fixed_protocol.json"
    print(f"[REPORT] Verify every cached outcome and write {comparison}")
    print("\n[CONSTANT-ACTION INPUT PROBE]")
    print(shlex.join(constant_command(args)))
    print("\n[ONE-STEP FLOW TIME-ALIGNMENT PROBE AND FIGURES]")
    print(shlex.join(evidence_command(args, comparison)))
    print("\n[FULL DENOISING: CORRECT / ZERO / REVERSED]")
    print("Same Time-Input weights, observations and pure initial noise; report first 10 and all 50 steps.")
    print(shlex.join(full_sampling_command(args)))
    print(f"\n[COMPLETE ONLY AFTER ALL STAGES SUCCEED] {args.output / 'pipeline_state.json'}")


class Backend:
    """Thin adapter around the existing validated training/evaluation code."""

    def __init__(self) -> None:
        from lerobot.scripts import pi05_task7_dualseed_eval as dual
        from lerobot.scripts import pi05_task7_timepath_experiment as timepath

        self.dual = dual
        self.timepath = timepath

    def check_plan(self, directory: Path) -> dict:
        return self.timepath.check_plan(directory)

    def build_jobs(self, record: dict, args) -> list[dict]:
        return [
            self.timepath.build_jobs(
                record, args.directory, args.output_root, selection=case, repeat=args.repeat
            )[0]
            for case in CASES
        ]

    def audit(self, record: dict, jobs: list[dict]) -> None:
        self.timepath.audit(record, jobs)

    def fingerprint(self, checkpoint: Path) -> list:
        return self.dual.fingerprint(checkpoint)

    def precreate_config(self, record: dict, args, job: dict, scene: int, policy: int) -> None:
        self.dual.precreate_configs(record, args, [job], scene, policy)

    def evaluate_one(self, record: dict, args, job: dict, scene: int, policy: int, gpu: int) -> dict:
        path = self.dual.run_location(record, args, job, scene, policy)
        if path.exists():
            return self.dual.verified_result(record, args, job, scene, policy)
        command = self.dual.evaluation_command(record, args, job, scene, policy, gpu)
        command[0] = sys.executable
        command.extend(["--log-root", str(args.log_root)])
        log = args.log_dir / f"scene{scene}-policy{policy}-{job['case']}-gpu{gpu}.log"
        env = process_environment()
        env["GPU_ID"] = str(gpu)
        run_logged(command, log, env)
        return self.dual.verified_result(record, args, job, scene, policy)

    def verified_result(self, record: dict, args, job: dict, scene: int, policy: int) -> dict:
        return self.dual.verified_result(record, args, job, scene, policy)

    def summary(self, rows: list[dict]) -> dict:
        return self.dual.render_table(rows, CASES, "fixed_protocol")


class Journal:
    def __init__(self, path: Path, settings: dict):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if self.data.get("protocol") != PROTOCOL or self.data.get("settings") != settings:
                raise ValueError(
                    f"Pipeline settings differ from {path}. Choose a different --output; "
                    "existing results and evaluation conditions will not be replaced."
                )
        else:
            self.data = {
                "protocol": PROTOCOL, "created_at": utc_now(),
                "settings": settings, "stages": {},
            }
        self.data["status"] = "running"
        self.data["updated_at"] = utc_now()
        self.save()

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        atomic_json(self.path, self.data)

    def start(self, stage: str, *, command=None, log=None) -> None:
        old = self.data["stages"].get(stage, {})
        self.data["stages"][stage] = {
            "status": "running", "attempt": old.get("attempt", 0) + 1,
            "started_at": utc_now(), "command": command,
            "log": str(log) if log is not None else None,
        }
        self.data["current_stage"] = stage
        self.save()

    def finish(self, stage: str, **details) -> None:
        self.data["stages"][stage].update(
            status="completed", finished_at=utc_now(), **details
        )
        self.save()

    def fail(self, error: BaseException) -> None:
        stage = self.data.get("current_stage")
        if stage in self.data["stages"]:
            self.data["stages"][stage].update(
                status="failed", finished_at=utc_now(), error=str(error)
            )
        self.data["status"] = "failed"
        self.data["error"] = f"{type(error).__name__}: {error}"
        self.save()

    def complete(self, **details) -> None:
        self.data.update(status="completed", completed_at=utc_now(), **details)
        self.data.pop("error", None)
        self.data.pop("current_stage", None)
        self.save()


def protocol_settings(args, record: dict) -> dict:
    return {
        "cases": list(CASES),
        "directory": str(args.directory),
        "preparation_sha256": file_sha256(args.directory / "manifest.json"),
        "uniform_plan_sha256": file_sha256(args.directory / "uniform.npz"),
        "training_episode": int(record["selection"]["episode_index"]),
        "training_repeat": args.repeat,
        "training_recipe": {
            "steps": 1000, "batch_per_gpu": 32, "world_size": 2,
            "chunk_size": 50, "prompt_seed": 1, "noise_seed": 1,
            "flow_time_seed": 1, "global_seed": 0,
        },
        "source": str(args.source),
        "prompt": str(args.prompt) if args.prompt is not None else None,
        "prompt_dir": str(args.prompt_dir),
        "tokenizer": str(args.tokenizer),
        "dataset_repo_id": args.dataset_repo_id,
        "output_root": str(args.output_root),
        "control_eval_root": str(args.control_eval_root),
        "scene_seeds": list(args.scene_seeds),
        "test_policy_seeds": list(args.test_policy_seeds),
        "init_seed": args.init_seed,
        "fixed_init_states": list(range(10)),
        "shift_m": args.shift_m,
        "flow_times": list(args.flow_times),
        "noise_seeds": list(args.noise_seeds),
        "probe_demos": args.probe_demos,
        "windows_per_demo": args.windows_per_demo,
        "selection": "Frozen before this run; no outcome-based scene selection or recalibration.",
    }


def check_run_states(jobs: list[dict], args) -> dict[str, bool]:
    """Preflight every run before starting expensive work; never overwrite one."""
    existing = {}
    for job in jobs:
        run = Path(job["run"])
        checkpoint = Path(job["checkpoint"])
        exists = run.exists()
        existing[job["case"]] = exists
        if not exists:
            if args.skip_training:
                raise FileNotFoundError(f"--skip-training requires a trained checkpoint: {checkpoint}")
            continue
        if not args.reuse_existing:
            raise FileExistsError(
                f"Training run already exists: {run}. Use --reuse-existing to audit/reuse "
                "completed models, or choose a new --repeat. Nothing was overwritten."
            )
        required = [
            checkpoint / "config.json", checkpoint / "train_config.json",
            checkpoint / "model.safetensors", run / "gripper_event_experiment.json",
        ]
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise ValueError(
                f"Existing {job['case']} run is incomplete; refusing to retrain over it. "
                f"Missing/empty: {missing}. Finish it with its original training workflow "
                "or choose a new --repeat."
            )
    return existing


def validate_cell(row: dict, job: dict, scene: int, policy: int) -> None:
    flags = row.get("per_episode")
    if (
        row.get("scene_seed") != scene
        or row.get("policy_seed") != policy
        or row.get("case") != job["case"]
        or row.get("episodes") != 10
        or not isinstance(flags, list)
        or len(flags) != 10
        or any(type(value) is not bool for value in flags)
        or row.get("successes") != sum(flags)
        or Path(row.get("checkpoint", "")).resolve() != Path(job["checkpoint"]).resolve()
    ):
        raise ValueError("Evaluation cell does not match the frozen scene/policy/model protocol")


def evaluate_model(backend, record: dict, args, job: dict, journal: Journal) -> list[dict]:
    cells = [(scene, policy) for scene in args.scene_seeds for policy in args.test_policy_seeds]
    rows = []
    stage = f"evaluate:{job['case']}"
    for start in range(0, len(cells), len(args.gpu_ids)):
        batch = cells[start:start + len(args.gpu_ids)]
        for scene, policy in batch:
            backend.precreate_config(record, args, job, scene, policy)
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = [
                pool.submit(
                    backend.evaluate_one, record, args, job, scene, policy, args.gpu_ids[index]
                )
                for index, (scene, policy) in enumerate(batch)
            ]
            for (scene, policy), future in zip(batch, futures, strict=True):
                row = future.result()
                validate_cell(row, job, scene, policy)
                rows.append(row)
                journal.data["stages"][stage]["completed_cells"] = rows
                journal.save()
    return rows


def verify_comparison(backend, record: dict, args, jobs: list[dict]) -> list[dict]:
    """Re-read and verify every model/physics manifest before reporting totals."""
    cells = []
    for scene in args.scene_seeds:
        for policy in args.test_policy_seeds:
            results = {}
            for job in jobs:
                row = backend.verified_result(record, args, job, scene, policy)
                validate_cell(row, job, scene, policy)
                results[job["case"]] = row
            cells.append({"scene_seed": scene, "policy_seed": policy, "results": results})
    return cells


def write_comparison(path: Path, settings: dict, rows: list[dict], summary: dict) -> dict:
    report = {
        "protocol": REPORT_PROTOCOL, "stage": "fixed_protocol",
        "settings": settings, "cases": list(CASES),
        "evaluations": rows, "summary": summary,
        "note": (
            "Historical task7 stress preset frozen before this run. No new baseline "
            "calibration, no success-band filtering, and no calibration episodes in "
            "these counts. This is not an unselected population benchmark. Policy "
            "seeds control inference RNG; repeat changes a run namespace, not the "
            "original fixed training seeds."
        ),
    }
    atomic_json(path, report)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "scene_seed", "policy_seed", "case", "successes", "episodes", "rate_pct"])
        for cell in rows:
            for case in CASES:
                result = cell["results"][case]
                writer.writerow([
                    "fixed_protocol", cell["scene_seed"], cell["policy_seed"], case,
                    result["successes"], 10, result["successes"] * 10.0,
                ])
    return report


def required_outputs(root: Path, names: tuple[str, ...]) -> list[str]:
    paths = [root / name for name in names]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Stage returned success but did not produce required artifacts: {missing}")
    return [str(path) for path in paths]


def run_pipeline(args, *, backend=None, command_runner=run_logged) -> dict | None:
    if args.dry_run:
        show_plan(args)
        return None
    backend = Backend() if backend is None else backend
    record = backend.check_plan(args.directory)
    jobs = backend.build_jobs(record, args)
    if [job["case"] for job in jobs] != list(CASES):
        raise ValueError("Pipeline requires exactly RoPE followed by Time-Input")
    existing = check_run_states(jobs, args)
    settings = protocol_settings(args, record)
    with output_lock(args.output):
        preset = args.output / "fixed_protocol.json"
        if preset.exists() and json.loads(preset.read_text()) != settings:
            raise ValueError("Saved fixed protocol changed; choose a separate --output")
        journal = Journal(args.output / "pipeline_state.json", settings)
        fingerprints = {}
        try:
            journal.start("freeze_protocol")
            env = process_environment()
            atomic_json(preset, settings)
            journal.finish("freeze_protocol", protocol_file=str(preset))
            for job in jobs:
                case = job["case"]
                stage = f"train:{case}"
                log = args.output / "logs" / f"train-{case}.log"
                command = training_command(args, case)
                journal.start(stage, command=command, log=log)
                if existing[case]:
                    print(f"\nReusing {case}; validating its final checkpoint and training records.", flush=True)
                    journal.finish(stage, action="reuse_existing", checkpoint=str(job["checkpoint"]))
                else:
                    command_runner(command, log, env)
                    backend.fingerprint(job["checkpoint"])
                    journal.finish(stage, action="trained", checkpoint=str(job["checkpoint"]))
                stage = f"audit:{case}"
                journal.start(stage)
                backend.audit(record, [job])
                fingerprints[case] = backend.fingerprint(job["checkpoint"])
                journal.finish(stage, checkpoint_files=fingerprints[case])
                stage = f"evaluate:{case}"
                journal.start(stage, log=args.log_dir)
                print(f"\nTraining ready: immediately evaluating {case} on the frozen task7 cells.", flush=True)
                rows = evaluate_model(backend, record, args, job, journal)
                journal.finish(stage, completed_cells=rows)
            journal.start("audit:paired_training")
            backend.audit(record, jobs)
            for job in jobs:
                if backend.fingerprint(job["checkpoint"]) != fingerprints[job["case"]]:
                    raise ValueError(f"Checkpoint changed during the pipeline: {job['checkpoint']}")
            journal.finish("audit:paired_training")
            journal.start("report")
            cells = verify_comparison(backend, record, args, jobs)
            comparison = args.output / "comparison-fixed_protocol.json"
            write_comparison(comparison, settings, cells, backend.summary(cells))
            journal.finish("report", comparison_json=str(comparison))
            checkpoints = {job["case"]: Path(job["checkpoint"]) for job in jobs}
            command = constant_command(args, checkpoints["time_input"])
            log = args.output / "logs" / "constant-action.log"
            journal.start("constant_action", command=command, log=log)
            command_runner(command, log, env)
            constant_files = required_outputs(args.output / "constant_action", (
                "constant_action_50_summary.json", "fig00_constant_action_50.png",
                "fig00_constant_action_50.svg", "constant_action_50_embeddings.npz",
                "constant_action_50_positions.csv",
            ))
            constant_report = json.loads(Path(constant_files[0]).read_text())
            if constant_report.get("source", {}).get("kind") != "trained_checkpoint":
                raise ValueError("Pipeline requires actual trained weights, not an illustrative drawing")
            journal.finish("constant_action", outputs=constant_files)
            command = evidence_command(args, comparison, checkpoints)
            log = args.output / "logs" / "time-evidence.log"
            journal.start("time_evidence", command=command, log=log)
            command_runner(command, log, env)
            evidence_files = required_outputs(args.output / "evidence", (
                "offline_probe.json", "time_alignment_evidence.json",
                "fig04_time_alignment_evidence.png", "fig04_time_alignment_evidence.svg",
            ))
            journal.finish("time_evidence", outputs=evidence_files)
            command = full_sampling_command(args, checkpoints["time_input"])
            log = args.output / "logs" / "full-sampling.log"
            journal.start("full_sampling", command=command, log=log)
            command_runner(command, log, env)
            full_sampling_files = required_outputs(args.output / "full_sampling", (
                "full_sampling_evidence.json", "full_sampling_arrays.npz",
                "fig05_full_sampling_time_alignment.png", "fig05_full_sampling_time_alignment.svg",
            ))
            journal.finish(
                "full_sampling", outputs=full_sampling_files,
                conditions=["correct", "zero", "reversed"],
                checkpoint=str(checkpoints["time_input"]),
                executed_steps=EXECUTED_ACTION_STEPS,
                inference_steps="checkpoint_config",
            )
            for job in jobs:
                if backend.fingerprint(job["checkpoint"]) != fingerprints[job["case"]]:
                    raise ValueError(f"Checkpoint changed while generating evidence: {job['checkpoint']}")
            journal.complete(
                comparison_json=str(comparison), constant_outputs=constant_files,
                evidence_outputs=evidence_files,
                full_sampling_outputs=full_sampling_files,
            )
        except BaseException as exc:
            journal.fail(exc)
            raise
    print(f"\nPipeline completed: {args.output / 'pipeline_state.json'}", flush=True)
    print(f"Paired task success: {comparison}", flush=True)
    print(f"One-step Flow figure: {args.output / 'evidence' / 'fig04_time_alignment_evidence.png'}", flush=True)
    print(
        f"Full-sampling alignment figure: {args.output / 'full_sampling' / 'fig05_full_sampling_time_alignment.png'}",
        flush=True,
    )
    print(
        "The plots report measured gains or losses; completing the pipeline does not "
        "prejudge whether correct time correspondence improves prediction.", flush=True,
    )
    return journal.data


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get(
        "OUTPUT_ROOT", os.environ.get("CKPT_ROOT", str(DEFAULT_OUTPUT))
    )))
    parser.add_argument("--output", "--pipeline-output", type=Path, default=None)
    parser.add_argument("--control-eval-root", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=Path(os.environ.get(
        "LOG_ROOT", "/data1/wyn/chkpt/2601-lerobot/logs"
    )))
    parser.add_argument("--source", type=Path, default=Path(os.environ.get(
        "PRETRAINED_PATH", os.environ.get(
            "BASE_PRETRAINED_PATH", "/home/wyn/models/lerobot/pi05_libero_base"
        )
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
    parser.add_argument("--probe-gpu", type=int, default=None)
    parser.add_argument("--port", type=int, default=int(os.environ.get("MAIN_PROCESS_PORT", "29589")))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--scene-seeds", default=DEFAULT_SCENES)
    parser.add_argument("--test-policy-seeds", default=DEFAULT_POLICY_SEEDS)
    parser.add_argument("--init-seed", type=int, default=3000)
    parser.add_argument("--shift-m", type=float, default=0.12)
    parser.add_argument("--flow-times", default=DEFAULT_FLOW_TIMES)
    parser.add_argument("--noise-seeds", default=DEFAULT_NOISE_SEEDS)
    parser.add_argument("--probe-demos", type=int, default=8)
    parser.add_argument("--windows-per-demo", type=int, default=3)
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Audit and reuse complete checkpoints; train only runs that do not exist.",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Require both complete checkpoints; audit, evaluate, and visualize them.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        default=os.environ.get("DRY_RUN", "false").lower() == "true",
        help="Print the full sequence without reading a plan, allocating GPUs, or writing files.",
    )
    return parser


def parse_args(argv=None):
    args = make_parser().parse_args(argv)
    args.gpu_ids = integer_list(args.gpus, "--gpus")
    args.scene_seeds = integer_list(args.scene_seeds, "--scene-seeds")
    args.test_policy_seeds = integer_list(args.test_policy_seeds, "--test-policy-seeds")
    args.noise_seeds = integer_list(args.noise_seeds, "--noise-seeds")
    args.flow_times = flow_times(args.flow_times)
    if len(args.gpu_ids) != 2:
        raise ValueError("The unchanged task7 training recipe requires exactly two GPUs")
    if (
        args.repeat < 1 or not 1 <= args.port <= 65535 or args.num_workers < 0
        or args.init_seed < 0 or args.init_seed >= 2**31 - 10
        or not math.isfinite(args.shift_m) or not 0 < args.shift_m <= 0.14
        or args.probe_demos < 1 or args.windows_per_demo < 1
    ):
        raise ValueError("Invalid repeat, port, workers, init seed, shift, or probe sample count")
    args.probe_gpu = args.gpu_ids[0] if args.probe_gpu is None else args.probe_gpu
    if args.probe_gpu not in args.gpu_ids:
        raise ValueError("--probe-gpu must be one of the two declared --gpus")
    args.gpus = comma(args.gpu_ids)
    args.reuse_existing = args.reuse_existing or args.skip_training
    for key in ("directory", "output_root", "log_root", "source", "tokenizer", "prompt_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.prompt is not None:
        args.prompt = Path(args.prompt).expanduser().resolve()
    args.output = (
        args.output or args.output_root / "analysis" / "task7-train-eval" / f"r{args.repeat}"
    ).expanduser().resolve()
    args.control_eval_root = (
        args.control_eval_root or args.output_root / "eval" / "task7-timepath-qkv"
    ).expanduser().resolve()
    # Compatibility attributes consumed by the existing dual-seed evaluator.
    args.optimized_eval_root = args.output_root / "eval" / "task7-input-spectral"
    args.log_dir = args.output / "logs" / "evaluation"
    return args


def main(argv=None) -> None:
    parser = make_parser()
    try:
        args = parse_args(argv)
        run_pipeline(args)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Task7 train/evaluate pipeline failed: {exc}\n")


if __name__ == "__main__":
    main()
