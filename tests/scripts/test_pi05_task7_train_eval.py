"""Orchestration tests use fake training/evaluation, never GPUs or real weights."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.scripts import pi05_task7_train_eval as pipeline


def arguments(tmp_path, *extra):
    return pipeline.parse_args([
        "--directory", str(tmp_path / "plan"),
        "--output-root", str(tmp_path / "runs"),
        "--output", str(tmp_path / "analysis"),
        "--log-root", str(tmp_path / "logs"),
        "--repeat", "3",
        "--gpus", "1,3",
        *extra,
    ])


class FakeBackend:
    """Record scheduling and produce explicit synthetic success fixtures."""

    def __init__(self, args):
        self.args = args
        self.events = []
        self.commands = []
        self.fail_eval = None
        self.rows = {}
        self.precreated = set()
        self.record = {"selection": {"episode_index": 7}}
        self.jobs = [
            {
                "case": case,
                "run": args.output_root / f"r{args.repeat}-{case}",
                "checkpoint": args.output_root / f"r{args.repeat}-{case}" / "checkpoints"
                / "001000" / "pretrained_model",
            }
            for case in pipeline.CASES
        ]
        args.directory.mkdir(parents=True)
        (args.directory / "manifest.json").write_text(json.dumps(self.record))
        (args.directory / "uniform.npz").write_bytes(b"synthetic plan fixture")

    def check_plan(self, directory):
        assert directory == self.args.directory
        return self.record

    def build_jobs(self, record, args):
        assert record == self.record
        assert args.repeat == 3
        return self.jobs

    def materialize(self, case):
        job = next(job for job in self.jobs if job["case"] == case)
        job["checkpoint"].mkdir(parents=True)
        for name in ("config.json", "train_config.json", "model.safetensors"):
            (job["checkpoint"] / name).write_bytes(b"synthetic checkpoint fixture")
        (job["run"] / "gripper_event_experiment.json").write_text("{}")
        return job

    def audit(self, record, jobs):
        self.events.append("audit:" + ",".join(job["case"] for job in jobs))
        for job in jobs:
            assert (job["checkpoint"] / "model.safetensors").is_file()

    def fingerprint(self, checkpoint):
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint)
        return [
            [path.name, path.stat().st_size, path.stat().st_mtime_ns]
            for path in sorted(checkpoint.iterdir())
        ]

    def precreate_config(self, record, args, job, scene, policy):
        self.precreated.add((job["case"], scene, policy))

    def evaluate_one(self, record, args, job, scene, policy, gpu):
        case = job["case"]
        assert (case, scene, policy) in self.precreated
        assert gpu in (1, 3)
        self.events.append(f"eval:{case}:{scene}:{policy}")
        if self.fail_eval == (case, scene, policy):
            raise RuntimeError("synthetic evaluation failure")
        flags = [True] * (2 if case == "rope" else 3)
        flags += [False] * (10 - len(flags))
        row = {
            "case": case, "scene_seed": scene, "policy_seed": policy,
            "episodes": 10, "successes": sum(flags), "per_episode": flags,
            "checkpoint": str(job["checkpoint"]),
            "eval_info": str(job["run"] / f"scene{scene}-policy{policy}.json"),
        }
        self.rows[(case, scene, policy)] = row
        return row

    def verified_result(self, record, args, job, scene, policy):
        self.events.append(f"verify:{job['case']}:{scene}:{policy}")
        return self.rows[(job["case"], scene, policy)]

    def summary(self, rows):
        return {
            "aggregate": {
                case: [sum(row["results"][case]["successes"] for row in rows), 10 * len(rows)]
                for case in pipeline.CASES
            }
        }

    def command_runner(self, command, log, env):
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert env["DRY_RUN"] == "false"
        self.commands.append(command)
        module = command[command.index("-m") + 1]
        if module == pipeline.TIMEPATH_MODULE:
            case = command[command.index("train") + 1]
            self.events.append("train:" + case)
            self.materialize(case)
            return
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if module == pipeline.CONSTANT_MODULE:
            self.events.append("constant")
            (output / "constant_action_50_summary.json").write_text(json.dumps({
                "source": {"kind": "trained_checkpoint"},
            }))
            for name in (
                "fig00_constant_action_50.png", "fig00_constant_action_50.svg",
                "constant_action_50_embeddings.npz", "constant_action_50_positions.csv",
            ):
                (output / name).write_bytes(b"synthetic output fixture")
        elif module == pipeline.EVIDENCE_MODULE:
            self.events.append("evidence")
            # Deliberately no positive conclusion: the orchestration must not demand one.
            for name in ("offline_probe.json", "time_alignment_evidence.json"):
                (output / name).write_text(json.dumps({"assessment": "no measured advantage"}))
            for name in ("fig04_time_alignment_evidence.png", "fig04_time_alignment_evidence.svg"):
                (output / name).write_bytes(b"synthetic output fixture")
        elif module == pipeline.FULL_SAMPLING_MODULE:
            self.events.append("full_sampling")
            # A completed experiment is valid even when time alignment has no gain.
            (output / "full_sampling_evidence.json").write_text(json.dumps({
                "assessment": "no measured advantage",
            }))
            for name in (
                "full_sampling_arrays.npz", "fig05_full_sampling_time_alignment.png",
                "fig05_full_sampling_time_alignment.svg",
            ):
                (output / name).write_bytes(b"synthetic output fixture")
        else:
            raise AssertionError(f"Unexpected child module: {module}")


def test_dry_run_works_without_site_packages_saved_plan_or_files(tmp_path):
    script = Path(pipeline.__file__)
    output = tmp_path / "must-not-be-created"
    completed = subprocess.run([
        sys.executable, "-S", str(script),
        "--directory", str(tmp_path / "missing-plan"),
        "--output", str(output),
        "--output-root", str(tmp_path / "missing-models"),
        "--repeat", "2", "--gpus", "1,3", "--dry-run",
    ], text=True, capture_output=True, check=True)
    assert not output.exists()
    assert not (tmp_path / "missing-models").exists()
    text = completed.stdout
    assert text.index("[TRAIN rope]") < text.index("[EVALUATE rope IMMEDIATELY]")
    assert text.index("[EVALUATE rope IMMEDIATELY]") < text.index("[TRAIN time_input]")
    assert "--repeat 2" in text
    assert "--gpu 1" in text and "--gpu 3" in text
    assert "1000 updates, 2 DDP GPUs, batch32/GPU" in text
    assert "No new calibration" in text
    assert text.index("[ONE-STEP FLOW") < text.index("[FULL DENOISING:")
    assert text.index("[FULL DENOISING:") < text.index("[COMPLETE ONLY AFTER")
    sampling_line = next(line for line in text.splitlines() if pipeline.FULL_SAMPLING_MODULE in line)
    assert "--executed-steps 10" in sampling_line
    assert "--noise-seeds 3000,4000,5000" in sampling_line
    assert "--flow-times" not in sampling_line


def test_each_model_is_tested_immediately_then_paired_probe_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    args = arguments(tmp_path)
    backend = FakeBackend(args)
    result = pipeline.run_pipeline(args, backend=backend, command_runner=backend.command_runner)
    events = backend.events
    rope_eval = [i for i, event in enumerate(events) if event.startswith("eval:rope:")]
    time_eval = [i for i, event in enumerate(events) if event.startswith("eval:time_input:")]
    assert len(rope_eval) == len(time_eval) == 6
    assert events.index("train:rope") < min(rope_eval)
    assert max(rope_eval) < events.index("train:time_input") < min(time_eval)
    assert max(time_eval) < events.index("audit:rope,time_input")
    assert events.index("audit:rope,time_input") < events.index("constant") < events.index("evidence")
    assert events.index("evidence") < events.index("full_sampling")
    assert result["status"] == "completed"
    report = json.loads((args.output / "comparison-fixed_protocol.json").read_text())
    assert report["protocol"] == "pi05-task7-fixed-protocol-v1"
    assert report["stage"] == "fixed_protocol"
    assert report["cases"] == ["rope", "time_input"]
    assert report["settings"]["training_repeat"] == 3
    assert report["settings"]["fixed_init_states"] == list(range(10))
    # RoPE is 20%, outside the old calibration band, and still reaches all tests.
    assert report["summary"]["aggregate"]["rope"] == [12, 60]
    assert "lock" not in report
    evidence = next(command for command in backend.commands if pipeline.EVIDENCE_MODULE in command)
    assert evidence[evidence.index("--repeat") + 1] == "3"
    assert evidence[evidence.index("--gpu") + 1] == "1"
    assert "--flow-times" in evidence and "--noise-seeds" in evidence
    assert evidence[evidence.index("--rope-checkpoint") + 1] == str(backend.jobs[0]["checkpoint"])
    assert evidence[evidence.index("--time-input-checkpoint") + 1] == str(backend.jobs[1]["checkpoint"])
    sampling = next(command for command in backend.commands if pipeline.FULL_SAMPLING_MODULE in command)
    assert sampling[sampling.index("--time-input-checkpoint") + 1] == str(backend.jobs[1]["checkpoint"])
    for option, expected in (
        ("--repeat", "3"), ("--gpu", "1"), ("--executed-steps", "10"),
        ("--noise-seeds", "3000,4000,5000"), ("--probe-demos", "8"),
        ("--windows-per-demo", "3"), ("--tokenizer", str(args.tokenizer)),
        ("--output", str(args.output / "full_sampling")),
    ):
        assert sampling[sampling.index(option) + 1] == expected
    assert "--rope-checkpoint" not in sampling and "--flow-times" not in sampling
    stage = result["stages"]["full_sampling"]
    assert stage["status"] == "completed"
    assert stage["conditions"] == ["correct", "zero", "reversed"]
    assert stage["executed_steps"] == 10
    assert stage["command"] == sampling
    assert len(result["full_sampling_outputs"]) == 4


def test_training_failure_stops_before_any_evaluation_and_records_failure(tmp_path):
    args = arguments(tmp_path)
    backend = FakeBackend(args)

    def fail(command, log, env):
        raise subprocess.CalledProcessError(9, command)

    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(args, backend=backend, command_runner=fail)
    assert not any(event.startswith("eval:") for event in backend.events)
    assert not (args.output / "comparison-fixed_protocol.json").exists()
    state = json.loads((args.output / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["stages"]["train:rope"]["status"] == "failed"
    assert "train:time_input" not in state["stages"]


def test_evaluation_failure_stops_before_next_model_and_no_partial_report(tmp_path):
    args = arguments(tmp_path)
    backend = FakeBackend(args)
    backend.fail_eval = ("rope", args.scene_seeds[0], args.test_policy_seeds[0])
    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        pipeline.run_pipeline(args, backend=backend, command_runner=backend.command_runner)
    assert "train:time_input" not in backend.events
    assert "constant" not in backend.events
    assert not (args.output / "comparison-fixed_protocol.json").exists()
    state = json.loads((args.output / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["stages"]["evaluate:rope"]["status"] == "failed"


def test_existing_run_requires_explicit_reuse_before_any_work(tmp_path):
    args = arguments(tmp_path)
    backend = FakeBackend(args)
    backend.materialize("rope")
    with pytest.raises(FileExistsError, match="--reuse-existing"):
        pipeline.run_pipeline(args, backend=backend, command_runner=backend.command_runner)
    assert not backend.events
    assert not args.output.exists()


def test_explicit_skip_training_audits_both_models_and_runs_all_diagnostics(tmp_path):
    args = arguments(tmp_path, "--skip-training")
    backend = FakeBackend(args)
    for case in pipeline.CASES:
        backend.materialize(case)
    result = pipeline.run_pipeline(args, backend=backend, command_runner=backend.command_runner)
    assert result["status"] == "completed"
    assert not any(event.startswith("train:") for event in backend.events)
    assert "audit:rope" in backend.events and "audit:time_input" in backend.events
    assert "evidence" in backend.events
    assert "full_sampling" in backend.events
    assert result["stages"]["train:rope"]["action"] == "reuse_existing"


def test_reuse_never_overwrites_an_incomplete_training_run(tmp_path):
    args = arguments(tmp_path, "--reuse-existing")
    backend = FakeBackend(args)
    backend.jobs[1]["run"].mkdir(parents=True)
    with pytest.raises(ValueError, match="incomplete"):
        pipeline.run_pipeline(args, backend=backend, command_runner=backend.command_runner)
    assert not backend.events
    assert not args.output.exists()
    assert list(backend.jobs[1]["run"].iterdir()) == []


def test_skip_training_rejects_missing_model_without_training(tmp_path):
    args = arguments(tmp_path, "--skip-training")
    backend = FakeBackend(args)
    backend.materialize("rope")
    with pytest.raises(FileNotFoundError, match="--skip-training"):
        pipeline.run_pipeline(args, backend=backend, command_runner=backend.command_runner)
    assert not backend.events


def test_fixed_settings_cannot_be_replaced_in_the_same_output(tmp_path):
    args = arguments(tmp_path)
    backend = FakeBackend(args)
    pipeline.run_pipeline(args, backend=backend, command_runner=backend.command_runner)
    state_path = args.output / "pipeline_state.json"
    original_state = state_path.read_bytes()
    args.reuse_existing = True
    args.scene_seeds = (999,)
    before = list(backend.events)
    with pytest.raises(ValueError, match="Saved fixed protocol changed"):
        pipeline.run_pipeline(args, backend=backend, command_runner=backend.command_runner)
    assert backend.events == before
    assert state_path.read_bytes() == original_state


def test_child_success_without_evidence_files_cannot_mark_pipeline_complete(tmp_path):
    args = arguments(tmp_path)
    backend = FakeBackend(args)

    def missing_evidence(command, log, env):
        if pipeline.EVIDENCE_MODULE not in command:
            backend.command_runner(command, log, env)

    with pytest.raises(RuntimeError, match="required artifacts"):
        pipeline.run_pipeline(args, backend=backend, command_runner=missing_evidence)
    state = json.loads((args.output / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["stages"]["time_evidence"]["status"] == "failed"
    assert "full_sampling" not in state["stages"]


@pytest.mark.parametrize("missing_name", (
    "full_sampling_evidence.json", "full_sampling_arrays.npz",
    "fig05_full_sampling_time_alignment.png", "fig05_full_sampling_time_alignment.svg",
))
def test_every_full_sampling_artifact_is_required_for_completion(tmp_path, missing_name):
    args = arguments(tmp_path)
    backend = FakeBackend(args)

    def missing_sampling_artifact(command, log, env):
        backend.command_runner(command, log, env)
        if pipeline.FULL_SAMPLING_MODULE in command:
            (args.output / "full_sampling" / missing_name).unlink()

    with pytest.raises(RuntimeError, match="required artifacts"):
        pipeline.run_pipeline(args, backend=backend, command_runner=missing_sampling_artifact)
    state = json.loads((args.output / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["stages"]["time_evidence"]["status"] == "completed"
    assert state["stages"]["full_sampling"]["status"] == "failed"
    assert missing_name in state["error"]
    assert "completed_at" not in state


def test_full_sampling_child_failure_is_recorded_after_old_evidence(tmp_path):
    args = arguments(tmp_path)
    backend = FakeBackend(args)

    def fail_sampling(command, log, env):
        if pipeline.FULL_SAMPLING_MODULE in command:
            raise subprocess.CalledProcessError(8, command)
        backend.command_runner(command, log, env)

    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(args, backend=backend, command_runner=fail_sampling)
    state = json.loads((args.output / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["stages"]["report"]["status"] == "completed"
    assert state["stages"]["time_evidence"]["status"] == "completed"
    assert state["stages"]["full_sampling"]["status"] == "failed"
    assert state["stages"]["full_sampling"]["log"].endswith("full-sampling.log")


def test_old_completed_journal_can_gain_sampling_stage_without_changing_protocol(tmp_path):
    args = arguments(tmp_path, "--skip-training")
    backend = FakeBackend(args)
    for case in pipeline.CASES:
        backend.materialize(case)
    args.output.mkdir(parents=True)
    old_settings = pipeline.protocol_settings(args, backend.record)
    # Pin the v1 persisted schema: adding a diagnostic must not invalidate r1/r2.
    assert set(old_settings) == {
        "cases", "directory", "preparation_sha256", "uniform_plan_sha256",
        "training_episode", "training_repeat", "training_recipe", "source",
        "prompt", "prompt_dir", "tokenizer", "dataset_repo_id", "output_root",
        "control_eval_root", "scene_seeds", "test_policy_seeds", "init_seed",
        "fixed_init_states", "shift_m", "flow_times", "noise_seeds", "probe_demos",
        "windows_per_demo", "selection",
    }
    # This fixture has the old protocol/settings and no full_sampling stage.
    preset = args.output / "fixed_protocol.json"
    preset.write_text(json.dumps(old_settings))
    state_path = args.output / "pipeline_state.json"
    state_path.write_text(json.dumps({
        "protocol": "pi05-task7-train-eval-v1", "settings": old_settings,
        "status": "completed", "stages": {"time_evidence": {"status": "completed", "attempt": 1}},
    }))
    result = pipeline.run_pipeline(args, backend=backend, command_runner=backend.command_runner)
    assert result["status"] == "completed"
    assert result["settings"] == old_settings
    assert json.loads(preset.read_text()) == old_settings
    assert result["stages"]["full_sampling"]["status"] == "completed"
    assert result["stages"]["time_evidence"]["attempt"] == 2
    assert not any(event.startswith("train:") for event in backend.events)


def test_bad_cell_pairing_is_rejected_before_reporting():
    job = {"case": "time_input", "checkpoint": Path("/tmp/explicit-model")}
    row = {
        "case": "time_input", "scene_seed": 10, "policy_seed": 11,
        "episodes": 10, "successes": 10, "per_episode": [True] * 10,
        "checkpoint": str(job["checkpoint"]),
    }
    pipeline.validate_cell(row, job, 10, 11)
    row["policy_seed"] = 12
    with pytest.raises(ValueError, match="frozen"):
        pipeline.validate_cell(row, job, 10, 11)


def test_logged_subprocess_propagates_failure_and_keeps_log(tmp_path):
    log = tmp_path / "child.log"
    command = [sys.executable, "-c", "print('controlled failure'); raise SystemExit(7)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        pipeline.run_logged(command, log, pipeline.process_environment())
    assert error.value.returncode == 7
    assert "controlled failure" in log.read_text()


def test_inherited_gpu_and_intervention_overrides_are_removed(monkeypatch):
    for key in (
        "CUDA_VISIBLE_DEVICES", "LIBERO_TASK7_DUALSEED_SHIFT_CONFIG",
        "LIBERO_FIXED_NOISE_CONFIG", "PI05_CADENCE_PLAN",
    ):
        monkeypatch.setenv(key, "unexpected")
    env = pipeline.process_environment()
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "LIBERO_TASK7_DUALSEED_SHIFT_CONFIG" not in env
    assert "LIBERO_FIXED_NOISE_CONFIG" not in env
    assert "PI05_CADENCE_PLAN" not in env
    assert env["DRY_RUN"] == "false"


def test_two_gpu_recipe_cannot_silently_be_changed(tmp_path):
    with pytest.raises(ValueError, match="exactly two GPUs"):
        arguments(tmp_path, "--gpus", "0,1,2,3")


def test_freezing_failure_is_recorded_in_journal(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    backend = FakeBackend(args)
    original = pipeline.atomic_json

    def fail_preset(path, data):
        if path.name == "fixed_protocol.json":
            raise OSError("synthetic disk error")
        return original(path, data)

    monkeypatch.setattr(pipeline, "atomic_json", fail_preset)
    with pytest.raises(OSError, match="synthetic disk error"):
        pipeline.run_pipeline(args, backend=backend, command_runner=backend.command_runner)
    state = json.loads((args.output / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["stages"]["freeze_protocol"]["status"] == "failed"
    assert not backend.events
