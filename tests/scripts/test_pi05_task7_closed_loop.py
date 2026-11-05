"""Closed-loop orchestration fixtures: no weights, GPU or simulator execution."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.scripts import pi05_task7_closed_loop as closed
from lerobot.scripts import pi05_task7_timepath_experiment as timepath


def arguments(tmp_path, *extra):
    return closed.parse_args([
        "--directory", str(tmp_path / "plan"), "--output-root", str(tmp_path / "models"),
        "--output", str(tmp_path / "analysis"), "--tokenizer", str(tmp_path / "tokenizer"),
        "--scene-seeds", "13,14", "--test-policy-seeds", "101,102", "--gpus", "1,3", *extra,
    ])


class FakeBackend:
    def __init__(self, args):
        self.args = args
        self.timepath = timepath
        self.record = {"selection": {"episode_index": 7}}
        args.directory.mkdir()
        args.tokenizer.mkdir()
        (args.tokenizer / "tokenizer.json").write_text("SYNTHETIC TOKENIZER FIXTURE")
        for name in ("manifest.json", "uniform.npz"):
            (args.directory / name).write_bytes(b"SYNTHETIC FIXTURE")
        self.job = {
            "name": "fixture-time-input",
            "checkpoint": args.output_root / "time-input" / "checkpoints" / "001000" / "pretrained_model",
        }
        self.commands = []
        self.verified = []

    def prepare(self, args):
        return self.record, self.job, {"num_inference_steps": 10}

    def fingerprint(self, checkpoint):
        return [["model.safetensors", 1, 1]]

    def spec(self, args, record, job, mode, scene, policy):
        # Use the native directory contract: the shared shift config and its
        # lock live one level above the model's result/.resume.lock directory.
        return closed.Backend.spec(self, args, record, job, mode, scene, policy)

    def command_runner(self, command, log, env):
        config_path = Path(env["LIBERO_TASK7_TIME_INTERVENTION_CONFIG"])
        config = json.loads(config_path.read_text())
        assert command[command.index("eval") + 1] == "time_input"
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert "LIBERO_TASK7_DUALSEED_SHIFT_CONFIG" not in env
        assert env["GPU_ID"] in ("1", "3")
        self.commands.append(config["intervention"])
        (config_path.parent / "eval_info.json").write_text(json.dumps({
            "intervention": config["intervention"], "successes": [config["intervention"] == "zero"] * 10,
        }))

    def verify(self, args, record, job, spec):
        value = json.loads(spec["result"].read_text())
        if value["intervention"] != spec["mode"]:
            raise ValueError("wrong intervention cache")
        self.verified.append((spec["mode"], spec["scene_seed"], spec["policy_seed"]))
        return {"successes": value["successes"], "traces": [f"synthetic-{i}" for i in range(10)],
                "eval_info": str(spec["result"])}


def report_fixture(path, output):
    data = json.loads(path.read_text())
    assert data["settings"]["primary_contrast"] == "correct_vs_zero"
    assert len(data["cells"]) == 4
    # A negative result is a valid completed experiment.
    assert all(not cell["results"]["correct"]["successes"][0] for cell in data["cells"])
    assert all(cell["results"]["zero"]["successes"][0] for cell in data["cells"])


@pytest.fixture
def isolated_hashes(monkeypatch):
    monkeypatch.setattr(closed, "source_hashes", lambda: {"synthetic-source": "fixture-only"})


def test_dry_run_needs_no_site_packages_files_or_gpus(tmp_path):
    output = tmp_path / "absent"
    result = subprocess.run([
        sys.executable, "-S", str(Path(closed.__file__)), "--dry-run", "--output", str(output),
    ], text=True, capture_output=True, check=True)
    assert not output.exists()
    assert "180 total" in result.stdout
    assert "later observations may differ" in result.stdout
    assert result.stdout.count("-m lerobot.scripts.pi05_task7_timepath_experiment eval time_input") == 18


def test_interventions_have_distinct_caches_and_native_rng_protocol(tmp_path, isolated_hashes):
    args = arguments(tmp_path)
    backend = FakeBackend(args)
    state = closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert state["status"] == "completed"
    assert len(backend.commands) == 12
    assert len(set(backend.verified)) == 12
    report = json.loads((args.output / "comparison-time-interventions.json").read_text())
    assert report["protocol"] == closed.COMPARISON_PROTOCOL
    assert report["settings"]["fixed_init_states"] == list(range(10))
    assert report["settings"]["executed_steps"] == 10
    assert report["settings"]["horizon"] == 50
    configs = list((args.output / "evaluations").rglob("time_intervention_config.json"))
    assert len(configs) == 12
    assert len({json.loads(path.read_text())["trace_directory"] for path in configs}) == 12


def test_resume_checks_every_cache_without_reexecuting(tmp_path, isolated_hashes):
    args = arguments(tmp_path)
    backend = FakeBackend(args)
    for _ in range(2):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert len(backend.commands) == 12
    assert len(backend.verified) == 48  # Before reuse and again before each final comparison.


def test_wrong_intervention_cached_outcomes_fail_instead_of_being_reused(tmp_path, isolated_hashes):
    args = arguments(tmp_path)
    backend = FakeBackend(args)
    closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    path = next((args.output / "evaluations" / "zero").rglob("eval_info.json"))
    value = json.loads(path.read_text())
    value["intervention"] = "correct"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="wrong intervention cache"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert len(backend.commands) == 12
    assert json.loads((args.output / "closed_loop_state.json").read_text())["status"] == "failed"


def test_frozen_scene_protocol_is_not_overwritten(tmp_path, isolated_hashes):
    args = arguments(tmp_path)
    backend = FakeBackend(args)
    closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    frozen = (args.output / "closed_loop_protocol.json").read_bytes()
    args.shift_m = 0.1
    with pytest.raises(ValueError, match="Frozen closed-loop settings changed"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert (args.output / "closed_loop_protocol.json").read_bytes() == frozen
    assert len(backend.commands) == 12


def test_failed_child_is_not_scored_as_failure_episode(tmp_path, isolated_hashes):
    args = arguments(tmp_path)
    backend = FakeBackend(args)

    def fail(command, log, env):
        raise subprocess.CalledProcessError(7, command)

    with pytest.raises(subprocess.CalledProcessError):
        closed.run(args, backend=backend, command_runner=fail, reporter=report_fixture)
    assert not (args.output / "comparison-time-interventions.json").exists()
    state = json.loads((args.output / "closed_loop_state.json").read_text())
    assert state["status"] == "failed"
    assert state["completed_cells"] == []


@pytest.fixture
def empty_failed_attempt(tmp_path, isolated_hashes):
    args = arguments(tmp_path)
    backend = FakeBackend(args)

    def fail(command, log, env):
        directory = Path(env["LIBERO_TASK7_TIME_INTERVENTION_CONFIG"]).parent
        config = json.loads((directory / "time_intervention_config.json").read_text())
        spec = backend.spec(args, backend.record, backend.job, config["intervention"],
                            config["scene_seed"], config["policy_seed"])
        # The native timepath launcher always uses this real shared writer,
        # even for object shifts with image noise disabled. Its historical
        # lock filename must survive a failure before the first policy action.
        timepath.previous.ensure_noise_config(spec["shift_path"], spec["shift_config"])
        assert spec["shift_path"].parent == directory.parent
        (directory / "resume_manifest.json").write_text(json.dumps({"source": "before-fix"}))
        (directory / "resume_summary.json").write_text(json.dumps({
            "per_suite": {"libero_10": {"saved_episodes": 0, "successes": 0, "pc_success_saved": None}},
            "saved_episodes": 0, "successes": 0, "pc_success_saved": None,
        }))
        (directory / ".resume.lock").touch()
        (directory / "task_results").mkdir()
        (directory / "trajectories").mkdir()
        (directory / "videos" / "libero_10_7").mkdir(parents=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Original first-action exception\n")
        native_log_root = Path(command[command.index("--log-root") + 1]) / "eval" / "task7-timepath-qkv"
        native_log_root.mkdir(parents=True)
        (native_log_root / "eval_time-input-001000-libero_10-task7.log").write_text("Native exception\n")
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        closed.run(args, backend=backend, command_runner=fail, reporter=report_fixture)
    return args, backend


def changed_sources(monkeypatch):
    monkeypatch.setattr(closed, "source_hashes", lambda: {"synthetic-source": "after-fix"})


def test_source_fix_archives_empty_failed_attempt_and_preserves_unrelated_files(empty_failed_attempt, monkeypatch):
    import fcntl

    args, backend = empty_failed_attempt
    artifacts = {path.relative_to(args.output): path.read_bytes() for path in args.output.rglob("*")
                 if path.is_file() and path.name not in (".pipeline.lock", ".resume.lock", ".fixed_image_noise.lock")}
    locks = {path: path.stat().st_ino for path in args.output.rglob("*.lock")}
    assert {path.name for path in locks} == {".pipeline.lock", ".resume.lock", ".fixed_image_noise.lock"}
    for path in locks:
        if path.name == ".fixed_image_noise.lock":
            assert (path.parent / "fixed_object_shift_config.json").is_file()
            assert not (path.parent / ".resume.lock").exists()
    unrelated = [args.output / "notes.txt", args.output / "logs" / "other-experiment.log"]
    for path in unrelated:
        path.write_text("Keep unrelated user work\n")
    changed_sources(monkeypatch)

    def evaluate_with_lock_check(command, log, env):
        with (args.output / ".pipeline.lock").open("a") as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backend.command_runner(command, log, env)

    state = closed.run(args, backend=backend, command_runner=evaluate_with_lock_check, reporter=report_fixture)
    archive = Path(state["recovered_empty_attempt"])
    assert archive.parent == args.output / "empty_failed_attempts"
    assert state["status"] == "completed"
    assert len(backend.commands) == 12
    for relative, content in artifacts.items():
        assert (archive / relative).read_bytes() == content
    assert all(path.read_text() == "Keep unrelated user work\n" for path in unrelated)
    assert all(path.stat().st_ino == inode for path, inode in locks.items())
    assert not list(archive.rglob("*.lock"))
    assert not list((args.output / "evaluations").rglob("resume_manifest.json"))
    assert state["settings"]["source_sha256"] == {"synthetic-source": "after-fix"}
    assert json.loads((archive / "closed_loop_state.json").read_text())["status"] == "failed"
    assert set(json.loads((archive / "recovery.json").read_text())["files"]) == {str(path) for path in artifacts}
    # Recovery does not weaken the usual verified-cache reuse on a subsequent run.
    closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert len(backend.commands) == 12
    assert len(list((args.output / "empty_failed_attempts").iterdir())) == 1


@pytest.mark.parametrize("evidence", [
    "eval_info.json",
    "task_results/libero_10_7.json",
    "trajectories/episode-000.json",
    "trajectories/.episode-000.json.partial",
    "trajectories/interrupted-write.bin",
    "videos/libero_10_7/eval_episode_0.mp4",
    "recordings/episode_000000.parquet",
    "unknown_partial_result.tmp",
    ".unknown.lock",
    ".fixed_image_noise.lock",  # Legitimate only beside the shared shift config.
])
def test_source_fix_cannot_archive_any_complete_or_partial_episode_evidence(empty_failed_attempt, monkeypatch, evidence):
    args, backend = empty_failed_attempt
    cell = next((args.output / "evaluations").rglob("resume_manifest.json")).parent
    path = cell / evidence
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"successes": [False], "partial": True}))
    before = {path: path.read_bytes() for path in args.output.rglob("*") if path.is_file()}
    changed_sources(monkeypatch)
    with pytest.raises(ValueError, match="episode/task outputs, traces.*separate --output"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not (args.output / "empty_failed_attempts").exists()
    assert backend.commands == []


@pytest.mark.parametrize("scope", ["overall", "suite", "invalid"])
def test_source_fix_rejects_nonempty_or_unverifiable_native_progress(empty_failed_attempt, monkeypatch, scope):
    args, backend = empty_failed_attempt
    path = next((args.output / "evaluations").rglob("resume_summary.json"))
    summary = json.loads(path.read_text())
    if scope == "invalid":
        path.write_text("incomplete JSON")
    else:
        row = summary if scope == "overall" else summary["per_suite"]["libero_10"]
        row["saved_episodes"] = 1  # A completed failure still rules out an empty attempt.
        path.write_text(json.dumps(summary))
    changed_sources(monkeypatch)
    with pytest.raises(ValueError, match="saved episodes or unverifiable progress.*separate --output"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert not (args.output / "empty_failed_attempts").exists()
    assert backend.commands == []


@pytest.mark.parametrize("change", ["completed_cell", "running", "completed", "mismatched_settings"])
def test_source_fix_requires_matching_failed_state_without_completed_cells(empty_failed_attempt, monkeypatch, change):
    args, backend = empty_failed_attempt
    path = args.output / "closed_loop_state.json"
    state = json.loads(path.read_text())
    if change == "completed_cell":
        state["completed_cells"] = [{"mode": "correct", "scene_seed": 13, "policy_seed": 101}]
    elif change == "mismatched_settings":
        state["settings"]["shift_m"] = 0.1
    else:
        state["status"] = change
    path.write_text(json.dumps(state))
    changed_sources(monkeypatch)
    with pytest.raises(ValueError, match="matching failed state with zero completed cells.*separate --output"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert json.loads(path.read_text()) == state
    assert not (args.output / "empty_failed_attempts").exists()


@pytest.mark.parametrize("change", ["scene", "policy", "init", "shift", "checkpoint", "tokenizer"])
def test_empty_failure_cannot_reset_changed_noncode_protocol(empty_failed_attempt, monkeypatch, change):
    args, backend = empty_failed_attempt
    frozen = (args.output / "closed_loop_protocol.json").read_bytes()
    if change == "scene":
        args.scene_seeds = (999,)
    elif change == "policy":
        args.test_policy_seeds = (999,)
    elif change == "init":
        args.init_seed = 123
    elif change == "shift":
        args.shift_m = 0.1
    elif change == "checkpoint":
        backend.job["checkpoint"] = args.output_root / "different-checkpoint"
    else:
        (args.tokenizer / "tokenizer.json").write_text("Changed tokenizer content")
    changed_sources(monkeypatch)
    with pytest.raises(ValueError, match="identical non-code settings.*separate --output"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert (args.output / "closed_loop_protocol.json").read_bytes() == frozen
    assert not (args.output / "empty_failed_attempts").exists()
    assert backend.commands == []


def test_empty_failure_recovery_keeps_existing_comparison(empty_failed_attempt, monkeypatch):
    args, backend = empty_failed_attempt
    path = args.output / "comparison-time-interventions.json"
    path.write_text(json.dumps({"cells": [{"successes": [False]}]}))
    changed_sources(monkeypatch)
    with pytest.raises(ValueError, match="existing comparison.*separate --output"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert path.exists()
    assert not (args.output / "empty_failed_attempts").exists()


@pytest.mark.parametrize("name", ["fixed_object_shift_config.json", "time_intervention_config.json"])
def test_empty_failure_recovery_rejects_changed_cell_configuration(empty_failed_attempt, monkeypatch, name):
    args, backend = empty_failed_attempt
    path = next((args.output / "evaluations").rglob(name))
    path.write_text(json.dumps({"scene_seed": 999}))
    changed_sources(monkeypatch)
    with pytest.raises(ValueError, match="per-cell configuration changed.*separate --output"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert json.loads(path.read_text()) == {"scene_seed": 999}
    assert not (args.output / "empty_failed_attempts").exists()


def test_empty_failure_archive_error_restores_every_original_file(empty_failed_attempt, monkeypatch):
    args, backend = empty_failed_attempt
    before = {path: path.read_bytes() for path in args.output.rglob("*") if path.is_file()}
    original_rename = Path.rename

    def fail_on_protocol_move(path, target):
        if path == args.output / "closed_loop_protocol.json":
            raise OSError("synthetic archive I/O failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_on_protocol_move)
    changed_sources(monkeypatch)
    with pytest.raises(OSError, match="synthetic archive I/O failure"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not [path for path in (args.output / "empty_failed_attempts").rglob("*") if path.is_file()]
    assert backend.commands == []


@pytest.mark.parametrize("name", [".resume.lock", ".fixed_image_noise.lock"])
def test_empty_failure_recovery_cannot_replace_a_live_native_lock(empty_failed_attempt, monkeypatch, name):
    import fcntl

    args, backend = empty_failed_attempt
    path = next((args.output / "evaluations").rglob(name))
    inode = path.stat().st_ino
    changed_sources(monkeypatch)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="native evaluator is still using.*separate --output"):
            closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert path.stat().st_ino == inode
    assert not (args.output / "empty_failed_attempts").exists()


def test_tokenizer_changed_in_place_invalidates_cached_evaluation(tmp_path, isolated_hashes):
    args = arguments(tmp_path)
    backend = FakeBackend(args)
    closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    (args.tokenizer / "tokenizer.json").write_text("CHANGED TOKENIZER FIXTURE CONTENT")
    with pytest.raises(ValueError, match="Frozen closed-loop settings changed"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=report_fixture)
    assert len(backend.commands) == 12


def test_report_pairing_failure_cannot_mark_state_completed(tmp_path, isolated_hashes):
    args = arguments(tmp_path)
    backend = FakeBackend(args)

    def fail_report(path, output):
        raise ValueError("paired noise differs")

    with pytest.raises(ValueError, match="paired noise differs"):
        closed.run(args, backend=backend, command_runner=backend.command_runner, reporter=fail_report)
    assert json.loads((args.output / "closed_loop_state.json").read_text())["status"] == "failed"


def test_inherited_interventions_cannot_contaminate_new_cell(tmp_path, monkeypatch):
    for key in ("LIBERO_TASK7_TIME_INTERVENTION_CONFIG", "LIBERO_TASK7_DUALSEED_SHIFT_CONFIG", "CUDA_VISIBLE_DEVICES", "LEROBOT_EVAL_ATTENTION"):
        monkeypatch.setenv(key, "unrelated")
    config = tmp_path / "chosen.json"
    env = closed.cell_environment(config, 3)
    assert env["LIBERO_TASK7_TIME_INTERVENTION_CONFIG"] == str(config)
    assert not any(key in env for key in ("LIBERO_TASK7_DUALSEED_SHIFT_CONFIG", "CUDA_VISIBLE_DEVICES", "LEROBOT_EVAL_ATTENTION"))
    assert env["GPU_ID"] == "3"


@pytest.mark.parametrize("extra", [("--gpus", "1,1"), ("--scene-seeds", "13,13"), ("--shift-m", "nan"), ("--repeat", "0")])
def test_bad_evaluation_protocol_is_rejected(tmp_path, extra):
    with pytest.raises(ValueError):
        arguments(tmp_path, *extra)
