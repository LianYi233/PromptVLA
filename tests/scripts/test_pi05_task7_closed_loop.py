"""Closed-loop orchestration fixtures: no weights, GPU or simulator execution."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.scripts import pi05_task7_closed_loop as closed


def arguments(tmp_path, *extra):
    return closed.parse_args([
        "--directory", str(tmp_path / "plan"), "--output-root", str(tmp_path / "models"),
        "--output", str(tmp_path / "analysis"), "--tokenizer", str(tmp_path / "tokenizer"),
        "--scene-seeds", "13,14", "--test-policy-seeds", "101,102", "--gpus", "1,3", *extra,
    ])


class FakeBackend:
    def __init__(self, args):
        self.args = args
        args.directory.mkdir()
        args.tokenizer.mkdir()
        (args.tokenizer / "tokenizer.json").write_text("SYNTHETIC TOKENIZER FIXTURE")
        for name in ("manifest.json", "uniform.npz"):
            (args.directory / name).write_bytes(b"SYNTHETIC FIXTURE")
        self.job = {"checkpoint": args.output_root / "time-input" / "checkpoints" / "001000" / "pretrained_model"}
        self.commands = []
        self.verified = []

    def prepare(self, args):
        return {}, self.job, {"num_inference_steps": 10}

    def fingerprint(self, checkpoint):
        return [["model.safetensors", 1, 1]]

    def spec(self, args, record, job, mode, scene, policy):
        directory = args.output / "evaluations" / mode / f"scene{scene}-policy{policy}"
        result = directory / "eval_info.json"
        return {
            "mode": mode, "scene_seed": scene, "policy_seed": policy,
            "result": result, "shift_path": directory / "shift.json",
            "shift_config": {"scene": scene, "policy": policy},
            "config_path": directory / "time_intervention_config.json",
            "config": closed.intervention_config(args, job, mode, scene, policy, result),
        }

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
    configs = list(args.output.glob("evaluations/*/*/time_intervention_config.json"))
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
    path = next(args.output.glob("evaluations/zero/*/eval_info.json"))
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
