"""Preserve completed B, archive only incomplete C, and refuse active jobs."""

import importlib.util
import json
from pathlib import Path

import pytest

from lerobot.scripts import pi05_curriculum_ablation as suite

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "curriculum_recovery", ROOT / "examples/analysis/recover_curriculum_ablation.py"
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    log = tmp_path / "logs"
    log.mkdir()
    jobs = suite.build_jobs({"OUTPUT_ROOT": str(tmp_path / "runs"), "LOG_ROOT": str(log)}, [])
    (log / "jobs.json").write_text(json.dumps(jobs))
    (log / "status.json").write_text(json.dumps({"B": {"state": "failed"}, "C": {"state": "interrupted"}}))
    for job in jobs[:2]:
        Path(job["run"]).mkdir(parents=True)
    (Path(jobs[0]["run"]) / "keep-model").write_text("B")
    (Path(jobs[1]["run"]) / "keep-trace").write_text("C")
    (log / f"{jobs[1]['name']}.log").write_text("old C training")
    checks = []
    monkeypatch.setattr(suite, "checkpoint_complete", lambda job: checks.append(("checkpoint", job["cell"])))
    monkeypatch.setattr(
        suite, "check_current_runtime", lambda job, _: checks.append(("runtime", job["cell"]))
    )
    monkeypatch.setattr(recovery, "runtime_info", lambda: {})
    return log, jobs, checks


def test_keep_b_archive_c_without_deleting_traces_or_status(setup):
    log, jobs, checks = setup
    before = (log / "status.json").read_bytes()
    report = recovery.recover(log, ["C"])
    assert checks == [("checkpoint", "B"), ("runtime", "B")]
    assert (Path(jobs[0]["run"]) / "keep-model").read_text() == "B"
    assert not Path(jobs[1]["run"]).exists()
    archived = Path(report[0]["archive"])
    assert (archived / "keep-trace").read_text() == "C"
    assert (archived / "previous-training.log").read_text() == "old C training"
    assert (log / "status.json").read_bytes() == before
    assert recovery.recover(log, ["C"]) == []


def test_never_archive_a_final_model_even_if_status_says_interrupted(setup):
    log, jobs, _ = setup
    checkpoint = Path(jobs[1]["run"]) / "checkpoints/006000/pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_text("model")
    with pytest.raises(ValueError, match="Final model files"):
        recovery.recover(log, ["C"])
    assert Path(jobs[1]["run"]).exists()


def test_failed_b_verification_leaves_c_untouched(setup, monkeypatch):
    log, jobs, _ = setup

    def reject(_):
        raise ValueError("B audit failed")

    monkeypatch.setattr(suite, "checkpoint_complete", reject)
    with pytest.raises(ValueError, match="B audit failed"):
        recovery.recover(log, ["C"])
    assert (Path(jobs[1]["run"]) / "keep-trace").exists()


def test_refuse_running_process_group_before_moving_anything(setup, monkeypatch):
    log, jobs, _ = setup
    (log / "status.json").write_text(
        json.dumps({"B": {"state": "failed", "pid": 123}, "C": {"state": "interrupted"}})
    )
    monkeypatch.setattr(recovery.os, "killpg", lambda *_: None)
    with pytest.raises(ValueError, match="process group still exists"):
        recovery.recover(log, ["C"])
    assert Path(jobs[1]["run"]).exists()
