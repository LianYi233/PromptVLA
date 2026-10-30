"""CPU coverage for A-H scope validation, four queues, and complete result accounting."""

import importlib.util
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from lerobot.scripts import libero_robustness_checkpoint as checkpoint
from tests.scripts import test_pi05_vlm_only as fixtures

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "curriculum_pro_eval", ROOT / "examples/evaluation/eval_curriculum_ablation_pro.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def env(tmp_path, **extra):
    return dict(
        os.environ,
        CKPT_ROOT=str(tmp_path / "models"),
        BASE_OUTPUT=str(tmp_path / "results"),
        EVAL_LOG_ROOT=str(tmp_path / "logs"),
        **extra,
    )


def test_jobs_are_distinct_correctly_scoped_and_queued(tmp_path):
    jobs, _ = runner.settings(env(tmp_path, RUN_NAME="stale", TRAIN_SCOPE="wrong", BASE_CKPT="bad"))
    assert [(j["cell"], j["gpu"]) for j in jobs] == list(zip("ABCDEFGH", "01230123", strict=True))
    assert len({j["name"] for j in jobs}) == 8
    assert [j["env"]["TRAIN_SCOPE"] for j in jobs] == ["joint"] * 6 + ["expert_only", "prompt_only"]
    assert all("BASE_CKPT" not in j["env"] and j["env"]["RUN_NAME"] == j["name"] for j in jobs)
    assert all("-r2-libero-" in j["name"] for j in jobs)
    assert len({j["manifest"] for j in jobs}) == 1
    assert len({j["result"] for j in jobs}) == 8
    for bad in (
        {"GPU_IDS": "0,0"},
        {"GPU_IDS": "0,00"},
        {"CELLS": "A A"},
        {"CELLS": "X"},
        {"PRO_DIMENSIONS": "object,object"},
    ):
        with pytest.raises(ValueError):
            runner.settings(env(tmp_path, **bad))


@pytest.mark.parametrize("scope", checkpoint.SCOPES)
def test_actual_saved_scope_and_processors_are_validated(scope):
    fixture = fixtures.VLMOnlyTest()
    fixture.setUp()
    try:
        path = fixture.work / "model"
        fixture.make_model(path, "base", tokens=2)
        config = json.loads((path / "config.json").read_text())
        config.update(
            vlm_prompt_only=scope == "prompt_only",
            train_action_expert=scope != "prompt_only",
            freeze_vlm_prompt=scope == "expert_only",
        )
        (path / "config.json").write_text(json.dumps(config))
        weights = load_file(path / "model.safetensors")
        for prefix in (
            "paligemma_with_expert.gemma_expert.model",
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ):
            weights[f"model.{prefix}.weight"] = np.ones(1, dtype=np.float32)
        save_file(weights, path / "model.safetensors")
        assert checkpoint.check(path, scope)["scope"] == scope
        for other in set(checkpoint.SCOPES) - {scope}:
            with pytest.raises(ValueError):
                checkpoint.check(path, other)
        (path / "policy_preprocessor.json").unlink()
        with pytest.raises((ValueError, FileNotFoundError)):
            checkpoint.check(path, scope)
    finally:
        fixture.doCleanups()


def test_workers_preflight_all_then_use_one_process_per_gpu_and_continue_failure(tmp_path, monkeypatch):
    jobs, batch = runner.settings(env(tmp_path))
    calls, active, maximum = [], {}, [0]
    guard = threading.Lock()

    def fake_call(command, *, env, **kwargs):
        cell = next(j["cell"] for j in jobs if j["name"] == env["RUN_NAME"])
        if env["DRY_RUN"] == "true":
            calls.append(("preflight", cell))
            return SimpleNamespace(returncode=0)
        with guard:
            assert len([c for c in calls if c[0] == "preflight"]) == 8
            gpu = env["GPU_ID"]
            assert gpu not in active
            active[gpu] = cell
            maximum[0] = max(maximum[0], len(active))
            calls.append(("eval", cell))
        time.sleep(0.03)
        with guard:
            del active[gpu]
        return SimpleNamespace(returncode=1 if cell == "A" else 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_call)
    monkeypatch.setattr(runner, "read_result", lambda job: {})
    monkeypatch.setattr(runner, "summary", lambda jobs, batch: True)
    assert not runner.run(jobs, batch)
    assert maximum[0] == 4
    assert len(calls) == 16
    assert calls.index(("eval", "A")) < calls.index(("eval", "E"))
    status = json.loads((batch / "status.json").read_text())
    assert status["A"]["state"] == "failed" and status["E"]["state"] == "completed"
    assert "exit=1" in (batch / "A.progress.log").read_text()


def test_preflight_failure_never_launches_gpu_and_dry_run_is_only_preflight(tmp_path, monkeypatch):
    calls = []

    def fake_call(command, *, env, **kwargs):
        calls.append(env["DRY_RUN"])
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", fake_call)
    jobs, batch = runner.settings(env(tmp_path))
    with pytest.raises(RuntimeError, match="preflight failed"):
        runner.run(jobs, batch)
    assert calls == ["true"]
    calls.clear()

    def success(command, *, env, **kwargs):
        calls.append(env["DRY_RUN"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", success)
    jobs, batch = runner.settings(env(tmp_path, DRY_RUN="true"))
    assert runner.run(jobs, batch) and calls == ["true"] * 8


def write_results(job):
    suites = runner.robust.selected_suites("pro", "all", job["env"]["PRO_DIMENSIONS"])
    tasks = [{"suite": s, "task_id": t, "category": s.rsplit("_", 1)[1]} for s in suites for t in range(10)]
    manifest = {
        "protocol": runner.robust.PROTOCOL,
        "benchmark": "pro",
        "mode": "all",
        "seed": 1000,
        "episodes_per_task": 10,
        "dimensions": job["env"]["PRO_DIMENSIONS"],
        "suites": suites,
        "tasks": tasks,
    }
    job["manifest"].parent.mkdir(parents=True, exist_ok=True)
    job["manifest"].write_text(json.dumps(manifest))
    path = job["result"]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"task_group": t["suite"], "task_id": t["task_id"], "metrics": {"successes": [True] * 9 + [False]}}
        for t in tasks
    ]
    path.write_text(json.dumps({"per_task": rows}))
    args = [
        f"--policy.path={job['checkpoint']}",
        "--seed=1000",
        "--eval.n_episodes=10",
        "--policy.n_action_steps=10",
        "--eval.batch_size=1",
        "--env.control_mode=relative",
        "--policy.use_amp=false",
        "--policy.compile_model=false",
        "--env.observation_height=224",
        "--env.observation_width=224",
        "--env.task=" + ",".join(suites),
    ]
    path.with_name("resume_manifest.json").write_text(
        json.dumps({"arguments": args, "robustness": {"manifest_sha256": runner.robust.json_hash(manifest)}})
    )
    return rows


def test_result_accounting_rejects_partial_duplicate_and_wrong_protocol(tmp_path):
    jobs, batch = runner.settings(env(tmp_path, CELLS="A C"))
    job = jobs[0]
    rows = write_results(job)
    report = runner.read_result(job)
    assert report["overall"] == {"successes": 1440, "episodes": 1600, "success_rate": 90.0}
    assert not report["full_benchmark"]
    assert not runner.summary(jobs, batch)  # C has no result: never display 0% or mark complete.
    assert "未完成" in (batch / "summary.txt").read_text()
    for bad in (rows[:-1], rows + rows[:1]):
        job["result"].write_text(json.dumps({"per_task": bad}))
        with pytest.raises(ValueError):
            runner.read_result(job)
    write_results(job)
    resume = job["result"].with_name("resume_manifest.json")
    saved = json.loads(resume.read_text())
    saved["arguments"][1] = "--seed=42"
    resume.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="settings differ"):
        runner.read_result(job)


def test_launcher_shell_syntax():
    for name in ("run_eval_libero_robustness.sh", "run_eval_libero_curriculum_ablation_pro.sh"):
        subprocess.run(["bash", "-n", str(ROOT / name)], check=True)
