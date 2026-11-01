"""CPU tests for standard LIBERO isolation, simultaneous queues and result accounting."""

import json
import subprocess
import threading
from types import SimpleNamespace

import pytest

from examples.evaluation import eval_curriculum_seed_repeat as evaluator


def env(tmp_path, **extra):
    return {
        "CKPT_ROOT": str(tmp_path / "models"),
        "LIBERO_OUTPUT_ROOT": str(tmp_path / "libero-results"),
        "LIBERO_EVAL_LOG_ROOT": str(tmp_path / "libero-logs"),
        "TOKENIZER_PATH": str(tmp_path / "tokenizer"),
        **extra,
    }


def write_result(job):
    path = job["result"]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"task_group": suite, "task_id": task, "metrics": {"successes": [True] * 8 + [False] * 2}}
        for suite in evaluator.SUITES
        for task in range(10)
    ]
    path.write_text(json.dumps({"per_task": rows}))
    args = {
        "--policy.path": str(job["checkpoint"]),
        "--env.type": "libero",
        "--env.task": ",".join(evaluator.SUITES),
        "--seed": "1000",
        "--eval.n_episodes": "10",
        "--env.control_mode": "relative",
        "--env.max_parallel_tasks": "1",
        "--eval.batch_size": "1",
        "--policy.n_action_steps": "10",
        "--policy.use_amp": "false",
        "--policy.compile_model": "false",
        "--policy.gradient_checkpointing": "false",
    }
    path.with_name("resume_manifest.json").write_text(
        json.dumps(
            {
                "arguments": [f"{key}={value}" for key, value in args.items()],
                "finetuned_prompt_evaluation": {"tokenizer_path": job["env"]["TOKENIZER_PATH"]},
            }
        )
    )
    return rows


def test_standard_settings_ignore_leftover_pro_paths_and_modes(tmp_path):
    dirty = env(
        tmp_path,
        BASE_OUTPUT="/old/pro-output",
        EVAL_LOG_ROOT="/old/pro-logs",
        LIBERO_ROBUSTNESS_MANIFEST="/old/pro.json",
        LIBERO_BENCHMARK="pro",
        LIBERO_CONFIG_PATH="/old/pro-config",
        BENCHMARK_ROOT="/old/PRO",
        PRO_DIMENSIONS="all",
        RUN_NAME="old-model",
        LEROBOT_EVAL_ATTENTION="1",
        BASE_CKPT="/old/weights",
        LIBERO_PROGRESS_FD="99",
        LIBERO_PROGRESS_LABEL="old",
    )
    jobs, batch, concurrent = evaluator.settings(dirty)
    assert concurrent == 2 and len(jobs) == 6
    assert [(j["cell"], j["gpu"]) for j in jobs] == [
        ("D1", "0"),
        ("D2", "1"),
        ("E1", "2"),
        ("E2", "3"),
        ("F1", "0"),
        ("F2", "1"),
    ]
    assert batch == tmp_path / "libero-logs/libero-batch-seed1000-ep10"
    for job in jobs:
        assert job["result"].is_relative_to(tmp_path / "libero-results/seed1000-ep10")
        assert "-006000-libero-all4-resume" in str(job["result"])
        assert job["env"]["LIBERO_CONFIG_PATH"] == "/home/wyn/.libero"
        assert job["env"]["LEROBOT_EVAL_FINETUNED_PROMPT"] == "1"
        for key in (
            "LIBERO_ROBUSTNESS_MANIFEST",
            "LIBERO_BENCHMARK",
            "PRO_DIMENSIONS",
            "BENCHMARK_ROOT",
            "BASE_CKPT",
            "LEROBOT_EVAL_ATTENTION",
            "LIBERO_PROGRESS_FD",
        ):
            assert key not in job["env"]
        # The single child invocation must resolve the identical model/GPU/result, without double suffixes.
        children, child_batch, _ = evaluator.settings(job["env"])
        assert len(children) == 1 and child_batch == batch
        assert children[0]["checkpoint"] == job["checkpoint"]
        assert children[0]["result"] == job["result"] and children[0]["gpu"] == job["gpu"]


def test_six_simultaneous_evaluations_continue_after_one_failure(tmp_path, monkeypatch):
    jobs, batch, concurrent = evaluator.settings(env(tmp_path))
    barrier = threading.Barrier(6, timeout=4)
    guard = threading.Lock()
    active, max_per_gpu, counts = {}, {}, {"active": 0, "max": 0, "preflight": 0}

    def fake_call(command, *, env, **kwargs):
        assert command == ["bash", str(evaluator.ENTRY), "single"]
        if env["DRY_RUN"] == "true":
            counts["preflight"] += 1
            return SimpleNamespace(returncode=0)
        with guard:
            assert counts["preflight"] == 6
            gpu = env["GPU_ID"]
            active[gpu] = active.get(gpu, 0) + 1
            max_per_gpu[gpu] = max(max_per_gpu.get(gpu, 0), active[gpu])
            counts["active"] += 1
            counts["max"] = max(counts["max"], counts["active"])
        barrier.wait()
        with guard:
            active[gpu] -= 1
            counts["active"] -= 1
        return SimpleNamespace(returncode=1 if env["CELLS"] == "D1" else 0)

    monkeypatch.setattr(evaluator.queue.subprocess, "run", fake_call)
    assert not evaluator.queue.run(
        jobs,
        batch,
        entry=evaluator.ENTRY,
        arguments=("single",),
        reader=lambda job: {},
        reporter=lambda jobs, batch: True,
        jobs_per_gpu=concurrent,
    )
    assert counts["max"] == 6
    assert max_per_gpu == {"0": 2, "1": 2, "2": 1, "3": 1}
    states = json.loads((batch / "status.json").read_text())
    assert states["D1"]["state"] == "failed"
    assert all(states[c]["state"] == "completed" for c in ("D2", "E1", "E2", "F1", "F2"))


def test_standard_summary_rejects_incomplete_and_wrong_protocol(tmp_path):
    jobs, batch, _ = evaluator.settings(env(tmp_path, CELLS="D1 D2"))
    job = jobs[0]
    rows = write_result(job)
    assert evaluator.read_result(job)["overall"] == {"successes": 320, "episodes": 400, "success_rate": 80.0}
    assert not evaluator.summary(jobs, batch)
    assert "未完成" in (batch / "summary.txt").read_text()
    for bad in (rows[:-1], rows + rows[:1]):
        job["result"].write_text(json.dumps({"per_task": bad}))
        with pytest.raises(ValueError):
            evaluator.read_result(job)
    write_result(job)
    path = job["result"].with_name("resume_manifest.json")
    record = json.loads(path.read_text())
    for key, value in (("robustness", {"benchmark": "pro"}), ("finetuned_prompt_evaluation", {})):
        path.write_text(json.dumps(record | {key: value}))
        with pytest.raises(ValueError):
            evaluator.read_result(job)
    write_result(job)
    rows[0]["metrics"]["successes"][0] = 1
    job["result"].write_text(json.dumps({"per_task": rows}))
    with pytest.raises(ValueError, match="nonboolean"):
        evaluator.read_result(job)


def test_single_routes_progress_separately_from_verbose_logs(tmp_path, monkeypatch, capfd):
    from lerobot.scripts import libero_robustness_checkpoint as checkpoint

    jobs, batch, _ = evaluator.settings(env(tmp_path, CELLS="F2", GPU_IDS="3", PROGRESS="true"))
    job = jobs[0]
    (tmp_path / "tokenizer").mkdir()
    write_result(job)
    fake = tmp_path / "run_eval_libero-full.sh"
    fake.write_text(
        'set -eu\n[ "$2" = all ]\n[ "$3" = 3 ]\n[ "$GPU_ID" = 3 ]\n'
        '[ -z "${LIBERO_ROBUSTNESS_MANIFEST:-}" ]\n'
        'printf "episode progress\\n" >&"$LIBERO_PROGRESS_FD"\n'
        'printf "verbose mp4 path\\n"\n'
    )
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(checkpoint, "check", lambda *args: None)
    monkeypatch.setattr(evaluator.seeds, "check_training_identity", lambda *args: None)
    evaluator.single(job, batch)
    captured = capfd.readouterr()
    assert "episode progress" in captured.err
    assert "overall: 80.00% (320/400)" in captured.out
    assert "verbose mp4" not in captured.out + captured.err
    assert "verbose mp4" in (batch / "full/F2.launch.log").read_text()


def test_cpu_dry_run_uses_standard_full_entry(tmp_path, monkeypatch, capfd):
    from lerobot.scripts import libero_robustness_checkpoint as checkpoint

    jobs, batch, _ = evaluator.settings(env(tmp_path, CELLS="E1", DRY_RUN="true"))
    job = jobs[0]
    job["checkpoint"].mkdir(parents=True)
    (tmp_path / "tokenizer").mkdir()
    monkeypatch.setattr(checkpoint, "check", lambda *args: None)
    monkeypatch.setattr(evaluator.seeds, "check_training_identity", lambda *args: None)
    evaluator.single(job, batch)
    text = capfd.readouterr().out
    assert "training_seed=2" in text and "--seed=1000" in text
    assert "--env.task=libero_spatial" in text and "libero_10" in text
    assert "-006000-libero-all4-resume" in text and "-pro-libero" not in text


def test_cli_and_concurrency_validation(tmp_path):
    for value in ("0", "3", "two"):
        with pytest.raises(ValueError, match="JOBS_PER_GPU"):
            evaluator.settings(env(tmp_path, JOBS_PER_GPU=value))
    assert evaluator.settings(env(tmp_path, JOBS_PER_GPU="1"))[2] == 1
    subprocess.run(["bash", "-n", str(evaluator.ENTRY)], check=True)
