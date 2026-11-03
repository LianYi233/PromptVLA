"""Prevent inherited curriculum paths and occupied GPUs in additive evaluation."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from lerobot.scripts import pi05_gripper_event_experiment as experiment


ROOT = Path(__file__).resolve().parents[2]


def _args(tmp_path):
    return SimpleNamespace(
        output_root=tmp_path / "checkpoints",
        eval_root=None,
        eval_seed=1000,
        episodes=30,
        repeat=1,
        noise_std=0.10,
        noise_seed=20261102,
        noise_scope="task8",
        fixed_eval=True,
    )


def test_additive_result_root_independent_of_old_base_output(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_OUTPUT", "/tmp/eval/curriculum-def-seeds-2-3")
    args = _args(tmp_path)
    record = {
        "protocol": "pi05-gripper-additive-long-v1",
        "selection": {"task_id": 8, "episode_index": 10},
    }
    value = experiment.evaluation_root(args, record)
    assert "curriculum-def-seeds-2-3" not in str(value)
    assert "/gripper-additive/fixed-image-noise/" in str(value)
    assert "r1-seed1000-n30" in str(value)


def test_legacy_root_is_still_distinct(tmp_path):
    args = _args(tmp_path)
    record = {
        "protocol": "pi05-gripper-event-long-v1",
        "selection": {"task_id": 8, "episode_index": 10},
    }
    assert "/gripper-event/" in str(experiment.evaluation_root(args, record))


def test_parallel_preflight_blocks_occupied_gpus_before_any_job(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_smi = binaries / "nvidia-smi"
    fake_smi.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-i\" ]; then echo 512; exit 0; fi\n"
        "exit 1\n"
    )
    fake_smi.chmod(0o755)
    logs = tmp_path / "logs"
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "LOG_ROOT": str(logs),
        "GPU_IDS": "0,1,2,3",
        "DRY_RUN": "false",
        "MIN_FREE_GPU_MIB": "20000",
        "BASE_OUTPUT": "/tmp/eval/curriculum-def-seeds-2-3",
        "EVAL_LOG_ROOT": "/tmp/logs/eval/curriculum-def-seeds-2-3",
    }
    run = subprocess.run(
        ["bash", str(ROOT / "run_eval_libero_gripper_additive_parallel.sh")],
        env=env, text=True, capture_output=True, timeout=15,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    assert "GPU memory preflight failed" in run.stderr
    assert "GPU 0: 512 MiB free" in run.stdout
    assert not logs.exists() or not list(logs.glob("eval-additive-*.log"))


def test_wrappers_ignore_stale_output_settings_and_use_thirty_episode_default():
    for name in (
        "run_eval_libero_gripper_additive.sh",
        "run_eval_libero_gripper_additive_parallel.sh",
    ):
        source = (ROOT / name).read_text()
        assert "unset BASE_OUTPUT EVAL_LOG_ROOT" in source
        assert "GRIPPER_EVAL_DEFAULT_EPISODES" in source
        assert ":-30}" in source
    runner = (ROOT / "src/lerobot/scripts/pi05_gripper_event_experiment.py").read_text()
    assert 'default=os.environ.get("GRIPPER_EVAL_ROOT")' in runner
    assert 'EVAL_LOG_ROOT=str(' in runner
