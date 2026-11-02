"""Ensure additive launcher really passes the larger matched budget to LeRobot."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

from lerobot.scripts import pi05_gripper_additive as additive
from lerobot.scripts import pi05_gripper_event_experiment as runner
from tests.scripts.test_pi05_gripper_event_experiment import prepared  # noqa: F401


def test_additive_shell_dry_run_offline_and_four_complete_plans(prepared, tmp_path):
    fixture, old_dir, prompt = prepared
    new_dir = tmp_path / "new additive plan"
    additive.prepare(
        fixture.dataset, new_dir,
        base_plan=old_dir / "uniform.npz",
        task_id=8, episode_index=1, base_steps=3, extra_steps=1,
        batch_size=2, world_size=2,
    )
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(runner.ROOT / "src"),
        "GRIPPER_EVENT_DIR": str(new_dir),
        "PROMPT_INIT_PATH": str(prompt),
        "PRETRAINED_PATH": str(fixture.source("base")),
        "TOKENIZER_PATH": str(fixture.tokenizer),
        "OUTPUT_ROOT": str(tmp_path / "ckpts"),
        "LOG_ROOT": str(tmp_path / "logs"),
        "GPU_IDS": "0,1",
        "DRY_RUN": "true",
    }
    command = [
        "bash", str(runner.ROOT / "examples/training/train_pi05_gripper_additive.sh"),
        "all",
    ]
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]
    assert len(commands) == 4
    for i, command in enumerate(commands):
        values = dict(arg.split("=", 1) for arg in command[command.index("lerobot-train") + 1 :])
        assert values["--steps"] == "4"
        assert values["--batch_size"] == "2"
        assert values["--pdet.audit_steps"] == "4"
        assert values["--pdet.flow_binding"] == "occurrence"
        assert values["--wandb.mode"] == "offline"
        assert values["--wandb.enable"] == "true"
        assert values["--policy.action_time_encoding"] == ("true" if i >= 2 else "false")
        assert values["--policy.action_time_encoding_fps"] == "10.0"
        assert values["--pdet.order_plan"] == str(new_dir / ("uniform.npz" if i % 2 == 0 else "event.npz"))
        assert values["--job_name"].startswith("pi05-gripper-additive-r1-")
    assert not (tmp_path / "ckpts").exists()
    assert not (new_dir / "runs").exists()
