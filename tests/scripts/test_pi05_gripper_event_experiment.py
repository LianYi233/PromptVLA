"""Exercise the actual launcher boundary and strict long-only result reporting."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.scripts import pi05_gripper_event as sampling, pi05_gripper_event_experiment as experiment
from lerobot.utils.prompt_initialization import prompt_hash, write_prompt
from tests.scripts import test_pi05_vlm_only as legacy


@pytest.fixture
def prepared(tmp_path):
    fixture = legacy.VLMOnlyTest()
    fixture.setUp()
    try:
        root = fixture.dataset
        (root / "data").mkdir()
        actions = np.zeros((160, 7), dtype=np.float32)
        actions[:, -1] = -1
        actions[100:140, -1] = 1
        pq.write_table(
            pa.table(
                {
                    "index": np.arange(160),
                    "episode_index": np.repeat([0, 1], 80),
                    "task_index": np.repeat([0, 7], 80),
                    "frame_index": np.tile(np.arange(80), 2),
                    "action": actions.tolist(),
                }
            ),
            root / "data/data.parquet",
        )
        pq.write_table(
            pa.table({"task_index": [0, 7], "task": ["another suite task", sampling.LONG_TASKS[8]]}),
            root / "meta/tasks.parquet",
        )
        (root / "meta/info.json").write_text(json.dumps({"fps": 10, "total_frames": 160}))
        source = fixture.source("base")
        config = experiment.read_json(source / "config.json")
        config["chunk_size"] = 50
        (source / "config.json").write_text(json.dumps(config))
        directory = tmp_path / "plan directory"
        sampling.prepare(root, directory, steps=3, batch_size=2, world_size=2)
        prompt = tmp_path / "P1.safetensors"
        values = np.zeros((2, 2048), dtype=np.float32)
        write_prompt(
            prompt,
            values,
            {
                "schema_version": 1,
                "source_step": 0,
                "source_training_seed": 1,
                "source_prompt_init_path": None,
                "num_vlm_prompt_tokens": 2,
                "source_pretrained_path": str(source),
                "prompt_sha256": prompt_hash(values),
            },
        )
        yield fixture, directory, prompt
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize("wandb_mode,repeat", [(None, 1), ("offline", 2), ("disabled", 3)])
def test_actual_training_launcher_dry_run_matches_plans_and_physical_fps(
    prepared, tmp_path, wandb_mode, repeat
):
    fixture, directory, prompt = prepared
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(experiment.ROOT / "src"),
        "GRIPPER_EVENT_DIR": str(directory),
        "PROMPT_INIT_PATH": str(prompt),
        "PRETRAINED_PATH": str(fixture.source("base")),
        "TOKENIZER_PATH": str(fixture.tokenizer),
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "LOG_ROOT": str(tmp_path / "logs"),
        "GPU_IDS": "0,1",
        "DRY_RUN": "true",
        "REPEAT_ID": "1",
        # These inherited values must not change the new experiment.
        "FREEZE_VLM_PROMPT": "true",
        "VLM_PROMPT_TOKENS": "32",
        "TRAIN_ACTION_EXPERT": "false",
    }
    command = ["bash", str(experiment.ROOT / "examples/training/train_pi05_gripper_event.sh"), "all"]
    if wandb_mode is not None:
        # Use the actual recovery CLI, not only WANDB_MODE: the wrapper supplies
        # an explicit online default, which the experiment must override last.
        command.extend(["--wandb-mode", wandb_mode, "--repeat", str(repeat)])
    result = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]
    assert len(commands) == 4
    for i, command in enumerate(commands):
        arguments = command[command.index("lerobot-train") + 1 :]
        # The last override wins, exactly as in the real config parser.
        values = dict(arg.split("=", 1) for arg in arguments)
        assert values["--policy.train_action_expert"] == "true"
        assert values["--policy.vlm_prompt_only"] == "false"
        assert values["--policy.num_vlm_prompt_tokens"] == "2"
        assert "--policy.freeze_vlm_prompt" not in values
        assert values["--policy.action_time_encoding_fps"] == "10.0"
        assert values["--policy.action_time_encoding"] == ("true" if i >= 2 else "false")
        assert values["--steps"] == "3" and values["--batch_size"] == "2"
        assert values["--seed"] == "0"
        assert values["--pdet.prompt_seed"] == "1"
        assert values["--pdet.data_seed"] == "0"
        assert values["--pdet.noise_seed"] == "1"
        assert values["--pdet.time_seed"] == "1"
        assert values["--job_name"].startswith(f"pi05-gripper-long-r{repeat}-")
        assert Path(values["--output_dir"]).name == values["--job_name"]
        assert values["--wandb.enable"] == ("false" if wandb_mode == "disabled" else "true")
        assert values["--wandb.mode"] == ("offline" if wandb_mode == "offline" else "online")
        assert values["--pdet.order_plan"] == str(directory / ("uniform.npz" if i % 2 == 0 else "event.npz"))
        assert values["--keyframe_loss.mode"] == '"off"'
        assert "--dataset.episodes" not in values
    assert not (tmp_path / "outputs").exists()
    assert not (directory / "runs").exists()


def test_build_jobs_shares_plan_for_time_controls_and_has_distinct_outputs(prepared, tmp_path):
    _, directory, _ = prepared
    record = sampling.check(directory)
    jobs = experiment.build_jobs(record, directory, tmp_path / "outputs")
    assert jobs[0]["plan"] == jobs[2]["plan"]
    assert jobs[1]["plan"] == jobs[3]["plan"]
    assert len({job["run"] for job in jobs}) == 4
    assert all("task8-ep1" in job["name"] for job in jobs)
    assert record["selection"]["task_index"] == 7


def eval_payload(success_count=1, episodes=2):
    return {
        "per_task": [
            {
                "task_group": "libero_10",
                "task_id": task,
                "metrics": {"successes": [True] * success_count + [False] * (episodes - success_count)},
            }
            for task in range(10)
        ]
    }


@pytest.mark.parametrize("defect", ("missing", "duplicate", "other_suite", "episodes", "not_boolean"))
def test_reject_incomplete_or_wrong_suite_results(tmp_path, defect):
    info = eval_payload()
    if defect == "missing":
        info["per_task"].pop()
    elif defect == "duplicate":
        info["per_task"][-1] = info["per_task"][0]
    elif defect == "other_suite":
        info["per_task"][0]["task_group"] = "libero_plus"
    elif defect == "episodes":
        info["per_task"][0]["metrics"]["successes"].pop()
    else:
        info["per_task"][0]["metrics"]["successes"][0] = "false"
    path = tmp_path / "eval_info.json"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError):
        experiment.read_eval_result(path, 0, 2)


def test_report_distinguishes_training_task_and_long_overall(prepared, tmp_path):
    _, directory, _ = prepared
    record = sampling.check(directory)
    args = argparse.Namespace(
        eval_root=tmp_path / "eval",
        output_root=tmp_path / "outputs",
        repeat=1,
        eval_seed=1000,
        episodes=2,
    )
    jobs = experiment.build_jobs(record, directory, args.output_root)
    for i, job in enumerate(jobs):
        info = eval_payload(success_count=1)
        info["per_task"][8]["metrics"]["successes"] = [i >= 1, i >= 2]
        path = experiment.result_path(job, record, args)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(info))
    report = experiment.summarize(record, jobs, args)
    assert report["results"]["uniform"]["selected_task_rate"] == 0
    assert report["results"]["uniform"]["overall_rate"] == 45
    assert report["contrasts"]["selected_task_rate"]["event_minus_uniform_pp"] == 50
    assert report["contrasts"]["overall_rate"]["event_minus_uniform_pp"] == 5
    assert (experiment.evaluation_root(args, record) / "summary-all.json").is_file()
    assert (experiment.evaluation_root(args, record) / "summary-all.csv").is_file()


def test_different_preparations_never_overwrite_summary_paths(prepared, tmp_path):
    _, directory, _ = prepared
    record = sampling.check(directory)
    args = argparse.Namespace(eval_root=None, output_root=tmp_path, repeat=1, eval_seed=1000, episodes=10)
    changed = json.loads(json.dumps(record))
    changed["settings"]["event_weight"] = 5.0
    assert experiment.evaluation_root(args, record) != experiment.evaluation_root(args, changed)
