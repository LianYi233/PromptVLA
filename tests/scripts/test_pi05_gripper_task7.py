"""Focused task7 experiment: single demo, 1000+200 updates, task7-only clean evaluation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.scripts import pi05_gripper_task7 as task7
from lerobot.scripts import pi05_gripper_event as base
from lerobot.scripts import pi05_gripper_additive as additive
from lerobot.scripts import pi05_gripper_event_experiment as runner
from lerobot.utils.pdet_order import load_plan, read_frame_metadata, validate_plan_dataset


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "task7_dataset"
    (root / "data").mkdir(parents=True)
    (root / "meta").mkdir()
    n = 360
    actions = np.zeros((n, 7), dtype=np.float32)
    actions[:, -1] = -1
    # First task7 episode is ineligible, second has two closure events.
    for center in (20, 70):
        actions[120 + center : 120 + center + 12, -1] = 1
    pq.write_table(
        pa.table(
            {
                "index": np.arange(n),
                "episode_index": np.arange(n) // 120,
                "frame_index": np.arange(n) % 120,
                "task_index": np.repeat([31, 31, 0], 120),
                "action": actions.tolist(),
            }
        ), root / "data" / "data.parquet",
    )
    pq.write_table(
        pa.table({
            "task_index": [0, 31],
            "task": [base.LONG_TASKS[0], base.LONG_TASKS[7]],
        }), root / "meta" / "tasks.parquet",
    )
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 10, "total_frames": n}))
    return root


@pytest.fixture
def prepared(dataset, tmp_path):
    root = tmp_path / "experiment"
    first = task7.prepare(work_dir=root, dataset_root=dataset, batch_size=2, world_size=2)
    return root, first


def test_auto_selects_one_full_task7_episode_instead_of_task8(prepared):
    root, actual = prepared
    assert actual["task_id"] == 7
    assert actual["task"] == base.LONG_TASKS[7]
    assert actual["episode_index"] == 1
    assert actual["episode_length"] == 120
    assert actual["closure_frames"] == [20, 70]
    assert actual["base_steps"] == 1000
    assert actual["extra_steps"] == 200
    assert actual["total_steps"] == 1200
    assert actual["batch_size"] == 2
    assert actual["world_size"] == 2
    assert task7.check(root) == actual


def test_four_training_plans_matched_and_contiguous(dataset, prepared, tmp_path):
    root, actual = prepared
    base_plan = load_plan(root / "base" / "uniform.npz")
    uniform = load_plan(root / "additive" / "uniform.npz")
    event = load_plan(root / "additive" / "event.npz")
    slots = additive.additive_slots(1000, 200)
    metadata = read_frame_metadata(dataset)
    original = base_plan["index"][base_plan["order"]]
    assert original.shape == (4000,)
    assert len(slots) == 1200 and int(slots.sum()) == 200
    for plan in (uniform, event):
        validate_plan_dataset(plan, metadata, steps=1200, batch_size=2, world_size=2)
        stream = plan["index"].reshape(1200, 4)
        assert np.array_equal(stream[~slots].reshape(-1), original)
        assert len(plan["index"]) == 4800
        assert set(plan["episode_index"]) == {1}
        assert set(plan["task_index"]) == {31}
        # No 50-step target crosses an episode boundary.
        target = plan["index"][:, None] + np.arange(50)[None, :]
        assert np.all(metadata["episode_index"][target] == 1)
        assert np.all(np.diff(metadata["frame_index"][target], axis=1) == 1)
    assert actual["sampling_summary"]["event"]["extra_near_closure_fraction"] == 1
    assert actual["sampling_summary"]["uniform"]["extra_draws"] == 800
    assert actual["sampling_summary"]["event"]["extra_draws"] == 800
    jobs = runner.build_jobs(actual_manifest(root), root / "additive", tmp_path / "models")
    assert [j["case"] for j in jobs] == list(runner.CASES)
    assert jobs[0]["plan"] == jobs[2]["plan"]
    assert jobs[1]["plan"] == jobs[3]["plan"]
    assert all("-task7-ep1-base1000-extra200-steps1200-" in j["name"] for j in jobs)
    assert all(j["checkpoint"].parts[-2:] == ("001200", "pretrained_model") for j in jobs)


def actual_manifest(root):
    return additive.check(root / "additive")


def _info(tasks=(7,), n=10):
    return {
        "per_task": [
            {
                "task_group": "libero_10",
                "task_id": task_id,
                "metrics": {"successes": [i % 2 == 0 for i in range(n)]},
            }
            for task_id in tasks
        ]
    }


def test_task7_only_report_never_counts_other_nine_tasks(tmp_path):
    path = tmp_path / "eval_info.json"
    path.write_text(json.dumps(_info()))
    result = runner.read_eval_result(path, 7, 10, only_selected=True)
    assert result["overall_episodes"] == 10
    assert result["overall_successes"] == 5
    assert set(result["per_task"]) == {7}
    with pytest.raises(ValueError, match="coverage mismatch"):
        runner.read_eval_result(path, 7, 10)
    path.write_text(json.dumps(_info(tasks=range(10))))
    with pytest.raises(ValueError, match="coverage mismatch"):
        runner.read_eval_result(path, 7, 10, only_selected=True)
    path.write_text(json.dumps(_info(tasks=(8,))))
    with pytest.raises(ValueError, match="coverage mismatch"):
        runner.read_eval_result(path, 7, 10, only_selected=True)


def test_four_case_summary_is_only_task7_and_clean(prepared, tmp_path):
    root, _ = prepared
    record = actual_manifest(root)
    args = SimpleNamespace(
        eval_root=tmp_path / "eval", output_root=tmp_path / "models",
        repeat=1, eval_seed=1000, episodes=10,
        only_selected_task=True, noise_std=0.0, fixed_eval=False,
        noise_seed=20261102, noise_scope="task8",
    )
    jobs = runner.build_jobs(record, root / "additive", args.output_root)
    for job in jobs:
        output = runner.result_path(job, record, args)
        assert "-libero_10-task7-resume" in str(output)
        assert "only-clean" in str(output)
        assert "fixed-image-noise" not in str(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_info()))
    summary = runner.summarize(record, jobs, args)
    assert summary["evaluated_task_ids"] == [7]
    assert summary["fixed_observation_noise"] is None
    assert all(v["overall_episodes"] == 10 for v in summary["results"].values())
    csv = (runner.evaluation_root(args, record) / "summary-all.csv").read_text()
    header = csv.splitlines()[0].split(",")
    assert header == ["case", "selected_task_rate", "overall_rate", "task_7"]


def test_native_eval_launcher_filters_only_task7_and_no_corruption():
    root = Path(__file__).resolve().parents[2]
    shell = (root / "run_eval_libero-full.sh").read_text()
    assert 'EVAL_ARGS+=(--env.task_ids="[$LIBERO_SELECTED_TASK_ID]")' in shell
    assert 'SUITE_TAG="${SUITE_TAG}-task${LIBERO_SELECTED_TASK_ID}"' in shell
    task7_eval = (root / "run_eval_libero_gripper_task7.sh").read_text()
    assert "--only-selected-task --noise-std 0" in task7_eval
    assert "GRIPPER_EVAL_DEFAULT_EPISODES=10" in task7_eval
    assert "GRIPPER_FIXED_EVAL=false" in task7_eval
    parallel = (root / "run_eval_libero_gripper_task7_2gpu.sh").read_text()
    assert 'run_group "${gpus[0]}" uniform event &' in parallel
    assert 'run_group "${gpus[1]}" uniform_time event_time &' in parallel


def test_task7_prepare_idempotent_and_rejects_wrong_episode(dataset, prepared):
    root, record = prepared
    assert task7.prepare(work_dir=root, dataset_root=dataset, batch_size=2, world_size=2) == record
    with pytest.raises(ValueError, match="does not belong|No eligible"):
        task7.prepare(
            work_dir=root / "wrong", dataset_root=dataset,
            episode_index=0, batch_size=2, world_size=2,
        )


def test_training_launcher_uses_1200_steps_and_saved_additive_plans(prepared, tmp_path):
    root, _ = prepared
    record = actual_manifest(root)
    jobs = runner.build_jobs(record, root / "additive", tmp_path / "models")
    args = SimpleNamespace(
        source=tmp_path / "source",
        tokenizer=tmp_path / "tokenizer",
        dataset_repo_id="libero",
        output_root=tmp_path / "models",
        log_root=tmp_path / "logs",
        gpus="1,3",
        port=29589,
        dry_run=True,
        num_workers=0,
        wandb_mode="offline",
    )
    for job in jobs:
        command, environment = runner.training_command(
            job, record, args, tmp_path / "P1.safetensors"
        )
        assert environment["FLOW_STEPS"] == "1200"
        assert environment["BATCH_SIZE"] == "2"
        assert environment["NUM_PROCESSES"] == "2"
        assert environment["GPU_IDS"] == "1,3"
        assert "--pdet.audit_steps=1200" in command
        assert f"--pdet.order_plan={job['plan']}" in command
        assert f"--policy.action_time_encoding={str(job['time_encoding']).lower()}" in command
        assert command[-2:] == ["--wandb.mode=offline", "--wandb.project=prompt-learning"]
        assert job["checkpoint"].parent.name == "001200"
