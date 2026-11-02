"""Single-episode, event-biased action-chunk sampling without target resampling."""

import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.scripts import pi05_gripper_event as experiment
from lerobot.utils.pdet_order import (
    PlannedSampler,
    load_plan,
    read_frame_metadata,
    save_plan,
    validate_plan_dataset,
)


def test_closure_detection_is_directional_debounced_and_does_not_count_closed_dwell():
    values = np.r_[np.ones(5), -np.ones(5), [1], -np.ones(4), np.ones(7), [-1], np.ones(3)]
    events, threshold = experiment.closure_events(values)
    assert events.tolist() == [15]
    assert threshold == 0.0
    decreasing, _ = experiment.closure_events(-values, closed_direction="decreasing")
    assert np.array_equal(decreasing, events)
    assert experiment.closure_events(np.ones(10))[0].size == 0
    assert experiment.closure_events(-np.ones(10))[0].size == 0
    assert experiment.closure_events(np.r_[np.zeros(3), np.ones(4)])[0].tolist() == [3]
    with pytest.raises(ValueError, match="explicit"):
        experiment.closure_events([0.1, 0.1, 0.9, 0.9])
    assert experiment.closure_events([0.1, 0.1, 0.9, 0.9], threshold=0.5)[0].tolist() == [2]
    with pytest.raises(ValueError, match="finite"):
        experiment.closure_events([0, np.nan])
    with pytest.raises(ValueError, match="positive integer"):
        experiment.closure_events([0, 1], min_state_frames=0)


def test_task_mapping_uses_exact_long_language_not_mixed_dataset_id():
    tasks = {0: "pick up the alphabet soup and place it in the basket", 28: experiment.LONG_TASKS[0]}
    task = experiment.resolve_task(tasks, 0)
    assert (task["suite"], task["task_id"], task["task_index"]) == ("libero_10", 0, 28)
    tasks[29] = experiment.canonical_language(experiment.LONG_TASKS[0])
    with pytest.raises(ValueError, match="one exact"):
        experiment.resolve_task(tasks, 0)
    with pytest.raises(ValueError, match="one exact"):
        experiment.resolve_task({0: tasks[0]}, 0)
    with pytest.raises(ValueError, match="suite ID"):
        experiment.resolve_task(tasks, 10)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "data").mkdir(parents=True)
    (root / "meta").mkdir()
    n = 480
    actions = np.zeros((n, 7), dtype=np.float32)
    actions[:, -1] = -1
    # The first episode of target long task has no closures; selection must skip
    # it instead of silently falling back to a uniform event plan.
    for start in (20, 70):
        actions[240 + start : 240 + start + 20, -1] = 1
    for start in (15, 60):
        actions[360 + start : 360 + start + 20, -1] = 1
    table = pa.table(
        {
            "index": range(n),
            "episode_index": np.arange(n) // 120,
            "frame_index": np.arange(n) % 120,
            "task_index": np.repeat([0, 28, 28, 29], 120),
            "action": actions.tolist(),
        }
    )
    pq.write_table(table, root / "data/part.parquet")
    pq.write_table(
        pa.table(
            {
                "task_index": [0, 28, 29],
                "task": [
                    "pick up the alphabet soup and place it in the basket",
                    experiment.canonical_language(experiment.LONG_TASKS[0]),
                    experiment.canonical_language(experiment.LONG_TASKS[1]),
                ],
            }
        ),
        root / "meta/tasks.parquet",
    )
    (root / "meta/info.json").write_text(json.dumps({"fps": 20, "total_frames": n}))
    (root / "meta/stats.json").write_text(json.dumps({"action": {"min": [-1] * 7, "max": [1] * 7}}))
    return root, actions


def selection_for(dataset, **kwargs):
    root, actions = dataset
    metadata = read_frame_metadata(root)
    task = experiment.resolve_task(experiment.read_tasks(root))
    return metadata, experiment.select_episode(metadata, actions, task, **kwargs)


def test_selection_preserves_full_original_targets_and_chooses_only_one_long_episode(dataset):
    metadata, (selection, starts, weights, near, contains) = selection_for(dataset)
    assert selection["episode_index"] == 2
    assert selection["task_index"] == 28
    assert selection["closure_frames"] == [20, 70]
    assert starts.tolist() == list(range(240, 311))
    assert selection["num_candidate_anchors"] == 71
    assert near.sum() == 17  # 15..25 and 65..70 (tail starts are excluded).
    assert weights[near].tolist() == [3.0] * 17
    assert (weights[~near] == 1).all()
    assert contains.sum() == 71  # Every complete chunk contains at least one closure.
    targets = starts[:, None] + np.arange(50)
    assert (metadata["episode_index"][targets] == 2).all()
    assert (np.diff(metadata["frame_index"][targets], axis=1) == 1).all()
    with pytest.raises(ValueError, match="does not belong"):
        selection_for(dataset, episode_index=0)
    with pytest.raises(ValueError, match="No eligible"):
        selection_for(dataset, episode_index=1)
    with pytest.raises(ValueError, match="No eligible"):
        selection_for(dataset, window_before=500, window_after=500)


def test_event_only_in_excluded_episode_tail_does_not_create_a_fake_treatment(dataset):
    root, actions = dataset
    actions[240:360, -1] = -1
    actions[350:360, -1] = 1
    metadata = read_frame_metadata(root)
    task = experiment.resolve_task(experiment.read_tasks(root))
    with pytest.raises(ValueError, match="No eligible"):
        experiment.select_episode(metadata, actions, task, episode_index=2)


def test_plans_match_trainer_global_indices_budget_rng_and_probability_ratio(dataset, tmp_path):
    metadata, (selection, indices, weights, near, _) = selection_for(dataset)
    before = torch.get_rng_state().clone(), np.random.get_state()
    outputs = {}
    for sampling in ("uniform", "event"):
        plan, probabilities = experiment.make_plan(
            metadata,
            indices,
            weights,
            selection,
            steps=2000,
            batch_size=2,
            world_size=2,
            data_seed=0,
            sampling=sampling,
        )
        filename = tmp_path / f"{sampling}.npz"
        save_plan(filename, plan)
        loaded = load_plan(filename)
        validate_plan_dataset(loaded, metadata, steps=2000, batch_size=2, world_size=2)
        assert len(plan["index"]) == 8000
        assert set(plan["episode_index"]) == {2}
        assert set(plan["task_index"]) == {28}
        assert set(plan["index"]) == set(indices)
        assert all(probabilities > 0)
        assert len(list(PlannedSampler(loaded, start_step=1999))) == 4
        outputs[sampling] = plan, probabilities
    uniform, pu = outputs["uniform"]
    event, pe = outputs["event"]
    assert np.all(pu == pu[0])
    assert pe[near][0] / pe[~near][0] == pytest.approx(3)
    assert pe[near].sum() > pu[near].sum()
    assert uniform["manifest"]["task_schedule_sha256"] == event["manifest"]["task_schedule_sha256"]
    repeated, _ = experiment.make_plan(
        metadata,
        indices,
        weights,
        selection,
        steps=2000,
        batch_size=2,
        world_size=2,
        data_seed=0,
        sampling="event",
    )
    assert repeated["manifest"] == event["manifest"]
    assert np.array_equal(repeated["index"], event["index"])
    assert torch.equal(before[0], torch.get_rng_state())
    np.testing.assert_equal(before[1], np.random.get_state())


def test_prepare_reuse_check_and_time_variants_share_exact_file(dataset, tmp_path):
    root, _ = dataset
    output = tmp_path / "prepared"
    kwargs = {"steps": 30, "batch_size": 2, "world_size": 2}
    record = experiment.prepare(root, output, **kwargs)
    assert record == experiment.check(output)
    assert record == experiment.prepare(root, output, **kwargs)
    assert record["fps"] == 20
    assert record["sampling_summary"]["event"]["draws"] == 120
    assert (
        record["sampling_summary"]["event"]["event_anchor_probability"]
        > record["sampling_summary"]["uniform"]["event_anchor_probability"]
    )
    for case in experiment.CASES:
        settings = experiment.case_config(case)
        path = output / f"{settings['sampling']}.npz"
        assert path.is_file()
        assert settings["time_encoding"] == case.endswith("_time")
        name = experiment.prefix(record, case)
        assert name == experiment.prefix(output, case)
        assert "task0-ep2-steps30-aexpert" in name
    assert sorted(path.name for path in output.glob("*.npz")) == ["event.npz", "uniform.npz"]
    with pytest.raises(ValueError, match="different inputs"):
        experiment.prepare(root, output, **{**kwargs, "event_weight": 5})
    with pytest.raises(ValueError, match="positive integer"):
        experiment.prefix(record, "event", repeat=0)


def test_check_detects_action_changes_even_when_metadata_and_gripper_unchanged(dataset, tmp_path):
    root, actions = dataset
    output = tmp_path / "prepared"
    experiment.prepare(root, output, steps=4, batch_size=2, world_size=2)
    path = root / "data/part.parquet"
    table = pq.read_table(path)
    actions[242, 0] = 0.75
    table = table.set_column(table.column_names.index("action"), "action", pa.array(actions.tolist()))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="changed after"):
        experiment.check(output)


def test_recomputed_recipe_rejects_a_different_validly_signed_plan(dataset, tmp_path):
    root, _ = dataset
    output = tmp_path / "prepared"
    experiment.prepare(root, output, steps=4, batch_size=2, world_size=2)
    altered = copy.deepcopy(load_plan(output / "event.npz"))
    altered["manifest"]["gripper_event"]["sampling"] = "uniform"
    (output / "event.npz").unlink()
    save_plan(output / "event.npz", altered)
    with pytest.raises(ValueError, match="deterministic sampling recipe"):
        experiment.check(output)


def test_legacy_task_metadata_and_invalid_fps(dataset, tmp_path):
    root, _ = dataset
    tasks = experiment.read_tasks(root)
    (root / "meta/tasks.parquet").unlink()
    (root / "meta/tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": key, "task": value}) for key, value in tasks.items())
    )
    assert experiment.read_tasks(root) == tasks
    (root / "meta/info.json").write_text(json.dumps({"fps": 0, "total_frames": 480}))
    with pytest.raises(ValueError, match="positive FPS"):
        experiment.prepare(root, tmp_path / "prepared", steps=4)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), 1.0])
def test_invalid_event_boost_is_rejected(dataset, bad):
    with pytest.raises(ValueError, match="event_weight"):
        selection_for(dataset, event_weight=bad)


def test_case_config_rejects_unknown_case():
    with pytest.raises(ValueError, match="Unknown case"):
        experiment.case_config("other")
