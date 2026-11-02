"""Regression checks for preserving the original uniform stream while adding event exposure."""

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.scripts import pi05_gripper_additive as additive
from lerobot.scripts import pi05_gripper_event as legacy
from lerobot.scripts import pi05_gripper_event_experiment as runner
from lerobot.utils.pdet_order import load_plan, read_frame_metadata, validate_plan_dataset
from tests.scripts.test_pi05_gripper_event import dataset as sample_dataset  # noqa: F401


@pytest.fixture
def prepared_additive(sample_dataset, tmp_path):
    root, _ = sample_dataset
    old_dir = tmp_path / "old-base"
    old_record = legacy.prepare(
        root, old_dir, task_id=8, episode_index=2, steps=6,
        batch_size=2, world_size=2,
    )
    new_dir = tmp_path / "additive"
    new_record = additive.prepare(
        root, new_dir, base_plan=old_dir / "uniform.npz",
        task_id=8, episode_index=2,
        base_steps=6, extra_steps=3, batch_size=2, world_size=2,
    )
    return root, old_dir, old_record, new_dir, new_record


def test_interleaved_extra_slots_preserve_every_original_base_step():
    assert additive.additive_slots(4, 1).tolist() == [False, False, False, False, True]
    assert additive.additive_slots(4, 2).tolist() == [False, False, True, False, False, True]
    assert additive.additive_slots(2, 5).tolist() == [
        False, True, True, False, True, True, True
    ]
    with pytest.raises(ValueError, match="positive integer"):
        additive.additive_slots(0, 1)


def test_both_cases_retain_exact_base_stream_and_equal_compute(prepared_additive):
    root, old_dir, _, new_dir, record = prepared_additive
    old = load_plan(old_dir / "uniform.npz")
    slots = additive.additive_slots(6, 3)
    base = old["index"][old["order"]]
    uniform = load_plan(new_dir / "uniform.npz")
    event = load_plan(new_dir / "event.npz")
    assert record == additive.check(new_dir)
    assert record["settings"]["steps"] == 9
    assert record["selection"]["episode_index"] == 2
    assert record["selection"]["task_id"] == 8

    for name, plan in (("uniform", uniform), ("event", event)):
        validate_plan_dataset(
            plan, read_frame_metadata(root), steps=9, batch_size=2, world_size=2,
        )
        stream = plan["index"].reshape(9, 4)
        assert np.array_equal(stream[~slots].reshape(-1), base)
        assert len(stream[slots].reshape(-1)) == 12
        assert len(plan["index"]) == 36
        assert set(plan["episode_index"]) == {2}
        assert set(plan["task_index"]) == {28}
        assert plan["manifest"]["additive"]["base_steps"] == 6
        assert plan["manifest"]["additive"]["extra_steps"] == 3
        assert record["sampling_summary"][name]["base_draws_unchanged"] == 24
        assert record["sampling_summary"][name]["extra_draws"] == 12
        assert record["sampling_summary"][name]["total_draws"] == 36
    assert np.array_equal(
        uniform["index"].reshape(9, 4)[~slots], event["index"].reshape(9, 4)[~slots]
    )


def test_only_event_extras_are_closure_starts_and_chunks_stay_contiguous(prepared_additive):
    root, _, _, new_dir, record = prepared_additive
    metadata = read_frame_metadata(root)
    uniform = load_plan(new_dir / "uniform.npz")
    event = load_plan(new_dir / "event.npz")
    slots = additive.additive_slots(6, 3)
    start_events = event["index"].reshape(9, 4)[slots].reshape(-1)
    episode_start = record["selection"]["episode_first_index"]
    closure_frames = record["selection"]["closure_frames"]
    valid_event_frames = {
        f for center in closure_frames
        for f in range(center - 5, center + 6)
        if 0 <= f <= record["selection"]["candidate_last_index"] - episode_start
    }
    assert all(int(metadata["frame_index"][idx]) in valid_event_frames for idx in start_events)
    for plan in (event, uniform):
        starts = plan["index"]
        targets = starts[:, None] + np.arange(50)
        assert np.all(metadata["episode_index"][targets] == 2)
        assert np.all(np.diff(metadata["frame_index"][targets], axis=1) == 1)

    assert record["sampling_summary"]["event"]["extra_near_closure_fraction"] == 1.0
    assert record["sampling_summary"]["event"]["total_near_closure_fraction"] > (
        record["sampling_summary"]["uniform"]["total_near_closure_fraction"]
    )


def test_four_time_variants_reuse_two_plans_and_use_new_run_names(prepared_additive, tmp_path):
    _, _, _, new_dir, record = prepared_additive
    jobs = runner.build_jobs(record, new_dir, tmp_path / "outputs")
    assert len(jobs) == 4
    assert jobs[0]["plan"] == jobs[2]["plan"]
    assert jobs[1]["plan"] == jobs[3]["plan"]
    assert all("-task8-ep2-base6-extra3-steps9-" in job["name"] for job in jobs)
    assert all("gripper-additive" in job["name"] for job in jobs)
    assert all(job["checkpoint"].parts[-2:] == ("000009", "pretrained_model") for job in jobs)
    assert runner.preparation_module(record) is additive
    assert runner.preparation_module({"protocol": legacy.PROTOCOL}) is legacy


def test_reject_changed_base_plan_and_reuse_directory_safely(prepared_additive, tmp_path):
    root, old_dir, _, new_dir, record = prepared_additive
    assert additive.prepare(
        root, new_dir, base_plan=old_dir / "uniform.npz", task_id=8,
        episode_index=2, base_steps=6, extra_steps=3, batch_size=2, world_size=2,
    ) == record
    with pytest.raises(ValueError, match="different"):
        additive.prepare(
            root, new_dir, base_plan=old_dir / "uniform.npz", task_id=8,
            episode_index=2, base_steps=6, extra_steps=4,
            batch_size=2, world_size=2,
        )
    wrong_base = tmp_path / "wrong-base.npz"
    wrong_base.write_bytes((old_dir / "event.npz").read_bytes())
    with pytest.raises(ValueError, match="uniform.npz"):
        additive.prepare(
            root, tmp_path / "bad", base_plan=wrong_base, task_id=8,
            episode_index=2, base_steps=6, extra_steps=3,
            batch_size=2, world_size=2,
        )
    with (old_dir / "uniform.npz").open("ab") as handle:
        handle.write(b"trailing content changes source checksum")
    with pytest.raises(ValueError, match="changed"):
        additive.check(new_dir)


def test_manifest_contains_unmodified_source_provenance(prepared_additive):
    _, _, old_record, new_dir, record = prepared_additive
    old_selection = old_record["selection"]
    assert old_selection["episode_index"] == record["selection"]["episode_index"]
    assert old_record["settings"]["steps"] == record["settings"]["base_steps"]
    assert record["original_uniform_plan"]["original_steps"] == 6
    assert (new_dir / "anchors.csv").is_file()
    assert json.loads((new_dir / "manifest.json").read_text()) == record
