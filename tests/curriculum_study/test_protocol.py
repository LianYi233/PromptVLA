"""Dependency-light tests: no model downloads, GPU, dataset or simulator needed."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from examples.analysis.curriculum_study import core
from examples.analysis.curriculum_study.analysis import aggregate, paired_summary, _validate_measurement
from examples.analysis.curriculum_study.perturb import feasible_delta, trajectories_match
from examples.analysis.curriculum_study.semantic import probabilities, validate_questions


def fixture():
    n, length = 240, 20
    meta = {k: np.zeros(n, dtype=np.int64) for k in core.META}
    meta.update(index=np.arange(n), frame_index=np.arange(n) % length,
                episode_index=np.arange(n)//length, task_index=np.arange(n)//120)
    scores = {**meta, "score": (np.arange(n) % length).astype(float)}
    info = {"dataset_metadata_sha256": core.arrays_hash(meta), "arrays_sha256": core.arrays_hash(scores)}
    arrays = {**meta, "order": np.arange(n, dtype=np.int64)}
    manifest = {"protocol": core.PLAN_PROTOCOL, "flow_protocol": core.FLOW_PROTOCOL,
                "steps": 60, "batch_size": 2, "world_size": 2, "data_seed": 0, "within_seed": 0,
                "dataset_frames": n, "dataset_metadata_sha256": info["dataset_metadata_sha256"],
                "instance_pool_sha256": core.arrays_hash(meta), "task_schedule_sha256": core.arrays_hash({"task": meta["task_index"]}),
                "arrays_sha256": core.arrays_hash(arrays)}
    return {**arrays, "manifest": manifest}, scores, info


@pytest.mark.parametrize("steps", [80, 100])
@pytest.mark.parametrize("percent", [30, 50, 70])
def test_paired_occurrence_pool_holdout_and_global_rng(tmp_path, steps, percent):
    base, scores, info = fixture()
    snapshot = copy.deepcopy(base)
    state = np.random.get_state()
    plans, probe = core.make_family(base, scores, info, steps=steps, easy_percent=percent,
                                   holdout_per_task=2, probe_per_group=2, warmup_steps=5)
    np.testing.assert_equal(state, np.random.get_state())
    for key in base:
        if key != "manifest":
            np.testing.assert_array_equal(base[key], snapshot[key])
    assert len(probe["heldout_episodes"]) == 4
    assert len(probe["samples"]) == 8
    assert plans["H2E"]["manifest"]["instance_pool_sha256"] == plans["RND"]["manifest"]["instance_pool_sha256"]
    for direction, plan in plans.items():
        assert not set(plan["episode_index"]) & set(probe["heldout_episodes"])
        np.testing.assert_array_equal(plan["order"][:20], np.arange(20))
        np.testing.assert_array_equal(plan["task_index"][plan["order"]], plan["task_index"])
        for task in (0, 1):
            slots = np.flatnonzero((plan["task_index"] == task) & (np.arange(steps*4) >= 20))
            values = scores["score"][plan["index"][plan["order"][slots]]]
            if direction != "RND":
                assert np.all(np.diff(values) <= 0) if direction == "H2E" else np.all(np.diff(values) >= 0)
        path = tmp_path / (direction + ".npz")
        core.save_plan(path, plan)
        loaded = core.load_plan(path)
        assert loaded["manifest"] == plan["manifest"]
        with pytest.raises(FileExistsError):
            core.save_plan(path, plan)
    budgets = plans["H2E"]["manifest"]["study"]["budgets"]
    assert sum(r["low_uses"] for r in budgets) == (steps*4*percent+50)//100
    # Immutable occurrence IDs still map to precisely the same sample after order changes.
    for direction in core.ORDERS:
        for key in core.META:
            np.testing.assert_array_equal(plans[direction][key], plans["RND"][key])


def test_corruption_and_overlap_fail_closed(tmp_path):
    base, scores, info = fixture()
    plans, probe = core.make_family(base, scores, info, steps=80, warmup_steps=5)
    tampered = copy.deepcopy(probe)
    tampered["samples"][0]["group"] = "critical"
    with pytest.raises(ValueError, match="checksum"):
        core.validate_family(plans, tampered)
    plans["H2E"]["episode_index"][0] = probe["heldout_episodes"][0]
    with pytest.raises(ValueError):
        core.validate_family(plans, probe)


@pytest.mark.parametrize("options", [{"steps": 0}, {"warmup_steps": 100}, {"easy_percent": 0},
                                      {"holdout_per_task": 6}, {"probe_per_group": 500}])
def test_invalid_plan_settings(options):
    base, scores, info = fixture()
    args = {"steps": 80, "warmup_steps": 5} | options
    with pytest.raises(ValueError):
        core.make_family(base, scores, info, **args)


def test_independent_labels_are_not_derived_from_scores():
    base, scores, info = fixture()
    labels = {i: "critical" if i % 2 else "routine" for i in range(240)}
    _, probe = core.make_family(base, scores, info, steps=80, warmup_steps=5, independent_labels=labels)
    assert probe["group_definition"] == "independent_labels"
    assert all(row["group"] == labels[row["index"]] for row in probe["samples"])


def test_same_warmup_then_actual_constant_or_cosine():
    kwargs = {"steps": 6000, "warmup": 200, "peak": 2.5e-5}
    c = [core.lr_value(i, **kwargs, end=2.5e-5) for i in range(6001)]
    d = [core.lr_value(i, **kwargs, end=1e-5) for i in range(6001)]
    assert c[:200] == d[:200]
    assert all(v == 2.5e-5 for v in c[200:])
    assert d[-1] == pytest.approx(1e-5)
    assert d[200] > d[3000] > d[6000]
    assert c[0] == 2.5e-5 / 201


def sample(group, task, episode, index, error, count):
    return {"group": group, "task_index": task, "episode_index": episode,
            "sample_index": index, "dataset_index": index, "squared_error_sum": error,
            "valid_coordinates": count, "time": .5}


def test_masked_reduction_and_task_balance():
    rows = [sample("low_score", 0, 0, 0, 10., 10), sample("low_score", 0, 0, 0, 20., 10),
            sample("low_score", 1, 1, 1, 30., 10), sample("high_score", 0, 0, 2, 4., 1),
            sample("high_score", 1, 1, 3, 50., 10)]
    assert aggregate(rows) == {"low_score": 2.25, "high_score": 4.5}
    assert aggregate(rows, weighting="micro")["high_score"] == pytest.approx(54/11)
    with pytest.raises(ValueError, match="same tasks"):
        aggregate(rows[:-1])
    rows[0]["valid_coordinates"] = 0
    with pytest.raises(ValueError):
        aggregate(rows)


def test_incomplete_measurement_cannot_be_plotted():
    rows = [{"sample_index": 0, "task_index": 0, "time": .5,
             "squared_error_sum": 1., "valid_coordinates": 4}]
    _validate_measurement({"samples": rows}, [{"task_index": 0}], [.5])
    with pytest.raises(ValueError):
        _validate_measurement({"samples": rows + rows}, [{"task_index": 0}], [.5])
    with pytest.raises(ValueError):
        _validate_measurement({"samples": rows}, [{"task_index": 0}], [.2, .5])


@pytest.mark.parametrize("action", [np.zeros(7), np.array([1., 1., 1., 0, 0, 0, -1.]),
                                    np.array([-1., .9, -.9, 0, 0, 0, 1.])])
def test_perturbation_fixed_norm_bounded_no_gripper_and_independent_rng(action):
    state = np.random.get_state()
    delta = feasible_delta(action, .05, [0, 1, 2], 7)
    np.testing.assert_equal(state, np.random.get_state())
    assert np.linalg.norm(delta) == pytest.approx(.05)
    assert (np.abs(action + delta) <= 1).all()
    assert (delta[3:] == 0).all()
    np.testing.assert_array_equal(delta, feasible_delta(action, .05, [0, 1, 2], 7))
    assert not feasible_delta(action, 0, [0, 1, 2], 7).any()


@pytest.mark.parametrize("dims,delta", [([6], .1), ([0, 0], .1), ([], .1), ([0], -1), ([0], float("nan"))])
def test_invalid_perturbation_rejected(dims, delta):
    with pytest.raises(ValueError):
        feasible_delta(np.zeros(7), delta, dims, 1)


def pair(trial, clean, pert):
    return {"state_uid": "abc", "trial": trial, "suite": "libero_goal", "task_id": 0,
            "episode": 0, "anchor_step": 20, "clean_success": clean, "zero_success": clean,
            "perturbed_success": pert, "zero_matches": True, "replay_matches": True}


def test_risk_summary_retains_negative_and_floor_values_and_rejects_bad_controls():
    rows = [pair(0, True, False), pair(1, True, True), pair(2, False, True)]
    summary = paired_summary(rows)[0]
    assert summary["control_sensitivity"] == 0
    assert summary["trials"] == 3
    assert paired_summary([pair(0, False, True)])[0]["control_sensitivity"] == -1
    assert paired_summary([pair(0, False, False)])[0]["all_clean_failed"]
    with pytest.raises(ValueError):
        paired_summary(rows + rows[:1])
    rows[0]["zero_matches"] = False
    with pytest.raises(ValueError):
        paired_summary(rows)


def test_zero_control_checks_whole_trajectory_not_only_success():
    a = {"success": True, "terminal": True, "actions": np.zeros((3, 7)), "states": np.zeros((3, 5))}
    b = copy.deepcopy(a)
    assert trajectories_match(a, b)
    b["states"][1, 0] = .1
    assert not trajectories_match(a, b)


def test_semantic_scores_need_independent_ground_truth():
    p = probabilities([-1000, -1001])
    assert np.isclose(p.sum(), 1) and p[0] > p[1]
    rows = [{"state_uid": "a", "question": "Which object is left of the plate?",
             "choices": ["cup", "bowl"], "correct_index": 0}]
    validate_questions(rows)
    rows[0]["correct_index"] = None
    with pytest.raises(ValueError):
        validate_questions(rows)


def test_exclusive_atomic_output_never_overwrites(tmp_path):
    path = tmp_path / "result.json"
    core.write_json(path, {"a": 1}, exclusive=True)
    with pytest.raises(FileExistsError):
        core.write_json(path, {"a": 2}, exclusive=True)
    assert core.read_json(path) == {"a": 1}
