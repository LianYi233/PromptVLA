"""Synthetic execution traces exercise reporting only; no real policy or MuJoCo run."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from PIL import Image

from lerobot.scripts import pi05_task7_closed_loop_report as report


def _hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def _state(scene, policy, initial, step=0, mode="correct", *, objects=True):
    offset = {"correct": 0, "zero": .002, "reversed": -.001}[mode] if step else 0
    position = [.4 + initial * .001 + step * .001, .02 + offset * step, .2 + step * .0005]
    return {
        "eef_position": position, "eef_quaternion": [0, 0, 0, 1],
        "eef_rotation_matrix": np.eye(3).tolist(),
        "gripper_qpos": [.03 - step * .0002, -.03 + step * .0002],
        "gripper_qvel": [-.0002, .0002], "sim_time": step / 20,
        "simulator_state_sha256": _hash((scene, policy, initial, step, offset)),
        "controller_state": {"class": "SyntheticController", "use_delta": True,
                             "goal_pos": position, "goal_ori": np.eye(3).tolist()},
        "objects": ({"fixture_cube": {"position": [.42, .04, .10 + max(step - 11, 0) * .001],
                                      "quaternion": [1, 0, 0, 0], "quaternion_order": "wxyz"}}
                    if objects else {}),
        "contact_pairs": [] if objects else None,
    }


def _trace(scene, policy, initial, mode, success, *, steps=13, objects=True):
    chunks = []
    for index in range((steps + 9) // 10):
        actions = (np.arange(350).reshape(50, 7) * .001 + index * .01).tolist()
        chunks.append({
            "chunk_index": index, "start_step": index * 10,
            "initial_noise_sha256": _hash((policy, initial, index, "shared noise")),
            "observation_sha256": _hash((scene, policy, initial, index, "shared" if index == 0 else mode)),
            "denoising_calls": 2, "flow_times": [1., .5], "actions": actions,
            "starts_from_native_pure_noise": True,
        })
    rows = []
    for index in range(steps):
        normalized = chunks[index // 10]["actions"][index % 10]
        action = [*np.multiply(normalized[:6], .1).tolist(), 1. if index < 7 else -1.]
        rows.append({
            "step": index, "chunk_index": index // 10, "chunk_step": index % 10,
            "executed_action": action, "normalized_action": normalized,
            "state_after": _state(scene, policy, initial, index + 1, mode, objects=objects),
            "success": bool(success and index == steps - 1),
            "terminated": index == steps - 1, "truncated": False,
        })
    return {
        "protocol": report.TRACE_PROTOCOL, "status": "completed", "intervention": mode,
        "scene_seed": scene, "policy_seed": policy, "init_state_id": initial,
        "rollout_seed": 3000 + initial, "policy_rng_seed": 9000 + policy + initial,
        "checkpoint": "/synthetic/task7-time-input/checkpoints/001000/pretrained_model",
        "executed_steps": 10, "horizon": 50, "success": bool(success),
        "num_inference_steps": 2,
        "initial_state": _state(scene, policy, initial, objects=objects),
        "steps": rows, "chunks": chunks, "synthetic_fixture": True,
    }


def _comparison(tmp_path, *, scenes=(11,), policies=(1001,), objects=True):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    flags = {
        "correct": [True, True, True, False, False, False, False, False, False, False],
        "zero": [True, False, False, True, False, False, False, False, False, False],
        "reversed": [False, True, False, False, True, True, False, False, False, False],
    }
    cells = []
    for scene in scenes:
        for policy in policies:
            results = {}
            for mode in report.MODES:
                paths = []
                for initial, success in enumerate(flags[mode]):
                    path = tmp_path / f"scene{scene}-policy{policy}-{mode}-{initial}.json"
                    # Real closed-loop interventions may terminate at different times.
                    trace = _trace(scene, policy, initial, mode, success,
                                   steps={"correct": 13, "zero": 21, "reversed": 11}[mode], objects=objects)
                    path.write_text(json.dumps(trace), encoding="utf-8")
                    paths.append(str(path.resolve()))
                results[mode] = {"successes": flags[mode].copy(), "traces": paths,
                                 "eval_info": str(tmp_path / f"scene{scene}-policy{policy}-{mode}-eval_info.json")}
            cells.append({"scene_seed": scene, "policy_seed": policy, "results": results})
    result = {
        "protocol": report.COMPARISON_PROTOCOL, "status": "completed",
        "settings": {
            "scene_seeds": list(scenes), "policy_seeds": list(policies),
            "fixed_init_states": list(range(10)), "init_seed": 3000,
            "checkpoint": "/synthetic/task7-time-input/checkpoints/001000/pretrained_model",
            "executed_steps": 10, "horizon": 50, "num_inference_steps": 2, "synthetic_fixture": True,
        },
        "cells": cells,
    }
    path = tmp_path / "comparison-closed-loop.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path, result


def _modify_trace(comparison, mutator, mode="zero", episode=0):
    path = Path(comparison["cells"][0]["results"][mode]["traces"][episode])
    value = json.loads(path.read_text())
    mutator(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_complete_grid_pairs_outcomes_without_overcounting_inference_seeds(tmp_path):
    path, _ = _comparison(tmp_path, scenes=(12, 11), policies=(1002, 1001))
    loaded = report.load_comparison(path)
    summary = report.summarize_comparison(loaded)
    assert len(loaded["episodes"]) == 40
    assert summary["summary"]["totals"] == {
        "correct": {"successes": 12, "episodes": 40, "success_rate": .3},
        "zero": {"successes": 8, "episodes": 40, "success_rate": .2},
        "reversed": {"successes": 12, "episodes": 40, "success_rate": .3},
    }
    assert summary["summary"]["paired"]["correct_vs_zero"] == {
        "both_success": 4, "both_failure": 24, "correct_only_wins": 8,
        "intervention_only_wins": 4, "net_correct_wins": 4,
        "success_rate_difference_pp": 10.0, "episodes": 40,
    }
    reversed_pair = summary["summary"]["paired"]["correct_vs_reversed"]
    assert reversed_pair["correct_only_wins"] == reversed_pair["intervention_only_wins"] == 8
    assert reversed_pair["net_correct_wins"] == 0
    assert all(row["totals"]["correct"]["episodes"] == 20 for row in summary["by_scene"].values())
    assert all(row["totals"]["correct"]["episodes"] == 20 for row in summary["by_policy_seed"].values())
    assert summary["representative_episode"] == {"scene_seed": 12, "policy_seed": 1002, "init_state_id": 0}
    assert summary["all_episode_page_order"][10] == {"page": 11, "scene_seed": 12, "policy_seed": 1001, "init_state_id": 0}
    assert len(summary["source_trace_hashes"]) == 120
    assert summary["primary_contrast"] == "correct_vs_zero"
    assert all(row["audit"]["matched_shared_chunk_indices"] == 2 for row in summary["episodes"])
    assert all(row["audit"]["later_observations_required_equal"] is False for row in summary["episodes"])


def test_missing_objects_and_optional_orientation_are_explicit_not_inferred(tmp_path):
    path, comparison = _comparison(tmp_path, objects=False)
    for mode in report.MODES:
        for trace_path in comparison["cells"][0]["results"][mode]["traces"]:
            trace = json.loads(Path(trace_path).read_text())
            for state in [trace["initial_state"], *[step["state_after"] for step in trace["steps"]]]:
                state["eef_rotation_matrix"] = None
            Path(trace_path).write_text(json.dumps(trace))
    summary = report.summarize_comparison(report.load_comparison(path))
    for mode in report.MODES:
        info = summary["episodes"][0]["conditions"][mode]
        assert info["objects"] == {}
        assert info["orientation_matrix_samples"] == 0
        assert info["contact_samples_available"] == 0
        assert info["gripper_command_sign_transition_steps"] == [7]
        assert "grasp" in info["note"]


@pytest.mark.parametrize(("mutator", "message"), [
    (lambda value: value.update(status="running"), "completed"),
    (lambda value: value["settings"].update(fixed_init_states=list(range(9))), "fixed initial states"),
    (lambda value: value["settings"].update(executed_steps=50), "execution prefix10"),
    (lambda value: value["cells"].append(copy.deepcopy(value["cells"][0])), "Duplicate"),
    (lambda value: value["settings"].update(policy_seeds=[1001, 1002]), "Incomplete scene"),
    (lambda value: value["cells"][0]["results"].pop("reversed"), "three|Correct"),
    (lambda value: value["cells"][0]["results"]["zero"]["successes"].pop(), "Incomplete official"),
    (lambda value: value["cells"][0]["results"]["zero"]["successes"].__setitem__(0, 1), "Incomplete official"),
])
def test_rejects_partial_duplicate_or_mixed_comparison(tmp_path, mutator, message):
    path, value = _comparison(tmp_path)
    mutator(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        report.load_comparison(path)


@pytest.mark.parametrize(("mutator", "message"), [
    (lambda trace: trace.update(checkpoint="/other-model"), "identity"),
    (lambda trace: trace.update(init_state_id=1), "identity"),
    (lambda trace: trace.update(rollout_seed=3001), "identity"),
    (lambda trace: trace.update(success=False), "Official success"),
    (lambda trace: trace["initial_state"].update(simulator_state_sha256=_hash("different state")), "initial simulator"),
    (lambda trace: trace["chunks"][0].update(observation_sha256=_hash("different image")), "initial observations"),
    (lambda trace: trace.update(policy_rng_seed=22), "policy RNG seeds"),
    (lambda trace: trace["initial_state"]["controller_state"].update(goal_pos=[0., 0., 0.]), "controller_state"),
    (lambda trace: trace["initial_state"].update(gripper_qpos=[.1, -.1]), "gripper_qpos"),
    (lambda trace: trace.update(num_inference_steps=3), "identity"),
    (lambda trace: trace["chunks"][0].update(denoising_calls=3), "denoising count"),
    (lambda trace: trace["chunks"][0].update(starts_from_native_pure_noise=False), "pure-noise"),
    (lambda trace: trace["chunks"][1].update(initial_noise_sha256=_hash("different noise")), "sampling noise"),
    (lambda trace: trace["chunks"].pop(), "chunk count"),
    (lambda trace: trace["chunks"][0].update(flow_times=[.8, .4]), "complete native"),
    (lambda trace: trace["steps"][0].update(chunk_step=1), "aligned"),
    (lambda trace: trace["steps"][0].update(normalized_action=[0.] * 7), "normalized action"),
    (lambda trace: trace["steps"][0].update(terminated=True), "after episode termination"),
    (lambda trace: trace["steps"][-1].update(terminated=False), "native termination"),
    (lambda trace: trace["steps"][0]["state_after"].update(eef_position=[float("nan"), 0, 0]), "Non-finite"),
    (lambda trace: trace.update(source_fingerprints={"model_sha256": _hash("changed source")}), "Source implementation"),
])
def test_rejects_unpaired_or_invalid_execution_trace(tmp_path, mutator, message):
    path, comparison = _comparison(tmp_path)
    _modify_trace(comparison, mutator)
    with pytest.raises(ValueError, match=message):
        report.load_comparison(path)


def test_does_not_require_later_observations_or_episode_lengths_to_match(tmp_path):
    path, _ = _comparison(tmp_path)
    loaded = report.load_comparison(path)
    first = loaded["episodes"][0]
    assert [len(first["traces"][mode]["steps"]) for mode in report.MODES] == [13, 21, 11]
    assert len({first["traces"][mode]["chunks"][1]["observation_sha256"] for mode in report.MODES}) == 3
    assert first["audit"]["initial_observation_equal"] is True
    assert first["audit"]["later_observations_required_equal"] is False


def test_command_sign_transition_skips_zero_plateaus_without_naming_open_or_close():
    values = [1., 0., -1., 0., -1., 1.]
    trace = {"steps": [{"executed_action": [0.] * 6 + [value]} for value in values]}
    assert report.gripper_sign_transitions(trace) == [2, 5]


def test_measured_path_summary_uses_positions_not_action_integration():
    trace = _trace(11, 1001, 0, "correct", True)
    expected = np.linalg.norm([.001, 0, .0005]) * len(trace["steps"])
    for step in trace["steps"]:
        step["executed_action"][:6] = [999.] * 6
    summary = report.summarize_trace(trace)
    assert summary["eef_measured_path_length"] == pytest.approx(expected)
    assert summary["objects"]["fixture_cube"]["max_height_change_from_initial"] == pytest.approx(.002)


def test_cached_controller_orientation_is_identified_in_report_and_figure():
    traces = {mode: _trace(11, 1001, 0, mode, False) for mode in report.MODES}
    cached = "controller.ee_ori_mat; may be cached before the final physics step"
    for state in [traces["zero"]["initial_state"], *[row["state_after"] for row in traces["zero"]["steps"]]]:
        state["eef_rotation_source"] = cached
    summary = report.summarize_trace(traces["zero"])
    assert summary["orientation_matrix_sources"] == {cached: 14}
    figure = report.trajectory_figure({"scene_seed": 11, "policy_seed": 1001, "init_state_id": 0,
                                      "traces": traces}, synthetic=True)
    assert any("controller-cache fallback used" in item.get_text() for item in figure.texts)
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_completed_report_renders_every_prespecified_episode_and_preserves_source(tmp_path):
    path, comparison = _comparison(tmp_path / "inputs")
    source_hashes = {source: _hash(Path(source).read_bytes())
                     for mode in report.MODES for source in comparison["cells"][0]["results"][mode]["traces"]}
    output = tmp_path / "report"
    result = report.run_report(path, output)
    assert result == json.loads((output / "closed_loop_evidence.json").read_text())
    assert result["status"] == "completed"
    assert len(result["all_episode_page_order"]) == 10
    assert len(result["figure_paths"]) == 5
    for filename in report.ARTIFACTS:
        assert (output / filename).is_file()
        assert (output / filename).stat().st_size > 0
    rows = (output / "closed_loop_paired_outcomes.csv").read_text().splitlines()
    assert len(rows) == 11
    with Image.open(output / "fig09_execution_trajectories.png") as image:
        assert image.width >= 1600 and image.height >= 1000
        assert np.asarray(image.convert("RGB")).std() > 20
    with Image.open(output / "fig08_closed_loop_success.png") as image:
        assert image.width >= 1600
    assert ElementTree.parse(output / "fig09_execution_trajectories.svg").getroot().tag.endswith("svg")
    assert (output / "all_episode_execution_traces.pdf").read_bytes().startswith(b"%PDF")
    assert source_hashes == {source: _hash(Path(source).read_bytes()) for source in source_hashes}


def test_failed_report_marks_stale_completion_failed(tmp_path):
    path, comparison = _comparison(tmp_path / "input")
    output = tmp_path / "report"
    output.mkdir()
    target = output / "closed_loop_evidence.json"
    target.write_text(json.dumps({"status": "completed"}))
    _modify_trace(comparison, lambda trace: trace["chunks"][0].update(initial_noise_sha256=_hash("changed")))
    with pytest.raises(ValueError, match="sampling noise"):
        report.run_report(path, output)
    failed = json.loads(target.read_text())
    assert failed["status"] == "failed"
    assert "ValueError" in failed["error"]


def test_dry_run_does_not_require_a_saved_comparison_or_create_files(tmp_path):
    nonexistent = tmp_path / "not-created"
    result = subprocess.run([
        sys.executable, "-m", "lerobot.scripts.pi05_task7_closed_loop_report",
        "--comparison-json", str(nonexistent / "comparison.json"), "--output", str(nonexistent), "--dry-run",
    ], text=True, capture_output=True, env=os.environ.copy(), check=True)
    assert "CPU-only report" in result.stdout
    assert not nonexistent.exists()
