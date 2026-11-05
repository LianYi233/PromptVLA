"""Synthetic, CPU-only tests of seven-command decomposition and provenance.

No checkpoint inference or robot execution is represented by these fixtures.
"""

import csv
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest
from PIL import Image

from lerobot.scripts import pi05_task7_seven_dim as seven


def _condition(episode, frame, noise, *, local=None, fraction=0.0):
    return {"episode_index": episode, "global_frame_index": frame,
            "frame_index": frame if local is None else local,
            "noise_seed": noise, "chunk_start_fraction": fraction}


def _noise_hash(noise):
    digest = hashlib.sha256(f"torch.float32:{(1, 50, noise.shape[-1])}:".encode())
    digest.update(np.ascontiguousarray(noise).tobytes())
    return digest.hexdigest()


def _source_fixture(directory):
    """Known MSE: demo 9 averages (1 + 9)/2=5, demo 74 has 25; pooled=15."""
    directory.mkdir(parents=True, exist_ok=True)
    conditions = [_condition(episode, frame, seed, local=local, fraction=fraction)
                  for episode, frame, local, fraction in ((9, 0, 0, 0.0), (9, 50, 50, 0.5), (74, 100, 0, 0.0))
                  for seed in (3000, 4000)]
    prediction = np.ones((6, 50, 7), dtype=np.float32) * np.asarray([1, 1, 3, 3, 5, 5])[:, None, None]
    noise = np.random.default_rng(42).standard_normal((6, 50, 32)).astype(np.float32)
    arrays = {"clean_target": np.zeros((6, 50, 7), dtype=np.float32), "initial_noise": noise,
              **{f"{mode}_actions": prediction.copy() for mode in seven.MODES}}
    array_path = directory / "full_sampling_arrays.npz"
    np.savez_compressed(array_path, **arrays)
    source = {
        "protocol": seven.SOURCE_PROTOCOL, "status": "completed", "interventions": list(seven.MODES),
        "array_file": "/original/server/path/full_sampling_arrays.npz", "array_row_order": "conditions",
        "array_sha256": seven._sha256(array_path), "conditions": conditions,
        "episode_ids": [9, 74], "number_of_demonstrations": 2, "number_of_conditions": 6,
        "number_of_anchors": 3, "noise_seeds": [3000, 4000],
        "regions": {"executed_prefix": [0, 10], "full_chunk": [0, 50]},
        "sampling": {"executed_prefix_steps": 10, "native_num_inference_steps": 4,
                     "prediction_horizon": 50, "targets_passed_to_inference": False,
                     "initial_state": "pure_gaussian_noise", "one_checkpoint_loaded_for_all_interventions": True},
        "representative_row": 2, "representative_condition": conditions[2],
        "representative_selection": "First demo, nearest middle window, smallest fixed noise seed; independent of errors",
        "checkpoint": "/SYNTHETIC_TEST_FIXTURE/not_a_model", "checkpoint_fingerprint": {"synthetic_fixture": True},
        "metrics": {region: {group: {"mean_mse": {mode: 15.0 for mode in seven.MODES}}
                             for group in ("all", "gripper")} for region in ("executed_prefix", "full_chunk")},
        "sampling_audits": [{mode: {
            "intervention": mode, "initial_noise_sha256": _noise_hash(noise[index]),
            "observation_sha256": f"{condition['global_frame_index']:064x}", "denoising_calls": 4,
            "starts_from_supplied_pure_noise": True, "flow_times": [1.0, 0.75, 0.5, 0.25],
        } for mode in seven.MODES} for index, condition in enumerate(conditions)],
    }
    (directory / "full_sampling_evidence.json").write_text(json.dumps(source), encoding="utf-8")
    return source, arrays


def test_hierarchical_aggregation_weights_demos_windows_and_noises_separately():
    conditions = [_condition(9, 10, 1), _condition(9, 10, 2), _condition(9, 20, 1), _condition(74, 30, 1)]
    # Window means: demo 9 -> (1+3)/2=2, then 10; demo 74 -> 20.
    # Demo means are 6 and 20, regardless of the unequal repeat/window counts.
    episodes, values = seven.demo_means([1, 3, 10, 20], conditions)
    assert episodes == [9, 74]
    np.testing.assert_allclose(values, [6, 20])
    assert values.mean() == 13
    assert values.mean() != np.mean([1, 3, 10, 20])


def test_dimension_groups_prefix_and_error_sign_have_known_analytic_values():
    conditions = [_condition(9, 0, 3000), _condition(74, 100, 3000)]
    target = np.zeros((2, 50, 7))
    predictions = {mode: np.ones_like(target) for mode in seven.MODES}
    # Zero is better on the first ten translation commands; Reverse is worse
    # only on the unexecuted gripper tail. No arm is forced to look beneficial.
    predictions["zero"][:, :10, :3] = 0.0
    predictions["reversed"][:, 10:, 6] = 2.0
    report = seven.summarize(predictions, target, conditions, 10)
    prefix = report["metrics"]["executed_prefix"]
    full = report["metrics"]["full_chunk"]
    assert prefix["translation"]["mean_paired_deltas"]["zero_minus_correct"] == -1.0
    assert prefix["all"]["mean_paired_deltas"]["zero_minus_correct"] == pytest.approx(-3 / 7)
    assert prefix["all"]["mean_paired_deltas"]["reversed_minus_correct"] == 0.0
    assert full["gripper"]["mean_paired_deltas"]["reversed_minus_correct"] == pytest.approx(2.4)
    assert full["all"]["mean_paired_deltas"]["reversed_minus_correct"] == pytest.approx(2.4 / 7)
    assert full["all"]["mean_paired_deltas"]["zero_minus_correct"] == pytest.approx(-0.2 * 3 / 7)
    assert prefix["rotation"]["mean_output_change_rms"] == {"zero": 0.0, "reversed": 0.0}
    assert prefix["all"]["mean_output_change_rms"]["zero"] == pytest.approx(np.sqrt(3 / 7))
    assert prefix["dim0"]["paired_bootstrap_95pct"]["zero_minus_correct"] == [-1.0, -1.0]
    changes = report["mean_position_dimension_mse_difference"]
    zero = np.asarray(changes["zero_minus_correct"])
    reverse = np.asarray(changes["reversed_minus_correct"])
    assert zero.shape == reverse.shape == (7, 50)
    np.testing.assert_array_equal(zero[:3, :10], -1.0)
    np.testing.assert_array_equal(zero[:, 10:], 0.0)
    np.testing.assert_array_equal(reverse[6, 10:], 3.0)
    # The two effects tie in absolute all-dimension MSE; ties retain a fixed
    # demo/position order and include both prefix and unexecuted-tail cells.
    assert any(not row["within_executed_prefix"]
               for row in report["largest_absolute_demo_position_differences"])


def test_complete_component_errors_do_not_imply_any_recorded_action_improvement():
    conditions = [_condition(9, 0, 3000)]
    target = np.zeros((1, 50, 7))
    correct = np.ones_like(target)
    zero = -np.ones_like(target)
    # Identical MSE does not mean identical generated commands.
    report = seven.summarize({"correct": correct, "zero": zero, "reversed": correct}, target, conditions, 10)
    metric = report["metrics"]["full_chunk"]["all"]
    assert metric["mean_paired_deltas"]["zero_minus_correct"] == 0.0
    assert metric["mean_output_change_rms"]["zero"] == 2.0
    assert metric["paired_bootstrap_95pct"]["zero_minus_correct"] is None


def test_loader_accepts_portable_source_paths_and_reproduces_original_mse(tmp_path):
    directory = tmp_path / "source"
    source, arrays = _source_fixture(directory)
    loaded, predictions, target, report, provenance = seven.load_sampling_outputs(directory)
    assert loaded == source
    assert provenance["arrays"] == str(directory / "full_sampling_arrays.npz")
    assert report["metrics"]["full_chunk"]["all"]["per_demo_mse"]["correct"] == [5.0, 25.0]
    assert report["metrics"]["full_chunk"]["all"]["mean_mse"]["correct"] == 15.0
    np.testing.assert_array_equal(target, arrays["clean_target"])
    np.testing.assert_array_equal(predictions["correct"], arrays["correct_actions"])


@pytest.mark.parametrize("corruption", [
    "hash", "condition_order", "duplicate", "count", "missing_noise", "nonfinite", "shape",
    "target_changes", "observation_audit", "native_noise", "schedule", "source_mse", "prefix", "incomplete_status",
])
def test_loader_rejects_unpaired_stale_or_changed_inputs(tmp_path, corruption):
    directory = tmp_path / "source"
    source, arrays = _source_fixture(directory)
    if corruption == "hash":
        source["array_sha256"] = "0" * 64
    elif corruption == "condition_order":
        # Preserve metadata counts, but rotate the raw-noise rows without the
        # corresponding audit rows. Recomputed NPZ hash alone is insufficient.
        arrays["initial_noise"] = np.roll(arrays["initial_noise"], 1, axis=0)
    elif corruption == "duplicate":
        source["conditions"][1] = source["conditions"][0].copy()
    elif corruption == "count":
        source["number_of_conditions"] = 8
    elif corruption == "missing_noise":
        source["noise_seeds"].append(5000)
    elif corruption == "nonfinite":
        arrays["zero_actions"][0, 0, 0] = np.nan
    elif corruption == "shape":
        arrays["zero_actions"] = arrays["zero_actions"][:, :, :6]
    elif corruption == "target_changes":
        arrays["clean_target"][1, 0, 0] = 0.5
    elif corruption == "observation_audit":
        source["sampling_audits"][0]["reversed"]["observation_sha256"] = "f" * 64
    elif corruption == "native_noise":
        source["sampling_audits"][0]["zero"]["initial_noise_sha256"] = "e" * 64
    elif corruption == "schedule":
        source["sampling_audits"][0]["zero"]["flow_times"] = [0.5]
    elif corruption == "source_mse":
        source["metrics"]["full_chunk"]["all"]["mean_mse"]["zero"] = 999.0
    elif corruption == "prefix":
        source["sampling"]["executed_prefix_steps"] = 50
    elif corruption == "incomplete_status":
        source["status"] = "failed"
    if corruption in {"condition_order", "nonfinite", "shape", "target_changes"}:
        array_path = directory / "full_sampling_arrays.npz"
        np.savez_compressed(array_path, **arrays)
        source["array_sha256"] = seven._sha256(array_path)
    (directory / "full_sampling_evidence.json").write_text(json.dumps(source))
    with pytest.raises(ValueError):
        seven.load_sampling_outputs(directory)


def test_every_position_csv_contribution_reproduces_primary_metrics(tmp_path):
    conditions = [_condition(9, 10, 1), _condition(9, 10, 2), _condition(9, 20, 1), _condition(74, 30, 1)]
    rng = np.random.default_rng(45)
    target = rng.standard_normal((4, 50, 7))
    predictions = {mode: rng.standard_normal(target.shape) for mode in seven.MODES}
    summary = seven.summarize(predictions, target, conditions, 10)
    path = tmp_path / "positions.csv.gz"
    seven.write_position_csv(predictions, target, conditions, 10, path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * 50 * 7 * 2
    for mode in ("zero", "reversed"):
        for region, column in (("executed_prefix", "contribution_to_prefix_all_mse_delta"),
                               ("full_chunk", "contribution_to_full_chunk_all_mse_delta")):
            actual = sum(float(row[column]) for row in rows if row["intervention"] == mode)
            expected = summary["metrics"][region]["all"]["mean_paired_deltas"][f"{mode}_minus_correct"]
            assert actual == pytest.approx(expected, abs=1e-12)
    assert {float(row["condition_mean_weight"]) for row in rows} == {0.125, 0.25, 0.5}
    assert {row["dimension"] for row in rows} == set(map(str, range(7)))
    assert any(float(row["error_difference"]) < 0 for row in rows)
    assert any(float(row["error_difference"]) > 0 for row in rows)


def test_real_artifacts_are_complete_and_leave_sources_unchanged(tmp_path):
    source_dir, output_dir = tmp_path / "source", tmp_path / "output"
    source, _ = _source_fixture(source_dir)
    before = {path.name: path.read_bytes() for path in source_dir.iterdir()}
    result = seven.run(SimpleNamespace(input=source_dir, output=output_dir))
    saved = json.loads((output_dir / "seven_dim_evidence.json").read_text())
    assert saved == result and saved["status"] == "completed"
    assert saved["protocol"] == seven.PROTOCOL
    assert saved["source_protocol"] == source["protocol"]
    assert set(seven.ARTIFACT_NAMES) <= {path.name for path in output_dir.iterdir()}
    assert before == {path.name: path.read_bytes() for path in source_dir.iterdir()}
    assert saved["representative_row"] == 2
    pages = saved["all_windows_collection"]["pages"]
    assert len(pages) == 3
    assert [page["row_index"] for page in pages] == [0, 2, 4]
    assert {page["noise_seed"] for page in pages} == {3000}
    assert saved["number_of_conditions"] == 6  # No noise draws removed from the statistics.
    for stem in ("fig06_seven_dim_actions", "fig07_seven_dim_error_changes"):
        with Image.open(output_dir / f"{stem}.png") as image:
            assert image.width >= 1600 and image.height >= 1000
        assert ElementTree.parse(output_dir / f"{stem}.svg").getroot().tag == "{http://www.w3.org/2000/svg}svg"
    pdf_bytes = (output_dir / "all_windows_seven_dim.pdf").read_bytes()
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(re.findall(rb"/Type /Page\b", pdf_bytes)) == 3
    with (output_dir / "seven_dim_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * len(seven.GROUPS)
    for name, digest in saved["artifact_sha256"].items():
        assert digest == seven._sha256(output_dir / name)


def test_source_failure_does_not_leave_stale_completed_report(tmp_path):
    source_dir, output_dir = tmp_path / "source", tmp_path / "output"
    _source_fixture(source_dir)
    (source_dir / "full_sampling_arrays.npz").write_bytes(b"invalid synthetic input")
    output_dir.mkdir()
    report = output_dir / "seven_dim_evidence.json"
    report.write_text(json.dumps({"status": "completed", "figure_paths": ["stale.png"]}))
    with pytest.raises(ValueError, match="SHA256"):
        seven.run(SimpleNamespace(input=source_dir, output=output_dir))
    saved = json.loads(report.read_text())
    assert saved["status"] == "failed"
    assert "figure_paths" not in saved and "metrics" not in saved


def test_dry_run_reads_no_sources_writes_nothing_and_imports_no_heavy_libraries(tmp_path):
    output = tmp_path / "absent" / "seven_dim"
    program = (
        "import sys; from lerobot.scripts import pi05_task7_seven_dim as s; "
        f"s.main(['--input', {str(tmp_path / 'missing_source')!r}, '--output', {str(output)!r}, '--dry-run']); "
        "assert not ({'numpy', 'matplotlib', 'torch', 'mujoco'} & set(sys.modules))"
    )
    env = os.environ.copy()
    source_root = str(Path(seven.__file__).resolve().parents[2])
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env, check=True)
    assert "CPU postprocessing only" in completed.stdout
    assert not output.exists()
