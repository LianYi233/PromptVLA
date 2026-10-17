"""Offline metrics and alignment tests require NumPy, not torch, a checkpoint or hardware."""

import contextlib
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_piper_offline as offline  # noqa: E402

NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
CAMERA = "observation.images.cam_high"


class RecordedDataset:
    """Two disjoint episodes: relative positions differ from absolute frame indices."""

    fps = 10

    def __init__(self):
        self.meta = SimpleNamespace(
            episodes={
                7: {"dataset_from_index": 100, "dataset_to_index": 104},
                9: {"dataset_from_index": 200, "dataset_to_index": 202},
            }
        )
        self.rows = []
        for ep, length, start in ((7, 4, 100), (9, 2, 200)):
            for frame in range(length):
                self.rows.append(
                    {
                        "episode_index": ep,
                        "frame_index": frame,
                        "index": start + frame,
                        "task": "move the test object",
                        "action": np.arange(7, dtype=np.float32) / 10 + frame / 100,
                        "observation.state": np.arange(7, dtype=np.float32)[::-1] / 20,
                        CAMERA: np.full((3, 8, 12), 128, dtype=np.uint8),
                    }
                )

    def __getitem__(self, relative):
        item = dict(self.rows[relative])
        ep = self.meta.episodes[item["episode_index"]]
        indices = [min(item["index"] + h, ep["dataset_to_index"] - 1) for h in range(3)]
        by_index = {row["index"]: row for row in self.rows}
        item["action"] = np.stack([by_index[idx]["action"] for idx in indices])
        item["action_is_pad"] = item["index"] + np.arange(3) >= ep["dataset_to_index"]
        return item

    def get_raw_item(self, relative):
        return self.rows[relative]


class OfflineTests(unittest.TestCase):
    def test_selection_uses_relative_positions_and_spans_episodes(self):
        dataset = RecordedDataset()
        self.assertEqual(offline.select_indices(dataset.rows, 2, 0), [0, 2, 4])
        self.assertEqual(offline.select_indices(dataset.rows, 1, 2), [0, 5])
        self.assertEqual(offline.select_indices(dataset.rows, 1, 0), list(range(6)))

    def test_training_episode_subset_is_respected(self):
        self.assertEqual(offline.parse_episodes("all", 10, [7, 9]), [7, 9])
        with self.assertRaises(ValueError):
            offline.parse_episodes("0", 10, [7, 9])
        with self.assertRaises(ValueError):
            offline.parse_episodes("7,7", 10)

    def test_tail_padding_is_excluded_without_crossing_episode(self):
        dataset = RecordedDataset()
        item = dataset[3]
        target, valid = offline.recorded_targets(item, dataset.meta.episodes[7], 3, 7)
        np.testing.assert_array_equal(valid, [True, False, False])
        np.testing.assert_array_equal(target[0], dataset.rows[3]["action"])
        item["action_is_pad"][:] = False
        with self.assertRaisesRegex(ValueError, "边界"):
            offline.recorded_targets(item, dataset.meta.episodes[7], 3, 7)

    def test_alignment_and_target_shape_fail_fast(self):
        dataset = RecordedDataset()
        item = dataset[0]
        item["index"] = 101
        with self.assertRaisesRegex(ValueError, "错位"):
            offline.recorded_targets(item, dataset.meta.episodes[7], 3, 7)
        item = dataset[0]
        item["action"] = item["action"][:2]
        with self.assertRaisesRegex(ValueError, "形状"):
            offline.recorded_targets(item, dataset.meta.episodes[7], 3, 7)

    def test_hold_baseline_matches_feature_names_not_column_positions(self):
        hold = offline.hold_baseline(np.arange(7), NAMES[::-1], NAMES, 3)
        np.testing.assert_array_equal(hold, np.repeat(np.arange(7)[::-1][None], 3, axis=0))

    def test_observation_never_contains_targets_or_padding_and_does_not_mutate_raw(self):
        item = RecordedDataset()[0]
        observation = offline.observation_inputs(item, ["observation.state", CAMERA])
        self.assertEqual(set(observation), {"observation.state", CAMERA, "task"})
        observation[CAMERA][:] = 0
        self.assertEqual(item[CAMERA].max(), 128)

    def test_processor_action_none_placeholder_is_allowed_but_labels_are_rejected(self):
        batch = {"action": None, "observation.state": np.zeros((1, 7))}
        self.assertNotIn("action", offline.remove_empty_action(batch))
        with self.assertRaisesRegex(ValueError, "标签"):
            offline.remove_empty_action({"action": np.zeros((1, 3, 7))})

    def test_exact_metrics_mask_padding_and_keep_units_separate(self):
        target = np.zeros((1, 3, 7))
        prediction = np.ones_like(target)
        prediction[:, 1] = 3
        prediction[:, 2] = 1e8
        target[:, 2] = np.nan  # Even nonfinite padded labels must not affect metrics.
        metrics = offline.error_metrics(prediction, target, np.array([[True, True, False]]), NAMES, 3)
        self.assertEqual(metrics["valid_action_pairs"], 2)
        self.assertAlmostEqual(metrics["joint_mae_deg"], 2 * 180 / math.pi)
        self.assertAlmostEqual(metrics["gripper_mae_dataset_units"], 2)
        self.assertAlmostEqual(metrics["per_dimension_dataset_units"][NAMES[0]]["rmse"], math.sqrt(5))
        self.assertAlmostEqual(metrics["per_dimension_dataset_units"][NAMES[0]]["p95_absolute_error"], 2.9)
        self.assertEqual(
            offline.error_metrics(prediction, target, np.array([[True, True, False]]), NAMES, 1)[
                "gripper_mae_dataset_units"
            ],
            1,
        )

    def test_zero_error_baseline_ratio_is_null_not_inf(self):
        zero = np.zeros((1, 3, 7))
        result = offline.summarize(
            {"predicted": zero, "target": zero, "hold": zero, "valid": np.ones((1, 3), bool)}, NAMES, [1]
        )
        self.assertIsNone(result["1"]["joint_mae_ratio_to_hold"])

    def test_full_evaluation_no_label_leakage_and_csv_npz_alignment(self):
        dataset = RecordedDataset()
        features = {"action": {"names": NAMES}, "observation.state": {"names": NAMES[::-1]}, CAMERA: {}}
        calls = []

        def predict(observation, absolute_index):
            self.assertEqual(set(observation), {"observation.state", CAMERA, "task"})
            calls.append(absolute_index)
            relative = next(i for i, row in enumerate(dataset.rows) if row["index"] == absolute_index)
            recorded = dataset[relative]
            result = recorded["action"] + 0.1
            result[recorded["action_is_pad"]] = 999
            return result

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            result = offline.evaluate(
                dataset,
                [0, 3, 5],
                predict,
                features,
                output,
                chunk_size=3,
                execution_steps=2,
                plots=0,
                manifest={"protocol": "test"},
            )
            self.assertEqual(calls, [100, 103, 201])
            self.assertEqual(result["horizons"]["3"]["model"]["valid_action_pairs"], 5)
            self.assertAlmostEqual(
                result["horizons"]["3"]["model"]["gripper_mae_dataset_units"], 0.1, places=6
            )
            self.assertTrue(json.loads((output / "summary.json").read_text())["completed"])
            with np.load(output / "predictions.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["predicted"].shape, (3, 3, 7))
                np.testing.assert_array_equal(arrays["index"], [100, 103, 201])
            with (output / "actions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["label_index"]) for row in rows], [100, 101, 102, 103, 201])
            self.assertEqual(len(rows), 5)

    def test_invalid_prediction_leaves_no_success_summary(self):
        dataset = RecordedDataset()
        features = {"action": {"names": NAMES}, "observation.state": {"names": NAMES[::-1]}, CAMERA: {}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "NaN"):
                offline.evaluate(
                    dataset,
                    [0],
                    lambda *_: np.full((3, 7), np.nan),
                    features,
                    output,
                    chunk_size=3,
                    execution_steps=2,
                    plots=0,
                    manifest={},
                )
            self.assertFalse((output / "summary.json").exists())

    def test_dependency_light_help_does_not_load_robot_or_model(self):
        with (
            patch.dict(sys.modules, {"torch": None, "piper_sdk": None, "pyrealsense2": None}),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            offline.main(["--help"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
