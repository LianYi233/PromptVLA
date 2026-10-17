"""Run directly without torch: python tests/test_training_diagnostics.py."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "src/lerobot/utils/training_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("training_diagnostics", FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiagnosticsTests(unittest.TestCase):
    def test_window_averages_skip_frequency_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_diagnostics.jsonl"
            writer = MODULE.TrainingDiagnostics(path, {"training_stage": "flow"})
            writer.update({"loss_per_dim": [1.0, 3.0], "cabo/action_prompt_scale": 0.2, "loss": 0.9})
            writer.update({"loss_per_dim": [3.0, float("nan")], "optimizer_step/skipped": 1})
            record = writer.write(2, {"loss": 0.3, "grad_norm": float("inf")})
            curves = MODULE.diagnostic_means(record)
            self.assertEqual(curves["loss_per_dim/0"], 2.0)
            self.assertEqual(curves["loss_per_dim/1/nonfinite"], 1)
            self.assertEqual(curves["optimizer_step/skipped"], 0.5)
            self.assertNotIn("loss", curves)
            self.assertEqual(curves["policy_rank0/loss"], 0.9)
            self.assertTrue(all(isinstance(value, (int, float)) for value in curves.values()))
            result = json.loads(path.read_text())
            self.assertEqual(result["step"], 2)
            self.assertEqual(result["window_updates"], 2)
            self.assertIsNone(result["train"]["grad_norm"])
            metrics = result["diagnostics_rank0"]
            self.assertEqual(metrics["loss_per_dim/0"]["mean"], 2.0)
            self.assertEqual(metrics["loss_per_dim/1"]["mean"], 3.0)
            self.assertEqual(metrics["loss_per_dim/1"]["nonfinite"], 1)
            self.assertEqual(metrics["optimizer_step/skipped"]["mean"], 0.5)
            # Missing CABO during action-only priming is not an invented zero scale.
            self.assertEqual(metrics["cabo/action_prompt_scale"]["count"], 1)
            writer.write(2, {})
            writer.update({"loss_per_dim": [5.0, 7.0]})
            writer.write(3, {"loss": 0.2})
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1]["window_updates"], 1)
            self.assertEqual(records[1]["diagnostics_rank0"]["loss_per_dim/0"]["mean"], 5.0)

    def test_resume_appends_and_all_nonfinite_metric_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_diagnostics.jsonl"
            for step in (50, 100):
                writer = MODULE.TrainingDiagnostics(path, {})
                writer.update({"update": float("inf")})
                writer.write(step, {})
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["step"] for row in records], [50, 100])
            value = records[1]["diagnostics_rank0"]["update"]
            self.assertIsNone(value["mean"])
            self.assertEqual(value["count"], 0)
            self.assertEqual(value["nonfinite"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
