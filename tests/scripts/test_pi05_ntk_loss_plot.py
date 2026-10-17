"""CPU-only checks for exact loss timelines and the NTK/loss composite figure."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from lerobot.scripts.plot_pi05_ntk_with_loss import load_loss_csv, load_ntk, load_training_log, make_figure


def training_log():
    # The real MetricsTracker prints rounded K suffixes, even at distinct steps.
    pretrain = ("200", "400", "600", "800", "1K")
    flow = ("200", "400", "600", "800", "1K", "1K", "1K", "2K", "2K", "2K", "2K", "2K", "3K", "3K", "3K")
    lines = [
        "Starting integrated PI0.5 Stage 1 with 750 action-only updates",
        "{'log_freq': 200}",
        "Start offline training on a fixed dataset",
    ]
    lines += [
        f"INFO step:{step} smpl:64 ep:1 epch:0.1 loss:{2.0 / (index + 1):.3f}"
        for index, step in enumerate(pretrain)
    ]
    lines += [
        "PI0.5 Stage 1 complete; rebuilding the model",
        "{'log_freq': 200}",
        "Start offline training on a fixed dataset",
    ]
    lines += [
        f"INFO step:{step} smpl:64 ep:1 epch:0.1 loss:{0.3 / (index + 1):.3f}"
        for index, step in enumerate(flow)
    ]
    return "\n".join(lines)


def ntk_document():
    stages = [
        {"name": name, "total_step": step}
        for name, step in zip(("before", "priming", "stage2", "final"), (0, 750, 1000, 4000), strict=True)
    ]
    records = []
    for index, stage in enumerate(stages):
        for seed in range(3):
            groups = {
                f"{scope}/{module}": {
                    "effective_rank": 3 + index + seed * 0.2,
                    "parameter_normalized_energy": scale * (1 + index * 0.1 + seed * 0.02),
                }
                for scope in ("backbone", "prompts")
                for module, scale in (("vlm", 0.01), ("action", 0.1))
            }
            records.append({"stage": stage["name"], "seed": seed, "groups": groups})
    return {"manifest": {"stages": stages, "seeds": [0, 1, 2]}, "records": records}


class TestLossTimeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        return path

    def test_integrated_log_recovers_rounded_steps_and_offset(self):
        path = self.write("training.log", training_log())
        losses = load_training_log(path)
        self.assertEqual([row["step"] for row in losses], list(range(200, 4001, 200)))
        self.assertEqual(
            [row["phase"] for row in losses[:6]],
            ["priming", "priming", "priming", "transition", "bridge", "flow"],
        )
        self.assertNotIn(0, [row["step"] for row in losses])

    def test_incomplete_or_multiple_logs_are_rejected(self):
        examples = (
            training_log().replace("step:400", "step:450", 1),
            training_log() + "\nStarting integrated PI0.5 Stage 1",
            "INFO step:200 smpl:64 loss:0.1",
        )
        for text in examples:
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                load_training_log(self.write("bad.log", text))

    def test_explicit_log_interval_must_match_config(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            load_training_log(self.write("training.log", training_log()), log_freq=100)

    def test_wandb_csv_offset_uses_optimizer_step_not_internal_step(self):
        path = self.write("flow.csv", "_step,train/steps,train/loss\n1,200,0.3\n2,3000,0.1\n")
        losses = load_loss_csv(path, step_offset=1000)
        self.assertEqual([row["step"] for row in losses], [1200, 4000])
        self.assertEqual([row["phase"] for row in losses], ["flow", "flow"])

    def test_csv_requires_one_run_and_exact_distinct_steps(self):
        examples = (
            "Step,a - train/loss,b - train/loss\n200,.3,.4\n",
            "Step,loss\n200,.3\n200,.4\n",
            "Step,loss\n200.5,.3\n",
            "Step,loss\n200,nan\n",
        )
        for text in examples:
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_loss_csv(self.write("bad.csv", text))

    def test_layout_has_four_scatter_panels_above_loss_and_exact_arrows(self):
        path = self.write("results.json", json.dumps(ntk_document()))
        stages, seeds, lookup, scopes = load_ntk(path, "both")
        losses = load_training_log(self.write("training.log", training_log()))
        fig = make_figure(stages, seeds, lookup, scopes[0], losses)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        self.assertEqual(len(fig.axes), 5)
        self.assertEqual(len({axis.get_position().y0 for axis in fig.axes[:4]}), 1)
        self.assertLess(fig.axes[4].get_position().y1, fig.axes[0].get_position().y0)
        self.assertEqual(
            [artist.xy2[0] for artist in fig.artists if isinstance(artist, ConnectionPatch)],
            [0, 750, 1000, 4000],
        )
        self.assertEqual(len({axis.get_ylim() for axis in fig.axes[:4]}), 1)

    def test_shell_replots_without_checkpoints_and_preserves_ntk_input(self):
        repo = Path(__file__).resolve().parents[2]
        run = self.root / "run"
        ntk = run / "ntk_stages"
        ntk.mkdir(parents=True)
        result = ntk / "results.json"
        original = json.dumps(ntk_document())
        result.write_text(original)
        log = self.write("training.log", training_log())
        environment = {**os.environ, "TRAIN_LOG": str(log)}
        output = subprocess.run(
            ["bash", str(repo / "examples/analysis/replot_pi05_ntk_with_loss.sh"), str(run)],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(output.returncode, 0, output.stderr)
        self.assertEqual(result.read_text(), original)
        for scope in ("backbone", "prompts"):
            for suffix in ("png", "pdf", "svg"):
                self.assertTrue((ntk / f"with_loss/{scope}_ntk_with_loss.{suffix}").is_file())
        rows = load_loss_csv(ntk / "with_loss/loss_curve.csv")
        self.assertEqual(rows[-1]["step"], 4000)


if __name__ == "__main__":
    unittest.main()
