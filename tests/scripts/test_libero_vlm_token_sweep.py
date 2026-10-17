"""Exercise evaluation shell dispatch and preflight without CUDA or LIBERO."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SINGLE = ROOT / "run_eval_libero-full.sh"
SWEEP = ROOT / "run_eval_libero_vlm_token_sweep.sh"


class LiberoTokenSweepTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="libero sweep test ")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "CKPT_ROOT": str(self.work / "checkpoints"),
            "BASE_OUTPUT": str(self.work / "results"),
            "DRY_RUN": "true",
        }
        for count in (1, 2, 8, 16):
            self.checkpoint(count)

    def checkpoint(self, count, *, seed=0, step=3000, name=None):
        name = name or f"pi05-vlm_only-vlm{count}-seed{seed}"
        path = self.work / "checkpoints" / name / "checkpoints" / f"{step:06d}" / "pretrained_model"
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps({"num_vlm_prompt_tokens": count, "num_prompt_tokens": 0})
        )
        return path

    def run_script(self, *args, script=SWEEP, **overrides):
        return subprocess.run(
            ["bash", str(script), *args],
            env={**self.env, **overrides},
            capture_output=True,
            text=True,
            check=False,
        )

    def commands(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("python -u - ")]

    @staticmethod
    def arg(command, name):
        return next(arg.split("=", 1)[1] for arg in command if arg.startswith(name + "="))

    def test_default_dry_run_checks_four_models_and_isolates_outputs(self):
        commands = self.commands(self.run_script("all", "0"))
        self.assertEqual(len(commands), 4)
        outputs = []
        for count, command in zip((1, 2, 8, 16), commands, strict=True):
            self.assertEqual(self.arg(command, "--policy.path"), str(self.checkpoint(count)))
            output = self.arg(command, "--output_dir")
            self.assertIn(f"pi05-vlm_only-vlm{count}-seed0-003000-libero-all4-resume", output)
            outputs.append(output)
            self.assertIn("--eval.n_episodes=10", command)
            self.assertIn("--policy.n_action_steps=10", command)
            self.assertIn("--env.task=libero_spatial,libero_object,libero_goal,libero_10", command)
            self.assertFalse(any(arg.startswith("--policy.num_") for arg in command))
        self.assertEqual(len(set(outputs)), 4)
        self.assertFalse((self.work / "results").exists())

    def test_legacy_name_and_single_token_selection(self):
        legacy = self.checkpoint(16, name="pi05-vlm_only-seed0")
        (command,) = self.commands(self.run_script("vlm_only", "all", "0", script=SINGLE))
        self.assertEqual(self.arg(command, "--policy.path"), str(legacy))
        (command,) = self.commands(
            self.run_script("vlm_only", "libero_10", "0", script=SINGLE, VLM_PROMPT_TOKENS="8")
        )
        self.assertEqual(self.arg(command, "--policy.path"), str(self.checkpoint(8)))
        self.assertIn("--env.task=libero_10", command)

    def test_subset_seed_step_and_episode_overrides(self):
        for count in (8, 16):
            self.checkpoint(count, seed=2, step=6000)
        commands = self.commands(
            self.run_script(
                "no10",
                "2",
                VLM_PROMPT_TOKEN_COUNTS="8 16",
                CHECKPOINT_STEP="6000",
                EPISODES_PER_TASK="1",
                GPU_ID="2",
            )
        )
        for count, command in zip((8, 16), commands, strict=True):
            self.assertIn(f"vlm{count}-seed2", self.arg(command, "--policy.path"))
            self.assertIn("/006000/", self.arg(command, "--policy.path"))
            self.assertIn("--eval.n_episodes=1", command)
            self.assertIn("--env.task=libero_spatial,libero_object,libero_goal", command)

    def test_missing_or_mislabeled_last_model_prevents_any_evaluation(self):
        config = self.checkpoint(16) / "config.json"
        for data in (
            {"num_vlm_prompt_tokens": 8, "num_prompt_tokens": 0},
            {"num_vlm_prompt_tokens": 16, "num_prompt_tokens": 16},
        ):
            config.write_text(json.dumps(data))
            result = self.run_script("all", "0", DRY_RUN="false")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Checkpoint prompt mismatch", result.stderr)
            self.assertNotIn("Evaluating vlm_only:", result.stdout)
            self.assertFalse((self.work / "results").exists())
        config.unlink()
        config.parent.rmdir()
        result = self.run_script("all", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint not found", result.stderr)
        self.assertNotIn("Evaluating vlm_only:", result.stdout)

    def test_explicit_checkpoint_override_is_still_validated(self):
        base = str(self.checkpoint(2).parent.parent)
        result = self.run_script("vlm_only", "all", "0", script=SINGLE, VLM_PROMPT_TOKENS="8", BASE_CKPT=base)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checkpoint prompt mismatch", result.stderr)
        (command,) = self.commands(
            self.run_script(
                "vlm_only",
                "all",
                "0",
                script=SINGLE,
                VLM_PROMPT_TOKENS="2",
                BASE_CKPT=base,
                RUN_NAME="custom-2",
            )
        )
        self.assertIn("custom-2", self.arg(command, "--output_dir"))

    def test_invalid_settings_fail_before_evaluation(self):
        for overrides in (
            {"VLM_PROMPT_TOKEN_COUNTS": ""},
            {"VLM_PROMPT_TOKEN_COUNTS": "1 1"},
            {"VLM_PROMPT_TOKEN_COUNTS": "0"},
            {"VLM_PROMPT_TOKEN_COUNTS": "1 bad"},
            {"BASE_CKPT": "/wrong/model"},
            {"VLM_PROMPT_TOKENS": "8"},
            {"EPISODES_PER_TASK": "0"},
            {"CHECKPOINT_STEP": "003000"},
            {"DRY_RUN": "yes"},
        ):
            with self.subTest(overrides=overrides):
                result = self.run_script("all", "0", **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Evaluating vlm_only:", result.stdout)

    def install_fake_evaluator(self):
        bin_dir = self.work / "bin"
        bin_dir.mkdir()
        capture = self.work / "launches.jsonl"
        python = bin_dir / "python"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1:3] != ['-u', '-']:\n"
            f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
            "with open(os.environ['CAPTURE_FILE'], 'a') as f:\n"
            "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "args = dict(arg.split('=', 1) for arg in sys.argv[3:])\n"
            "count = json.loads((Path(args['--policy.path']) / 'config.json').read_text())['num_vlm_prompt_tokens']\n"
            "if str(count) == os.environ.get('FAIL_COUNT'):\n"
            "    sys.exit(7)\n"
            "output = Path(args['--output_dir'])\n"
            "output.mkdir(parents=True, exist_ok=True)\n"
            "data = {'per_task': [{'task_group': 'libero_10', 'task_id': 0, 'metrics': {'successes': [True]}}]}\n"
            "(output / 'eval_info.json').write_text(json.dumps(data))\n"
        )
        python.chmod(0o755)
        self.env.update(PATH=f"{bin_dir}{os.pathsep}{self.env['PATH']}", CAPTURE_FILE=str(capture))
        return capture

    def test_real_shell_dispatch_writes_separate_model_logs(self):
        capture = self.install_fake_evaluator()
        result = self.run_script("libero_10", "0", DRY_RUN="false")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(capture.read_text().splitlines()), 4)
        for count in (1, 2, 8, 16):
            summary = self.work / "results" / f"eval_pi05-vlm_only-vlm{count}-seed0-libero_10-summary.log"
            self.assertIn("libero_10: 100.00% (1/1)", summary.read_text())
        self.assertEqual(len(list((self.work / "results").glob("*/eval_info.json"))), 4)

    def test_first_evaluation_failure_stops_remaining_models(self):
        capture = self.install_fake_evaluator()
        result = self.run_script("all", "0", DRY_RUN="false", FAIL_COUNT="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(capture.read_text().splitlines()), 2)
        self.assertNotIn("Evaluating vlm_only: 8", result.stdout)


if __name__ == "__main__":
    unittest.main()
