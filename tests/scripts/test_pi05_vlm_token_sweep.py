"""Check expensive-run configuration and shell failure handling without ML dependencies."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SINGLE = ROOT / "examples/training/train_pi05_prompt_ablation.sh"
SWEEP = ROOT / "examples/training/train_pi05_vlm_token_sweep.sh"


class VLMTokenSweepTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vlm sweep test ")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}
        for key in ("DATASET_ROOT", "PRETRAINED_PATH", "TOKENIZER_PATH"):
            path = self.work / key
            path.mkdir()
            self.env[key] = str(path)
        self.env.update(
            DRY_RUN="true",
            OUTPUT_ROOT=str(self.work / "checkpoints"),
            LOG_ROOT=str(self.work / "logs"),
        )

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
        return [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]

    def test_default_sweep_supplements_completed_runs_with_matched_recipes(self):
        for count in (1, 2, 8, 16):
            (self.work / "checkpoints" / f"pi05-vlm_only-vlm{count}-seed0").mkdir(parents=True)
        wandb_args = (
            "--wandb.enable=true",
            "--wandb.project=prompt-learning",
            "--wandb.mode=online",
            "--wandb.disable_artifact=true",
        )
        commands = self.commands(self.run_script("0", *wandb_args))
        self.assertEqual(len(commands), 2)
        for tokens, command in zip((4, 32), commands, strict=True):
            expected = (
                f"--policy.num_vlm_prompt_tokens={tokens}",
                "--policy.num_prompt_tokens=0",
                "--policy.cabo_enabled=false",
                "--policy.next_action_pretrain_steps=0",
                "--policy.next_action_bridge_steps=0",
                "--steps=3000",
                "--seed=0",
                "--batch_size=32",
                *wandb_args,
                f"--policy.pretrained_path={self.env['PRETRAINED_PATH']}",
                f"--output_dir={self.work / 'checkpoints' / f'pi05-vlm_only-vlm{tokens}-seed0'}",
            )
            for arg in expected:
                self.assertIn(arg, command)
        self.assertEqual(
            {path.name for path in (self.work / "checkpoints").iterdir()},
            {f"pi05-vlm_only-vlm{count}-seed0" for count in (1, 2, 8, 16)},
        )
        self.assertFalse((self.work / "logs").exists())

    def test_single_run_preserves_legacy_default_and_labels_explicit_lengths(self):
        (legacy,) = self.commands(self.run_script("vlm_only", "0", script=SINGLE))
        self.assertIn("--policy.num_vlm_prompt_tokens=16", legacy)
        self.assertIn("--job_name=pi05-vlm_only-seed0", legacy)
        for count in (1, 2, 4, 8, 16, 32):
            with self.subTest(count=count):
                (command,) = self.commands(
                    self.run_script("vlm_only", "0", script=SINGLE, VLM_PROMPT_TOKENS=str(count))
                )
                self.assertIn(f"--policy.num_vlm_prompt_tokens={count}", command)
                self.assertIn(f"--job_name=pi05-vlm_only-vlm{count}-seed0", command)

    def test_subset_and_common_training_overrides(self):
        commands = self.commands(
            self.run_script(
                "3", VLM_PROMPT_TOKEN_COUNTS="8 16", GPU_IDS="1", BATCH_SIZE="16", FLOW_STEPS="6000"
            )
        )
        self.assertEqual(len(commands), 2)
        for command in commands:
            for arg in ("CUDA_VISIBLE_DEVICES=1", "--seed=3", "--batch_size=16", "--steps=6000"):
                self.assertIn(arg, command)

    def test_all_destinations_checked_before_any_training(self):
        (self.work / "checkpoints/pi05-vlm_only-vlm32-seed0").mkdir(parents=True)
        result = self.run_script("0", DRY_RUN="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)
        self.assertNotIn("Starting vlm_only", result.stdout)
        self.assertFalse((self.work / "logs").exists())

    def test_invalid_lengths_and_conflicting_overrides_fail_before_launch(self):
        for counts in ("", " ", "0", "-1", "1.5", "01", "1 1", "1 bad"):
            with self.subTest(counts=counts):
                result = self.run_script("0", VLM_PROMPT_TOKEN_COUNTS=counts)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Starting vlm_only", result.stdout)
        for arg in (
            "--policy.num_vlm_prompt_tokens=4",
            "--policy.num_prompt_tokens=8",
            "--output_dir=/tmp/x",
        ):
            result = self.run_script("0", arg)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Sweep controls", result.stderr)
        result = self.run_script("vlm_only", "0", script=SINGLE, VLM_PROMPT_TOKENS="-1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive integer", result.stderr)

    def test_training_stops_on_failed_count_and_keeps_separate_logs(self):
        bin_dir = self.work / "bin"
        bin_dir.mkdir()
        capture = self.work / "launches.jsonl"
        accelerate = bin_dir / "accelerate"
        accelerate.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['CAPTURE_FILE'], 'a') as f:\n"
            "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "print('simulated training', flush=True)\n"
            "sys.exit(7 if '--policy.num_vlm_prompt_tokens=2' in sys.argv else 0)\n"
        )
        trainer = bin_dir / "lerobot-train"
        trainer.write_text("#!/bin/sh\nexit 0\n")
        for path in (accelerate, trainer):
            path.chmod(0o755)
        cache_env = dict.fromkeys(
            ("TMPDIR", "TMP", "TEMP", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "HF_HOME"),
            str(self.work / "cache"),
        )
        result = self.run_script(
            "0",
            VLM_PROMPT_TOKEN_COUNTS="1 2 8 16",
            DRY_RUN="false",
            PATH=f"{bin_dir}{os.pathsep}{self.env['PATH']}",
            CAPTURE_FILE=str(capture),
            **cache_env,
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        launched = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(len(launched), 2)
        self.assertIn("--policy.num_vlm_prompt_tokens=1", launched[0])
        self.assertIn("--policy.num_vlm_prompt_tokens=2", launched[1])
        self.assertEqual(
            sorted(path.name for path in (self.work / "logs").glob("*.log")),
            ["pi05-vlm_only-vlm1-seed0.log", "pi05-vlm_only-vlm2-seed0.log"],
        )


if __name__ == "__main__":
    unittest.main()
