"""Check continuation controls and normalization parity without multi-GB weights or CUDA."""

import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lerobot.scripts.libero_finetuned_prompt import (  # noqa: E402
    install_finetuned_evaluation,
    validate_experiment_checkpoint,
)
from lerobot.scripts.pretrained_normalization import preserve_pretrained_normalization  # noqa: E402

TRAIN = ROOT / "examples/training/train_pi05_finetuned_prompt.sh"
EVAL = ROOT / "run_eval_libero_finetuned_prompt.sh"
RUN = "pi05-ftv044-dual-vlm16-act16-seed0"


class FinetunedPromptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="finetuned prompt ")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.source = self.work / "source model"
        self.tokenizer = self.work / "tokenizer"
        self.tokenizer.mkdir()
        (self.work / "dataset").mkdir()
        self.source_config = {
            "type": "pi05",
            "dtype": "bfloat16",
            "normalization_mapping": {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
        }
        self.checkpoint(self.source, prompts=False)
        for step in (0, 3000):
            self.checkpoint(self.model(step), prompts=True)
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "DRY_RUN": "true",
            "PRETRAINED_PATH": str(self.source),
            "DATASET_ROOT": str(self.work / "dataset"),
            "TOKENIZER_PATH": str(self.tokenizer),
            "OUTPUT_ROOT": str(self.work / "train"),
            "LOG_ROOT": str(self.work / "logs"),
            "CKPT_ROOT": str(self.work / "checkpoints"),
            "BASE_OUTPUT": str(self.work / "results"),
        }

    def model(self, step):
        return self.work / "checkpoints" / RUN / "checkpoints" / f"{step:06d}" / "pretrained_model"

    def checkpoint(self, directory, prompts):
        directory.mkdir(parents=True)
        config = copy.deepcopy(self.source_config)
        weights = {"model.backbone.weight": np.ones(1, dtype=np.float32)}
        if prompts:
            config.update(
                num_vlm_prompt_tokens=16,
                num_prompt_tokens=16,
                training_stage="flow",
                cabo_enabled=False,
                train_action_projections=False,
                use_peft=False,
                next_action_pretrain_steps=0,
                next_action_bridge_steps=0,
            )
            for key in ("vlm_prompt_tokens", "prompt_tokens"):
                weights[f"model.{key}.weight"] = np.ones((16, 4), dtype=np.float32)
        (directory / "config.json").write_text(json.dumps(config))
        save_file(weights, directory / "model.safetensors")
        for name, registry, sizes in (
            ("preprocessor", "normalizer_processor", {"observation.state": 8, "action": 7}),
            ("postprocessor", "unnormalizer_processor", {"action": 7}),
        ):
            state = f"{registry}.safetensors"
            features = {
                key: {"type": "ACTION" if key == "action" else "STATE", "shape": [size]}
                for key, size in sizes.items()
            }
            pipeline = {
                "steps": [
                    {
                        "registry_name": registry,
                        "state_file": state,
                        "config": {
                            "features": features,
                            "norm_map": config["normalization_mapping"],
                            "eps": 1e-8,
                        },
                    }
                ]
            }
            (directory / f"policy_{name}.json").write_text(json.dumps(pipeline))
            stats = {
                f"{key}.{stat}": np.ones(size, dtype=np.float32)
                for key, size in sizes.items()
                for stat in ("mean", "std")
            }
            save_file(stats, directory / state)

    def run_script(self, script, *args, **env):
        return subprocess.run(
            ["bash", str(script), *args], env={**self.env, **env}, text=True, capture_output=True, check=False
        )

    def commands(self, result, prefix):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [shlex.split(line) for line in result.stdout.splitlines() if line.startswith(prefix)]

    def test_training_inherits_source_config_and_locks_dual_prompt_recipe(self):
        (command,) = self.commands(self.run_script(TRAIN, "0"), "env ")
        for arg in (
            f"--policy.path={self.source}",
            "--policy.num_vlm_prompt_tokens=16",
            "--policy.num_prompt_tokens=16",
            "--policy.train_action_projections=false",
            "--policy.use_peft=false",
            "--policy.cabo_enabled=false",
            "--policy.next_action_pretrain_steps=0",
            "--policy.next_action_bridge_steps=0",
            "--policy.ntk_save_stage_snapshots=true",
            "--preserve_pretrained_normalization=true",
            "--steps=3000",
            "--save_freq=1000",
            "--wandb.project=prompt-learning",
            "--wandb.mode=online",
        ):
            self.assertIn(arg, command)
        self.assertFalse(any(arg.startswith(("--policy.type=", "--policy.dtype=")) for arg in command))
        self.assertFalse((self.work / "train").exists())

    def test_prompt_counts_and_training_overrides_are_explicit(self):
        (command,) = self.commands(
            self.run_script(
                TRAIN,
                "2",
                "--wandb.mode=offline",
                "--policy.optimizer_lr=0.00001",
                VLM_PROMPT_TOKENS="8",
                ACTION_PROMPT_TOKENS="4",
                FLOW_STEPS="2000",
            ),
            "env ",
        )
        for arg in (
            "--job_name=pi05-ftv044-dual-vlm8-act4-seed2",
            "--steps=2000",
            "--policy.num_vlm_prompt_tokens=8",
            "--policy.num_prompt_tokens=4",
        ):
            self.assertIn(arg, command)
        for arg in (
            "--policy.train_action_projections=true",
            "--preserve_pretrained_normalization=false",
            "--policy.dtype=float32",
            "--peft.method=LORA",
            "--policy.path=wrong",
        ):
            result = self.run_script(TRAIN, "0", arg)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported override", result.stderr)

    def test_existing_output_is_not_overwritten(self):
        (self.work / "train" / RUN).mkdir(parents=True)
        result = self.run_script(TRAIN, "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)

    def test_training_failure_is_reported_through_log_pipeline(self):
        bin_dir = self.work / "bin"
        bin_dir.mkdir()
        for name, script in (
            ("accelerate", "#!/bin/sh\necho simulated-training-failure\nexit 7\n"),
            ("lerobot-train", "#!/bin/sh\nexit 0\n"),
        ):
            path = bin_dir / name
            path.write_text(script)
            path.chmod(0o755)
        caches = dict.fromkeys(
            ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"), str(self.work / "cache")
        )
        result = self.run_script(
            TRAIN, "0", DRY_RUN="false", PATH=f"{bin_dir}{os.pathsep}{self.env['PATH']}", **caches
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("simulated-training-failure", (self.work / "logs" / f"{RUN}.log").read_text())

    def test_saved_prompt_evaluation_overrides_only_tokenizer_location(self):
        original = Mock(return_value=("saved_pre", "saved_post"))
        evaluator = SimpleNamespace(make_pre_post_processors=original)
        manifest = install_finetuned_evaluation(evaluator, self.tokenizer)
        result = evaluator.make_pre_post_processors(
            pretrained_path=self.model(0), preprocessor_overrides={"device_processor": {"device": "cuda"}}
        )
        self.assertEqual(result, ("saved_pre", "saved_post"))
        self.assertEqual(
            original.call_args.kwargs["preprocessor_overrides"],
            {
                "device_processor": {"device": "cuda"},
                "tokenizer_processor": {"tokenizer_name": str(self.tokenizer)},
            },
        )
        self.assertNotIn("prompt_counts", manifest["finetuned_prompt_evaluation"])

    def test_all_conditions_use_matched_evaluation_settings_and_separate_paths(self):
        commands = self.commands(self.run_script(EVAL, "all", "all", "0", EVAL_SEED="42"), "python -u - ")
        self.assertEqual(len(commands), 3)
        paths = []
        for command, path in zip(commands, (self.source, self.model(0), self.model(3000)), strict=True):
            self.assertIn(f"--policy.path={path}", command)
            for arg in ("--policy.n_action_steps=10", "--eval.n_episodes=10", "--seed=42"):
                self.assertIn(arg, command)
            paths.append(next(arg for arg in command if arg.startswith("--output_dir=")))
        self.assertEqual(len(set(paths)), 3)
        self.assertIn("--policy.num_vlm_prompt_tokens=0", commands[0])
        self.assertIn("--policy.num_prompt_tokens=0", commands[0])
        self.assertFalse(any(arg.startswith("--policy.num_prompt_tokens") for arg in commands[1]))
        self.assertFalse((self.work / "results").exists())

    def test_changed_statistics_stop_all_evaluation_before_launch(self):
        state = self.model(3000) / "normalizer_processor.safetensors"
        tensors = load_file(state)
        tensors["action.mean"][0] += 1
        save_file(tensors, state)
        result = self.run_script(EVAL, "all", "all", "0", DRY_RUN="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("statistics differ", result.stderr)
        self.assertNotIn("Evaluating finetuned", result.stdout)
        self.assertFalse((self.work / "results").exists())

    def test_missing_source_stats_and_wrong_prompt_weights_fail(self):
        save_file({"model.backbone.weight": np.ones(1)}, self.model(0) / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "Missing or mismatched learned prompt"):
            validate_experiment_checkpoint(self.source, self.model(0))
        (self.source / "normalizer_processor.safetensors").unlink()
        with self.assertRaises(OSError):
            validate_experiment_checkpoint(self.source)

    def test_modes_and_model_dtype_must_match_source(self):
        config_path = self.model(0) / "config.json"
        config = json.loads(config_path.read_text())
        config["dtype"] = "float32"
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "differs from source in dtype"):
            validate_experiment_checkpoint(self.source, self.model(0))

    def test_preservation_removes_stat_overrides_without_mutating_existing_recipe(self):
        kwargs = {
            "dataset_stats": {"action": {"mean": [99]}},
            "preprocessor_overrides": {
                "normalizer_processor": {"stats": "new stats", "norm_map": "QUANTILES"},
                "device_processor": {"device": "cuda"},
                "tokenizer_processor": {"tokenizer_name": "local"},
                "rename_observations_processor": {"rename_map": {}},
            },
            "postprocessor_overrides": {
                "unnormalizer_processor": {"stats": "new stats"},
                "absolute_actions_processor": {"enabled": True},
            },
        }
        before = copy.deepcopy(kwargs)
        preserved = preserve_pretrained_normalization(kwargs, self.source)
        self.assertEqual(kwargs, before)
        self.assertNotIn("dataset_stats", preserved)
        self.assertNotIn("normalizer_processor", preserved["preprocessor_overrides"])
        self.assertNotIn("unnormalizer_processor", preserved["postprocessor_overrides"])
        self.assertEqual(
            preserved["preprocessor_overrides"]["tokenizer_processor"], {"tokenizer_name": "local"}
        )
        self.assertEqual(
            preserved["postprocessor_overrides"], {"absolute_actions_processor": {"enabled": True}}
        )
        with self.assertRaisesRegex(ValueError, "requires a pretrained checkpoint"):
            preserve_pretrained_normalization(kwargs, None)


if __name__ == "__main__":
    unittest.main()
