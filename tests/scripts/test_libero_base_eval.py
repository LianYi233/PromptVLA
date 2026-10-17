"""Validate base evaluation without loading policy weights, CUDA, or LIBERO."""

import importlib.util
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "libero_base_eval", ROOT / "src/lerobot/scripts/libero_base_eval.py"
)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


class LiberoBaseEvalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="libero base test ")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.model = self.work / "base model"
        self.model.mkdir()
        self.tokenizer = self.work / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text("{}")
        (self.model / "config.json").write_text(json.dumps({"type": "pi05"}))
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            (self.model / name).write_text("{}")
        self.save_weights()
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "BASE_MODEL_PATH": str(self.model),
            "TOKENIZER_PATH": str(self.tokenizer),
            "BASE_OUTPUT": str(self.work / "results"),
            "DRY_RUN": "true",
        }

    def save_weights(self, **prompts):
        save_file(
            {"model.weight": np.ones((2, 2), dtype=np.float32), **prompts}, self.model / "model.safetensors"
        )

    def run_script(self, *args, **overrides):
        return subprocess.run(
            ["bash", str(ROOT / "run_eval_libero_base.sh"), *args],
            env={**self.env, **overrides},
            capture_output=True,
            text=True,
            check=False,
        )

    def test_base_forces_zero_prompts_even_if_config_has_old_defaults(self):
        for fields in ({}, {"num_vlm_prompt_tokens": 16, "num_prompt_tokens": 16, "cabo_enabled": True}):
            with self.subTest(fields=fields):
                (self.model / "config.json").write_text(json.dumps({"type": "pi05", **fields}))
                before = {p.name: p.read_bytes() for p in self.model.iterdir()}
                result = self.run_script("all")
                self.assertEqual(result.returncode, 0, result.stderr)
                (command,) = [
                    shlex.split(s) for s in result.stdout.splitlines() if s.startswith("python -u - ")
                ]
                for argument in (
                    f"--policy.path={self.model}",
                    "--policy.num_vlm_prompt_tokens=0",
                    "--policy.num_prompt_tokens=0",
                    "--policy.cabo_enabled=false",
                    "--policy.training_stage=flow",
                    "--policy.use_peft=false",
                    "--policy.n_action_steps=10",
                    "--eval.n_episodes=10",
                    "--env.task=libero_spatial,libero_object,libero_goal,libero_10",
                ):
                    self.assertIn(argument, command)
                self.assertIn("pi05_libero_base-no_prompt-base-libero-all4-resume", result.stdout)
                self.assertNotIn("/003000/", result.stdout)
                self.assertFalse((self.work / "results").exists())
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.model.iterdir()})

    def test_rejects_learned_prompt_weights_before_evaluation(self):
        for key in ("model.vlm_prompt_tokens.weight", "model.prompt_tokens.weight"):
            with self.subTest(key=key):
                self.save_weights(**{key: np.zeros((1, 4), dtype=np.float32)})
                result = self.run_script("all")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("nonempty prompt bank", result.stderr)
                self.assertNotIn("python -u - ", result.stdout)

    def test_accepts_explicit_empty_banks(self):
        self.save_weights(
            **{
                "model.vlm_prompt_tokens.weight": np.zeros((0, 4), dtype=np.float32),
                "model.prompt_tokens.weight": np.zeros((0, 4), dtype=np.float32),
            }
        )
        report = BASE.validate_base_checkpoint(self.model, self.tokenizer)
        self.assertEqual(report["effective_prompt_counts"], {"vlm": 0, "action": 0})

    def test_missing_processor_and_wrong_model_type_fail_preflight(self):
        path = self.model / "policy_postprocessor.json"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing base checkpoint file"):
            BASE.validate_base_checkpoint(self.model)
        path.write_text("{}")
        (self.model / "config.json").write_text(json.dumps({"type": "pi0"}))
        with self.assertRaisesRegex(ValueError, "Expected PI05 config"):
            BASE.validate_base_checkpoint(self.model)

    def test_suite_episode_override_and_ambiguous_path_rejection(self):
        result = self.run_script("libero_10", EPISODES_PER_TASK="2", GPU_ID="3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--env.task=libero_10", result.stdout)
        self.assertIn("--eval.n_episodes=2", result.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=3", result.stdout)
        for overrides in ({"BASE_CKPT": "/wrong/path"}, {"VLM_PROMPT_TOKENS": "1"}):
            result = self.run_script("all", **overrides)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unset BASE_CKPT and VLM_PROMPT_TOKENS", result.stderr)

    def test_tokenizer_override_preserves_saved_pipeline_and_other_overrides(self):
        original = Mock(return_value=("saved_pre", "saved_post"))
        evaluator = SimpleNamespace(make_pre_post_processors=original)
        manifest = BASE.install_base_evaluation(evaluator, str(self.tokenizer))
        overrides = {
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": {"old": "new"}},
            "tokenizer_processor": {"max_length": 200},
        }
        kwargs = {
            "policy_cfg": object(),
            "pretrained_path": str(self.model),
            "preprocessor_overrides": overrides,
            "postprocessor_overrides": {"device_processor": {"device": "cpu"}},
        }
        self.assertEqual(evaluator.make_pre_post_processors(**kwargs), ("saved_pre", "saved_post"))
        expected = {
            **overrides,
            "tokenizer_processor": {"max_length": 200, "tokenizer_name": str(self.tokenizer)},
        }
        original.assert_called_once_with(**{**kwargs, "preprocessor_overrides": expected})
        self.assertEqual(overrides["tokenizer_processor"], {"max_length": 200})
        self.assertEqual(manifest["base_evaluation"]["tokenizer_files"][0][0], "tokenizer.json")
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)

    def test_no_tokenizer_override_leaves_factory_unchanged(self):
        original = Mock()
        evaluator = SimpleNamespace(make_pre_post_processors=original)
        manifest = BASE.install_base_evaluation(evaluator)
        self.assertIs(evaluator.make_pre_post_processors, original)
        self.assertIsNone(manifest["base_evaluation"]["tokenizer_path"])


if __name__ == "__main__":
    unittest.main()
