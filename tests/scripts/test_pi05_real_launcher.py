"""Exercise real shell dispatch using metadata fixtures, without loading ML dependencies."""

import contextlib
import importlib.util
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/training/train_pi05_real.sh"
AUTODL = ROOT / "examples/training/train_piper_autodl.sh"
FIT = ROOT / "examples/training/train_piper_fit.sh"
SPEC = importlib.util.spec_from_file_location("real_launcher", LAUNCHER.with_suffix(".py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RealLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pi05 real test ")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.env = dict(
            os.environ,
            WORK_ROOT=str(self.work),
            DRY_RUN="true",
            PYTHON=sys.executable,
            GPU_IDS="0",
            NUM_PROCESSES="1",
        )
        for name in (
            "DATASET_BASE",
            "PRETRAINED_PATH",
            "TOKENIZER_PATH",
            "OUTPUT_ROOT",
            "LOG_ROOT",
            "FLOW_STEPS",
            "CHUNK_SIZE",
            "MASKED_STEPS",
            "N_ACTION_STEPS",
            "NORMALIZATION_MODE",
            "SAVE_FREQ",
            "SAVE_STEPS",
            "KEEP_PRETRAIN_CHECKPOINT",
            "BATCH_SIZE",
            "RUN_NAME",
            "FIT_EPISODES",
            "COMPILE_MODEL",
        ):
            self.env.pop(name, None)
        for filename in (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ):
            self.touch(self.work / "models/pi05_libero_base" / filename)
        for filename in ("tokenizer_config.json", "tokenizer.json"):
            self.touch(self.work / "models/google/paligemma-3b-pt-224" / filename)
        for folder in MODULE.TASKS.values():
            self.make_dataset(folder)

    @staticmethod
    def touch(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def make_dataset(self, folder):
        root = self.work / "datasets/May-pick-and-place" / folder
        for filename in (
            "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-000.parquet",
            "data/chunk-000/file-000.parquet",
            "videos/observation.images.wrist/chunk-000/file-000.mp4",
        ):
            self.touch(root / filename)
        info = {
            "codebase_version": "v3.0",
            "fps": 30,
            "total_episodes": 2,
            "total_frames": 120,
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [7]},
                "action": {"dtype": "float32", "shape": [7]},
                "observation.images.wrist": {"dtype": "video", "shape": [240, 320, 3]},
            },
        }
        (root / "meta/info.json").write_text(json.dumps(info))
        stats = {
            key: {"q01": [0.0] * 7, "q99": [1.0] * 7, "mean": [0.0] * 7, "std": [1.0] * 7}
            for key in ("observation.state", "action")
        }
        (root / "meta/stats.json").write_text(json.dumps(stats))

    def run_launcher(self, *args, **env):
        return subprocess.run(
            ["bash", str(LAUNCHER), *args],
            env=dict(self.env, **env),
            capture_output=True,
            text=True,
            check=False,
        )

    def run_autodl(self, *args, script=AUTODL, **env):
        auto_env = dict(self.env, RUN_GROUP="retrain-test")
        for name in (
            "GPU_IDS",
            "NUM_PROCESSES",
            "DTYPE",
            "MIXED_PRECISION",
            "CABO_RATIO",
            "NTK_SAVE_STAGE_SNAPSHOTS",
            "WANDB_ENABLE",
            "NUM_WORKERS",
            "WANDB_PROJECT",
            "WANDB_MODE",
            "ACTION_EVAL_FREQ",
            "ACTION_EVAL_SAMPLES",
            "ACTION_EVAL_SEED",
            "ACTION_UPDATE_FREQ",
            "EVAL_SPLIT",
        ):
            auto_env.pop(name, None)
        return subprocess.run(
            ["bash", str(script), *args],
            env=dict(auto_env, **env),
            capture_output=True,
            text=True,
            check=False,
        )

    def command_args(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]

    def test_all_dispatches_four_isolated_runs_and_preserves_quoted_args(self):
        result = self.run_launcher("all", "full_reference", "2", "--wandb.notes=real robot run")
        commands = self.command_args(result)
        self.assertEqual(len(commands), 4)
        outputs = []
        for command, folder in zip(commands, MODULE.TASKS.values(), strict=True):
            self.assertIn(f"--dataset.root={self.work / 'datasets/May-pick-and-place' / folder}", command)
            self.assertIn(f"--dataset.repo_id=may-pick-and-place/{folder}", command)
            self.assertIn("--steps=3000", command)
            self.assertIn("--policy.next_action_pretrain_steps=1000", command)
            self.assertIn("--policy.next_action_bridge_steps=250", command)
            self.assertIn("--wandb.notes=real robot run", command)
            self.assertIn("--save_freq=500", command)
            outputs.extend(item for item in command if item.startswith("--output_dir="))
        self.assertEqual(len(set(outputs)), 4)
        self.assertFalse((self.work / "chkpt").exists())
        self.assertFalse((self.work / "logs").exists())

    def test_variants_share_existing_recipe(self):
        for variant, pretrain, bridge, cabo in (
            ("no_bridge", 1000, 0, "true"),
            ("direct_dual", 0, 0, "true"),
            ("dual_prompt_only", 0, 0, "false"),
            ("vlm_only", 0, 0, "false"),
            ("action_only", 0, 0, "false"),
            ("no_cabo", 1000, 250, "false"),
        ):
            with self.subTest(variant=variant):
                (command,) = self.command_args(self.run_launcher("1", variant, "0"))
                self.assertIn(f"--policy.next_action_pretrain_steps={pretrain}", command)
                self.assertIn(f"--policy.next_action_bridge_steps={bridge}", command)
                self.assertIn(f"--policy.cabo_enabled={cabo}", command)

    def test_short_chunks_multi_gpu_and_full_task_name(self):
        (command,) = self.command_args(
            self.run_launcher(
                MODULE.TASKS["3"], "full_reference", "0", CHUNK_SIZE="8", GPU_IDS="0,1", NUM_PROCESSES="2"
            )
        )
        for arg in (
            "--policy.chunk_size=8",
            "--policy.n_action_steps=8",
            "--policy.next_action_masked_steps=6",
            "--multi_gpu",
            "--num_processes=2",
            "CUDA_VISIBLE_DEVICES=0,1",
        ):
            self.assertIn(arg, command)

    def test_checks_all_datasets_before_first_launch(self):
        root = self.work / "datasets/May-pick-and-place" / MODULE.TASKS["4"]
        (root / "meta/tasks.parquet").unlink()
        result = self.run_launcher("all")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("meta/tasks.parquet", result.stderr)
        self.assertNotIn("Launching:", result.stdout)

    def test_quantile_failure_and_explicit_mean_std(self):
        path = self.work / "datasets/May-pick-and-place" / MODULE.TASKS["1"] / "meta/stats.json"
        stats = json.loads(path.read_text())
        del stats["action"]["q01"]
        path.write_text(json.dumps(stats))
        result = self.run_launcher("1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing action.q01", result.stderr)
        (command,) = self.command_args(self.run_launcher("1", NORMALIZATION_MODE="MEAN_STD"))
        norm_arg = next(item for item in command if item.startswith("--policy.normalization_mapping="))
        self.assertEqual(json.loads(norm_arg.split("=", 1)[1])["ACTION"], "MEAN_STD")

    def test_existing_priming_output_and_invalid_gpu_configuration(self):
        result = self.run_launcher("1", GPU_IDS="0,1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NUM_PROCESSES", result.stderr)
        name = f"pi05-may-{MODULE.TASKS['1']}-full_reference-seed0_next_action_pretrain"
        (self.work / "chkpt/2601-lerobot/prompt-ablation-real" / name).mkdir(parents=True)
        result = self.run_launcher("1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)

    def test_feature_and_path_overrides_cannot_bypass_preflight(self):
        result = self.run_launcher("1", "full_reference", "0", "--policy.path=/wrong/model")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("environment variables", result.stderr)

    def test_autodl_defaults_train_four_separate_fp32_models(self):
        commands = self.command_args(self.run_autodl())
        self.assertEqual(len(commands), 4)
        for command, folder in zip(commands, MODULE.TASKS.values(), strict=True):
            for arg in (
                "CUDA_VISIBLE_DEVICES=0",
                "--num_processes=1",
                "--mixed_precision=no",
                "--policy.dtype=float32",
                "--steps=12000",
                "--save_freq=3000",
                "--save_steps=[6000,9000,12000]",
                "--batch_size=8",
                "--policy.num_vlm_prompt_tokens=16",
                "--policy.num_prompt_tokens=16",
                "--policy.cabo_enabled=true",
                "--policy.cabo_prompt_update_ratio=2.0",
                "--policy.next_action_pretrain_steps=1000",
                "--policy.next_action_bridge_steps=250",
                "--policy.chunk_size=50",
                "--policy.n_action_steps=8",
                "--policy.next_action_masked_steps=40",
                "--policy.ntk_save_stage_snapshots=false",
                "--wandb.enable=true",
                "--wandb.project=piper-action-fit",
                "--wandb.mode=online",
                "--wandb.disable_artifact=true",
                "--piper_eval.freq=500",
                "--piper_eval.samples=16",
                "--piper_eval.update_freq=50",
                "--piper_eval.execution_steps=8",
                "--dataset.eval_split=0",
                f"--policy.pretrained_path={self.work / 'models/pi05_libero_base'}",
                f"--dataset.root={self.work / 'datasets/May-pick-and-place' / folder}",
                f"--output_dir={self.work / 'chkpt/2601-lerobot/piper/retrain-test' / f'pi05-may-{folder}-full_reference-seed0'}",
            ):
                self.assertIn(arg, command)
            self.assertNotIn("--multi_gpu", command)
        self.assertFalse((self.work / "chkpt").exists())
        self.assertFalse((self.work / "logs").exists())

    def test_autodl_single_gpu_precision_overrides_and_quoted_paths(self):
        (command,) = self.command_args(
            self.run_autodl(
                "3",
                "no_bridge",
                "7",
                "--wandb.notes=AutoDL tennis rerun",
                GPU_IDS="1",
                DTYPE="bfloat16",
                FLOW_STEPS="3000",
                SAVE_STEPS="[3000]",
                BATCH_SIZE="4",
                N_ACTION_STEPS="50",
                OUTPUT_ROOT=str(self.work / "custom output"),
            )
        )
        for arg in (
            "CUDA_VISIBLE_DEVICES=1",
            "--num_processes=1",
            "--mixed_precision=bf16",
            "--policy.dtype=bfloat16",
            "--steps=3000",
            "--save_steps=[3000]",
            "--batch_size=4",
            "--seed=7",
            "--policy.n_action_steps=50",
            "--policy.next_action_bridge_steps=0",
            "--wandb.notes=AutoDL tennis rerun",
        ):
            self.assertIn(arg, command)
        self.assertNotIn("--multi_gpu", command)
        output = next(arg for arg in command if arg.startswith("--output_dir="))
        self.assertIn(str(self.work / "custom output"), output)

    def test_offline_wandb_and_action_diagnostic_overrides(self):
        (command,) = self.command_args(
            self.run_autodl(
                "4",
                WANDB_MODE="offline",
                WANDB_PROJECT="my-piper",
                ACTION_EVAL_FREQ="100",
                ACTION_EVAL_SAMPLES="8",
                ACTION_EVAL_SEED="3",
                ACTION_UPDATE_FREQ="25",
                EVAL_SPLIT="0.1",
            )
        )
        for arg in (
            "--wandb.mode=offline",
            "--wandb.project=my-piper",
            "--piper_eval.freq=100",
            "--piper_eval.samples=8",
            "--piper_eval.seed=3",
            "--piper_eval.update_freq=25",
            "--dataset.eval_split=0.1",
        ):
            self.assertIn(arg, command)
        (command,) = self.command_args(
            self.run_autodl("4", WANDB_ENABLE="false", ACTION_EVAL_FREQ="0", ACTION_UPDATE_FREQ="0")
        )
        self.assertIn("--wandb.enable=false", command)
        self.assertIn("--piper_eval.freq=0", command)
        self.assertIn("--piper_eval.update_freq=0", command)

    def test_fit_profiles_preserve_data_and_change_explicit_training_controls(self):
        for profile, lr, cabo, projections in (
            ("reference", "0.000025", "true", "false"),
            ("prompt_lr", "0.0001", "true", "false"),
            ("no_cabo", "0.0001", "false", "false"),
            ("projections", "0.0001", "false", "true"),
        ):
            with self.subTest(profile=profile):
                (command,) = self.command_args(self.run_autodl("4", profile, "0", script=FIT))
                for arg in (
                    "--steps=2000",
                    "--save_steps=[2000]",
                    "--dataset.episodes=[0]",
                    "--dataset.image_transforms.enable=false",
                    "--log_freq=50",
                    "--policy.compile_model=false",
                    "--policy.chunk_size=50",
                    "--policy.n_action_steps=8",
                    "--policy.next_action_pretrain_steps=1000",
                    "--policy.next_action_bridge_steps=250",
                    f"--policy.optimizer_lr={lr}",
                    f"--policy.cabo_enabled={cabo}",
                    f"--policy.train_action_projections={projections}",
                ):
                    self.assertIn(arg, command)
        self.assertFalse((self.work / "chkpt").exists())

    def test_fit_full_data_schedule_multi_gpu_and_default_profile(self):
        (command,) = self.command_args(
            self.run_autodl(
                "4",
                script=FIT,
                FIT_EPISODES="all",
                FLOW_STEPS="12000",
                SAVE_STEPS="[6000,9000,12000]",
                GPU_IDS="0,1",
                OUTPUT_ROOT=str(self.work / "custom fit output"),
            )
        )
        for arg in (
            "--steps=12000",
            "--save_steps=[6000,9000,12000]",
            "--num_processes=2",
            "--multi_gpu",
            "--policy.train_action_projections=true",
            "--policy.cabo_enabled=false",
        ):
            self.assertIn(arg, command)
        self.assertFalse(any(arg.startswith("--dataset.episodes=") for arg in command))
        output = next(arg for arg in command if arg.startswith("--output_dir="))
        self.assertIn(str(self.work / "custom fit output"), output)

    def test_unknown_fit_profile_fails_before_launch(self):
        result = self.run_autodl("4", "unknown", script=FIT)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown profile", result.stderr)
        self.assertNotIn("Launching:", result.stdout)

    def test_autodl_invalid_settings_fail_before_launch(self):
        for env in (
            {"DTYPE": "float32", "MIXED_PRECISION": "bf16"},
            {"GPU_IDS": "0", "NUM_PROCESSES": "2"},
            {"FLOW_STEPS": "0"},
            {"SAVE_FREQ": "0"},
            {"BATCH_SIZE": "-1"},
            {"RUN_GROUP": "../old-run"},
            {"SAVE_STEPS": "[0,6000]"},
            {"SAVE_STEPS": "[6000,13000]"},
            {"SAVE_STEPS": "[true,6000]"},
            {"SAVE_STEPS": "6000"},
            {"SAVE_STEPS": "bad-json"},
            {"KEEP_PRETRAIN_CHECKPOINT": "invalid"},
            {"NTK_SAVE_STAGE_SNAPSHOTS": "true"},
        ):
            with self.subTest(env=env):
                result = self.run_autodl("all", **env)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Launching:", result.stdout)

    def test_explicit_schedule_takes_precedence_and_supports_final_only(self):
        for schedule, expected in (("[9000,6000,6000]", "[6000,9000]"), ("[]", "[]")):
            with self.subTest(schedule=schedule):
                (command,) = self.command_args(self.run_autodl("1", SAVE_STEPS=schedule, SAVE_FREQ="1"))
                self.assertIn(f"--save_steps={expected}", command)
        (command,) = self.command_args(
            self.run_autodl("1", "dual_prompt_only", FLOW_STEPS="2", SAVE_STEPS="[2]")
        )
        self.assertIn("--steps=2", command)
        self.assertIn("--save_steps=[2]", command)

    def simulate_training(self, fail_task=None, keep_pretrain="true", incomplete_final=False):
        launched = []

        def train(command, *, env, cwd, check):
            launched.append(env)
            output = Path(env["OUTPUT_ROOT"]) / env["RUN_NAME"]
            output.mkdir(parents=True)
            pretrain = output.with_name(f"{output.name}_next_action_pretrain")
            self.touch(pretrain / "checkpoints/001000/pretrained_model/model.safetensors")
            for step in ("006000", "009000", "012000"):
                self.touch(output / "checkpoints" / step / "pretrained_model/config.json")
                if not incomplete_final or step != "012000":
                    self.touch(output / "checkpoints" / step / "pretrained_model/model.safetensors")
            failed = Path(env["DATASET_ROOT"]).name == MODULE.TASKS.get(fail_task)
            return subprocess.CompletedProcess(command, 17 if failed else 0)

        with (
            patch.dict(
                os.environ,
                dict(
                    self.env,
                    DRY_RUN="false",
                    FLOW_STEPS="12000",
                    SAVE_STEPS="[6000,9000,12000]",
                    KEEP_PRETRAIN_CHECKPOINT=keep_pretrain,
                    NTK_SAVE_STAGE_SNAPSHOTS="false",
                ),
                clear=True,
            ),
            patch.object(sys, "argv", [str(LAUNCHER), "all"]),
            patch.object(MODULE, "check_cuda"),
            patch.object(MODULE.subprocess, "run", side_effect=train),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = MODULE.main()
        return code, launched

    def test_completed_tasks_export_their_own_deployment_metadata(self):
        for number, folder in MODULE.TASKS.items():
            path = self.work / "datasets/May-pick-and-place" / folder / "meta/info.json"
            info = json.loads(path.read_text())
            info["total_frames"] = int(number) * 120
            path.write_text(json.dumps(info))
        code, launched = self.simulate_training()
        self.assertEqual(code, 0)
        self.assertEqual(len(launched), 4)
        for env in launched:
            output = Path(env["OUTPUT_ROOT"]) / env["RUN_NAME"] / "dataset_info.json"
            source = Path(env["DATASET_ROOT"]) / "meta/info.json"
            self.assertEqual(output.read_bytes(), source.read_bytes())

    def test_failed_training_stops_queue_and_does_not_export_success_metadata(self):
        code, launched = self.simulate_training(fail_task="2")
        self.assertEqual(code, 17)
        self.assertEqual(len(launched), 2)
        first, second = launched
        self.assertTrue((Path(first["OUTPUT_ROOT"]) / first["RUN_NAME"] / "dataset_info.json").is_file())
        self.assertFalse((Path(second["OUTPUT_ROOT"]) / second["RUN_NAME"] / "dataset_info.json").exists())

    def test_success_removes_only_temporary_weights_and_failure_keeps_recovery(self):
        code, launched = self.simulate_training(fail_task="2", keep_pretrain="false")
        self.assertEqual(code, 17)
        self.assertEqual(len(launched), 2)
        for index, env in enumerate(launched):
            output = Path(env["OUTPUT_ROOT"]) / env["RUN_NAME"]
            pretrain = output.with_name(f"{output.name}_next_action_pretrain")
            self.assertEqual((pretrain / "checkpoints").exists(), index == 1)
            for step in ("006000", "009000", "012000"):
                self.assertTrue(
                    (output / "checkpoints" / step / "pretrained_model/model.safetensors").exists()
                )

    def test_cleanup_requires_a_complete_final_model(self):
        with self.assertRaisesRegex(ValueError, "Keeping pretraining weights"):
            self.simulate_training(keep_pretrain="false", incomplete_final=True)
        output = self.work / "chkpt/2601-lerobot/prompt-ablation-real"
        self.assertEqual(len(list(output.glob("*_next_action_pretrain/checkpoints/001000"))), 1)

    def test_cleanup_rejects_symlinked_stage_directory(self):
        output = self.work / "output"
        previous = self.work / "previous_run"
        self.touch(previous / "checkpoints/001000/pretrained_model/model.safetensors")
        output.with_name("output_next_action_pretrain").symlink_to(previous, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "linked pretraining directory"):
            MODULE.cleanup_pretrain_checkpoint(output, 12000)
        self.assertTrue((previous / "checkpoints/001000/pretrained_model/model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
