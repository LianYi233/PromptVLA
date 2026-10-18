"""Test two-source GPU dispatch and normalization-matched controls without CUDA."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lerobot.scripts.libero_vlm_only import prepare_baseline, validate_checkpoint  # noqa: E402

TRAIN = ROOT / "examples/training/train_pi05_vlm_only.sh"
PAIR = ROOT / "examples/training/train_pi05_vlm_only_pair.sh"
SWEEP = ROOT / "examples/training/train_pi05_vlm_only_token_sweep.sh"
RANK_SWEEP = ROOT / "examples/training/train_pi05_vlm_only_rank_sweep.sh"
EVAL = ROOT / "run_eval_libero_vlm_only.sh"
EVAL_SWEEP = ROOT / "run_eval_libero_vlm_only_token_sweep.sh"
PDET_EVAL_SWEEP = ROOT / "run_eval_libero_vlm_only_pdet_sweep.sh"


class VLMOnlyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vlm only ")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.dataset = self.work / "dataset"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "meta/stats.json").write_text(
            json.dumps(
                {
                    key: {"q01": [0] * size, "q99": [1] * size}
                    for key, size in (("observation.state", 8), ("action", 7))
                }
            )
        )
        self.tokenizer = self.work / "tokenizer"
        self.tokenizer.mkdir()
        for kind in ("base", "finetuned"):
            self.make_model(self.source(kind), kind, source=True)
            for step in (0, 3000):
                self.make_model(self.model(kind, step), kind)
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "DRY_RUN": "true",
            "DATASET_ROOT": str(self.dataset),
            "TOKENIZER_PATH": str(self.tokenizer),
            "BASE_PRETRAINED_PATH": str(self.source("base")),
            "FINETUNED_PRETRAINED_PATH": str(self.source("finetuned")),
            "OUTPUT_ROOT": str(self.work / "training"),
            "LOG_ROOT": str(self.work / "logs"),
            "CKPT_ROOT": str(self.work / "checkpoints"),
            "BASE_OUTPUT": str(self.work / "eval"),
            "EVAL_LOG_ROOT": str(self.work / "eval/logs"),
        }

    def source(self, kind):
        return self.work / "sources" / kind

    def model(self, kind, step, tokens=1):
        return (
            self.work
            / "checkpoints"
            / f"pi05-{kind}-vlm{tokens}-seed0/checkpoints/{step:06d}/pretrained_model"
        )

    def make_model(self, path, kind, source=False, tokens=1):
        path.mkdir(parents=True)
        mode = "MEAN_STD" if kind == "finetuned" or source else "QUANTILES"
        norm = {"ACTION": mode, "STATE": mode, "VISUAL": "IDENTITY"}
        config = {
            "type": "pi05",
            "dtype": "bfloat16",
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "normalization_mapping": norm,
        }
        weights = {"model.backbone.weight": np.ones((4, 4), dtype=np.float32)}
        if not source:
            config.update(
                vlm_prompt_only=True,
                num_vlm_prompt_tokens=tokens,
                num_prompt_tokens=0,
                training_stage="flow",
                next_action_pretrain_steps=0,
                next_action_bridge_steps=0,
                cabo_enabled=False,
                train_action_projections=False,
                use_peft=False,
            )
            weights["model.vlm_prompt_tokens.weight"] = np.ones((tokens, 2048), dtype=np.float32)
            weights["model.prompt_tokens.weight"] = np.empty((0, 1024), dtype=np.float32)
            (path / "train_config.json").write_text(
                json.dumps({"policy": {"pretrained_path": str(self.source(kind))}})
            )
        (path / "config.json").write_text(json.dumps(config))
        save_file(weights, path / "model.safetensors")
        for filename, registry, features in (
            ("policy_preprocessor.json", "normalizer_processor", {"observation.state": 8, "action": 7}),
            ("policy_postprocessor.json", "unnormalizer_processor", {"action": 7}),
        ):
            missing_stats = source and kind == "base"
            feature_map = (
                {}
                if missing_stats
                else {
                    key: {"type": "ACTION" if key == "action" else "STATE", "shape": [size]}
                    for key, size in features.items()
                }
            )
            step = {
                "registry_name": registry,
                "config": {"eps": 1e-8, "norm_map": norm, "features": feature_map},
            }
            if not missing_stats:
                step["state_file"] = registry + ".safetensors"
                stats = {
                    f"{key}.{stat}": np.ones(size, dtype=np.float32)
                    for key, size in features.items()
                    for stat in (("mean", "std") if mode == "MEAN_STD" else ("q01", "q99"))
                }
                save_file(stats, path / step["state_file"])
            (path / filename).write_text(json.dumps({"steps": [step]}))

    def run_script(self, script, *args, **env):
        return subprocess.run(
            ["bash", str(script), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def make_eval_sweep(self):
        for kind in ("base", "finetuned"):
            for count in (2, 4):
                self.make_model(self.model(kind, 3000, count), kind, tokens=count)

    def install_fake_evaluator(self):
        bin_dir = self.work / "bin"
        bin_dir.mkdir()
        capture = self.work / "evaluations.jsonl"
        python = bin_dir / "python"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1:3] != ['-u', '-']:\n"
            f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
            "args = dict(arg.split('=', 1) for arg in sys.argv[3:])\n"
            "with open(os.environ['CAPTURE_FILE'], 'a') as f:\n"
            "    f.write(json.dumps({'gpu': os.environ['CUDA_VISIBLE_DEVICES'], **args}) + '\\n')\n"
            "name = Path(args['--policy.path']).parents[2].name\n"
            "print('progress-noise 50%\\rprogress-noise 100%', file=sys.stderr)\n"
            "print({'video_paths': ['/verbose/video-path.mp4'] * 30})\n"
            "if os.environ.get('LIBERO_PROGRESS_FD'):\n"
            "    line = '\\r' + os.environ['LIBERO_PROGRESS_LABEL'] + ': 50% 200/400 [ETA 10:00]'\n"
            "    os.write(int(os.environ['LIBERO_PROGRESS_FD']), line.encode())\n"
            "if name == os.environ.get('FAIL_RUN'): sys.exit(7)\n"
            "output = Path(args['--output_dir'])\n"
            "output.mkdir(parents=True, exist_ok=True)\n"
            "episodes = int(args['--eval.n_episodes'])\n"
            "tasks = [{'task_group': suite, 'task_id': tid,\n"
            "          'metrics': {'successes': [True] * (episodes - 1) + [False]}}\n"
            "         for suite in args['--env.task'].split(',') for tid in range(10)]\n"
            "if os.environ.get('BAD_RESULT') == 'missing': tasks.pop()\n"
            "if os.environ.get('BAD_RESULT') == 'duplicate': tasks[-1] = tasks[0]\n"
            "if os.environ.get('BAD_RESULT') == 'episodes': tasks[-1]['metrics']['successes'].pop()\n"
            "(output / 'eval_info.json').write_text(json.dumps({'per_task': tasks}))\n"
        )
        python.chmod(0o755)
        self.env.update(PATH=f"{bin_dir}{os.pathsep}{self.env['PATH']}", CAPTURE_FILE=str(capture))
        return capture

    def test_eval_sweep_dry_run_matches_four_final_checkpoints_without_writes(self):
        self.make_eval_sweep()
        result = self.run_script(EVAL_SWEEP, "both", "all", "0", GPU_ID="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [
            shlex.split(line) for line in result.stdout.splitlines() if line.startswith("python -u -")
        ]
        self.assertEqual(len(commands), 4)
        for (kind, count), command in zip(
            (("base", 2), ("base", 4), ("finetuned", 2), ("finetuned", 4)), commands, strict=True
        ):
            self.assertIn(f"--policy.path={self.model(kind, 3000, count)}", command)
            self.assertIn("--seed=1000", command)
            self.assertIn("--eval.n_episodes=10", command)
            self.assertIn("--policy.n_action_steps=10", command)
            self.assertIn("--env.task=libero_spatial,libero_object,libero_goal,libero_10", command)
        self.assertFalse((self.work / "eval").exists())

    def test_eval_sweep_checks_last_model_before_launching_first(self):
        self.make_eval_sweep()
        capture = self.install_fake_evaluator()
        config_path = self.model("finetuned", 3000, 4) / "config.json"
        config = json.loads(config_path.read_text())
        config["num_vlm_prompt_tokens"] = 2
        config_path.write_text(json.dumps(config))
        result = self.run_script(EVAL_SWEEP, DRY_RUN="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("Preflight failed: pi05-finetuned-vlm4", result.stderr)
        self.assertFalse(capture.exists())

    def test_eval_sweep_prints_only_fresh_rates_and_retains_noisy_logs(self):
        self.make_eval_sweep()
        capture = self.install_fake_evaluator()
        output = self.work / "eval"
        output.mkdir()
        log_root = Path(self.env["EVAL_LOG_ROOT"])
        log_root.mkdir(parents=True)
        stale = log_root / "eval_pi05-base-vlm2-seed0-libero-all4-summary.log"
        stale.write_text("STALE SUCCESS RATE: 12.34%\n")
        result = self.run_script(EVAL_SWEEP, DRY_RUN="false", GPU_ID="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        launches = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual(
            [Path(run["--policy.path"]).parents[2].name for run in launches],
            [f"pi05-{kind}-vlm{n}-seed0" for kind in ("base", "finetuned") for n in (2, 4)],
        )
        self.assertEqual({run["gpu"] for run in launches}, {"1"})
        self.assertEqual(result.stdout.count("overall: 90.00% (360/400)"), 4)
        self.assertEqual(result.stdout.count("libero_10: 90.00% (90/100)"), 4)
        self.assertEqual(len(result.stdout.strip().splitlines()), 27)  # 6 lines/model + blank separators
        self.assertEqual(result.stderr, "")
        for noise in ("STALE", "progress-noise", ".mp4", "per-task", "No step-0"):
            self.assertNotIn(noise, result.stdout)
        logs = list(log_root.glob("pi05-*.log"))  # one timestamped sweep log per model
        self.assertEqual(len(logs), 4)
        for log in logs:
            self.assertIn("progress-noise", log.read_text())
            self.assertIn("video-path.mp4", log.read_text())
        # A failed rerun must not print the previous valid eval_info.json as a new score.
        failed = self.run_script(EVAL_SWEEP, DRY_RUN="false", FAIL_RUN="pi05-base-vlm2-seed0")
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(failed.stdout, "")
        self.assertIn("Evaluation failed: pi05-base-vlm2", failed.stderr)
        self.assertEqual(len(capture.read_text().splitlines()), 5)

    def test_eval_sweep_rejects_partial_or_duplicate_results_and_stops(self):
        self.make_eval_sweep()
        capture = self.install_fake_evaluator()
        for index, bad in enumerate(("missing", "duplicate", "episodes"), 1):
            with self.subTest(bad=bad):
                result = self.run_script(EVAL_SWEEP, DRY_RUN="false", BAD_RESULT=bad)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("Result validation failed", result.stderr)
                self.assertEqual(len(capture.read_text().splitlines()), index)

    def test_eval_sweep_progress_stream_bypasses_logs_without_leaking_other_output(self):
        self.make_eval_sweep()
        self.install_fake_evaluator()
        result = self.run_script(EVAL_SWEEP, DRY_RUN="false", PROGRESS="true")
        self.assertEqual(result.returncode, 0, result.stderr)
        for index, (kind, count) in enumerate(
            (("base", 2), ("base", 4), ("finetuned", 2), ("finetuned", 4)), 1
        ):
            self.assertIn(f"[{index}/4] {kind}-vlm{count}: 50% 200/400 [ETA 10:00]", result.stderr)
        self.assertIn("loading model/environments", result.stderr)
        self.assertIn("\x1b[K", result.stderr)
        self.assertEqual(result.stdout.count("overall: 90.00% (360/400)"), 4)
        for noise in ("progress-noise", ".mp4", "per-task", "No step-0"):
            self.assertNotIn(noise, result.stdout + result.stderr)
        for log in (self.work / "eval/logs").glob("*.log"):
            self.assertNotIn("ETA 10:00", log.read_text())

    def commands(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [
            shlex.split(line)
            for line in result.stdout.splitlines()
            if line.startswith("env CUDA_VISIBLE_DEVICES=")
        ]

    def make_pdet_eval_model(self, case, published=True):
        name = "pi05-pdet-base-vlm2-" + "-".join(
            f"{key}{value}" for key, value in zip("PDET", case, strict=True)
        )
        checkpoint = self.work / "checkpoints" / name / "checkpoints/003000"
        model = checkpoint / "pretrained_model"
        self.make_model(model, "base", tokens=2)
        config_path = model / "train_config.json"
        config = json.loads(config_path.read_text())
        config.update(
            seed=0,
            pdet={
                "enabled": True,
                **dict(
                    zip(
                        ("prompt_seed", "data_seed", "noise_seed", "time_seed"),
                        map(int, case),
                        strict=True,
                    )
                ),
            },
        )
        config_path.write_text(json.dumps(config))
        if published:
            (checkpoint.parent / "last").symlink_to("003000")
        return name, checkpoint

    def test_pdet_eval_order_protocol_progress_and_quiet_results(self):
        cases = "1111 0000 0111 1011 1101 1110 1000".split()
        names = [self.make_pdet_eval_model(case)[0] for case in cases]
        capture = self.install_fake_evaluator()
        result = self.run_script(PDET_EVAL_SWEEP, DRY_RUN="false", PROGRESS="true")
        self.assertEqual(result.returncode, 0, result.stderr)
        launches = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual([Path(run["--policy.path"]).parents[2].name for run in launches], names)
        for run in launches:
            self.assertEqual(run["gpu"], "2")
            self.assertEqual(run["--seed"], "1000")
            self.assertEqual(run["--eval.n_episodes"], "10")
            self.assertEqual(run["--policy.n_action_steps"], "10")
            self.assertEqual(run["--env.task"], "libero_spatial,libero_object,libero_goal,libero_10")
        self.assertEqual(result.stdout.count("overall: 90.00% (360/400)"), 7)
        self.assertIn("[7/7] P1-D0-E0-T0: 50% 200/400 [ETA 10:00]", result.stderr)
        for noise in ("progress-noise", ".mp4", "No step-0"):
            self.assertNotIn(noise, result.stdout + result.stderr)
        self.assertTrue(
            any("video-path.mp4" in p.read_text() for p in Path(self.env["EVAL_LOG_ROOT"]).glob("*.log"))
        )

    def test_pdet_eval_does_not_block_earlier_runs_or_read_unpublished_weights(self):
        name, _ = self.make_pdet_eval_model("1111")
        self.make_pdet_eval_model("1110", published=False)
        capture = self.install_fake_evaluator()
        result = self.run_script(
            PDET_EVAL_SWEEP,
            DRY_RUN="false",
            PDET_CASES="1111 1110",
            WAIT_TIMEOUT_SECONDS="1",
            WAIT_INTERVAL_SECONDS="1",
        )
        self.assertNotEqual(result.returncode, 0)
        launches = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual([Path(run["--policy.path"]).parents[2].name for run in launches], [name])
        self.assertEqual(result.stdout.count("overall:"), 1)
        self.assertIn("Timed out waiting for pi05-pdet-base-vlm2-P1-D1-E1-T0", result.stderr)

    def test_pdet_eval_waits_for_publication_then_continues(self):
        first, _ = self.make_pdet_eval_model("1111")
        second, checkpoint = self.make_pdet_eval_model("1110", published=False)
        capture = self.install_fake_evaluator()
        stop = threading.Event()

        def publish_after_first_eval():
            deadline = time.monotonic() + 20
            while not stop.is_set() and time.monotonic() < deadline:
                if capture.exists() and capture.stat().st_size > 0:
                    if not stop.wait(0.5):
                        (checkpoint.parent / "last").symlink_to("003000")
                    return
                stop.wait(0.05)

        publisher = threading.Thread(target=publish_after_first_eval)
        publisher.start()
        try:
            result = self.run_script(
                PDET_EVAL_SWEEP,
                DRY_RUN="false",
                PDET_CASES="1111 1110",
                WAIT_TIMEOUT_SECONDS="10",
                WAIT_INTERVAL_SECONDS="1",
            )
        finally:
            stop.set()
            publisher.join()
        self.assertEqual(result.returncode, 0, result.stderr)
        launches = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual([Path(run["--policy.path"]).parents[2].name for run in launches], [first, second])
        self.assertEqual(result.stdout.count("overall:"), 2)

    def test_pdet_eval_rejects_mislabeled_checkpoint_and_failed_or_incomplete_results(self):
        first, checkpoint = self.make_pdet_eval_model("1111")
        self.make_pdet_eval_model("0000")
        capture = self.install_fake_evaluator()
        config_path = checkpoint / "pretrained_model/train_config.json"
        config = json.loads(config_path.read_text())
        config["pdet"]["time_seed"] = 0
        config_path.write_text(json.dumps(config))
        result = self.run_script(PDET_EVAL_SWEEP, DRY_RUN="false", PDET_CASES="1111 0000")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(capture.exists())
        self.assertIn("Evaluation/preflight failed", result.stderr)
        config["pdet"]["time_seed"] = 1
        config_path.write_text(json.dumps(config))
        good = self.run_script(PDET_EVAL_SWEEP, DRY_RUN="false", PDET_CASES="1111")
        self.assertEqual(good.returncode, 0, good.stderr)
        for overrides in ({"FAIL_RUN": first}, {"BAD_RESULT": "missing"}, {"BAD_RESULT": "episodes"}):
            with self.subTest(overrides=overrides):
                result = self.run_script(
                    PDET_EVAL_SWEEP, DRY_RUN="false", PDET_CASES="1111 0000", **overrides
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("overall:", result.stdout)
        launches = [json.loads(line) for line in capture.read_text().splitlines()]
        self.assertEqual([Path(run["--policy.path"]).parents[2].name for run in launches], [first] * 4)

    def test_pdet_eval_dry_run_does_not_write_or_wait(self):
        self.make_pdet_eval_model("1111")
        result = self.run_script(PDET_EVAL_SWEEP, PDET_CASES="1111")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--seed=1000", result.stdout)
        self.assertFalse((self.work / "eval").exists())
        result = self.run_script(PDET_EVAL_SWEEP, PDET_CASES="1110")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Completed checkpoint not ready", result.stderr)
        result = self.run_script(PDET_EVAL_SWEEP, PDET_CASES="1111 1111")
        self.assertEqual(result.returncode, 2)

    def test_default_pair_assigns_distinct_gpus_sources_ports_and_prompt_only_recipe(self):
        commands = self.commands(self.run_script(PAIR, "0"))
        self.assertEqual(len(commands), 2)
        for index, (kind, command) in enumerate(zip(("base", "finetuned"), commands, strict=True)):
            for arg in (
                f"CUDA_VISIBLE_DEVICES={index}",
                "--num_processes=1",
                "--batch_size=32",
                f"--main_process_port={29531 + index}",
                f"--policy.path={self.source(kind)}",
                "--policy.vlm_prompt_only=true",
                "--policy.num_vlm_prompt_tokens=1",
                "--policy.num_prompt_tokens=0",
                "--policy.cabo_enabled=false",
                "--policy.train_action_projections=false",
                "--policy.next_action_pretrain_steps=0",
                "--policy.next_action_bridge_steps=0",
                "--policy.ntk_save_stage_snapshots=true",
                "--steps=3000",
                "--policy.dtype=bfloat16",
                "--policy.optimizer_lr=2.5e-5",
            ):
                self.assertIn(arg, command)
            self.assertIn(
                f"--preserve_pretrained_normalization={'true' if kind == 'finetuned' else 'false'}", command
            )
            self.assertNotIn("--multi_gpu", command)
        self.assertFalse((self.work / "training").exists())

    def test_ddp_pair_keeps_global_batch_32_and_supports_custom_tokens(self):
        commands = self.commands(
            self.run_script(PAIR, "2", TRAIN_LAYOUT="ddp", GPU_IDS="2,3", VLM_PROMPT_TOKENS="16")
        )
        for command in commands:
            for arg in (
                "CUDA_VISIBLE_DEVICES=2,3",
                "--num_processes=2",
                "--multi_gpu",
                "--batch_size=16",
                "--policy.num_vlm_prompt_tokens=16",
                "--seed=2",
            ):
                self.assertIn(arg, command)

    def test_two_source_token_sweep_uses_ddp_final_only_and_no_compile(self):
        result = self.run_script(
            SWEEP, "both", "0", FLOW_STEPS="6000", SAVE_ONLY_FINAL="false", SAVE_FREQ="1"
        )
        commands = self.commands(result)
        self.assertEqual(len(commands), 8)
        for (source, count), command in zip(
            [(source, count) for source in ("base", "finetuned") for count in (8, 16, 32, 64)],
            commands,
            strict=True,
        ):
            args = dict(arg[2:].split("=", 1) for arg in command if arg.startswith("--") and "=" in arg)
            self.assertIn("CUDA_VISIBLE_DEVICES=0,1", command)
            self.assertIn("--multi_gpu", command)
            for key, value in {
                "num_processes": "2",
                "batch_size": "32",
                "steps": "3000",
                "save_freq": "3000",
                "save_steps": "[]",
                "policy.path": str(self.source(source)),
                "policy.num_vlm_prompt_tokens": str(count),
                "policy.num_prompt_tokens": "0",
                "policy.vlm_prompt_only": "true",
                "policy.ntk_save_stage_snapshots": "false",
                "policy.compile_model": "false",
                "policy.optimizer_lr": "2.5e-5",
                "wandb.enable": "true",
                "wandb.mode": "online",
                "wandb.project": "prompt-learning",
                "wandb.disable_artifact": "true",
                "output_dir": str(self.work / "training" / f"pi05-{source}-vlm{count}-seed0"),
            }.items():
                self.assertEqual(args[key], value, key)
        self.assertFalse((self.work / "training").exists())
        self.assertFalse((self.work / "logs").exists())

    def test_rank_sweep_six_base_seed1_runs_and_fixed_probe_settings(self):
        commands = self.commands(self.run_script(RANK_SWEEP))
        self.assertEqual(len(commands), 6)
        for count, command in zip((1, 2, 4, 8, 16, 32), commands, strict=True):
            args = dict(arg[2:].split("=", 1) for arg in command if arg.startswith("--") and "=" in arg)
            for key, value in {
                "seed": "1",
                "policy.path": str(self.source("base")),
                "policy.num_vlm_prompt_tokens": str(count),
                "policy.num_prompt_tokens": "0",
                "policy.compile_model": "false",
                "steps": "3000",
                "save_steps": "[]",
                "policy.ntk_save_stage_snapshots": "false",
                "prompt_rank.freq": "250",
                "prompt_rank.samples": "16",
                "prompt_rank.output_probes": "4",
                "prompt_rank.seed": "2026",
                "prompt_rank.timesteps": "[0.2,0.5,0.8]",
                "job_name": f"pi05-base-vlm{count}-seed1",
            }.items():
                self.assertEqual(args[key], value, key)
        self.assertFalse((self.work / "training").exists())
        result = self.run_script(RANK_SWEEP, "1", "--policy.compile_model=true")
        self.assertNotEqual(result.returncode, 0)

    def test_rank_sweep_imports_seed1_prompt_but_retains_seed0_training_and_probe_seed(self):
        from lerobot.utils.prompt_initialization import prompt_hash, write_prompt

        prompt = np.full((2, 2048), 0.03125, dtype=np.float32)
        path = self.work / "seed1 init.safetensors"
        write_prompt(
            path,
            prompt,
            {
                "schema_version": 1,
                "source_step": 0,
                "source_training_seed": 1,
                "num_vlm_prompt_tokens": 2,
                "prompt_sha256": prompt_hash(prompt),
            },
        )
        result = self.run_script(
            RANK_SWEEP,
            "0",
            VLM_PROMPT_TOKEN_COUNTS="2",
            PROMPT_INIT_PATH=str(path),
            RUN_PREFIX="pi05-init1-train0",
        )
        (command,) = self.commands(result)
        args = dict(arg[2:].split("=", 1) for arg in command if arg.startswith("--") and "=" in arg)
        self.assertEqual(args["seed"], "0")
        self.assertEqual(args["prompt_rank.seed"], "2026")
        self.assertEqual(args["vlm_prompt_init_path"], str(path))
        self.assertEqual(args["policy.path"], str(self.source("base")))
        self.assertEqual(args["job_name"], "pi05-init1-train0-base-vlm2-seed0")
        self.assertFalse((self.work / "training").exists())
        bad = self.run_script(RANK_SWEEP, "0", VLM_PROMPT_TOKEN_COUNTS="2 4", PROMPT_INIT_PATH=str(path))
        self.assertNotEqual(bad.returncode, 0)
        self.assertNotIn("Starting source=", bad.stdout)

    def test_rank_sweep_initialization_export_controls_and_preflight(self):
        output = self.work / "shared inits/seed0-vlm8.safetensors"
        reference = self.work / "old swap.json"
        reference.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_step": 0,
                    "source_training_seed": 1,
                    "training_seed": 0,
                    "num_vlm_prompt_tokens": 8,
                    "replaced_prompt_sha256": "a" * 64,
                }
            )
        )
        settings = {
            "VLM_PROMPT_TOKEN_COUNTS": "8",
            "RUN_PREFIX": "pi05-init-rebuild",
            "PROMPT_EXPORT_PATH": str(output),
            "PROMPT_EXPORT_REFERENCE": str(reference),
        }
        result = self.run_script(RANK_SWEEP, "0", **settings)
        (command,) = self.commands(result)
        args = dict(arg[2:].split("=", 1) for arg in command if arg.startswith("--") and "=" in arg)
        self.assertEqual(args["seed"], "0")
        self.assertEqual(args["policy.num_vlm_prompt_tokens"], "8")
        self.assertEqual(args["vlm_prompt_export_path"], str(output))
        self.assertEqual(args["vlm_prompt_export_reference"], str(reference))
        self.assertEqual(args["policy.compile_model"], "false")
        self.assertIn("0 updates", result.stdout)
        self.assertNotIn("Starting source=", result.stdout)
        self.assertFalse(output.exists())
        self.assertFalse((self.work / "training").exists())
        for overrides in (
            {"VLM_PROMPT_TOKEN_COUNTS": "2 8"},
            {"PROMPT_INIT_PATH": "/seed1.safetensors"},
            {"PROMPT_EXPORT_REFERENCE": str(self.work / "absent.json")},
        ):
            bad = self.run_script(RANK_SWEEP, "0", **{**settings, **overrides})
            self.assertNotEqual(bad.returncode, 0)
            self.assertNotIn("Starting initialization export", bad.stdout)
        output.parent.mkdir()
        output.write_text("existing initialization")
        bad = self.run_script(RANK_SWEEP, "0", **settings)
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual(output.read_text(), "existing initialization")

    def test_token_sweep_selects_remaining_runs_and_rejects_ambiguous_settings(self):
        (command,) = self.commands(
            self.run_script(
                SWEEP,
                "finetuned",
                "2",
                "--wandb.mode=offline",
                VLM_PROMPT_TOKEN_COUNTS="4",
                GPU_IDS="2,3",
                BATCH_SIZE="16",
                RUN_PREFIX="retry",
            )
        )
        self.assertIn("--job_name=retry-finetuned-vlm4-seed2", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=2,3", command)
        self.assertIn("--batch_size=16", command)
        self.assertEqual(
            [arg for arg in command if arg.startswith("--wandb.mode=")][-1], "--wandb.mode=offline"
        )
        for env in (
            {"VLM_PROMPT_TOKEN_COUNTS": ""},
            {"VLM_PROMPT_TOKEN_COUNTS": "2 2"},
            {"VLM_PROMPT_TOKEN_COUNTS": "2 0"},
            {"VLM_PROMPT_TOKEN_COUNTS": "2 x"},
            {"RUN_NAME": "old-run"},
            {"VLM_PROMPT_TOKENS": "1"},
            {"PRETRAINED_PATH": "old-source"},
        ):
            with self.subTest(env=env):
                self.assertNotEqual(self.run_script(SWEEP, "both", "0", **env).returncode, 0)

    def test_token_sweep_checks_last_destination_before_starting_any_training(self):
        output = self.work / "training/pi05-finetuned-vlm64-seed0"
        output.mkdir(parents=True)
        result = self.run_script(SWEEP, "both", "0", DRY_RUN="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)
        self.assertNotIn("Starting source=", result.stdout)
        self.assertEqual(list((self.work / "training").iterdir()), [output])
        self.assertFalse((self.work / "logs").exists())

    def test_token_sweep_stops_on_training_failure_before_later_runs(self):
        bin_dir = self.work / "bin"
        bin_dir.mkdir()
        accelerate = bin_dir / "accelerate"
        accelerate.write_text(
            f"#!{sys.executable}\nimport os, sys\nfrom pathlib import Path\n"
            "job = next(x.split('=', 1)[1] for x in sys.argv if x.startswith('--job_name='))\n"
            "with Path(os.environ['CAPTURE_LOG']).open('a') as stream: stream.write(job + '\\n')\n"
            "print(job, flush=True)\nsys.exit(7 if '-base-vlm16-' in job else 0)\n"
        )
        trainer = bin_dir / "lerobot-train"
        trainer.write_text("#!/bin/sh\nexit 0\n")
        for path in (accelerate, trainer):
            path.chmod(0o755)
        capture = self.work / "launches.txt"
        cache = dict.fromkeys(
            ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"), str(self.work / "cache")
        )
        result = self.run_script(
            SWEEP,
            "both",
            "0",
            DRY_RUN="false",
            PATH=f"{bin_dir}{os.pathsep}{self.env['PATH']}",
            CAPTURE_LOG=str(capture),
            **cache,
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(capture.read_text().splitlines(), ["pi05-base-vlm8-seed0", "pi05-base-vlm16-seed0"])
        self.assertIn("Sweep stopped: source=base VLM=16 exit=7", result.stderr)
        self.assertFalse((self.work / "logs/pi05-base-vlm32-seed0.log").exists())
        self.assertFalse((self.work / "logs/pi05-finetuned-vlm8-seed0.log").exists())

    def test_final_only_saves_3000_without_initial_or_periodic_checkpoints(self):
        from lerobot.configs.default import DatasetConfig
        from lerobot.configs.train import TrainPipelineConfig

        result = self.run_script(
            TRAIN, "finetuned", "0", SAVE_ONLY_FINAL="true", GPU_IDS="0,1", BATCH_SIZE="32", SAVE_FREQ="1"
        )
        (command,) = self.commands(result)
        for arg in (
            "--num_processes=2",
            "--multi_gpu",
            "--batch_size=32",
            "--policy.ntk_save_stage_snapshots=false",
            "--save_steps=[]",
            "--save_freq=3000",
        ):
            self.assertIn(arg, command)
        self.assertIn(f"--policy.path={self.source('finetuned')}", command)
        self.assertIn("final only=3000", result.stdout)
        args = dict(arg[2:].split("=", 1) for arg in command if arg.startswith("--") and "=" in arg)
        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(repo_id="test/data"),
            steps=int(args["steps"]),
            save_steps=json.loads(args["save_steps"]),
            save_freq=int(args["save_freq"]),
        )
        cfg.validate_checkpoint_schedule()
        self.assertEqual([step for step in range(3001) if cfg.should_save_checkpoint(step)], [3000])

    def test_invalid_or_conflicting_settings_fail_before_launch(self):
        for env in (
            {"GPU_IDS": "0,0"},
            {"GPU_IDS": "0"},
            {"TRAIN_LAYOUT": "wrong"},
            {"NUM_PROCESSES": "2"},
            {"PRETRAINED_PATH": "wrong"},
            {"DRY_RUN": "yes"},
            {"SAVE_ONLY_FINAL": "yes"},
        ):
            self.assertNotEqual(self.run_script(PAIR, "0", **env).returncode, 0)
        for arg in (
            "--policy.num_prompt_tokens=16",
            "--policy.vlm_prompt_only=false",
            "--policy.cabo_enabled=true",
        ):
            self.assertNotEqual(self.run_script(TRAIN, "base", "0", arg).returncode, 0)
        self.assertNotEqual(
            self.run_script(TRAIN, "base", "0", GPU_IDS="0,1", NUM_PROCESSES="1").returncode, 0
        )

    def test_missing_second_source_or_existing_output_prevents_both_launches(self):
        (self.work / "training/pi05-finetuned-vlm1-seed0").mkdir(parents=True)
        result = self.run_script(PAIR, "0", DRY_RUN="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)
        self.assertNotIn("Both sources passed", result.stdout)
        self.assertFalse((self.work / "logs").exists())

    def test_matched_base_uses_original_weights_and_initial_quantile_statistics(self):
        output = self.work / "baseline"
        source_before = {p.name: p.read_bytes() for p in self.source("base").iterdir()}
        prepare_baseline(self.source("base"), self.model("base", 0), output, 1, "base")
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["num_vlm_prompt_tokens"], 0)
        self.assertFalse(config["vlm_prompt_only"])
        self.assertEqual(config["normalization_mapping"]["ACTION"], "QUANTILES")
        self.assertEqual((output / "model.safetensors").resolve(), self.source("base") / "model.safetensors")
        self.assertEqual(
            (output / "normalizer_processor.safetensors").read_bytes(),
            (self.model("base", 0) / "normalizer_processor.safetensors").read_bytes(),
        )
        self.assertEqual(source_before, {p.name: p.read_bytes() for p in self.source("base").iterdir()})
        # Reuse succeeds; changed processor contents cannot silently reuse the old baseline.
        prepare_baseline(self.source("base"), self.model("base", 0), output, 1, "base")
        (output / "policy_postprocessor.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Baseline processor changed"):
            prepare_baseline(self.source("base"), self.model("base", 0), output, 1, "base")

    def test_processor_drift_and_wrong_source_are_rejected(self):
        state = self.model("finetuned", 3000) / "normalizer_processor.safetensors"
        tensors = load_file(state)
        tensors["action.mean"][0] += 1
        save_file(tensors, state)
        with self.assertRaisesRegex(ValueError, "statistics differ"):
            validate_checkpoint(self.model("finetuned", 3000), 1, reference=self.model("finetuned", 0))
        with self.assertRaisesRegex(ValueError, "different source path"):
            validate_checkpoint(self.model("base", 0), 1, source=self.source("finetuned"))
        result = self.run_script(EVAL, "both", "all", "all", "0", DRY_RUN="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Evaluating source=", result.stdout)
        self.assertFalse((self.work / "eval").exists())

    def test_finetuned_processor_load_to_save_roundtrip_preserves_comparison(self):
        from safetensors.torch import load_file as load_torch, save_file as save_torch

        from lerobot.processor.normalize_processor import NormalizerProcessorStep

        source = self.source("finetuned")
        filename = "policy_preprocessor.json"
        pipeline = json.loads((source / filename).read_text())
        step = pipeline["steps"][0]
        tensors = load_file(source / step["state_file"])
        for feature in ("action", "observation.state"):
            tensors[f"{feature}.count"] = np.array([100], dtype=np.float32)
        # Dataset metadata is saved alongside model feature statistics. The
        # same scalar conversion affects every statistic, not just counts.
        for feature in ("episode_index", "frame_index", "index", "task_index", "timestamp", "extra_metadata"):
            for stat in ("min", "max", "mean", "std", "q01", "q99", "count"):
                tensors[f"{feature}.{stat}"] = np.array([100], dtype=np.float32)
        # v044 contains 1-D counts even though its image mean/std are already 3-D.
        for camera in ("image", "image2"):
            feature = f"observation.images.{camera}"
            step["config"]["features"][feature] = {"type": "VISUAL", "shape": [3, 256, 256]}
            tensors[f"{feature}.count"] = np.array([100], dtype=np.float32)
            tensors[f"{feature}.mean"] = np.ones((3, 1, 1), dtype=np.float32)
        (source / filename).write_text(json.dumps(pipeline))
        save_file(tensors, source / step["state_file"])
        # Use the real loader/saver, rather than assuming what serialization does.
        processor = NormalizerProcessorStep(**step["config"])
        processor.load_state_dict(load_torch(source / step["state_file"]))
        self.assertEqual(processor.state_dict()["observation.images.image.count"].shape, (1, 1, 1))
        # Training moves processors to a device/dtype, rebuilding tensors from
        # stats whose singleton values have already been converted to scalars.
        processor.to(device="cpu")
        saved = processor.state_dict()
        for key in tensors:
            if tensors[key].size == 1:
                self.assertEqual(saved[key].shape, ())
        for training_step in (0, 3000):
            model = self.model("finetuned", training_step)
            (model / filename).write_text(json.dumps(pipeline))
            save_torch(saved, model / step["state_file"])
        validate_checkpoint(self.model("finetuned", 0), 1, source=source, kind="finetuned")
        result = self.run_script(EVAL, "finetuned", "all", "all", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Evaluating source="), 3)
        # Canonicalizing shape must not hide changed values, even for image counts.
        saved["observation.images.image.count"] += 1
        save_torch(saved, self.model("finetuned", 3000) / step["state_file"])
        with self.assertRaisesRegex(ValueError, r"observation.images.image.count.*max_abs_diff_after_load=1"):
            validate_checkpoint(self.model("finetuned", 3000), 1, reference=self.model("finetuned", 0))

    def test_float32_loader_conversion_is_allowed_but_not_numeric_tolerance(self):
        from safetensors.torch import load_file as load_torch, save_file as save_torch

        from lerobot.processor.normalize_processor import NormalizerProcessorStep

        source = self.source("finetuned")
        step = json.loads((source / "policy_preprocessor.json").read_text())["steps"][0]
        tensors = load_file(source / step["state_file"])
        tensors["action.mean"] = np.linspace(0.123456789012, 0.765432198765, 7, dtype=np.float64)
        save_file(tensors, source / step["state_file"])
        processor = NormalizerProcessorStep(**step["config"])
        processor.load_state_dict(load_torch(source / step["state_file"]))
        save_torch(processor.state_dict(), self.model("finetuned", 0) / step["state_file"])
        validate_checkpoint(self.model("finetuned", 0), 1, source=source, kind="finetuned")
        tensors = load_file(self.model("finetuned", 0) / step["state_file"])
        # One float32 ULP would pass a broad allclose(), but is genuine drift.
        tensors["action.mean"][0] = np.nextafter(tensors["action.mean"][0], np.float32(np.inf))
        save_file(tensors, self.model("finetuned", 0) / step["state_file"])
        with self.assertRaisesRegex(ValueError, r"reference=.*candidate=.*action.mean.*max_abs_diff"):
            validate_checkpoint(self.model("finetuned", 0), 1, source=source, kind="finetuned")

    def test_final_only_eval_accepts_singleton_count_shapes_in_both_processors(self):
        source = self.source("finetuned")
        model = self.model("finetuned", 3000)
        shutil.rmtree(self.model("finetuned", 0).parent)
        for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
            step = json.loads((source / filename).read_text())["steps"][0]
            tensors = load_file(source / step["state_file"])
            for feature in step["config"]["features"]:
                tensors[f"{feature}.count"] = np.array([100], dtype=np.float32)
            save_file(tensors, source / step["state_file"])
            for shape in ((), (1,), (1, 1, 1)):
                with self.subTest(processor=filename, count_shape=shape):
                    saved = {
                        key: value.reshape(shape) if key.endswith(".count") else value
                        for key, value in tensors.items()
                    }
                    save_file(saved, model / step["state_file"])
                    result = self.run_script(EVAL, "finetuned", "trained", "all", "0")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("No step-0 checkpoint", result.stdout)
                    self.assertIn("Evaluating source=finetuned condition=trained", result.stdout)

    def test_count_values_cardinality_and_missing_keys_remain_strict(self):
        state = self.model("finetuned", 3000) / "normalizer_processor.safetensors"
        tensors = load_file(state)
        for step in (0, 3000):
            save_file(
                {**tensors, "action.count": np.array([100], dtype=np.float32)},
                self.model("finetuned", step) / state.name,
            )
        for count in (np.array(101, dtype=np.float32), np.array([100, 100], dtype=np.float32)):
            with self.subTest(count=count):
                save_file({**tensors, "action.count": count}, state)
                with self.assertRaisesRegex(ValueError, r"action.count: saved shapes"):
                    validate_checkpoint(
                        self.model("finetuned", 3000), 1, reference=self.model("finetuned", 0)
                    )
        save_file(tensors, state)
        with self.assertRaisesRegex(ValueError, r"missing=\['action.count'\]"):
            validate_checkpoint(self.model("finetuned", 3000), 1, reference=self.model("finetuned", 0))

    def test_count_shape_compatibility_does_not_relax_normalization_shapes(self):
        state = self.model("finetuned", 3000) / "normalizer_processor.safetensors"
        tensors = load_file(state)
        for feature in ("action", "observation.state"):
            for statistic in ("mean", "std"):
                key = f"{feature}.{statistic}"
                with self.subTest(statistic=key):
                    save_file({**tensors, key: tensors[key].reshape(1, -1)}, state)
                    with self.assertRaisesRegex(ValueError, rf"invalid normalization tensor: {key}"):
                        validate_checkpoint(
                            self.model("finetuned", 3000), 1, reference=self.model("finetuned", 0)
                        )

    def test_final_only_metadata_roundtrip_in_both_processors(self):
        from safetensors.torch import load_file as load_torch, save_file as save_torch

        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep

        source, model = self.source("finetuned"), self.model("finetuned", 3000)
        shutil.rmtree(self.model("finetuned", 0).parent)
        for filename, processor_type in (
            ("policy_preprocessor.json", NormalizerProcessorStep),
            ("policy_postprocessor.json", UnnormalizerProcessorStep),
        ):
            step = json.loads((source / filename).read_text())["steps"][0]
            tensors = load_file(source / step["state_file"])
            tensors["episode_index.max"] = np.array([100], dtype=np.float32)
            tensors["timestamp.mean"] = np.array([0.5], dtype=np.float32)
            save_file(tensors, source / step["state_file"])
            processor = processor_type(**step["config"])
            processor.load_state_dict(load_torch(source / step["state_file"]))
            processor.to(device="cpu")
            saved = processor.state_dict()
            self.assertEqual(saved["episode_index.max"].shape, ())
            save_torch(saved, model / step["state_file"])
        result = self.run_script(EVAL, "finetuned", "trained", "all", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No step-0 checkpoint", result.stdout)
        self.assertIn("Evaluating source=finetuned condition=trained", result.stdout)

    def test_metadata_values_cardinality_missing_keys_and_all_differences_remain_checked(self):
        source, model = self.source("finetuned"), self.model("finetuned", 3000)
        filename = "normalizer_processor.safetensors"
        reference = load_file(source / filename)
        reference["episode_index.max"] = np.array([100], dtype=np.float32)
        reference["timestamp.mean"] = np.array([0.5], dtype=np.float32)
        save_file(reference, source / filename)
        for changed in (np.array(101, dtype=np.float32), np.array([100, 100], dtype=np.float32)):
            with self.subTest(episode_max=changed):
                save_file({**reference, "episode_index.max": changed}, model / filename)
                with self.assertRaisesRegex(ValueError, r"episode_index.max: saved shapes"):
                    validate_checkpoint(model, 1, source=source, kind="finetuned")
        save_file(
            {key: value for key, value in reference.items() if key != "episode_index.max"}, model / filename
        )
        with self.assertRaisesRegex(ValueError, r"missing=\['episode_index.max'\]"):
            validate_checkpoint(model, 1, source=source, kind="finetuned")
        # Report all differences in this state file so debugging does not stop
        # at another metadata key on each retry.
        candidate = {**reference, "episode_index.max": np.array(101, dtype=np.float32)}
        candidate["timestamp.mean"] = np.array(0.75, dtype=np.float32)
        save_file(candidate, model / filename)
        with self.assertRaisesRegex(ValueError, "statistics differ") as error:
            validate_checkpoint(model, 1, source=source, kind="finetuned")
        self.assertIn("episode_index.max", str(error.exception))
        self.assertIn("timestamp.mean", str(error.exception))

    def test_dry_eval_plans_six_conditions_without_writing_baselines(self):
        result = self.run_script(EVAL, "both", "all", "all", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Evaluating source="), 6)
        self.assertIn("/000000/pretrained_model", result.stdout)
        self.assertIn("/003000/pretrained_model", result.stdout)
        self.assertIn("matched_baseline/pretrained_model", result.stdout)
        self.assertFalse(list((self.work / "checkpoints").glob("*/matched_baseline")))
        self.assertFalse((self.work / "eval").exists())

    def test_trained_evaluation_without_initial_and_clear_error_for_initial_controls(self):
        for kind in ("base", "finetuned"):
            shutil.rmtree(self.model(kind, 0).parent)
            result = self.run_script(EVAL, kind, "trained", "all", "0")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No step-0 checkpoint", result.stdout)
            self.assertIn("/003000/pretrained_model", result.stdout)
            command = next(
                shlex.split(line) for line in result.stdout.splitlines() if line.startswith("python -u")
            )
            self.assertIn("--env.task=libero_spatial,libero_object,libero_goal,libero_10", command)
            for condition in ("baseline", "initial", "all"):
                result = self.run_script(EVAL, kind, condition, "all", "0")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires the saved step-0", result.stderr)
                self.assertNotIn("Evaluating source=", result.stdout)
        self.assertFalse((self.work / "eval").exists())

    def test_final_only_eval_still_rejects_statistics_drift_and_wrong_source(self):
        shutil.rmtree(self.model("finetuned", 0).parent)
        state = self.model("finetuned", 3000) / "normalizer_processor.safetensors"
        original = state.read_bytes()
        tensors = load_file(state)
        tensors["action.mean"][0] += 1
        save_file(tensors, state)
        result = self.run_script(EVAL, "finetuned", "trained", "all", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("statistics differ", result.stderr)
        self.assertIn("action.mean", result.stderr)
        self.assertNotIn("Evaluating source=", result.stdout)
        state.write_bytes(original)
        (state.parent / "train_config.json").write_text(
            json.dumps({"policy": {"pretrained_path": str(self.source("base"))}})
        )
        result = self.run_script(EVAL, "finetuned", "trained", "all", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different source path", result.stderr)
        self.assertNotIn("Evaluating source=", result.stdout)

    def test_real_parallel_dispatch_waits_for_both_and_propagates_failure(self):
        bin_dir = self.work / "bin"
        bin_dir.mkdir()
        accelerate = bin_dir / "accelerate"
        accelerate.write_text(
            f"#!{sys.executable}\nimport json, os, sys, time\nfrom pathlib import Path\n"
            "source = next(x.split('=',1)[1] for x in sys.argv if x.startswith('--policy.path='))\n"
            "kind = Path(source).name\nwork = Path(os.environ['CAPTURE_ROOT'])\n"
            "(work / (kind + '.ready')).touch()\nlimit = time.monotonic() + 10\n"
            "while not all((work / (x + '.ready')).exists() for x in ['base', 'finetuned']):\n"
            "    if time.monotonic() > limit: sys.exit(19)\n    time.sleep(0.02)\n"
            "(work / (kind + '.json')).write_text(json.dumps({'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'args': sys.argv}))\n"
            "print('mock completed ' + kind, flush=True)\nsys.exit(7 if kind == 'base' else 0)\n"
        )
        trainer = bin_dir / "lerobot-train"
        trainer.write_text("#!/bin/sh\nexit 0\n")
        for path in (accelerate, trainer):
            path.chmod(0o755)
        cache = dict.fromkeys(
            ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"), str(self.work / "cache")
        )
        result = self.run_script(
            PAIR,
            "0",
            DRY_RUN="false",
            PATH=f"{bin_dir}{os.pathsep}{self.env['PATH']}",
            CAPTURE_ROOT=str(self.work),
            **cache,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        for kind, gpu in (("base", "0"), ("finetuned", "1")):
            self.assertEqual(json.loads((self.work / f"{kind}.json").read_text())["gpu"], gpu)
            self.assertIn(
                f"mock completed {kind}", (self.work / f"logs/pi05-{kind}-vlm1-seed0.log").read_text()
            )


if __name__ == "__main__":
    unittest.main()
