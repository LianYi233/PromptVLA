import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as Namespace
from unittest.mock import Mock, patch

import numpy as np

FILE = Path(__file__).resolve().parents[1] / "deploy_piper_vlaa.py"
sys.path.insert(0, str(FILE.parent))
spec = importlib.util.spec_from_file_location("deploy", FILE)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)
NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]


def config():
    return Namespace(
        type="pi05",
        training_stage="flow",
        use_peft=False,
        chunk_size=50,
        n_action_steps=50,
        num_inference_steps=10,
        num_vlm_prompt_tokens=16,
        num_prompt_tokens=16,
        compile_model=True,
        gradient_checkpointing=True,
        use_amp=False,
        rtc_config=Namespace(enabled=True),
        input_features={
            "observation.state": Namespace(shape=(7,)),
            "observation.images.wrist": Namespace(shape=(3, 4, 6)),
        },
        output_features={"action": Namespace(shape=(7,))},
        image_features={"observation.images.wrist": Namespace(shape=(3, 4, 6))},
    )


def metadata():
    return {
        "fps": 30,
        "features": {
            "observation.state": {"shape": [7], "names": NAMES[::-1]},
            "action": {"shape": [7], "names": NAMES[2:] + NAMES[:2]},
            "observation.images.wrist": {"shape": [4, 6, 3]},
        },
    }


class DeploymentTests(unittest.TestCase):
    def args(self, *extra):
        return deploy.parse_args(["--policy_path", "/tmp/checkpoint", "--dry_run", *extra])

    def test_inference_overrides_preserve_trained_architecture(self):
        cfg = config()
        hz, steps = deploy.configure_deployment(cfg, self.args("--steps_per_chunk", "8"), 30)
        self.assertEqual((hz, steps), (30, 8))
        self.assertEqual(cfg.chunk_size, 50)
        self.assertEqual((cfg.num_vlm_prompt_tokens, cfg.num_prompt_tokens), (16, 16))
        self.assertFalse(cfg.compile_model)
        self.assertFalse(cfg.gradient_checkpointing)
        self.assertIsNone(cfg.rtc_config)
        cfg.training_stage = "next_action"
        with self.assertRaisesRegex(ValueError, "flow"):
            deploy.configure_deployment(cfg, self.args(), 30)
        cfg.training_stage = "flow"
        with self.assertRaisesRegex(ValueError, "1..50"):
            deploy.configure_deployment(cfg, self.args("--steps_per_chunk", "51"), 30)
        with self.assertRaisesRegex(ValueError, "有限正数"):
            deploy.configure_deployment(cfg, self.args("--control_hz", "nan"), 30)

    def test_parser_requires_metadata_only_for_hardware(self):
        self.assertTrue(self.args().dry_run)
        self.assertTrue(deploy.parse_args(["--check_env"]).check_env)
        self.assertTrue(deploy.parse_args(["--check_robot"]).check_robot)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            deploy.parse_args(["--dry_run"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            deploy.parse_args(["--policy_path", "/tmp/checkpoint"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--quantile_sampling")

    def test_robot_diagnostic_does_not_import_model_or_require_checkpoint(self):
        with (
            patch.object(deploy, "check_robot_connection") as check,
            patch.dict(sys.modules, {"torch": None, "transformers": None}),
        ):
            deploy.main(["--check_robot", "--can_port", "can0"])
        check.assert_called_once_with("can0")

    def test_checkout_precedes_other_lerobot_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            old_package = Path(directory) / "lerobot"
            old_package.mkdir()
            (old_package / "__init__.py").write_text('raise RuntimeError("old checkout loaded")')
            code = """
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[2]).parent))
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location("deployment", sys.argv[2])
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)
source = deployment.activate_checkout_source()
import lerobot
assert Path(lerobot.__file__).resolve() == source / "lerobot/__init__.py"
"""
            subprocess.run(
                [sys.executable, "-c", code, directory, str(FILE)], check=True, capture_output=True
            )

    def test_already_imported_foreign_source_is_rejected(self):
        with (
            patch.dict(sys.modules, {"lerobot": Namespace(__file__="/old-checkout/lerobot/__init__.py")}),
            self.assertRaisesRegex(RuntimeError, "已加载其他目录"),
        ):
            deploy.activate_checkout_source()

    def test_config_source_and_prompt_fields_are_verified(self):
        source = FILE.parent / "src"
        module = Namespace(__file__=str(source / "lerobot/policies/pi05/configuration_pi05.py"))
        cfg_class = Namespace(
            __module__="fake_pi05_config",
            __dataclass_fields__={
                "num_vlm_prompt_tokens": None,
                "num_prompt_tokens": None,
                "training_stage": None,
            },
        )
        with patch.dict(sys.modules, {"fake_pi05_config": module}), contextlib.redirect_stdout(io.StringIO()):
            deploy.validate_pi05_source(cfg_class, source)
            del cfg_class.__dataclass_fields__["num_vlm_prompt_tokens"]
            with self.assertRaisesRegex(RuntimeError, "num_vlm_prompt_tokens"):
                deploy.validate_pi05_source(cfg_class, source)
            module.__file__ = "/old-checkout/configuration_pi05.py"
            with self.assertRaisesRegex(RuntimeError, "来自其他源码"):
                deploy.validate_pi05_source(cfg_class, source)

    def test_missing_driver_dependency_is_not_hidden(self):
        missing_sdk = ModuleNotFoundError("No module named piper_sdk", name="piper_sdk")
        with (
            patch("builtins.__import__", side_effect=missing_sdk),
            self.assertRaises(ModuleNotFoundError) as result,
        ):
            deploy.create_robot(None, None, None)
        self.assertIs(result.exception, missing_sdk)
        missing_driver = ModuleNotFoundError("No Piper", name="lerobot.robots.piper")
        with (
            patch("builtins.__import__", side_effect=missing_driver),
            self.assertRaisesRegex(ImportError, "缺少自定义 Piper 驱动"),
        ):
            deploy.create_robot(None, None, None)

    def test_metadata_order_and_camera_conversion(self):
        features = deploy.deployment_features(config(), metadata())
        self.assertEqual(features["action"]["names"], NAMES[2:] + NAMES[:2])
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[..., 0] = 19
        raw = dict(zip(NAMES, range(7), strict=True), wrist_real=frame)
        mapping = deploy.resolve_camera_map(features, {"observation.images.wrist": "wrist_real"})
        result = deploy.build_observation(raw, features, mapping, "bgr")
        np.testing.assert_equal(result["observation.state"], np.arange(7)[::-1])
        self.assertTrue((result["observation.images.wrist"][..., 2] == 19).all())
        self.assertTrue(result["observation.images.wrist"].flags.c_contiguous)
        self.assertTrue((frame[..., 0] == 19).all())

    def test_wrong_metadata_and_missing_camera_rejected(self):
        meta = metadata()
        meta["features"]["action"]["names"][0] = "unknown_joint"
        with self.assertRaisesRegex(ValueError, "不能猜测顺序"):
            deploy.deployment_features(config(), meta)
        meta = metadata()
        del meta["features"]["observation.images.wrist"]
        with self.assertRaisesRegex(ValueError, "相机"):
            deploy.deployment_features(config(), meta)
        features = deploy.deployment_features(config())
        with self.assertRaisesRegex(ValueError, "camera_map"):
            deploy.resolve_camera_map(features, {"observation.images.wrong": "camera"})
        cfg = config()
        cfg.output_features["action"].shape = (32,)
        with self.assertRaisesRegex(ValueError, "7 维"):
            deploy.deployment_features(cfg)

    def test_checkpoint_requires_processor_statistics(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").touch()
            (root / "policy_preprocessor.json").write_text(
                json.dumps({"steps": [{"state_file": "stats.safetensors"}]})
            )
            (root / "policy_postprocessor.json").write_text(json.dumps({"steps": []}))
            with self.assertRaisesRegex(ValueError, "stats.safetensors"):
                deploy.check_checkpoint_files(root)
            (root / "stats.safetensors").touch()
            deploy.check_checkpoint_files(root)
            (root / "config.json").write_text('{"action_generation_mode": "regression"}')
            with self.assertRaisesRegex(ValueError, "quantile/regression"):
                deploy.check_checkpoint_files(root)

    def test_missing_prompt_or_mismatched_shape_rejected(self):
        cfg = config()
        state = Mock()
        state.keys.return_value = []
        fake = Namespace(safe_open=lambda *a, **k: contextlib.nullcontext(state))
        with patch.dict(sys.modules, {"safetensors": fake}):
            with self.assertRaisesRegex(ValueError, "缺少"):
                deploy.check_prompt_weights(Path("/tmp/model"), cfg)
            state.keys.return_value = ["model.vlm_prompt_tokens.weight", "model.prompt_tokens.weight"]
            state.get_slice.return_value.get_shape.return_value = [8, 2048]
            with self.assertRaisesRegex(ValueError, "形状与 config 不符"):
                deploy.check_prompt_weights(Path("/tmp/model"), cfg)

    def test_nonfinite_actions_fail_and_limits_remain(self):
        action = dict.fromkeys(NAMES, 100.0)
        clamped = deploy.clamp_action(action)
        self.assertEqual(clamped["gripper.pos"], 1.0)
        self.assertAlmostEqual(clamped["joint_1.pos"], np.deg2rad(150))
        self.assertEqual(clamped["joint_3.pos"], 0.0)
        action["joint_1.pos"] = float("nan")
        with self.assertRaisesRegex(ValueError, "非有限"):
            deploy.clamp_action(action)

    def make_worker(self, cfg, **kwargs):
        return deploy.ChunkInferenceWorker(
            policy=kwargs.get("policy", Mock()),
            preprocessor=kwargs.get("pre", Mock()),
            postprocessor=kwargs.get("post", Mock()),
            policy_cfg=cfg,
            ds_features=deploy.deployment_features(cfg),
            device=Namespace(type="cpu"),
            task="Put the apple on the yellow plate",
            robot_type="piper",
            steps_per_chunk=3,
        )

    def test_one_flow_call_per_chunk_and_saved_postprocessor_used(self):
        cfg = config()
        values = np.zeros((1, 50, 7), dtype=np.float32)
        policy = Namespace(predict_action_chunk=Mock(return_value=values))
        pre = Mock(side_effect=lambda batch: dict(batch, processed=True))
        post = Mock(side_effect=lambda array: array + 0.125)
        prepare = Mock(side_effect=lambda obs, device, task, robot_type: dict(obs, task=task))
        policies = types.ModuleType("lerobot.policies")
        policies.prepare_observation_for_inference = prepare
        policies.make_robot_action = lambda val, ft: dict(zip(ft["action"]["names"], val[0], strict=True))
        torch = Namespace(inference_mode=contextlib.nullcontext, isfinite=np.isfinite)
        worker = self.make_worker(cfg, policy=policy, pre=pre, post=post)
        with patch.dict(sys.modules, {"torch": torch, "lerobot.policies": policies}):
            chunk, _ = worker._infer_chunk({"observation.state": np.zeros(7, dtype=np.float32)})
            self.assertEqual(len(chunk), 3)
            self.assertEqual(policy.predict_action_chunk.call_count, 1)
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(pre.call_count, 1)
            self.assertEqual(post.call_count, 3)
            self.assertAlmostEqual(chunk[0]["joint_1.pos"], 0.125)
            self.assertTrue(policy.predict_action_chunk.call_args.args[0]["processed"])
            values[0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "非有限"):
                worker._infer_chunk({})

    def test_background_inference_error_reaches_main_thread(self):
        worker = self.make_worker(config())
        worker._infer_chunk = Mock(side_effect=ValueError("bad checkpoint"))
        worker.start()
        try:
            worker.trigger({"frame": 1})
            with self.assertRaisesRegex(RuntimeError, "后台策略推理失败") as result:
                worker.next_chunk(timeout=1)
            self.assertIsInstance(result.exception.__cause__, ValueError)
        finally:
            worker.stop()
        self.assertFalse(worker._thread.is_alive())

    def test_dry_run_never_constructs_or_imports_piper(self):
        worker = self.make_worker(config())
        worker._policy.config = config()
        worker._policy.predict_action_chunk = Mock(return_value=None)

        def infer(obs):
            self.assertIn("observation.images.wrist", obs)
            worker._policy.predict_action_chunk(obs)
            return [dict.fromkeys(NAMES, 0.0)] * 3, 5.0

        worker._infer_chunk = infer
        before = set(sys.modules)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            deploy.run_dry_run(
                worker, worker._ds_features, worker._policy, worker._preprocessor, worker._postprocessor
            )
        self.assertIn("DRY_RUN_OK", output.getvalue())
        self.assertFalse(
            any(
                key.startswith(("piper_sdk", "pyrealsense2", "lerobot.robots.piper"))
                for key in set(sys.modules) - before
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
