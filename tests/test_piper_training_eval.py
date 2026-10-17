"""CPU numerical/inference-contract tests; run directly with torch and numpy installed."""

import importlib.util
import json
import random
import socket
import sys
import tempfile
import unittest
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "piper_training_eval", ROOT / "src/lerobot/utils/piper_training_eval.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONFIG_SPEC = importlib.util.spec_from_file_location(
    "piper_eval_config", ROOT / "src/lerobot/configs/piper.py"
)
CONFIG = importlib.util.module_from_spec(CONFIG_SPEC)
sys.modules[CONFIG_SPEC.name] = CONFIG
CONFIG_SPEC.loader.exec_module(CONFIG)


def distributed_worker(rank, rendezvous, output):
    import torch.distributed as dist

    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
        )
    except RuntimeError as error:
        if "gloo/transport/tcp/device.cc" in str(error) and "Operation not permitted" in str(error):
            Path(output, f"rank-{rank}.blocked").write_text(str(error))
            return
        raise

    class Accelerator:
        is_main_process = rank == 0
        device = torch.device("cpu")

        @staticmethod
        def reduce(tensor, reduction):
            assert reduction == "sum"
            dist.all_reduce(tensor)
            return tensor

    def successful_callback():
        assert rank == 0
        return 123

    def failing_callback():
        assert rank == 0
        raise ValueError("bad feedback sample")

    try:
        value = MODULE.on_main_process(Accelerator(), successful_callback)
        assert value == (123 if rank == 0 else None)
        try:
            MODULE.on_main_process(Accelerator(), failing_callback)
        except RuntimeError as error:
            assert "rank 0" in str(error)
            Path(output, f"rank-{rank}.ok").write_text("propagated")
        else:
            raise AssertionError("Every rank must receive the failure")
    finally:
        dist.destroy_process_group()


class FakeDataset:
    def __init__(self, offset=0):
        self.offset = offset
        self.image_transforms = object()
        self.meta = SimpleNamespace(
            features={
                key: {"names": MODULE.PIPER_NAMES, "shape": [7]} for key in ("action", "observation.state")
            },
            episodes={
                2: {"dataset_from_index": offset + 10, "dataset_to_index": offset + 13},
                7: {"dataset_from_index": offset + 20, "dataset_to_index": offset + 22},
            },
        )

    def __len__(self):
        return 5

    def clear_image_transforms(self):
        self.image_transforms = None

    def set_image_transforms(self, transforms):
        self.image_transforms = transforms

    def __getitem__(self, row):
        assert self.image_transforms is None
        ep, frame, length, start = (2, row, 3, 10) if row < 3 else (7, row - 3, 2, 20)
        random.random()
        np.random.random()
        torch.rand(1)
        return {
            "episode_index": torch.tensor(ep),
            "frame_index": torch.tensor(frame),
            "index": torch.tensor(self.offset + start + frame),
            "action": torch.ones(3, 7),
            "action_is_pad": torch.arange(3) + frame >= length,
            "observation.state": torch.zeros(7),
            "task": "pick cube",
            "observation.images.cam_high": torch.full((3, 4, 4), 255, dtype=torch.uint8),
            "observation.images.cam_wrist": torch.full((3, 4, 4), 255, dtype=torch.uint8),
        }


class Processor:
    def __init__(self, post=False):
        self.post, self.calls = post, 99

    def reset(self):
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        if self.post:
            return value * 2 + 1
        assert "action" not in value
        value["action"] = None  # Production pipeline's empty action placeholder.
        return value


class Policy(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SimpleNamespace(config=config)
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.fail = False

    def predict_action_chunk(self, batch, noise):
        assert self.config.training_stage == "flow"
        assert not self.training
        assert "action" not in batch and "action_is_pad" not in batch and "index" not in batch
        assert batch["observation.images.cam_high"].max() == 1
        if self.fail:
            raise RuntimeError("inference failed")
        return noise[:, :, :7] * self.weight


class ActionMetricTests(unittest.TestCase):
    def test_config_and_named_piper_schema_validation(self):
        CONFIG.PiperEvalConfig().validate()
        for kwargs in (
            {"freq": -1},
            {"samples": 0},
            {"execution_steps": 0},
            {"seed": True},
            {"moving_threshold_deg": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CONFIG.PiperEvalConfig(**kwargs).validate()
        features = FakeDataset().meta.features
        self.assertEqual(MODULE.piper_names(features)[0], MODULE.PIPER_NAMES)
        features["action"]["names"] = None
        with self.assertRaisesRegex(ValueError, "explicitly named"):
            MODULE.piper_names(features)

    def test_rank_zero_callback_errors_are_not_swallowed(self):
        accelerator = SimpleNamespace(
            is_main_process=True, device="cpu", reduce=lambda value, reduction: value
        )
        self.assertEqual(MODULE.on_main_process(accelerator, lambda: 42), 42)

        def fail():
            raise ValueError("broken sample")

        with self.assertRaisesRegex(RuntimeError, "broken sample"):
            MODULE.on_main_process(accelerator, fail)

    def test_two_rank_evaluation_and_error_propagation(self):
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
        except PermissionError:
            self.skipTest("Execution environment forbids the loopback sockets required by Gloo")
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                distributed_worker, args=(str(Path(directory) / "rendezvous"), directory), nprocs=2, join=True
            )
            if list(Path(directory).glob("*.blocked")):
                self.skipTest("Gloo transport initialization is denied by the execution environment")
            self.assertTrue(all((Path(directory) / f"rank-{rank}.ok").exists() for rank in (0, 1)))

    def test_units_padding_baseline_and_delta_error(self):
        target = np.zeros((2, 3, 7))
        target[:, :, :6] = np.pi / 2
        target[:, :, 6] = 0.5
        predicted = target.copy()
        predicted[:, :, :6] += np.pi / 180
        predicted[:, :, 6] += 0.1
        valid = np.array([[True, True, False], [True, False, False]])
        predicted[~valid] = np.nan
        scores = MODULE.action_metrics(
            predicted, target, valid, np.zeros_like(target), MODULE.PIPER_NAMES, [1, 3]
        )
        self.assertAlmostEqual(scores["h3/joint_mae_deg"], 1)
        self.assertAlmostEqual(scores["h3/joint_rmse_deg"], 1)
        self.assertAlmostEqual(scores["h3/gripper_mae"], 0.1)
        self.assertAlmostEqual(scores["h3/joint_mae_deg_ratio_to_hold"], 1 / 90)
        self.assertEqual(scores["h3/valid_pairs"], 3)
        self.assertEqual(scores["h3/adjacent_pairs"], 1)
        self.assertAlmostEqual(scores["h3/joint_delta_mae_deg_per_step"], 0)
        self.assertEqual(scores["h3/moving_pairs"], 3)
        predicted[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            MODULE.action_metrics(predicted, target, valid, target, MODULE.PIPER_NAMES, [1])

    def test_zero_hold_error_and_empty_moving_subset_are_not_fake_zero_scores(self):
        zeros = np.zeros((1, 1, 7))
        scores = MODULE.action_metrics(
            zeros, zeros, np.ones((1, 1), dtype=bool), zeros, MODULE.PIPER_NAMES, [1]
        )
        self.assertNotIn("h1/joint_mae_deg_ratio_to_hold", scores)
        self.assertNotIn("h1/moving_joint_mae_deg", scores)
        self.assertEqual(scores["h1/moving_pairs"], 0)

    def test_fixed_inference_repeats_without_changing_rng_stage_or_training_processors(self):
        dataset, val_dataset = FakeDataset(), FakeDataset(100)
        original_transforms = dataset.image_transforms
        config = SimpleNamespace(
            training_stage="next_action",
            chunk_size=3,
            max_action_dim=9,
            num_inference_steps=10,
            input_features=[
                "observation.state",
                "observation.images.cam_high",
                "observation.images.cam_wrist",
            ],
            image_features=["observation.images.cam_high", "observation.images.cam_wrist"],
        )
        options = SimpleNamespace(seed=10, samples=2, execution_steps=2, moving_threshold_deg=1)
        policy, pre, post = Policy(config), Processor(), Processor(post=True)
        python_state, numpy_state, torch_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        with tempfile.TemporaryDirectory() as directory:
            evaluator = MODULE.PiperActionEvaluator(
                {"train": dataset, "val": val_dataset},
                config,
                pre,
                post,
                options,
                directory,
                torch.device("cpu"),
            )
            a = evaluator.evaluate(policy, 0, nullcontext)
            b = evaluator.evaluate(policy, 50, nullcontext)
            a.pop("action_eval_seconds")
            b.pop("action_eval_seconds")
            self.assertEqual(a, b)
            self.assertEqual(a["action_train/h3/valid_pairs"], 4)
            self.assertIn("action_val/h2/joint_mae_deg", a)
            # Compare against independently generated noise and the postprocessor's physical scale.
            noises = [
                torch.randn((1, 3, 9), generator=torch.Generator().manual_seed(10 + index))[0, :, :7].numpy()
                for index in (10, 21)
            ]
            valid = np.array([[True, True, True], [True, False, False]])
            expected = MODULE.action_metrics(
                np.array(noises) * 2 + 1,
                np.ones((2, 3, 7)),
                valid,
                np.zeros((2, 3, 7)),
                MODULE.PIPER_NAMES,
                [2],
            )
            self.assertAlmostEqual(a["action_train/h2/joint_mae_deg"], expected["h2/joint_mae_deg"])
            policy.fail = True
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                evaluator.evaluate(policy, 100, nullcontext)
            records = [
                json.loads(line) for line in (Path(directory) / "action_eval.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["step"] for row in records], [0, 50])
        self.assertEqual(config.training_stage, "next_action")
        self.assertTrue(policy.training)
        self.assertIs(dataset.image_transforms, original_transforms)
        self.assertEqual((pre.calls, post.calls), (99, 99))
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_actual_updates_distinguish_frozen_layers_and_zero_learning_rate(self):
        model = torch.nn.Module()
        model.vlm_prompt_tokens = torch.nn.Embedding(2, 3)
        model.prompt_tokens = torch.nn.Embedding(2, 3)
        model.action_in_proj = torch.nn.Linear(3, 3).requires_grad_(False)
        model.action_out_proj = torch.nn.Linear(3, 3)
        policy = SimpleNamespace(model=model)
        monitor = MODULE.ActionParameterMonitor(policy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=0.1)
        # The VLM has no gradient in action-only priming.
        (model.prompt_tokens.weight.sum() + model.action_out_proj.weight.sum()).backward()
        monitor.before_step()
        optimizer.step()
        zero_update = monitor.after_step()
        self.assertGreater(zero_update["action_update/prompt_tokens/grad_rms"], 0)
        self.assertEqual(zero_update["action_update/prompt_tokens/relative_update"], 0)
        self.assertEqual(zero_update["action_update/vlm_prompt_tokens/grad_numel"], 0)
        optimizer.param_groups[0]["lr"] = 0.1
        monitor.before_step()
        optimizer.step()
        metrics = monitor.after_step()
        self.assertGreater(metrics["action_update/action_out_proj/update_rms"], 0)
        self.assertEqual(metrics["action_update/action_in_proj/trainable_numel"], 0)
        self.assertEqual(metrics["action_update/action_in_proj/update_rms"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
