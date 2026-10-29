"""Exercise the real LIBERO adapter methods without a GPU or simulator install.

Run directly with ``python tests/envs/test_libero_placement_retry.py``. Extracting
the methods avoids importing optional rendering dependencies; their executable
bodies are compiled unchanged, and only the Gym base and simulator are fakes.
"""

from __future__ import annotations

import ast
import gc
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "src/lerobot/envs/libero.py"


class RandomizationError(Exception):
    pass


class GymEnv:
    def reset(self, seed=None):
        self.base_reset_seeds.append(seed)


class Simulator:
    def __init__(self):
        self.events = []
        self.robots = [types.SimpleNamespace(controller=types.SimpleNamespace(use_delta=None))]

    def reset(self):
        self.events.append(("reset",))
        return {"observation": "random placement"}

    def seed(self, seed):
        self.events.append(("seed", seed))
        np.random.seed(seed)

    def set_init_state(self, state):
        self.events.append(("initial_state", state))
        return {"observation": "recorded initial state"}

    def step(self, action):
        self.events.append(("step", action))
        return {"observation": "settled recorded state"}, 0, False, {}


def load_adapter(constructor):
    """Compile production methods unchanged, retaining their normal super()."""
    tree = ast.parse(SOURCE.read_text())
    original = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LiberoEnv"
    )
    methods = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_ensure_env", "reset"}
    ]
    dummy_action = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_libero_dummy_action"
    )
    adapter = ast.ClassDef(
        name="LiberoEnv", bases=[ast.Name(id="GymEnv", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[]
    )
    namespace = {
        "__file__": str(SOURCE),
        "__name__": "libero_adapter_under_test",
        "GymEnv": GymEnv,
        "OffScreenRenderEnv": constructor,
        "RandomizationError": RandomizationError,
        "gc": gc,
        "logging": logging,
        "logger": logging.getLogger("libero_adapter_under_test"),
        "Path": Path,
        "np": np,
    }
    # Keep literal configuration constants from production without importing its
    # optional dependencies. No retry behavior is reimplemented here.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                namespace[target.id] = literal
    module = ast.fix_missing_locations(ast.Module(body=[dummy_action, adapter], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    instance = namespace["LiberoEnv"]()
    instance.__dict__.update(
        _env=None,
        task="test_recorded_scene",
        task_id=2076,
        _task_bddl_file="/benchmark/test_recorded_scene.bddl",
        observation_height=224,
        observation_width=224,
        init_states=True,
        is_libero_plus=True,
        _init_states=[np.array([10.0]), np.array([20.0]), np.array([30.0])],
        init_state_id=4,
        _reset_stride=2,
        num_steps_wait=2,
        control_mode="relative",
        base_reset_seeds=[],
        _format_raw_obs=lambda observation: observation,
    )
    return instance


class PlacementRetryTests(unittest.TestCase):
    def setUp(self):
        errors = types.ModuleType("robosuite.utils.errors")
        errors.RandomizationError = RandomizationError
        stubs = {
            "robosuite": types.ModuleType("robosuite"),
            "robosuite.utils": types.ModuleType("robosuite.utils"),
            "robosuite.utils.errors": errors,
        }
        self.modules = patch.dict(sys.modules, stubs)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)

    def test_first_attempt_and_repeated_ensure_preserve_existing_behavior(self):
        simulator = Simulator()
        arguments = []

        def construct(**kwargs):
            arguments.append(kwargs)
            return simulator

        env = load_adapter(construct)
        env._ensure_env()
        env._ensure_env()
        self.assertIs(env._env, simulator)
        self.assertEqual(
            arguments,
            [{"bddl_file_name": env._task_bddl_file, "camera_heights": 224, "camera_widths": 224}],
        )
        self.assertEqual(simulator.events, [("reset",)])

    def test_transient_placement_errors_advance_rng_before_success(self):
        simulator = Simulator()
        samples = []
        np.random.seed(1000)
        expected = np.random.RandomState(1000).random_sample(4)

        def construct(**kwargs):
            samples.append(np.random.random())
            if len(samples) <= 2:
                raise RandomizationError("Cannot place all objects ):")
            return simulator

        env = load_adapter(construct)
        with self.assertLogs(level="WARNING") as logged:
            env._ensure_env()
        self.assertIs(env._env, simulator)
        np.testing.assert_array_equal(samples, expected[:3])
        self.assertEqual(np.random.random(), expected[3])
        self.assertEqual(simulator.events, [("reset",)])
        self.assertTrue(any(env._task_bddl_file in message for message in logged.output))

    def test_persistent_placement_failure_is_bounded_and_remains_an_error(self):
        calls = []
        failure = RandomizationError("Cannot place all objects ):")

        def construct(**kwargs):
            calls.append(kwargs)
            raise failure

        env = load_adapter(construct)
        with self.assertLogs(level="WARNING"):
            with self.assertRaises(RandomizationError) as raised:
                env._ensure_env()
        self.assertIs(raised.exception, failure)
        self.assertEqual(len(calls), 20)
        self.assertIsNone(env._env)
        self.assertEqual(env.init_state_id, 4)

    def test_unrelated_errors_and_interrupts_are_not_retried(self):
        for failure in (ValueError("bad input"), RuntimeError("CUDA out of memory"), KeyboardInterrupt()):
            with self.subTest(error=type(failure).__name__):
                construct = Mock(side_effect=failure)
                env = load_adapter(construct)
                with self.assertRaises(type(failure)) as raised:
                    env._ensure_env()
                self.assertIs(raised.exception, failure)
                self.assertEqual(construct.call_count, 1)
                self.assertIsNone(env._env)

    def test_other_libero_benchmarks_keep_original_constructor_behavior(self):
        calls = []
        failure = RandomizationError("Cannot place all objects ):")

        def construct(**kwargs):
            calls.append(kwargs)
            raise failure

        env = load_adapter(construct)
        env.is_libero_plus = False
        with self.assertRaises(RandomizationError) as raised:
            env._ensure_env()
        self.assertIs(raised.exception, failure)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(env._env)

    def test_recovered_constructor_keeps_episode_seed_and_recorded_initial_state(self):
        simulator = Simulator()
        calls = []

        def construct(**kwargs):
            calls.append(kwargs)
            np.random.random()
            if len(calls) == 1:
                raise RandomizationError("Cannot place all objects ):")
            return simulator

        env = load_adapter(construct)
        expected_state = env._init_states[1]
        with self.assertLogs(level="WARNING"):
            observation, info = env.reset(seed=4321)
        self.assertEqual(len(calls), 2)
        self.assertEqual(env.base_reset_seeds, [4321])
        self.assertEqual(
            [event[0] for event in simulator.events],
            ["reset", "seed", "reset", "initial_state", "step", "step"],
        )
        self.assertEqual(simulator.events[1], ("seed", 4321))
        self.assertIs(simulator.events[3][1], expected_state)
        self.assertEqual(env.init_state_id, 6)
        self.assertTrue(simulator.robots[0].controller.use_delta)
        self.assertEqual(observation, {"observation": "settled recorded state"})
        self.assertEqual(info, {"is_success": False})
        self.assertEqual(np.random.random(), np.random.RandomState(4321).random_sample())


if __name__ == "__main__":
    unittest.main(verbosity=2)
