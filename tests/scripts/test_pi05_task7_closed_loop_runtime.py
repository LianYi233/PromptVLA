"""CPU integration fixtures, not real model or robot performance tests.

Actual native PI0.5 sampling/queue methods, evaluator rollout, and LiberoEnv
reset/step methods are loaded by AST to avoid optional GPU/transformer/MuJoCo
imports. Native processor converters are loaded directly, so the policy sees
the same transition envelope as deployment. Only the expensive backbone and
simulator physics are tiny fixtures.
"""

import ast
import copy
import importlib.util
import json
import os
from collections import deque
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import einops
import numpy as np
import pytest
import torch

from lerobot.policies.pi05.action_time import RelativeActionTimeEncoding
from lerobot.scripts import pi05_task7_closed_loop_runtime as runtime
from lerobot.scripts import pi05_task7_dualseed_shift as dual
from lerobot.scripts import pi05_task7_object_shift as objects
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _native_methods():
    path = ROOT / "src/lerobot/policies/pi05/modeling_pi05.py"
    tree = ast.parse(path.read_text())
    nodes = []
    selected = {
        "PI05Pytorch": {"sample_actions", "sample_noise"},
        "PI05Policy": {"predict_action_chunk", "select_action"},
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "make_pi05_att_2d_masks":
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name in selected:
            nodes.extend(item for item in node.body if isinstance(item, ast.FunctionDef)
                         and item.name in selected[node.name])
    assert len(nodes) == 5, "Native PI0.5 API moved; update the integration fixture"
    namespace = {
        "torch": torch, "ACTION": "action", "OBS_LANGUAGE_TOKENS": OBS_LANGUAGE_TOKENS,
        "OBS_LANGUAGE_ATTENTION_MASK": OBS_LANGUAGE_ATTENTION_MASK,
    }
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                           *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace


def _configuration():
    return {
        "action_time_encoding": True, "action_time_basis": "index", "action_time_injection": "input",
        "chunk_size": 50, "max_action_dim": 32, "n_action_steps": 10,
        "num_inference_steps": 4, "training_stage": "flow", "num_vlm_prompt_tokens": 2,
        "num_prompt_tokens": 0, "rtc_config": None, "compile_model": False,
        "output_features": {"action": {"shape": [7]}},
    }


class TinyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_time_encoder = RelativeActionTimeEncoding(32, encoding_dim=8, fps=10)
        with torch.no_grad():
            self.action_time_encoder.weight.copy_(torch.arange(256).reshape(32, 8) / 2560)
        self.paligemma_with_expert = SimpleNamespace(forward=self._prefix)
        self.rtc_processor = None
        self.sample_actions = _native_methods()["sample_actions"].__get__(self)
        self.sample_noise = _native_methods()["sample_noise"].__get__(self)
        self.raise_on_denoise = False

    def _rtc_enabled(self):
        return False

    def embed_prefix(self, images, img_masks, tokens, masks):
        embeddings = tokens.float().unsqueeze(-1).expand(-1, -1, 4) + images[0].mean()
        return embeddings, masks, torch.zeros_like(masks)

    def _prepare_attention_masks_4d(self, value):
        return value.unsqueeze(1)

    def _prefix(self, **kwargs):
        embeddings = kwargs["inputs_embeds"][0]
        return [embeddings, None], {"condition": embeddings.mean()}

    def denoise_step(self, *, prefix_pad_masks, past_key_values, x_t, timestep):
        if self.raise_on_denoise:
            raise RuntimeError("synthetic denoising failure")
        residual = self.action_time_encoder(x_t) - x_t
        return 0.2 * x_t + 0.4 * residual + 0.01 * past_key_values["condition"]


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = _configuration()
        config["output_features"] = {"action": SimpleNamespace(shape=(7,))}
        self.config = SimpleNamespace(**config)
        self.model = TinyModel(self.config)
        self.predict_action_chunk = _native_methods()["predict_action_chunk"].__get__(self)
        self.select_action = _native_methods()["select_action"].__get__(self)
        self.reset()

    def reset(self):
        self._action_queue = deque()

    def _rtc_enabled(self):
        return False

    def _preprocess_images(self, batch):
        return [batch["observation.images.camera"]], [torch.ones(1, dtype=torch.bool)]


def _observation(value=0.25):
    return {
        "observation.images.camera": torch.full((1, 3, 4, 4), float(value)),
        "observation.state": torch.zeros(1, 7),
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 5, 7]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "task": ["put both objects in the basket"],
    }


@lru_cache(maxsize=1)
def _native_transition_converters():
    # Import the actual converter module without importing the optional tokenizer
    # and every other processor from lerobot.processor.__init__.
    path = ROOT / "src/lerobot/processor/converters.py"
    spec = importlib.util.spec_from_file_location("task7_native_converters", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_processor_batch(batch):
    converters = _native_transition_converters()
    return converters.transition_to_batch(converters.batch_to_transition(batch))


def _actions(policy, count=13):
    return torch.stack([
        policy.select_action(_native_processor_batch(_observation(index / 20)))
        for index in range(count)
    ])


def test_native_processor_envelope_preserves_inputs_actions_rng_and_observation_hash():
    from lerobot.scripts.pi05_task7_full_sampling import observation_sha256

    raw = _observation()
    batch = _native_processor_batch(raw)
    # These are inserted even when the raw deployment input has no labels.
    assert set(batch) - set(raw) == {"action", "next.reward", "next.done", "next.truncated", "info"}
    assert batch["action"] is None and batch["info"] == {}
    assert batch["next.reward"] == 0.0
    assert batch["next.done"] is False and batch["next.truncated"] is False
    assert all(batch[key] is value for key, value in raw.items())

    policy = TinyPolicy()
    torch.manual_seed(234)
    baseline = policy.select_action(batch)
    expected_rng = torch.random.get_rng_state().clone()
    policy.reset()
    original_predict = policy.predict_action_chunk
    forwarded = []

    def capture_batch(item):
        forwarded.append(item)
        return original_predict(item)

    policy.predict_action_chunk = capture_batch
    torch.manual_seed(234)
    with runtime.audit_policy(policy, "correct") as audit:
        observed = policy.select_action(batch)
    assert len(forwarded) == 1 and forwarded[0] is batch
    torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert audit["chunks"][0]["observation_sha256"] == observation_sha256(raw)
    assert batch["action"] is None and batch["info"] == {}
    assert all(batch[key] is value for key, value in raw.items())


@pytest.mark.parametrize("reward,done,truncated", [
    (0.0, False, False),
    (torch.tensor(0.0), torch.tensor(False), torch.tensor(False)),
    (torch.tensor([0.0]), torch.tensor([False]), torch.tensor([False])),
])
def test_default_transition_metadata_may_be_scalar_or_singleton_tensor(reward, done, truncated):
    raw = _observation()
    batch = _native_processor_batch(raw)
    batch.update({"next.reward": reward, "next.done": done, "next.truncated": truncated})
    observation = runtime._deployment_observation_batch(batch)
    assert set(observation) == set(raw)
    assert all(observation[key] is value for key, value in raw.items())
    assert batch["next.reward"] is reward and batch["next.done"] is done
    assert batch["next.truncated"] is truncated and batch["action"] is None


@pytest.mark.parametrize("field,value", [
    ("action", torch.zeros(1, 50, 7)),
    ("action", torch.empty(0)),
    ("action", []),
    ("action_is_pad", torch.zeros(1, 50, dtype=torch.bool)),
    ("future_action", None),
    ("next.reward", 1.0),
    ("next.reward", float("nan")),
    ("next.reward", torch.zeros(2)),
    ("next.done", True),
    ("next.truncated", torch.tensor([True])),
    ("info", {"future_action": torch.zeros(1, 50, 7)}),
    ("info", []),
])
def test_populated_or_unknown_deployment_fields_fail_before_native_sampling(field, value):
    policy = TinyPolicy()
    batch = _native_processor_batch(_observation())
    batch[field] = value
    torch.manual_seed(234)
    expected_rng = torch.random.get_rng_state().clone()
    with runtime.audit_policy(policy, "correct") as audit:
        with pytest.raises(ValueError) as error:
            policy.select_action(batch)
    assert field in str(error.value)
    assert audit == {"chunks": [], "normalized_actions": []}
    assert not policy._action_queue
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)


def test_correct_instrumentation_preserves_native_actions_rng_and_queue():
    policy = TinyPolicy()
    torch.manual_seed(234)
    baseline = _actions(policy)
    expected_rng = torch.random.get_rng_state().clone()
    policy.reset()
    torch.manual_seed(234)
    with runtime.audit_policy(policy, "correct") as audit:
        observed = _actions(policy)
    torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert len(audit["chunks"]) == 2 and len(audit["normalized_actions"]) == 13
    assert [chunk["start_step"] for chunk in audit["chunks"]] == [0, 10]
    assert all(chunk["flow_times"] == [1.0, 0.75, 0.5, 0.25] for chunk in audit["chunks"])


def test_all_interventions_keep_native_noise_and_restore_weights_methods_after_failure():
    policy = TinyPolicy()
    weights = copy.deepcopy(policy.state_dict())
    original_predict, original_select = policy.predict_action_chunk, policy.select_action
    original_noise, original_denoise = policy.model.sample_noise, policy.model.denoise_step
    predictions, noises, inputs, end_rng = {}, {}, {}, {}
    for mode in runtime.MODES:
        torch.manual_seed(1401)
        policy.reset()
        with runtime.audit_policy(policy, mode) as audit:
            predictions[mode] = _actions(policy)
        noises[mode] = [chunk["initial_noise_sha256"] for chunk in audit["chunks"]]
        inputs[mode] = [chunk["observation_sha256"] for chunk in audit["chunks"]]
        end_rng[mode] = torch.random.get_rng_state().clone()
    assert noises["correct"] == noises["zero"] == noises["reversed"]
    assert inputs["correct"] == inputs["zero"] == inputs["reversed"]
    assert not torch.equal(predictions["correct"], predictions["zero"])
    assert not torch.equal(predictions["correct"], predictions["reversed"])
    torch.testing.assert_close(end_rng["correct"], end_rng["zero"], rtol=0, atol=0)
    torch.testing.assert_close(end_rng["correct"], end_rng["reversed"], rtol=0, atol=0)
    policy.reset()
    policy.model.raise_on_denoise = True
    with pytest.raises(RuntimeError, match="synthetic denoising failure"):
        with runtime.audit_policy(policy, "reversed"):
            _actions(policy, 1)
    assert policy.predict_action_chunk == original_predict and policy.select_action == original_select
    assert policy.model.sample_noise == original_noise and policy.model.denoise_step == original_denoise
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [("n_action_steps", 50), ("compile_model", True),
                                         ("action_time_basis", "delta_t")])
def test_runtime_rejects_changed_inference_contract(field, value):
    policy = TinyPolicy()
    setattr(policy.config, field, value)
    with pytest.raises(ValueError):
        with runtime.audit_policy(policy, "zero"):
            pass


class _BaseEnv:
    def reset(self, **kwargs):
        pass


@lru_cache(maxsize=1)
def _native_env_class():
    path = ROOT / "src/lerobot/envs/libero.py"
    parsed = ast.parse(path.read_text())
    node = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "LiberoEnv")
    methods = {"reset", "step", "task7_start_trace", "task7_get_trace"}
    node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
    assert len(node.body) == len(methods)
    node.bases = [ast.Name(id="_BaseEnv", ctx=ast.Load())]
    namespace = {"_BaseEnv": _BaseEnv, "os": os, "np": np,
                 "get_libero_dummy_action": lambda: [0, 0, 0, 0, 0, 0, -1]}
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                           node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace["LiberoEnv"]


class FakePhysics:
    def __init__(self, stop_after=15):
        self.count = 0
        self.state_id = 0
        self.stop_after = stop_after
        self.seed_calls = []
        self.commands = []
        self.robots = [SimpleNamespace(eef_site_id=0, controller=SimpleNamespace(
            ee_ori_mat=np.eye(3) * -1, use_delta=True, goal_pos=np.zeros(3), goal_ori=np.eye(3),
        ))]
        self.env = SimpleNamespace(objects_dict={name: SimpleNamespace(joints=[name])
                                               for name in objects.TARGET_OBJECTS + objects.OTHER_OBJECTS})
        self.sim = SimpleNamespace(data=SimpleNamespace(
            get_joint_qpos=self.pose, time=0.0, ncon=1,
            contact=[SimpleNamespace(geom1=2, geom2=5)], site_xmat=np.eye(3).reshape(1, 9),
        ), model=SimpleNamespace(geom_id2name=lambda number: f"fixture_geom{number}"))

    def pose(self, name):
        return np.array([0.01 * self.count, 0.01 * self.state_id, 0.8, 1, 0, 0, 0])

    def observation(self):
        return {
            "robot0_eef_pos": np.array([0.01 * self.count, 0.0, 0.9]),
            "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
            "robot0_gripper_qpos": np.array([0.02, -0.02]),
            "robot0_gripper_qvel": np.zeros(2),
        }

    def seed(self, seed):
        self.seed_calls.append(seed)

    def reset(self):
        self.count = 0
        self.sim.data.time = 0.0
        return self.observation()

    def set_init_state(self, state):
        self.state_id = int(state)
        return self.observation()

    def step(self, action):
        self.count += 1
        self.sim.data.time += 0.05
        self.commands.append(np.asarray(action).copy())
        return self.observation(), 0.0, False, {}

    def check_success(self):
        return self.count >= self.stop_after

    def get_sim_state(self):
        return np.array([self.sim.data.time, self.count, self.state_id, 0.8])


def _physical_env(stop_after=15):
    env = _native_env_class()()
    env.task_id, env.task_suite_name, env.is_libero_plus = 7, "libero_10", False
    env.task, env.task_description = "synthetic task7", "put both objects in the basket"
    env.control_mode = "relative"
    env.init_states, env._init_states, env.init_state_id = True, list(range(10)), 0
    env._reset_stride, env.num_steps_wait, env._max_episode_steps = 1, 2, 20
    env._env = FakePhysics(stop_after)
    env._ensure_env = lambda: None
    env._format_raw_obs = lambda observation: observation
    return env


class FakeVector:
    num_envs = 1

    def __init__(self, env):
        self.env = env

    def call(self, name, *args):
        value = getattr(self.env, name)
        return (value(*args) if callable(value) else value,)

    def reset(self, seed):
        raw, info = self.env.reset(seed=seed[0])
        return _observation(raw["robot0_eef_pos"][0]), info

    def step(self, action):
        raw, reward, terminated, truncated, info = self.env.step(action[0])
        return (_observation(raw["robot0_eef_pos"][0]), np.array([reward]),
                np.array([terminated]), np.array([truncated]), {"is_success": np.array([info["is_success"]])})


@lru_cache(maxsize=1)
def _native_rollout():
    path = ROOT / "src/lerobot/scripts/lerobot_eval.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rollout")
    namespace = {
        "torch": torch, "np": np, "nn": torch.nn, "einops": einops, "deepcopy": copy.deepcopy,
        "ACTION": "action", "OBS_STR": "observation", "inside_slurm": lambda: False,
        "trange": lambda *args, **kwargs: SimpleNamespace(set_postfix=lambda *a, **k: None,
                                                         update=lambda *a, **k: None),
        "preprocess_observation": lambda value: value, "check_env_attributes_and_types": lambda env: None,
    }
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                           node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace["rollout"]


def _config(tmp_path, mode="correct"):
    checkpoint = tmp_path / "synthetic-checkpoint"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text(json.dumps(_configuration()))
    return {
        "protocol": runtime.PROTOCOL, "intervention": mode,
        "executed_steps": 10, "horizon": 50, "episodes": 10,
        "scene_seed": 20261103, "policy_seed": 1001, "init_seed": 3000,
        "checkpoint": str(checkpoint), "trace_directory": str(tmp_path / mode / "trajectories"),
    }


def _installed_fixture(tmp_path, monkeypatch, mode="correct", stop_after=15):
    config = _config(tmp_path, mode)
    path = tmp_path / f"{mode}-config.json"
    path.write_text(json.dumps(config))
    dual_config = {"protocol": dual.PROTOCOL, "suite": "libero_10", "task_id": 7,
                   "scene_seed": config["scene_seed"], "policy_seed": config["policy_seed"],
                   "eval_seed": config["init_seed"], "shift_m": 0.12}
    dual_path = tmp_path / "dual-shift.json"
    dual_path.write_text(json.dumps(dual_config))
    monkeypatch.setenv(runtime.CONFIG_ENV, str(path))
    monkeypatch.setenv("LIBERO_TASK7_DUALSEED_SHIFT_CONFIG", str(dual_path))
    # Fake physics has the same known scene objects but no MuJoCo forward method.
    monkeypatch.setattr(dual, "apply_shift", lambda env, config, seed: env._env.observation())
    module = SimpleNamespace(rollout=_native_rollout())

    def run_one(task_group, task_id, env, *, policy, n_episodes, start_seed):
        successes = []
        for index in range(n_episodes):
            result = module.rollout(
                env=env, policy=policy, env_preprocessor=_native_processor_batch,
                env_postprocessor=lambda item: item, preprocessor=_native_processor_batch,
                # Distinguish the actual controller command from normalized output.
                postprocessor=lambda action: action * 0.1 + 0.02,
                seeds=[start_seed + index],
            )
            successes.append(bool(result["success"].any()))
        return task_group, task_id, {"successes": successes}

    module.run_one = run_one
    dual.install(module, dual_path)
    manifest = runtime.install(module, path)
    return module, FakeVector(_physical_env(stop_after)), TinyPolicy(), config, manifest


def test_installed_native_loop_keeps_terminal_state_and_logs_all_ten_seeded_episodes(tmp_path, monkeypatch):
    module, env, policy, config, manifest = _installed_fixture(tmp_path, monkeypatch)
    outcome = module.run_one("libero_10", 7, env, policy=policy, n_episodes=10, start_seed=3000)
    assert outcome[2]["successes"] == [True] * 10
    paths = runtime.require_saved_traces(config, successes=outcome[2]["successes"])
    assert len(paths) == 10 and len(env.env._env.seed_calls) == 20
    assert env.env._env.seed_calls == [value for seed in range(3000, 3010) for value in (seed, None)]
    for slot, path in enumerate(paths):
        trace = json.loads(path.read_text())
        assert trace["init_state_id"] == slot and trace["rollout_seed"] == 3000 + slot
        assert trace["configuration"] == config and trace["source_fingerprints"] == runtime.source_fingerprints()
        assert trace["initial_state"]["eef_position"][0] == 0.02
        assert trace["steps"][-1]["state_after"]["eef_position"][0] == 0.15
        # Automatic reset has happened, but its state must not replace the terminal one.
        assert len(trace["steps"]) == 13 and len(trace["chunks"]) == 2
        assert trace["termination_reason"] == "environment_termination"
        assert trace["steps"][-1]["terminated"] is True
        assert trace["initial_state"]["eef_rotation_source"] == "post-physics simulator site_xmat"
        assert trace["initial_state"]["eef_rotation_matrix"] == np.eye(3).tolist()
        for step in trace["steps"]:
            np.testing.assert_allclose(step["executed_action"], np.asarray(step["normalized_action"]) * 0.1 + 0.02,
                                       rtol=1e-6, atol=1e-8)
    assert manifest["task7_time_intervention"]["config"] == config


def test_installed_modes_pair_initial_states_inputs_and_native_noise(tmp_path, monkeypatch):
    traces = {}
    for mode in runtime.MODES:
        module, env, policy, config, _ = _installed_fixture(tmp_path, monkeypatch, mode)
        module.run_one("libero_10", 7, env, policy=policy, n_episodes=10, start_seed=3000)
        traces[mode] = json.loads((Path(config["trace_directory"]) / "episode-000.json").read_text())
    for mode in ("zero", "reversed"):
        assert traces[mode]["policy_rng_seed"] == traces["correct"]["policy_rng_seed"]
        assert traces[mode]["initial_state"] == traces["correct"]["initial_state"]
        assert [chunk["initial_noise_sha256"] for chunk in traces[mode]["chunks"]] == [
            chunk["initial_noise_sha256"] for chunk in traces["correct"]["chunks"]]
        assert traces[mode]["chunks"][0]["observation_sha256"] == traces["correct"]["chunks"][0]["observation_sha256"]
    assert traces["correct"]["chunks"][0]["actions"] != traces["zero"]["chunks"][0]["actions"]


def test_native_time_limit_is_logged_as_truncation_without_extra_reset(tmp_path, monkeypatch):
    module, env, policy, config, _ = _installed_fixture(tmp_path, monkeypatch, stop_after=100)
    result = module.run_one("libero_10", 7, env, policy=policy, n_episodes=10, start_seed=3000)
    assert result[2]["successes"] == [False] * 10
    trace = json.loads((Path(config["trace_directory"]) / "episode-000.json").read_text())
    assert len(trace["steps"]) == 20 and trace["steps"][-1]["truncated"] is True
    assert trace["termination_reason"] == "native_evaluator_time_limit"
    assert env.env._env.seed_calls == list(range(3000, 3010))


def test_cache_requires_complete_matching_traces_and_native_inference_count(tmp_path, monkeypatch):
    module, env, policy, config, _ = _installed_fixture(tmp_path, monkeypatch)
    module.run_one("libero_10", 7, env, policy=policy, n_episodes=10, start_seed=3000)
    paths = runtime.require_saved_traces(config, successes=[True] * 10)
    original = json.loads(paths[0].read_text())
    for field, replacement in (("intervention", "zero"), ("policy_rng_seed", 0),
                                ("source_fingerprints", {}), ("status", "running")):
        broken = copy.deepcopy(original)
        broken[field] = replacement
        paths[0].write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            runtime.require_saved_traces(config, successes=[True] * 10)
    paths[0].write_text(json.dumps(original))
    with pytest.raises(ValueError, match="official paired outcome"):
        runtime.require_saved_traces(config, successes=[False] + [True] * 9)
    checkpoint_path = Path(config["checkpoint"]) / "config.json"
    changed = json.loads(checkpoint_path.read_text())
    changed["num_inference_steps"] = 5
    checkpoint_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="denoising count"):
        runtime.require_saved_traces(config)
    checkpoint_path.write_text(json.dumps(_configuration()))
    paths[-1].unlink()
    with pytest.raises(ValueError, match="exactly ten"):
        runtime.require_saved_traces(config)


def test_failed_new_attempt_cannot_leave_stale_completed_trace(tmp_path, monkeypatch):
    module, env, policy, config, _ = _installed_fixture(tmp_path, monkeypatch)
    module.run_one("libero_10", 7, env, policy=policy, n_episodes=10, start_seed=3000)
    target = Path(config["trace_directory"]) / "episode-000.json"
    assert target.exists()
    policy.model.raise_on_denoise = True
    with pytest.raises(RuntimeError, match="synthetic denoising failure"):
        module.run_one("libero_10", 7, env, policy=policy, n_episodes=10, start_seed=3000)
    assert not target.exists()


def test_shell_default_manifest_unchanged_and_opt_in_validates_traces():
    source = (ROOT / "run_eval_libero-full.sh").read_text()
    assert 'time_intervention_manifest = {}' in source
    assert '**time_intervention_manifest,' in source
    assert source.index("dualseed_shift_manifest = install_dualseed_shift") < source.index(
        "time_intervention_manifest = install_time_intervention") < source.index("original_rollout = module.rollout")
    assert 'require_saved_traces(trace_config, successes=metrics["successes"])' in source
