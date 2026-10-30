"""Exercise the actual new probes/hooks/replay code with small CPU stand-ins."""
from __future__ import annotations

import copy
import json
import random
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.analysis.curriculum_study import core, train, perturb
from examples.analysis.curriculum_study.analysis import read_new_run, plot_new, export_trajectories
from examples.analysis.curriculum_study.__main__ import prepare, launch, training_args
from test_protocol import fixture


class LocalAccelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1
    device = torch.device("cpu")
    autocast = staticmethod(nullcontext)
    @staticmethod
    def reduce(value, reduction="sum"):
        return value
    @staticmethod
    def gather(value):
        return value
    @staticmethod
    def unwrap_model(policy, **kwargs):
        return policy


class Data:
    image_transforms = "preserve me"
    meta = SimpleNamespace(camera_keys=["observation.images.image"])
    def clear_image_transforms(self):
        self.image_transforms = None
    def set_image_transforms(self, value):
        self.image_transforms = value
    def __getitem__(self, index):
        assert self.image_transforms is None
        return {"observation.images.image": torch.full((3, 4, 4), index, dtype=torch.uint8),
                "action": torch.tensor([[1., 2.], [3., 4.], [100., 100.]]),
                "action_is_pad": torch.tensor([False, False, True]),
                "observation.language.tokens": torch.tensor([1, 2]),
                "observation.language.attention_mask": torch.tensor([True, True]),
                "index": index, "episode_index": index, "frame_index": 0, "task_index": 0}


def collate(items):
    return {k: torch.stack([row[k] for row in items]) if isinstance(items[0][k], torch.Tensor)
            else torch.tensor([row[k] for row in items]) for k in items[0]}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
    def predict_velocity(self, images, masks, tokens, text_mask, noisy, time):
        return noisy * self.weight


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.config = SimpleNamespace(chunk_size=3, max_action_dim=4, dtype="float32",
                      image_features={"observation.images.image": None}, output_features={"action": SimpleNamespace(shape=(2,))})
    def prepare_action(self, batch):
        return torch.nn.functional.pad(batch["action"], (0, 2))
    def _preprocess_images(self, batch):
        return [batch["observation.images.image"]], [torch.tensor([True])]


def protocol():
    return {"protocol": core.SCHEMA, "scope": "test held-out", "seed": 2026,
            "timesteps": [.2, .8], "noise_repeats": 2, "probe_id": "test-id",
            "samples": [{"sample_index": i, "dataset_index": i, "index": i, "task_index": 0,
                         "episode_index": i, "frame_index": 0, "group": g, "importance": float(i)}
                        for i, g in enumerate(("low_score", "high_score"))]}


def test_probe_reproducibility_padding_and_no_training_interference(tmp_path):
    policy, data = Policy().train(), Data()
    policy.model.weight.grad = torch.tensor(.123)
    old_param = policy.model.weight
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    torch_state = torch.get_rng_state().clone()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    probe = train.FixedProbe(data, lambda x: x, policy, protocol(), tmp_path, "cpu", collate_fn=collate)
    assert data.image_transforms == "preserve me"
    probe.save(LocalAccelerator())
    first = probe.measure(policy, 0, LocalAccelerator())
    second = probe.measure(policy, 1, LocalAccelerator())
    assert first == second
    assert torch.equal(torch_state, torch.get_rng_state())
    np.testing.assert_equal(numpy_state, np.random.get_state())
    assert python_state == random.getstate()
    assert policy.model.weight is old_param and policy.training and policy.model.training
    assert policy.model.weight.grad.item() == pytest.approx(.123)
    for key, value in policy.state_dict().items():
        assert torch.equal(value, before[key])
    record = core.read_json(tmp_path / "curriculum_probe/step_000000.json")
    assert len(record["samples"]) == 8
    assert all(row["valid_coordinates"] == 4 for row in record["samples"])
    # Verify actual numbers against independently computed, valid-coordinate-only error.
    row = record["samples"][0]
    a = policy.prepare_action(probe.samples[0])
    eps = probe.noise[0, 0]
    target = (eps-a)[..., :2]
    predicted = (.2*eps+.8*a)[..., :2] * .2
    expected = (predicted[:, :2]-target[:, :2]).square().double().sum().item()
    assert row["squared_error_sum"] == pytest.approx(expected)
    manifest, rows = read_new_run(tmp_path)
    assert len(rows) == 2
    # The same parameter object still supports the next training update.
    optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
    optimizer.zero_grad()
    policy.model.weight.square().backward()
    optimizer.step()
    assert policy.model.weight.item() != .2


def test_probe_exception_restores_all_flags_rng(tmp_path, monkeypatch):
    policy, data = Policy().train(), Data()
    probe = train.FixedProbe(data, lambda x: x, policy, protocol(), tmp_path, "cpu", collate_fn=collate)
    before = torch.get_rng_state().clone()
    def fail(*args):
        torch.rand(4)
        raise ValueError("deliberate failure")
    monkeypatch.setattr(policy.model, "predict_velocity", fail)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        probe.measure(policy, 0, LocalAccelerator())
    assert policy.training and policy.model.training and policy.model.weight.requires_grad
    assert torch.equal(before, torch.get_rng_state())


def test_hooks_delegate_actual_optimizer_once_and_log_used_lr(tmp_path):
    cfg = SimpleNamespace(resume=False, dataset=SimpleNamespace(eval_split=0, image_transforms=SimpleNamespace(enable=False)),
                          prompt_rank=SimpleNamespace(freq=0), env_eval_freq=0, eval_steps=0,
                          policy=SimpleNamespace(type="pi05", train_action_expert=True, compile_model=False))
    calls = []
    policy = Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: core.lr_value(s, steps=4, warmup=1, peak=.1, end=.05)/.1)
    def update_policy(policy, optimizer, accelerator, lr_scheduler):
        calls.append("original update")
        optimizer.zero_grad()
        policy.model.weight.square().backward()
        optimizer.step()
        lr_scheduler.step()
        return object(), {}
    trainer = SimpleNamespace(make_train_eval_datasets=lambda cfg: (Data(), None),
                              make_pre_post_processors=lambda: (lambda x: x, lambda x: x),
                              update_policy=update_policy)
    originals = vars(trainer).copy()
    class Probe:
        def __init__(self, dataset, processor, policy, protocol, output, device):
            calls.append("build")
            (tmp_path / "curriculum_probe").mkdir()
        def save(self, acc):
            calls.append("save")
        def measure(self, policy, step, acc):
            calls.append(step)
    spec = {"run": str(tmp_path), "steps": 4, "warmup_steps": 1, "peak_lr": .1,
            "end_lr": .05, "probe_frequency": 2, "probe_steps": [1]}
    with train.instrument_trainer(trainer, spec, {}, probe_class=Probe):
        trainer.make_train_eval_datasets(cfg)
        trainer.make_pre_post_processors()
        for _ in range(4):
            trainer.update_policy(policy, optimizer, LocalAccelerator(), scheduler)
    assert vars(trainer) == originals
    assert calls == ["build", "save", 0, "original update", 1, "original update", 2,
                     "original update", "original update", 4]
    lrs = [json.loads(line) for line in (tmp_path / "curriculum_probe/lr.jsonl").read_text().splitlines()]
    assert [r["used_lr"][0] for r in lrs] == [core.lr_value(i, steps=4, warmup=1, peak=.1, end=.05) for i in range(4)]
    assert lrs[0]["next_lr"] != lrs[0]["used_lr"]


def test_hook_restoration_even_if_entry_fails():
    trainer = SimpleNamespace(make_train_eval_datasets=lambda: None, make_pre_post_processors=lambda: None,
                              update_policy=lambda: None)
    old = vars(trainer).copy()
    with pytest.raises(RuntimeError):
        with train.instrument_trainer(trainer, {}, {}):
            raise RuntimeError("failure")
    assert vars(trainer) == old


class Env:
    _max_episode_steps = 6
    task_description = "move object"
    _task_bddl_file = None
    def __init__(self):
        self.x = 0.
        self.t = 0
    def close(self):
        pass


class SimRuntime:
    score_manifest = None
    def __init__(self, bddl):
        self.bddl = bddl
        self.policy = SimpleNamespace(_action_queue=deque())
    def reset_policy(self):
        self.policy._action_queue.clear()
    def open(self, suite, task, episode, seed):
        perturb.seed_everything(seed)
        env = Env()
        env._task_bddl_file = self.bddl
        self.reset_policy()
        return env, self.obs(env)
    def obs(self, env):
        return {"pixels": {"image": np.full((4, 4, 3), env.t, dtype=np.uint8)},
                "state": np.array([env.x])}
    def state(self, env):
        return np.array([env.x, env.t], dtype=float)
    def action(self, obs, instruction, features=False):
        return np.array([.2 + torch.rand(1).item()*.01, 0, 0, 0, 0, 0, -1.]), None
    def step(self, env, action):
        env.x += action[0]
        env.t += 1
        success = env.x >= 1.
        return self.obs(env), success, success


def test_real_pairing_engine_replays_prefix_and_zero_control(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text("task")
    runtime = SimRuntime(bddl)
    ref = perturb.reference_rollout(runtime, "libero_goal", 0, 0, 123, [1, 3])
    assert set(ref["anchors"]) == {1, 3}
    common = runtime, ref, "libero_goal", 0, 0, 123, 3, 987
    clean = perturb.continuation(*common, 0., [0], 1)
    zero = perturb.continuation(*common, 0., [0], 1)
    assert perturb.trajectories_match(clean, zero)
    changed = perturb.continuation(*common, .05, [0], 1)
    assert np.isclose(np.linalg.norm(np.array(changed["perturbation"])), .05)
    assert not np.array_equal(clean["actions"], changed["actions"])
    ref["anchors"][3]["state"][0] += 1
    with pytest.raises(RuntimeError, match="Replayed"):
        perturb.continuation(*common, 0., [0], 1)


def test_plots_export_actual_points_and_no_overwrite(tmp_path):
    values = {"H2E": [{"step": 0, "low_score": 2., "high_score": 3.},
                      {"step": 10, "low_score": 1., "high_score": 2.}]}
    export_trajectories(values, tmp_path / "plot", {"scope": "SYNTHETIC UNIT TEST ONLY"})
    assert (tmp_path / "plot/trajectory.svg").is_file()
    assert "2.0,3.0" in (tmp_path / "plot/trajectory.csv").read_text()
    with pytest.raises(FileExistsError):
        export_trajectories(values, tmp_path / "plot", {})


def prep_args(tmp_path):
    base, scores, info = fixture()
    pair, dataset, tokenizer = tmp_path / "pair", tmp_path / "dataset", tmp_path / "tokenizer"
    (pair / "libero").mkdir(parents=True)
    (dataset / "meta").mkdir(parents=True)
    tokenizer.mkdir()
    core.write_json(pair / "pair.json", {"dataset_root": str(dataset)})
    core.write_json(pair / "libero/config.json", {"paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m"})
    (pair / "libero/model.safetensors").write_bytes(b"fixture-not-real-weights")
    (pair / "P1.safetensors").write_bytes(b"fixture-not-real-prompt")
    core.write_json(dataset / "meta/stats.json", {})
    core.save_plan(pair / "D0-W0.npz", base)
    score_path = tmp_path / "scores.npz"
    with score_path.open("wb") as stream:
        np.savez_compressed(stream, **scores, manifest=json.dumps(info))
    return SimpleNamespace(source_pair=pair, scores=score_path, output=tmp_path / "study", dataset_root=dataset,
                tokenizer=tokenizer, seed=0, steps=80, warmup_steps=5, easy_percent=30, holdout_per_task=2,
                probe_per_group=2, labels=None, peak_lr=2.5e-5, decay_lr=1e-5, probe_frequency=10,
                checkpoints=[20, 40, 80], no_wandb=True, wandb_project="test")


def test_prepare_and_launch_dry_run_covers_six_isolated_runs(tmp_path):
    args = prep_args(tmp_path)
    study = prepare(args)
    assert len(study["specs"]) == 6
    assert not (args.output / "runs").exists()
    reference = None
    for name, path in study["specs"].items():
        spec, probe = train.check_spec(path)
        flags = dict(arg[2:].split("=", 1) for arg in spec["training_args"])
        assert flags["policy.num_vlm_prompt_tokens"] == "2"
        assert flags["policy.train_action_expert"] == "true"
        assert flags["prompt_rank.freq"] == "0"
        assert json.loads(flags["save_steps"]) == [20, 40, 80]
        if reference is None:
            reference = probe
        assert probe == reference
    result = launch(SimpleNamespace(study=args.output, all=True, order="H2E", lr="cosine", gpus="0,1", port=29620, dry_run=True))
    assert result["dry_run"] and result["launched"] == []
    assert not (args.output / "runs").exists()
    with pytest.raises(FileExistsError):
        prepare(args)
    # Read-only preflight must catch input tampering before any process launches.
    (args.source_pair / "P1.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        train.check_spec(next(iter(study["specs"].values())))


def test_perturb_command_resume_coverage_and_resource_change_guard(tmp_path, monkeypatch):
    """Execute the real command orchestration; only policy/physics are small stand-ins."""
    import sys
    import types
    checkpoint = tmp_path / "checkpoint"
    tokenizer = tmp_path / "tokenizer"
    checkpoint.mkdir()
    tokenizer.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
    core.write_json(checkpoint / "config.json", {"type": "pi05"})
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        core.write_json(checkpoint / name, {"steps": []})
    paths = {key: tmp_path / key for key in ("bddl_files", "init_states", "assets")}
    for path in paths.values():
        (path / "libero_goal").mkdir(parents=True)
    bddl = paths["bddl_files"] / "libero_goal/task.bddl"
    bddl.write_text("test task")
    (paths["init_states"] / "libero_goal/task.pruned_init").write_bytes(b"test state")
    asset = paths["assets"] / "scene.xml"
    asset.write_text("test scene")
    task = SimpleNamespace(name="test", problem_folder="libero_goal", bddl_file="task.bddl",
                           init_states_file="task.pruned_init")
    suite = SimpleNamespace(tasks=[task], get_task=lambda i: task)
    for name in ("lerobot", "lerobot.envs", "lerobot.envs.libero", "lerobot.utils", "lerobot.utils.pdet",
                 "libero", "libero.libero"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["lerobot.envs.libero"]._get_suite = lambda name: suite
    sys.modules["libero.libero"].get_libero_path = lambda key: str(paths[key])
    sys.modules["lerobot.utils.pdet"].runtime_info = lambda: {"source_tree_sha256": "fixture"}
    monkeypatch.setattr(perturb, "LiberoRuntime", lambda *a, **kw: SimRuntime(bddl))
    args = SimpleNamespace(episodes=1, trials=2, seed=1000, state_atol=1e-8, action_atol=1e-5,
            anchor_steps=[1, 3, 100], action_steps=1, delta=.05, dims=[0], tasks="libero_goal:0",
            checkpoint=checkpoint, output=tmp_path / "risk", tokenizer=tokenizer, scores=None,
            device="cpu", image_size=4)
    first = perturb.run(args)
    assert first["states"] == 2 and first["paired_trials"] == 4
    assert core.read_json(args.output / "coverage.json")[0]["unreached_anchors"] == [100]
    results_before = {p.name: p.read_bytes() for p in args.output.glob("pair-*.json")}
    features = args.output / "features_template.csv"
    features.write_text(features.read_text() + "# hand edited sentinel\n")
    assert perturb.run(args) == first
    assert results_before == {p.name: p.read_bytes() for p in args.output.glob("pair-*.json")}
    assert features.read_text().endswith("# hand edited sentinel\n")
    asset.write_text("changed scene")
    with pytest.raises(ValueError, match="changed"):
        perturb.run(args)
