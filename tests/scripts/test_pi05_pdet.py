"""Factor interventions must alter only the intended inputs, including through policy forward."""

import ast
import copy
import importlib.util
import json
import random
import shlex
import subprocess
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch

from lerobot.configs.pdet import PDETConfig
from lerobot.scripts.pdet_experiment import audit_runs, check_initializations
from lerobot.utils.pdet import PDETStreams, sample_flow
from lerobot.utils.prompt_initialization import prompt_hash, write_prompt

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "examples/training/train_pi05_vlm_only_pdet_sweep.sh"
_spec = importlib.util.spec_from_file_location("pdet_test_sampler", ROOT / "src/lerobot/datasets/sampler.py")
_sampler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sampler)
EpisodeAwareSampler = _sampler.EpisodeAwareSampler


def real_functions(class_name, names):
    """Use real PI05 math/forward code without loading optional vision libraries or 4B weights."""
    tree = ast.parse((ROOT / "src/lerobot/policies/pi05/modeling_pi05.py").read_text())
    nodes = (
        tree.body
        if class_name is None
        else next(n for n in tree.body if getattr(n, "name", None) == class_name).body
    )
    selected = [copy.deepcopy(n) for n in nodes if getattr(n, "name", None) in names]
    for node in selected:
        node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return compile(module, "real_pi05_methods", "exec")


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_vlm_prompt_tokens=2, pretrained_path="/base", max_action_dim=32)
        self.model = torch.nn.Module()
        self.model.vlm_prompt_tokens = torch.nn.Embedding(2, 2048)
        self.model.config = SimpleNamespace(
            time_sampling_beta_alpha=1.5,
            time_sampling_beta_beta=1.0,
            time_sampling_scale=0.999,
            time_sampling_offset=0.001,
        )
        namespace = {"torch": torch}
        exec(real_functions(None, {"sample_beta"}), namespace)
        exec(real_functions("PI05Pytorch", {"sample_time"}), namespace)
        self.model.sample_time = namespace["sample_time"].__get__(self.model)

    def prepare_action(self, batch):
        return torch.nn.functional.pad(batch["action"], (0, 32 - batch["action"].shape[-1]))


@pytest.fixture
def policy():
    return TinyPolicy()


@pytest.fixture
def batch():
    return {"action": torch.ones(8, 50, 7), "index": torch.arange(8)}


def test_factors_rank_step_and_global_rng_are_independent(policy, batch):
    config = PDETConfig(enabled=True, prompt_seed=1, data_seed=1, noise_seed=1, time_seed=1)
    python_state, numpy_state, cpu_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    baseline = sample_flow(policy, batch, config, 7, 0)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert baseline["noise"].shape == (8, 50, 32)
    assert baseline["time"].shape == (8,)
    assert baseline["noise"].dtype == baseline["time"].dtype == torch.float32
    assert ((baseline["time"] > 0) & (baseline["time"] < 1)).all()
    for factor in ("prompt_seed", "data_seed", "noise_seed", "time_seed"):
        intervention = sample_flow(policy, batch, replace(config, **{factor: 0}), 7, 0)
        assert torch.equal(intervention["noise"], baseline["noise"]) == (factor != "noise_seed")
        assert torch.equal(intervention["time"], baseline["time"]) == (factor != "time_seed")
    for step, rank in ((8, 0), (7, 1)):
        other = sample_flow(policy, batch, config, step, rank)
        assert all(not torch.equal(other[key], baseline[key]) for key in baseline)
    # A fresh call at a resumed step is identical despite arbitrary intervening RNG use/probes.
    torch.randn(200)
    again = sample_flow(policy, batch, config, 7, 0)
    assert all(torch.equal(again[key], baseline[key]) for key in baseline)


def test_data_order_depends_only_on_d():
    def indices(config):
        return list(EpisodeAwareSampler([0], [101], shuffle=True, seed=config.data_seed))

    cfg = PDETConfig(data_seed=1)
    baseline = indices(cfg)
    for factor in ("prompt_seed", "noise_seed", "time_seed"):
        torch.randn(100)
        assert indices(replace(cfg, **{factor: 1})) == baseline
    assert indices(replace(cfg, data_seed=0)) != baseline


def make_config(path, factors):
    return SimpleNamespace(
        pdet=factors,
        output_dir=path,
        seed=0,
        batch_size=8,
        steps=3000,
        dataset=SimpleNamespace(root="/dataset"),
        resume=False,
        to_dict=lambda: {"seed": 0, "steps": 3000},
    )


def accelerator(rank=0, size=2):
    return SimpleNamespace(process_index=rank, num_processes=size, device=torch.device("cpu"))


def test_audit_detects_unintended_coupling_and_resume_changes(tmp_path, policy, batch):
    runs = []
    for name, factor in (("one", 1), ("zero", 0)):
        cfg = make_config(tmp_path / name, PDETConfig(enabled=True, noise_seed=factor, audit_steps=2))
        runs.append(cfg.output_dir)
        for rank in (0, 1):
            streams = PDETStreams(cfg, policy, accelerator(rank))
            for step in range(2):
                streams.draw(policy, batch, step, batch["index"])
        cfg.resume = True
        PDETStreams(cfg, policy, accelerator())
        with pytest.raises(ValueError, match="resume must preserve"):
            PDETStreams(cfg, policy, accelerator(size=1))
    assert audit_runs(runs)["audit"] == "passed"
    path = runs[1] / "pdet/inputs-rank0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["time_sha256"] = "corrupt"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="time_seed"):
        audit_runs(runs)


def test_explicit_draws_reach_real_policy_forward_and_flow_loss(policy, batch):
    # Execute the actual public forward, observation-flow preparation and loss reduction.
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    tree = ast.parse((ROOT / "src/lerobot/policies/pi05/configuration_pi05.py").read_text())
    enums = {}
    exec(
        compile(
            ast.Module(
                body=[
                    n
                    for n in tree.body
                    if isinstance(n, ast.ClassDef)
                    and n.name in {"PI05TrainingObjective", "PI05TrainingStage"}
                ],
                type_ignores=[],
            ),
            "real_pi05_enums",
            "exec",
        ),
        {"StrEnum": StrEnum},
        enums,
    )
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        **enums,
        "ACTION": ACTION,
        "OBS_LANGUAGE_TOKENS": OBS_LANGUAGE_TOKENS,
        "OBS_LANGUAGE_ATTENTION_MASK": OBS_LANGUAGE_ATTENTION_MASK,
        "_PI05_LOSS_VALID_STEPS_KEY": "valid_steps",
    }
    exec(real_functions(None, {"_valid_action_steps", "_reduce_action_losses"}), namespace)
    exec(real_functions("PI05Policy", {"forward", "_forward_observation_flow"}), namespace)
    for name in ("forward", "_forward_observation_flow"):
        setattr(policy, name, namespace[name].__get__(policy))
    exec(real_functions("PI05Pytorch", {"forward"}), namespace)
    policy.model.forward = namespace["forward"].__get__(policy.model)
    policy.config.training_stage = "flow"
    policy.config.output_features = {ACTION: SimpleNamespace(shape=(7,))}
    policy._preprocess_images = lambda batch: ([], [])
    # The captured prediction input must be x_t = t*epsilon + (1-t)*action.
    seen = {}

    def velocity(images, masks, tokens, text_masks, x_t, time):
        seen.update(x_t=x_t, time=time)
        return x_t * policy.model.vlm_prompt_tokens.weight.mean()

    policy.model.predict_velocity = velocity
    batch = {**batch, OBS_LANGUAGE_TOKENS: torch.ones(8, 2), OBS_LANGUAGE_ATTENTION_MASK: torch.ones(8, 2)}
    draws = sample_flow(policy, batch, PDETConfig(), 0, 0)
    loss, _ = policy(batch, **draws)
    t = draws["time"][:, None, None]
    assert torch.equal(seen["x_t"], t * draws["noise"] + (1 - t) * policy.prepare_action(batch))
    assert seen["time"] is draws["time"]
    loss.backward()
    assert policy.model.vlm_prompt_tokens.weight.grad.abs().sum() > 0


def test_config_roundtrip():
    cfg = draccus.parse(PDETConfig, args=["--enabled=true", "--data_seed=1", "--noise_seed=0"])
    assert draccus.decode(PDETConfig, draccus.encode(cfg)) == cfg
    with pytest.raises(ValueError):
        replace(cfg, time_seed=-1).validate()


def test_training_config_validates_prompt_label_and_disallows_augmentation(tmp_path):
    from lerobot.configs.train import TrainPipelineConfig

    path = tmp_path / "p1.safetensors"
    prompt = np.ones((2, 2048), dtype=np.float32)
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
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=libero",
            "--seed=0",
            "--env_eval_freq=0",
            "--pdet.enabled=true",
            "--pdet.prompt_seed=1",
            f"--vlm_prompt_init_path={path}",
        ],
    )
    cfg = draccus.decode(TrainPipelineConfig, cfg.to_dict())
    cfg.policy = SimpleNamespace(
        type="pi05",
        vlm_prompt_only=True,
        num_prompt_tokens=0,
        num_vlm_prompt_tokens=2,
        compile_model=False,
        use_peft=False,
        training_stage="flow",
        next_action_pretrain_steps=0,
        next_action_bridge_steps=0,
    )
    cfg.pdet.validate_training(cfg)
    cfg.pdet.prompt_seed = 0
    with pytest.raises(ValueError, match="P label"):
        cfg.pdet.validate_training(cfg)
    cfg.pdet.prompt_seed = 1
    cfg.dataset.image_transforms.enable = True
    with pytest.raises(ValueError, match="without augmentation"):
        cfg.pdet.validate_training(cfg)


def test_sweep_uses_two_gpus_seven_labeled_cases_and_protects_existing_outputs(tmp_path):
    from tests.scripts.test_pi05_vlm_only import VLMOnlyTest

    fixture = VLMOnlyTest()
    fixture.setUp()
    try:
        directory = tmp_path / "inits"
        for seed in (0, 1):
            prompt = np.full((2, 2048), seed, dtype=np.float32)
            write_prompt(
                directory / f"pi05-base-vlm2-init-seed{seed}.safetensors",
                prompt,
                {
                    "schema_version": 1,
                    "source_step": 0,
                    "source_training_seed": seed,
                    "num_vlm_prompt_tokens": 2,
                    "prompt_sha256": prompt_hash(prompt),
                },
            )
        assert len(set(check_initializations(directory).values())) == 2

        def run_script():
            return subprocess.run(
                ["bash", str(SWEEP)],
                env={**fixture.env, "PROMPT_INIT_DIR": str(directory)},
                capture_output=True,
                text=True,
                timeout=180,
            )

        result = run_script()
        assert result.returncode == 0, result.stderr
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]
        assert len(commands) == 7
        for digits, command in zip(
            ("1111", "0000", "0111", "1011", "1101", "1110", "1000"), commands, strict=True
        ):
            p, d, e, t = digits
            for arg in (
                "CUDA_VISIBLE_DEVICES=0,1",
                "--num_processes=2",
                "--batch_size=32",
                "--seed=0",
                "--steps=3000",
                "--save_freq=3000",
                "--save_steps=[]",
                "--policy.ntk_save_stage_snapshots=false",
                "--policy.compile_model=false",
                "--policy.num_vlm_prompt_tokens=2",
                "--pdet.enabled=true",
                f"--pdet.prompt_seed={p}",
                f"--pdet.data_seed={d}",
                f"--pdet.noise_seed={e}",
                f"--pdet.time_seed={t}",
            ):
                assert arg in command
            assert any(f"pi05-pdet-base-vlm2-P{p}-D{d}-E{e}-T{t}" in arg for arg in command)
        (Path(fixture.env["OUTPUT_ROOT"]) / "pi05-pdet-base-vlm2-P1-D1-E1-T1").mkdir(parents=True)
        result = run_script()
        assert result.returncode != 0 and "Output already exists" in result.stderr
        assert "Starting P" not in result.stdout
    finally:
        fixture.doCleanups()
