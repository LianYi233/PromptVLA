"""Relative action-time conditioning: initialization, gradient path and reloads."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from lerobot.policies.pi05.action_time import RelativeActionTimeEncoding
from lerobot.utils.pi05_trainable import validate_vlm_trainables
from tests.policies.pi0_pi05.test_pi05_prompt import (
    PI05Policy,
    _actions,
    _config,
    _inputs,
    _save_checkpoint,
    _tiny_real_pi05,  # noqa: F401 -- run production layers with tiny shapes
)
from tests.scripts.test_pi05_gripper_event_experiment import (
    prepared as prepared_experiment,  # noqa: F401 -- synthetic dataset/source/prompt fixture
)

TIME_WEIGHT = "model.action_time_encoder.weight"


def config(**kwargs):
    options = {
        "train_action_expert": True,
        "num_prompt_tokens": 0,
        "cabo_enabled": False,
        "next_action_pretrain_steps": 0,
        "next_action_bridge_steps": 0,
        "action_time_encoding": True,
        "action_time_encoding_dim": 8,
        "action_time_encoding_fps": 20.0,
    }
    options.update(kwargs)
    return _config(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"train_action_expert": False},
        {"action_time_encoding": 1},
        {"action_time_encoding_dim": 0},
        {"action_time_encoding_dim": 3},
        {"action_time_encoding_dim": True},
        {"action_time_encoding_dim": 4.5},
        {"action_time_encoding_fps": 0},
        {"action_time_encoding_fps": -1},
        {"action_time_encoding_fps": float("nan")},
        {"action_time_encoding_fps": float("inf")},
        {"action_time_encoding_fps": True},
    ],
)
def test_invalid_time_encoding_config_rejected(options):
    with pytest.raises(ValueError, match="action_time_encoding"):
        config(**options)


def test_time_features_use_physical_offsets_and_do_not_consume_rng():
    rng_before = torch.random.get_rng_state().clone()
    module = RelativeActionTimeEncoding(16, encoding_dim=8, fps=20)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    features = module.features(5, torch.device("cpu"))
    slow = RelativeActionTimeEncoding(16, encoding_dim=8, fps=10)
    torch.testing.assert_close(features[::2], slow.features(3, torch.device("cpu")))
    assert not torch.equal(features[0], features[1])
    assert torch.isfinite(features).all()
    inputs = torch.randn(2, 5, 16)
    torch.testing.assert_close(module(inputs), inputs, atol=0, rtol=0)
    module(inputs).sum().backward()
    assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_optional_module_preserves_initial_model_rng_and_predictions(dtype):
    torch.manual_seed(101)
    ordinary = PI05Policy(config(action_time_encoding=False, dtype=dtype))
    next_random_off = torch.randn(8)
    torch.manual_seed(101)
    timed = PI05Policy(config(dtype=dtype))
    next_random_on = torch.randn(8)
    assert torch.equal(next_random_off, next_random_on)
    assert TIME_WEIGHT not in ordinary.state_dict()
    assert set(timed.state_dict()) - set(ordinary.state_dict()) == {TIME_WEIGHT}
    for key, value in ordinary.state_dict().items():
        torch.testing.assert_close(value, timed.state_dict()[key], atol=0, rtol=0, msg=key)
    inputs, actions = _inputs(), _actions()
    noise = torch.randn_like(actions)
    time = torch.tensor([0.2, 0.7])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == "bfloat16"):
        loss_off = ordinary.model(**inputs, actions=actions, noise=noise, time=time)
        loss_on = timed.model(**inputs, actions=actions, noise=noise, time=time)
    torch.testing.assert_close(loss_on, loss_off, atol=0, rtol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_time_projection_trains_and_survives_full_checkpoint(tmp_path, checkpointing):
    cfg = config(gradient_checkpointing=checkpointing)
    policy = PI05Policy(cfg).train()
    validate_vlm_trainables(policy)
    parameters = policy.get_optim_params()
    projection = policy.model.action_time_encoder.weight
    assert any(parameter is projection for parameter in parameters)
    assert projection in list(policy._action_projection_modules()[-1].parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0)
    inputs, actions = _inputs(), _actions()
    original_actions = actions.clone()
    noise, time = torch.randn_like(actions), torch.tensor([0.25, 0.75])
    loss = policy.model(**inputs, actions=actions, noise=noise, time=time).mean()
    loss.backward()
    assert projection.grad is not None and torch.isfinite(projection.grad).all()
    assert projection.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.model.paligemma_with_expert.paligemma.parameters())
    optimizer.step()
    assert projection.detach().abs().sum() > 0
    torch.testing.assert_close(actions, original_actions, atol=0, rtol=0)

    checkpoint = tmp_path / "trained"
    _save_checkpoint(checkpoint, policy)
    cfg.save_pretrained(checkpoint)
    loaded = PI05Policy.from_pretrained(checkpoint, strict=True).eval()
    validate_vlm_trainables(loaded)
    assert loaded.config.action_time_encoding
    assert loaded.config.action_time_encoding_fps == 20
    torch.testing.assert_close(loaded.state_dict()[TIME_WEIGHT], projection, atol=0, rtol=0)
    # Production sampling uses the same embed_suffix path and needs no training
    # labels, absolute episode position, or ground-truth closing-event metadata.
    policy.eval()
    predicted = policy.model.sample_actions(**inputs, noise=noise)
    restored = loaded.model.sample_actions(**inputs, noise=noise)
    assert predicted.shape == actions.shape
    assert torch.isfinite(predicted).all()
    torch.testing.assert_close(predicted, restored, atol=0, rtol=0)


def test_base_checkpoint_can_initialize_zero_time_projection(tmp_path):
    source = PI05Policy(config(action_time_encoding=False))
    checkpoint = tmp_path / "base"
    _save_checkpoint(checkpoint, source)
    source.config.save_pretrained(checkpoint)
    loaded = PI05Policy.from_pretrained(checkpoint, config=config(), strict=True)
    assert loaded.model.action_time_encoder.weight.detach().count_nonzero() == 0
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize("strict", [False, True])
def test_enabled_checkpoint_missing_time_weight_is_rejected(tmp_path, strict):
    source = PI05Policy(config())
    checkpoint = tmp_path / "corrupt"
    _save_checkpoint(checkpoint, source)
    source.config.save_pretrained(checkpoint)
    state = load_file(checkpoint / "model.safetensors")
    state.pop(TIME_WEIGHT)
    save_file(state, checkpoint / "model.safetensors")
    with pytest.raises(RuntimeError, match="action_time_encoding checkpoint is missing"):
        PI05Policy.from_pretrained(checkpoint, strict=strict)


def test_actual_experiment_cli_parses_and_validates_all_four_conditions(
    request,
    tmp_path,
    monkeypatch,
):
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import pi05_gripper_event as sampling, pi05_gripper_event_experiment as experiment

    fixture, directory, prompt = request.getfixturevalue("prepared_experiment")
    record = sampling.check(directory)
    args = argparse.Namespace(
        source=fixture.source("base"),
        tokenizer=fixture.tokenizer,
        dataset_repo_id="libero",
        output_root=tmp_path / "outputs",
        log_root=tmp_path / "logs",
        gpus="0,1",
        port=29589,
        dry_run=True,
        num_workers=0,
        wandb_mode="disabled",
    )

    @parser.wrap()
    def validated_config(cfg: TrainPipelineConfig):
        cfg.validate()
        return cfg

    jobs = experiment.build_jobs(record, directory, args.output_root)
    for job in jobs:
        command, env = experiment.training_command(job, record, args, prompt)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        launches = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]
        assert len(launches) == 1
        launch = launches[0]
        native_args = launch[launch.index("lerobot-train") + 1 :]
        monkeypatch.setattr(sys, "argv", ["lerobot-train", *native_args])
        # Exercise the real CLI .path extraction, dataclass parsing, pretrained
        # config overrides, imported prompt checks, and PDET plan validation.
        cfg = validated_config()
        assert cfg.policy.train_action_expert and not cfg.policy.vlm_prompt_only
        assert cfg.policy.action_time_encoding is job["time_encoding"]
        assert cfg.policy.action_time_encoding_dim == 32
        assert cfg.policy.action_time_encoding_fps == record["fps"]
        assert cfg.policy.num_vlm_prompt_tokens == 2 and cfg.policy.num_prompt_tokens == 0
        assert not cfg.policy.compile_model and cfg.policy.gradient_checkpointing
        assert cfg.pdet.enabled and cfg.pdet.prompt_seed == 1
        assert cfg.pdet.order_plan == str(job["plan"])
        assert cfg.pdet.flow_binding == "occurrence"
        assert cfg.vlm_prompt_init_path == str(prompt.resolve())
        assert cfg.dataset.episodes is None
        assert cfg.steps == record["settings"]["steps"]
        assert cfg.batch_size == record["settings"]["batch_size"]
        assert cfg.keyframe_loss.mode == "off"
        assert cfg.optimizer is not None and cfg.scheduler is not None
