"""Initialization swaps must preserve the control run's subsequent stochastic training inputs."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from lerobot.utils.prompt_initialization import (
    KEY,
    METADATA_KEY,
    apply_initial_prompt,
    constructed_prompt_packet,
    export_constructed_prompt,
    export_initial_prompt,
    prompt_hash,
    read_initial_prompt,
    save_import_record,
    validate_export_request,
    write_prompt,
)


def packet(tmp_path, prompt=None):
    if prompt is None:
        prompt = np.random.default_rng(1).normal(0, 0.02, (2, 2048)).astype(np.float32)
    metadata = {
        "schema_version": 1,
        "source_step": 0,
        "source_training_seed": 1,
        "num_vlm_prompt_tokens": prompt.shape[0],
        "prompt_sha256": prompt_hash(prompt),
    }
    path = tmp_path / "seed1.safetensors"
    write_prompt(path, prompt, metadata)
    return path, prompt, metadata


class Policy(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.config = SimpleNamespace(vlm_prompt_only=True, num_prompt_tokens=0, num_vlm_prompt_tokens=2)
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Linear(2048, 7).to(dtype=dtype).requires_grad_(False)
        self.model.vlm_prompt_tokens = torch.nn.Embedding(2, 2048).to(dtype=dtype)


def training_draws():
    return (
        random.random(),
        np.random.rand(8),
        torch.normal(0.0, 1.0, (2, 50, 32)),
        torch.distributions.Beta(1.5, 1.0).sample((2,)),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("full_expert", [False, True])
def test_swap_preserves_rng_future_flow_draws_parameter_identity_and_frozen_weights(
    tmp_path, dtype, full_expert
):
    original = np.random.default_rng(1).normal(0, 0.02, (2, 2048)).astype(np.float32)
    prompt = torch.from_numpy(original).to(dtype=dtype).float().numpy()
    path, prompt, _ = packet(tmp_path, prompt)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    policy = Policy(dtype)
    if full_expert:
        policy.config.vlm_prompt_only = False
        policy.config.train_action_expert = True
        policy.model.paligemma_with_expert = torch.nn.Module()
        policy.model.paligemma_with_expert.gemma_expert = torch.nn.Module()
        policy.model.paligemma_with_expert.gemma_expert.model = torch.nn.Linear(4, 4)
        for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            setattr(policy.model, name, torch.nn.Linear(4, 4))
    other_weights = {n: p.clone() for n, p in policy.named_parameters() if "vlm_prompt_tokens" not in n}
    parameter = policy.model.vlm_prompt_tokens.weight
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    backbone = {name: value.clone() for name, value in policy.model.backbone.state_dict().items()}
    states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    expected = [training_draws() for _ in range(3)]
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    record = apply_initial_prompt(policy, path)
    assert all(torch.equal(other_weights[n], p) for n, p in policy.named_parameters() if n in other_weights)
    # Persistence must not advance RNG either.
    save_import_record(policy, record, tmp_path / "new_run", 0)
    assert random.getstate() == states[0]
    np.testing.assert_array_equal(np.random.get_state()[1], states[1][1])
    assert np.random.get_state()[2:] == states[1][2:]
    assert torch.equal(torch.get_rng_state(), states[2])
    for actual, reference in zip([training_draws() for _ in range(3)], expected, strict=True):
        assert actual[0] == reference[0]
        np.testing.assert_array_equal(actual[1], reference[1])
        assert torch.equal(actual[2], reference[2])
        assert torch.equal(actual[3], reference[3])
    assert policy.model.vlm_prompt_tokens.weight is parameter
    assert optimizer.param_groups[0]["params"][0] is parameter
    np.testing.assert_array_equal(parameter.detach().float().numpy(), prompt)
    for name, value in policy.model.backbone.state_dict().items():
        assert torch.equal(value, backbone[name])
    assert record["loaded_prompt_sha256"] == prompt_hash(prompt)
    assert record["replaced_prompt_sha256"] != record["loaded_prompt_sha256"]
    path.unlink()
    transferred, metadata = read_initial_prompt(tmp_path / "new_run/prompt_init.safetensors", 2)
    np.testing.assert_array_equal(transferred, prompt)
    assert metadata["training_seed"] == 0 and metadata["source_training_seed"] == 1


def make_run(tmp_path):
    run = tmp_path / "pi05-base-vlm2-seed1"
    (run / "prompt_rank").mkdir(parents=True)
    model = run / "checkpoints/003000/pretrained_model"
    model.mkdir(parents=True)
    (model / "train_config.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "policy": {
                    "type": "pi05",
                    "vlm_prompt_only": True,
                    "num_vlm_prompt_tokens": 2,
                    "num_prompt_tokens": 0,
                },
            }
        )
    )
    (run / "prompt_rank/manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "num_vlm_prompt_tokens": 2,
                "forward_precision": "bfloat16",
                "pretrained_path": "/old-server/models/pi05_libero_base",
            }
        )
    )
    prompt = np.random.default_rng(1).normal(0, 0.02, (2, 2048)).astype(np.float32)
    np.savez(run / "prompt_rank/step_000000.npz", prompt=prompt)
    np.savez(run / "prompt_rank/step_003000.npz", prompt=prompt + 1)
    return run, prompt


def test_export_uses_actual_step_zero_and_survives_transfer_without_source_files(tmp_path):
    run, initial = make_run(tmp_path)
    output = tmp_path / "transfer/init.safetensors"
    metadata = export_initial_prompt(run, output, expect_seed=1)
    imported, actual = read_initial_prompt(output, 2)
    np.testing.assert_array_equal(imported, initial)
    assert metadata == actual
    assert metadata["source_training_seed"] == 1
    assert metadata["source_step"] == 0
    with pytest.raises(FileExistsError):
        export_initial_prompt(run, output, expect_seed=1)
    with pytest.raises(ValueError, match="expected 0"):
        export_initial_prompt(run, tmp_path / "bad.safetensors", expect_seed=0)
    (run / "prompt_rank/step_000000.npz").unlink()
    with pytest.raises(FileNotFoundError):
        export_initial_prompt(run, tmp_path / "bad.safetensors", expect_seed=1)


@pytest.mark.parametrize("change", ["nan", "shape", "checksum", "step", "full_model"])
def test_rejects_corrupt_wrong_shape_and_noninitial_files(tmp_path, change):
    path, prompt, metadata = packet(tmp_path)
    arrays = {KEY: prompt}
    if change == "nan":
        prompt[0, 0] = np.nan
    elif change == "shape":
        arrays[KEY] = prompt[:1]
    elif change == "checksum":
        prompt[0, 0] += 0.01
    elif change == "step":
        metadata["source_step"] = 3000
    else:
        arrays["backbone"] = prompt
    save_file(arrays, path, metadata={METADATA_KEY: json.dumps(metadata)})
    with pytest.raises(ValueError):
        read_initial_prompt(path, 2)


def test_precision_loss_and_unfrozen_backbone_fail_before_mutation(tmp_path):
    path, _, _ = packet(tmp_path)
    policy = Policy(torch.bfloat16)
    original = policy.model.vlm_prompt_tokens.weight.detach().clone()
    with pytest.raises(ValueError, match="exactly"):
        apply_initial_prompt(policy, path)
    assert torch.equal(policy.model.vlm_prompt_tokens.weight, original)
    policy.model.backbone.requires_grad_(True)
    with pytest.raises(ValueError, match="Trainable partition mismatch"):
        apply_initial_prompt(policy, path)


def test_export_cli(tmp_path):
    import os
    import subprocess
    import sys

    run, initial = make_run(tmp_path)
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "cli.safetensors"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.export_pi05_prompt_init",
            "export",
            str(run),
            "--output",
            str(output),
            "--expect-seed",
            "1",
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(read_initial_prompt(output, 2)[0], initial)


def generation_config(tmp_path):
    return SimpleNamespace(
        seed=0,
        resume=False,
        vlm_prompt_init_path=None,
        vlm_prompt_export_path=str(tmp_path / "regenerated.safetensors"),
        vlm_prompt_export_reference=None,
        output_dir=tmp_path / "generation-run",
        to_dict=lambda: {"seed": 0, "steps": 3000},
    )


class ExportAccelerator:
    device = torch.device("cpu")
    num_processes = 2
    process_index = 0
    is_main_process = True

    def reduce(self, tensor, reduction):
        return tensor

    def gather(self, tensor):
        return torch.cat((tensor, tensor))

    def wait_for_everyone(self):
        pass


def generation_policy(dtype=torch.float32):
    policy = Policy(dtype)
    policy.config.pretrained_path = "/source/pi05_libero_base"
    policy.config.dtype = str(dtype).split(".")[-1]
    return policy


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_regeneration_exports_exact_constructed_values_without_rng_or_parameter_changes(tmp_path, dtype):
    policy = generation_policy(dtype)
    cfg = generation_config(tmp_path)
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    parameter = policy.model.vlm_prompt_tokens.weight
    states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    metadata = export_constructed_prompt(policy, cfg, ExportAccelerator())
    prompt, saved = read_initial_prompt(cfg.vlm_prompt_export_path, 2)
    np.testing.assert_array_equal(prompt, parameter.detach().float().numpy())
    assert saved == metadata
    assert metadata["source_kind"] == "regenerated_training_initialization"
    assert metadata["source_training_seed"] == 0 and metadata["source_step"] == 0
    assert metadata["historical_hash_verified"] is False
    assert policy.model.vlm_prompt_tokens.weight is parameter
    for name, value in policy.state_dict().items():
        assert torch.equal(value, before[name])
    assert random.getstate() == states[0]
    np.testing.assert_array_equal(np.random.get_state()[1], states[1][1])
    assert np.random.get_state()[2:] == states[1][2:]
    assert torch.equal(torch.get_rng_state(), states[2])
    assert not cfg.output_dir.exists()  # no full model/optimizer checkpoint
    with pytest.raises(FileExistsError):
        constructed_prompt_packet(policy, cfg)
    recipient = generation_policy(dtype)
    apply_initial_prompt(recipient, cfg.vlm_prompt_export_path)
    assert torch.equal(recipient.model.vlm_prompt_tokens.weight, parameter)


def test_regeneration_matches_preimport_hash_not_imported_seed1_hash(tmp_path):
    policy = generation_policy()
    cfg = generation_config(tmp_path)
    original = prompt_hash(policy.model.vlm_prompt_tokens.weight.detach().numpy())
    source, _, _ = packet(tmp_path)
    record = apply_initial_prompt(policy, source)
    save_import_record(policy, record, tmp_path / "cross", 0)
    cfg.vlm_prompt_export_reference = str(tmp_path / "cross/prompt_init.json")
    # Imported seed1 prompt is NOT the seed0 initialization we are trying to regenerate.
    with pytest.raises(ValueError, match="does not match"):
        constructed_prompt_packet(policy, cfg)
    assert not Path(cfg.vlm_prompt_export_path).exists()
    reference = validate_export_request(cfg.vlm_prompt_export_path, cfg.vlm_prompt_export_reference, 0, 2)
    assert reference["replaced_prompt_sha256"] == original
    with pytest.raises(ValueError, match="TRAINING seed/tokens"):
        validate_export_request(cfg.vlm_prompt_export_path, cfg.vlm_prompt_export_reference, 1, 2)
    with pytest.raises(ValueError, match="TRAINING seed/tokens"):
        validate_export_request(cfg.vlm_prompt_export_path, cfg.vlm_prompt_export_reference, 0, 8)


def test_regeneration_verifies_historical_hash_and_rejects_rank_disagreement(tmp_path):
    policy = generation_policy()
    cfg = generation_config(tmp_path)
    cfg.vlm_prompt_export_reference = str(tmp_path / "reference.json")
    digest = prompt_hash(policy.model.vlm_prompt_tokens.weight.detach().numpy())
    Path(cfg.vlm_prompt_export_reference).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_step": 0,
                "source_training_seed": 1,
                "training_seed": 0,
                "num_vlm_prompt_tokens": 2,
                "replaced_prompt_sha256": digest,
            }
        )
    )
    accelerator = ExportAccelerator()
    accelerator.gather = lambda value: torch.cat((value, value ^ 1))
    with pytest.raises(ValueError, match="differs across DDP ranks"):
        export_constructed_prompt(policy, cfg, accelerator)
    assert not Path(cfg.vlm_prompt_export_path).exists()
    record = export_constructed_prompt(policy, cfg, ExportAccelerator())
    assert record["historical_hash_verified"] is True
    assert record["reference_replaced_prompt_sha256"] == digest


def test_trainer_export_exits_before_processors_optimizer_and_rank_diagnostics(tmp_path, monkeypatch):
    """Execute the real trainer function with tiny dataset/model factories, without optional GPU deps."""
    import ast
    import logging
    import sys
    import types

    from lerobot.utils.random_utils import set_seed

    source = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/lerobot_train.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_train_single_stage"
    )
    cfg = generation_config(tmp_path)
    cfg.trainable_config = SimpleNamespace()
    cfg.wandb = SimpleNamespace(enable=True, project="should-not-be-created")
    cfg.cudnn_deterministic = False
    cfg.env_eval_freq = 0
    cfg.is_reward_model_training = False
    cfg.policy = SimpleNamespace()
    cfg.rename_map = {}
    calls = []
    expected = {}

    def dataset_factory(config):
        calls.append("dataset")
        torch.rand(17)  # model initialization must follow prior RNG consumption
        return SimpleNamespace(meta="dataset-metadata"), None

    def policy_factory(**kwargs):
        assert kwargs["ds_meta"] == "dataset-metadata"
        calls.append("model")
        policy = generation_policy()
        expected["prompt"] = policy.model.vlm_prompt_tokens.weight.detach().numpy().copy()
        return policy

    def forbidden(*args, **kwargs):
        pytest.fail("Export must exit before W&B, processors, optimizer or diagnostics")

    fake_accelerate = types.ModuleType("accelerate")
    fake_utils = types.ModuleType("accelerate.utils")
    fake_utils.DistributedType = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", fake_utils)
    namespace = {
        "TrainPipelineConfig": object,
        "torch": torch,
        "logging": logging,
        "pformat": repr,
        "colored": lambda text, *a, **kw: text,
        "init_logging": lambda **kw: None,
        "set_seed": lambda seed, **kw: set_seed(seed),
        "make_train_eval_datasets": dataset_factory,
        "make_policy": policy_factory,
        "WandBLogger": forbidden,
        "make_pre_post_processors": forbidden,
        "make_optimizer_and_scheduler": forbidden,
    }
    old_flags = (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
    )
    try:
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        namespace["_train_single_stage"](cfg, ExportAccelerator())
    finally:
        (
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32,
        ) = old_flags
    assert calls == ["dataset", "model"]
    np.testing.assert_array_equal(read_initial_prompt(cfg.vlm_prompt_export_path, 2)[0], expected["prompt"])


def test_export_config_cli_roundtrip_and_incompatible_modes(tmp_path, monkeypatch):
    import draccus

    from lerobot.configs.train import TrainPipelineConfig

    output = str(tmp_path / "init.safetensors")
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=libero",
            "--seed=0",
            "--env_eval_freq=0",
            f"--vlm_prompt_export_path={output}",
            f"--output_dir={tmp_path / 'run'}",
        ],
    )
    decoded = draccus.decode(TrainPipelineConfig, cfg.to_dict())
    assert decoded.vlm_prompt_export_path == output and decoded.seed == 0
    monkeypatch.setattr(TrainPipelineConfig, "_resolve_pretrained_from_cli", lambda self: None)
    cfg.policy = SimpleNamespace(
        type="pi05",
        vlm_prompt_only=True,
        use_peft=False,
        compile_model=False,
        training_stage="flow",
        next_action_pretrain_steps=0,
        next_action_bridge_steps=0,
        num_vlm_prompt_tokens=2,
        pretrained_path="/source/base",
        cabo_enabled=False,
        get_optimizer_preset=lambda: None,
        get_scheduler_preset=lambda: None,
    )
    cfg.validate()
    for key, value in (
        ("resume", True),
        ("vlm_prompt_init_path", "/seed1.safetensors"),
        ("seed", None),
        ("seed", -1),
        ("seed", 2**32),
        ("env_eval_freq", 1),
    ):
        before = getattr(cfg, key)
        setattr(cfg, key, value)
        with pytest.raises(ValueError, match="fresh seeded"):
            cfg.validate()
        setattr(cfg, key, before)
    cfg.vlm_prompt_export_path = None
    cfg.vlm_prompt_export_reference = "/reference.json"
    with pytest.raises(ValueError, match="requires vlm_prompt_export_path"):
        cfg.validate()
