"""Full velocity-network adaptation without accidentally unfreezing the VLM."""

import pytest
import torch

from lerobot.utils.pi05_trainable import validate_vlm_trainables
from lerobot.utils.prompt_rank import isolated_prompt
from tests.policies.pi0_pi05.test_pi05_prompt import (
    PI05Policy,
    _actions,
    _config,
    _inputs,
    _save_checkpoint,
    _tiny_real_pi05,  # noqa: F401 -- shared real-model fixture
)


def config(**kwargs):
    options = {
        "train_action_expert": True,
        "num_prompt_tokens": 0,
        "cabo_enabled": False,
        "next_action_pretrain_steps": 0,
        "next_action_bridge_steps": 0,
    }
    options.update(kwargs)
    return _config(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"vlm_prompt_only": True},
        {"train_action_projections": True},
        {"cabo_enabled": True},
        {"use_peft": True},
        {"num_prompt_tokens": 1},
        {"num_vlm_prompt_tokens": 0},
        {"training_stage": "next_action"},
        {"next_action_pretrain_steps": 1},
        {"next_action_bridge_steps": 1},
    ],
)
def test_reject_incompatible_expert_training(options):
    with pytest.raises(ValueError, match="train_action_expert"):
        config(**options)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_real_backward_updates_every_action_group_and_prompt_only(dtype, checkpointing, tmp_path):
    cfg = config(dtype=dtype, gradient_checkpointing=checkpointing)
    policy = PI05Policy(cfg)
    policy.requires_grad_(True)  # The partition must also repair accidental global unfreezing.
    policy.eval().train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=0.01, weight_decay=0)
    prefixes = (
        "model.vlm_prompt_tokens.",
        "model.paligemma_with_expert.gemma_expert.model.",
        "model.action_in_proj.",
        "model.action_out_proj.",
        "model.time_mlp_in.",
        "model.time_mlp_out.",
    )
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    selected = {n for n, p in policy.named_parameters() if p.requires_grad}
    assert selected == {n for n in before if n.startswith(prefixes)}
    assert {id(p) for p in policy.get_optim_params()} == {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    assert policy.model.paligemma_with_expert.gemma_expert.model.training
    assert not policy.model.paligemma_with_expert.paligemma.training
    # Fresh tiny AdaRMS starts with zero conditioning weights: the first update
    # opens the time-MLP gradient path, and the second updates those MLPs too.
    for _ in range(2):
        optimizer.zero_grad()
        loss = policy.model(**_inputs(), actions=_actions()).mean()
        loss.backward()
        for name, parameter in policy.named_parameters():
            if name in selected:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            else:
                assert parameter.grad is None, name
        optimizer.step()
    changed = {n for n, p in policy.named_parameters() if not torch.equal(before[n], p)}
    assert changed <= selected
    assert all(any(name.startswith(prefix) for name in changed) for prefix in prefixes)
    validate_vlm_trainables(policy)
    checkpoint = tmp_path / "checkpoint"
    _save_checkpoint(checkpoint, policy)
    cfg.save_pretrained(checkpoint)
    reloaded = PI05Policy.from_pretrained(checkpoint, strict=True)
    assert reloaded.config.train_action_expert and not reloaded.config.vlm_prompt_only
    validate_vlm_trainables(reloaded)
    for name, p in reloaded.named_parameters():
        torch.testing.assert_close(p, dict(policy.named_parameters())[name], atol=0, rtol=0)


def test_probe_exception_restores_all_expert_flags_gradients_and_parameter_objects():
    policy = PI05Policy(config()).train()
    before = list(policy.parameters())
    for p in before:
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    gradients = [p.grad for p in before]
    flags = [p.requires_grad for p in before]
    modes = [m.training for m in policy.modules()]
    with pytest.raises(RuntimeError, match="probe error"), isolated_prompt(policy) as probe:
        assert [id(p) for p in policy.parameters() if p.requires_grad] == [id(probe)]
        raise RuntimeError("probe error")
    assert all(a is b for a, b in zip(before, policy.parameters(), strict=True))
    assert [p.requires_grad for p in policy.parameters()] == flags
    assert [m.training for m in policy.modules()] == modes
    assert all(p.grad is g for p, g in zip(before, gradients, strict=True))
    policy.model.paligemma_with_expert.paligemma.requires_grad_(True)
    with pytest.raises(ValueError, match="partition mismatch"), isolated_prompt(policy):
        pass


def test_training_pipeline_accepts_imported_p_and_rank_probes_for_full_expert(tmp_path, monkeypatch):
    import draccus
    import numpy as np

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.prompt_initialization import prompt_hash, write_prompt

    prompt = np.zeros((2, 2048), dtype=np.float32)
    path = tmp_path / "P1.safetensors"
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
            "--prompt_rank.freq=250",
            f"--output_dir={tmp_path / 'new'}",
        ],
    )
    cfg.policy = config(num_vlm_prompt_tokens=2, compile_model=False, push_to_hub=False)
    monkeypatch.setattr("sys.argv", ["lerobot-train"])
    cfg.validate()
    assert cfg.policy.train_action_expert
