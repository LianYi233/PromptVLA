"""Task7 1000-update matched action-time Q/K/V vs input ablation.

Lightweight CPU tests of stateless temporal projection hooks; training-plan
tests rely only on the small synthetic task7 fixture and saved NumPy plans.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.utils.checkpoint
from torch import nn

from lerobot.policies.pi05.action_time import (
    RelativeActionTimeEncoding,
    make_temporal_projection_hook,
)
from lerobot.scripts import pi05_gripper_event as original
from lerobot.scripts import pi05_task7_timepath_experiment as timepath
from tests.scripts.test_pi05_gripper_task7 import dataset as sample_task7_dataset  # noqa: F401


def test_zero_init_preserves_exact_projection_and_training_gradient():
    torch.manual_seed(4321)
    encoder = RelativeActionTimeEncoding(16, 32, 10)
    linear = nn.Linear(16, 24)
    x = torch.randn(3, 50, 16, requires_grad=True)
    clean = linear(x).detach()
    handle = linear.register_forward_hook(
        make_temporal_projection_hook(encoder, horizon=50)
    )
    torch.testing.assert_close(linear(x), clean, rtol=0, atol=0)
    torch.testing.assert_close(encoder(x), x, rtol=0, atol=0)
    out = torch.utils.checkpoint.checkpoint(
        lambda y: linear(y).square().mean(), x, use_reentrant=False
    )
    out.backward()
    assert encoder.weight.grad is not None
    assert encoder.weight.grad.abs().sum() > 0
    handle.remove()


def test_temporal_path_qk_and_v_only_do_not_change_other_projections():
    torch.manual_seed(100)
    q = nn.Linear(16, 32)
    k = nn.Linear(16, 8)
    v = nn.Linear(16, 8)
    encoder = RelativeActionTimeEncoding(16, 32, 10)
    with torch.no_grad():
        encoder.weight.fill_(0.07)
    x = torch.randn(2, 50, 16)

    clean_q, clean_k, clean_v = q(x), k(x), v(x)
    hq = q.register_forward_hook(make_temporal_projection_hook(encoder, horizon=50))
    hk = k.register_forward_hook(make_temporal_projection_hook(encoder, horizon=50))
    assert not torch.equal(q(x), clean_q)
    assert not torch.equal(k(x), clean_k)
    torch.testing.assert_close(v(x), clean_v, rtol=0, atol=0)
    hq.remove()
    hk.remove()

    hv = v.register_forward_hook(make_temporal_projection_hook(encoder, horizon=50))
    torch.testing.assert_close(q(x), clean_q, rtol=0, atol=0)
    torch.testing.assert_close(k(x), clean_k, rtol=0, atol=0)
    assert not torch.equal(v(x), clean_v)
    hv.remove()


def test_hook_fails_loudly_on_wrong_token_horizon():
    encoder = RelativeActionTimeEncoding(16, 32, 10)
    layer = nn.Linear(16, 8)
    layer.register_forward_hook(make_temporal_projection_hook(encoder, horizon=50))
    with pytest.raises(RuntimeError, match="first-layer action tokens"):
        layer(torch.randn(1, 40, 16))


def test_temporal_modules_share_exact_32768_parameter_budget():
    for case in ("time_qk", "time_v", "time_input"):
        enabled, mode = timepath.MODES[case]
        assert enabled and mode in ("qk", "v", "input")
        encoder = RelativeActionTimeEncoding(1024, 32, 10)
        assert sum(p.numel() for p in encoder.parameters()) == 32768
        assert torch.count_nonzero(encoder.weight) == 0
    assert timepath.MODES["rope"] == (False, "input")


@pytest.fixture
def prepared(sample_task7_dataset, tmp_path):
    path = tmp_path / "prepared-base"
    record = original.prepare(
        sample_task7_dataset, path, task_id=7, episode_index=1,
        steps=1000, batch_size=32, world_size=2,
    )
    return path, record


def test_four_cases_use_identical_1000_step_one_episode_uniform_plan(prepared, tmp_path):
    from lerobot.utils.pdet_order import load_plan

    directory, record = prepared
    assert timepath.check_plan(directory) == record
    jobs = timepath.build_jobs(record, directory, tmp_path / "models")
    assert [j["case"] for j in jobs] == list(timepath.CASES)
    assert len({j["plan"] for j in jobs}) == 1
    assert len({j["run"] for j in jobs}) == 4
    assert all(j["checkpoint"].parts[-2:] == ("001000", "pretrained_model") for j in jobs)
    assert all("task7-ep1-steps1000" in j["name"] for j in jobs)
    plan = load_plan(jobs[0]["plan"])
    assert len(plan["index"]) == 1000 * 32 * 2
    assert np.array_equal(np.unique(plan["episode_index"]), np.array([1]))


def test_train_command_preserves_source_budget_noise_and_injection_modes(prepared, tmp_path):
    directory, record = prepared
    args = SimpleNamespace(
        source=tmp_path / "source",
        tokenizer=tmp_path / "tokenizer",
        dataset_repo_id="libero",
        output_root=tmp_path / "models", log_root=tmp_path / "logs",
        gpus="0,1", port=29589, dry_run=True, num_workers=0,
        wandb_mode="offline",
    )
    jobs = timepath.build_jobs(record, directory, args.output_root)
    for job in jobs:
        cmd, env = timepath.training_command(job, record, args, tmp_path / "prompt.safetensors")
        assert f"--policy.action_time_injection={job['time_injection']}" in cmd
        assert f"--policy.action_time_encoding={str(job['time_encoding']).lower()}" in cmd
        assert f"--pdet.order_plan={directory.resolve() / 'uniform.npz'}" in cmd
        assert "--pdet.audit_steps=1000" in cmd
        assert "--pdet.prompt_seed=1" in cmd
        assert "--pdet.noise_seed=1" in cmd
        assert "--pdet.time_seed=1" in cmd
        assert env["GPU_IDS"] == "0,1"
        assert env["FLOW_STEPS"] == "1000"
        assert env["BATCH_SIZE"] == "32"
        assert env["NUM_PROCESSES"] == "2"
        assert env["TRAIN_ACTION_EXPERT"] == "true"
        assert env["VLM_PROMPT_TOKENS"] == "2"


def _args(tmp_path, seed=3000, magnitude=0.12):
    return SimpleNamespace(
        eval_root=tmp_path / "eval", output_root=tmp_path / "models",
        log_root=tmp_path / "logs", directory=tmp_path / "plan",
        repeat=1, eval_seed=seed, episodes=10, shift_m=magnitude,
        shift_seed=20261103,
    )


def test_result_roots_preserve_seed_and_displacement(prepared, tmp_path):
    directory, record = prepared
    args = _args(tmp_path)
    a = timepath.evaluation_root(args, record)
    assert "task7-ep1" in str(a)
    assert "seed3000" in str(a)
    assert "meters-0.120" in str(a)
    args.eval_seed = 4000
    b = timepath.evaluation_root(args, record)
    assert a != b
    args.shift_m = 0.1
    assert timepath.evaluation_root(args, record) != b
    args = _args(tmp_path)
    jobs = timepath.build_jobs(record, directory, args.output_root)
    assert len({timepath.result_path(job, record, args) for job in jobs}) == 4


def test_summary_refuses_mismatched_physics_or_unexpected_suite(prepared, tmp_path):
    from lerobot.scripts.pi05_gripper_event_experiment import file_hash, save_record

    directory, record = prepared
    args = _args(tmp_path)
    jobs = timepath.build_jobs(record, directory, args.output_root)
    cfg = timepath.shift_config(args)
    cfg_file = timepath.evaluation_config_path(args, record)
    save_record(cfg_file, cfg)
    for i, job in enumerate(jobs):
        target = timepath.result_path(job, record, args)
        target.parent.mkdir(parents=True, exist_ok=True)
        success = [True] * (4+i) + [False] * (6-i)
        target.write_text(json.dumps({"per_task": [{
            "task_group": "libero_10", "task_id": 7,
            "metrics": {"successes": success},
        }]}))
        (target.parent / "resume_manifest.json").write_text(json.dumps({
            "task7_object_shift": {
                "config": cfg, "config_sha256": file_hash(cfg_file),
                "helper_sha256": "same-helper-sha",
            },
        }))
    summary = timepath.summarize(record, jobs, args)
    assert summary["results"]["rope"]["selected_task_rate"] == 40
    assert summary["results"]["time_input"]["selected_task_rate"] == 70
    assert summary["contrasts"]["input_minus_qk_pp"] == 20
    path = timepath.result_path(jobs[-1], record, args)
    m = path.parent / "resume_manifest.json"
    altered = json.loads(m.read_text())
    altered["task7_object_shift"]["helper_sha256"] = "changed"
    m.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="Different object-shift"):
        timepath.summarize(record, jobs, args)


def test_two_gpu_shell_phases_and_no_image_corruption():
    root = Path(__file__).resolve().parents[2]
    parallel = (root / "run_eval_libero_task7_timepath_2gpu.sh").read_text()
    assert parallel.index('run_pair rope time_qk "Phase 1"') < parallel.index(
        'run_pair time_v time_input "Phase 2"'
    )
    assert parallel.index('run_pair time_v time_input "Phase 2"') < parallel.index("summarize all")
    single = (root / "run_eval_libero_task7_timepath.sh").read_text()
    assert "LIBERO_FIXED_NOISE_CONFIG" in single
    assert "pi05_task7_timepath_experiment eval" in single
    source = (root / "src/lerobot/policies/pi05/modeling_pi05.py").read_text()
    assert "self._temporal_projection_hooks.append" in source
    assert "apply_rotary_pos_emb" in source


def test_mode_validation_guards_qk_and_v_without_active_time():
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    for mode in ("qk", "v"):
        with pytest.raises(ValueError, match="requires action_time_encoding"):
            PI05Config(action_time_encoding=False, action_time_injection=mode)
    with pytest.raises(ValueError, match="input, qk or v"):
        PI05Config(action_time_injection="invalid")
    args = dict(
        num_vlm_prompt_tokens=2, num_prompt_tokens=0,
        train_action_expert=True,
        cabo_enabled=False, use_peft=False,
        next_action_pretrain_steps=0, next_action_bridge_steps=0,
    )
    for mode in ("input", "qk", "v"):
        cfg = PI05Config(**args, action_time_encoding=True, action_time_injection=mode)
        assert cfg.action_time_injection == mode
        assert cfg.action_time_encoding is True
