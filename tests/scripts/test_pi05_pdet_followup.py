"""Causal-isolation checks for task slots, occurrence-bound flow and diagnostics."""

import json
import shlex
import subprocess
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from accelerate.data_loader import BatchSamplerShard

from lerobot.configs.pdet import PDETConfig
from lerobot.configs.prompt_rank import PromptRankConfig
from lerobot.scripts.pdet_followup import audit_order, compare_inputs, summarize
from lerobot.utils.pdet import PDETStreams, PromptUpdateMonitor, sample_flow
from lerobot.utils.pdet_order import (
    PlannedDataset,
    PlannedSampler,
    create_plan,
    load_plan,
    read_frame_metadata,
    save_plan,
    validate_plan_dataset,
)
from lerobot.utils.prompt_initialization import prompt_hash, write_prompt
from lerobot.utils.prompt_rank import PromptRankMonitor
from tests.scripts.test_pi05_pdet import TinyPolicy, accelerator, make_config

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/training/train_pi05_pdet_followup.sh"


def make_dataset(root):
    path = Path(root) / "data/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 24
    pq.write_table(
        pa.table(
            {
                "index": list(range(count)),
                "episode_index": [i // 6 for i in range(count)],
                "frame_index": [i % 6 for i in range(count)],
                "task_index": [i // 12 for i in range(count)],
                "action": [[0.0] * 6 + [float(i % 6 >= 3)] for i in range(count)],
            }
        ),
        path,
    )
    return read_frame_metadata(root)


def plan_for(metadata, seed=0, steps=5):
    return create_plan(metadata, steps=steps, batch_size=2, world_size=2, data_seed=0, within_seed=seed)


def shard_batches(plan, rank, start_step=0):
    sampler = torch.utils.data.BatchSampler(PlannedSampler(plan, start_step), batch_size=2, drop_last=False)
    return list(BatchSamplerShard(sampler, num_processes=2, process_index=rank))


def test_fixed_pool_task_slots_and_resume_through_actual_accelerate_sharder(tmp_path):
    metadata = make_dataset(tmp_path)
    plans = [plan_for(metadata, seed, steps=17) for seed in (0, 1)]
    assert plans[0]["manifest"]["instance_pool_sha256"] == plans[1]["manifest"]["instance_pool_sha256"]
    assert not np.array_equal(plans[0]["order"], plans[1]["order"])
    for rank in (0, 1):
        a, b = [shard_batches(plan, rank) for plan in plans]
        assert [[int(metadata["task_index"][idx]) for idx, _ in batch] for batch in a] == [
            [int(metadata["task_index"][idx]) for idx, _ in batch] for batch in b
        ]
        for plan in plans:
            assert shard_batches(plan, rank, start_step=3) == shard_batches(plan, rank)[3:]
    path = tmp_path / "plan.npz"
    save_plan(path, plans[0])
    save_plan(path, plans[0])
    assert load_plan(path)["manifest"] == plans[0]["manifest"]
    with pytest.raises(ValueError, match="Different order plan"):
        save_plan(path, plans[1])
    with pytest.raises(ValueError, match="GPU count"):
        validate_plan_dataset(plans[0], metadata, steps=17, batch_size=4, world_size=1)
    changed = {key: value.copy() for key, value in metadata.items()}
    changed["task_index"][0] += 1
    with pytest.raises(ValueError, match="metadata differ"):
        validate_plan_dataset(plans[0], changed, steps=17, batch_size=2, world_size=2)


def test_occurrence_flow_follows_samples_not_rank_step_or_batch_position():
    policy = TinyPolicy()
    cfg = PDETConfig(enabled=True, order_plan="plan", noise_seed=1, time_seed=1)
    batch = {"action": torch.zeros(3, 50, 7)}
    state = torch.get_rng_state().clone()
    a = sample_flow(policy, batch, cfg, 0, 0, [4, 2, 17])
    b = sample_flow(policy, batch, cfg, 1024, 1, [17, 4, 2])
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(b["noise"], a["noise"][[2, 0, 1]])
    assert torch.equal(b["time"], a["time"][[2, 0, 1]])
    for key, factor in (("noise", "noise_seed"), ("time", "time_seed")):
        other = sample_flow(policy, batch, replace(cfg, **{factor: 0}), 0, 0, [4, 2, 17])
        for name in a:
            assert torch.equal(a[name], other[name]) == (name != key)
    with pytest.raises(ValueError, match="occurrence ID"):
        sample_flow(policy, batch, cfg, 0, 0)


def test_full_realized_trace_audit_detects_late_noise_corruption(tmp_path):
    root = tmp_path / "dataset"
    metadata = make_dataset(root)
    policy = TinyPolicy()
    runs = []
    for w in (0, 1):
        plan = plan_for(metadata, w, steps=50)
        plan_path = tmp_path / f"plan{w}.npz"
        save_plan(plan_path, plan)
        cfg = make_config(
            tmp_path / f"run{w}",
            PDETConfig(
                enabled=True,
                prompt_seed=1,
                data_seed=0,
                noise_seed=1,
                time_seed=1,
                order_plan=str(plan_path),
                audit_steps=50,
                diagnostics=True,
            ),
        )
        cfg.steps, cfg.batch_size, cfg.dataset.root = 50, 2, root
        runs.append(cfg.output_dir)
        for rank in (0, 1):
            streams = PDETStreams(cfg, policy, accelerator(rank))
            for step, items in enumerate(shard_batches(plan, rank)):
                ids, occurrences = zip(*items, strict=True)
                streams.draw(policy, {"action": torch.zeros(2, 50, 7)}, step, ids, occurrences)
        cfg.resume = True
        PDETStreams(cfg, policy, accelerator())
    assert audit_order(runs)["audit"] == "passed"
    assert Path(summarize(runs[0])["output"], "coverage.csv").is_file()
    # A startup-only patch still changes provenance: explicit inputs-only mode
    # can check the trace, but must never certify different source as equivalent.
    for rank in (0, 1):
        manifest_path = runs[1] / f"pdet/manifest-rank{rank}.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["git_commit"] = "a" * 40
        manifest["source_tree_sha256"] = "b" * 64
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="isolation failed"):
        audit_order(runs)
    report = audit_order(runs, inputs_only=True)
    assert report["audit"] == "inputs_only_passed"
    assert report["code_revision"]["status"] == "not_verified"
    assert "NOT verified" in report["scope"]
    path = runs[1] / "pdet/inputs-rank1.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[30]["occurrence_noise_sha256"][0] = "corrupt"
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(ValueError, match="isolation failed"):
        audit_order(runs, inputs_only=True)


def test_plan_dataset_identity_and_prompt_update_monitor():
    class Dataset:
        def __len__(self):
            return 10

        def __getitem__(self, index):
            return {"index": index}

    wrapped = PlannedDataset(Dataset())
    assert wrapped[(4, 9)] == {"index": 4, "_pdet_occurrence_id": 9}
    policy = TinyPolicy()
    monitor = PromptUpdateMonitor(policy)
    parameter = policy.model.vlm_prompt_tokens.weight
    parameter.grad = torch.ones_like(parameter)
    before = parameter.detach().clone()
    monitor.before_step()
    with torch.no_grad():
        parameter.add_(0.01)
    metrics = monitor.after_step()
    assert metrics["prompt_update/grad_norm"] == pytest.approx(np.sqrt(parameter.numel()))
    assert metrics["prompt_update/update_norm"] == pytest.approx(float((parameter - before).norm()))


def test_repeat_comparison_checks_only_common_input_prefix(tmp_path):
    root = tmp_path / "dataset"
    make_dataset(root)
    policy = TinyPolicy()
    runs = []
    for name, audit_steps, diagnostics in (("old", 8, False), ("repeat", 50, True)):
        cfg = make_config(
            tmp_path / name,
            PDETConfig(
                enabled=True,
                prompt_seed=1,
                data_seed=0,
                noise_seed=1,
                time_seed=1,
                audit_steps=audit_steps,
                diagnostics=diagnostics,
            ),
        )
        cfg.steps, cfg.batch_size, cfg.dataset.root = 50, 2, root
        runs.append(cfg.output_dir)
        for rank in (0, 1):
            streams = PDETStreams(cfg, policy, accelerator(rank))
            for step in range(audit_steps):
                streams.draw(policy, {"action": torch.zeros(2, 50, 7)}, step, [step % 24, (step + 1) % 24])
    report = compare_inputs(*runs)
    assert report["matched_steps"] == 8 and report["matched_step_rank_pairs"] == 16
    path = runs[1] / "pdet/inputs-rank0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["noise_sha256"] = "corrupt"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="Repeat changed noise_sha256"):
        compare_inputs(*runs)


def test_fixed_probe_loss_does_not_change_gradients_parameters_or_rng(tmp_path):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    policy = TinyPolicy()
    policy._preprocess_images = lambda batch: ([], [])
    policy.model.predict_velocity = lambda images, masks, tokens, text_masks, noisy, time: (
        noisy * policy.model.vlm_prompt_tokens.weight.mean()
    )
    monitor = PromptRankMonitor.__new__(PromptRankMonitor)
    monitor.options = PromptRankConfig(
        freq=250, samples=1, output_probes=1, timesteps=[0.5], fixed_flow_loss=True
    )
    monitor.device = torch.device("cpu")
    monitor.output_dir = tmp_path
    monitor.action_dim, monitor.horizon = 7, 50
    monitor.samples = [
        {
            "action": torch.ones(1, 50, 7),
            OBS_LANGUAGE_TOKENS: torch.ones(1, 2),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2),
        }
    ]
    monitor.noise = [torch.zeros(1, 50, 32)]
    monitor.probes = torch.ones(1, 1, 50, 7)
    monitor.manifest = {"samples": [{"task_index": 0}]}
    acc = SimpleNamespace(
        is_main_process=True,
        num_processes=1,
        process_index=0,
        device=torch.device("cpu"),
        autocast=nullcontext,
        gather=lambda x: x,
        reduce=lambda x, op: x,
    )
    parameter = policy.model.vlm_prompt_tokens.weight
    parameter.grad = torch.ones_like(parameter)
    before, gradient, state = (
        parameter.detach().clone(),
        parameter.grad.clone(),
        torch.get_rng_state().clone(),
    )
    expected = float((0.5 * parameter.float().mean() + 1).square())
    metrics = monitor.measure(policy, 0, acc)
    assert metrics["fixed_training_probe_flow_mse"] == pytest.approx(expected)
    assert policy.model.vlm_prompt_tokens.weight is parameter
    assert torch.equal(before, parameter) and torch.equal(gradient, parameter.grad)
    assert torch.equal(state, torch.get_rng_state())
    assert (tmp_path / "fixed_training_probe_loss.jsonl").is_file()


@pytest.mark.parametrize(
    "mode,gpus,port,count", [("repeats", "0,1", "29541", 4), ("order", "2,3", "29542", 2)]
)
def test_launchers_preflight_isolate_names_ports_and_final_only(tmp_path, mode, gpus, port, count):
    from tests.scripts.test_pi05_vlm_only import VLMOnlyTest

    fixture = VLMOnlyTest()
    fixture.setUp()
    try:
        make_dataset(fixture.dataset)
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
        env = {**fixture.env, "PROMPT_INIT_DIR": str(directory), "PLAN_DIR": str(tmp_path / "plans")}
        result = subprocess.run(
            ["bash", str(LAUNCHER), mode], env=env, text=True, capture_output=True, timeout=180
        )
        assert result.returncode == 0, result.stderr
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]
        assert len(commands) == count
        for command in commands:
            for value in (
                f"CUDA_VISIBLE_DEVICES={gpus}",
                f"--main_process_port={port}",
                "--num_processes=2",
                "--batch_size=32",
                "--seed=0",
                "--steps=3000",
                "--save_steps=[]",
                "--policy.ntk_save_stage_snapshots=false",
                "--policy.compile_model=false",
                "--pdet.audit_steps=3000",
                "--prompt_rank.fixed_flow_loss=true",
            ):
                assert value in command
        output = Path(next(arg.split("=", 1)[1] for arg in commands[-1] if arg.startswith("--output_dir=")))
        output.mkdir(parents=True)
        result = subprocess.run(
            ["bash", str(LAUNCHER), mode], env=env, text=True, capture_output=True, timeout=180
        )
        assert result.returncode != 0 and "Output already exists" in result.stderr
        assert "Starting " not in result.stdout
    finally:
        fixture.doCleanups()
