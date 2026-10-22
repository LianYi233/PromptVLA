"""Controls for temporal identity, finite curriculum counts and realized flow pairing."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate.data_loader import BatchSamplerShard

from lerobot.configs.pdet import PDETConfig
from lerobot.scripts.frameskip_experiment import audit_plans, audit_runs
from lerobot.scripts.frameskip_scores import gripper_threshold
from lerobot.utils.frameskip_curriculum import (
    ScoreConfig,
    dependent_round,
    load_scores,
    make_plans,
    prepare_plans,
    save_scores,
    score_episode,
    source_mask,
)
from lerobot.utils.pdet import audit_record, sample_flow, tensor_hash
from lerobot.utils.pdet_order import (
    FLOW_PROTOCOL,
    META_COLUMNS,
    PlannedDataset,
    PlannedSampler,
    arrays_hash,
    save_plan,
    validate_plan_dataset,
)


def fixture_scores(tmp_path):
    size = 60
    episodes = 8
    n = size * episodes
    metadata = {
        "index": np.arange(n, dtype=np.int64),
        "task_index": np.repeat(np.arange(4, dtype=np.int64), size * 2),
        "episode_index": np.repeat(np.arange(episodes, dtype=np.int64), size),
        "frame_index": np.tile(np.arange(size, dtype=np.int64), episodes),
        "phase_bin": np.tile(np.arange(size, dtype=np.int64) * 4 // size, episodes),
        "gripper_change": np.zeros(n, dtype=np.int64),
    }
    pieces = []
    rng = np.random.default_rng(42)
    for episode in range(episodes):
        actions = rng.normal(size=(size, 7))
        actions[:, -1] = np.r_[np.zeros(30), np.ones(30)]
        metadata["gripper_change"][episode * size + 30] = 1
        pieces.append(score_episode(actions, rng.random((size, 2)), np.full(6, -1), np.ones(6), 0.5, episode))
    scores = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    path = tmp_path / "scores.npz"
    save_scores(path, metadata, scores, {"test": True})
    return path, metadata, scores


def test_score_constant_inputs_and_protected_random_control():
    actions = np.zeros((50, 7))
    actions[25:, -1] = 1
    actions[10, -1] = 1  # a one-frame glitch must not become an event
    scores = score_episode(actions, np.zeros((50, 2)), np.zeros(6), np.ones(6), 0.5, 7)
    assert np.flatnonzero(scores["event"]).tolist() == [25]
    assert np.flatnonzero(scores["event_neighborhood"]).tolist() == list(range(15, 28))
    assert np.flatnonzero(scores["protected"]).tolist() == [0, 25, 49]
    assert scores["selected"].sum() == scores["random_selected"].sum() == 10
    assert np.all(scores["selected"][scores["protected"]])
    assert np.all(scores["random_selected"][scores["protected"]])
    assert np.all(scores["action_score"] == 0)
    assert np.all(scores["visual_score"] == 0)
    assert np.isfinite(scores["score"]).all()
    # Quota expands, identically for both pools, if protection exceeds nominal retention.
    expanded = score_episode(
        actions, np.zeros((50, 2)), np.zeros(6), np.ones(6), 0.5, 7, ScoreConfig(retention=0.01)
    )
    assert expanded["selected"].sum() == expanded["random_selected"].sum() == 3
    with pytest.raises(ValueError, match="Nonfinite actions"):
        score_episode(np.full((3, 7), np.nan), np.zeros((3, 2)), np.zeros(6), np.ones(6), 0.5, 1)
    assert gripper_threshold(np.array([[0] * 7, [0] * 6 + [1]])) == 0.5
    assert gripper_threshold(np.array([[0] * 6 + [-1], [0] * 6 + [1]])) == 0
    with pytest.raises(ValueError, match="Nonbinary"):
        gripper_threshold(np.array([[0] * 6 + [0.2], [0] * 6 + [0.8]]))


def test_fixed_size_rounding_preserves_marginals_and_rng_isolation():
    p = np.linspace(0.2, 0.8, 100)
    values = np.array([dependent_round(p, np.random.default_rng(seed)) for seed in range(800)])
    assert np.all(values.sum(axis=1) == 50)
    assert np.max(np.abs(values.mean(axis=0) - p)) < 0.075


def test_family_counts_warmup_order_noise_keys_and_actual_sharding(tmp_path):
    path, metadata, scores = fixture_scores(tmp_path)
    state = torch.get_rng_state().clone()
    kwargs = {"steps": 120, "batch_size": 2, "world_size": 2, "warmup_steps": 20}
    plans = prepare_plans(path, tmp_path / "plans", **kwargs)
    assert torch.equal(state, torch.get_rng_state())
    assert audit_plans(tmp_path / "plans", path, production=False)["audit"] == "plans_passed"
    warm = 80
    s, c = plans["S"], plans["C"]
    for key in META_COLUMNS:
        assert np.array_equal(s[key], c[key])
    assert not np.array_equal(s["order"], c["order"])
    assert source_mask(s).sum() == (len(s["order"]) - warm) // 2
    source = source_mask(c)[c["order"]][warm:]
    assert source[:100].mean() < 0.4
    assert source[-100:].mean() > 0.6
    for plan in plans.values():
        validate_plan_dataset(plan, metadata, steps=120, batch_size=2, world_size=2)
        assert np.array_equal(plan["index"][plan["order"][:warm]], s["index"][s["order"][:warm]])
        assert np.array_equal(plan["task_index"][plan["order"]], s["task_index"])
        batches = torch.utils.data.BatchSampler(PlannedSampler(plan), batch_size=2, drop_last=False)
        for rank in (0, 1):
            shard = list(BatchSamplerShard(batches, num_processes=2, process_index=rank))
            for step, batch in enumerate(shard):
                offset = (step * 2 + rank) * 2
                assert [o for _, o in batch] == plan["order"][offset : offset + 2].tolist()
            resumed = torch.utils.data.BatchSampler(PlannedSampler(plan, 37), batch_size=2, drop_last=False)
            assert list(BatchSamplerShard(resumed, num_processes=2, process_index=rank)) == shard[37:]
    # Same source-frame can have multiple independent USE IDs, never a fixed noise per frame.
    assert len(np.unique(s["index"])) < len(s["index"])
    changed = {**metadata, "frame_index": metadata["frame_index"].copy()}
    changed["frame_index"][0] += 1
    with pytest.raises(ValueError, match="metadata differ"):
        validate_plan_dataset(s, changed, **{k: kwargs[k] for k in ("steps", "batch_size", "world_size")})

    # The wrapper delegates the untouched chunk-start index to the original dataset.
    class OriginalDataset:
        def __len__(self):
            return 500

        def __getitem__(self, index):
            return {"index": index, "action": torch.arange(index, index + 50)}

    item = PlannedDataset(OriginalDataset())[(31, 8)]
    assert item["action"].tolist() == list(range(31, 81))
    assert item["_pdet_occurrence_id"] == 8
    with pytest.raises(ValueError, match="Different order plan"):
        save_plan(tmp_path / "plans/S.npz", c)
    again = make_plans(metadata, scores, load_scores(path)[1], **kwargs)
    assert again["C"]["manifest"] == c["manifest"]


class TinyPolicy:
    def __init__(self):
        self.model = SimpleNamespace(
            sample_time=lambda n, device: (
                torch.distributions.Beta(1.5, 1.0).sample((n,)) * 0.999 + 0.001
            ).to(device)
        )

    def prepare_action(self, batch):
        return batch["action"]


def write_runs(tmp_path):
    path, metadata, _ = fixture_scores(tmp_path)
    plans = prepare_plans(path, tmp_path / "plans", steps=8, batch_size=2, world_size=2, warmup_steps=2)
    runs = []
    for label, plan in plans.items():
        for bank in (0, 1):
            run = tmp_path / f"pi05-fskip-r1-{label}-Bocc-base-vlm2-P1-D0-E{bank}-T{bank}"
            (run / "pdet").mkdir(parents=True)
            factors = {
                "enabled": True,
                "prompt_seed": 1,
                "data_seed": 0,
                "noise_seed": bank,
                "time_seed": bank,
                "audit_steps": 8,
                "diagnostics": True,
                "order_plan": str(tmp_path / f"plans/{label}.npz"),
                "flow_binding": "occurrence",
            }
            manifest = {
                "protocol": FLOW_PROTOCOL,
                "flow_key_namespace": FLOW_PROTOCOL,
                "factors": factors,
                "global_seed": 0,
                "world_size": 2,
                "per_device_batch_size": 2,
                "tokens": 2,
                "dtype": "torch.bfloat16",
                "prompt_sha256": "p",
                "source_pretrained_path": "base",
                "dataset_root": "dataset",
                "dataset_metadata_sha256": arrays_hash(metadata),
                "versions": {},
                "cuda": None,
                "recipe": {},
                "git_commit": "commit",
                "source_tree_sha256": "code",
                "total_steps": 8,
                "order_plan": plan["manifest"],
            }
            config = PDETConfig(**factors)
            for rank in (0, 1):
                (run / f"pdet/manifest-rank{rank}.json").write_text(json.dumps(manifest))
                records = []
                for step in range(8):
                    start = (step * 2 + rank) * 2
                    ids = plan["order"][start : start + 2].tolist()
                    draws = sample_flow(
                        TinyPolicy(), {"action": torch.zeros(2, 50, 7)}, config, step, rank, ids
                    )
                    record = audit_record(plan["index"][ids], draws, step=step, rank=rank)
                    record.update(occurrence_ids=ids, flow_key_ids=ids)
                    for factor in ("noise", "time"):
                        record[f"occurrence_{factor}_sha256"] = [
                            tensor_hash(draws[factor][i]) for i in range(2)
                        ]
                    record.update({key: plan[key][ids].tolist() for key in META_COLUMNS if key != "index"})
                    records.append(record)
                (run / f"pdet/inputs-rank{rank}.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
            runs.append(run)
    return runs


def test_real_flow_bank_audit_and_late_corruption(tmp_path):
    runs = write_runs(tmp_path)
    assert audit_runs(runs, production=False)["audit"] == "passed"
    assert audit_runs(runs[:2], production=False, allow_partial=True)["audit"] == "partial_inputs_passed"
    with pytest.raises(ValueError, match="all eight"):
        audit_runs(runs[:2], production=False)
    trace = runs[-1] / "pdet/inputs-rank1.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[-1]["occurrence_noise_sha256"][0] = "a" * 64
    trace.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="different realized noise"):
        audit_runs(runs, production=False)


def test_corrupt_score_and_plan_files_fail_closed(tmp_path):
    path, _, _ = fixture_scores(tmp_path)
    arrays, manifest = load_scores(path)
    arrays["score"][0] += 1
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays, manifest=json.dumps(manifest))
    with pytest.raises(ValueError, match="corrupted score"):
        load_scores(path)


def test_vision_only_load_and_batch_boundary_continuity(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from transformers import SiglipVisionConfig, SiglipVisionModel

    from lerobot.scripts.frameskip_scores import extract_episode, load_vision

    config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=4304,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
        vision_use_head=False,
        projection_dim=2048,
    )
    original = SiglipVisionModel(config).eval()
    state = {
        f"model.paligemma_with_expert.paligemma.model.vision_tower.{key}": value
        for key, value in original.state_dict().items()
    }
    state["model.unused_language_weight"] = torch.ones(2, 2)
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05", "image_resolution": [28, 28]}))
    config._attn_implementation = "eager"
    monkeypatch.setattr("lerobot.scripts.frameskip_scores.vision_config_for", lambda resolution: config)
    loaded, resolution, digest = load_vision(tmp_path, "cpu")
    assert len(digest) == 64
    assert not any(p.requires_grad for p in loaded.parameters())
    for key, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])

    class Images:
        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(index)
            return {
                "index": index,
                "external": torch.rand(3, 32, 32, generator=generator),
                "wrist": torch.rand(3, 32, 32, generator=generator),
            }

    a = extract_episode(Images(), np.arange(5), ["external", "wrist"], loaded, resolution, "cpu", 2)
    b = extract_episode(Images(), np.arange(5), ["external", "wrist"], loaded, resolution, "cpu", 5)
    assert np.all(a[0] == 0)
    assert np.all(a[1:] > 0)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    # Strict loader must fail instead of silently filling a missing layer at random.
    del state[next(iter(state))]
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_vision(tmp_path, "cpu")


def test_launchers_reject_invalid_selection_and_existing_output(tmp_path):
    import os
    import subprocess

    root = Path(__file__).resolve().parents[2]
    script = root / "examples/training/train_pi05_frameskip_curriculum.sh"
    env = os.environ.copy()
    for key in (
        "PRETRAINED_PATH",
        "RUN_NAME",
        "RUN_PREFIX",
        "VLM_PROMPT_TOKENS",
        "PROMPT_INIT_PATH",
        "PROMPT_EXPORT_PATH",
        "PROMPT_EXPORT_REFERENCE",
        "PDET_CASES",
    ):
        env.pop(key, None)
    env.update(OUTPUT_ROOT=str(tmp_path), CELLS="S0 S0")
    result = subprocess.run(["bash", str(script), "a"], env=env, text=True, capture_output=True)
    assert result.returncode == 2 and "duplicate cell" in result.stderr
    env["CELLS"] = "S0"
    output = tmp_path / "pi05-fskip-r1-S-Bocc-base-vlm2-P1-D0-E0-T0"
    output.mkdir()
    (output / "keep.txt").write_text("existing results")
    result = subprocess.run(["bash", str(script), "a"], env=env, text=True, capture_output=True)
    assert result.returncode == 1 and "Output already exists" in result.stderr
    assert (output / "keep.txt").read_text() == "existing results"


def test_dry_run_shell_wires_both_balanced_queues(tmp_path):
    """Stub only CLI preflights; exercise the real sweep and underlying Bash launcher."""
    import os
    import re
    import shlex
    import subprocess

    root = Path(__file__).resolve().parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text("#!/bin/sh\n# Preflights tested separately with real Python.\nexit 0\n")
    python_stub.chmod(0o755)
    env = os.environ.copy()
    for key in (
        "PRETRAINED_PATH",
        "RUN_NAME",
        "RUN_PREFIX",
        "VLM_PROMPT_TOKENS",
        "PROMPT_INIT_PATH",
        "PROMPT_EXPORT_PATH",
        "PROMPT_EXPORT_REFERENCE",
        "PDET_CASES",
        "CELLS",
        "GPU_IDS",
        "MAIN_PROCESS_PORT",
        "EXPERIMENT_ID",
    ):
        env.pop(key, None)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        DRY_RUN="true",
        OUTPUT_ROOT=str(tmp_path / "outputs"),
        LOG_ROOT=str(tmp_path / "logs"),
        DATASET_ROOT=str(tmp_path),
        BASE_PRETRAINED_PATH=str(tmp_path),
        TOKENIZER_PATH=str(tmp_path),
        PROMPT_INIT_DIR=str(tmp_path),
        SCORE_DIR=str(tmp_path),
        PLAN_DIR=str(tmp_path / "plans"),
    )
    for queue, cells, gpus, port in (
        ("a", ["U0", "R1", "S0", "C1"], "0,1", "29561"),
        ("b", ["U1", "R0", "S1", "C0"], "2,3", "29562"),
    ):
        result = subprocess.run(
            [
                "bash",
                str(root / "examples/training/train_pi05_frameskip_curriculum.sh"),
                queue,
                "--wandb.enable=false",
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        commands = [
            line for line in result.stdout.splitlines() if line.startswith("env CUDA_VISIBLE_DEVICES")
        ]
        assert len(commands) == 4
        for cell, command in zip(cells, commands, strict=True):
            command = " ".join(shlex.split(command))
            bank = cell[1]
            assert f"CUDA_VISIBLE_DEVICES={gpus}" in command
            assert f"--main_process_port={port}" in command
            assert f"--pdet.order_plan={tmp_path}/plans/{cell[0]}.npz" in command
            assert f"--pdet.noise_seed={bank}" in command and f"--pdet.time_seed={bank}" in command
            assert "--pdet.flow_binding=occurrence" in command
            assert "--save_freq=3000" in command and "--policy.ntk_save_stage_snapshots=false" in command
            assert "--policy.num_vlm_prompt_tokens=2" in command and "--batch_size=32" in command
            assert "--num_processes=2" in command and "--policy.compile_model=false" in command
            assert re.search(rf"pi05-fskip-r1-{cell[0]}-Bocc-base-vlm2-P1-D0-E{bank}-T{bank}", command)
    assert not (tmp_path / "outputs").exists()
