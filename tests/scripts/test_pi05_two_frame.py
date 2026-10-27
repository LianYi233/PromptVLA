"""Two-frame selection, paired flow, launch/eval and realized trace invariants."""

# ruff: noqa: F811 -- shared pytest fixture
import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.numpy import save_file

from lerobot.configs.pdet import PDETConfig
from lerobot.scripts import pi05_source_compare as sources, pi05_two_frame as experiment
from lerobot.scripts.frameskip_scores import file_inventory, read_actions, source_inventory
from lerobot.utils.frameskip_curriculum import save_scores
from lerobot.utils.pdet import PDETStreams, sample_flow
from lerobot.utils.pdet_order import (
    META_COLUMNS,
    arrays_hash,
    create_plan,
    fixed_probe_indices,
    load_plan,
    read_frame_metadata,
    save_plan,
    validate_plan_dataset,
)
from tests.scripts.test_pi05_pdet import TinyPolicy, accelerator, make_config
from tests.scripts.test_pi05_source_compare import inputs  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "examples/training/train_pi05_two_frame.sh"
EVAL = ROOT / "run_eval_libero_two_frame.sh"


def small_inputs():
    n = 256
    meta = {key: np.zeros(n, dtype=np.int64) for key in META_COLUMNS}
    meta.update(index=np.arange(n), frame_index=np.arange(n) % 128, episode_index=np.arange(n) // 128)
    scores = {**meta, "score": (np.arange(n) % 128) / 128.0}
    # Include all frames in the candidate pool, allowing full 50-step chunks.
    plan = create_plan(meta, steps=64, batch_size=2, world_size=2, data_seed=0, within_seed=0)
    return plan, scores


def test_selection_same_episode_nonoverlapping_complete_chunks_and_manual_controls():
    base, scores = small_inputs()
    selection = experiment.select_pair(base, scores)
    easy, hard = selection["easy"]["index"], selection["hard"]["index"]
    assert selection["episode_index"] == 0
    assert scores["score"][easy] < scores["score"][hard]
    assert hard - easy >= 50 and hard + 50 <= 128
    assert experiment.select_pair(base, scores, easy_index=easy, hard_index=hard)["rule"] == "manual"
    with pytest.raises(ValueError, match="both"):
        experiment.select_pair(base, scores, easy_index=easy)
    with pytest.raises(ValueError, match="non-overlapping"):
        experiment.select_pair(base, scores, easy_index=10, hard_index=20)
    with pytest.raises(ValueError, match="No eligible"):
        experiment.select_pair(base, scores, easy_index=10, hard_index=128 + 60)
    with pytest.raises(ValueError, match="No eligible"):
        experiment.select_pair(base, {**scores, "score": np.zeros(256)})
    with pytest.raises(ValueError, match="no candidates"):
        experiment.select_pair(base, scores, task_index=3)


def test_two_balanced_blocks_preserve_sample_noise_pairs_across_steps_and_ranks(tmp_path):
    base, scores = small_inputs()
    selection = experiment.select_pair(base, scores)
    plans, banks = [], []
    before = torch.get_rng_state().clone(), np.random.get_state()
    for direction in experiment.DIRECTIONS:
        plan = experiment.make_plan(base, scores, selection, steps=20, direction=direction)
        save_plan(tmp_path / f"{direction}.npz", plan)
        plan = load_plan(tmp_path / f"{direction}.npz")
        plans.append(plan)
        indices = [selection[label]["index"] for label in ("easy", "hard")]
        assert fixed_probe_indices(plan) == indices
        validate_plan_dataset(
            plan, {k: scores[k] for k in META_COLUMNS}, steps=20, batch_size=2, world_size=2
        )
        assert np.unique(plan["index"]).tolist() == indices
        realized = plan["index"][plan["order"]]
        first, second = indices if direction == "E2H" else indices[::-1]
        assert (realized[:40] == first).all() and (realized[40:] == second).all()
        bank = {}
        for start in range(0, 80, 2):
            ids = plan["order"][start : start + 2].tolist()
            flow = sample_flow(
                TinyPolicy(),
                {"action": torch.zeros(2, 50, 7)},
                PDETConfig(enabled=True, order_plan="plan", noise_seed=1, time_seed=1),
                start // 4,
                start // 2 % 2,
                ids,
            )
            for i, use in enumerate(ids):
                bank[use] = (int(plan["index"][use]), flow["noise"][i], flow["time"][i])
        banks.append(bank)
    assert plans[0]["manifest"]["instance_pool_sha256"] == plans[1]["manifest"]["instance_pool_sha256"]
    for use, (index, noise, time) in banks[0].items():
        other = banks[1][use]
        assert index == other[0] and torch.equal(noise, other[1]) and torch.equal(time, other[2])
    assert not torch.equal(banks[0][0][1], banks[0][1][1])
    # TinyPolicy construction consumes RNG; plan construction alone must not.
    before = torch.get_rng_state().clone(), np.random.get_state()
    experiment.make_plan(base, scores, selection)
    assert torch.equal(before[0], torch.get_rng_state())
    np.testing.assert_equal(before[1], np.random.get_state())
    bad = copy.deepcopy(plans[0])
    bad["manifest"]["two_frame"]["probe_indices"] = [0, 1]
    with pytest.raises(ValueError, match="exact two"):
        fixed_probe_indices(bad)
    assert fixed_probe_indices(base) is None


@pytest.fixture
def prepared(inputs, tmp_path):
    data = inputs["dataset_root"]
    n = 512
    pq.write_table(
        pa.table(
            {
                "index": range(n),
                "episode_index": [i // 128 for i in range(n)],
                "frame_index": [i % 128 for i in range(n)],
                "task_index": [i // 256 for i in range(n)],
                "action": [[0.0] * 7] * n,
            }
        ),
        data / "data/part.parquet",
    )
    meta = read_frame_metadata(data)
    plan = create_plan(meta, steps=3000, batch_size=32, world_size=2, data_seed=0, within_seed=0)
    inputs["plan_path"].unlink()
    save_plan(inputs["plan_path"], plan)
    path = inputs["reference_run"] / "pdet/manifest-rank0.json"
    record = sources.read_json(path)
    record["order_plan"] = plan["manifest"]
    path.write_text(json.dumps(record))
    sources.prepare(**inputs)
    scores = {"score": (np.arange(n) % 128).astype(float)}
    scores.update({key: np.zeros(n, dtype=bool) for key in ("protected", "selected", "random_selected")})
    provenance = {
        "source": str(inputs["libero"]),
        "dataset_root": str(data),
        "actions_sha256": arrays_hash({"action": read_actions(data)}),
        "stats_sha256": sources.digest(data / "meta/stats.json"),
        "source_inventory": source_inventory(inputs["libero"]),
        "video_inventory_sha256": hashlib.sha256(json.dumps(file_inventory(data, [])).encode()).hexdigest(),
    }
    scores_path = tmp_path / "scores.npz"
    save_scores(scores_path, meta, scores, provenance)
    args = {"source_pair": inputs["output"], "scores_path": scores_path, "output": tmp_path / "twoframe"}
    return args


def test_prepare_reuses_pair_and_detects_resigned_changed_order(prepared):
    assert experiment.prepare(**prepared)["distinct_training_starts"] == 2
    assert experiment.prepare(**prepared)["prepared"] == "reused"
    record = experiment.check(prepared["output"])
    with pytest.raises(ValueError, match="different inputs"):
        experiment.prepare(**prepared, task_index=1)
    plan_path = prepared["output"] / "E2H.npz"
    plan = load_plan(plan_path)
    assert len(plan["order"]) == 384000
    plan["order"] = plan["order"][::-1].copy()
    plan["manifest"]["arrays_sha256"] = arrays_hash({k: plan[k] for k in (*META_COLUMNS, "order")})
    plan_path.unlink()
    save_plan(plan_path, plan)
    record["plans"]["E2H"] = plan["manifest"]
    (prepared["output"] / "pair.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="block order"):
        experiment.check(prepared["output"])


def env_for(prepared, tmp_path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    return {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "SOURCE_PAIR_DIR": str(prepared["source_pair"]),
        "PAIR_DIR": str(prepared["output"]),
        "DATASET_ROOT": sources.check(prepared["source_pair"])["dataset_root"],
        "TOKENIZER_PATH": str(tokenizer),
        "OUTPUT_ROOT": str(tmp_path / "runs"),
        "LOG_ROOT": str(tmp_path / "logs"),
        "DRY_RUN": "true",
    }


def test_actual_training_commands_differ_only_by_order_name_and_port(prepared, tmp_path):
    experiment.prepare(**prepared)
    parsed = []
    for direction, gpu, port in (("E2H", "0,1", "29586"), ("H2E", "2,3", "29587")):
        result = subprocess.run(
            ["bash", str(TRAIN), direction],
            env=env_for(prepared, tmp_path) | {"GPU_IDS": gpu, "MAIN_PROCESS_PORT": port},
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        line = next(
            line for line in result.stdout.splitlines() if line.startswith("env CUDA_VISIBLE_DEVICES=")
        )
        flags = dict(
            token[2:].split("=", 1) for token in shlex.split(line) if token.startswith("--") and "=" in token
        )
        parsed.append(flags)
        assert flags["steps"] == flags["save_freq"] == flags["pdet.audit_steps"] == "6000"
        assert flags["policy.train_action_expert"] == "true" and flags["policy.vlm_prompt_only"] == "false"
        assert flags["policy.num_vlm_prompt_tokens"] == "2" and flags["policy.num_prompt_tokens"] == "0"
        assert flags["prompt_rank.samples"] == "2" and flags["prompt_rank.fixed_flow_loss"] == "true"
        assert json.loads(flags["prompt_rank.extra_steps"]) == [50, 100, 3000]
        assert flags["save_steps"] == "[]" and flags["policy.ntk_save_stage_snapshots"] == "false"
        assert flags["batch_size"] == "32" and flags["num_processes"] == "2"
        assert flags["policy.path"] == str(prepared["source_pair"] / "libero")
        assert flags["pdet.order_plan"] == str(prepared["output"] / f"{direction}.npz")
    assert {key for key in parsed[0] if parsed[0][key] != parsed[1][key]} == {
        "pdet.order_plan",
        "job_name",
        "output_dir",
        "main_process_port",
    }


def test_full_evaluator_loads_trained_action_weights(prepared, tmp_path):
    experiment.prepare(**prepared)
    record = experiment.check(prepared["output"])
    run = tmp_path / (experiment.prefix(record, "H2E") + "-base-vlm2-P1-D0-E1-T1")
    checkpoint = run / "checkpoints/006000/pretrained_model"
    shutil.copytree(prepared["source_pair"] / "libero", checkpoint)
    config = sources.read_json(checkpoint / "config.json")
    config.update(
        vlm_prompt_only=False,
        train_action_expert=True,
        num_vlm_prompt_tokens=2,
        num_prompt_tokens=0,
        training_stage="flow",
        next_action_pretrain_steps=0,
        next_action_bridge_steps=0,
        cabo_enabled=False,
        train_action_projections=False,
        use_peft=False,
        pretrained_path=str(prepared["source_pair"] / "libero"),
    )
    (checkpoint / "config.json").write_text(json.dumps(config))
    factors = {
        "enabled": True,
        "prompt_seed": 1,
        "data_seed": 0,
        "noise_seed": 1,
        "time_seed": 1,
        "flow_binding": "occurrence",
        "order_plan": str(prepared["output"] / "H2E.npz"),
    }
    (checkpoint / "train_config.json").write_text(
        json.dumps({"policy": config, "pdet": factors, "seed": 0, "steps": 6000, "batch_size": 32})
    )
    weights = {
        "model.vlm_prompt_tokens.weight": np.ones((2, 2048), dtype=np.float32),
        "model.prompt_tokens.weight": np.empty((0, 1024), dtype=np.float32),
    }
    for module in (
        "paligemma_with_expert.gemma_expert.model",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    ):
        weights[f"model.{module}.weight"] = np.ones((2, 2), dtype=np.float32)
    save_file(weights, checkpoint / "model.safetensors")
    (run / "checkpoints/last").symlink_to("006000")
    assert experiment.check_checkpoint(prepared["output"], run)["direction"] == "H2E"
    result = subprocess.run(
        ["bash", str(EVAL), "H2E", "all"],
        env=env_for(prepared, tmp_path) | {"CKPT_ROOT": str(tmp_path)},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "train_action_expert=True" in result.stdout and f"--policy.path={checkpoint}" in result.stdout


def test_paired_audit_requires_same_realized_noise_and_reports_per_frame_losses(tmp_path, monkeypatch):
    base, scores = small_inputs()
    selection = experiment.select_pair(base, scores)
    record = {"requested": {"steps": 20}, "selection": selection}
    indices = [selection[k]["index"] for k in ("easy", "hard")]
    monkeypatch.setattr(
        "lerobot.utils.pdet_order.read_frame_metadata", lambda _: {k: scores[k] for k in META_COLUMNS}
    )
    runs = []
    policy = TinyPolicy()
    for direction in experiment.DIRECTIONS:
        run = tmp_path / (experiment.prefix(record, direction) + "-base-vlm2-P1-D0-E1-T1")
        runs.append(run)
        plan = experiment.make_plan(base, scores, selection, steps=20, direction=direction)
        path = tmp_path / f"{direction}.npz"
        save_plan(path, plan)
        cfg = make_config(
            run,
            PDETConfig(
                enabled=True,
                prompt_seed=1,
                data_seed=0,
                noise_seed=1,
                time_seed=1,
                order_plan=str(path),
                audit_steps=20,
                diagnostics=True,
            ),
        )
        cfg.steps, cfg.batch_size, cfg.dataset.root = 20, 2, tmp_path / "data"
        cfg.to_dict = lambda: {"policy": {"vlm_prompt_only": False, "train_action_expert": True}}
        for rank in (0, 1):
            streams = PDETStreams(cfg, policy, accelerator(rank))
            for step in range(20):
                start = (step * 2 + rank) * 2
                ids = plan["order"][start : start + 2].tolist()
                streams.draw(
                    policy, {"action": torch.zeros(2, 50, 7)}, step, plan["index"][ids].tolist(), ids
                )
        probe = run / "prompt_rank"
        probe.mkdir()
        (probe / "manifest.json").write_text(
            json.dumps(
                {
                    "fixed_dataset_indices": indices,
                    "samples": [{"dataset_index": i} for i in indices],
                    "options": {"timesteps": [0.2, 0.5, 0.8]},
                }
            )
        )
        rows = [
            {
                "step": step,
                "samples": [
                    {"sample_index": i, "time": t, "squared_error_sum": 2.0 + i, "valid_coordinates": 2}
                    for i in (0, 1)
                    for t in (0.2, 0.5, 0.8)
                ],
            }
            for step in (0, 10, 20)
        ]
        (probe / "fixed_training_probe_loss.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    monkeypatch.setattr(experiment, "check", lambda _: record)
    monkeypatch.setattr(
        experiment,
        "check_checkpoint",
        lambda _, run: {"direction": experiment.NAME.fullmatch(Path(run).name)[2]},
    )
    monkeypatch.setattr(experiment, "assert_same_processors", lambda *_: None)
    report = experiment.audit(tmp_path, runs)
    assert report["audit"] == "paired_order_only_passed"
    assert report["runs"][0]["bank_sha256"] == report["runs"][1]["bank_sha256"]
    assert report["runs"][0]["checkpoints_of_loss"][-1]["balanced_mse"] == 1.25
    assert (runs[0] / "two_frame_losses.csv").exists()
    path = runs[1] / "pdet/inputs-rank1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["occurrence_noise_sha256"][0] = "0" * 64
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="differs beyond order"):
        experiment.audit(tmp_path, runs)
