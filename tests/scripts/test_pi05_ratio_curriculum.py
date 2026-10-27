"""Exposure counts, frame/flow identity, launcher scope and trace audits."""

# ruff: noqa: F811 -- fixtures imported from the existing source-pair/H2E tests

import copy
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file

from lerobot.configs.pdet import PDETConfig
from lerobot.scripts import pi05_ratio_curriculum as experiment, pi05_source_compare as sources
from lerobot.utils.pdet import PDETStreams, sample_flow
from lerobot.utils.pdet_order import META_COLUMNS, arrays_hash, load_plan, save_plan, validate_plan_dataset
from tests.scripts.test_pi05_hard_to_easy import prepared_inputs, small_inputs  # noqa: F401
from tests.scripts.test_pi05_pdet import TinyPolicy, accelerator, make_config
from tests.scripts.test_pi05_source_compare import inputs  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "examples/training/train_pi05_ratio_curriculum.sh"
EVAL = ROOT / "run_eval_libero_ratio_curriculum.sh"


def run_name(direction="easy_to_hard", percent=70, seed=0):
    return f"pi05-curriculum-{experiment.label(direction, percent, seed)}-aexpert-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1"


@pytest.mark.parametrize("direction", experiment.DIRECTIONS)
@pytest.mark.parametrize("percent", [30, 50, 70, 80])
def test_exposure_order_coverage_and_fixed_budget(tmp_path, direction, percent):
    base, scores, manifest = small_inputs()
    before = copy.deepcopy(base)
    np_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    plan = experiment.make_plan(base, scores, manifest, direction=direction, easy_percent=percent)
    np.testing.assert_equal(np_state, np.random.get_state())
    assert torch.equal(torch_state, torch.get_rng_state())
    assert base["manifest"] == before["manifest"]
    for key in (*META_COLUMNS, "order"):
        np.testing.assert_array_equal(base[key], before[key])
    path = tmp_path / "plan.npz"
    save_plan(path, plan)
    plan = load_plan(path)
    validate_plan_dataset(plan, {k: scores[k] for k in META_COLUMNS}, steps=20, batch_size=2, world_size=2)
    np.testing.assert_array_equal(plan["task_index"], base["task_index"])
    np.testing.assert_array_equal(plan["task_index"][plan["order"]], base["task_index"])
    assert len(plan["order"]) == len(base["order"]) == 80
    easy = experiment.partition(base, scores, manifest)
    assert easy[plan["index"]].sum() == round(len(plan["order"]) * percent / 100)
    for task in np.unique(plan["task_index"]):
        slots = np.flatnonzero(plan["task_index"] == task)
        frames = plan["index"][plan["order"][slots]]
        delta = np.diff(scores["score"][frames])
        assert (delta >= 0).all() if direction == "easy_to_hard" else (delta <= 0).all()
        groups = easy[frames].astype(int)
        assert (np.diff(groups) <= 0).all() if direction == "easy_to_hard" else (np.diff(groups) >= 0).all()
        assert abs(groups.sum() - len(slots) * percent / 100) < 1
        for group in (False, True):
            candidates = base["index"][slots][easy[base["index"][slots]] == group]
            selected = frames[easy[frames] == group]
            assert set(selected) <= set(candidates)
            if len(selected) >= len(candidates):
                assert set(selected) == set(candidates)


def test_direction_only_keeps_new_sample_flow_pairs_and_repeats_get_distinct_draws():
    base, scores, manifest = small_inputs()
    plans = [
        experiment.make_plan(base, scores, manifest, direction=d, easy_percent=70)
        for d in experiment.DIRECTIONS
    ]
    assert plans[0]["manifest"]["instance_pool_sha256"] == plans[1]["manifest"]["instance_pool_sha256"]
    assert not np.array_equal(plans[0]["order"], plans[1]["order"])
    policy, config = TinyPolicy(), PDETConfig(enabled=True, order_plan="plan", noise_seed=1, time_seed=1)
    banks = []
    for plan in plans:
        bank = {}
        for start in range(0, len(plan["order"]), 2):
            ids = plan["order"][start : start + 2].tolist()
            flow = sample_flow(
                policy, {"action": torch.zeros(2, 50, 7)}, config, start // 4, start // 2 % 2, ids
            )
            for i, use in enumerate(ids):
                bank[use] = (int(plan["index"][use]), flow["noise"][i], flow["time"][i])
        banks.append(bank)
    for use, (frame, noise, time) in banks[0].items():
        other = banks[1][use]
        assert frame == other[0] and torch.equal(noise, other[1]) and torch.equal(time, other[2])
    indices = plans[0]["index"]
    repeated = next(np.flatnonzero(indices == frame) for frame in indices if (indices == frame).sum() > 1)
    assert not torch.equal(banks[0][int(repeated[0])][1], banks[0][int(repeated[1])][1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"easy_percent": 0},
        {"easy_percent": 100},
        {"easy_percent": 70.5},
        {"sample_seed": -1},
        {"sample_seed": 2**32},
        {"direction": "random"},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        experiment.options(**({"direction": "easy_to_hard", "easy_percent": 70, "sample_seed": 0} | kwargs))


def test_bad_scores_and_changed_metadata_rejected():
    base, scores, manifest = small_inputs()
    for values in (np.zeros_like(scores["score"]), np.full_like(scores["score"], np.nan)):
        with pytest.raises(ValueError):
            experiment.partition(base, {**scores, "score": values}, manifest)
    with pytest.raises(ValueError, match="dataset metadata"):
        experiment.partition(base, scores, {**manifest, "dataset_metadata_sha256": "bad"})
    base["index"][0] = 23 - base["index"][0]
    with pytest.raises(ValueError, match="sample pool"):
        experiment.partition(base, scores, manifest)


def test_prepare_reuses_only_identical_settings_and_detects_resigned_tampering(prepared_inputs):
    args = prepared_inputs
    settings = {"direction": "easy_to_hard", "easy_percent": 70}
    assert experiment.prepare(**args, **settings)["prepared"] == "passed"
    assert experiment.prepare(**args, **settings)["prepared"] == "reused"
    record = experiment.check(args["output"])
    with (args["output"] / "schedule.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3000
    assert sum(int(row["easy_uses"]) for row in rows) == 134400
    assert sum(int(row["hard_uses"]) for row in rows) == 57600
    with pytest.raises(ValueError, match="different settings"):
        experiment.prepare(**args, direction="hard_to_easy", easy_percent=70)
    plan_path = args["output"] / "plan.npz"
    plan = load_plan(plan_path)
    slots = np.flatnonzero(plan["task_index"] == plan["task_index"][0])
    plan["order"][slots] = plan["order"][slots][::-1]
    plan["manifest"]["arrays_sha256"] = arrays_hash({key: plan[key] for key in (*META_COLUMNS, "order")})
    plan_path.unlink()
    save_plan(plan_path, plan)
    record["plan"] = plan["manifest"]
    (args["output"] / "curriculum.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="declared ratio"):
        experiment.check(args["output"])


def launcher_env(args, tmp_path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    return {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "SOURCE_PAIR_DIR": str(args["pair"]),
        "DATASET_ROOT": sources.check(args["pair"])["dataset_root"],
        "SCORES_PATH": str(args["scores_path"]),
        "REFERENCE_RUN": str(args["reference"]),
        "ORDER_DIR": str(args["output"]),
        "OUTPUT_ROOT": str(tmp_path / "runs"),
        "LOG_ROOT": str(tmp_path / "logs"),
        "TOKENIZER_PATH": str(tokenizer),
        "DRY_RUN": "true",
    }


@pytest.mark.parametrize(
    "direction,percent,gpus,port",
    [("easy_to_hard", 70, "0,1", "29584"), ("hard_to_easy", 30, "2,3", "29585")],
)
def test_actual_training_dry_run_and_no_overwrite(prepared_inputs, tmp_path, direction, percent, gpus, port):
    args = prepared_inputs
    env = launcher_env(args, tmp_path) | {
        "CURRICULUM_DIRECTION": direction,
        "GPU_IDS": gpus,
        "MAIN_PROCESS_PORT": port,
    }
    result = subprocess.run(
        ["bash", str(TRAIN), "--wandb.project=prompt-learning"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("env CUDA_VISIBLE_DEVICES="))
    tokens = shlex.split(line)
    flags = dict(token[2:].split("=", 1) for token in tokens if token.startswith("--") and "=" in token)
    assert f"CUDA_VISIBLE_DEVICES={gpus}" in tokens
    assert flags["main_process_port"] == port
    assert flags["policy.train_action_expert"] == "true"
    assert flags["policy.vlm_prompt_only"] == "false"
    assert flags["policy.num_vlm_prompt_tokens"] == "2"
    assert flags["policy.num_prompt_tokens"] == "0"
    assert flags["policy.compile_model"] == "false"
    assert flags["steps"] == flags["save_freq"] == "3000"
    assert flags["policy.ntk_save_stage_snapshots"] == "false"
    assert flags["save_steps"] == "[]"
    assert flags["batch_size"] == "32" and flags["num_processes"] == "2"
    assert flags["policy.path"] == str(args["pair"] / "libero")
    assert flags["vlm_prompt_init_path"] == str(args["pair"] / "P1.safetensors")
    assert flags["pdet.order_plan"] == str(args["output"] / "plan.npz")
    assert flags["pdet.flow_binding"] == "occurrence"
    assert flags["pdet.noise_seed"] == flags["pdet.time_seed"] == "1"
    assert Path(flags["output_dir"]).name == run_name(direction, percent)
    assert not (tmp_path / "runs").exists()
    (tmp_path / "runs" / run_name(direction, percent)).mkdir(parents=True)
    result = subprocess.run(["bash", str(TRAIN)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "Output already exists" in result.stderr


def test_saved_checkpoint_and_full_evaluator_chain(prepared_inputs, tmp_path):
    args = prepared_inputs
    experiment.prepare(**args, direction="easy_to_hard", easy_percent=70)
    run = tmp_path / run_name()
    checkpoint = run / "checkpoints/003000/pretrained_model"
    shutil.copytree(args["reference"] / "checkpoints/003000/pretrained_model", checkpoint)
    config = sources.read_json(checkpoint / "train_config.json")
    config["pdet"].update(enabled=True, order_plan=str(args["output"] / "plan.npz"))
    config["policy"].update(train_action_expert=True, vlm_prompt_only=False)
    (checkpoint / "train_config.json").write_text(json.dumps(config))
    (checkpoint / "config.json").write_text(json.dumps(config["policy"]))
    weights = load_file(checkpoint / "model.safetensors")
    for prefix in (
        "paligemma_with_expert.gemma_expert.model",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    ):
        weights[f"model.{prefix}.weight"] = np.ones((2, 2), dtype=np.float32)
    save_file(weights, checkpoint / "model.safetensors")
    assert experiment.check_checkpoint(args["output"], run)["easy_percent"] == 70
    wrong_name = tmp_path / run_name(percent=30)
    run.rename(wrong_name)
    with pytest.raises(ValueError, match="disagrees"):
        experiment.check_checkpoint(args["output"], wrong_name)
    wrong_name.rename(run)
    config["keyframe_loss"] = {"mode": "K2"}
    (checkpoint / "train_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="loss weighting"):
        experiment.check_checkpoint(args["output"], run)
    config["keyframe_loss"] = {"mode": "off"}
    (checkpoint / "train_config.json").write_text(json.dumps(config))
    (run / "checkpoints/last").symlink_to("003000")
    env = launcher_env(args, tmp_path) | {"CKPT_ROOT": str(tmp_path), "GPU_ID": "3"}
    result = subprocess.run(["bash", str(EVAL), "all"], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "train_action_expert=True" in result.stdout
    assert f"--policy.path={checkpoint}" in result.stdout
    assert "--policy.vlm_prompt_only=true" not in result.stdout


def test_all_step_audit_allows_only_declared_pool_scope_order_changes(tmp_path, monkeypatch):
    base, scores, manifest = small_inputs()
    changed = experiment.make_plan(base, scores, manifest, direction="easy_to_hard", easy_percent=70)
    monkeypatch.setattr(
        "lerobot.utils.pdet_order.read_frame_metadata", lambda _: {k: scores[k] for k in META_COLUMNS}
    )
    policy = TinyPolicy()
    runs = [tmp_path / "pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1", tmp_path / run_name()]
    for run, plan in zip(runs, (base, changed), strict=True):
        plan_path = tmp_path / f"{run.name}.npz"
        save_plan(plan_path, plan)
        cfg = make_config(
            run,
            PDETConfig(
                enabled=True,
                prompt_seed=1,
                data_seed=0,
                noise_seed=1,
                time_seed=1,
                order_plan=str(plan_path),
                audit_steps=20,
                diagnostics=True,
            ),
        )
        cfg.steps, cfg.batch_size, cfg.dataset.root = 20, 2, tmp_path / "data"
        expert = run == runs[1]
        cfg.to_dict = lambda expert=expert: {
            "seed": 0,
            "steps": 20,
            "policy": {"vlm_prompt_only": not expert, "train_action_expert": expert},
        }
        for rank in (0, 1):
            streams = PDETStreams(cfg, policy, accelerator(rank))
            for step in range(20):
                start = (step * 2 + rank) * 2
                ids = plan["order"][start : start + 2].tolist()
                streams.draw(
                    policy, {"action": torch.zeros(2, 50, 7)}, step, plan["index"][ids].tolist(), ids
                )
    monkeypatch.setattr(
        experiment,
        "check",
        lambda _: {
            "reference_run": str(runs[0]),
            "settings": {"direction": "easy_to_hard", "easy_percent": 70, "sample_seed": 0},
            "plan": changed["manifest"],
        },
    )
    monkeypatch.setattr(experiment, "check_checkpoint", lambda *_: None)
    monkeypatch.setattr(experiment, "assert_same_processors", lambda *_: None)
    report = experiment.audit(tmp_path, runs[1])
    assert report["audit"] == "inputs_and_recipe_passed"
    assert report["sample_pools"]["reference"] != report["sample_pools"]["curriculum"]
    assert report["reference"]["bank_sha256"] == report["curriculum"]["bank_sha256"]
    path = runs[1] / "pdet/inputs-rank1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["occurrence_noise_sha256"][0] = "0" * 64
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="bank changed"):
        experiment.audit(tmp_path, runs[1])
