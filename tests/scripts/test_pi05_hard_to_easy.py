"""Order-only intervention: preserve actual sample/flow pairs and launch recipe."""

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
import pytest
import torch
from safetensors.numpy import save_file

from lerobot.configs.pdet import PDETConfig
from lerobot.scripts import pi05_hard_to_easy as experiment, pi05_source_compare as sources
from lerobot.scripts.frameskip_scores import file_inventory, read_actions, source_inventory
from lerobot.utils.frameskip_curriculum import save_scores
from lerobot.utils.pdet import PDETStreams, sample_flow
from lerobot.utils.pdet_order import (
    META_COLUMNS,
    arrays_hash,
    create_plan,
    load_plan,
    read_frame_metadata,
    save_plan,
)
from tests.scripts.test_pi05_pdet import TinyPolicy, accelerator, make_config
from tests.scripts.test_pi05_source_compare import inputs  # noqa: F401 -- shared pytest fixture

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "examples/training/train_pi05_hard_to_easy.sh"
EVAL = ROOT / "run_eval_libero_hard_to_easy.sh"
RUN_NAME = "pi05-hard2easy-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1"
REF_NAME = "pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1"


def small_inputs():
    n = 24
    meta = {key: np.zeros(n, dtype=np.int64) for key in META_COLUMNS}
    meta.update(
        index=np.arange(n),
        frame_index=np.arange(n) % 6,
        episode_index=np.arange(n) // 6,
        task_index=np.arange(n) // 12,
    )
    scores = {**meta, "score": (np.arange(n) % 4).astype(float)}
    plan = create_plan(meta, steps=20, batch_size=2, world_size=2, data_seed=0, within_seed=0)
    manifest = {"arrays_sha256": arrays_hash(scores), "dataset_metadata_sha256": arrays_hash(meta)}
    return plan, scores, manifest


def test_sort_preserves_all_uses_task_slots_ties_and_global_rng():
    base, scores, manifest = small_inputs()
    before = copy.deepcopy(base)
    state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    changed = experiment.reorder(base, scores, manifest)
    np.testing.assert_equal(state, np.random.get_state())
    assert torch.equal(torch_state, torch.get_rng_state())
    assert not np.array_equal(base["order"], changed["order"])
    for key in META_COLUMNS:
        assert np.array_equal(changed[key], base[key])
        assert np.array_equal(before[key], base[key])
    assert np.array_equal(np.sort(changed["order"]), np.arange(len(base["order"])))
    assert np.array_equal(changed["task_index"][changed["order"]], base["task_index"])
    difficulty = scores["score"][base["index"]]
    for task in np.unique(base["task_index"]):
        slots = np.flatnonzero(base["task_index"] == task)
        old, new = base["order"][slots], changed["order"][slots]
        assert np.all(np.diff(difficulty[new]) <= 0)
        for value in np.unique(difficulty[new]):
            assert np.array_equal(old[difficulty[old] == value], new[difficulty[new] == value])
    assert base["manifest"] == before["manifest"]
    assert np.array_equal(base["order"], before["order"])


def test_rejects_uninformative_scores_and_mismatched_pool():
    base, scores, manifest = small_inputs()
    with pytest.raises(ValueError, match="did not change"):
        experiment.reorder(base, {**scores, "score": np.zeros_like(scores["score"])}, manifest)
    changed = copy.deepcopy(base)
    changed["index"][0] = (changed["index"][0] + 1) % 24
    with pytest.raises(ValueError, match="sample pool"):
        experiment.reorder(changed, scores, manifest)
    with pytest.raises(ValueError, match="dataset metadata"):
        experiment.reorder(base, scores, {**manifest, "dataset_metadata_sha256": "wrong"})


def test_same_sample_use_keeps_exact_noise_time_after_moving_steps_and_ranks():
    base, scores, manifest = small_inputs()
    changed = experiment.reorder(base, scores, manifest)
    config = PDETConfig(enabled=True, order_plan="plan", noise_seed=1, time_seed=1)
    policy = TinyPolicy()
    banks = []
    for plan in (base, changed):
        bank = {}
        for slot in range(0, len(plan["order"]), 2):
            ids = plan["order"][slot : slot + 2].tolist()
            flow = sample_flow(
                policy, {"action": torch.zeros(2, 50, 7)}, config, slot // 4, (slot // 2) % 2, ids
            )
            for i, use in enumerate(ids):
                bank[use] = (int(plan["index"][use]), flow["noise"][i], flow["time"][i])
        banks.append(bank)
    for use, (sample, noise, time) in banks[0].items():
        other = banks[1][use]
        assert sample == other[0] and torch.equal(noise, other[1]) and torch.equal(time, other[2])


@pytest.fixture
def prepared_inputs(inputs, tmp_path):  # noqa: F811 -- injected shared pytest fixture
    sources.prepare(**inputs)
    pair, data = inputs["output"], inputs["dataset_root"]
    reference = tmp_path / REF_NAME
    checkpoint = reference / "checkpoints/003000/pretrained_model"
    shutil.copytree(pair / "libero", checkpoint)
    config = sources.read_json(checkpoint / "config.json")
    config.update(
        vlm_prompt_only=True,
        num_vlm_prompt_tokens=2,
        num_prompt_tokens=0,
        training_stage="flow",
        next_action_pretrain_steps=0,
        next_action_bridge_steps=0,
        cabo_enabled=False,
        train_action_projections=False,
        use_peft=False,
    )
    (checkpoint / "config.json").write_text(json.dumps(config))
    save_file(
        {
            "model.vlm_prompt_tokens.weight": np.ones((2, 2048), dtype=np.float32),
            "model.prompt_tokens.weight": np.empty((0, 1024), dtype=np.float32),
        },
        checkpoint / "model.safetensors",
    )
    manifest = sources.read_json(inputs["reference_run"] / "pdet/manifest-rank0.json")
    manifest["source_pretrained_path"] = str(pair / "libero")
    (reference / "pdet").mkdir()
    (reference / "pdet/manifest-rank0.json").write_text(json.dumps(manifest))
    (checkpoint / "train_config.json").write_text(
        json.dumps(
            {
                "policy": {**config, "pretrained_path": str(pair / "libero")},
                "seed": 0,
                "steps": 3000,
                "batch_size": 32,
                "pdet": manifest["factors"],
            }
        )
    )
    metadata = read_frame_metadata(data)
    n = len(metadata["index"])
    scores = {"score": (np.arange(n) % 5).astype(float)}
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
    save_scores(scores_path, metadata, scores, provenance)
    return {
        "pair": pair,
        "scores_path": scores_path,
        "reference": reference,
        "output": tmp_path / "curriculum",
    }


def test_prepare_and_check_exact_plan_reuse_and_detect_resigned_wrong_order(prepared_inputs):
    args = prepared_inputs
    before = {
        p: sources.digest(p)
        for p in [args["scores_path"], args["pair"] / "D0-W0.npz", args["pair"] / "P1.safetensors"]
    }
    assert experiment.prepare(**args)["prepared"] == "passed"
    assert experiment.prepare(**args)["prepared"] == "reused"
    assert before == {p: sources.digest(p) for p in before}
    record = experiment.check(args["output"])
    assert len((args["output"] / "schedule.csv").read_text().splitlines()) == 3001
    plan_path = args["output"] / "H2E.npz"
    plan = load_plan(plan_path)
    slots = np.flatnonzero(plan["task_index"] == plan["task_index"][0])
    plan["order"][slots] = plan["order"][slots][::-1]
    plan["manifest"]["arrays_sha256"] = arrays_hash({k: plan[k] for k in (*META_COLUMNS, "order")})
    plan_path.unlink()
    save_plan(plan_path, plan)
    record["plan"] = plan["manifest"]
    (args["output"] / "curriculum.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="exact descending"):
        experiment.check(args["output"])


def test_training_command_only_changes_order_and_output_labels(prepared_inputs, tmp_path):
    args = prepared_inputs
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    env = {
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
    parsed = []
    for command in (
        ["bash", str(ROOT / "examples/training/train_pi05_source_compare.sh"), "libero"],
        ["bash", str(TRAIN)],
    ):
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        line = next(
            line for line in result.stdout.splitlines() if line.startswith("env CUDA_VISIBLE_DEVICES=")
        )
        tokens = shlex.split(line)
        assert "CUDA_VISIBLE_DEVICES=0,1" in tokens
        flags = dict(token[2:].split("=", 1) for token in tokens if token.startswith("--") and "=" in token)
        parsed.append(flags)
    differences = {
        key for key in parsed[0].keys() | parsed[1].keys() if parsed[0].get(key) != parsed[1].get(key)
    }
    assert differences == {"main_process_port", "pdet.order_plan", "output_dir", "job_name"}
    assert parsed[1]["pdet.order_plan"] == str(args["output"] / "H2E.npz")
    assert parsed[1]["policy.path"] == str(args["pair"] / "libero")
    assert parsed[1]["vlm_prompt_init_path"] == str(args["pair"] / "P1.safetensors")
    assert not (tmp_path / "runs").exists()
    (tmp_path / "runs" / RUN_NAME).mkdir(parents=True)
    result = subprocess.run(["bash", str(TRAIN)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "Output already exists" in result.stderr


def test_checkpoint_rejects_wrong_binding_or_keyframe_loss(prepared_inputs, tmp_path):
    args = prepared_inputs
    experiment.prepare(**args)
    run = tmp_path / RUN_NAME
    checkpoint = run / "checkpoints/003000/pretrained_model"
    shutil.copytree(args["reference"] / "checkpoints/003000/pretrained_model", checkpoint)
    path = checkpoint / "train_config.json"
    config = sources.read_json(path)
    config["pdet"].update(enabled=True, order_plan=str(args["output"] / "H2E.npz"))
    path.write_text(json.dumps(config))
    assert experiment.check_checkpoint(args["output"], run)["checkpoint"] == "passed"
    config["pdet"]["flow_binding"] = "slot"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="occurrence"):
        experiment.check_checkpoint(args["output"], run)
    config["pdet"]["flow_binding"] = "occurrence"
    config["keyframe_loss"] = {"enabled": True}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="loss weighting"):
        experiment.check_checkpoint(args["output"], run)


def test_audit_compares_actual_all_step_per_occurrence_bank(tmp_path, monkeypatch):
    base, scores, manifest = small_inputs()
    changed = experiment.reorder(base, scores, manifest)
    data = tmp_path / "data"
    # Isolate file access, retaining real stream generation and actual trace audit.
    monkeypatch.setattr(
        "lerobot.utils.pdet_order.read_frame_metadata", lambda _: {k: scores[k] for k in META_COLUMNS}
    )
    policy = TinyPolicy()
    runs = [tmp_path / REF_NAME, tmp_path / RUN_NAME]
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
        cfg.steps, cfg.batch_size, cfg.dataset.root = 20, 2, data
        for rank in (0, 1):
            streams = PDETStreams(cfg, policy, accelerator(rank))
            for step in range(20):
                start = (step * 2 + rank) * 2
                ids = plan["order"][start : start + 2].tolist()
                streams.draw(
                    policy, {"action": torch.zeros(2, 50, 7)}, step, plan["index"][ids].tolist(), ids
                )
    monkeypatch.setattr(experiment, "check", lambda _: {"reference_run": str(runs[0])})
    monkeypatch.setattr(experiment, "check_checkpoint", lambda *_: None)
    monkeypatch.setattr(experiment, "assert_same_processors", lambda *_: None)
    assert experiment.audit(tmp_path, runs[1])["audit"] == "inputs_and_recipe_passed"
    for rank in (0, 1):
        path = runs[1] / f"pdet/manifest-rank{rank}.json"
        record = sources.read_json(path)
        record["git_commit"], record["source_tree_sha256"] = "a" * 40, "b" * 64
        path.write_text(json.dumps(record))
    assert experiment.audit(tmp_path, runs[1])["code_revision"]["status"] == "different_not_certified"
    path = runs[1] / "pdet/inputs-rank1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["occurrence_noise_sha256"][0] = "0" * 64
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="bank changed"):
        experiment.audit(tmp_path, runs[1])


def test_evaluator_selects_correct_source_name_and_progress(tmp_path):
    repo, binaries = tmp_path / "repo", tmp_path / "bin"
    repo.mkdir()
    binaries.mkdir()
    (repo / EVAL.name).write_text(EVAL.read_text())
    (binaries / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
    (binaries / "python").chmod(0o755)
    (repo / "run_eval_libero_vlm_only_pdet_sweep.sh").write_text(
        '#!/usr/bin/env bash\nprintf "%s|%s|%s|%s|%s|%s|%s\\n" '
        '"$BASE_PRETRAINED_PATH" "$RUN_PREFIX" "$PDET_CASES" "$EVAL_SEED" "$EPISODES_PER_TASK" "$PROGRESS" "$1"\n'
    )
    result = subprocess.run(
        ["bash", str(repo / EVAL.name), "all"],
        env={
            "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
            "SOURCE_PAIR_DIR": str(tmp_path / "pair"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout.strip() == f"{tmp_path}/pair/libero|pi05-hard2easy-r1-libero-Bocc|1011|1000|10|true|all"
    )
