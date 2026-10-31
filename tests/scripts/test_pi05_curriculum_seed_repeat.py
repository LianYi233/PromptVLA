"""Seed identity, matched curriculum inputs, native launch arguments and queue safety."""

import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from lerobot.configs.pdet import PDETConfig
from lerobot.scripts import pi05_curriculum_seed_repeat as suite
from lerobot.utils.pdet import sample_flow
from lerobot.utils.pdet_order import META_COLUMNS, arrays_hash, load_plan, save_plan
from lerobot.utils.prompt_initialization import prompt_hash, write_prompt


def jobs_at(tmp_path, **changes):
    env = {
        key: str(tmp_path / key.lower())
        for key in (
            "OUTPUT_ROOT",
            "LOG_ROOT",
            "SOURCE_PAIR_DIR",
            "SCORES_PATH",
            "DATASET_ROOT",
            "TOKENIZER_PATH",
            "SEED_WORK_DIR",
        )
    }
    return suite.build_jobs(env | changes, ["--wandb.enable=false"])


def inputs():
    n = 72
    meta = {key: np.zeros(n, dtype=np.int64) for key in META_COLUMNS}
    meta.update(
        index=np.arange(n),
        task_index=np.arange(n) // 24,
        episode_index=np.arange(n) // 8,
        frame_index=np.arange(n) % 8,
    )
    scores = {**meta, "score": (np.arange(n) % 24).astype(float)}
    return scores, {"dataset_metadata_sha256": arrays_hash(meta), "arrays_sha256": arrays_hash(scores)}


def make_plans(seed):
    return suite.plans_for_seed(*inputs(), seed, parent_steps=30, steps=60, batch=2, world=2)


def test_six_jobs_preserve_budget_and_really_set_seeds(tmp_path, monkeypatch):
    for key in suite.runner.CONTROLLED:
        monkeypatch.setenv(key, "stale")
    jobs, logs = jobs_at(tmp_path)
    assert [j["cell"] for j in jobs] == ["D1", "D2", "E1", "E2", "F1", "F2"]
    assert len({j["run"] for j in jobs}) == 6
    assert logs.name == "curriculum-def-seeds-2-3"
    for job in jobs:
        seed = job["seed"]
        env = suite.runner.child_env(job)
        assert env["FLOW_STEPS"] == "6000" and env["SAVE_ONLY_FINAL"] == "true"
        assert env["BATCH_SIZE"] == "32" and env["NUM_PROCESSES"] == "2"
        assert env["TRAIN_ACTION_EXPERT"] == "true" and env["FREEZE_VLM_PROMPT"] == "false"
        assert env["PRETRAINED_PATH"].endswith("source_pair_dir/libero")
        assert env["VLM_PROMPT_TOKENS"] == "2"
        assert env["GPU_IDS"] == ("0,1" if job["block"] == 1 else "2,3")
        assert env["SAMPLE_SEED"] == str(seed)
        assert "PROMPT_EXPORT_PATH" not in env and "PDET_CASES" not in env
        assert job["command"][2:4] == ["base", str(seed)]
        for factor in ("prompt", "data", "noise", "time"):
            assert f"--pdet.{factor}_seed={seed}" in job["command"]
        assert "--prompt_rank.seed=2026" in job["command"]
        assert "--pdet.flow_binding=occurrence" in job["command"]
        assert f"-P{seed}-D{seed}-E{seed}-T{seed}" in job["name"]


@pytest.mark.parametrize(
    "bad",
    [
        {"SEED_SET_1": "1"},
        {"SEED_SET_1": "3"},
        {"SEED_SET_2": str(2**32)},
        {"GPU_IDS_A": "0"},
        {"GPU_IDS_B": "1,2"},
        {"PORT_A": "29593"},
    ],
)
def test_rejects_old_duplicate_seeds_and_bad_gpu_pairs(tmp_path, bad):
    with pytest.raises(ValueError):
        jobs_at(tmp_path, **bad)


def test_pools_and_flow_stay_matched_but_new_seed_changes_both(tmp_path):
    state = torch.get_rng_state().clone()
    plans = make_plans(2)
    repeated = make_plans(2)
    other = make_plans(3)
    assert torch.equal(state, torch.get_rng_state())
    assert len({p["manifest"]["instance_pool_sha256"] for p in plans.values()}) == 1
    assert len({p["manifest"]["task_schedule_sha256"] for p in plans.values()}) == 1
    assert len({arrays_hash({"order": p["order"]}) for p in plans.values()}) == 3
    assert plans["D"]["manifest"]["instance_pool_sha256"] != other["D"]["manifest"]["instance_pool_sha256"]
    for letter, plan in plans.items():
        assert plan["manifest"] == repeated[letter]["manifest"]
        assert plan["manifest"]["difficulty_sampling"]["easy_percent"] == 50
        assert plan["manifest"]["difficulty_sampling"]["sample_seed"] == 2
        save_plan(tmp_path / f"{letter}.npz", plan)
        assert load_plan(tmp_path / f"{letter}.npz")["manifest"] == plan["manifest"]
        scores, _ = inputs()
        values = scores["score"][plan["index"][plan["order"]]]
        tasks = plan["task_index"][plan["order"]]
        if letter != "F":
            for task in np.unique(tasks):
                changes = np.diff(values[tasks == task])
                assert (changes <= 0).all() if letter == "D" else (changes >= 0).all()
    policy = SimpleNamespace(
        prepare_action=lambda batch: torch.zeros((len(batch), 3, 7)),
        model=SimpleNamespace(sample_time=lambda n, device: torch.distributions.Beta(1.5, 1).sample((n,))),
    )
    banks = []
    for plan in plans.values():
        cfg = PDETConfig(
            enabled=True, order_plan="plan", noise_seed=2, time_seed=2, flow_binding="occurrence"
        )
        bank = {}
        for offset in range(0, len(plan["order"]), 2):
            ids = plan["order"][offset : offset + 2]
            draws = sample_flow(policy, ids, cfg, offset // 2, 0, occurrences=ids)
            for i, occurrence in enumerate(ids):
                bank[int(occurrence)] = (draws["noise"][i].numpy().tobytes(), draws["time"][i].item())
        banks.append(bank)
    assert banks[0] == banks[1] == banks[2]
    cfg.noise_seed = cfg.time_seed = 3
    changed = sample_flow(policy, [0], cfg, 0, 0, occurrences=[0])
    assert changed["noise"][0].numpy().tobytes() != banks[0][0][0]


def write_init(job):
    prompt = np.full((2, 2048), job["seed"], dtype=np.float32)
    metadata = {
        "schema_version": 1,
        "source_step": 0,
        "source_training_seed": job["seed"],
        "num_vlm_prompt_tokens": 2,
        "prompt_sha256": prompt_hash(prompt),
        "source_kind": "regenerated_training_initialization",
        "source_prompt_init_path": None,
        "source_pretrained_path": job["env"]["PRETRAINED_PATH"],
    }
    write_prompt(job["env"]["PROMPT_INIT_PATH"], prompt, metadata)
    return metadata


def test_native_launcher_has_expected_training_and_save_arguments(tmp_path):
    jobs, _ = jobs_at(tmp_path)
    for job in jobs[:2]:
        env = suite.runner.child_env(job, dry_run=True)
        source, data, tokenizer = [
            Path(env[key]) for key in ("PRETRAINED_PATH", "DATASET_ROOT", "TOKENIZER_PATH")
        ]
        for path in (source, data / "meta", tokenizer):
            path.mkdir(parents=True, exist_ok=True)
        (source / "config.json").write_text(
            json.dumps(
                {
                    "type": "pi05",
                    "paligemma_variant": "gemma_2b",
                    "action_expert_variant": "gemma_300m",
                }
            )
        )
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            (source / name).write_text("{}")
        save_file({"model.backbone": np.zeros((1,), dtype=np.float32)}, source / "model.safetensors")
        (data / "meta/stats.json").write_text(
            json.dumps(
                {
                    key: {"q01": [0] * n, "q99": [1] * n}
                    for key, n in (("action", 7), ("observation.state", 8))
                }
            )
        )
        write_init(job)
        result = subprocess.run(job["command"], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        argv = shlex.split(result.stdout.strip().splitlines()[-1])
        for expected in (
            "--steps=6000",
            "--batch_size=32",
            "--save_freq=6000",
            "--save_steps=[]",
            "--policy.ntk_save_stage_snapshots=false",
            "--policy.train_action_expert=true",
            "--policy.vlm_prompt_only=false",
            "--policy.num_vlm_prompt_tokens=2",
            "--policy.optimizer_lr=2.5e-5",
            "--num_processes=2",
            "--mixed_precision=bf16",
        ):
            assert expected in argv
        assert argv[-len(job["command"][4:]) :] == job["command"][4:]
        assert suite.validate_init(job)["source_training_seed"] == job["seed"]
        assert not Path(job["run"]).exists()


def test_wrong_seed_prompt_is_rejected(tmp_path):
    jobs, _ = jobs_at(tmp_path)
    write_init(jobs[0])
    jobs[0]["seed"] = 3
    with pytest.raises(ValueError, match="fresh native initialization"):
        suite.validate_init(jobs[0])


def test_round_callback_and_resume_skip_with_real_children(tmp_path):
    jobs, _ = jobs_at(tmp_path)
    jobs = jobs[:2]
    for job in jobs:
        job["command"] = [sys.executable, "-c", "print('child finished')"]
    statuses = {job["cell"]: {"state": "pending"} for job in jobs}
    checked = []
    suite.runner.run_round(
        jobs,
        statuses,
        tmp_path / "status.json",
        lambda _: None,
        checkpoint_validator=lambda j: checked.append(j["cell"]),
        poll_seconds=0.01,
    )
    assert set(checked) == {"D1", "D2"}
    assert all(v["state"] == "complete" for v in statuses.values())
    checked.clear()
    suite.runner.run_round(
        jobs,
        statuses,
        tmp_path / "status.json",
        lambda _: None,
        checkpoint_validator=lambda j: checked.append(j["cell"]),
        poll_seconds=0.01,
    )
    assert not checked


def test_failed_child_prevents_later_jobs(tmp_path):
    jobs, _ = jobs_at(tmp_path)
    jobs = jobs[:2]
    jobs[0]["command"] = [sys.executable, "-c", "raise SystemExit(7)"]
    jobs[1]["command"] = [sys.executable, "-c", "import time; time.sleep(10)"]
    statuses = {job["cell"]: {"state": "pending"} for job in jobs}
    with pytest.raises(RuntimeError, match="exit=7"):
        suite.runner.run_round(
            jobs,
            statuses,
            tmp_path / "status.json",
            lambda _: None,
            checkpoint_validator=lambda _: None,
            poll_seconds=0.01,
        )
    assert statuses["D1"]["state"] == "failed"
    assert statuses["D2"]["state"] == "interrupted"


def test_suite_audit_rejects_cross_order_or_cross_seed_contamination(tmp_path, monkeypatch):
    jobs, _ = jobs_at(tmp_path)
    records = {}
    for job in jobs:
        seed = job["seed"]
        records[job["cell"]] = (
            {
                "global_seed": seed,
                "prompt_sha256": f"prompt{seed}",
                "experiment": job["block"],
                "pool": f"pool{seed}",
                "tasks": f"tasks{seed}",
                "recipe": {"seed": seed, "steps": 6000},
                "factors": {"prompt_seed": seed, "data_seed": seed},
            },
            {"bank_sha256": f"bank{seed}", "order_sha256": job["cell"], "run": job["run"]},
        )
    monkeypatch.setattr(suite, "check_checkpoint", lambda j: copy.deepcopy(records[j["cell"]]))
    assert suite.audit_suite(jobs)["audit"] == "passed"
    records["E1"][1]["bank_sha256"] = "wrong"
    with pytest.raises(ValueError, match="realized E/T bank"):
        suite.audit_suite(jobs)
    records["E1"][1]["bank_sha256"] = "bank2"
    for cell in ("D2", "E2", "F2"):
        records[cell][0]["recipe"]["steps"] = 3000
    with pytest.raises(ValueError, match="Non-seed recipe"):
        suite.audit_suite(jobs)


def test_init_export_uses_native_zero_update_path_and_reuses_verified_file(tmp_path, monkeypatch):
    jobs, directory = jobs_at(tmp_path)
    directory.mkdir(parents=True)
    job = jobs[0]
    calls = []

    def export(command, **kwargs):
        calls.append(command)
        env = kwargs["env"]
        assert env["TRAIN_ACTION_EXPERT"] == "false"
        assert "PROMPT_INIT_PATH" not in env
        assert env["PROMPT_EXPORT_PATH"] == job["env"]["PROMPT_INIT_PATH"]
        assert "--wandb.enable=false" in command
        assert not any(arg.startswith(("--pdet.", "--prompt_rank.")) for arg in command)
        write_init(job)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", export)
    suite.ensure_init(job, directory, dry=True)
    assert not calls
    suite.ensure_init(job, directory)
    suite.ensure_init(job, directory)
    assert len(calls) == 1


def test_one_invocation_runs_three_rounds_without_continuing_previous_weights(tmp_path, monkeypatch):
    jobs, directory = jobs_at(tmp_path)
    events = tmp_path / "events"
    script = tmp_path / "fake_train.py"
    script.write_text(
        "import os, time\n"
        "def event(kind):\n"
        "    with open(os.environ['EVENTS'], 'a') as f: f.write(kind + ' ' + os.environ['CELL'] + '\\n')\n"
        "if os.environ['DRY_RUN'] == 'true': event('PREFLIGHT')\n"
        "else:\n"
        "    event('START')\n"
        "    time.sleep(0.15)\n"
        "    event('END')\n"
    )
    for job in jobs:
        job["command"] = [sys.executable, str(script)]
        job["env"].update(EVENTS=str(events), CELL=job["cell"])
    monkeypatch.setattr(suite, "build_jobs", lambda *_: (jobs, directory))
    monkeypatch.setattr(suite, "prepare", lambda _: {})
    monkeypatch.setattr(suite, "ensure_init", lambda *_: None)
    monkeypatch.setattr(suite, "check_checkpoint", lambda _: ({}, {}))
    monkeypatch.setattr(suite, "audit_suite", lambda _: {"audit": "passed"})
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setattr(sys, "argv", ["seed-repeat", "run"])
    suite.main()
    rows = events.read_text().splitlines()
    assert rows[:6] == [f"PREFLIGHT {j['cell']}" for j in jobs]
    for round_id in (1, 2, 3):
        selected = [j["cell"] for j in jobs if j["round"] == round_id]
        assert max(rows.index(f"START {c}") for c in selected) < min(rows.index(f"END {c}") for c in selected)
        if round_id < 3:
            later = [j["cell"] for j in jobs if j["round"] == round_id + 1]
            assert max(rows.index(f"END {c}") for c in selected) < min(
                rows.index(f"START {c}") for c in later
            )
    assert {v["state"] for v in json.loads((directory / "status.json").read_text()).values()} == {"complete"}
