"""Real subprocess scheduling and fixed ablation recipes, without GPU training."""

# ruff: noqa: F811 -- shared fixtures

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.scripts import pi05_curriculum_ablation as suite, pi05_ratio_curriculum as curriculum
from tests.scripts.test_pi05_hard_to_easy import prepared_inputs  # noqa: F401
from tests.scripts.test_pi05_ratio_curriculum import launcher_env
from tests.scripts.test_pi05_source_compare import inputs  # noqa: F401


def test_eight_cells_have_unique_outputs_fixed_budget_scope_and_barriers():
    jobs = suite.build_jobs({}, ["--wandb.mode=offline"])
    assert [job["cell"] for job in jobs] == list("BCDFEAGH")
    assert len({job["run"] for job in jobs}) == 8
    for job, (round_id, lane, _, direction, easy, scope) in zip(jobs, suite.CELLS, strict=True):
        env = job["env"]
        assert job["round"] == round_id and env["GPU_IDS"] == ("0,1", "2,3")[lane]
        assert env["FLOW_STEPS"] == "6000" and env["REPEAT_ID"] == "2"
        assert env["SAVE_ONLY_FINAL"] == "true" and env["BATCH_SIZE"] == "32"
        assert env["TRAIN_SCOPE"] == scope and env["EASY_PERCENT"] == str(easy)
        assert env["CURRICULUM_DIRECTION"] == direction
        assert curriculum.scope_from_run(job["run"]) == scope
    assert len({job["env"]["ORDER_DIR"] for job in jobs[-3:]}) == 1
    assert jobs[-2]["name"].endswith("frozenp-aexpert-r2-libero-Bocc-base-vlm2-P1-D0-E1-T1")
    assert jobs[-1]["name"].endswith("promptonly-r2-libero-Bocc-base-vlm2-P1-D0-E1-T1")


@pytest.mark.parametrize(
    "overrides",
    [
        {"GPU_IDS_A": "0"},
        {"GPU_IDS_B": "1,2"},
        {"PORT_A": "29589"},
        {"PORT_B": "65536"},
        {"REPEAT_ID": "0"},
        {"FLOW_STEPS": "0"},
    ],
)
def test_invalid_resources_rejected_before_launch(overrides):
    with pytest.raises(ValueError):
        suite.build_jobs(overrides, [])


@pytest.mark.parametrize("scope", curriculum.SCOPES)
def test_real_scope_launcher_command_and_final_only(prepared_inputs, tmp_path, scope):
    env = launcher_env(prepared_inputs, tmp_path) | {
        "TRAIN_SCOPE": scope,
        "CURRICULUM_DIRECTION": "random",
        "EASY_PERCENT": "30",
        "REPEAT_ID": "2",
    }
    result = subprocess.run(["bash", str(suite.TRAIN)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("env CUDA_VISIBLE_DEVICES="))
    flags = dict(t[2:].split("=", 1) for t in shlex.split(line) if t.startswith("--") and "=" in t)
    assert flags["steps"] == flags["save_freq"] == flags["pdet.audit_steps"] == "6000"
    assert flags["save_steps"] == "[]" and flags["policy.ntk_save_stage_snapshots"] == "false"
    assert flags.get("policy.train_action_expert", "false") == str(scope != "prompt_only").lower()
    assert flags.get("policy.freeze_vlm_prompt", "false") == str(scope == "expert_only").lower()
    assert flags["policy.vlm_prompt_only"] == str(scope == "prompt_only").lower()
    assert flags["policy.num_vlm_prompt_tokens"] == "2"
    assert flags["policy.path"] == str(prepared_inputs["pair"] / "libero")
    assert flags["vlm_prompt_init_path"] == str(prepared_inputs["pair"] / "P1.safetensors")
    assert flags["output_dir"].endswith(curriculum.run_name("random", 30, scope=scope))
    assert not Path(flags["output_dir"]).exists()


@pytest.fixture
def fake_suite(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("REPEAT_ID", "2")
    monkeypatch.setenv("FLOW_STEPS", "6000")
    monkeypatch.setattr(sys, "argv", ["ablation", "run"])
    events = tmp_path / "events"
    script = tmp_path / "trainer.py"
    script.write_text("""
import os, sys, time
from pathlib import Path
cell = os.environ["CELL"]
def record(event):
    with open(os.environ["EVENTS"], "a") as f:
        f.write(event + " " + cell + "\\n")
if os.environ["DRY_RUN"] == "true":
    record("PREFLIGHT")
    sys.exit(0)
record("START")
time.sleep(.04 if cell == os.environ.get("FAIL_CELL") else (10 if os.environ.get("FAIL_CELL") else .15))
if cell == os.environ.get("FAIL_CELL"):
    sys.exit(3)
record("END")
""")
    jobs = suite.build_jobs(dict(os.environ), [])
    for job in jobs:
        job["command"] = [sys.executable, str(script)]
        job["env"].update(CELL=job["cell"], EVENTS=str(events))
    monkeypatch.setattr(suite, "build_jobs", lambda *_: jobs)
    monkeypatch.setattr(suite, "checkpoint_complete", lambda job: {"complete": True})
    monkeypatch.setattr(suite, "check_current_runtime", lambda *_: None)
    monkeypatch.setattr(
        suite, "audit_suite", lambda jobs: {"audit": "passed", "cells": [j["cell"] for j in jobs]}
    )
    return jobs, events, tmp_path / "logs/curriculum-ablation-r2-steps6000"


def test_one_invocation_preflights_every_run_then_respects_all_four_barriers(fake_suite):
    jobs, events, directory = fake_suite
    suite.main()
    rows = events.read_text().splitlines()
    assert rows[:8] == [f"PREFLIGHT {job['cell']}" for job in jobs]
    for round_id in range(1, 5):
        cells = [job["cell"] for job in jobs if job["round"] == round_id]
        # Both jobs overlap, rather than running eight experiments serially.
        assert max(rows.index(f"START {cell}") for cell in cells) < min(
            rows.index(f"END {cell}") for cell in cells
        )
        if round_id > 1:
            previous = [job["cell"] for job in jobs if job["round"] == round_id - 1]
            assert min(rows.index(f"START {cell}") for cell in cells) > max(
                rows.index(f"END {cell}") for cell in previous
            )
    assert {s["state"] for s in json.loads((directory / "status.json").read_text()).values()} == {"complete"}
    assert json.loads((directory / "audit.json").read_text())["audit"] == "passed"


def test_failure_stops_sibling_and_all_later_rounds(fake_suite):
    jobs, events, directory = fake_suite
    for job in jobs:
        job["env"]["FAIL_CELL"] = "C"
    with pytest.raises(RuntimeError, match="Cell C failed"):
        suite.main()
    rows = events.read_text().splitlines()
    assert {r.split()[1] for r in rows if r.startswith("START")} == {"B", "C"}
    states = json.loads((directory / "status.json").read_text())
    assert states["C"]["state"] == "failed"
    assert states["B"]["state"] == "interrupted"
    assert all(states[c]["state"] == "pending" for c in "DFEAGH")
    assert not (directory / "audit.json").exists()


def test_existing_completed_outputs_are_skipped_without_retraining(fake_suite):
    jobs, events, directory = fake_suite
    for job in jobs[:2]:
        Path(job["run"]).mkdir(parents=True)
    suite.main()
    rows = events.read_text().splitlines()
    assert not any(r.endswith((" B", " C")) for r in rows)
    assert all(s["state"] == "complete" for s in json.loads((directory / "status.json").read_text()).values())


def test_partial_output_stops_before_launch(fake_suite, monkeypatch):
    jobs, events, directory = fake_suite
    Path(jobs[0]["run"]).mkdir(parents=True)

    def incomplete(_):
        raise ValueError("Incomplete existing run")

    monkeypatch.setattr(suite, "checkpoint_complete", incomplete)
    with pytest.raises(ValueError, match="Incomplete"):
        suite.main()
    assert not events.exists()


def test_old_code_is_not_silently_mixed_into_a_new_suite(tmp_path):
    recorded = {"git_commit": "old", "source_tree_sha256": "same"}
    path = tmp_path / "pdet"
    path.mkdir()
    (path / "manifest-rank0.json").write_text(json.dumps(recorded))
    suite.check_current_runtime({"run": str(tmp_path)}, recorded)
    with pytest.raises(ValueError, match="Missing/invalid recorded commit"):
        suite.check_current_runtime({"run": str(tmp_path)}, {**recorded, "git_commit": "new"})


def test_dry_run_never_launches_training(fake_suite, monkeypatch):
    jobs, events, directory = fake_suite
    monkeypatch.setenv("DRY_RUN", "true")
    suite.main()
    assert events.read_text().splitlines() == [f"PREFLIGHT {job['cell']}" for job in jobs]
    assert not (directory / "audit.json").exists()


def test_cross_run_audit_checks_realized_noise_bank_and_module_controls(tmp_path, monkeypatch):
    import torch

    from lerobot.configs.pdet import PDETConfig
    from lerobot.utils.pdet import PDETStreams
    from lerobot.utils.pdet_order import META_COLUMNS, save_plan
    from tests.scripts.test_pi05_hard_to_easy import small_inputs
    from tests.scripts.test_pi05_pdet import TinyPolicy, accelerator, make_config

    base, scores, manifest = small_inputs()
    monkeypatch.setattr(
        "lerobot.utils.pdet_order.read_frame_metadata", lambda _: {k: scores[k] for k in META_COLUMNS}
    )
    monkeypatch.setattr(curriculum, "check_checkpoint", lambda *_: None)
    jobs = suite.build_jobs({"OUTPUT_ROOT": str(tmp_path / "runs"), "FLOW_STEPS": "20"}, [])
    policy = TinyPolicy()
    plans = {}
    for job in jobs:
        env = job["env"]
        key = env["CURRICULUM_DIRECTION"], int(env["EASY_PERCENT"])
        if key not in plans:
            plan = curriculum.make_plan(
                base, scores, manifest, direction=key[0], easy_percent=key[1], steps=20
            )
            plan_path = tmp_path / f"{key[0]}-{key[1]}.npz"
            save_plan(plan_path, plan)
            plans[key] = plan, plan_path
        plan, plan_path = plans[key]
        cfg = make_config(
            Path(job["run"]),
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
        flags = curriculum.scope_flags(env["TRAIN_SCOPE"])
        cfg.to_dict = lambda flags=flags: {"seed": 0, "steps": 20, "policy": dict(flags)}
        for rank in (0, 1):
            streams = PDETStreams(cfg, policy, accelerator(rank))
            for step in range(20):
                start = (step * 2 + rank) * 2
                ids = plan["order"][start : start + 2].tolist()
                streams.draw(
                    policy, {"action": torch.zeros(2, 50, 7)}, step, plan["index"][ids].tolist(), ids
                )
    assert suite.audit_suite(jobs)["audit"] == "passed"
    path = Path(jobs[-1]["run"]) / "pdet/inputs-rank1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["occurrence_noise_sha256"][0] = "0" * 64
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="noise/time bank"):
        suite.audit_suite(jobs)
