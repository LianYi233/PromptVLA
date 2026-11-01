"""Check seeded model identity and reuse of the original four-GPU PRO protocol."""

import json
import subprocess

import pytest

from examples.evaluation import eval_curriculum_seed_repeat_pro as evaluator
from lerobot.scripts import pi05_curriculum_seed_repeat as training
from tests.scripts import test_curriculum_pro_eval as fixtures


def test_names_match_training_and_preserve_original_eval_protocol(tmp_path):
    env = fixtures.env(tmp_path, RUN_NAME="old-A", TRAIN_SCOPE="prompt_only", REPEAT_ID="999")
    jobs, batch = evaluator.settings(env)
    trained, _ = training.build_jobs(env, [])
    assert [j["name"] for j in jobs] == [j["name"] for j in trained]
    assert [(j["cell"], j["gpu"]) for j in jobs] == [
        ("D1", "0"),
        ("D2", "1"),
        ("E1", "2"),
        ("E2", "3"),
        ("F1", "0"),
        ("F2", "1"),
    ]
    assert len({j["result"] for j in jobs}) == 6
    assert len({j["manifest"] for j in jobs}) == 1
    assert batch.name == "pro-batch-seed1000-ep10-dimsobject-swap-lan-task"
    for job in jobs:
        assert job["env"]["TRAIN_SCOPE"] == "joint"
        assert job["env"]["EVAL_SEED"] == "1000"
        assert job["env"]["EPISODES_PER_TASK"] == "10"
        assert job["checkpoint"].parent.name == "006000"
        assert job["result"].parent.name == f"libero060-all-{job['name']}-006000-pro-libero-all4-resume"


def test_custom_seeds_subset_and_separate_default_results():
    jobs, batch = evaluator.settings(
        {"SEED_SET_1": "4", "SEED_SET_2": "5", "CELLS": "E2 F1", "GPU_IDS": "3,2"}
    )
    assert [(j["cell"], j["training_seed"], j["gpu"]) for j in jobs] == [("E2", 5, "3"), ("F1", 4, "2")]
    assert "curriculum-def-seeds-4-5" in str(batch)
    assert all("curriculum-def-seeds-4-5" in str(j["result"]) for j in jobs)


@pytest.mark.parametrize(
    "env",
    [
        {"CELLS": "D E F"},
        {"CELLS": "D1 D1"},
        {"CELLS": ""},
        {"SEED_SET_1": "3"},
        {"SEED_SET_2": "1"},
        {"CHECKPOINT_STEP": "3000"},
        {"GPU_IDS": "0,0"},
    ],
)
def test_reject_invalid_selection(env):
    with pytest.raises(ValueError):
        evaluator.settings(env)


def test_checkpoint_seed_identity_is_checked_from_saved_config(tmp_path):
    jobs, _ = evaluator.settings(fixtures.env(tmp_path))
    for job in jobs:
        job["checkpoint"].mkdir(parents=True)
        seed = job["training_seed"]
        config = {
            "seed": seed,
            "steps": 6000,
            "batch_size": 32,
            "pdet": {
                "enabled": True,
                "flow_binding": "occurrence",
                "prompt_seed": seed,
                "data_seed": seed,
                "noise_seed": seed,
                "time_seed": seed,
            },
        }
        (job["checkpoint"] / "train_config.json").write_text(json.dumps(config))
    evaluator.check_training_identity(jobs)
    path = jobs[0]["checkpoint"] / "train_config.json"
    config = json.loads(path.read_text())
    config["pdet"]["noise_seed"] = 1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="D1: checkpoint"):
        evaluator.check_training_identity(jobs)


def test_summary_uses_seeded_names_and_rejects_other_checkpoint(tmp_path):
    jobs, batch = evaluator.settings(fixtures.env(tmp_path, CELLS="D1 E1 F1"))
    for job in jobs:
        fixtures.write_results(job)
    assert evaluator.baseline.summary(jobs, batch)
    saved = json.loads((batch / "summary.json").read_text())
    assert set(saved["reports"]) == {"D1", "E1", "F1"}
    assert all(r["overall"]["episodes"] == 1600 for r in saved["reports"].values())
    resume = jobs[0]["result"].with_name("resume_manifest.json")
    record = json.loads(resume.read_text())
    record["arguments"][0] = f"--policy.path={jobs[1]['checkpoint']}"
    resume.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="different checkpoint"):
        evaluator.baseline.read_result(jobs[0])


def test_launcher_syntax():
    subprocess.run(
        ["bash", "-n", str(evaluator.baseline.ROOT / "run_eval_libero_curriculum_seed_repeat_pro.sh")],
        check=True,
    )
