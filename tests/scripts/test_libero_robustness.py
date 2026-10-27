"""Benchmark isolation, actual commands, perturbed instructions and complete accounting."""

import json
import os
import shlex
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from lerobot.scripts import libero_robustness as robust
from tests.scripts import test_pi05_vlm_only as fixtures

ROOT = Path(__file__).resolve().parents[2]


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def fork(tmp_path):
    root = tmp_path / "test PRO"
    package = root / "libero/libero"
    write(package / "__init__.py", "")
    write(package / "assets/scenes/libero_tabletop_base_style.xml", "<mujoco/>")
    write(package / "envs/__init__.py", "")
    write(
        package / "envs/bddl_utils.py",
        "import json\nfrom pathlib import Path\ndef get_problem_info(path):\n    return json.loads(Path(path).read_text())\n",
    )
    write(
        package / "benchmark/__init__.py",
        """from collections import namedtuple
Task = namedtuple("Task", "name language problem_folder bddl_file init_states_file")
def get_benchmark_dict():
    result = {}
    for dim in ("object", "swap", "lan", "task", "env"):
        name = "libero_goal_" + dim
        def factory(task_order_index=0, name=name):
            assert task_order_index == 0
            class Suite:
                pass
            suite = Suite()
            suite.tasks = [Task(f"original_{i}", "OLD FILENAME LANGUAGE", name, f"original_{i}.bddl", f"original_{i}.pruned_init") for i in range(10)]
            return suite
        result[name] = factory
    return result
""",
    )
    for dim in robust.PRO_DIMENSIONS:
        for i in range(10):
            write(
                package / f"bddl_files/libero_goal_{dim}/original_{i}.bddl",
                json.dumps({"language_instruction": f"PERTURBED {dim} {i}"}),
            )
            write(package / f"init_files/libero_goal_{dim}/original_{i}.pruned_init", "fixture state")
    return root


@pytest.fixture(autouse=True)
def isolated_modules(monkeypatch):
    saved = {k: v for k, v in sys.modules.items() if k == "libero" or k.startswith("libero.")}
    for key in saved:
        del sys.modules[key]
    monkeypatch.setenv("LIBERO_CONFIG_PATH", "unused-test-config")
    yield
    for key in list(sys.modules):
        if key == "libero" or key.startswith("libero."):
            del sys.modules[key]
    sys.modules.update(saved)


def test_suite_selection_counts_and_plus_one_trial_only(tmp_path):
    assert len(robust.selected_suites("pro", "all")) == 20
    assert sum(robust.PLUS_COUNTS.values()) == 10030
    assert robust.selected_suites("pro", "libero_goal", "lan,task") == ["libero_goal_lan", "libero_goal_task"]
    for kind, mode, dimensions in (
        ("pro", "all", "object,object"),
        ("plus", "all", "env"),
        ("pro", "bad", "all"),
    ):
        with pytest.raises(ValueError):
            robust.selected_suites(kind, mode, dimensions)
    with pytest.raises(ValueError, match="PLUS requires 1"):
        robust.build_manifest(tmp_path, tmp_path, "plus", "all", "all", 10, 1000)


def test_preflight_rejects_changed_resources_and_reads_bddl_language(fork, tmp_path):
    directory = tmp_path / "protocol"
    result = robust.prepare(fork, directory, "pro", "libero_goal", "all", 10, 1000)
    assert len(result["tasks"]) == 50 and result["expected_episodes"] == 500
    assert result["tasks"][0]["instruction"] == "PERTURBED object 0"
    assert robust.prepare(fork, directory, "pro", "libero_goal", "all", 10, 1000) == result
    path = fork / "libero/libero/bddl_files/libero_goal_lan/original_0.bddl"
    path.write_text(json.dumps({"language_instruction": "CHANGED"}))
    with pytest.raises(ValueError, match="changed"):
        robust.prepare(fork, directory, "pro", "libero_goal", "all", 10, 1000)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="task resource"):
        robust.prepare(fork, directory, "pro", "libero_goal", "all", 10, 1000)


def test_plus_virtual_bddl_and_layout_init_paths(tmp_path):
    task = SimpleNamespace(
        problem_folder="libero_goal",
        bddl_file="pick_language_1_view_0_initstate_2.bddl",
        init_states_file="pick_language_1_view_0_initstate_2.pruned_init",
    )
    bddl = tmp_path / "bddl_files/libero_goal/pick_language_1.bddl"
    initial = tmp_path / "init_files/libero_goal/pick.pruned_init"
    write(bddl, "language perturbation")
    write(initial, "state")
    assert robust.task_files(task, tmp_path, "plus") == (bddl, initial)
    task.bddl_file = "pick_add_1.bddl"
    task.init_states_file = "pick_add_1.pruned_init"
    bddl = tmp_path / "bddl_files/libero_goal/pick_add_1.bddl"
    initial = tmp_path / "init_files/libero_newobj/libero_goal/pick_add_1.pruned_init"
    write(bddl, "layout")
    write(initial, "flat state")
    assert robust.task_files(task, tmp_path, "plus") == (bddl, initial)


def test_plus_coverage_and_categories_join_by_name_not_off_by_one_id(fork, tmp_path, monkeypatch):
    package = fork / "libero/libero"
    write(
        package / "benchmark/__init__.py",
        """from collections import namedtuple
Task = namedtuple("Task", "name language problem_folder bddl_file init_states_file")
def get_benchmark_dict():
    def factory(task_order_index=0):
        class Suite:
            pass
        suite = Suite()
        suite.tasks = [Task(n, "WRONG filename", "libero_goal", n + ".bddl", n + ".pruned_init")
                       for n in ["pick_view_0_initstate_0", "pick_language_1"]]
        return suite
    return {"libero_goal": factory}
""",
    )
    write(package / "bddl_files/libero_goal/pick.bddl", json.dumps({"language_instruction": "pick object"}))
    write(
        package / "bddl_files/libero_goal/pick_language_1.bddl",
        json.dumps({"language_instruction": "grasp object"}),
    )
    write(package / "init_files/libero_goal/pick.pruned_init", "initial")
    write(
        package / "benchmark/task_classification.json",
        json.dumps(
            {
                "libero_goal": [
                    {"id": 1, "name": "pick_language_1", "category": "Language Instructions"},
                    {"id": 2, "name": "pick_view_0_initstate_0", "category": "Camera Viewpoints"},
                ]
            }
        ),
    )
    monkeypatch.setitem(robust.PLUS_COUNTS, "libero_goal", 2)
    result = robust.prepare(fork, tmp_path / "protocol", "plus", "libero_goal", "all", 1, 1000)
    assert result["expected_episodes"] == 2
    assert [r["category"] for r in result["tasks"]] == ["Camera Viewpoints", "Language Instructions"]
    assert [r["instruction"] for r in result["tasks"]] == ["pick object", "grasp object"]
    monkeypatch.setitem(robust.PLUS_COUNTS, "libero_goal", 2591)
    with pytest.raises(ValueError, match="task coverage"):
        robust.prepare(fork, tmp_path / "another", "plus", "libero_goal", "all", 1, 1000)


def test_runtime_changes_instruction_not_virtual_path_and_resets_task_rng(fork, tmp_path, monkeypatch):
    import torch

    import lerobot.envs

    directory = tmp_path / "protocol"
    robust.prepare(fork, directory, "pro", "libero_goal", "all", 1, 1000)
    from libero.libero import benchmark

    env_module = types.ModuleType("lerobot.envs.libero")
    env_module._get_suite = lambda name: benchmark.get_benchmark_dict()[name]()
    env_module.TASK_SUITE_MAX_STEPS = {}
    monkeypatch.setitem(sys.modules, "lerobot.envs.libero", env_module)
    monkeypatch.setattr(lerobot.envs, "libero", env_module, raising=False)

    def original_one(group, task_id, env, **kwargs):
        return torch.randn(3), kwargs

    evaluator = SimpleNamespace(run_one=original_one)
    robust.install(evaluator, directory / "manifest.json")
    task = env_module._get_suite("libero_goal_lan").tasks[0]
    assert task.language == "PERTURBED lan 0"
    assert task.bddl_file == "original_0.bddl"
    assert env_module.TASK_SUITE_MAX_STEPS["libero_goal_lan"] == 300
    first, kwargs = evaluator.run_one(
        "libero_goal_lan", 0, None, videos_dir="would-write", max_episodes_rendered=10
    )
    torch.randn(100)
    second, _ = evaluator.run_one("libero_goal_lan", 0, None)
    assert torch.equal(first, second)
    assert kwargs["videos_dir"] is None and kwargs["max_episodes_rendered"] == 0


def test_summary_uses_all_tasks_and_rejects_missing_duplicate_or_empty(fork, tmp_path):
    directory = tmp_path / "protocol"
    manifest = robust.prepare(fork, directory, "pro", "libero_goal", "all", 2, 1000)
    path = tmp_path / "eval_info.json"
    rows = [
        {"task_group": t["suite"], "task_id": t["task_id"], "metrics": {"successes": [True, False]}}
        for t in manifest["tasks"]
    ]
    write(path, json.dumps({"per_task": rows}))
    report = robust.summarize(directory / "manifest.json", path)
    assert report["overall"] == {"episodes": 100, "successes": 50, "success_rate": 50.0}
    assert len(report["per_category"]) == 5
    for bad in (rows[:-1], rows + rows[:1], []):
        write(path, json.dumps({"per_task": bad}))
        with pytest.raises(ValueError):
            robust.summarize(directory / "manifest.json", path)


def test_actual_launcher_uses_full_checkpoint_pro_suites_and_rejects_invalid_weights(fork, tmp_path):
    fixture = fixtures.VLMOnlyTest()
    fixture.setUp()
    try:
        checkpoint = fixture.work / "checkpoints/robust-model/checkpoints/006000/pretrained_model"
        fixture.make_model(checkpoint, "base", tokens=2)
        config = json.loads((checkpoint / "config.json").read_text())
        config.update(vlm_prompt_only=False, train_action_expert=True)
        (checkpoint / "config.json").write_text(json.dumps(config))
        weights = load_file(checkpoint / "model.safetensors")
        for name in (
            "paligemma_with_expert.gemma_expert.model",
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ):
            weights[f"model.{name}.weight"] = np.ones(1, dtype=np.float32)
        save_file(weights, checkpoint / "model.safetensors")
        env = fixture.env | {
            "RUN_NAME": "robust-model",
            "BENCHMARK_ROOT": str(fork),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        }
        result = subprocess.run(
            ["bash", str(ROOT / "run_eval_libero_robustness.sh"), "pro", "libero_goal"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        command = shlex.split(
            next(line for line in result.stdout.splitlines() if line.startswith("python -u - "))
        )
        assert f"--policy.path={checkpoint}" in command
        assert "--env.type=libero" in result.stdout and "--env.observation_width=224" in result.stdout
        assert "--eval.n_episodes=10" in result.stdout and "--policy.n_action_steps=10" in result.stdout
        assert (
            "--env.task=libero_goal_object,libero_goal_swap,libero_goal_lan,libero_goal_task,libero_goal_env"
            in command
        )
        config["train_action_expert"] = False
        (checkpoint / "config.json").write_text(json.dumps(config))
        result = subprocess.run(
            ["bash", str(ROOT / "run_eval_libero_robustness.sh"), "pro", "libero_goal"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode != 0 and "Expected train_action_expert=True" in result.stderr
    finally:
        fixture.doCleanups()


def test_embedded_runner_still_compiles():
    shell = (ROOT / "run_eval_libero-full.sh").read_text()
    body = shell.split("<<'PY' 2>&1 | tee -a", 1)[1].split("\n", 1)[1].split("\nPY\n", 1)[0]
    compile(body, "progress-wrapper", "exec")
