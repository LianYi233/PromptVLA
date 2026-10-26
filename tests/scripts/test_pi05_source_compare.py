"""CPU checks for a weights-only source intervention and reproducible launch recipes."""

import copy
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
from safetensors.numpy import save_file

from lerobot.scripts import pi05_source_compare as experiment
from lerobot.utils.pdet_order import create_plan, read_frame_metadata, save_plan
from lerobot.utils.prompt_initialization import prompt_hash, write_prompt

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "examples/training/train_pi05_source_compare.sh"
EVAL = ROOT / "run_eval_libero_source_compare.sh"


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "dataset"
    (data / "meta").mkdir(parents=True)
    (data / "data").mkdir()
    stats = {key: {"q01": [0] * n, "q99": [1] * n} for key, n in (("observation.state", 8), ("action", 7))}
    (data / "meta/stats.json").write_text(json.dumps(stats))
    count = 512
    pq.write_table(
        pa.table(
            {
                "index": range(count),
                "episode_index": [i // 32 for i in range(count)],
                "frame_index": [i % 32 for i in range(count)],
                "task_index": [i // 256 for i in range(count)],
                "action": [[0.0] * 7] * count,
            }
        ),
        data / "data/part.parquet",
    )
    plan = create_plan(
        read_frame_metadata(data), steps=3000, batch_size=32, world_size=2, data_seed=0, within_seed=0
    )
    plan_path = tmp_path / "old-plan.npz"
    save_plan(plan_path, plan)
    config = {
        "type": "pi05",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "max_state_dim": 32,
        "max_action_dim": 32,
        "chunk_size": 50,
        "empty_cameras": 1,
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
        "input_features": {"observation.state": {"type": "STATE", "shape": [8]}},
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
    }
    libero, generic = [tmp_path / name for name in ("pi05_libero_base", "pi05_base")]
    for path, value in ((libero, 1.0), (generic, 2.0)):
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config))
        save_file(
            {"model.backbone.weight": np.full((4, 4), value, dtype=np.float32)}, path / "model.safetensors"
        )
    # Simulate generic checkpoint's incompatible data interface and absent processors.
    other = copy.deepcopy(config)
    other["input_features"] = {"observation.state": {"type": "STATE", "shape": [32]}}
    other["output_features"]["action"]["shape"] = [32]
    other["empty_cameras"] = 0
    (generic / "config.json").write_text(json.dumps(other))
    for filename, registry, features in (
        (
            "policy_preprocessor.json",
            "normalizer_processor",
            {**config["input_features"], **config["output_features"]},
        ),
        ("policy_postprocessor.json", "unnormalizer_processor", config["output_features"]),
    ):
        (libero / filename).write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "registry_name": registry,
                            "config": {"features": features, "norm_map": config["normalization_mapping"]},
                            "state_file": "stats.safetensors",
                        }
                    ]
                }
            )
        )
    save_file(
        {
            f"{key}.{stat}": np.asarray(value, dtype=np.float32)
            for key, stats_for_key in stats.items()
            for stat, value in stats_for_key.items()
        },
        libero / "stats.safetensors",
    )
    prompt_path = tmp_path / "P1.safetensors"
    prompt = np.ones((2, 2048), dtype=np.float32)
    write_prompt(
        prompt_path,
        prompt,
        {
            "schema_version": 1,
            "source_step": 0,
            "source_training_seed": 1,
            "num_vlm_prompt_tokens": 2,
            "prompt_sha256": prompt_hash(prompt),
        },
    )
    reference = tmp_path / "historical"
    (reference / "pdet").mkdir(parents=True)
    ref = {
        "order_plan": plan["manifest"],
        "prompt_sha256": prompt_hash(prompt),
        "source_pretrained_path": str(libero),
        "global_seed": 0,
        "tokens": 2,
        "total_steps": 3000,
        "factors": {
            "prompt_seed": 1,
            "data_seed": 0,
            "noise_seed": 1,
            "time_seed": 1,
            "flow_binding": "occurrence",
        },
    }
    (reference / "pdet/manifest-rank0.json").write_text(json.dumps(ref))
    return {
        "libero": libero,
        "generic": generic,
        "plan_path": plan_path,
        "prompt_path": prompt_path,
        "dataset_root": data,
        "reference_run": reference,
        "output": tmp_path / "pair",
    }


def test_prepare_preserves_all_inputs_and_uses_correct_weights_without_modifying_originals(inputs):
    originals = {
        p: experiment.digest(p) for source in (inputs["libero"], inputs["generic"]) for p in source.iterdir()
    }
    experiment.prepare(**inputs)
    output = inputs["output"]
    record = experiment.check(output, full_hash=True)
    for label in experiment.LABELS:
        assert (output / label / "model.safetensors").is_symlink()
        assert (output / label / "model.safetensors").resolve() == inputs[label] / "model.safetensors"
        assert (output / label / "config.json").read_bytes() == (
            inputs["libero"] / "config.json"
        ).read_bytes()
    assert (output / "P1.safetensors").read_bytes() == inputs["prompt_path"].read_bytes()
    assert (output / "D0-W0.npz").read_bytes() == inputs["plan_path"].read_bytes()
    assert record["sources"]["generic"]["original_config"]["output_features"]["action"]["shape"] == [32]
    assert originals == {p: experiment.digest(p) for p in originals}
    with pytest.raises(FileExistsError):
        experiment.prepare(**inputs)


@pytest.mark.parametrize("change", ["weights", "config", "stats", "plan", "prompt", "dataset"])
def test_preflight_detects_input_changes(inputs, change):
    experiment.prepare(**inputs)
    paths = {
        "weights": inputs["generic"] / "model.safetensors",
        "config": inputs["output"] / "generic/config.json",
        "stats": inputs["output"] / "libero/stats.safetensors",
        "plan": inputs["output"] / "D0-W0.npz",
        "prompt": inputs["output"] / "P1.safetensors",
        "dataset": inputs["dataset_root"] / "meta/stats.json",
    }
    with paths[change].open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        experiment.check(inputs["output"])


@pytest.mark.parametrize(
    "change,match",
    [
        ("shape", "key/shape"),
        ("same", "identical weight"),
        ("prompt", "Plan or exact prompt"),
        ("architecture", "architecture"),
    ],
)
def test_prepare_rejects_invalid_comparisons(inputs, change, match):
    if change == "shape":
        save_file(
            {"model.backbone.weight": np.ones((8, 4), dtype=np.float32)},
            inputs["generic"] / "model.safetensors",
        )
    elif change == "same":
        (inputs["generic"] / "model.safetensors").write_bytes(
            (inputs["libero"] / "model.safetensors").read_bytes()
        )
    elif change == "prompt":
        path = inputs["reference_run"] / "pdet/manifest-rank0.json"
        record = json.loads(path.read_text())
        record["prompt_sha256"] = "incorrect"
        path.write_text(json.dumps(record))
    else:
        path = inputs["generic"] / "config.json"
        config = json.loads(path.read_text())
        config["max_action_dim"] = 64
        path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        experiment.prepare(**inputs)
    assert not inputs["output"].exists()


def test_schema_uses_supported_loader_aliases_and_rejects_trained_prompts(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_file({"action_time_mlp_in.weight": np.ones((3, 2), dtype=np.float32)}, path)
    assert experiment.weight_schema(str(path)) == {"model.time_mlp_in.weight": [3, 2]}
    save_file({"model.vlm_prompt_tokens.weight": np.ones((2, 2048), dtype=np.float32)}, path)
    with pytest.raises(ValueError, match="learned prompts"):
        experiment.weight_schema(str(path))


def test_checkpoint_checks_actual_source_not_just_model_name(inputs):
    experiment.prepare(**inputs)
    run = inputs["output"].parent / "pi05-source-r1-generic-W0-Bocc-base-vlm2-P1-D0-E1-T1"
    checkpoint = run / "checkpoints/003000/pretrained_model"
    shutil.copytree(inputs["output"] / "generic", checkpoint)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
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
    config_path.write_text(json.dumps(config))
    save_file(
        {
            "model.vlm_prompt_tokens.weight": np.ones((2, 2048), dtype=np.float32),
            "model.prompt_tokens.weight": np.empty((0, 1024), dtype=np.float32),
        },
        checkpoint / "model.safetensors",
    )
    training = {
        "policy": {**config, "pretrained_path": str(inputs["output"] / "generic")},
        "steps": 3000,
        "seed": 0,
        "pdet": {
            "prompt_seed": 1,
            "data_seed": 0,
            "noise_seed": 1,
            "time_seed": 1,
            "flow_binding": "occurrence",
        },
    }
    config_file = checkpoint / "train_config.json"
    config_file.write_text(json.dumps(training))
    assert experiment.check_checkpoint(inputs["output"], run)["weight_source"] == "generic"
    training["policy"]["pretrained_path"] = str(inputs["output"] / "libero")
    config_file.write_text(json.dumps(training))
    with pytest.raises(ValueError, match="different source path"):
        experiment.check_checkpoint(inputs["output"], run)


def test_two_launchers_differ_only_in_source_name_gpu_and_port(inputs, tmp_path):
    experiment.prepare(**inputs)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "SOURCE_PAIR_DIR": str(inputs["output"]),
        "DATASET_ROOT": str(inputs["dataset_root"]),
        "OUTPUT_ROOT": str(tmp_path / "runs"),
        "LOG_ROOT": str(tmp_path / "logs"),
        "TOKENIZER_PATH": str(tokenizer),
        "DRY_RUN": "true",
    }
    parsed = []
    for label, gpu, port in (("libero", "0,1", "29581"), ("generic", "2,3", "29582")):
        result = subprocess.run(["bash", str(TRAIN), label], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr + result.stdout
        line = next(
            line for line in result.stdout.splitlines() if line.startswith("env CUDA_VISIBLE_DEVICES=")
        )
        args = shlex.split(line)
        flags = dict(a[2:].split("=", 1) for a in args if a.startswith("--") and "=" in a)
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in args
        assert flags["main_process_port"] == port
        assert flags["policy.path"] == str(inputs["output"] / label)
        assert flags["vlm_prompt_init_path"] == str(inputs["output"] / "P1.safetensors")
        assert flags["pdet.order_plan"] == str(inputs["output"] / "D0-W0.npz")
        assert (
            flags["seed"],
            flags["pdet.prompt_seed"],
            flags["pdet.noise_seed"],
            flags["pdet.time_seed"],
        ) == ("0", "1", "1", "1")
        assert (flags["policy.num_vlm_prompt_tokens"], flags["policy.num_prompt_tokens"], flags["steps"]) == (
            "2",
            "0",
            "3000",
        )
        assert (flags["batch_size"], flags["num_processes"], flags["save_freq"], flags["save_steps"]) == (
            "32",
            "2",
            "3000",
            "[]",
        )
        assert flags["policy.ntk_save_stage_snapshots"] == "false"
        assert flags["policy.compile_model"] == "false"
        for key in ("main_process_port", "policy.path", "job_name", "output_dir"):
            flags.pop(key)
        parsed.append(flags)
    assert parsed[0] == parsed[1]
    assert not (tmp_path / "runs").exists()
    blocked = subprocess.run(
        ["bash", str(TRAIN), "generic"],
        env={**env, "PRETRAINED_PATH": "/other"},
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0 and "Unset PRETRAINED_PATH" in blocked.stderr
    run = tmp_path / "runs/pi05-source-r1-libero-W0-Bocc-base-vlm2-P1-D0-E1-T1"
    run.mkdir(parents=True)
    blocked = subprocess.run(["bash", str(TRAIN), "libero"], env=env, capture_output=True, text=True)
    assert blocked.returncode != 0 and "Output already exists" in blocked.stderr


def test_pair_audit_allows_only_weight_source_change(inputs, monkeypatch):
    from lerobot.scripts import pdet_order_binding

    experiment.prepare(**inputs)
    record = experiment.check(inputs["output"])
    runs = [
        inputs["output"].parent / f"pi05-source-r1-{label}-W0-Bocc-base-vlm2-P1-D0-E1-T1"
        for label in experiment.LABELS
    ]
    original = {
        "prompt_sha256": record["prompt_sha256"],
        "pool": "pool",
        "tasks": "tasks",
        "shapes": [[32, 50, 32], [32]],
        "global_seed": 0,
        "dataset_metadata_sha256": "data",
        "source_pretrained_path": "unused",
        "recipe": {
            "dataset": {"root": record["dataset_root"]},
            "policy": {"pretrained_path": "unused", "optimizer_lr": 2.5e-5},
        },
    }
    changed = {}

    def inspect(run, **_):
        invariant = copy.deepcopy(original)
        match = experiment.NAME.fullmatch(Path(run).name)
        source = inputs["output"] / match[2] if match else inputs["libero"]
        invariant["source_pretrained_path"] = str(source)
        invariant["recipe"]["policy"]["pretrained_path"] = str(source)
        report = {"run": str(run), "bank_sha256": "bank", "order_sha256": "order"}
        if match and match[2] == "generic":
            if "lr" in changed:
                invariant["recipe"]["policy"]["optimizer_lr"] = 1e-3
            report.update(changed.get("report", {}))
        return invariant, report

    monkeypatch.setattr(pdet_order_binding, "inspect_run", inspect)
    monkeypatch.setattr(experiment, "check_checkpoint", lambda *_: None)
    monkeypatch.setattr(experiment, "assert_same_processors", lambda *_: None)
    assert experiment.audit(inputs["output"], runs)["audit"] == "passed"
    changed["lr"] = True
    with pytest.raises(ValueError, match="remaining recipe"):
        experiment.audit(inputs["output"], runs)
    changed.clear()
    changed["report"] = {"bank_sha256": "another bank"}
    with pytest.raises(ValueError, match="sample/flow"):
        experiment.audit(inputs["output"], runs)


def test_eval_dispatch_preserves_names_source_and_clean_progress(tmp_path):
    # Fake Python validates shell dispatch only; real checkpoint checks have their own tests.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / EVAL.name).write_text(EVAL.read_text())
    (repo / "run_eval_libero_vlm_only_pdet_sweep.sh").write_text(
        '#!/usr/bin/env bash\nprintf "%s|%s|%s|%s|%s|%s\\n" "$BASE_PRETRAINED_PATH" "$RUN_PREFIX" "$PDET_CASES" "$EVAL_SEED" "$EPISODES_PER_TASK" "$PROGRESS"\n'
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
    (binaries / "python").chmod(0o755)
    env = {"PATH": str(binaries) + os.pathsep + os.environ["PATH"], "SOURCE_PAIR_DIR": str(tmp_path / "pair")}
    result = subprocess.run(
        ["bash", str(repo / EVAL.name), "both", "all"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    for label, line in zip(experiment.LABELS, lines, strict=True):
        assert line == f"{tmp_path}/pair/{label}|pi05-source-r1-{label}-W0-Bocc|1011|1000|10|true"
