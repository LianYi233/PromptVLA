"""Order/binding controls must share actual indexed draws, not just seed labels."""

import json
import shlex
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs.pdet import PDETConfig
from lerobot.scripts.pdet_followup import audit_order
from lerobot.scripts.pdet_order_binding import audit_runs, check_checkpoint
from lerobot.utils.pdet import PDETStreams, sample_flow
from lerobot.utils.pdet_order import save_plan
from lerobot.utils.prompt_initialization import prompt_hash, write_prompt
from tests.scripts.test_pi05_pdet import TinyPolicy, accelerator, make_config
from tests.scripts.test_pi05_pdet_followup import make_dataset, plan_for, shard_batches

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "examples/training/train_pi05_order_binding.sh"
EVAL = ROOT / "run_eval_libero_order_binding.sh"


def name_for(w, binding, bank):
    short = "occ" if binding == "occurrence" else "slot"
    return f"pi05-orderbind-r1-W{w}-B{short}-base-vlm2-P1-D0-E{bank}-T{bank}"


def test_bindings_use_identical_cpu_bank_and_preserve_rng():
    policy = TinyPolicy()
    batch = {"action": torch.zeros(3, 50, 7)}
    config = PDETConfig(enabled=True, order_plan="plan", noise_seed=1, time_seed=1)
    state = torch.get_rng_state().clone()
    a = sample_flow(policy, batch, config, 0, 0, [8, 2, 7])
    b = sample_flow(policy, batch, replace(config, flow_binding="slot"), 19, 1, [999, 123, 5], slot_ids=[8, 2, 7])
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert torch.equal(state, torch.get_rng_state())
    c = sample_flow(policy, batch, replace(config, flow_binding="slot"), 19, 1, [5, 999, 123], slot_ids=[8, 2, 7])
    assert all(torch.equal(b[key], c[key]) for key in b)
    for field, changed in (("noise_seed", "noise"), ("time_seed", "time")):
        other = sample_flow(policy, batch, replace(config, flow_binding="slot", **{field: 0}), 19, 1, [5, 999, 123], slot_ids=[8, 2, 7])
        for key in b:
            assert torch.equal(b[key], other[key]) == (key != changed)
    with pytest.raises(ValueError, match="global slot ID"):
        sample_flow(policy, batch, replace(config, flow_binding="slot"), 0, 0, [8, 2, 7])
    with pytest.raises(ValueError, match="finite"):
        PDETConfig(enabled=True, flow_binding="slot").validate()
    with pytest.raises(ValueError, match="occurrence or slot"):
        replace(config, flow_binding="typo").validate()


def test_full_factorial_audit_checks_late_draws_keys_sources_and_missing_cells(tmp_path):
    root = tmp_path / "dataset"
    metadata = make_dataset(root)
    policy = TinyPolicy()
    plans = [plan_for(metadata, w, steps=20) for w in (0, 1)]
    for w, plan in enumerate(plans):
        save_plan(tmp_path / f"W{w}.npz", plan)
    runs = []
    for bank in (1, 0):
        for binding in ("occurrence", "slot"):
            for w, plan in enumerate(plans):
                run = tmp_path / name_for(w, binding, bank)
                cfg = make_config(run, PDETConfig(
                    enabled=True, prompt_seed=1, data_seed=0, noise_seed=bank, time_seed=bank,
                    order_plan=str(tmp_path / f"W{w}.npz"), flow_binding=binding, audit_steps=20,
                ))
                cfg.steps, cfg.batch_size, cfg.dataset.root = 20, 2, root
                for rank in (0, 1):
                    streams = PDETStreams(cfg, policy, accelerator(rank))
                    for step, items in enumerate(shard_batches(plan, rank)):
                        ids, occurrences = zip(*items, strict=True)
                        streams.draw(policy, {"action": torch.zeros(2, 50, 7)}, step, ids, occurrences)
                cfg.resume = True
                PDETStreams(cfg, policy, accelerator())
                runs.append(run)
    report = audit_runs(runs)
    assert report["audit"] == "passed" and report["complete_factorial"]
    assert report["noise_banks"] == 2
    assert len({row["bank_sha256"] for row in report["runs"]}) == 2
    assert audit_order(runs[:2])["audit"] == "passed"
    with pytest.raises(ValueError, match="Do not mix"):
        audit_order(runs[2:4])
    with pytest.raises(ValueError, match="Incomplete"):
        audit_runs(runs[:-1])
    assert audit_runs(runs[:2], allow_partial=True)["audit"] == "partial_inputs_passed"

    # Beyond the original eight-step audits: corruption must still be detected.
    path = runs[-1] / "pdet/inputs-rank1.jsonl"
    original = path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[15]["occurrence_noise_sha256"][0] = "0" * 64
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="BANK differs"):
        audit_runs(runs)
    path.write_text(original)
    rows = [json.loads(line) for line in original.splitlines()]
    rows[15]["flow_key_ids"].reverse()
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="declared binding"):
        audit_runs(runs)
    path.write_text(original)
    manifest_path = runs[-1] / "pdet/manifest-rank1.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_tree_sha256"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="across ranks"):
        audit_runs(runs)


@pytest.mark.parametrize("mode,gpus,port", [("occ", "0,1", "29551"), ("slot", "2,3", "29552")])
def test_train_dispatch_two_banks_two_orders_and_preflight_all_outputs(tmp_path, mode, gpus, port):
    from tests.scripts.test_pi05_vlm_only import VLMOnlyTest

    fixture = VLMOnlyTest()
    fixture.setUp()
    try:
        make_dataset(fixture.dataset)
        prompt = np.ones((2, 2048), dtype=np.float32)
        write_prompt(tmp_path / "inits/pi05-base-vlm2-init-seed1.safetensors", prompt, {
            "schema_version": 1, "source_step": 0, "source_training_seed": 1,
            "num_vlm_prompt_tokens": 2, "prompt_sha256": prompt_hash(prompt),
        })
        env = {**fixture.env, "PROMPT_INIT_DIR": str(tmp_path / "inits"), "PLAN_DIR": str(tmp_path / mode)}
        result = subprocess.run(["bash", str(TRAIN), mode], env=env, capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stderr
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]
        assert len(commands) == 4
        binding = "occurrence" if mode == "occ" else "slot"
        for command, (bank, w) in zip(commands, ((1, 0), (1, 1), (0, 0), (0, 1)), strict=True):
            for flag in (
                f"CUDA_VISIBLE_DEVICES={gpus}", f"--main_process_port={port}", "--num_processes=2",
                "--batch_size=32", "--steps=3000", "--save_steps=[]", "--policy.ntk_save_stage_snapshots=false",
                "--policy.compile_model=false", "--seed=0", "--pdet.prompt_seed=1", "--pdet.data_seed=0",
                f"--pdet.flow_binding={binding}", f"--pdet.noise_seed={bank}", f"--pdet.time_seed={bank}",
                "--pdet.audit_steps=3000", "--prompt_rank.fixed_flow_loss=true",
            ):
                assert flag in command
            output = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output_dir=")))
            assert output.name == name_for(w, binding, bank)
        output.mkdir(parents=True)
        blocked = subprocess.run(["bash", str(TRAIN), mode], env=env, capture_output=True, text=True, timeout=30)
        assert blocked.returncode != 0 and "Output already exists" in blocked.stderr
        assert "Starting " not in blocked.stdout
    finally:
        fixture.doCleanups()


def test_eval_uses_saved_bindings_and_separate_result_names(tmp_path):
    from tests.scripts.test_pi05_vlm_only import VLMOnlyTest

    fixture = VLMOnlyTest()
    fixture.setUp()
    try:
        metadata = make_dataset(fixture.dataset)
        plan_path = tmp_path / "W0.npz"
        save_plan(plan_path, plan_for(metadata, 0, steps=3000))
        for binding in ("occurrence", "slot"):
            run = Path(fixture.env["CKPT_ROOT"]) / name_for(0, binding, 1)
            checkpoint = run / "checkpoints/003000"
            model = checkpoint / "pretrained_model"
            fixture.make_model(model, "base", tokens=2)
            path = model / "train_config.json"
            config = json.loads(path.read_text())
            config["policy"]["num_vlm_prompt_tokens"] = 2
            config.update(seed=0, steps=3000, pdet=asdict(PDETConfig(
                enabled=True, prompt_seed=1, noise_seed=1, time_seed=1,
                order_plan=str(plan_path), flow_binding=binding,
            )))
            path.write_text(json.dumps(config))
            (checkpoint.parent / "last").symlink_to("003000")
            assert check_checkpoint(run)["checkpoint"] == "name_and_config_match"
        capture = fixture.install_fake_evaluator()
        for mode in ("occ", "slot"):
            result = subprocess.run(
                ["bash", str(EVAL), mode, "all"],
                env={**fixture.env, "FLOW_BANKS": "1", "WITHIN_SEEDS": "0", "PROGRESS": "true", "DRY_RUN": "false"},
                capture_output=True, text=True, timeout=90,
            )
            assert result.returncode == 0, result.stderr
            assert "overall: 90.00% (360/400)" in result.stdout
            assert "video-path.mp4" not in result.stdout
        commands = [json.loads(line) for line in capture.read_text().splitlines()]
        assert len(commands) == 2
        for command, binding, gpu in zip(commands, ("occurrence", "slot"), ("0", "2"), strict=True):
            assert command["gpu"] == gpu
            assert command["--seed"] == "1000" and command["--eval.n_episodes"] == "10"
            assert name_for(0, binding, 1) in command["--output_dir"]
        config["pdet"]["flow_binding"] = "occurrence"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="does not match"):
            check_checkpoint(run)
    finally:
        fixture.doCleanups()
