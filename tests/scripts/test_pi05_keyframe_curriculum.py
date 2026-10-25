"""Loss-weight interventions must preserve action chunks, sample order and flow draws."""

import hashlib
import json
import random
import shlex
import subprocess
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.keyframe_loss import KeyframeLossConfig
from lerobot.configs.pdet import PDETConfig
from lerobot.configs.prompt_rank import PromptRankConfig
from lerobot.scripts.frameskip_scores import file_inventory, read_actions, source_inventory
from lerobot.scripts.keyframe_experiment import audit_runs, check_checkpoint, validate_prepared
from lerobot.utils.keyframe_loss import (
    MODES,
    KeyframeLossRuntime,
    build_archive,
    calibrated_curve,
    chunk_masks,
    load_archive,
    normalized_key_mass,
    save_archive,
    shifted_masks,
    weighted_action_loss,
)
from lerobot.utils.pdet import PDETStreams, sample_flow
from lerobot.utils.pdet_order import arrays_hash, create_plan, save_plan
from lerobot.utils.prompt_initialization import prompt_hash, write_prompt
from lerobot.utils.prompt_rank import PromptRankMonitor
from tests.scripts.test_pi05_pdet import TinyPolicy, accelerator, make_config, real_functions
from tests.scripts.test_pi05_pdet_followup import make_dataset, plan_for, shard_batches

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "examples/training/train_pi05_keyframe_curriculum.sh"
EVAL = ROOT / "run_eval_libero_keyframe_curriculum.sh"


def prepared(tmp_path, *, production=False, root=None, source=None):
    root = root or tmp_path / "dataset"
    metadata = make_dataset(root)
    recipe = {
        "config": {"event_before": 10, "event_after": 2, "debounce": 2},
        "dataset_metadata_sha256": arrays_hash(metadata),
        "dataset_root": str(root.resolve()),
        "source": str(source or "/base"),
    }
    if source:
        recipe.update(
            actions_sha256=arrays_hash({"action": read_actions(root)}),
            stats_sha256=hashlib.sha256((root / "meta/stats.json").read_bytes()).hexdigest(),
            source_inventory=source_inventory(source),
            video_inventory_sha256=hashlib.sha256(
                json.dumps(file_inventory(root, sorted((root / "videos").rglob("*")))).encode()
            ).hexdigest(),
        )
    plan = (
        create_plan(metadata, steps=3000, batch_size=32, world_size=2, data_seed=0, within_seed=0)
        if production
        else plan_for(metadata, 0, steps=20)
    )
    # Synthetic event mask: two steps in each episode; task1 has no events.
    scores = {
        **metadata,
        "event_neighborhood": (metadata["frame_index"] >= 2)
        & (metadata["frame_index"] <= 3)
        & (metadata["task_index"] == 0),
    }
    arrays, manifest = build_archive(scores, recipe, plan)
    path = tmp_path / "prepared"
    save_plan(path / "D0-W0.npz", plan)
    save_archive(path / "archive.npz", arrays, manifest)
    return path, arrays, manifest, plan


def name(mode, bank):
    return f"pi05-key-r1-{mode}-W0-Bocc-base-vlm2-P1-D0-E{bank}-T{bank}"


def test_chunks_stop_at_episode_boundary_and_placebo_matches_every_sample(tmp_path):
    _, arrays, _, _ = prepared(tmp_path)
    mask, valid = chunk_masks(arrays, [1, 3, 5, 12, 23])
    assert valid.sum(1).tolist() == [5, 3, 1, 6, 1]
    assert mask.sum(1).tolist() == [2, 1, 0, 0, 0]
    assert not mask[~valid].any()
    shifted = shifted_masks(mask, valid, [1, 2, 3, 4, 5])
    assert np.array_equal(shifted.sum(1), mask.sum(1))
    assert not shifted[~valid].any() and not np.array_equal(mask, shifted)
    for strength in (1.5, 2, 2.5):
        assert np.array_equal(
            np.where(valid, 1 + (strength - 1) * mask, 0).sum(1),
            np.where(valid, 1 + (strength - 1) * shifted, 0).sum(1),
        )


def test_calibration_matches_normalized_budget_and_preserves_all_global_rng(tmp_path):
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    directory, arrays, manifest, _ = prepared(tmp_path)
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    validate_prepared(directory, production=False)
    # Deliberately strongly nonstationary key fraction; raw average2 is insufficient.
    key, valid = np.array([0, 1, 2, 4, 8, 10]), np.full(6, 12)
    target = normalized_key_mass(2, key, valid).sum()
    for direction, sign in (("early", -1), ("late", 1)):
        weights = calibrated_curve(direction, key, valid)
        assert np.all(sign * np.diff(weights) > 0)
        assert normalized_key_mass(weights.astype(float), key, valid).sum() == pytest.approx(target, rel=1e-7)
    assert not np.isclose(normalized_key_mass(np.linspace(2.5, 1.5, 6), key, valid).sum(), target)
    changed = {**arrays, "weight_Kearly": np.ones_like(arrays["weight_Kearly"])}
    with (directory / "bad.npz").open("wb") as handle:
        np.savez_compressed(
            handle, **changed, manifest=json.dumps({**manifest, "arrays_sha256": arrays_hash(changed)})
        )
    with pytest.raises(ValueError, match="budgets differ"):
        load_archive(directory / "bad.npz")


def actual_policy():
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    namespace = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "ACTION": ACTION,
        "OBS_LANGUAGE_TOKENS": OBS_LANGUAGE_TOKENS,
        "OBS_LANGUAGE_ATTENTION_MASK": OBS_LANGUAGE_ATTENTION_MASK,
        "_PI05_LOSS_VALID_STEPS_KEY": "valid_steps",
        "PI05TrainingStage": SimpleNamespace(NEXT_ACTION="next_action"),
        "PI05TrainingObjective": SimpleNamespace(OBSERVATION_FLOW="flow"),
    }
    exec(real_functions(None, {"_valid_action_steps", "_reduce_action_losses"}), namespace)
    exec(
        real_functions(
            "PI05Policy", {"forward", "_forward_observation_flow", "get_distributed_loss_normalizer"}
        ),
        namespace,
    )
    policy = TinyPolicy()
    policy.config.training_stage = "flow"
    policy.config.output_features = {ACTION: SimpleNamespace(shape=(7,))}
    policy._preprocess_images = lambda batch: ([], [])
    for method in ("forward", "_forward_observation_flow", "get_distributed_loss_normalizer"):
        setattr(policy, method, namespace[method].__get__(policy))
    batch = {
        "action": torch.zeros(2, 50, 7),
        "action_is_pad": torch.arange(50)[None, :] >= torch.tensor([[50], [13]]),
        OBS_LANGUAGE_TOKENS: torch.ones(2, 2),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 2),
    }
    return policy, batch, namespace["_reduce_action_losses"]


def test_real_forward_b_is_bitwise_identical_and_weighted_ddp_gradient_is_global_mean():
    policy, batch, old_reduce = actual_policy()
    parameter = torch.linspace(-1, 1, 700).reshape(2, 50, 7).requires_grad_()
    policy.model.forward = lambda *args, **kwargs: parameter.square()
    valid = ~batch["action_is_pad"]
    key = torch.arange(50)[None, :].expand(2, -1) < 10
    old_loss = old_reduce(parameter.square(), valid)[0]
    old_gradient = torch.autograd.grad(old_loss, parameter)[0]
    baseline, metrics = policy(batch, action_key_mask=key)
    assert torch.equal(baseline, old_loss)
    assert torch.equal(torch.autograd.grad(baseline, parameter)[0], old_gradient)
    assert "_pi05_action_loss_weights" not in batch
    assert metrics["keyframe/key_coordinates"] == 140
    assert metrics["keyframe/nonkey_coordinates"] == int(valid.sum()) * 7 - 140
    weights = 1 + key.float()
    loss, _ = policy(batch, action_loss_weights=weights, action_key_mask=key)
    denominator = torch.where(valid, weights, 0).sum() * 7
    assert policy.get_distributed_loss_normalizer(batch) == denominator
    expected = (parameter.square() * torch.where(valid, weights, 0)[..., None]).sum() / denominator
    assert torch.equal(loss, expected)
    # Simulate DDP averaging two local gradients AFTER its world*local/global correction.
    gradients = []
    for rank in (0, 1):
        local = weighted_action_loss(
            parameter[rank : rank + 1].square(), valid[rank : rank + 1], weights[rank : rank + 1]
        )[0]
        local_denominator = torch.where(valid[rank], weights[rank], 0).sum() * 7
        gradients.append(
            torch.autograd.grad(local * 2 * local_denominator / denominator, parameter, retain_graph=True)[0]
        )
    assert torch.allclose(
        sum(gradients) / 2, torch.autograd.grad(expected, parameter)[0], rtol=1e-6, atol=1e-8
    )
    # Reusing a weighted batch for B must remove the weighted normalizer.
    again, _ = policy(batch, action_key_mask=key)
    assert torch.equal(again, old_loss)
    assert policy.get_distributed_loss_normalizer(batch) == valid.sum() * 7
    zero, _, _ = weighted_action_loss(parameter.square(), torch.zeros_like(valid), weights)
    assert zero.item() == 0 and torch.equal(
        torch.autograd.grad(zero, parameter)[0], torch.zeros_like(parameter)
    )


def test_full_audit_preserves_flow_draws_and_detects_late_mask_noise_source_corruption(tmp_path):
    directory, arrays, _, plan = prepared(tmp_path)
    policy = TinyPolicy()
    runs = []
    for mode in MODES:
        for bank in (0, 1):
            run = tmp_path / name(mode, bank)
            factors = PDETConfig(
                enabled=True,
                prompt_seed=1,
                data_seed=0,
                noise_seed=bank,
                time_seed=bank,
                order_plan=str(directory / "D0-W0.npz"),
                audit_steps=20,
            )
            cfg = make_config(run, factors)
            cfg.dataset.root, cfg.steps, cfg.batch_size, cfg.log_freq = tmp_path / "dataset", 20, 2, 10
            cfg.keyframe_loss = KeyframeLossConfig(mode, str(directory / "archive.npz"))
            cfg.to_dict = lambda cfg=cfg: {"seed": 0, "steps": 20, "keyframe_loss": asdict(cfg.keyframe_loss)}
            for rank in (0, 1):
                acc = accelerator(rank)
                acc.is_main_process = rank == 0
                streams = PDETStreams(cfg, policy, acc)
                runtime = KeyframeLossRuntime(cfg, streams, acc)
                for step, items in enumerate(shard_batches(plan, rank)):
                    indices, occurrences = zip(*items, strict=True)
                    _, valid = chunk_masks(arrays, indices)
                    batch = {"action": torch.zeros(2, 50, 7), "action_is_pad": torch.from_numpy(~valid)}
                    expected = sample_flow(policy, batch, factors, step, rank, occurrences)
                    inputs = streams.draw(policy, batch, step, indices, occurrences)
                    state = torch.get_rng_state().clone()
                    inputs.update(runtime.inputs(batch, step, indices, occurrences))
                    assert torch.equal(torch.get_rng_state(), state)
                    assert all(torch.equal(inputs[k], expected[k]) for k in ("noise", "time"))
                    assert ("action_loss_weights" in inputs) == (mode != "B")
                wrong_batch = {
                    "action": torch.zeros(2, 50, 7),
                    "action_is_pad": torch.zeros(2, 50, dtype=torch.bool),
                }
                indices, occurrences = zip(*shard_batches(plan, rank)[0], strict=True)
                with pytest.raises(ValueError, match="padding"):
                    runtime.inputs(wrong_batch, 0, indices, occurrences)
            runs.append(run)
    assert audit_runs(runs, "curriculum")["audit"] == "passed"
    assert audit_runs(runs[:2], "baseline")["complete"]
    with pytest.raises(ValueError, match="Incomplete"):
        audit_runs(runs[:-1], "curriculum")
    path = runs[-1] / "keyframe_loss/inputs-rank1.jsonl"
    original = path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[18]["weight_sha256"] = "0" * 64
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="weight_sha256"):
        audit_runs(runs, "curriculum")
    path.write_text(original)
    path = runs[-1] / "pdet/inputs-rank1.jsonl"
    original = path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[18]["occurrence_noise_sha256"][0] = "0" * 64
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="BANK differs"):
        audit_runs(runs, "curriculum")
    path.write_text(original)
    path = runs[-1] / "pdet/manifest-rank1.json"
    manifest = json.loads(path.read_text())
    manifest["source_tree_sha256"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="across ranks"):
        audit_runs(runs, "curriculum")


def test_fixed_probe_key_split_does_not_add_forward_calls_or_change_rng(tmp_path):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    policy = TinyPolicy()
    policy._preprocess_images = lambda batch: ([], [])
    calls = []

    def velocity(images, masks, tokens, text_masks, noisy, time):
        calls.append(1)
        return noisy * policy.model.vlm_prompt_tokens.weight.mean()

    policy.model.predict_velocity = velocity
    monitor = PromptRankMonitor.__new__(PromptRankMonitor)
    monitor.options = PromptRankConfig(
        freq=250, samples=1, output_probes=1, timesteps=[0.5], fixed_flow_loss=True
    )
    monitor.device, monitor.output_dir, monitor.action_dim, monitor.horizon = (
        torch.device("cpu"),
        tmp_path,
        7,
        50,
    )
    monitor.samples = [
        {
            "action": torch.ones(1, 50, 7),
            "action_is_pad": torch.arange(50)[None, :] >= 20,
            OBS_LANGUAGE_TOKENS: torch.ones(1, 2),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2),
        }
    ]
    monitor.noise = [torch.zeros(1, 50, 32)]
    monitor.probes = torch.ones(1, 1, 50, 7)
    monitor.manifest = {"samples": [{"task_index": 0}]}
    monitor.keyframe_masks = (torch.arange(50) < 5)[None, None, :]
    acc = SimpleNamespace(
        is_main_process=True,
        num_processes=1,
        process_index=0,
        device=torch.device("cpu"),
        autocast=nullcontext,
        gather=lambda x: x,
        reduce=lambda x, op: x,
    )
    state = torch.get_rng_state().clone()
    metrics = monitor.measure(policy, 0, acc)
    assert len(calls) == 1 and torch.equal(torch.get_rng_state(), state)
    assert metrics["fixed_training_probe_key_coordinates"] == 35
    assert metrics["fixed_training_probe_nonkey_coordinates"] == 105
    assert metrics["fixed_training_probe_key_mse"] == pytest.approx(metrics["fixed_training_probe_flow_mse"])
    # Saved diagnostic masks survive offline reload and are protected by a content hash.
    from lerobot.utils.prompt_rank import _fingerprint

    monitor.manifest.update(
        schema_version=1,
        options=asdict(monitor.options),
        action_dim=7,
        chunk_size=50,
        sample_sha256=_fingerprint(monitor.samples),
        keyframe_masks_sha256=arrays_hash({"key": monitor.keyframe_masks[:, 0, :].numpy()}),
    )
    monitor.save_protocol(0)
    restored = PromptRankMonitor.from_cache(tmp_path, "cpu")
    assert torch.equal(restored.keyframe_masks, monitor.keyframe_masks)
    torch.save(~monitor.keyframe_masks, tmp_path / "keyframe_masks.pt")
    with pytest.raises(ValueError, match="masks changed"):
        PromptRankMonitor.from_cache(tmp_path, "cpu")


def test_keyframe_config_roundtrip_rejects_confounded_recipe(tmp_path):
    import draccus

    directory, _, _, _ = prepared(tmp_path)
    settings = draccus.parse(KeyframeLossConfig, args=["--mode=K2", f"--archive={directory}/archive.npz"])
    assert draccus.decode(KeyframeLossConfig, draccus.encode(settings)) == settings
    cfg = SimpleNamespace(
        trainable_config=SimpleNamespace(
            type="pi05", chunk_size=50, num_vlm_prompt_tokens=2, pretrained_path="/base"
        ),
        dataset=SimpleNamespace(root=tmp_path / "dataset"),
        seed=0,
        steps=20,
        sample_weighting=None,
        resume=False,
        pdet=PDETConfig(
            enabled=True,
            prompt_seed=1,
            data_seed=0,
            noise_seed=1,
            time_seed=1,
            audit_steps=20,
            order_plan=str(directory / "D0-W0.npz"),
        ),
    )
    settings.validate_training(cfg)
    cfg.pdet.flow_binding = "slot"
    with pytest.raises(ValueError, match="occurrence"):
        settings.validate_training(cfg)
    cfg.pdet.flow_binding = "occurrence"
    cfg.trainable_config.pretrained_path = "/other-source"
    with pytest.raises(ValueError, match="original scoring"):
        settings.validate_training(cfg)


@pytest.mark.parametrize(
    "queue,stage,cells,gpus,port",
    [
        ("a", "baseline", [("B", 0)], "0,1", 29571),
        ("b", "baseline", [("B", 1)], "2,3", 29572),
        ("a", "weight", [("K2", 1), ("R2", 0)], "0,1", 29571),
        ("b", "weight", [("K2", 0), ("R2", 1)], "2,3", 29572),
        ("a", "curriculum", [("Kearly", 1), ("Klate", 0)], "0,1", 29571),
        ("b", "curriculum", [("Kearly", 0), ("Klate", 1)], "2,3", 29572),
    ],
)
def test_training_dispatch_uses_old_plan_final_only_and_never_overwrites(
    tmp_path, queue, stage, cells, gpus, port
):
    from tests.scripts.test_pi05_vlm_only import VLMOnlyTest

    fixture = VLMOnlyTest()
    fixture.setUp()
    try:
        directory, _, _, _ = prepared(
            tmp_path, production=True, root=fixture.dataset, source=fixture.source("base")
        )
        prompt = np.ones((2, 2048), dtype=np.float32)
        write_prompt(
            tmp_path / "inits/pi05-base-vlm2-init-seed1.safetensors",
            prompt,
            {
                "schema_version": 1,
                "source_step": 0,
                "source_training_seed": 1,
                "num_vlm_prompt_tokens": 2,
                "prompt_sha256": prompt_hash(prompt),
            },
        )
        env = {**fixture.env, "PROMPT_INIT_DIR": str(tmp_path / "inits"), "KEYFRAME_DIR": str(directory)}
        result = subprocess.run(
            ["bash", str(TRAIN), queue, stage], env=env, capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, result.stderr
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]
        assert len(commands) == len(cells)
        for command, (mode, bank) in zip(commands, cells, strict=True):
            for flag in (
                f"CUDA_VISIBLE_DEVICES={gpus}",
                f"--main_process_port={port}",
                "--num_processes=2",
                "--batch_size=32",
                "--steps=3000",
                "--save_steps=[]",
                "--policy.ntk_save_stage_snapshots=false",
                "--policy.compile_model=false",
                "--seed=0",
                "--pdet.prompt_seed=1",
                "--pdet.data_seed=0",
                "--pdet.flow_binding=occurrence",
                f"--pdet.noise_seed={bank}",
                f"--pdet.time_seed={bank}",
                f"--keyframe_loss.mode={mode}",
                f"--pdet.order_plan={directory}/D0-W0.npz",
                "--prompt_rank.fixed_flow_loss=true",
            ):
                assert flag in command
            assert f"--output_dir={env['OUTPUT_ROOT']}/{name(mode, bank)}" in command
        (Path(env["OUTPUT_ROOT"]) / name(*cells[-1])).mkdir(parents=True)
        blocked = subprocess.run(
            ["bash", str(TRAIN), queue, stage], env=env, capture_output=True, text=True, timeout=30
        )
        assert (
            blocked.returncode != 0
            and "Output already exists" in blocked.stderr
            and "Starting " not in blocked.stdout
        )
    finally:
        fixture.doCleanups()


def test_checkpoint_rejects_mislabeled_mode_and_eval_has_clean_progress_defaults(tmp_path):
    directory, arrays, archive, _ = prepared(tmp_path, production=True)
    run = tmp_path / name("K2", 1)
    save_archive(run / "keyframe_loss/archive.npz", arrays, archive)
    path = run / "checkpoints/003000/pretrained_model/train_config.json"
    path.parent.mkdir(parents=True)
    config = {
        "seed": 0,
        "steps": 3000,
        "keyframe_loss": {"mode": "K2"},
        "policy": {"num_vlm_prompt_tokens": 2},
        "pdet": asdict(
            PDETConfig(
                enabled=True,
                prompt_seed=1,
                data_seed=0,
                noise_seed=1,
                time_seed=1,
                order_plan=str(directory / "D0-W0.npz"),
                audit_steps=3000,
            )
        ),
    }
    path.write_text(json.dumps(config))
    assert check_checkpoint(run)["checkpoint"] == "name_and_config_match"
    config["keyframe_loss"]["mode"] = "R2"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="differs"):
        check_checkpoint(run)
    # The real delegate owns quiet progress and result parsing. Ensure valid help/invalid-cell handling.
    result = subprocess.run(["bash", str(EVAL), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "one live progress line" in result.stdout


def test_eval_delegates_all_four_suites_and_preserves_quiet_output(tmp_path):
    from tests.scripts.test_pi05_vlm_only import VLMOnlyTest

    fixture = VLMOnlyTest()
    fixture.setUp()
    try:
        directory, arrays, archive, _ = prepared(tmp_path, production=True)
        run = Path(fixture.env["CKPT_ROOT"]) / name("K2", 1)
        checkpoint = run / "checkpoints/003000"
        model = checkpoint / "pretrained_model"
        fixture.make_model(model, "base", tokens=2)
        save_archive(run / "keyframe_loss/archive.npz", arrays, archive)
        path = model / "train_config.json"
        config = json.loads(path.read_text())
        config["policy"]["num_vlm_prompt_tokens"] = 2
        config.update(
            seed=0,
            steps=3000,
            keyframe_loss={"mode": "K2"},
            pdet=asdict(
                PDETConfig(
                    enabled=True,
                    prompt_seed=1,
                    noise_seed=1,
                    time_seed=1,
                    order_plan=str(directory / "D0-W0.npz"),
                    audit_steps=3000,
                )
            ),
        )
        path.write_text(json.dumps(config))
        (checkpoint.parent / "last").symlink_to("003000")
        capture = fixture.install_fake_evaluator()
        result = subprocess.run(
            ["bash", str(EVAL), "weight", "all"],
            env={**fixture.env, "CELLS": "K2:1", "GPU_ID": "3", "DRY_RUN": "false"},
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr
        assert "overall: 90.00% (360/400)" in result.stdout
        assert "video-path.mp4" not in result.stdout
        calls = [json.loads(line) for line in capture.read_text().splitlines()]
        assert len(calls) == 1 and calls[0]["gpu"] == "3"
        assert calls[0]["--seed"] == "1000" and calls[0]["--eval.n_episodes"] == "10"
        assert name("K2", 1) in calls[0]["--output_dir"]
    finally:
        fixture.doCleanups()


def test_prepare_reuses_score_cache_and_copies_old_plan_without_new_occurrences(tmp_path):
    from lerobot.scripts.keyframe_experiment import prepare
    from lerobot.utils.frameskip_curriculum import save_scores
    from lerobot.utils.pdet_order import META_COLUMNS, load_plan
    from tests.scripts.test_pi05_vlm_only import VLMOnlyTest

    fixture = VLMOnlyTest()
    fixture.setUp()
    try:
        directory, arrays, archive, plan = prepared(
            tmp_path, production=True, root=fixture.dataset, source=fixture.source("base")
        )
        count = len(arrays["index"])
        scores = {
            "event_neighborhood": arrays["event_mask"],
            "score": np.zeros(count),
            **{key: np.ones(count, dtype=bool) for key in ("selected", "random_selected", "protected")},
        }
        scores_path = tmp_path / "scores.npz"
        save_scores(scores_path, {key: arrays[key] for key in META_COLUMNS}, scores, archive["score_recipe"])
        out = tmp_path / "new-prepared"
        result = prepare(scores_path, directory / "D0-W0.npz", out, fixture.dataset, fixture.source("base"))
        assert result["prepared"] == "passed"
        copied = load_plan(out / "D0-W0.npz")
        assert copied["manifest"] == plan["manifest"]
        assert all(np.array_equal(copied[key], plan[key]) for key in (*META_COLUMNS, "order"))
        assert (out / "schedule.csv").is_file() and (out / "summary.json").is_file()
        with pytest.raises(FileNotFoundError, match="Original orderbind"):
            prepare(scores_path, tmp_path / "missing.npz", out, fixture.dataset, fixture.source("base"))
    finally:
        fixture.doCleanups()
