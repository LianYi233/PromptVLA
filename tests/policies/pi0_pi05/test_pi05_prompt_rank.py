"""Fixed-probe repeatability and non-interference through the real PI05 path."""

import json
import os
import random
import socket
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.multiprocessing.spawn import ProcessRaisedException

from lerobot.configs.prompt_rank import PromptRankConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.prompt_rank import PromptRankMonitor, isolated_prompt, tangent_spectrum
from tests.policies.pi0_pi05.test_pi05_prompt import (
    PI05Policy,
    _config,
    _tiny_real_pi05,  # noqa: F401 -- real PI05 with reduced transformer dimensions
)

CAMERA = "observation.images.camera"


class ProbeDataset:
    image_transforms = "training augmentation"
    root = "fixed-test-dataset"
    meta = SimpleNamespace(camera_keys=[CAMERA])

    def __len__(self):
        return 5

    def clear_image_transforms(self):
        self.image_transforms = None

    def set_image_transforms(self, value):
        self.image_transforms = value

    def __getitem__(self, index):
        assert self.image_transforms is None
        return {
            CAMERA: torch.rand(3, 16, 16),
            "action": torch.randn(4, 3),
            "action_is_pad": torch.tensor([False, False, False, index % 2 == 0]),
            OBS_LANGUAGE_TOKENS: torch.tensor([1, 2, 3]),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(3, dtype=torch.bool),
            "index": index,
            "task": f"task {index}",
        }


def identity(batch):
    return batch


class LocalAccelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1
    device = torch.device("cpu")
    autocast = staticmethod(nullcontext)

    @staticmethod
    def reduce(value, reduction):
        return value

    @staticmethod
    def gather(value):
        return value


def make_monitor(policy, tmp_path):
    return PromptRankMonitor(
        ProbeDataset(),
        identity,
        policy.config,
        PromptRankConfig(freq=250, samples=3, output_probes=2, timesteps=[0.2, 0.8]),
        tmp_path,
        "cpu",
    )


def make_policy(dtype="float32"):
    return PI05Policy(
        _config(
            dtype=dtype,
            vlm_prompt_only=True,
            num_prompt_tokens=0,
            cabo_enabled=False,
            next_action_pretrain_steps=0,
            next_action_bridge_steps=0,
            gradient_checkpointing=True,
            input_features={CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16))},
        )
    )


def test_known_spectrum_distinguishes_jacobian_from_kernel_entropy():
    metrics, gram, eigs, sigmas = tangent_spectrum(np.diag([3.0, 1.0, 0.0]))
    np.testing.assert_allclose(eigs, [9, 1, 0])
    np.testing.assert_allclose(sigmas, [3, 1, 0])
    assert metrics["jacobian_erank"] == pytest.approx(np.exp(-0.75 * np.log(0.75) - 0.25 * np.log(0.25)))
    assert metrics["ntk_erank"] == pytest.approx(np.exp(-0.9 * np.log(0.9) - 0.1 * np.log(0.1)))
    assert metrics["stable_rank"] == pytest.approx(10 / 9)
    assert metrics["rank95_energy"] == 2
    assert metrics["tangent_trace"] == 10
    assert not tangent_spectrum(np.zeros((4, 3)))[0]["jacobian_erank"]
    with pytest.raises(ValueError):
        tangent_spectrum([[float("nan")]])


def test_nested_cli_configuration_and_invalid_probe_settings():
    import draccus

    from lerobot.configs.train import TrainPipelineConfig

    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=libero",
            "--prompt_rank.freq=250",
            "--prompt_rank.samples=16",
            "--prompt_rank.output_probes=4",
            "--prompt_rank.timesteps=[0.2,0.5,0.8]",
        ],
    )
    config.prompt_rank.validate()
    assert config.prompt_rank.timesteps == [0.2, 0.5, 0.8]
    assert config.prompt_rank.freq == 250
    for options in (PromptRankConfig(samples=0), PromptRankConfig(timesteps=[0.5, 0.5])):
        with pytest.raises(ValueError):
            options.validate()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_measurement_preserves_training_state_rng_gradients_and_optimizer_step(tmp_path, dtype):
    policy = make_policy(dtype).train()
    prompt = policy.model.vlm_prompt_tokens.weight
    prompt.grad = torch.full_like(prompt, 0.123)
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    flags = [module.training for module in policy.modules()]
    state = torch.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    monitor = make_monitor(policy, tmp_path)
    assert ProbeDataset.image_transforms == "training augmentation"
    monitor.save_protocol(0)
    first = monitor.measure(policy, 0, LocalAccelerator())
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert policy.model.vlm_prompt_tokens.weight is prompt
    torch.testing.assert_close(prompt.grad, torch.full_like(prompt, 0.123), rtol=0, atol=0)
    assert flags == [module.training for module in policy.modules()]
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    second = monitor.measure(policy, 1, LocalAccelerator())
    assert 1 <= first["jacobian_erank"] <= first["rank_ceiling"] == 12
    assert first["tangent_trace"] > 0
    assert first["jacobian_erank"] == second["jacobian_erank"]
    with (
        np.load(monitor.output_dir / "step_000000.npz") as left,
        np.load(monitor.output_dir / "step_000001.npz") as right,
    ):
        np.testing.assert_array_equal(left["gram"], right["gram"])
    # A real training backward still reaches the original optimizer-owned leaf.
    sample = monitor.samples[0]
    optimizer = torch.optim.SGD([prompt], lr=0.001)
    optimizer.zero_grad()
    policy(sample)[0].backward()
    assert prompt.grad is not None and prompt.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(prompt, before["model.vlm_prompt_tokens.weight"])
    assert len((monitor.output_dir / "metrics.jsonl").read_text().splitlines()) == 2
    assert json.loads((monitor.output_dir / "manifest.json").read_text())["projected_rows"] == 12


def test_probe_failure_restores_parameter_identity_and_flags(tmp_path, monkeypatch):
    policy = make_policy().train()
    monitor = make_monitor(policy, tmp_path)
    original = policy.model.vlm_prompt_tokens.weight
    flags = [module.training for module in policy.modules()]
    state = torch.get_rng_state().clone()

    def fail(*args, **kwargs):
        raise ValueError("deliberate probe failure")

    monkeypatch.setattr(policy.model, "predict_velocity", fail)
    with pytest.raises(RuntimeError, match="deliberate probe failure"):
        monitor.measure(policy, 0, LocalAccelerator())
    assert policy.model.vlm_prompt_tokens.weight is original
    assert flags == [module.training for module in policy.modules()]
    torch.testing.assert_close(torch.get_rng_state(), state)


def test_same_probe_across_token_counts_and_no_mutation_of_model_output(tmp_path):
    policy = make_policy()
    monitor = make_monitor(policy, tmp_path / "first")
    config = _config(
        vlm_prompt_only=True,
        num_prompt_tokens=0,
        cabo_enabled=False,
        next_action_pretrain_steps=0,
        next_action_bridge_steps=0,
        num_vlm_prompt_tokens=1,
        input_features={CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16))},
    )
    other = make_monitor(PI05Policy(config), tmp_path / "second")
    assert monitor.manifest["sample_sha256"] == other.manifest["sample_sha256"]
    torch.testing.assert_close(monitor.probes, other.probes, rtol=0, atol=0)
    for a, b in zip(monitor.noise, other.noise, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sample = monitor.samples[0]
    policy.eval()
    images, masks = policy._preprocess_images(sample)
    args = (
        images,
        masks,
        sample[OBS_LANGUAGE_TOKENS],
        sample[OBS_LANGUAGE_ATTENTION_MASK],
        monitor.noise[0],
        torch.tensor([0.5]),
    )
    expected = policy.model.predict_velocity(*args).detach()
    with isolated_prompt(policy):
        actual = policy.model.predict_velocity(*args).detach()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_offline_cache_replay_and_residual_sign_match_real_velocity(tmp_path, dtype):
    from lerobot.scripts.analyze_pi05_prompt_alignment import collect_alignment

    policy = make_policy(dtype).train()
    original = policy.model.vlm_prompt_tokens.weight
    before = original.detach().clone()
    monitor = make_monitor(policy, tmp_path)
    monitor.save_protocol(0)
    replay = PromptRankMonitor.from_cache(monitor.output_dir, "cpu")
    arrays, mse = collect_alignment(policy, replay, heldout_probes=2, progress=False)
    assert policy.model.vlm_prompt_tokens.weight is original
    torch.testing.assert_close(original, before, rtol=0, atol=0)
    assert mse > 0

    # Match the exact same autocast path as the standalone replay.
    class NativeAccelerator(LocalAccelerator):
        @staticmethod
        def autocast():
            return torch.autocast("cpu", dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext()

    replay.measure(policy, 0, NativeAccelerator())
    with np.load(monitor.output_dir / "step_000000.npz") as saved:
        np.testing.assert_allclose(arrays["gram"], saved["gram"], rtol=1e-12, atol=1e-12)
    with isolated_prompt(policy), torch.enable_grad(), NativeAccelerator.autocast():
        velocity, target, probes, valid = replay._probe_forward(policy, 0, replay.options.timesteps[0])
        projected = [
            (float(((target.float() - velocity.float()).detach() * probe).sum())) for probe in probes
        ]
    np.testing.assert_allclose(arrays["residual"][:2], projected, rtol=0, atol=0)
    assert not valid[0, -1]  # This sample includes padding, masked by the stored projections.
    assert arrays["cross_gram"].shape == arrays["gram"].shape

    cache_path = monitor.output_dir / "fixed_probe.pt"
    cache = torch.load(cache_path, weights_only=True)
    cache["samples"][0]["action"][0, 0, 0] += 1
    torch.save(cache, cache_path)
    with pytest.raises(ValueError, match="hash"):
        PromptRankMonitor.from_cache(monitor.output_dir, "cpu")


def test_offline_analysis_restores_three_prompts_and_resumes_without_model_reload(tmp_path, monkeypatch):
    from copy import deepcopy

    from lerobot.scripts import analyze_pi05_prompt_alignment as script

    policy = make_policy().train()
    count = policy.config.num_vlm_prompt_tokens
    run = tmp_path / f"pi05-base-vlm{count}-seed1"
    monitor = make_monitor(policy, run)
    monitor.save_protocol(0)
    for step in (0, 1500, 3000):
        with torch.no_grad():
            policy.model.vlm_prompt_tokens.weight.add_(0.01)
        monitor.measure(policy, step, LocalAccelerator())
    checkpoint = run / "checkpoints/003000/pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"test loader supplies tiny actual PI05")
    calls = []

    def load(*args):
        calls.append(args)
        return deepcopy(policy).eval()

    monkeypatch.setattr(script, "load_policy", load)
    args = SimpleNamespace(
        ckpt_root=tmp_path,
        tokens=[count],
        seed=1,
        source="base",
        run_prefix="pi05",
        steps=[0, 1500, 3000],
        checkpoint_step=3000,
        heldout_probes=2,
        heldout_seed=2027,
        ridge_relative=0.01,
        gram_rtol=0.02,
        device="cpu",
        no_progress=True,
        analysis_name="prompt_alignment",
    )
    script.analyze(args)
    script.analyze(args)
    assert len(calls) == 1
    for step in args.steps:
        scores = json.loads((run / f"prompt_alignment/step_{step:06d}.json").read_text())
        assert scores["gram_relative_error"] < 1e-12
        assert scores["step"] == step
    args.heldout_seed += 1
    with pytest.raises(ValueError, match="protocol changed"):
        script.analyze(args)


def test_sharded_probe_order_matches_single_process_with_uneven_last_shard(tmp_path):
    policy = make_policy().train()
    single = make_monitor(policy, tmp_path / "single")
    single.save_protocol(0)
    single.measure(policy, 0, LocalAccelerator())
    sharded = make_monitor(policy, tmp_path / "sharded")
    sharded.save_protocol(0)

    class ReplayPeer(LocalAccelerator):
        num_processes = 2
        calls = 0

        def gather(self, local):
            block, t = divmod(self.calls, len(sharded.options.timesteps))
            peer = block * 2 + 1
            self.calls += 1
            if peer < len(sharded.samples):
                other = sharded._rows(
                    policy,
                    policy.model.vlm_prompt_tokens.weight,
                    peer,
                    sharded.options.timesteps[t],
                    self.autocast,
                )
            else:
                other = torch.zeros_like(local)
            return torch.cat([local, other])

    sharded.measure(policy, 0, ReplayPeer())
    with (
        np.load(single.output_dir / "step_000000.npz") as expected,
        np.load(sharded.output_dir / "step_000000.npz") as actual,
    ):
        np.testing.assert_array_equal(actual["gram"], expected["gram"])


def _ddp_worker(rank, root):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    root = Path(root)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        # The dynamic PiGemma compatibility class is intentionally not pickleable.
        # Construct the same tiny real model inside each spawned process.
        patch = pytest.MonkeyPatch()
        _tiny_real_pi05.__wrapped__(patch)
        policy = make_policy().train()
        accelerator = Accelerator(
            cpu=True, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )
        optimizer = torch.optim.SGD([policy.model.vlm_prompt_tokens.weight], lr=0.001)
        wrapped, optimizer = accelerator.prepare(policy, optimizer)
        bare = accelerator.unwrap_model(wrapped)
        original = bare.model.vlm_prompt_tokens.weight
        monitor = make_monitor(bare, root / "distributed")
        monitor.verify_across_ranks(accelerator)
        if rank == 0:
            monitor.save_protocol(0)
        monitor.measure(bare, 0, accelerator)
        # Exercise a monitor between real DDP backward/update cycles, not just a forward-only mock.
        for step in (1, 2):
            wrapped.train()
            optimizer.zero_grad()
            loss = wrapped(monitor.samples[rank])[0]
            accelerator.backward(loss)
            optimizer.step()
            if step == 1:
                previous_grad = original.grad.clone()
                monitor.measure(bare, step, accelerator)
                torch.testing.assert_close(original.grad, previous_grad, atol=0, rtol=0)
        assert bare.model.vlm_prompt_tokens.weight is original
        # One-rank Python failures must be reported on both ranks before gather, without deadlock.
        if rank == 1:

            def fail(*args, **kwargs):
                raise ValueError("one-rank probe failure")

            bare.model.predict_velocity = fail
        try:
            monitor.measure(bare, 2, accelerator)
        except RuntimeError as exc:
            assert "Prompt-rank diagnostic failed" in str(exc)
        else:
            raise AssertionError("Expected collective failure")
        assert bare.model.vlm_prompt_tokens.weight is original
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_probes_match_single_rank_and_training_continues(tmp_path):
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("This environment forbids local TCP sockets required by Gloo DDP")
    policy = make_policy().train()
    monitor = make_monitor(policy, tmp_path / "single")
    monitor.save_protocol(0)
    monitor.measure(policy, 0, LocalAccelerator())
    try:
        torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
    except ProcessRaisedException as exc:
        if "Operation not permitted" in str(exc) and "ProcessGroupGloo" in str(exc):
            pytest.skip("This environment blocks Gloo TCP device initialization")
        raise
    with (
        np.load(tmp_path / "single/prompt_rank/step_000000.npz") as single,
        np.load(tmp_path / "distributed/prompt_rank/step_000000.npz") as distributed,
    ):
        np.testing.assert_allclose(single["gram"], distributed["gram"], rtol=1e-6, atol=1e-10)
