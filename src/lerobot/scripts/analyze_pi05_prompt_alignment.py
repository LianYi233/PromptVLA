"""Replay small prompt snapshots and measure fixed-probe task-residual alignment.

Runs offline on one device, without a dataset/tokenizer or optimizer. Both the fitted
and held-out projections use the same observations; this is not task generalization.
"""

import argparse
import gc
import hashlib
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from lerobot.utils.prompt_rank_analysis import (
    read_json,
    residual_alignment,
    resolve_runs,
    snapshot,
    write_json,
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def collect_alignment(policy, monitor, heldout_probes=4, heldout_seed=2027, progress=True):
    import torch
    from tqdm import tqdm

    from lerobot.utils.piper_training_eval import evaluation_rng
    from lerobot.utils.prompt_rank import isolated_prompt

    generator = torch.Generator().manual_seed(heldout_seed)
    extra = torch.randint(0, 2, (heldout_probes, 1, monitor.horizon, monitor.action_dim), generator=generator)
    extra = (extra.float() * 2 - 1) / np.sqrt(heldout_probes)
    extra = extra.to(monitor.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(policy.config.dtype)
    autocast = (lambda: torch.autocast(monitor.device.type, dtype=dtype)) if dtype else nullcontext
    rows, residuals, held_rows, held_residuals = [], [], [], []
    squared_error, elements = 0.0, 0
    pairs = [(i, t) for i in range(len(monitor.samples)) for t in monitor.options.timesteps]
    with evaluation_rng(monitor.device, monitor.options.seed), isolated_prompt(policy) as prompt:
        for index, timestep in tqdm(pairs, desc="Residual/Jacobian", disable=not progress, leave=False):
            with torch.enable_grad(), autocast():
                velocity, target, probes, valid = monitor._probe_forward(policy, index, timestep)
                error = target.float() - velocity.float()
                squared_error += float((error.detach().square() * valid[..., None]).sum())
                elements += int(valid.sum()) * monitor.action_dim
                projections = torch.cat([probes, extra * valid[None, :, :, None]])
                block_rows, block_residuals = [], []
                for j, projection in enumerate(projections):
                    (gradient,) = torch.autograd.grad(
                        (velocity.float() * projection).sum(), prompt, retain_graph=j + 1 < len(projections)
                    )
                    block_rows.append(gradient.detach().float().flatten().cpu().numpy())
                    block_residuals.append(float((error.detach() * projection).sum()))
            r = monitor.options.output_probes
            rows.extend(block_rows[:r])
            residuals.extend(block_residuals[:r])
            held_rows.extend(block_rows[r:])
            held_residuals.extend(block_residuals[r:])
            del velocity, target, error, gradient
    g, h = np.asarray(rows, dtype=np.float64), np.asarray(held_rows, dtype=np.float64)
    arrays = {
        "gram": g @ g.T,
        "cross_gram": h @ g.T,
        "heldout_gram": h @ h.T,
        "residual": np.asarray(residuals),
        "heldout_residual": np.asarray(held_residuals),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Non-finite residual or tangent kernel")
    return arrays, squared_error / elements


def load_policy(run, checkpoint_step, device):
    import torch

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts.analyze_pi05_ntk_stages import load_pi05_config
    from lerobot.scripts.libero_vlm_only import validate_checkpoint

    checkpoint = run["path"] / f"checkpoints/{checkpoint_step:06d}/pretrained_model"
    validate_checkpoint(checkpoint, run["tokens"])
    training = read_json(checkpoint / "train_config.json")
    if training["seed"] != run["seed"]:
        raise ValueError("Checkpoint training seed does not match the run")
    if read_json(checkpoint.parent / "training_state/training_step.json")["step"] != checkpoint_step:
        raise ValueError("Checkpoint step does not match the requested backbone")
    config = load_pi05_config(checkpoint)
    if config.dtype != run["manifest"]["forward_precision"]:
        raise ValueError("Checkpoint precision differs from the recorded probe")
    config.device, config.compile_model, config.gradient_checkpointing = device, False, False
    policy = PI05Policy.from_pretrained(checkpoint, config=config, strict=True, local_files_only=True).eval()
    # The final prompt check catches absent/incorrect compatibility-loaded weights.
    saved = snapshot(run, checkpoint_step)["prompt"]
    actual = policy.model.vlm_prompt_tokens.weight.detach().float().cpu().numpy()
    if not np.array_equal(actual, saved):
        raise ValueError("Loaded final prompt differs from the recorded final snapshot")
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # Match lerobot_train.
    return policy


def analyze(args):
    import torch

    from lerobot.utils.prompt_rank import PromptRankMonitor, _fingerprint

    if (
        args.heldout_probes < 1
        or args.heldout_seed < 0
        or not np.isfinite(args.ridge_relative)
        or args.ridge_relative <= 0
        or not np.isfinite(args.gram_rtol)
        or args.gram_rtol <= 0
        or not args.steps
        or min(args.steps) < 0
        or max(args.steps) > args.checkpoint_step
        or len(set(args.steps)) != len(args.steps)
    ):
        raise ValueError("Invalid analysis settings")
    if Path(args.analysis_name).name != args.analysis_name or args.analysis_name in {
        "",
        ".",
        "..",
        "prompt_rank",
        "checkpoints",
    }:
        raise ValueError("--analysis-name must be a separate child directory name")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose --device=cpu explicitly for CPU analysis")
    runs = resolve_runs(args.ckpt_root, args.tokens, args.seed, args.run_prefix, args.source)
    common_hash = None
    for run in runs:
        directory = run["path"] / "prompt_rank"
        monitor = PromptRankMonitor.from_cache(directory, args.device)
        cache_hash = _fingerprint([{"probes": monitor.probes}, *[{"noise": x} for x in monitor.noise]])
        if common_hash is not None and cache_hash != common_hash:
            raise ValueError("Noise/projection caches differ across token counts")
        common_hash = cache_hash
        for step in args.steps:
            snapshot(run, step)  # Validate all selected snapshots before loading this model.
        checkpoint = run["path"] / f"checkpoints/{args.checkpoint_step:06d}/pretrained_model"
        weight = checkpoint / "model.safetensors"
        protocol = {
            "schema_version": 1,
            "run": run["name"],
            "torch_version": str(torch.__version__),
            "checkpoint": str(checkpoint.resolve()),
            "weight_size": weight.stat().st_size,
            "weight_mtime_ns": weight.stat().st_mtime_ns,
            "config_sha256": digest(checkpoint / "config.json"),
            "probe_manifest_sha256": digest(directory / "manifest.json"),
            "fixed_probe_sha256": digest(directory / "fixed_probe.pt"),
            "device_type": torch.device(args.device).type,
            "noise_projection_sha256": cache_hash,
            "heldout_probes": args.heldout_probes,
            "heldout_seed": args.heldout_seed,
            "ridge_relative": args.ridge_relative,
            "gram_rtol": args.gram_rtol,
            "scope": "velocity residual; independent output projections on SAME observations/noise; linear prediction only",
        }
        output = run["path"] / args.analysis_name
        output.mkdir(parents=True, exist_ok=True)
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise ValueError(f"Analysis protocol changed; choose a new --analysis-name: {output}")
        write_json(protocol_path, protocol)
        policy = None
        try:
            for step in args.steps:
                destination = output / f"step_{step:06d}"
                source = directory / f"step_{step:06d}.npz"
                snapshot_hash = digest(source)
                if destination.with_suffix(".json").exists() and destination.with_suffix(".npz").exists():
                    if read_json(destination.with_suffix(".json"))["snapshot_sha256"] != snapshot_hash:
                        raise ValueError("Prompt snapshot changed after analysis")
                    logging.info("Already complete: %s step=%d", run["name"], step)
                    continue
                if policy is None:
                    policy = load_policy(run, args.checkpoint_step, args.device)
                saved = snapshot(run, step)
                with torch.no_grad():
                    policy.model.vlm_prompt_tokens.weight.copy_(torch.from_numpy(saved["prompt"]))
                logging.info("Analyzing %s step=%d", run["name"], step)
                started = time.perf_counter()
                arrays, mse = collect_alignment(
                    policy, monitor, args.heldout_probes, args.heldout_seed, not args.no_progress
                )
                error = np.linalg.norm(arrays["gram"] - saved["gram"]) / max(
                    np.linalg.norm(saved["gram"]), 1e-30
                )
                if error > args.gram_rtol:
                    raise ValueError(
                        f"Replay Gram differs by {error:.3g}; inspect precision/model/probe provenance"
                    )
                scores = residual_alignment(
                    arrays["gram"],
                    arrays["residual"],
                    arrays["cross_gram"],
                    arrays["heldout_residual"],
                    args.ridge_relative,
                )
                scores.update(
                    step=step,
                    tokens=run["tokens"],
                    seed=run["seed"],
                    run=run["name"],
                    snapshot_sha256=snapshot_hash,
                    gram_relative_error=float(error),
                    velocity_mse=mse,
                    seconds=time.perf_counter() - started,
                    linear_delta_relative_norm=scores["linear_delta_norm"]
                    / max(float(np.linalg.norm(saved["prompt"])), 1e-30),
                )
                temporary = destination.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **arrays)
                temporary.replace(destination.with_suffix(".npz"))
                write_json(destination.with_suffix(".json"), scores)
                logging.info(
                    "step=%d MSE=%.5g top16=%s held-out linear reduction=%s",
                    step,
                    mse,
                    scores["top16_alignment"],
                    scores["heldout_linear_reduction"],
                )
        finally:
            del policy, monitor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-root", type=Path, default=Path("/root/autodl-tmp/chkpt/2601-lerobot/vlm-only")
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source", choices=["base", "finetuned"], default="base")
    parser.add_argument("--run-prefix", default="pi05")
    parser.add_argument("--checkpoint-step", type=int, default=3000)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1500, 3000])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heldout-probes", type=int, default=4)
    parser.add_argument("--heldout-seed", type=int, default=2027)
    parser.add_argument("--ridge-relative", type=float, default=0.01)
    parser.add_argument("--gram-rtol", type=float, default=0.02)
    parser.add_argument("--analysis-name", default="prompt_alignment")
    parser.add_argument("--no-progress", action="store_true")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
