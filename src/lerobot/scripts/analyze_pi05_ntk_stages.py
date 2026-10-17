"""Compare PI05 NTKs before training, after priming/bridge, and after final adaptation.

Run with ``python -m lerobot.scripts.analyze_pi05_ntk_stages --help``.
See examples/analysis/README_pi05_ntk_stages.md for the protocol and checkpoint layout.
"""

import argparse
import copy
import gc
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from lerobot.utils.ntk import (
    enable_analysis_gradients,
    parameter_groups,
    projected_jacobian_rows,
    spectral_metrics,
)

STAGES = ("before", "priming", "stage2", "final")
COMPARABLE_FIELDS = (
    "paligemma_variant",
    "action_expert_variant",
    "num_vlm_prompt_tokens",
    "num_prompt_tokens",
    "chunk_size",
    "max_action_dim",
    "max_state_dim",
    "image_resolution",
    "input_features",
    "output_features",
    "normalization_mapping",
    "tokenizer_max_length",
    "min_period",
    "max_period",
    "time_sampling_beta_alpha",
    "time_sampling_beta_beta",
    "time_sampling_scale",
    "time_sampling_offset",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_dir(path):
    path = Path(path).expanduser().resolve()
    return path / "pretrained_model" if (path / "pretrained_model").is_dir() else path


def resolve_stages(args):
    selected = STAGES[: STAGES.index(args.through_stage) + 1]
    if args.final_flow_steps <= 0:
        raise ValueError("--final-flow-steps must be positive (2000 means 3000 cumulative updates)")
    paths = {name: getattr(args, name) for name in STAGES}
    if args.run_dir:
        run = Path(args.run_dir).expanduser()
        pretrain = run.parent / f"{run.name}_next_action_pretrain"
        defaults = {
            "before": pretrain / "checkpoints/000000",
            "priming": pretrain / "checkpoints/000750",
            "stage2": pretrain / "checkpoints/001000",
            "final": run / f"checkpoints/{args.final_flow_steps:06d}",
        }
        paths = {key: paths[key] or defaults[key] for key in STAGES}
    if any(paths[name] is None for name in selected):
        raise ValueError("Supply --run-dir or explicit checkpoint paths for every selected stage")
    local_steps = (0, 750, 1000, args.final_flow_steps)
    total_steps = (0, 750, 1000, 1000 + args.final_flow_steps)
    stages = []
    for name, local_step, total_step in zip(STAGES, local_steps, total_steps, strict=True):
        if name not in selected:
            continue
        path = checkpoint_dir(paths[name])
        for filename in ("config.json", "train_config.json", "model.safetensors", "policy_preprocessor.json"):
            if not (path / filename).is_file():
                raise FileNotFoundError(
                    f"{name}: missing {path / filename}. Exact 0/750 snapshots require training with "
                    "--policy.ntk_save_stage_snapshots=true. Existing flow-3000 runs require "
                    "--final-flow-steps=3000 (cumulative step 4000)."
                )
        step_file = path.parent / "training_state/training_step.json"
        if not step_file.is_file():
            raise FileNotFoundError(f"Cannot verify checkpoint step: {step_file}")
        saved_step = read_json(step_file)["step"]
        if saved_step != local_step:
            raise ValueError(f"{name}: checkpoint has {saved_step} local updates, expected {local_step}")
        config = read_json(path / "config.json")
        expected_stage = "flow" if name == "final" else "next_action"
        if config.get("training_stage") != expected_stage:
            raise ValueError(
                f"{name}: expected {expected_stage} checkpoint, got {config.get('training_stage')}"
            )
        train_config = read_json(path / "train_config.json")
        if name != "final" and (
            train_config["steps"] != 1000 or config.get("next_action_bridge_steps") != 250
        ):
            raise ValueError(f"{name}: this protocol requires 750 priming + 250 bridge updates")
        if name == "final" and config.get("next_action_pretrain_steps") != 1000:
            raise ValueError("Final checkpoint must follow the same integrated 1000-step pretraining")
        stages.append(
            {
                "name": name,
                "local_step": local_step,
                "total_step": total_step,
                "checkpoint": str(path),
                "config": config,
            }
        )
    if len({stage["checkpoint"] for stage in stages}) != len(stages):
        raise ValueError("The selected checkpoints must be distinct")
    for field in COMPARABLE_FIELDS:
        if any(stage["config"].get(field) != stages[0]["config"].get(field) for stage in stages[1:]):
            raise ValueError(f"Incomparable checkpoint configuration: {field}")
    return stages


def validate_manifest_extension(previous, current):
    """Reuse completed stages only when all measurement settings stay identical."""
    previous_protocol = {key: value for key, value in previous.items() if key != "stages"}
    current_protocol = {key: value for key, value in current.items() if key != "stages"}
    old_stages, new_stages = previous["stages"], current["stages"]
    if previous_protocol != current_protocol or old_stages != new_stages[: len(old_stages)]:
        raise ValueError(
            "Output protocol/checkpoints have changed. Only appending later stages is supported; "
            "use a new --output-dir for other changes"
        )


def file_identity(path):
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fingerprint_samples(samples):
    import torch

    digest = hashlib.sha256()
    for sample in samples:
        for name, value in sorted(sample.items()):
            digest.update(name.encode())
            if isinstance(value, torch.Tensor):
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.contiguous().numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def load_pi05_config(path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    # The base registry consumes the saved `type` discriminator; calling the
    # concrete PI05Config loader directly fails with the repository's draccus.
    config = PreTrainedConfig.from_pretrained(path)
    if not isinstance(config, PI05Config):
        raise ValueError(f"Expected a PI05 checkpoint: {path}")
    return config


def load_samples(args, stages):
    import torch
    from torch.utils.data import default_collate

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    reference = stages[0]["checkpoint"]
    config = load_pi05_config(reference)
    config.device = "cpu"
    config.compile_model = False
    overrides = {"device_processor": {"device": "cpu"}}
    if args.tokenizer_path:
        overrides["tokenizer_processor"] = {"tokenizer_name": str(Path(args.tokenizer_path).expanduser())}
    processor, _ = make_pre_post_processors(
        config,
        pretrained_path=reference,
        preprocessor_overrides=overrides,
    )
    # Freeze the preprocessing protocol across checkpoints, including normalization statistics.
    # Loading the saved processor avoids recomputing or uploading dataset quantiles.
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
        video_backend=args.video_backend,
        return_uint8=True,
    )
    indices = (
        [int(value) for value in args.indices.split(",")] if args.indices else list(range(args.num_samples))
    )
    if (
        len(indices) < 2
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= len(dataset)
    ):
        raise ValueError("Need at least two distinct valid dataset indices")
    samples = []
    for index in indices:
        batch = default_collate([dataset[index]])
        for camera in metadata.camera_keys:
            if camera in batch and batch[camera].dtype == torch.uint8:
                batch[camera] = batch[camera].float() / 255
        samples.append(processor(batch))
    return samples, indices


def load_policy(stage, scope, device):
    from safetensors import safe_open

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    path = stage["checkpoint"]
    # PI05's compatibility loader initializes absent prompts. Never allow that for a time series.
    with safe_open(str(Path(path) / "model.safetensors"), framework="pt", device="cpu") as handle:
        saved_keys = set(handle.keys())
        for key in ("model.vlm_prompt_tokens.weight", "model.prompt_tokens.weight"):
            if key not in saved_keys:
                raise ValueError(
                    f"{stage['name']}: missing learned/initialized prompt {key}; save exact snapshots"
                )
    config = load_pi05_config(path)
    config.device = device
    config.dtype = "float32"
    config.compile_model = False
    config.gradient_checkpointing = False
    config.separate_frozen_observations = False
    config.attention_implementation = "sdpa"
    policy = PI05Policy.from_pretrained(path, config=config, strict=True)
    groups = parameter_groups(policy, scope)
    enable_analysis_gradients(policy, groups)
    return policy, groups


def analyze_seed(policy, groups, samples, seed, args):
    import torch
    from tqdm import tqdm

    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    device = torch.device(args.device)
    action_dim = samples[0]["action"].shape[-1]
    horizon = policy.config.chunk_size
    if action_dim > policy.config.max_action_dim:
        raise ValueError("Dataset action dimension exceeds the model action dimension")
    # A shared output probe across samples is essential: independent probes erase off-diagonal NTK entries.
    probe_rng = torch.Generator(device="cpu").manual_seed(args.probe_seed + seed)
    probes = torch.randint(0, 2, (args.output_probes, 1, horizon, action_dim), generator=probe_rng)
    probes = (probes * 2 - 1).to(device=device, dtype=torch.float32)
    rows = {group: [] for group in groups}
    traces = dict.fromkeys(groups, 0.0)
    for sample_index, source in enumerate(tqdm(samples, desc=f"NTK seed={seed}", unit="sample")):
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in source.items()
        }
        # Reset per sample independently of model initialization, checkpoint order or skipped seeds.
        noise_rng = torch.Generator(device="cpu").manual_seed(seed * 1000003 + sample_index)
        actions = policy.prepare_action(batch)
        noise = torch.randn(actions.shape, generator=noise_rng).to(device)
        time_rng = np.random.default_rng(seed * 1000003 + sample_index)
        time_value = time_rng.beta(
            policy.config.time_sampling_beta_alpha, policy.config.time_sampling_beta_beta
        )
        time_value = time_value * policy.config.time_sampling_scale + policy.config.time_sampling_offset
        time = torch.tensor([time_value], device=device, dtype=torch.float32)
        noisy_actions = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        images, image_masks = policy._preprocess_images(batch)
        output = policy.model.predict_velocity(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            noisy_actions,
            time,
        )[..., :action_dim]
        # Omit padded action dimensions and timesteps, keeping a common coordinate system.
        valid = ~batch.get("action_is_pad", torch.zeros((1, horizon), dtype=torch.bool, device=device)).bool()
        if not valid.any():
            raise ValueError(f"Sample {sample_index} contains no valid action timesteps")
        sample_probes = probes * valid[None, :, :, None]
        sample_rows, sample_traces = projected_jacobian_rows(
            output,
            sample_probes,
            groups,
            args.sketch_dim,
            args.sketch_seed,
        )
        for group in groups:
            rows[group].append(sample_rows[group])
            traces[group] += sum(sample_traces[group]) / args.output_probes
        del output, images, batch, sample_rows
    result = {}
    for group, members in groups.items():
        # rows: samples x output-probes x parameter-coordinates (or sketch coordinates).
        features = np.asarray(rows[group], dtype=np.float64)
        kernel = np.einsum("iqp,jqp->ij", features, features) / args.output_probes
        metrics = spectral_metrics(kernel, sum(p.numel() for _, p in members), traces[group])
        metrics["kernel"] = kernel.tolist()
        metrics["parameter_representation"] = "exact" if group.startswith("prompts/") else "countsketch"
        result[group] = metrics
        if metrics["tangent_trace"] == 0:
            raise RuntimeError(f"Disconnected NTK group {group}: zero gradients on every probe")
        if metrics["sketch_trace_relative_error"] > 0.1:
            logging.warning("%s sketch trace differs by >10%%; increase --sketch-dim", group)
    return result


def run(args):
    import torch

    stages = resolve_stages(args)
    print("Checkpoint timeline (completed updates):")
    for stage in stages:
        print(
            f"  {stage['name']:8s} cumulative={stage['total_step']:4d} "
            f"local={stage['local_step']:4d}  {stage['checkpoint']}"
        )
    if args.dry_run:
        return
    if not args.dataset_repo_id or not args.dataset_root:
        raise ValueError("--dataset-repo-id and --dataset-root are required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; activate the training environment and check the selected GPU"
        )
    if min(args.num_samples, args.output_probes, args.sketch_dim) < 1:
        raise ValueError("Sample, probe and sketch counts must be positive")
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    samples, indices = load_samples(args, stages)
    manifest = {
        "schema": 1,
        "protocol": "shared-Rademacher output NTK; full conditioned velocity at all stages; entropy rank",
        "scope": args.scope,
        "dtype": "float32",
        "device": args.device,
        "torch_version": torch.__version__,
        "seeds": seeds,
        "output_probes": args.output_probes,
        "probe_seed": args.probe_seed,
        "sketch_dim": args.sketch_dim,
        "sketch_seed": args.sketch_seed,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "sample_indices": indices,
        "sample_digest": fingerprint_samples(samples),
        "source_digest": hashlib.sha256(
            Path(__file__).read_bytes()
            + b"".join(
                (Path(__file__).parents[1] / name).read_bytes()
                for name in (
                    "utils/ntk.py",
                    "policies/pi05/modeling_pi05.py",
                    "policies/pi_gemma.py",
                    "policies/pi05/attention_pi05.py",
                )
            )
        ).hexdigest(),
        "stages": [
            {**stage, "weights": file_identity(Path(stage["checkpoint"]) / "model.safetensors")}
            for stage in stages
        ],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        validate_manifest_extension(read_json(manifest_path), manifest)
    write_json(manifest_path, manifest)
    records = []
    signature = None
    for stage in stages:
        pending = [seed for seed in seeds if not (output_dir / f"{stage['name']}_seed{seed}.json").exists()]
        if pending:
            logging.info("Loading %s: %s", stage["name"], stage["checkpoint"])
            policy, groups = load_policy(stage, args.scope, args.device)
            current_signature = {
                group: [[name, list(p.shape)] for name, p in members] for group, members in groups.items()
            }
            if signature is not None and signature != current_signature:
                raise ValueError("Parameter groups differ across checkpoints")
            signature = current_signature
            for seed in pending:
                record = {
                    "stage": stage["name"],
                    "total_step": stage["total_step"],
                    "local_step": stage["local_step"],
                    "seed": seed,
                    "groups": analyze_seed(policy, groups, samples, seed, args),
                    "parameter_groups": current_signature,
                }
                write_json(output_dir / f"{stage['name']}_seed{seed}.json", record)
            del policy, groups
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for seed in seeds:
            record = read_json(output_dir / f"{stage['name']}_seed{seed}.json")
            if any(record[key] != stage[key] for key in ("local_step", "total_step")) or (
                record["stage"] != stage["name"] or record["seed"] != seed
            ):
                raise ValueError(f"Cached record does not match {stage['name']} seed {seed}")
            if signature is not None and record["parameter_groups"] != signature:
                raise ValueError("Parameter groups differ across checkpoints or cached records")
            signature = record["parameter_groups"]
            records.append(record)
    write_json(output_dir / "results.json", {"manifest": manifest, "records": records})
    from lerobot.scripts.plot_pi05_ntk_stages import plot_results

    plot_results(output_dir / "results.json")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", help="Formal-flow output directory; auto-resolves its pretraining sibling"
    )
    parser.add_argument(
        "--through-stage",
        choices=STAGES,
        default="final",
        help="Analyze from before through this stage; stage2 needs only the 0/750/1000 snapshots",
    )
    for stage in STAGES:
        parser.add_argument(
            f"--{stage}", help="Explicit checkpoint directory or its pretrained_model subdirectory"
        )
    parser.add_argument(
        "--final-flow-steps",
        type=int,
        default=2000,
        help="Final LOCAL flow step: 2000 => cumulative 3000; 3000 => cumulative 4000",
    )
    parser.add_argument("--dataset-repo-id")
    parser.add_argument("--dataset-root")
    parser.add_argument("--tokenizer-path", help="Override a saved tokenizer path after moving machines")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scope", choices=("backbone", "prompts", "both"), default="both")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--indices", help="Comma-separated fixed dataset indices; overrides --num-samples")
    parser.add_argument("--seeds", default=",".join(map(str, range(10))))
    parser.add_argument("--output-probes", type=int, default=4)
    parser.add_argument("--probe-seed", type=int, default=1729)
    parser.add_argument("--sketch-dim", type=int, default=8192)
    parser.add_argument("--sketch-seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/ntk_stages")
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify checkpoint files and step metadata only"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Replot output-dir/results.json without models"
    )
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = make_parser().parse_args()
    if args.plot_only:
        from lerobot.scripts.plot_pi05_ntk_stages import plot_results

        plot_results(Path(args.output_dir) / "results.json")
    else:
        run(args)


if __name__ == "__main__":
    main()
