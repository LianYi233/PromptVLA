#!/usr/bin/env python
"""Train separate PI05 prompt policies on local May-pick-and-place v3.0 tasks.

Only the standard library is needed for --help and DRY_RUN=true. Training uses
the existing ablation launcher and the active LeRobot environment.
"""

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

TASKS = {
    "1": "1-put_the_apple_on_the_yellow_plate",
    "2": "2-remove_the_cuboid_from_blue_plate",
    "3": "3-move_the_tennis_from_yellow_plate_to_blue_plate",
    "4": "4-pick_the_red_cube_into_the_yellow_plate",
}
VARIANTS = (
    "full_reference",
    "no_bridge",
    "direct_dual",
    "dual_prompt_only",
    "vlm_only",
    "action_only",
    "no_cabo",
)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def cleanup_pretrain_checkpoint(output_dir: Path, flow_steps: int) -> None:
    """Discard this fresh run's transfer weights only after a final flow model exists."""
    stage_dir = output_dir.with_name(f"{output_dir.name}_next_action_pretrain")
    checkpoints = stage_dir / "checkpoints"
    if not checkpoints.exists():
        return
    if stage_dir.is_symlink() or checkpoints.is_symlink():
        raise ValueError(f"Refusing to clean a linked pretraining directory: {checkpoints}")
    step_id = str(flow_steps).zfill(max(6, len(str(flow_steps))))
    final_model = output_dir / "checkpoints" / step_id / "pretrained_model"
    if not all((final_model / name).is_file() for name in ("config.json", "model.safetensors")):
        raise ValueError(f"Keeping pretraining weights: final flow model is incomplete at {final_model}")
    shutil.rmtree(checkpoints)
    print(f"Removed temporary pretraining checkpoints: {checkpoints}", flush=True)
    if not any(stage_dir.iterdir()):
        stage_dir.rmdir()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    with path.open() as stream:
        return json.load(stream)


def check_dataset(root: Path, normalization: str) -> None:
    """Check local metadata before any model allocation or implicit Hub lookup."""
    for folder in ("data", "meta", "videos"):
        if not (root / folder).is_dir():
            raise ValueError(f"Missing {root / folder}; DATASET_BASE must contain the four task folders")
    info = read_json(root / "meta/info.json")
    if str(info.get("codebase_version", "")).lstrip("v") != "3.0":
        raise ValueError(f"{root}: expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    for key in ("fps", "total_episodes", "total_frames"):
        if not isinstance(info.get(key), (int, float)) or info[key] <= 0:
            raise ValueError(f"{root}: {key} must be positive")
    for pattern in ("meta/tasks.parquet", "meta/episodes/*/*.parquet", "data/*/*.parquet"):
        if not any(path.is_file() for path in root.glob(pattern)):
            raise ValueError(f"Missing v3.0 file(s): {root / pattern}")

    features = info.get("features", {})
    stats = read_json(root / "meta/stats.json")
    required_stats = {
        "QUANTILES": ("q01", "q99"),
        "MEAN_STD": ("mean", "std"),
        "MIN_MAX": ("min", "max"),
    }[normalization]
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        shape = feature.get("shape", [])
        if len(shape) != 1 or not isinstance(shape[0], int) or not 0 < shape[0] <= 32:
            raise ValueError(f"{root}: {key} must be a vector with 1..32 dimensions, got {shape}")
        for stat in required_stats:
            values = stats.get(key, {}).get(stat)
            if values is None:
                raise ValueError(
                    f"{root}: missing {key}.{stat} for {normalization}. "
                    "See README_pi05_real.md for quantile-stat preparation or NORMALIZATION_MODE."
                )
            if (
                not isinstance(values, list)
                or len(values) != shape[0]
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)
            ):
                raise ValueError(f"{root}: invalid {key}.{stat}; expected {shape[0]} finite values")

    cameras = {key: value for key, value in features.items() if value.get("dtype") in ("video", "image")}
    if not cameras:
        raise ValueError(f"{root}: no image/video features in meta/info.json")
    for key, feature in cameras.items():
        if not key.startswith("observation.images."):
            raise ValueError(f"{root}: expected an observation.images.* camera key, got {key}")
        if feature["dtype"] == "video":
            template = info.get("video_path")
            if not template:
                raise ValueError(f"{root}: video_path is missing from meta/info.json")
            pattern = template.replace("{video_key}", key)
            pattern = re.sub(r"\{(?:chunk_index|file_index)(?::[^}]+)?\}", "*", pattern)
            if not any(path.is_file() for path in root.glob(pattern)):
                raise ValueError(f"{root}: no local video found for {key}: {pattern}")
    print(
        f"Dataset: {root}\n"
        f"  episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']} "
        f"state_dim={features['observation.state']['shape'][0]} action_dim={features['action']['shape'][0]}\n"
        f"  cameras={list(cameras)} normalization={normalization}",
        flush=True,
    )


def positive_int(env: dict[str, str], name: str, default: int) -> int:
    value = int(env.setdefault(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def check_cuda(env: dict[str, str], num_processes: int) -> None:
    # Import lazily: dry runs work without PyTorch, models in RAM, or a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = env["GPU_IDS"]
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < num_processes:
        raise ValueError(f"CUDA unavailable or fewer than {num_processes} visible GPUs: {env['GPU_IDS']}")
    for index in range(num_processes):
        value = torch.ones(1, device=f"cuda:{index}")
        (value + 1).sum().item()
        print(f"CUDA {index}: {torch.cuda.get_device_name(index)} OK", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train one task or four independent policies on May-pick-and-place (LeRobot v3.0).",
        epilog="Set DATASET_BASE, PRETRAINED_PATH and TOKENIZER_PATH. DRY_RUN=true checks metadata and "
        "prints commands. See examples/training/README_pi05_real.md.",
    )
    parser.add_argument("task", choices=[*TASKS, "all", *TASKS.values()])
    parser.add_argument("variant", nargs="?", choices=VARIANTS, default="full_reference")
    parser.add_argument("seed", nargs="?", type=int, default=0)
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    # Keep the validated data/model paths and feature contract identical to the training command.
    managed = {
        "--dataset.root",
        "--dataset.repo_id",
        "--dataset.streaming",
        "--policy.type",
        "--policy.path",
        "--policy.pretrained_path",
        "--policy.tokenizer_name",
        "--policy.input_features",
        "--policy.output_features",
        "--policy.normalization_mapping",
        "--policy.chunk_size",
        "--policy.n_action_steps",
        "--policy.next_action_masked_steps",
        "--policy.max_state_dim",
        "--policy.max_action_dim",
        "--rename_map",
        "--output_dir",
        "--job_name",
        "--config_path",
        "--resume",
        "--steps",
        "--save_steps",
        "--save_checkpoint",
        "--policy.ntk_save_stage_snapshots",
    }
    if any(arg.split("=", 1)[0] in managed for arg in extra):
        parser.error(
            "Use the documented environment variables for paths, normalization, chunks and checkpoint steps; "
            "feature, output and resume overrides are not supported by this fresh-run launcher"
        )
    env = os.environ.copy()
    work_root = Path(env.get("WORK_ROOT", "/root/autodl-tmp")).expanduser().resolve()
    defaults = {
        "DATASET_BASE": str(work_root / "datasets/May-pick-and-place"),
        "PRETRAINED_PATH": str(work_root / "models/pi05_libero_base"),
        "TOKENIZER_PATH": str(work_root / "models/google/paligemma-3b-pt-224"),
        "OUTPUT_ROOT": str(work_root / "chkpt/2601-lerobot/prompt-ablation-real"),
        "LOG_ROOT": str(work_root / "logs/prompt-ablation-real"),
        "TMPDIR": str(work_root / "tmp"),
        "HF_HOME": str(work_root / "cache/huggingface"),
        "TORCHINDUCTOR_CACHE_DIR": str(work_root / "cache/torchinductor"),
        "TRITON_CACHE_DIR": str(work_root / "cache/triton"),
    }
    for name, default in defaults.items():
        env[name] = str(Path(env.get(name, default)).expanduser().resolve())
    env.setdefault("GPU_IDS", "0")
    env.setdefault("BATCH_SIZE", "8")
    env.setdefault("SAVE_FREQ", "500")
    env.setdefault("DRY_RUN", "false")
    env.setdefault("KEEP_PRETRAIN_CHECKPOINT", "true")
    if env["DRY_RUN"] not in ("true", "false"):
        raise ValueError("DRY_RUN must be true or false")
    if env["KEEP_PRETRAIN_CHECKPOINT"] not in ("true", "false"):
        raise ValueError("KEEP_PRETRAIN_CHECKPOINT must be true or false")
    if env["KEEP_PRETRAIN_CHECKPOINT"] == "false" and env.get("NTK_SAVE_STAGE_SNAPSHOTS") != "false":
        raise ValueError("Set NTK_SAVE_STAGE_SNAPSHOTS=false before discarding pretraining checkpoints")
    # Ensure training imports this branch even if another checkout is installed.
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    num_processes = positive_int(env, "NUM_PROCESSES", 1)
    positive_int(env, "BATCH_SIZE", 8)
    positive_int(env, "SAVE_FREQ", 500)
    flow_steps = positive_int(env, "FLOW_STEPS", 3000)
    if "SAVE_STEPS" in env:
        save_steps = json.loads(env["SAVE_STEPS"])
        if not isinstance(save_steps, list) or any(
            type(step) is not int or not 1 <= step <= flow_steps for step in save_steps
        ):
            raise ValueError("SAVE_STEPS must be a JSON list of integer steps between 1 and FLOW_STEPS")
        env["SAVE_STEPS"] = json.dumps(sorted(set(save_steps)), separators=(",", ":"))
    gpu_ids = [item.strip() for item in env["GPU_IDS"].split(",")]
    if any(not item or item == "-1" for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU_IDS must contain distinct GPU IDs, e.g. 0 or 0,1")
    if len(gpu_ids) != num_processes:
        raise ValueError("NUM_PROCESSES must equal the number of GPUs listed in GPU_IDS")
    chunk = positive_int(env, "CHUNK_SIZE", 50)
    n_action_steps = positive_int(env, "N_ACTION_STEPS", chunk)
    masked = positive_int(env, "MASKED_STEPS", max(1, chunk * 4 // 5))
    if n_action_steps > chunk or masked > chunk:
        raise ValueError("N_ACTION_STEPS and MASKED_STEPS cannot exceed CHUNK_SIZE")
    normalization = env.get("NORMALIZATION_MODE", "QUANTILES")
    if normalization not in ("QUANTILES", "MEAN_STD", "MIN_MAX"):
        raise ValueError("NORMALIZATION_MODE must be QUANTILES, MEAN_STD or MIN_MAX")
    for filename in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        if not (Path(env["PRETRAINED_PATH"]) / filename).is_file():
            raise ValueError(f"Missing pretrained model file: {Path(env['PRETRAINED_PATH']) / filename}")
    tokenizer = Path(env["TOKENIZER_PATH"])
    if not (tokenizer / "tokenizer_config.json").is_file() or not any(
        (tokenizer / name).is_file() for name in ("tokenizer.json", "tokenizer.model")
    ):
        raise ValueError(f"Incomplete local PaliGemma tokenizer: {tokenizer}")

    folders = list(TASKS.values()) if args.task == "all" else [TASKS.get(args.task, args.task)]
    commands = []
    for folder in folders:
        root = Path(env["DATASET_BASE"]) / folder
        check_dataset(root, normalization)
        run_name = f"pi05-may-{folder}-{args.variant}-seed{args.seed}"
        task_env = dict(
            env, DATASET_ROOT=str(root), DATASET_REPO_ID=f"may-pick-and-place/{folder}", RUN_NAME=run_name
        )
        # The integrated priming stage is a sibling output directory in this branch.
        for suffix in ("", "_next_action_pretrain"):
            output = Path(env["OUTPUT_ROOT"]) / f"{run_name}{suffix}"
            if output.exists():
                raise ValueError(f"Output already exists: {output}. Choose another OUTPUT_ROOT or seed.")
        command = [
            "bash",
            str(SCRIPT_DIR / "train_pi05_prompt_ablation.sh"),
            args.variant,
            str(args.seed),
            f"--policy.chunk_size={chunk}",
            f"--policy.n_action_steps={n_action_steps}",
            "--policy.normalization_mapping="
            + json.dumps({"VISUAL": "IDENTITY", "STATE": normalization, "ACTION": normalization}),
            "--dataset.use_imagenet_stats=false",
            "--env_eval_freq=0",
            *extra,
        ]
        commands.append((command, task_env))
    # All task folders/output paths have been checked before launching the first long run.
    if env["DRY_RUN"] != "true":
        check_cuda(env, num_processes)
    for command, task_env in commands:
        print(f"Launching: {shlex.join(command)}", flush=True)
        result = subprocess.run(command, env=task_env, cwd=REPO_ROOT, check=False)
        if result.returncode:
            return result.returncode
        if env["DRY_RUN"] != "true":
            # Keep each successful policy paired with the metadata required by --dataset_info.
            # Saved model processors already contain the task's normalization statistics.
            info_path = Path(task_env["OUTPUT_ROOT"]) / task_env["RUN_NAME"] / "dataset_info.json"
            shutil.copy2(Path(task_env["DATASET_ROOT"]) / "meta/info.json", info_path)
            print(f"Deployment metadata: {info_path}", flush=True)
            if env["KEEP_PRETRAIN_CHECKPOINT"] == "false":
                cleanup_pretrain_checkpoint(info_path.parent, flow_steps)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Preflight/training failed: {exc}", file=sys.stderr)
        sys.exit(1)
