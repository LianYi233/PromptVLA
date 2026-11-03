"""Offline, paired normalised flow-velocity MSE for task7 cadence experiment II.

Uses genuine recorded instantaneous actions at t+j*k frames (k=1,2, and
optionally unseen k=3), fixed Flow Matching t=0.5 and SHA256-keyed noise.
No action execution, simulation, or altered-frequency robot control.
"""

import hashlib
import json
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.scripts.pi05_task7_cadence import (
    HORIZON, file_hash, make_action_targets,
)
from lerobot.scripts.frameskip_scores import read_actions
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION

CASE_BASIS = {
    "rope": (False, "index"), "index": (True, "index"),
    "global_dt": (True, "global_dt"), "physical": (True, "physical"),
}


def flow_noise(seed, start, stride, shape):
    key = json.dumps(["cadence-fixed-flow-v1", seed, int(start), int(stride)])
    generator = torch.Generator(device="cpu").manual_seed(
        int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") & ((1<<63)-1)
    )
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _check_saved_checkpoint(checkpoint, expected_case):
    checkpoint = Path(checkpoint).resolve()
    cfg = json.loads((checkpoint / "config.json").read_text())
    active, basis = CASE_BASIS[expected_case]
    expected = {
        "action_time_encoding": active,
        "action_time_basis": basis,
        "action_time_injection": "input",
        "chunk_size": HORIZON,
        "train_action_expert": True,
        "num_vlm_prompt_tokens": 2,
        "num_prompt_tokens": 0,
    }
    for key, value in expected.items():
        if cfg.get(key, "index" if key == "action_time_basis" else None) != value:
            raise ValueError(f"Offline model {expected_case} has unexpected {key}")
    return {
        "config_sha256": file_hash(checkpoint / "config.json"),
        "weights_size": (checkpoint / "model.safetensors").stat().st_size,
        "weights_mtime_ns": (checkpoint / "model.safetensors").stat().st_mtime_ns,
    }


def evaluate_checkpoint(prepared, checkpoint, output, args, *, expected_case, eval_manifest):
    import os

    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    fingerprint = _check_saved_checkpoint(checkpoint, expected_case)
    if output.exists():
        stored = json.loads(output.read_text())
        if (
            stored.get("case") != expected_case
            or stored.get("checkpoint") != str(checkpoint)
            or stored.get("checkpoint_fingerprint") != fingerprint
            or stored.get("eval_manifest") != eval_manifest
        ):
            raise ValueError("Existing evaluation output is from a different experiment/checkpoint")
        print(f"Reusing verified existing offline evaluation {output}")
        return stored
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not os.environ.get("CADENCE_ALLOW_CPU_EVAL"):
        raise RuntimeError("GPU unavailable; set CADENCE_ALLOW_CPU_EVAL=1 only for local debugging")
    print(f"Loading {expected_case}: {checkpoint} on {device}", flush=True)
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    config = PI05Config.from_pretrained(checkpoint)
    config.device = str(device)
    config.tokenizer_name = str(args.tokenizer.resolve())
    config.compile_model = False
    config.gradient_checkpointing = False
    policy = PI05Policy.from_pretrained(
        checkpoint, config=config, local_files_only=True, strict=True
    ).to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(args.tokenizer.resolve())},
        },
    )
    root = Path(prepared["dataset_root"])
    fps = float(prepared["fps"])
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=root,
        delta_timestamps={ACTION:[j / fps for j in range(HORIZON)]},
        video_backend="pyav",
        return_uint8=True,
    )
    actions = read_actions(root)
    collate = lerobot_collate_fn if dataset.meta.has_language_columns else default_collate
    eval_starts = prepared["heldout_starts"]
    stride3_starts = prepared["stride3_heldout_starts"]
    metric_by_stride = {}
    total_per_stride = {}
    for stride in (1, 2, 3):
        starts = stride3_starts if stride == 3 else eval_starts
        starts = starts[:args.max_eval_starts]
        if not starts:
            if stride == 3:
                print("Skipping unseen stride3: task7 demonstration too short", flush=True)
                continue
            raise ValueError(f"No heldout examples for stride {stride}")
        losses = []
        for start in starts:
            labels = make_action_targets(
                actions, [start], [stride],
                first=prepared["episode_first"], last=prepared["episode_last"],
            )[0]
            item = dict(dataset[int(start)])
            if int(item["index"]) != int(start):
                raise ValueError("Heldout observation index differs from target anchor")
            item[ACTION] = torch.from_numpy(labels.copy())
            item[f"{ACTION}_is_pad"] = torch.zeros(HORIZON, dtype=torch.bool)
            for key in dataset.meta.camera_keys:
                if key in item and item[key].dtype == torch.uint8:
                    item[key] = item[key].float() / 255.0
            batch = preprocessor(collate([item]))
            batch["pi05_action_dt"] = torch.tensor(
                [stride/fps], dtype=torch.float32, device=device
            )
            with torch.inference_mode():
                actions_norm = policy.prepare_action(batch)
                noise = flow_noise(args.eval_seed, start, stride, actions_norm.shape).to(device)
                time = torch.tensor([args.flow_time], device=device, dtype=torch.float32)
                loss, _ = policy.forward(batch, noise=noise, time=time)
            losses.append(float(loss.item()))
        metric_by_stride[str(stride)] = {
            "n": len(losses),
            "mean_mse": mean(losses),
            "median_mse": median(losses),
            "std_mse": pstdev(losses),
            "sample_mse": losses,
            "start_indices": [int(s) for s in starts],
            "delta_t_seconds": stride/fps,
            "train_seen_cadence": stride in (1,2),
        }
        total_per_stride[str(stride)] = len(losses)
        print(
            f"{expected_case} stride={stride} dt={stride/fps:.4f}s "
            f"MSE={mean(losses):.6f} n={len(losses)}",
            flush=True,
        )
    report = {
        "protocol": "pi05-cadence-offline-eval-result-v1",
        "case": expected_case,
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": fingerprint,
        "eval_manifest": eval_manifest,
        "per_stride": metric_by_stride,
        "important": (
            "Instantaneous recorded action targets, not 2x/3x aggregated commands. "
            "All held-out starts come from the SAME training demonstration, "
            "so their overlapping target windows do not establish generalization."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report
