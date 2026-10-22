"""Offline two-camera scoring; loads only PI05's frozen vision tower, not its LLM/expert."""

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lerobot.utils.frameskip_curriculum import ScoreConfig, save_scores, score_episode
from lerobot.utils.pdet_order import arrays_hash, read_frame_metadata


def file_inventory(root, paths):
    return [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()]


def source_inventory(source):
    return file_inventory(source, sorted(source.glob("model*.safetensors")) + [source / "config.json"])


def check_inputs(scores_path, root, source):
    from lerobot.utils.frameskip_curriculum import load_scores

    _, manifest = load_scores(scores_path)
    if str(root.resolve()) != manifest["dataset_root"] or str(source.resolve()) != manifest["source"]:
        raise ValueError("Training must use the dataset/source used for scoring")
    checks = {
        "dataset_metadata_sha256": arrays_hash(read_frame_metadata(root)),
        "actions_sha256": arrays_hash({"action": read_actions(root)}),
        "stats_sha256": hashlib.sha256((root / "meta/stats.json").read_bytes()).hexdigest(),
        "source_inventory": source_inventory(source),
        "video_inventory_sha256": hashlib.sha256(
            json.dumps(file_inventory(root, sorted((root / "videos").rglob("*")))).encode()
        ).hexdigest(),
    }
    # JSON normalization handles tuple/list conversion in persisted inventories.
    for key, value in json.loads(json.dumps(checks)).items():
        if manifest[key] != value:
            raise ValueError(f"Training input changed after scoring: {key}")
    return {
        "inputs": "match_scoring_source",
        "scope": "actions/stats hashes; model/video size+mtime inventory",
    }


def read_actions(root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.concat_tables(
        [
            pq.read_table(p, columns=["index", "action"])
            for p in sorted((Path(root) / "data").rglob("*.parquet"))
        ]
    )
    order = np.argsort(np.asarray(table["index"].to_pylist()))
    return np.asarray(table["action"].to_pylist(), dtype=np.float32)[order]


def gripper_threshold(actions, supplied=None):
    values = actions[:, -1]
    if supplied is not None:
        if not np.isfinite(supplied) or not values.min() < supplied < values.max():
            raise ValueError("Explicit gripper threshold must lie inside observed action range")
        return float(supplied)
    unique = np.unique(values)
    if np.all(np.isin(unique, [0, 1])):
        return 0.5
    if np.all(np.isin(unique, [-1, 1])):
        return 0.0
    raise ValueError("Nonbinary gripper actions: inspect the schema and pass --gripper-threshold explicitly")


def vision_config_for(resolution):
    from transformers import PaliGemmaConfig

    # Matches PaliGemmaWithExpertModel in modeling_pi05.py.
    config = PaliGemmaConfig().vision_config
    config.image_size = resolution[0]
    config.intermediate_size = 4304
    config.projection_dim = 2048
    config.projector_hidden_act = "gelu_fast"
    config._attn_implementation = "eager"
    return config


def load_vision(source, device):
    import re

    import torch
    from safetensors import safe_open
    from transformers import SiglipVisionModel

    source = Path(source)
    config = json.loads((source / "config.json").read_text())
    resolution = config.get("image_resolution", [224, 224])
    if config.get("type") != "pi05" or len(resolution) != 2 or resolution[0] != resolution[1]:
        raise ValueError("Expected a native PI05 source with square image resolution")
    weights = sorted(source.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No model safetensors in {source}")
    prefix = re.compile(r"^(?:model\.)?paligemma_with_expert\.paligemma\.(?:model\.)?vision_tower\.")
    state = {}
    for path in weights:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safetensors is not a dict
                match = prefix.match(key)
                if match:
                    name = key[match.end() :]
                    if name in state:
                        raise ValueError(f"Duplicate vision weight: {name}")
                    state[name] = handle.get_tensor(key).float()
    if not state:
        raise ValueError("No PI05 vision_tower keys found; refusing random scoring weights")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    vision_config = vision_config_for(resolution)
    with torch.random.fork_rng(devices=[]):
        model = SiglipVisionModel(vision_config)
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    return model, resolution, digest.hexdigest()


def extract_episode(dataset, indices, cameras, model, resolution, device, batch_size):
    import torch
    import torch.nn.functional as F  # noqa: N812

    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

    result = np.zeros((len(indices), 2), dtype=np.float32)
    previous = [None, None]
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            positions = indices[start : start + batch_size]
            rows = [dataset[int(index)] for index in positions]
            if [int(row["index"]) for row in rows] != positions.tolist():
                raise ValueError("Dataset decoder changed global frame identities")
            for camera, key in enumerate(cameras):
                images = torch.stack([row[key] for row in rows]).to(device, dtype=torch.float32)
                if images.ndim != 4 or images.shape[1] != 3 or images.min() < 0 or images.max() > 1:
                    raise ValueError(f"Expected [N,3,H,W] images in [0,1]: {key}")
                images = images.permute(0, 2, 3, 1)
                if tuple(images.shape[1:3]) != tuple(resolution):
                    images = resize_with_pad_torch(images, *resolution)
                images = (images * 2 - 1).permute(0, 3, 1, 2)
                features = model(pixel_values=images).last_hidden_state.mean(dim=1)
                features = F.normalize(features.float(), dim=-1)
                if not torch.isfinite(features).all():
                    raise ValueError("Nonfinite frozen visual features")
                before = features[:1] if previous[camera] is None else previous[camera]
                joined = torch.cat([before, features])
                result[start : start + len(rows), camera] = (
                    (joined[1:] - joined[:-1]).norm(dim=-1).cpu().numpy()
                )
                previous[camera] = features[-1:]
    return result


def plot_episode(path, scores, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(scores["action_score"], label="normalized action change")
    axes[0].legend(loc="upper right")
    for key, label in (("vac_0", "camera 0"), ("vac_1", "camera 1")):
        axes[1].plot(scores[key], label=label)
    axes[1].legend(loc="upper right")
    axes[2].plot(scores["score"], label="combined importance")
    axes[2].fill_between(
        np.arange(len(scores["score"])), 0, scores["event_neighborhood"], alpha=0.15, label="gripper window"
    )
    axes[2].legend(loc="upper right")
    for row, key in enumerate(("selected", "random_selected", "protected")):
        ids = np.flatnonzero(scores[key])
        axes[3].scatter(ids, np.full(len(ids), row), s=8, label=key)
    axes[3].set_yticks(range(3), ["scored", "random", "protected"])
    axes[3].set_xlabel("Original frame / action-chunk start")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def score_dataset(args):
    import torch
    from tqdm import tqdm

    # Local scoring must not download a model, tokenizer or dataset implicitly.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = ScoreConfig()
    root, output = args.dataset_root, args.output
    metadata = read_frame_metadata(root)
    actions = read_actions(root)
    if len(actions) != len(metadata["index"]):
        raise ValueError("Action/metadata lengths differ")
    info = json.loads((root / "meta/info.json").read_text())
    cameras = args.cameras or [k for k, v in info["features"].items() if v["dtype"] in ("video", "image")]
    if len(cameras) != 2 or len(set(cameras)) != 2:
        raise ValueError("Specify exactly two distinct --cameras keys")
    for camera in cameras:
        if camera not in info["features"]:
            raise ValueError(f"Unknown camera: {camera}")
    stats = json.loads((root / "meta/stats.json").read_text())["action"]
    low, high = np.asarray(stats["q01"])[:6], np.asarray(stats["q99"])[:6]
    threshold = gripper_threshold(actions, args.gripper_threshold)
    model, resolution, weight_hash = load_vision(args.source, args.device)
    # File inventory identifies local video changes cheaply; this is not a video-content hash.
    videos = file_inventory(root, sorted((root / "videos").rglob("*")))
    source_files = [Path(__file__), Path(__file__).parents[1] / "utils/frameskip_curriculum.py"]
    provenance = {
        "config": asdict(config),
        "gripper_threshold": threshold,
        "action_low": low.tolist(),
        "action_high": high.tolist(),
        "actions_sha256": arrays_hash({"action": actions}),
        "stats_sha256": hashlib.sha256((root / "meta/stats.json").read_bytes()).hexdigest(),
        "source_inventory": source_inventory(args.source),
        "dataset_root": str(root.resolve()),
        "source": str(args.source.resolve()),
        "vision_weights_sha256": weight_hash,
        "resolution": resolution,
        "cameras": cameras,
        "encoder": "PI05 frozen SigLIP last_hidden_state mean pooled, L2 normalized, float32 eager",
        "video_inventory_sha256": hashlib.sha256(json.dumps(videos).encode()).hexdigest(),
        "video_identity_scope": "relative paths/sizes/mtime only, NOT full video content hashes",
        "scoring_code_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in source_files)).hexdigest(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    provenance = json.loads(json.dumps(provenance))
    cache_identity = {**provenance, "dataset_metadata_sha256": arrays_hash(metadata)}
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "vision_cache"
    cache.mkdir(exist_ok=True)
    marker = cache / "manifest.json"
    if marker.exists():
        if json.loads(marker.read_text()) != cache_identity:
            raise ValueError("Vision cache source/config/data changed; use a new scoring output directory")
    else:
        with marker.open("x") as handle:
            json.dump(cache_identity, handle, indent=2)
    edges = np.r_[0, np.flatnonzero(np.diff(metadata["episode_index"])) + 1, len(actions)]
    episodes = list(zip(edges[:-1], edges[1:], strict=True))
    pilot = set()
    for task in np.unique(metadata["task_index"]):
        candidates = [
            int(metadata["episode_index"][a]) for a, b in episodes if metadata["task_index"][a] == task
        ]
        rng = np.random.default_rng(np.random.SeedSequence([2026, int(task)]))
        pilot.update(rng.choice(candidates, min(2, len(candidates)), replace=False).tolist())
    dataset = LeRobotDataset(args.repo_id, root=root, video_backend="pyav")
    all_scores, reports = {}, []
    selected_episodes = [
        (a, b) for a, b in episodes if not args.pilot_only or metadata["episode_index"][a] in pilot
    ]
    for left, right in tqdm(selected_episodes, desc="FrameSkip scoring", dynamic_ncols=True):
        episode = int(metadata["episode_index"][left])
        task = int(metadata["task_index"][left])
        path = cache / f"episode-{episode:06d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                delta = archive["visual_delta"]
                if arrays_hash({"visual_delta": delta}) != str(archive["sha256"]):
                    raise ValueError(f"Corrupted cached visual deltas: {path}")
        else:
            delta = extract_episode(
                dataset, np.arange(left, right), cameras, model, resolution, args.device, args.batch_size
            )
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, visual_delta=delta, sha256=arrays_hash({"visual_delta": delta}))
            temporary.replace(path)
        scored = score_episode(actions[left:right], delta, low, high, threshold, episode, config)
        if not args.pilot_only:
            for key, value in scored.items():
                if key not in all_scores:
                    all_scores[key] = np.empty(len(actions), dtype=value.dtype)
                all_scores[key][left:right] = value
        reports.append(
            {
                "task": task,
                "episode": episode,
                "frames": right - left,
                "selected": int(scored["selected"].sum()),
                "random_selected": int(scored["random_selected"].sum()),
                "protected": int(scored["protected"].sum()),
                "retention": float(scored["selected"].mean()),
                "events": int(scored["event"].sum()),
            }
        )
        if episode in pilot:
            plot_episode(
                output / "preview" / f"task-{task:02d}-episode-{episode:06d}.png",
                scored,
                f"task {task}, episode {episode}; retained {scored['selected'].mean():.1%}",
            )
    report_name = "pilot_episodes.csv" if args.pilot_only else "episodes.csv"
    with (output / report_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    if not args.pilot_only:
        save_scores(output / "scores.npz", metadata, all_scores, provenance)
    print(
        json.dumps(
            {
                "episodes": len(reports),
                "pilot_only": args.pilot_only,
                "output": str(output),
                "next": "inspect preview, then prepare plans",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/wyn/datasets/libero"))
    parser.add_argument("--repo-id", default="libero")
    parser.add_argument("--source", type=Path, default=Path("/home/wyn/models/lerobot/pi05_libero_base"))
    parser.add_argument("--output", type=Path, default=Path("/data1/wyn/prompt-work/frameskip-r1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cameras", nargs=2)
    parser.add_argument("--gripper-threshold", type=float)
    parser.add_argument(
        "--pilot-only", action="store_true", help="2 episodes/task, cache reusable by full scoring"
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    score_dataset(args)


if __name__ == "__main__":
    main()
