"""Offline two-camera scoring; loads only PI05's frozen vision tower, not its LLM/expert."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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


def read_frame_batch(dataset, indices, cameras, decode_mode="batch"):
    """Reuse LeRobot's timestamps/episode offsets and PyAV decoder for a whole batch."""
    import torch

    reader = getattr(dataset, "reader", None)
    if decode_mode == "batch" and reader is not None and all(k in dataset.meta.video_keys for k in cameras):
        if dataset.delta_timestamps is not None or dataset.image_transforms is not None:
            raise ValueError(
                "Offline batch decoding requires raw single-frame observations without transforms"
            )
        if any(k in dataset.meta.depth_keys for k in cameras):
            raise ValueError("FrameSkip requires two RGB cameras")
        rows = dataset.hf_dataset.select_columns(["index", "episode_index", "timestamp"])[indices.tolist()]
        ids = torch.as_tensor(rows["index"]).reshape(-1)
        episodes = torch.as_tensor(rows["episode_index"]).reshape(-1)
        timestamps = torch.as_tensor(rows["timestamp"]).reshape(-1)
        if (
            ids.tolist() != indices.tolist()
            or len(episodes.unique()) != 1
            or not torch.isfinite(timestamps).all()
        ):
            raise ValueError("Batch decoding changed frame identities or crossed an episode")
        images = reader._query_videos({key: timestamps.tolist() for key in cameras}, int(episodes[0]))
        return {
            "index": ids,
            **{key: images[key].unsqueeze(0) if images[key].ndim == 3 else images[key] for key in cameras},
        }
    rows = [dataset[int(index)] for index in indices]
    return {
        "index": torch.as_tensor([int(row["index"]) for row in rows]),
        **{key: torch.stack([row[key] for row in rows]) for key in cameras},
    }


def verify_batch_decoding(dataset, indices, cameras):
    """Fail closed if the fast reader differs from the existing per-frame pixel path."""
    import torch

    fast = read_frame_batch(dataset, indices, cameras, "batch")
    slow = read_frame_batch(dataset, indices, cameras, "frame")
    for key in ("index", *cameras):
        if not torch.equal(fast[key], slow[key]):
            raise ValueError(f"Batch/frame decoding mismatch for {key}; rerun with --decode-mode frame")


def extract_episode(
    dataset, indices, cameras, model, resolution, device, batch_size, decode_mode="batch", progress=None
):
    import torch
    import torch.nn.functional as F  # noqa: N812

    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

    result = np.zeros((len(indices), 2), dtype=np.float32)
    previous = [None, None]
    # One reader thread overlaps CPU video decoding with GPU inference. No DataLoader fork.
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(read_frame_batch, dataset, indices[:batch_size], cameras, decode_mode)
        for start in range(0, len(indices), batch_size):
            positions = indices[start : start + batch_size]
            batch = future.result()
            stop = start + len(positions)
            if stop < len(indices):
                future = pool.submit(
                    read_frame_batch, dataset, indices[stop : stop + batch_size], cameras, decode_mode
                )
            if batch["index"].tolist() != positions.tolist():
                raise ValueError("Dataset decoder changed global frame identities")
            for camera, key in enumerate(cameras):
                images = batch[key].to(device, dtype=torch.float32)
                if (
                    images.ndim != 4
                    or images.shape[0] != len(positions)
                    or images.shape[1] != 3
                    or images.min() < 0
                    or images.max() > 1
                ):
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
                result[start:stop, camera] = (joined[1:] - joined[:-1]).norm(dim=-1).cpu().numpy()
                previous[camera] = features[-1:]
            if progress is not None:
                progress(len(positions))
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


# c15245d: identical score/encoder recipe before parallel execution and batch decoding.
# Accept only this reviewed predecessor, never arbitrary changed scoring code.
COMPATIBLE_CACHE_CODE = {"96c6487673626e1cd72ac975abd1121e0ff9e36defbe7dc446dec22e58c2b256"}


def scoring_code_hash():
    files = [
        Path(__file__),
        Path(__file__).with_name("frameskip_parallel.py"),
        Path(__file__).parents[1] / "utils/frameskip_curriculum.py",
    ]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


@contextmanager
def output_lock(path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another scoring process owns {path}; stop it before restarting") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def atomic_file(path, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, mode) as handle:
            yield handle
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path) as handle:
        json.dump(value, handle, indent=2)


def ensure_cache_manifest(cache, expected):
    """Keep the cache recipe immutable; actual execution provenance is recorded separately."""
    import fcntl

    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / "manifest.json"
    with (cache / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            saved = json.loads(marker.read_text())
            accepted = COMPATIBLE_CACHE_CODE | {expected["scoring_code_sha256"]}
            if saved.get("scoring_code_sha256") not in accepted or {
                k: v for k, v in saved.items() if k != "scoring_code_sha256"
            } != {k: v for k, v in expected.items() if k != "scoring_code_sha256"}:
                raise ValueError(
                    "Vision cache source/config/data changed; use a new scoring output directory"
                )
            return saved
        if any(cache.glob("episode-*.npz")):
            raise ValueError("Episode caches exist without their manifest; cannot establish provenance")
        write_json(marker, expected)
        return expected


def partition_episodes(episodes, count):
    """Deterministic longest-first balancing, independent of which caches already exist."""
    import heapq

    if count < 1:
        raise ValueError("Shard count must be positive")
    groups = [[] for _ in range(count)]
    loads = [(0, i) for i in range(count)]
    heapq.heapify(loads)
    for left, right in sorted(episodes, key=lambda pair: (-(pair[1] - pair[0]), pair[0])):
        load, shard = heapq.heappop(loads)
        groups[shard].append((left, right))
        heapq.heappush(loads, (load + right - left, shard))
    return [sorted(group) for group in groups]


def load_delta(path, length):
    with np.load(path, allow_pickle=False) as archive:
        delta = archive["visual_delta"]
        if (
            delta.shape != (length, 2)
            or delta.dtype != np.float32
            or not np.isfinite(delta).all()
            or np.any(delta < 0)
            or np.any(delta[0] != 0)
            or arrays_hash({"visual_delta": delta}) != str(archive["sha256"])
        ):
            raise ValueError(f"Corrupted cached visual deltas: {path}")
    return delta


def write_delta(path, delta):
    with atomic_file(path, "wb") as handle:
        np.savez_compressed(handle, visual_delta=delta, sha256=arrays_hash({"visual_delta": delta}))


def scoring_context(args):
    import torch

    # Local scoring must not download a model, tokenizer or dataset implicitly.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

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
    cache = output / "vision_cache"
    marker = cache / "manifest.json"
    model = None
    if marker.exists():
        saved = json.loads(marker.read_text())
        resolution, weight_hash = saved["resolution"], saved["vision_weights_sha256"]
    elif args.merge_only:
        raise FileNotFoundError(f"No vision cache manifest: {marker}")
    else:
        model, resolution, weight_hash = load_vision(args.source, args.device)
    # File inventory identifies local video changes cheaply; this is not a video-content hash.
    videos = file_inventory(root, sorted((root / "videos").rglob("*")))
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
        "scoring_code_sha256": scoring_code_hash(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    provenance = json.loads(json.dumps(provenance))
    cache_identity = {**provenance, "dataset_metadata_sha256": arrays_hash(metadata)}
    provenance = ensure_cache_manifest(cache, cache_identity)
    return metadata, actions, cameras, low, high, threshold, config, provenance, model


def episode_selection(metadata, pilot_only):
    edges = np.r_[0, np.flatnonzero(np.diff(metadata["episode_index"])) + 1, len(metadata["index"])]
    episodes = list(zip(edges[:-1], edges[1:], strict=True))
    pilot = set()
    for task in np.unique(metadata["task_index"]):
        candidates = [
            int(metadata["episode_index"][a]) for a, b in episodes if metadata["task_index"][a] == task
        ]
        rng = np.random.default_rng(np.random.SeedSequence([2026, int(task)]))
        pilot.update(rng.choice(candidates, min(2, len(candidates)), replace=False).tolist())
    selected = [(a, b) for a, b in episodes if not pilot_only or metadata["episode_index"][a] in pilot]
    return selected, pilot


def score_dataset(args):
    from tqdm import tqdm

    metadata, actions, cameras, low, high, threshold, config, provenance, model = scoring_context(args)
    selected, pilot = episode_selection(metadata, args.pilot_only)
    assigned = partition_episodes(selected, args.num_shards)[args.shard_index] if args.worker else selected
    cache, output = args.output / "vision_cache", args.output
    runtime = {
        "execution_code_sha256": scoring_code_hash(),
        "cache_recipe_code_sha256": provenance["scoring_code_sha256"],
        "decode_mode": args.decode_mode,
        "batch_size": args.batch_size,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "merge_only": args.merge_only,
        "pilot_only": args.pilot_only,
        "pid": os.getpid(),
        "started_at_unix": time.time(),
    }
    runtime_path = output / "scoring_logs" / f"execution-{time.time_ns()}-{os.getpid()}.json"
    write_json(runtime_path, runtime)
    dataset = None
    complete = 0
    total = sum(b - a for a, b in assigned)
    last_status = 0.0

    def progress(count, force=False):
        nonlocal complete, last_status
        complete += int(count)
        bar.update(count)
        now = time.monotonic()
        if args.status_path and (force or now - last_status >= 1):
            write_json(args.status_path, {"frames": complete, "total": int(total)})
            last_status = now

    if not args.merge_only:
        with tqdm(
            total=total, desc="FrameSkip frames", unit="frame", dynamic_ncols=True, disable=args.worker
        ) as bar:
            progress(0, force=True)
            for left, right in assigned:
                episode = int(metadata["episode_index"][left])
                path = cache / f"episode-{episode:06d}.npz"
                if path.exists():
                    load_delta(path, right - left)
                    progress(right - left)
                    continue
                if model is None:
                    model, resolution, weight_hash = load_vision(args.source, args.device)
                    if (
                        resolution != provenance["resolution"]
                        or weight_hash != provenance["vision_weights_sha256"]
                    ):
                        raise ValueError("Vision weights differ from the cache source")
                if dataset is None:
                    from lerobot.datasets.lerobot_dataset import LeRobotDataset

                    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, video_backend="pyav")
                    if args.decode_mode == "batch":
                        verify_batch_decoding(
                            dataset, np.arange(left, min(right, left + args.batch_size)), cameras
                        )
                delta = extract_episode(
                    dataset,
                    np.arange(left, right),
                    cameras,
                    model,
                    provenance["resolution"],
                    args.device,
                    args.batch_size,
                    args.decode_mode,
                    progress,
                )
                write_delta(path, delta)
            progress(0, force=True)
    runtime.update(finished_at_unix=time.time(), frames=int(total))
    write_json(runtime_path, runtime)
    if args.worker:
        return
    # Only the coordinator/single process merges. No model or video reader is needed on resume.
    all_scores, reports = {}, []
    for left, right in tqdm(selected, desc="Merge scores", dynamic_ncols=True):
        episode = int(metadata["episode_index"][left])
        task = int(metadata["task_index"][left])
        delta = load_delta(cache / f"episode-{episode:06d}.npz", right - left)
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
    parser.add_argument("--gpu-ids", help="Independent scoring workers, e.g. 0,1,2,3; no DDP")
    parser.add_argument("--decode-mode", choices=("batch", "frame"), default="batch")
    parser.add_argument("--merge-only", action="store_true", help="Validate complete caches and merge on CPU")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--status-path", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid shard index/count")
    if args.worker and (args.gpu_ids or args.merge_only):
        parser.error("Worker cannot coordinate or merge")
    if not args.worker and (args.num_shards != 1 or args.shard_index != 0):
        parser.error("Shard options are internal; use --gpu-ids")
    if args.gpu_ids and args.merge_only:
        parser.error("--merge-only does not use GPUs")
    if args.worker:
        score_dataset(args)
    else:
        with output_lock(args.output / ".scoring.lock"):
            if args.gpu_ids:
                from lerobot.scripts.frameskip_parallel import run_parallel

                run_parallel(args)
            else:
                score_dataset(args)


if __name__ == "__main__":
    main()
