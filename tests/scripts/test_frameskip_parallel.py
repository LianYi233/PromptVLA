"""Scoring shards preserve episodes, decoder pixels and resumable score identities."""

import json
import os
import subprocess
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.scripts import frameskip_parallel as parallel, frameskip_scores as scoring
from lerobot.utils.frameskip_curriculum import load_scores


def test_partition_covers_episodes_without_splitting_and_balances():
    edges = np.r_[0, np.cumsum([101, 100, 70, 55, 40, 35, 20, 10, 5])]
    episodes = list(zip(edges[:-1], edges[1:], strict=True))
    groups = scoring.partition_episodes(episodes, 4)
    assert sorted(pair for group in groups for pair in group) == episodes
    assert groups == scoring.partition_episodes(list(reversed(episodes)), 4)
    loads = [sum(b - a for a, b in group) for group in groups]
    assert max(loads) - min(loads) < 35
    assert scoring.partition_episodes([(0, 10)], 4) == [[(0, 10)], [], [], []]
    with pytest.raises(ValueError):
        scoring.partition_episodes(episodes, 0)


def test_cache_compatibility_concurrent_creation_atomicity_and_validation(tmp_path):
    expected = {"source": "base", "data": "same", "scoring_code_sha256": "new"}
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(4) as pool:
        assert all(
            result == expected
            for result in pool.map(lambda _: scoring.ensure_cache_manifest(cache, expected), range(4))
        )
    marker = cache / "manifest.json"
    old = {**expected, "scoring_code_sha256": next(iter(scoring.COMPATIBLE_CACHE_CODE))}
    marker.write_text(json.dumps(old))
    assert scoring.ensure_cache_manifest(cache, expected) == old
    assert json.loads(marker.read_text()) == old  # immutable recipe provenance
    with pytest.raises(ValueError, match="changed"):
        scoring.ensure_cache_manifest(cache, {**expected, "data": "different"})
    marker.write_text(json.dumps({**old, "scoring_code_sha256": "unreviewed"}))
    with pytest.raises(ValueError, match="changed"):
        scoring.ensure_cache_manifest(cache, expected)
    delta = np.array([[0, 0], [0.1, 0.2]], np.float32)
    path = cache / "episode-000000.npz"
    scoring.write_delta(path, delta)
    np.testing.assert_array_equal(scoring.load_delta(path, 2), delta)
    original = path.read_bytes()
    with pytest.raises(RuntimeError), scoring.atomic_file(path, "wb") as handle:
        handle.write(b"interrupted")
        raise RuntimeError("interruption")
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="Corrupted"):
        scoring.load_delta(path, 3)
    with path.open("wb") as handle:
        np.savez(handle, visual_delta=delta, sha256="wrong")
    with pytest.raises(ValueError, match="Corrupted"):
        scoring.load_delta(path, 2)
    with (
        scoring.output_lock(tmp_path / "lock"),
        pytest.raises(RuntimeError, match="Another"),
        scoring.output_lock(tmp_path / "lock"),
    ):
        pass


def test_batch_decoding_matches_original_reader_with_shared_video_offsets(tmp_path):
    import av
    import pandas as pd
    import torch
    from datasets import Dataset

    from lerobot.datasets.dataset_reader import DatasetReader
    from lerobot.datasets.io_utils import hf_transform_to_torch

    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.options = {"g": "30", "crf": "18"}
        rng = np.random.default_rng(4)
        for _ in range(45):
            frame = av.VideoFrame.from_ndarray(
                rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    cameras = ["external", "wrist"]
    meta = SimpleNamespace(
        video_keys=cameras,
        depth_keys=[],
        image_keys=[],
        episodes=[{f"videos/{key}/from_timestamp": offset for key in cameras} for offset in (0, 1.0)],
        get_video_file_path=lambda ep, key: Path("video.mp4"),
        tasks=pd.DataFrame(index=["task"]),
    )
    reader = DatasetReader(meta, tmp_path, None, 0.01, "pyav", None, None)
    reader.hf_dataset = Dataset.from_dict(
        {
            "index": list(range(24)),
            "episode_index": [0] * 12 + [1] * 12,
            "task_index": [0] * 24,
            "timestamp": list(np.arange(12) / 30) * 2,
        }
    )
    reader.hf_dataset.set_transform(hf_transform_to_torch)

    class Wrapped:
        def __init__(self):
            self.reader, self.meta, self.hf_dataset = reader, meta, reader.hf_dataset
            self.delta_timestamps = self.image_transforms = None

        def __getitem__(self, index):
            return self.reader.get_item(index)

    dataset = Wrapped()
    for indices in (np.arange(8), np.arange(12, 20), np.array([23])):
        scoring.verify_batch_decoding(dataset, indices, cameras)
        result = scoring.read_frame_batch(dataset, indices, cameras)
        assert result["external"].shape == (len(indices), 3, 48, 64)
        assert torch.equal(result["external"], result["wrist"])
    with pytest.raises(ValueError, match="crossed an episode"):
        scoring.read_frame_batch(dataset, np.array([11, 12]), cameras)
    dataset.image_transforms = lambda x: x
    with pytest.raises(ValueError, match="without transforms"):
        scoring.read_frame_batch(dataset, np.arange(3), cameras)


def fixture_args(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, source = tmp_path / "data", tmp_path / "model"
    (root / "data").mkdir(parents=True)
    (root / "meta").mkdir()
    source.mkdir()
    (source / "config.json").write_text('{"type": "pi05"}')
    n = 32
    actions = np.random.default_rng(2).normal(size=(n, 7)).astype(np.float32)
    actions[:, -1] = np.tile([0] * 4 + [1] * 4, 4)
    pq.write_table(
        pa.table(
            {
                "index": list(range(n)),
                "task_index": [0] * 16 + [1] * 16,
                "episode_index": np.repeat(np.arange(4), 8),
                "frame_index": np.tile(np.arange(8), 4),
                "action": actions.tolist(),
            }
        ),
        root / "data" / "part.parquet",
    )
    (root / "meta/info.json").write_text(
        json.dumps({"features": {"external": {"dtype": "video"}, "wrist": {"dtype": "video"}}})
    )
    (root / "meta/stats.json").write_text(json.dumps({"action": {"q01": [-1] * 7, "q99": [1] * 7}}))
    return Namespace(
        dataset_root=root,
        repo_id="libero",
        source=source,
        output=tmp_path / "out",
        device="cpu",
        batch_size=3,
        cameras=None,
        gripper_threshold=None,
        pilot_only=False,
        gpu_ids=None,
        decode_mode="batch",
        merge_only=False,
        worker=False,
        num_shards=1,
        shard_index=0,
        status_path=None,
    )


def test_four_shards_resume_and_cpu_merge_equal_single_scoring(tmp_path, monkeypatch):
    import lerobot.datasets.lerobot_dataset as dataset_module

    args = fixture_args(tmp_path)
    monkeypatch.setattr(scoring, "load_vision", lambda *a: (object(), [224, 224], "weights"))
    monkeypatch.setattr(dataset_module, "LeRobotDataset", lambda *a, **kw: object())
    monkeypatch.setattr(scoring, "verify_batch_decoding", lambda *a: None)
    monkeypatch.setattr(scoring, "plot_episode", lambda *a: None)
    calls = []

    def extract(dataset, indices, cameras, model, resolution, device, batch_size, mode, progress):
        calls.append(int(indices[0]))
        delta = np.random.default_rng(int(indices[0])).random((len(indices), 2)).astype(np.float32)
        delta[0] = 0
        progress(len(indices))
        return delta

    monkeypatch.setattr(scoring, "extract_episode", extract)
    scoring.score_dataset(args)
    single, manifest = load_scores(args.output / "scores.npz")
    parallel_args = Namespace(
        **{**vars(args), "output": tmp_path / "sharded", "num_shards": 4, "worker": True}
    )
    for shard in range(4):
        parallel_args.shard_index = shard
        parallel_args.status_path = tmp_path / f"status-{shard}.json"
        scoring.score_dataset(parallel_args)
        assert json.loads(parallel_args.status_path.read_text())["frames"] == 8
    assert sorted(calls) == [0, 0, 8, 8, 16, 16, 24, 24]
    assert not (parallel_args.output / "scores.npz").exists()

    def forbidden(*a, **kw):
        pytest.fail("Resume/merge should not construct the model or dataset")

    monkeypatch.setattr(scoring, "load_vision", forbidden)
    monkeypatch.setattr(dataset_module, "LeRobotDataset", forbidden)
    scoring.score_dataset(parallel_args)  # cached shard, no extraction
    merged = Namespace(
        **{
            **vars(parallel_args),
            "worker": False,
            "num_shards": 1,
            "shard_index": 0,
            "merge_only": True,
            "status_path": None,
        }
    )
    scoring.score_dataset(merged)
    other, other_manifest = load_scores(merged.output / "scores.npz")
    assert manifest == other_manifest
    for key in single:
        np.testing.assert_array_equal(single[key], other[key])
    scoring.score_dataset(merged)  # idempotent output
    (merged.output / "scores.npz").unlink()
    (merged.output / "vision_cache/episode-000003.npz").unlink()
    with pytest.raises(FileNotFoundError):
        scoring.score_dataset(merged)
    assert not (merged.output / "scores.npz").exists()
    # The legacy cache keeps its recipe hash but logs actual execution code separately.
    cache_marker = merged.output / "vision_cache/manifest.json"
    legacy = json.loads(cache_marker.read_text())
    legacy["scoring_code_sha256"] = next(iter(scoring.COMPATIBLE_CACHE_CODE))
    cache_marker.write_text(json.dumps(legacy))
    parallel_args.shard_index = 0
    scoring.score_dataset(parallel_args)
    records = [json.loads(p.read_text()) for p in (merged.output / "scoring_logs").glob("execution-*.json")]
    assert any(
        record["cache_recipe_code_sha256"] == legacy["scoring_code_sha256"]
        and record["execution_code_sha256"] == scoring.scoring_code_hash()
        for record in records
    )


def test_supervisor_real_processes_success_failure_and_cleanup(tmp_path, monkeypatch):
    def job(index, script):
        status = tmp_path / f"worker-{index}.json"
        return {
            "command": [sys.executable, "-c", script, str(status)],
            "env": dict(os.environ),
            "log": tmp_path / f"worker-{index}.log",
            "status": status,
        }

    script = (
        "import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({'frames':3}))"
    )
    parallel.supervise([job(i, script) for i in range(4)], 12, quiet=True)
    processes = []
    popen = subprocess.Popen

    def tracked(*a, **kw):
        process = popen(*a, **kw)
        processes.append(process)
        return process

    monkeypatch.setattr(parallel.subprocess, "Popen", tracked)
    jobs = [job(5, "import time; time.sleep(30)"), job(6, "raise RuntimeError('worker broken')")]
    with pytest.raises(RuntimeError, match="worker broken"):
        parallel.supervise(jobs, 6, quiet=True)
    assert all(process.poll() is not None for process in processes)
    with pytest.raises(RuntimeError, match="Incomplete"):
        parallel.supervise([job(7, "pass")], 3, quiet=True)


def test_launcher_arguments_preserve_recipe_and_merge_only_on_success(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    args.gpu_ids, args.device = "0,1,2,3", "cuda"
    args.cameras, args.gripper_threshold = ["external", "wrist"], 0.0
    merged = []

    def supervise(jobs, total):
        assert len(jobs) == 4 and total == 32
        for i, job in enumerate(jobs):
            assert job["env"]["CUDA_VISIBLE_DEVICES"] == str(i)
            command = job["command"]
            assert command[command.index("--num-shards") + 1] == "4"
            assert command[command.index("--shard-index") + 1] == str(i)
            assert command[command.index("--batch-size") + 1] == "3"
            assert command[command.index("--gripper-threshold") + 1] == "0.0"
            assert "--gpu-ids" not in command

    monkeypatch.setattr(parallel, "supervise", supervise)
    monkeypatch.setattr(scoring, "score_dataset", lambda cfg: merged.append(cfg))
    parallel.run_parallel(args)
    assert len(merged) == 1 and merged[0].merge_only and merged[0].device == "cpu"

    def failed(*a):
        raise RuntimeError("child failure")

    monkeypatch.setattr(parallel, "supervise", failed)
    with pytest.raises(RuntimeError, match="child failure"):
        parallel.run_parallel(args)
    assert len(merged) == 1
    args.gpu_ids = "0,0"
    with pytest.raises(ValueError, match="unique"):
        parallel.run_parallel(args)
