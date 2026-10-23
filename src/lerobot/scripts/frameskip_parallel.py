"""Supervise independent GPU scoring workers; merge only after every shard succeeds."""

import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from collections import deque


def worker_command(args, shard, count, status):
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.frameskip_scores",
        "--worker",
        "--num-shards",
        str(count),
        "--shard-index",
        str(shard),
        "--status-path",
        str(status),
        "--device",
        "cuda",
    ]
    for flag, value in (
        ("dataset-root", args.dataset_root),
        ("repo-id", args.repo_id),
        ("source", args.source),
        ("output", args.output),
        ("batch-size", args.batch_size),
        ("decode-mode", args.decode_mode),
    ):
        command.extend([f"--{flag}", str(value)])
    if args.cameras:
        command.extend(["--cameras", *args.cameras])
    if args.gripper_threshold is not None:
        command.extend(["--gripper-threshold", str(args.gripper_threshold)])
    if args.pilot_only:
        command.append("--pilot-only")
    return command


def stop_workers(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def supervise(jobs, total, *, quiet=False):
    """Jobs contain argv, env, log and status paths; useful without a GPU in process tests."""
    from tqdm import tqdm

    processes = []
    try:
        for job in jobs:
            with job["log"].open("w") as log:
                processes.append(
                    subprocess.Popen(job["command"], env=job["env"], stdout=log, stderr=subprocess.STDOUT)
                )
        with tqdm(
            total=total, desc="FrameSkip (all GPUs)", unit="frame", dynamic_ncols=True, disable=quiet
        ) as bar:
            while True:
                codes = [process.poll() for process in processes]
                for code, job in zip(codes, jobs, strict=True):
                    if code is not None and code != 0:
                        with job["log"].open() as handle:
                            tail = "".join(deque(handle, maxlen=25))
                        raise RuntimeError(f"Scoring worker exited {code}. Log: {job['log']}\n{tail}")
                done = 0
                for job in jobs:
                    if job["status"].exists():
                        done += json.loads(job["status"].read_text())["frames"]
                bar.update(done - bar.n)
                if all(code == 0 for code in codes):
                    if done != total:
                        raise RuntimeError(f"Incomplete scoring shards: {done}/{total} frames")
                    return
                time.sleep(0.5)
    finally:
        stop_workers(processes)


def run_parallel(args):
    from lerobot.scripts.frameskip_scores import episode_selection, score_dataset, write_json
    from lerobot.utils.pdet_order import read_frame_metadata

    ids = [value.strip() for value in args.gpu_ids.split(",")]
    if not ids or any(not value.isdecimal() for value in ids) or len(set(map(int, ids))) != len(ids):
        raise ValueError("--gpu-ids must contain unique device indices, e.g. 0,1,2,3")
    if args.device != "cuda":
        raise ValueError("--gpu-ids uses one CUDA device per worker; omit --device")
    selected, _ = episode_selection(read_frame_metadata(args.dataset_root), args.pilot_only)
    total = int(sum(b - a for a, b in selected))
    logs = args.output / "scoring_logs" / f"parallel-{time.time_ns()}-{os.getpid()}"
    logs.mkdir(parents=True)
    jobs = []
    for shard, gpu in enumerate(ids):
        status = logs / f"worker-{shard}.json"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(name, "1")
        jobs.append(
            {
                "command": worker_command(args, shard, len(ids), status),
                "env": env,
                "log": logs / f"worker-{shard}.log",
                "status": status,
            }
        )
    write_json(
        logs / "launch.json",
        {
            "gpus": ids,
            "frames": total,
            "commands": [job["command"] for job in jobs],
            "assignment": "whole-episode longest-first",
        },
    )
    print(
        f"Scoring on GPUs {','.join(ids)}; {len(selected)} episodes, {total} frames.\nLogs: {logs}",
        flush=True,
    )
    supervise(jobs, total)
    print("All shards complete. Validating caches and merging scores on CPU.", flush=True)
    merged = Namespace(**{**vars(args), "merge_only": True, "gpu_ids": None, "device": "cpu"})
    score_dataset(merged)
