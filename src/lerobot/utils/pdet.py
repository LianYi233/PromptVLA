"""Step/rank-keyed flow draws, independent of P, data order, and diagnostic RNG use.

This is a new stochastic protocol, not a replay of legacy global-seed trajectories.
E/T seeds address batch slots, not sample IDs: a D intervention changes which
observations receive the same flow draws. Keep batch size and world size fixed.
"""

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from lerobot.configs.pdet import PDETConfig
from lerobot.utils.prompt_initialization import prompt_hash

PROTOCOL = "pdet-step-rank-v1"


def stream_seed(factor: str, seed: int, step: int, rank: int) -> int:
    payload = f"{PROTOCOL}:{factor}:{seed}:{step}:{rank}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def sample_flow(policy, batch, config: PDETConfig, step: int, rank: int) -> dict:
    actions = policy.prepare_action(batch)
    generator = torch.Generator(device=actions.device).manual_seed(
        stream_seed("E", config.noise_seed, step, rank)
    )
    noise = torch.normal(
        0.0, 1.0, size=actions.shape, dtype=torch.float32, device=actions.device, generator=generator
    )
    # PI05 samples Beta on CPU, then transfers to the action device. Beta.sample has
    # no generator argument. Restore CPU RNG afterwards; do NOT torch.manual_seed,
    # which would also reseed CUDA. This context touches no CUDA default generator.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(stream_seed("T", config.time_seed, step, rank))
        time = policy.model.sample_time(actions.shape[0], actions.device)
    return {"noise": noise, "time": time}


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def audit_record(indices, draws: dict, *, step: int, rank: int) -> dict:
    if indices is None:
        raise ValueError("PDET audit requires dataset frame 'index' before preprocessing")
    ids = torch.as_tensor(indices).detach().cpu().reshape(-1).tolist()
    if len(ids) != draws["noise"].shape[0]:
        raise ValueError("PDET sample IDs do not match the per-rank batch size")
    return {
        "step": step,
        "rank": rank,
        "sample_ids": ids,
        "sample_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "noise_sha256": tensor_hash(draws["noise"]),
        "time_sha256": tensor_hash(draws["time"]),
        "noise_shape": list(draws["noise"].shape),
        "time_shape": list(draws["time"].shape),
    }


def runtime_info() -> dict:
    versions = {}
    for name in ("torch", "numpy", "transformers", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"versions": versions, "git_commit": commit, "cuda": torch.version.cuda}


class PDETStreams:
    def __init__(self, cfg, policy, accelerator):
        self.config = cfg.pdet
        self.rank = accelerator.process_index
        self.directory = Path(cfg.output_dir) / "pdet"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.directory / f"inputs-rank{self.rank}.jsonl"
        path = self.directory / f"manifest-rank{self.rank}.json"
        invariants = {
            "protocol": PROTOCOL,
            "factors": asdict(cfg.pdet),
            "global_seed": cfg.seed,
            "world_size": accelerator.num_processes,
            "per_device_batch_size": cfg.batch_size,
            "tokens": policy.config.num_vlm_prompt_tokens,
            "dtype": str(policy.model.vlm_prompt_tokens.weight.dtype),
        }
        if cfg.resume:
            previous = json.loads(path.read_text())
            if any(previous.get(key) != value for key, value in invariants.items()):
                raise ValueError(
                    "PDET resume must preserve factors, global seed, world size, batch size and dtype"
                )
        else:
            recipe = cfg.to_dict()
            for key in ("pdet", "output_dir", "job_name", "vlm_prompt_init_path", "wandb"):
                recipe.pop(key, None)
            record = {
                **invariants,
                **runtime_info(),
                "prompt_sha256": prompt_hash(
                    policy.model.vlm_prompt_tokens.weight.detach().float().cpu().numpy()
                ),
                "source_pretrained_path": str(policy.config.pretrained_path),
                "dataset_root": str(cfg.dataset.root),
                "total_steps": cfg.steps,
                "recipe": recipe,
                "device": str(accelerator.device),
                "gpu": torch.cuda.get_device_name(accelerator.device)
                if accelerator.device.type == "cuda"
                else None,
            }
            with path.open("x") as handle:
                json.dump(record, handle, indent=2)
                handle.write("\n")

    def draw(self, policy, batch, step: int, indices) -> dict:
        draws = sample_flow(policy, batch, self.config, step, self.rank)
        if step < self.config.audit_steps:
            record = audit_record(indices, draws, step=step, rank=self.rank)
            with self.audit_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return draws
