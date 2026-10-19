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


def sample_flow(policy, batch, config: PDETConfig, step: int, rank: int, occurrences=None) -> dict:
    actions = policy.prepare_action(batch)
    if config.order_plan is not None:
        from lerobot.utils.pdet_order import FLOW_PROTOCOL

        if occurrences is None or len(occurrences) != len(actions):
            raise ValueError("Occurrence-keyed flow requires one immutable occurrence ID per sample")
        noise_rows, time_rows = [], []
        # CPU draws make keys independent of GPU/rank/batch position. All generator state is local.
        for occurrence in torch.as_tensor(occurrences).cpu().tolist():

            def seed_for(factor, seed, occurrence=occurrence):
                payload = f"{FLOW_PROTOCOL}:{factor}:{seed}:{int(occurrence)}".encode()
                return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)

            generator = torch.Generator().manual_seed(seed_for("E", config.noise_seed))
            noise_rows.append(torch.randn(actions.shape[1:], generator=generator, dtype=torch.float32))
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed_for("T", config.time_seed))
                time_rows.append(policy.model.sample_time(1, torch.device("cpu")))
        return {
            "noise": torch.stack(noise_rows).to(actions.device),
            "time": torch.cat(time_rows).to(actions.device),
        }
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
    root = Path(__file__).resolve().parents[3]
    source_digest = hashlib.sha256()
    # Include on-disk edits, not just HEAD. No credentials/environment variables are recorded.
    paths = sorted((root / "src/lerobot").rglob("*.py")) + sorted((root / "examples/training").rglob("*.sh"))
    for path in paths:
        source_digest.update(str(path.relative_to(root)).encode() + b"\0")
        source_digest.update(hashlib.sha256(path.read_bytes()).digest())
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
    return {
        "versions": versions,
        "git_commit": commit,
        "cuda": torch.version.cuda,
        "source_tree_sha256": source_digest.hexdigest(),
    }


class PDETStreams:
    def __init__(self, cfg, policy, accelerator):
        self.config = cfg.pdet
        self.rank = accelerator.process_index
        self.directory = Path(cfg.output_dir) / "pdet"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.directory / f"inputs-rank{self.rank}.jsonl"
        path = self.directory / f"manifest-rank{self.rank}.json"
        self.frames = None
        self.plan = None
        if cfg.pdet.diagnostics or cfg.pdet.order_plan is not None:
            from lerobot.utils.pdet_order import read_frame_metadata

            self.frames = read_frame_metadata(cfg.dataset.root)
        protocol = PROTOCOL
        if cfg.pdet.order_plan is not None:
            from lerobot.utils.pdet_order import FLOW_PROTOCOL, load_plan, validate_plan_dataset

            self.plan = load_plan(cfg.pdet.order_plan)
            validate_plan_dataset(
                self.plan,
                self.frames,
                steps=cfg.steps,
                batch_size=cfg.batch_size,
                world_size=accelerator.num_processes,
            )
            protocol = FLOW_PROTOCOL
        invariants = {
            "protocol": protocol,
            "factors": asdict(cfg.pdet),
            "global_seed": cfg.seed,
            "world_size": accelerator.num_processes,
            "per_device_batch_size": cfg.batch_size,
            "tokens": policy.config.num_vlm_prompt_tokens,
            "dtype": str(policy.model.vlm_prompt_tokens.weight.dtype),
        }
        if self.plan is not None:
            invariants["order_plan"] = self.plan["manifest"]
        if self.frames is not None:
            from lerobot.utils.pdet_order import arrays_hash

            invariants["dataset_metadata_sha256"] = arrays_hash(self.frames)
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

    def draw(self, policy, batch, step: int, indices, occurrences=None) -> dict:
        draws = sample_flow(policy, batch, self.config, step, self.rank, occurrences)
        if step < self.config.audit_steps:
            record = audit_record(indices, draws, step=step, rank=self.rank)
            if self.frames is not None:
                for key in ("task_index", "episode_index", "frame_index", "phase_bin", "gripper_change"):
                    record[key] = self.frames[key][record["sample_ids"]].tolist()
            if self.plan is not None:
                occurrence_ids = torch.as_tensor(occurrences).cpu().reshape(-1).tolist()
                start = (step * self.plan["manifest"]["world_size"] + self.rank) * len(occurrence_ids)
                expected = self.plan["order"][start : start + len(occurrence_ids)].tolist()
                if occurrence_ids != expected:
                    raise ValueError("Actual DDP task-slot assignment differs from the recorded plan")
                if self.plan["index"][occurrence_ids].tolist() != record["sample_ids"]:
                    raise ValueError("Occurrence IDs no longer refer to the recorded frames")
                record["occurrence_ids"] = occurrence_ids
                # Transfer each tensor once; per-occurrence hashes verify E/T after reordering.
                cpu_draws = {key: value.detach().cpu() for key, value in draws.items()}
                for key, value in cpu_draws.items():
                    record[f"occurrence_{key}_sha256"] = [tensor_hash(row) for row in value]
            with self.audit_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return draws


class PromptUpdateMonitor:
    """Only 4096 prompt parameters are copied. No RNG, forward pass or optimizer changes."""

    def __init__(self, policy):
        self.parameter = policy.model.vlm_prompt_tokens.weight

    def before_step(self):
        self.before = self.parameter.detach().float().clone()
        gradient = self.parameter.grad
        self.gradient_norm = float(gradient.detach().float().norm()) if gradient is not None else 0.0

    def after_step(self):
        norm = float(self.before.norm())
        delta = float((self.parameter.detach().float() - self.before).norm())
        del self.before
        return {
            "prompt_update/grad_norm": self.gradient_norm,
            "prompt_update/parameter_norm": norm,
            "prompt_update/update_norm": delta,
            "prompt_update/relative_update": delta / max(norm, 1e-12),
        }
