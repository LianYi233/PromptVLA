"""Portable step-zero VLM prompts, loaded without drawing or reseeding random numbers."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save

KEY = "vlm_prompt"
METADATA_KEY = "lerobot_prompt_initialization"


def prompt_hash(prompt: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(prompt, dtype="<f4").tobytes(order="C")).hexdigest()


def validate_array(prompt: np.ndarray, tokens: int | None = None) -> None:
    if (
        prompt.dtype != np.float32
        or prompt.ndim != 2
        or prompt.shape[0] < 1
        or prompt.shape[1] != 2048
        or (tokens is not None and prompt.shape[0] != tokens)
        or not np.isfinite(prompt).all()
    ):
        raise ValueError(
            f"Expected finite float32 VLM prompt [{tokens or 'N'}, 2048], got {prompt.shape}/{prompt.dtype}"
        )


def write_prompt(path: str | Path, prompt: np.ndarray, metadata: dict) -> None:
    """Use exclusive creation: never overwrite an initialization or previous experiment."""
    validate_array(prompt)
    destination = Path(path)
    if destination.suffix != ".safetensors":
        raise ValueError("Prompt initialization must use a .safetensors filename")
    payload = save(
        {KEY: np.ascontiguousarray(prompt)},
        metadata={METADATA_KEY: json.dumps(metadata, sort_keys=True, allow_nan=False)},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)


def export_initial_prompt(run: str | Path, output: str | Path, expect_seed: int) -> dict:
    """Export only step 000000; final learned prompts are never used as initialization."""
    run = Path(run).resolve()
    snapshot = run / "prompt_rank/step_000000.npz"
    manifest = json.loads((run / "prompt_rank/manifest.json").read_text())
    configs = sorted((run / "checkpoints").glob("[0-9]*/pretrained_model/train_config.json"))
    if not configs:
        raise ValueError("No saved checkpoint train_config.json; cannot verify the source training seed")
    config_path = configs[-1]
    train_config = json.loads(config_path.read_text())
    policy = train_config.get("policy", {})
    seed = train_config.get("seed")
    if type(seed) is not int or seed != expect_seed:
        raise ValueError(f"Source training seed is {seed!r}, expected {expect_seed}")
    tokens = policy.get("num_vlm_prompt_tokens")
    if (
        manifest.get("schema_version") != 1
        or type(tokens) is not int
        or tokens < 1
        or tokens != manifest.get("num_vlm_prompt_tokens")
        or policy.get("type") != "pi05"
        or not policy.get("vlm_prompt_only")
        or policy.get("num_prompt_tokens") != 0
    ):
        raise ValueError("Source must be a recorded native PI05 VLM-only run with matching token count")
    with np.load(snapshot, allow_pickle=False) as arrays:
        prompt = arrays["prompt"].copy()
    validate_array(prompt, tokens)
    metadata = {
        "schema_version": 1,
        "source_step": 0,
        "source_run": str(run),
        "source_training_seed": seed,
        "source_prompt_init_path": train_config.get("vlm_prompt_init_path"),
        "source_pretrained_path": manifest.get("pretrained_path"),
        "source_forward_precision": manifest.get("forward_precision"),
        "source_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "source_train_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "num_vlm_prompt_tokens": tokens,
        "prompt_sha256": prompt_hash(prompt),
    }
    write_prompt(output, prompt, metadata)
    return metadata


def read_initial_prompt(path: str | Path, tokens: int | None = None) -> tuple[np.ndarray, dict]:
    with safe_open(str(path), framework="np") as handle:
        if list(handle.keys()) != [KEY]:
            raise ValueError("Expected an exported prompt-only file, not a full model checkpoint")
        metadata = json.loads((handle.metadata() or {}).get(METADATA_KEY, "{}"))
        prompt = handle.get_tensor(KEY).copy()
    validate_array(prompt, tokens)
    if (
        metadata.get("schema_version") != 1
        or metadata.get("source_step") != 0
        or type(metadata.get("source_training_seed")) is not int
        or metadata.get("num_vlm_prompt_tokens") != prompt.shape[0]
        or metadata.get("prompt_sha256") != prompt_hash(prompt)
    ):
        raise ValueError("Invalid step-zero prompt provenance or tensor checksum")
    return prompt, metadata


def apply_initial_prompt(policy, path: str | Path) -> dict:
    """Replace values in-place after normal seeded construction/loading, before DDP/optimizer.

    In particular, do NOT skip the normal random initialization, reseed training,
    replace the Parameter, or load a source run's optimizer/RNG state.
    """
    import torch

    from lerobot.utils.pi05_trainable import is_vlm_adaptation, validate_vlm_trainables

    config = policy.config
    if not is_vlm_adaptation(config) or getattr(config, "num_prompt_tokens", None) != 0:
        raise ValueError("Prompt initialization import requires native PI05 VLM adaptation")
    parameter = policy.model.vlm_prompt_tokens.weight
    validate_vlm_trainables(policy)
    prompt, metadata = read_initial_prompt(path, config.num_vlm_prompt_tokens)
    if tuple(parameter.shape) != tuple(prompt.shape):
        raise ValueError(f"Prompt shape mismatch: {tuple(parameter.shape)} vs {prompt.shape}")
    value = torch.from_numpy(prompt).to(device=parameter.device, dtype=parameter.dtype)
    # Reject silent precision changes: the new run must start from the exact saved values.
    if not np.array_equal(value.float().cpu().numpy(), prompt):
        raise ValueError("Saved initialization cannot be represented exactly in the target prompt dtype")
    before = prompt_hash(parameter.detach().float().cpu().numpy())
    with torch.no_grad():
        parameter.copy_(value)
    return {
        **metadata,
        "loaded_from": str(Path(path).resolve()),
        "replaced_prompt_sha256": before,
        "loaded_prompt_sha256": prompt_hash(parameter.detach().float().cpu().numpy()),
        "scope": "swap initial prompt only; preserve all remaining training RNG streams",
    }


def save_import_record(policy, record: dict, output_dir: str | Path, training_seed: int) -> None:
    """Keep a portable copy even if the original transfer file is later moved."""
    output_dir = Path(output_dir)
    prompt = policy.model.vlm_prompt_tokens.weight.detach().float().cpu().numpy()
    record = {**record, "training_seed": training_seed}
    write_prompt(output_dir / "prompt_init.safetensors", prompt, record)
    with (output_dir / "prompt_init.json").open("x") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def validate_export_request(output, reference, seed, tokens):
    """Check an export destination and optional historical PRE-import hash before loading a model."""
    destination = Path(output)
    if destination.suffix != ".safetensors":
        raise ValueError("Prompt export must use a .safetensors filename")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Prompt export already exists: {destination}")
    if type(seed) is not int or not 0 <= seed < 2**32 or type(tokens) is not int or tokens < 1:
        raise ValueError("Expected a seed in [0, 2**32) and a positive token count")
    if reference is None:
        return None
    record = json.loads(Path(reference).read_text())
    digest = record.get("replaced_prompt_sha256")
    if (
        record.get("schema_version") != 1
        or record.get("source_step") != 0
        or type(record.get("training_seed")) is not int
        or record["training_seed"] != seed
        or record.get("num_vlm_prompt_tokens") != tokens
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("Reference must be prompt_init.json from a swap with matching TRAINING seed/tokens")
    return record


def constructed_prompt_packet(policy, cfg):
    """Package the actual initialized tensor; never approximate construction with a standalone randn."""
    import torch

    if (
        cfg.resume
        or cfg.vlm_prompt_init_path is not None
        or not getattr(policy.config, "vlm_prompt_only", False)
        or getattr(policy.config, "num_prompt_tokens", None) != 0
    ):
        raise ValueError("Export requires fresh VLM-only initialization without resume or prompt import")
    parameter = policy.model.vlm_prompt_tokens.weight
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if len(trainable) != 1 or trainable[0] is not parameter:
        raise ValueError("Only the VLM prompt may be trainable when exporting initialization")
    tokens = policy.config.num_vlm_prompt_tokens
    record = validate_export_request(
        cfg.vlm_prompt_export_path, cfg.vlm_prompt_export_reference, cfg.seed, tokens
    )
    prompt = parameter.detach().float().cpu().numpy().copy()
    validate_array(prompt, tokens)
    digest = prompt_hash(prompt)
    if record is not None and digest != record["replaced_prompt_sha256"]:
        raise ValueError(
            "Regenerated prompt does not match the historical pre-import initialization: "
            f"expected {record['replaced_prompt_sha256']}, got {digest}. "
            "No export was written; check source weights, code, precision and package versions."
        )
    metadata = {
        "schema_version": 1,
        "source_step": 0,
        "source_kind": "regenerated_training_initialization",
        "source_run": str(cfg.output_dir),
        "source_training_seed": cfg.seed,
        "source_prompt_init_path": None,
        "source_pretrained_path": str(policy.config.pretrained_path),
        "source_forward_precision": policy.config.dtype,
        "source_prompt_dtype": str(parameter.dtype),
        "source_train_config": cfg.to_dict(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "num_vlm_prompt_tokens": tokens,
        "prompt_sha256": digest,
        "historical_hash_verified": record is not None,
    }
    if record is not None:
        metadata.update(
            reference_path=str(Path(cfg.vlm_prompt_export_reference).resolve()),
            reference_sha256=hashlib.sha256(Path(cfg.vlm_prompt_export_reference).read_bytes()).hexdigest(),
            reference_replaced_prompt_sha256=record["replaced_prompt_sha256"],
        )
    return prompt, metadata


def export_constructed_prompt(policy, cfg, accelerator):
    """All ranks verify identical tensors; only rank zero writes, and all ranks exit together."""
    import logging

    import torch

    from lerobot.utils.prompt_rank import collective_call

    prompt, metadata = collective_call(accelerator, lambda: constructed_prompt_packet(policy, cfg))
    digest = torch.tensor(
        list(bytes.fromhex(metadata["prompt_sha256"])), device=accelerator.device, dtype=torch.uint8
    )
    digests = accelerator.gather(digest).reshape(accelerator.num_processes, -1)
    if not (digests == digests[0]).all().item():
        raise ValueError("Constructed prompt initialization differs across DDP ranks; no export was written")
    collective_call(
        accelerator,
        lambda: (
            write_prompt(cfg.vlm_prompt_export_path, prompt, metadata)
            if accelerator.is_main_process
            else None
        ),
    )
    if accelerator.is_main_process:
        logging.info(
            "PROMPT_EXPORT path=%s seed=%s tokens=%s sha256=%s historical_hash_verified=%s; "
            "initialization only, 0 optimizer updates, no full checkpoint",
            cfg.vlm_prompt_export_path,
            cfg.seed,
            metadata["num_vlm_prompt_tokens"],
            metadata["prompt_sha256"],
            metadata["historical_hash_verified"],
        )
    return metadata
