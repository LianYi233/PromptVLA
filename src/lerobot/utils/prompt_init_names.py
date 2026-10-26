"""Unambiguous names for exported LIBERO-source prompts, with legacy path compatibility."""

import hashlib
import os
import tempfile
from pathlib import Path

from lerobot.utils.prompt_initialization import read_initial_prompt


def libero_prompt_names(directory, seed, tokens=2):
    if type(seed) is not int or seed < 0 or type(tokens) is not int or tokens < 1:
        raise ValueError("Seed must be nonnegative and token count positive")
    directory = Path(directory)
    return (
        directory / f"pi05_libero_base-vlm{tokens}-init-seed{seed}.safetensors",
        directory / f"pi05-base-vlm{tokens}-init-seed{seed}.safetensors",
    )


def resolve_libero_prompt(directory, seed, tokens=2):
    """Prefer the explicit source name; never silently choose between conflicting files."""
    canonical, legacy = libero_prompt_names(directory, seed, tokens)
    existing = [path for path in (canonical, legacy) if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"Missing saved P{seed}: expected {canonical} (legacy: {legacy})")
    for path in (canonical, legacy):
        if os.path.lexists(path) and not path.is_file():
            raise ValueError(f"Broken link or non-file at prompt path: {path}")
    if len(existing) == 2 and canonical.read_bytes() != legacy.read_bytes():
        raise ValueError(f"Conflicting canonical/legacy initialization files: {canonical} and {legacy}")
    _, metadata = read_initial_prompt(existing[0], tokens)
    if metadata["source_training_seed"] != seed or metadata.get("source_prompt_init_path"):
        raise ValueError(f"Expected original unswapped P{seed} initialization")
    return existing[0]


def rename_libero_prompts(directory, seeds=(0, 1), tokens=2):
    """Keep file bytes/provenance intact; replace old names with relative symlink aliases.

    Validate every selected file before writing anything. The new filename is justified
    by recorded construction provenance, never by the ambiguous old filename alone.
    """
    directory = Path(directory).resolve()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Select at least one unique seed")
    pending = []
    for seed in seeds:
        selected = resolve_libero_prompt(directory, seed, tokens)
        payload = selected.read_bytes()
        _, metadata = read_initial_prompt(selected, tokens)
        source = metadata.get("source_pretrained_path")
        if not isinstance(source, str) or Path(source.rstrip("/")).name != "pi05_libero_base":
            raise ValueError(
                f"Cannot label {selected} as pi05_libero_base: recorded source_pretrained_path={source!r}. "
                "Inspect provenance; no files have been renamed."
            )
        canonical, legacy = libero_prompt_names(directory, seed, tokens)
        if canonical.is_symlink():
            raise ValueError(f"Canonical path must be a regular file before migration: {canonical}")
        pending.append((canonical, legacy, payload, metadata))
    result = []
    for canonical, legacy, payload, metadata in pending:
        # Exclusive creation: an existing matching canonical file is reused, not overwritten.
        if not canonical.is_file():
            with canonical.open("xb") as stream:
                stream.write(payload)
        if canonical.read_bytes() != payload or (legacy.is_file() and legacy.read_bytes() != payload):
            raise ValueError("Initialization changed during rename; refusing to replace the old path")
        if not (legacy.is_symlink() and os.readlink(legacy) == canonical.name):
            # Atomic replacement keeps the old path valid for readers/history after migration.
            with tempfile.TemporaryDirectory(prefix=".prompt-name-", dir=directory) as temporary:
                alias = Path(temporary) / "alias"
                alias.symlink_to(canonical.name)
                os.replace(alias, legacy)
        result.append(
            {
                "canonical": str(canonical),
                "legacy_alias": str(legacy),
                "source_pretrained_path": metadata["source_pretrained_path"],
                "seed": metadata["source_training_seed"],
                "tokens": tokens,
                "prompt_sha256": metadata["prompt_sha256"],
                "file_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {"renamed": "passed", "tensor_values_and_metadata": "unchanged", "files": result}
