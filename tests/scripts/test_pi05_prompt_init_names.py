"""A source-name migration must not regenerate prompts or break historical paths."""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from lerobot.scripts.pdet_experiment import check_initializations
from lerobot.utils.prompt_init_names import (
    libero_prompt_names,
    rename_libero_prompts,
    resolve_libero_prompt,
)
from lerobot.utils.prompt_initialization import prompt_hash, read_initial_prompt, write_prompt


def make_prompt(directory, seed, *, source="/data/models/lerobot/pi05_libero_base", canonical=False):
    paths = libero_prompt_names(directory, seed)
    path = paths[0 if canonical else 1]
    prompt = np.full((2, 2048), seed + 0.125, dtype=np.float32)
    write_prompt(
        path,
        prompt,
        {
            "schema_version": 1,
            "source_step": 0,
            "source_training_seed": seed,
            "source_pretrained_path": source,
            "num_vlm_prompt_tokens": 2,
            "prompt_sha256": prompt_hash(prompt),
        },
    )
    return path


def test_migration_keeps_exact_bytes_provenance_and_legacy_readability(tmp_path):
    originals = {seed: make_prompt(tmp_path, seed).read_bytes() for seed in (0, 1)}
    before = check_initializations(tmp_path)
    report = rename_libero_prompts(tmp_path)
    assert report["renamed"] == "passed"
    assert report["tensor_values_and_metadata"] == "unchanged"
    for seed, row in enumerate(report["files"]):
        canonical, legacy = libero_prompt_names(tmp_path, seed)
        assert canonical.is_file() and not canonical.is_symlink()
        assert legacy.is_symlink() and os.readlink(legacy) == canonical.name
        assert canonical.read_bytes() == legacy.read_bytes() == originals[seed]
        assert row["file_sha256"] == hashlib.sha256(originals[seed]).hexdigest()
        assert resolve_libero_prompt(tmp_path, seed) == canonical
        assert (
            read_initial_prompt(legacy, 2)[1]["source_pretrained_path"]
            == "/data/models/lerobot/pi05_libero_base"
        )
    assert check_initializations(tmp_path) == before
    assert rename_libero_prompts(tmp_path) == report


def test_resolver_accepts_legacy_and_canonical_but_rejects_conflicts(tmp_path):
    legacy = make_prompt(tmp_path, 1)
    assert resolve_libero_prompt(tmp_path, 1) == legacy
    canonical = make_prompt(tmp_path, 1, canonical=True)
    assert resolve_libero_prompt(tmp_path, 1) == canonical
    canonical.unlink()
    make_prompt(tmp_path, 1, canonical=True, source="another/path/pi05_libero_base")
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_libero_prompt(tmp_path, 1)
    with pytest.raises(ValueError, match="Conflicting"):
        rename_libero_prompts(tmp_path, [1])
    assert not legacy.is_symlink()


@pytest.mark.parametrize("source", ["/models/pi05_base", "", "/other/unknown"])
def test_migration_validates_all_sources_before_mutating_any_file(tmp_path, source):
    old0 = make_prompt(tmp_path, 0)
    old1 = make_prompt(tmp_path, 1, source=source)
    with pytest.raises(ValueError, match="recorded source_pretrained_path"):
        rename_libero_prompts(tmp_path)
    assert not old0.is_symlink() and not old1.is_symlink()
    assert not libero_prompt_names(tmp_path, 0)[0].exists()
    assert not libero_prompt_names(tmp_path, 1)[0].exists()


def test_migration_refuses_circular_aliases_and_broken_links(tmp_path):
    old = make_prompt(tmp_path, 0)
    canonical, _ = libero_prompt_names(tmp_path, 0)
    canonical.symlink_to(old.name)
    with pytest.raises(ValueError, match="regular file"):
        rename_libero_prompts(tmp_path, [0])
    canonical.unlink()
    canonical.symlink_to("missing.safetensors")
    with pytest.raises(ValueError, match="Broken link"):
        resolve_libero_prompt(tmp_path, 0)


def test_explicit_canonical_export_can_get_a_compatibility_alias(tmp_path):
    canonical = make_prompt(tmp_path, 1, canonical=True)
    original = canonical.read_bytes()
    rename_libero_prompts(tmp_path, [1])
    legacy = libero_prompt_names(tmp_path, 1)[1]
    assert legacy.is_symlink() and legacy.read_bytes() == original


def test_cli_rename_and_resolve_emit_usable_paths(tmp_path):
    make_prompt(tmp_path, 0)
    make_prompt(tmp_path, 1)
    base = [sys.executable, "-m", "lerobot.scripts.export_pi05_prompt_init"]
    result = subprocess.run(base + ["rename-libero-inits", str(tmp_path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["renamed"] == "passed"
    result = subprocess.run(
        base + ["resolve-libero-init", str(tmp_path), "--seed", "1"], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(libero_prompt_names(tmp_path, 1)[0]) + "\n"


def test_missing_or_wrong_seed_is_not_silently_regenerated(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing saved P1"):
        resolve_libero_prompt(tmp_path, 1)
    path = make_prompt(tmp_path, 0)
    path.rename(libero_prompt_names(tmp_path, 1)[1])
    with pytest.raises(ValueError, match="unswapped P1"):
        resolve_libero_prompt(tmp_path, 1)
