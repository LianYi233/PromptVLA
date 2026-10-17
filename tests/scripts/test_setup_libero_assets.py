"""Exercise asset relocation without networking, LIBERO imports or CUDA."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from lerobot.scripts.setup_libero_assets import ASSET_DIRS, SCENE, configure_assets


def layout(tmp_path):
    assets, package, config = (tmp_path / name for name in ("download", "package", "config"))
    for name in ASSET_DIRS:
        (assets / name).mkdir(parents=True)
        (assets / name / "fixture").write_text("asset")
    (assets / SCENE).write_text("<mujoco/>")
    for name in ("bddl_files", "init_files"):
        (package / name).mkdir(parents=True)
    return assets, package, config


def test_preserves_assets_and_custom_paths_and_is_idempotent(tmp_path):
    assets, package, config = layout(tmp_path)
    (package / "assets").mkdir()
    (package / "assets/old").write_text("keep this")
    custom_tasks = tmp_path / "custom tasks"
    custom_tasks.mkdir()
    config.mkdir()
    original = {"bddl_files": str(custom_tasks), "init_states": "/missing/old/server", "extra": 42}
    (config / "config.yaml").write_text(yaml.safe_dump(original))
    configure_assets(assets, package, config)
    assert (package / "assets").resolve() == assets
    assert next(package.glob("assets.bak-*/old")).read_text() == "keep this"
    assert yaml.safe_load(next(config.glob("config.yaml.bak-*")).read_text()) == original
    settings = yaml.safe_load((config / "config.yaml").read_text())
    assert settings["bddl_files"] == str(custom_tasks)
    assert settings["init_states"] == str(package / "init_files")
    assert settings["assets"] == str(assets)
    assert settings["extra"] == 42
    configure_assets(assets, package, config)
    assert len(list(package.glob("assets.bak-*"))) == len(list(config.glob("config.yaml.bak-*"))) == 1


def test_incomplete_download_leaves_existing_installation_untouched(tmp_path):
    assets, package, config = layout(tmp_path)
    (assets / SCENE).unlink()
    (package / "assets").mkdir()
    with pytest.raises(FileNotFoundError):
        configure_assets(assets, package, config)
    assert not (package / "assets").is_symlink()
    assert not list(package.glob("assets.bak-*"))
    assert not config.exists()


def test_broken_link_backed_up_and_failed_replacement_restored(tmp_path):
    assets, package, config = layout(tmp_path)
    link = package / "assets"
    link.symlink_to("/missing/previous/assets")
    with patch.object(Path, "symlink_to", side_effect=PermissionError("test permission failure")):
        with pytest.raises(PermissionError):
            configure_assets(assets, package, config)
    assert link.is_symlink() and str(link.readlink()) == "/missing/previous/assets"
    assert not config.exists()
    configure_assets(assets, package, config)
    assert link.resolve() == assets
    assert next(package.glob("assets.bak-*")).is_symlink()
