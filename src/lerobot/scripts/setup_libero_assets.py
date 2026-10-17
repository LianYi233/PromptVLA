"""Download external LIBERO assets and configure the active Python installation.

Does not import LIBERO (its first import can prompt or start a download), load a
policy, or create a simulator. Existing asset locations/configs are backed up.
"""

import argparse
import importlib.util
import os
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

ASSET_DIRS = (
    "articulated_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
    "stable_hope_objects",
    "scenes",
    "textures",
)
SCENE = "scenes/libero_tabletop_base_style.xml"


def validate_assets(assets: Path) -> None:
    for name in ASSET_DIRS:
        directory = assets / name
        if not directory.is_dir() or not any(directory.rglob("*")):
            raise FileNotFoundError(f"Missing or empty asset directory: {directory}")
    # Reject a missing, empty or Git-LFS-pointer XML before changing any links.
    ET.parse(assets / SCENE)


def find_libero_root() -> Path:
    spec = importlib.util.find_spec("libero")
    for location in (spec.submodule_search_locations or []) if spec else []:
        candidate = Path(location) / "libero"
        if (candidate / "envs/bddl_base_domain.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot locate libero/libero in this Python environment; activate the LIBERO environment first"
    )


def backup_name(path: Path) -> Path:
    return path.with_name(f"{path.name}.bak-{uuid.uuid4().hex[:12]}")


def configure_assets(assets: Path, package: Path, config_dir: Path) -> dict:
    import yaml

    assets, package = assets.resolve(), package.resolve()
    validate_assets(assets)
    config_file = config_dir / "config.yaml"
    old_config = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
    if old_config is None:
        old_config = {}
    if not isinstance(old_config, dict):
        raise ValueError(f"Expected a mapping in {config_file}")
    config = dict(old_config)
    # Preserve valid custom task paths; repair absent/stale paths after a machine move.
    for key, default in (
        ("benchmark_root", package),
        ("bddl_files", package / "bddl_files"),
        ("init_states", package / "init_files"),
    ):
        value = config.get(key)
        if not isinstance(value, str) or not Path(value).is_dir():
            config[key] = str(default)
        if not Path(config[key]).is_dir():
            raise FileNotFoundError(f"LIBERO {key} directory not found: {config[key]}")
    config.setdefault("datasets", str(package.parent / "datasets"))
    config["assets"] = str(assets)

    link = package / "assets"
    if assets != link.resolve():
        if assets.is_relative_to(link.absolute()):
            raise ValueError("The external assets target must not be inside the package assets directory")
        backup = None
        if link.exists() or link.is_symlink():
            backup = backup_name(link)
            link.rename(backup)
        try:
            link.symlink_to(assets, target_is_directory=True)
        except OSError:
            if backup is not None:
                backup.rename(link)
            raise
        if backup is not None:
            print(f"Previous assets backed up: {backup}", flush=True)

    config_dir.mkdir(parents=True, exist_ok=True)
    if config != old_config:
        if config_file.exists():
            backup = backup_name(config_file)
            shutil.copy2(config_file, backup)
            print(f"Previous config backed up: {backup}", flush=True)
        fd, temporary = tempfile.mkstemp(dir=config_dir, prefix="config-", suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            os.replace(temporary, config_file)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if not (link / SCENE).is_file():
        raise FileNotFoundError(f"Package cannot resolve {link / SCENE}")
    return {"assets": str(assets), "package_assets": str(link), "config": str(config_file)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=Path("/data1/datasets/libero-assets"))
    parser.add_argument(
        "--config-dir", type=Path, default=Path(os.environ.get("LIBERO_CONFIG_PATH", "/home/wyn/.libero"))
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Validate/configure an already downloaded directory offline"
    )
    args = parser.parse_args()
    if not args.skip_download:
        # Import only after setting the transport option, before HF constants are loaded.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import snapshot_download

        print(f"Downloading lerobot/libero-assets (dataset) -> {args.assets_dir}", flush=True)
        snapshot_download(
            repo_id="lerobot/libero-assets", repo_type="dataset", local_dir=args.assets_dir, max_workers=1
        )
    result = configure_assets(args.assets_dir, find_libero_root(), args.config_dir)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("LIBERO assets ready (resource paths verified; simulator not started).")


if __name__ == "__main__":
    main()
