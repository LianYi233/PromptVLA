"""Validate VLM-only runs and build zero-prompt controls with matched processing.

No policy weights are loaded into RAM. A baseline references the original weights
and copies the initial checkpoint's config/processors, so normalization is not a
hidden difference between the zero-prompt and trained conditions.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from lerobot.scripts.libero_base_eval import validate_base_checkpoint
from lerobot.scripts.libero_finetuned_prompt import validate_experiment_checkpoint

PROCESSORS = ("policy_preprocessor.json", "policy_postprocessor.json")


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_source(source, kind, dataset_root=None):
    import numpy as np

    validate_base_checkpoint(source)
    config = read_json(Path(source) / "config.json")
    if config.get("paligemma_variant") != "gemma_2b" or config.get("action_expert_variant") != "gemma_300m":
        raise ValueError("Expected the official PI05 gemma_2b/gemma_300m architecture")
    if kind == "finetuned":
        validate_experiment_checkpoint(source)
    elif kind == "base":
        if dataset_root:
            stats = read_json(Path(dataset_root) / "meta/stats.json")
            for feature, size in (("observation.state", 8), ("action", 7)):
                for stat in ("q01", "q99"):
                    values = np.asarray(stats.get(feature, {}).get(stat, []))
                    if values.shape != (size,) or not np.isfinite(values).all():
                        raise ValueError(f"Base recipe needs valid {feature}.{stat} in meta/stats.json")
    else:
        raise ValueError(f"Unknown source: {kind}")


def processor_files(directory):
    directory = Path(directory)
    files = set(PROCESSORS)
    for filename in PROCESSORS:
        for step in read_json(directory / filename)["steps"]:
            if state_file := step.get("state_file"):
                if Path(state_file).is_absolute() or ".." in Path(state_file).parts:
                    raise ValueError(f"Processor state must be relative to its checkpoint: {state_file}")
                files.add(state_file)
    for filename in files:
        if not (directory / filename).is_file():
            raise ValueError(f"Missing processor file: {directory / filename}")
    return sorted(files)


def validate_processors(directory, config):
    import numpy as np
    from safetensors.numpy import load_file

    directory = Path(directory)
    processor_files(directory)
    for filename, registry, features in (
        (PROCESSORS[0], "normalizer_processor", {"observation.state": 8, "action": 7}),
        (PROCESSORS[1], "unnormalizer_processor", {"action": 7}),
    ):
        steps = [
            step for step in read_json(directory / filename)["steps"] if step["registry_name"] == registry
        ]
        if len(steps) != 1 or not steps[0].get("state_file"):
            raise ValueError(f"Missing saved normalization in {directory / filename}")
        step = steps[0]
        stats = load_file(directory / step["state_file"])
        if step["config"]["norm_map"] != config["normalization_mapping"]:
            raise ValueError("Policy and processor normalization modes differ")
        for feature, size in features.items():
            ft = step["config"]["features"].get(feature, {})
            if ft.get("shape") != [size]:
                raise ValueError(f"Wrong processor feature shape: {feature}")
            mode = config["normalization_mapping"].get(ft.get("type"))
            keys = {"QUANTILES": ("q01", "q99"), "MEAN_STD": ("mean", "std")}.get(mode)
            if keys is None:
                raise ValueError(f"Unsupported normalization: {mode}")
            for key in keys:
                value = stats.get(f"{feature}.{key}")
                if value is None or value.shape != (size,) or not np.isfinite(value).all():
                    raise ValueError(f"Missing/invalid normalization tensor: {feature}.{key}")


def _loaded_processor_stat(value, key, step):
    """Compare effective normalization statistics without loading torch or weights.

    The loader casts saved statistics to float32, then reshapes every 1-D visual
    statistic to (-1, 1, 1). Its reconstructed stats dict converts single-element
    tensors to Python scalars, so a later processor.to() saves them as shape ().
    Counts and stats for keys absent from the configured features (episode_index,
    timestamp, etc.) do not participate in normalization. Canonicalize their
    singleton shapes, keeping values exact. Do not allow numeric tolerances or
    reshape other action/state statistics, which could hide normalization drift.
    """
    import numpy as np

    if step["registry_name"] in {"normalizer_processor", "unnormalizer_processor"}:
        value = value.astype(np.float32)
        feature, statistic = key.rsplit(".", 1)
        features = step["config"]["features"]
        if value.size == 1 and (statistic == "count" or feature not in features):
            return value.reshape(())
        if features.get(feature, {}).get("type") == "VISUAL" and value.ndim == 1:
            value = value.reshape(-1, 1, 1)
    return value


def assert_same_processors(reference, candidate):
    import numpy as np
    from safetensors.numpy import load_file

    for filename in PROCESSORS:
        left = read_json(Path(reference) / filename)["steps"]
        right = read_json(Path(candidate) / filename)["steps"]
        context = f"{filename}: reference={reference}; candidate={candidate}"
        if len(left) != len(right):
            raise ValueError(f"Processor steps differ between checkpoints ({context})")
        for a, b in zip(left, right, strict=True):
            if a.get("registry_name") != b.get("registry_name"):
                raise ValueError(f"Processor order differs between checkpoints ({context})")
            # Only runtime device and local tokenizer location may move between machines.
            ac, bc = dict(a["config"]), dict(b["config"])
            if a["registry_name"] == "device_processor":
                ac.pop("device", None)
                bc.pop("device", None)
            if a["registry_name"] == "tokenizer_processor":
                ac.pop("tokenizer_name", None)
                bc.pop("tokenizer_name", None)
            if ac != bc or bool(a.get("state_file")) != bool(b.get("state_file")):
                raise ValueError(f"Processor configuration differs between checkpoints ({context})")
            if a.get("state_file"):
                x = load_file(Path(reference) / a["state_file"])
                y = load_file(Path(candidate) / b["state_file"])
                if x.keys() != y.keys():
                    raise ValueError(
                        f"Processor statistics differ between checkpoints ({context}): "
                        f"missing={sorted(x.keys() - y.keys())}; extra={sorted(y.keys() - x.keys())}"
                    )
                differences = []
                for key, value in x.items():
                    loaded_x = _loaded_processor_stat(value, key, a)
                    loaded_y = _loaded_processor_stat(y[key], key, b)
                    if not np.array_equal(loaded_x, loaded_y):
                        detail = (
                            f"{key}: saved shapes {value.shape} -> {y[key].shape}, "
                            f"dtypes {value.dtype} -> {y[key].dtype}"
                        )
                        if loaded_x.shape == loaded_y.shape:
                            delta = np.abs(loaded_x.astype(np.float64) - loaded_y.astype(np.float64))
                            detail += f", max_abs_diff_after_load={float(delta.max(initial=0)):.9g}"
                        differences.append(detail)
                if differences:
                    raise ValueError(
                        f"Processor statistics differ between checkpoints ({context}): "
                        + "; ".join(differences)
                    )


def validate_checkpoint(directory, tokens, reference=None, source=None, kind=None):
    from safetensors import safe_open

    directory = Path(directory)
    config = read_json(directory / "config.json")
    expected = {
        "type": "pi05",
        "vlm_prompt_only": True,
        "num_vlm_prompt_tokens": tokens,
        "num_prompt_tokens": 0,
        "training_stage": "flow",
        "next_action_pretrain_steps": 0,
        "next_action_bridge_steps": 0,
        "cabo_enabled": False,
        "train_action_projections": False,
        "use_peft": False,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"Expected {name}={value} in {directory}")
    with safe_open(directory / "model.safetensors", framework="np") as weights:
        keys = set(weights.keys())
        for key, shape in (
            ("model.vlm_prompt_tokens.weight", [tokens, 2048]),
            ("model.prompt_tokens.weight", [0, 1024]),
        ):
            if key not in keys or weights.get_slice(key).get_shape() != shape:
                raise ValueError(f"Prompt tensor mismatch: {key}, expected {shape}")
    validate_processors(directory, config)
    if source:
        training = read_json(directory / "train_config.json")
        if Path(training["policy"]["pretrained_path"]).resolve() != Path(source).resolve():
            raise ValueError("Checkpoint was trained from a different source path")
        if kind == "finetuned":
            assert_same_processors(source, directory)
    if reference:
        ref_config = read_json(Path(reference) / "config.json")
        for name in set(config) | set(ref_config):
            if name not in {
                "pretrained_path",
                "device",
                "tokenizer_name",
                "compile_model",
                "gradient_checkpointing",
            } and config.get(name) != ref_config.get(name):
                raise ValueError(f"Checkpoint configuration differs from initialization: {name}")
        assert_same_processors(reference, directory)
    return config


def prepare_baseline(source, initial, output, tokens, kind, dry_run=False):
    source, initial, output = Path(source).resolve(), Path(initial).resolve(), Path(output).resolve()
    validate_source(source, kind)
    config = validate_checkpoint(initial, tokens, source=source, kind=kind)
    config.update(num_vlm_prompt_tokens=0, num_prompt_tokens=0, vlm_prompt_only=False)
    files = processor_files(initial)
    weight = source / "model.safetensors"
    manifest = {
        "source": str(source),
        "initial": str(initial),
        "kind": kind,
        "weights_size": weight.stat().st_size,
        "weights_mtime_ns": weight.stat().st_mtime_ns,
        "config": config,
        "processor_sha256": {
            name: hashlib.sha256((initial / name).read_bytes()).hexdigest() for name in files
        },
    }
    if output.exists():
        if (
            read_json(output / "baseline_manifest.json") != manifest
            or read_json(output / "config.json") != config
        ):
            raise ValueError(f"Baseline inputs changed; choose a new output/run directory: {output}")
        if (output / "model.safetensors").resolve() != weight:
            raise ValueError("Baseline no longer references the original source weights")
        for name, digest in manifest["processor_sha256"].items():
            if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Baseline processor changed: {name}")
    elif not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".baseline-", dir=output.parent) as temp:
            target = Path(temp) / "model"
            target.mkdir()
            (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (target / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            for name in files:
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(initial / name, target / name)
            (target / "model.safetensors").symlink_to(weight)
            target.rename(output)
    return str(output)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--kind", choices=("base", "finetuned"), required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.tokens < 1:
            raise ValueError("VLM token count must be positive")
        validate_source(args.source, args.kind, args.dataset_root)
        if args.initial:
            validate_checkpoint(args.initial, args.tokens, source=args.source, kind=args.kind)
        if args.checkpoint:
            # Final-only runs have no initialized model on disk. Still validate
            # source provenance, prompt shapes/recipe and all saved processors;
            # v044 must preserve its source's normalization exactly after loading.
            validate_checkpoint(
                args.checkpoint, args.tokens, reference=args.initial, source=args.source, kind=args.kind
            )
        if args.baseline_output:
            if not args.initial:
                raise ValueError("--baseline-output requires --initial")
            prepare_baseline(
                args.source, args.initial, args.baseline_output, args.tokens, args.kind, args.dry_run
            )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"VLM-only preflight failed: {error}\n")
    print(f"VLM-only preflight OK: source={args.kind} VLM={args.tokens} action=0")


if __name__ == "__main__":
    main()
