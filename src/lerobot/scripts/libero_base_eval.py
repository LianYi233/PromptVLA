"""Read-only base-checkpoint preflight and evaluation-only tokenizer overrides."""

import functools
import hashlib
import json
from pathlib import Path


def validate_base_checkpoint(directory, tokenizer_path=None):
    from safetensors import safe_open

    directory = Path(directory)
    required = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    for name in required:
        if not (directory / name).is_file():
            raise ValueError(f"Missing base checkpoint file: {directory / name}")
    config = json.loads((directory / "config.json").read_text())
    if config.get("type") != "pi05":
        raise ValueError(f"Expected PI05 config: {directory / 'config.json'}")
    # Inspect tensor shapes without loading multi-GB weights or importing torch.
    with safe_open(directory / "model.safetensors", framework="np") as weights:
        for key in weights.keys():  # noqa: SIM118 - safe_open is not an iterable mapping.
            if key.endswith(("vlm_prompt_tokens.weight", "prompt_tokens.weight")):
                shape = weights.get_slice(key).get_shape()
                if not shape or shape[0] != 0:
                    raise ValueError(
                        f"Base checkpoint contains a nonempty prompt bank: {key} {shape}. "
                        "Use the original base weights, not a trained prompt checkpoint."
                    )
    if tokenizer_path and not Path(tokenizer_path).is_dir():
        raise ValueError(f"TOKENIZER_PATH does not exist: {tokenizer_path}")
    return {
        "model_directory": str(directory.resolve()),
        "saved_prompt_counts": {
            "vlm": config.get("num_vlm_prompt_tokens"),
            "action": config.get("num_prompt_tokens"),
        },
        "effective_prompt_counts": {"vlm": 0, "action": 0},
    }


def install_base_evaluation(evaluator, tokenizer_path=None):
    """Keep saved normalization/state/image processing; replace only tokenizer location."""
    tokenizer = str(Path(tokenizer_path).resolve()) if tokenizer_path else None
    if tokenizer:
        if not Path(tokenizer).is_dir():
            raise ValueError(f"TOKENIZER_PATH does not exist: {tokenizer}")
        original = evaluator.make_pre_post_processors

        @functools.wraps(original)
        def make_processors(*args, **kwargs):
            overrides = dict(kwargs.get("preprocessor_overrides") or {})
            overrides["tokenizer_processor"] = {
                **overrides.get("tokenizer_processor", {}),
                "tokenizer_name": tokenizer,
            }
            return original(*args, **{**kwargs, "preprocessor_overrides": overrides})

        evaluator.make_pre_post_processors = make_processors
    return {
        "base_evaluation": {
            "prompt_counts": {"vlm": 0, "action": 0},
            "processors": "base checkpoint processors/statistics; optional tokenizer location override",
            "tokenizer_path": tokenizer,
            "tokenizer_files": [
                [str(path.relative_to(tokenizer)), path.stat().st_size, path.stat().st_mtime_ns]
                for path in sorted(Path(tokenizer).rglob("*"))
                if path.is_file()
            ]
            if tokenizer
            else [],
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("tokenizer_path", nargs="?", default="")
    args = parser.parse_args()
    try:
        result = validate_base_checkpoint(args.directory, args.tokenizer_path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Base checkpoint preflight failed: {error}\n")
    print("Base evaluation: " + json.dumps(result))
