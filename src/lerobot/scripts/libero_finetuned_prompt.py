"""Preflight the v044 continuation experiment without loading policy weights or CUDA."""

import hashlib
import json
from pathlib import Path

from lerobot.scripts.libero_base_eval import install_base_evaluation, validate_base_checkpoint


def install_finetuned_evaluation(evaluator, tokenizer_path=None):
    """Use the same local tokenizer for baseline and both saved prompt checkpoints."""
    settings = install_base_evaluation(evaluator, tokenizer_path)["base_evaluation"]
    settings.pop("prompt_counts")  # Learned prompt counts come from the checkpoint config.
    settings["processors"] = "saved checkpoint processors/statistics; tokenizer location override"
    settings["experiment_helper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"finetuned_prompt_evaluation": settings}


def normalization_state(directory, filename, registry_name, required_features):
    import numpy as np
    from safetensors.numpy import load_file

    directory = Path(directory)
    pipeline = json.loads((directory / filename).read_text())
    matches = [step for step in pipeline["steps"] if step.get("registry_name") == registry_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {registry_name} in {directory / filename}")
    step = matches[0]
    config = step["config"]
    if not step.get("state_file"):
        raise ValueError(f"Missing saved normalization state in {directory / filename}")
    tensors = load_file(directory / step["state_file"])
    for feature, size in required_features.items():
        feature_config = config["features"].get(feature, {})
        if feature_config.get("shape") != [size]:
            raise ValueError(f"Expected {feature} shape [{size}] in {directory / filename}")
        if config["norm_map"].get(feature_config.get("type")) != "MEAN_STD":
            raise ValueError(f"Expected v044 MEAN_STD normalization for {feature}")
        for stat in ("mean", "std"):
            value = tensors.get(f"{feature}.{stat}")
            if value is None or value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Missing or invalid {feature}.{stat} in {directory / filename}")
            if stat == "std" and (value < 0).any():
                raise ValueError(f"Negative {feature}.std in {directory / filename}")
    return config, tensors


def validate_experiment_checkpoint(source, checkpoint=None, vlm_tokens=16, action_tokens=16):
    import numpy as np
    from safetensors import safe_open

    source = Path(source)
    validate_base_checkpoint(source)
    source_config = json.loads((source / "config.json").read_text())
    if source_config.get("normalization_mapping") != {
        "ACTION": "MEAN_STD",
        "STATE": "MEAN_STD",
        "VISUAL": "IDENTITY",
    }:
        raise ValueError("Expected the v044 policy normalization_mapping (MEAN_STD)")
    candidate = Path(checkpoint) if checkpoint else source
    config = json.loads((candidate / "config.json").read_text())
    if checkpoint:
        # Runtime switches (compile/device/tokenizer) may differ; model semantics must match.
        for key in (
            "type",
            "dtype",
            "input_features",
            "output_features",
            "normalization_mapping",
            "paligemma_variant",
            "action_expert_variant",
            "chunk_size",
            "max_state_dim",
            "max_action_dim",
            "image_resolution",
            "empty_cameras",
            "tokenizer_max_length",
            "num_inference_steps",
            "time_sampling_beta_alpha",
            "time_sampling_beta_beta",
            "time_sampling_scale",
            "time_sampling_offset",
            "min_period",
            "max_period",
        ):
            if config.get(key) != source_config.get(key):
                raise ValueError(f"Checkpoint differs from source in {key}: {candidate}")
        expected = {
            "num_vlm_prompt_tokens": vlm_tokens,
            "num_prompt_tokens": action_tokens,
            "training_stage": "flow",
            "cabo_enabled": False,
            "train_action_projections": False,
            "use_peft": False,
            "next_action_pretrain_steps": 0,
            "next_action_bridge_steps": 0,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Expected {key}={value} in {candidate}")
        with safe_open(candidate / "model.safetensors", framework="np") as weights:
            keys = set(weights.keys())
            for key, count in (
                ("model.vlm_prompt_tokens.weight", vlm_tokens),
                ("model.prompt_tokens.weight", action_tokens),
            ):
                if key not in keys or weights.get_slice(key).get_shape()[0] != count:
                    raise ValueError(f"Missing or mismatched learned prompt: {key}")
    for filename, registry, features in (
        ("policy_preprocessor.json", "normalizer_processor", {"observation.state": 8, "action": 7}),
        ("policy_postprocessor.json", "unnormalizer_processor", {"action": 7}),
    ):
        reference_config, reference_tensors = normalization_state(source, filename, registry, features)
        if checkpoint:
            saved_config, saved_tensors = normalization_state(candidate, filename, registry, features)
            if saved_config != reference_config or saved_tensors.keys() != reference_tensors.keys():
                raise ValueError(f"Normalization configuration differs from source: {candidate / filename}")
            for key, value in reference_tensors.items():
                if not np.array_equal(value, saved_tensors[key]):
                    raise ValueError(f"Normalization statistics differ from source: {key}")
    return {
        "source": str(source.resolve()),
        "checkpoint": str(candidate.resolve()),
        "normalization": "MEAN_STD",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm-tokens", type=int, default=16)
    parser.add_argument("--action-tokens", type=int, default=16)
    args = parser.parse_args()
    try:
        result = validate_experiment_checkpoint(
            args.source, args.checkpoint, args.vlm_tokens, args.action_tokens
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Finetuned prompt experiment preflight failed: {error}\n")
    print(json.dumps(result))
