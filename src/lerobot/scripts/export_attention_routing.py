"""Export per-head/pass all-key attention accounting without loading a policy.

Usage: python -m lerobot.scripts.export_attention_routing PATH [PATH ...]
Paths may be .attention.npz files or evaluation directories searched recursively.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def export_routing(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "routing_head_mass" not in data:
            raise ValueError(f"{path}: no full-key routing; rerun with ATTENTION_RECORD_ROUTING=1")
        names = data["routing_group_names"].tolist()
        masses = data["routing_head_mass"]  # prediction, pass, head, group
        counts = data["routing_visible_key_counts"]
        tokens = data["routing_token_mass"]
        ids = data["routing_key_group_ids"]
        token_ids = data["routing_token_ids"]
        steps = data["input_steps"]
    if not np.isfinite(masses).all() or (masses < 0).any():
        raise ValueError("Non-finite or negative routing probability")
    closure = float(np.max(np.abs(masses.sum(axis=-1) - 1)))
    if closure > 2e-6:
        raise ValueError(f"Group probability closure error: {closure}")
    metadata = json.loads(path.with_suffix(".json").read_text())
    denoise = metadata["denoise"]

    def reduce_pass(values):
        if denoise == "first":
            return values[:, 0]
        if denoise == "last":
            return values[:, -1]
        return values.mean(axis=1)

    rows_path = path.with_suffix(".routing.csv")
    with rows_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "input_step",
                "pass_zero_based",
                "head",
                "group",
                "mass",
                "mean_visible_keys",
                "mass_per_visible_key",
                "ratio_to_uniform_visible_attention",
            ]
        )
        for prediction, step in enumerate(steps):
            for denoising_pass in range(masses.shape[1]):
                total_visible = counts[prediction, denoising_pass].sum()
                for head in range(masses.shape[2]):
                    for group, name in enumerate(names):
                        mass = float(masses[prediction, denoising_pass, head, group])
                        count = float(counts[prediction, denoising_pass, group])
                        per_key = mass / count if count else None
                        writer.writerow(
                            [
                                int(step),
                                denoising_pass,
                                head,
                                name,
                                mass,
                                count,
                                per_key,
                                per_key * total_visible if per_key is not None else None,
                            ]
                        )
    selected = reduce_pass(masses).mean(axis=1)  # prediction, group
    summary = {
        "schema_version": 1,
        "source_recording": str(path),
        "source": metadata["source"],
        "layer_zero_based": metadata["resolved_layer_zero_based"],
        "prediction_count": len(steps),
        "passes_per_prediction": masses.shape[1],
        "head_count": masses.shape[2],
        "query_reduction": "selected query rows averaged; individual query rows not retained",
        "max_probability_closure_error": closure,
        "all_pass_head_prediction_mean_mass": dict(
            zip(names, masses.mean(axis=(0, 1, 2)).tolist(), strict=True)
        ),
        "display_denoise_selection": denoise,
        "display_prediction_mean_mass": dict(zip(names, selected.mean(axis=0).tolist(), strict=True)),
        "input_steps": steps.tolist(),
        "display_mass_by_prediction": selected.tolist(),
        "group_order": names,
        "note": "Attention routing only; mass is not causal importance. Predictions weighted equally within this episode.",
    }
    summary_path = path.with_suffix(".routing_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    top_path = path.with_suffix(".routing_top_tokens.csv")
    with top_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["input_step", "key_position", "group", "index_in_group", "token_id", "mass"])
        for prediction, values in enumerate(reduce_pass(tokens)):
            for key in np.argsort(-values, kind="stable")[:20]:
                group = ids[prediction, key]
                writer.writerow(
                    [
                        int(steps[prediction]),
                        int(key),
                        names[group],
                        int(np.count_nonzero(ids[prediction, :key] == group)),
                        int(token_ids[prediction, key]),
                        float(values[key]),
                    ]
                )
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    paths = sorted(
        {file for path in args.paths for file in (path.rglob("*.attention.npz") if path.is_dir() else [path])}
    )
    if not paths:
        parser.error("No .attention.npz files found")
    for path in paths:
        print(export_routing(path))


if __name__ == "__main__":
    main()
