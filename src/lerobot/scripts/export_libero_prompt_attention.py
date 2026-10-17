"""Export synchronized two-camera PNG figures and probability masses from saved diagnostics."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.policies.pi05.attention_visualization import (
    AttentionVideoConfig,
    load_attention_font,
    render_camera_attention_frame,
)


def export_figures(data_path, *, output_dir=None, every=5, steps=None, font_path=None, alpha=None, vmax=None):
    """Keep original input pixels and probabilities; never decode source images from video."""
    if every < 1:
        raise ValueError("--every must be positive")
    data_path = Path(data_path)
    metadata = json.loads(data_path.with_suffix(".json").read_text())
    with np.load(data_path, allow_pickle=False) as data:
        if "camera_images" not in data or "camera_maps" not in data:
            raise ValueError(f"{data_path}: requires a new ATTENTION_CAMERAS=0,1 recording")
        maps, images = data["camera_maps"], data["camera_images"]
        input_steps, camera_indices = data["input_steps"], data["camera_indices"]
    if (
        maps.ndim != 4
        or maps.shape[1] != 2
        or images.ndim != 5
        or images.shape[:2] != maps.shape[:2]
        or images.shape[-1] != 3
        or images.dtype != np.uint8
        or len(input_steps) != len(maps)
        or not len(maps)
        or np.any(np.diff(input_steps) <= 0)
        or not np.isfinite(maps).all()
        or np.any(maps < 0)
        or camera_indices.tolist() != metadata["camera_indices"]
    ):
        raise ValueError(f"Invalid two-camera data or step alignment: {data_path}")
    config = AttentionVideoConfig(
        source=metadata["source"],
        layer=metadata["layer"],
        cameras=",".join(str(int(i)) for i in camera_indices),
        alpha=metadata["alpha"] if alpha is None else alpha,
        vmax=metadata["vmax"] if vmax is None else vmax,
        font_path=font_path or metadata.get("resolved_font_path"),
    )
    font = load_attention_font(config.font_path)
    if steps is None:
        indices = sorted(set(range(0, len(maps), every)) | {len(maps) - 1})
    else:
        missing = set(steps) - set(input_steps.tolist())
        if missing:
            raise ValueError(
                f"Steps {sorted(missing)} are not prediction steps; available: {input_steps.tolist()}"
            )
        indices = [i for i, step in enumerate(input_steps) if int(step) in steps]
    stem = data_path.name.removesuffix(".attention.npz")
    destination = Path(output_dir) if output_dir else data_path.parent / f"{stem}_frames"
    destination.mkdir(parents=True, exist_ok=True)
    for index in indices:
        step = int(input_steps[index])
        frame = render_camera_attention_frame(
            images[index],
            maps[index],
            config,
            font,
            metadata["resolved_layer_zero_based"],
            step,
            step,
        )
        path = destination / f"step_{step:06d}.png"
        temporary = path.with_suffix(".tmp.png")
        Image.fromarray(frame).save(temporary, dpi=(300, 300))
        os.replace(temporary, path)
    # Include every prediction in the CSV, not just the selected snapshots.
    csv_path = destination / "camera_attention_mass.csv"
    temporary = csv_path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "input_step",
                *[f"camera_{i}_mass" for i in camera_indices],
                "other_visible_keys_mass",
                "display_vmax",
            ]
        )
        for step, values in zip(input_steps, maps, strict=True):
            masses = values.sum(axis=(-2, -1))
            writer.writerow(
                [int(step), *masses.tolist(), 1.0 - float(masses.sum()), config.vmax or float(values.max())]
            )
    os.replace(temporary, csv_path)
    (destination / "figure_settings.json").write_text(
        json.dumps(
            {
                "source": str(data_path.resolve()),
                "camera_indices": camera_indices.tolist(),
                "camera_features": metadata["camera_features"],
                "layer": metadata["resolved_layer_zero_based"],
                "input_steps": [int(input_steps[i]) for i in indices],
                "alpha": config.alpha,
                "vmax": config.vmax,
                "font": str(font.path),
                "scale": "shared across cameras; full-key softmax probabilities",
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One .attention.npz file or a directory")
    parser.add_argument("--output-dir", type=Path, help="Optional separate output directory")
    parser.add_argument("--every", type=int, default=5, help="Select every Nth prediction and the last")
    parser.add_argument("--steps", help="Exact prediction steps, e.g. 0,50,100; overrides --every")
    parser.add_argument("--font-path", default=os.environ.get("ATTENTION_FONT_PATH"))
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--vmax", type=float)
    args = parser.parse_args()
    files = [args.input] if args.input.is_file() else sorted(args.input.rglob("*.attention.npz"))
    if not files:
        parser.error("No .attention.npz files found")
    steps = {int(s) for s in args.steps.split(",")} if args.steps else None
    for path in files:
        destination = args.output_dir
        if destination and args.input.is_dir():
            relative = path.relative_to(args.input)
            destination = (
                destination / relative.parent / (relative.name.removesuffix(".attention.npz") + "_frames")
            )
        print(
            export_figures(
                path,
                output_dir=destination,
                every=args.every,
                steps=steps,
                font_path=args.font_path,
                alpha=args.alpha,
                vmax=args.vmax,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
