"""Restyle saved three-panel attention videos without model inference or a GPU."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from lerobot.policies.pi05 import attention_visualization
from lerobot.policies.pi05.attention_visualization import (
    AttentionVideoConfig,
    load_attention_font,
    render_attention_frame,
)


def replot_video(video, target, *, font_path=None, alpha=0.82, vmax=None):
    import av

    metadata_path = video.with_suffix(".attention.json")
    data_path = video.with_suffix(".attention.npz")
    metadata = json.loads(metadata_path.read_text())
    config = AttentionVideoConfig(
        source=metadata["source"],
        camera=metadata["camera"],
        layer=metadata["layer"],
        denoise=metadata["denoise"],
        alpha=alpha,
        vmax=metadata["vmax"] if vmax is None else vmax,
        font_path=font_path,
    )
    font = load_attention_font(font_path)
    with np.load(data_path, allow_pickle=False) as data:
        maps, input_steps = data["maps"], data["input_steps"]
    if (
        maps.ndim != 3
        or input_steps.ndim != 1
        or len(maps) != len(input_steps)
        or not len(maps)
        or input_steps[0] != 0
        or np.any(np.diff(input_steps) <= 0)
        or not np.isfinite(maps).all()
        or np.any(maps < 0)
    ):
        raise ValueError(f"Invalid saved attention maps or step alignment: {data_path}")
    signature = {
        "source_video": str(video.resolve()),
        "source_size": video.stat().st_size,
        "source_mtime_ns": video.stat().st_mtime_ns,
        "maps_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "font_sha256": hashlib.sha256(Path(font.path).read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(
            Path(__file__).read_bytes() + Path(attention_visualization.__file__).read_bytes()
        ).hexdigest(),
        "alpha": alpha,
        "vmax": config.vmax,
    }
    target_metadata = target.with_suffix(".attention.json")
    target_data = target.with_suffix(".attention.npz")
    if target.exists() and target_metadata.exists() and target_data.exists():
        previous = json.loads(target_metadata.read_text())
        if previous.get("restyle") == signature and target.stat().st_size:
            return "already done"
        raise ValueError(f"Different output already exists: {target}; choose a new --output-dir")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.mp4")
    try:
        with av.open(str(video)) as reader, av.open(str(temporary), mode="w") as writer:
            source_stream = reader.streams.video[0]
            if (source_stream.width, source_stream.height) != (768, 320):
                raise ValueError(f"Expected the saved 768x320 three-panel video: {video}")
            stream = writer.add_stream("libx264", rate=source_stream.average_rate)
            stream.width, stream.height, stream.pix_fmt = 768, 320, "yuv420p"
            frame_count = 0
            for frame_index, frame in enumerate(reader.decode(video=0)):
                rgb = frame.to_ndarray(format="rgb24")
                # The existing evaluator renders reset first, then after each action.
                step = frame_index - 1
                index = int(np.searchsorted(input_steps, step, side="right")) - 1
                patch_map = maps[index] if index >= 0 else None
                prediction_step = int(input_steps[index]) if index >= 0 else -1
                result = render_attention_frame(
                    rgb[24:280, :256],
                    rgb[24:280, 256:512],
                    patch_map,
                    config,
                    font,
                    metadata["resolved_layer_zero_based"],
                    step,
                    prediction_step,
                )
                for packet in stream.encode(av.VideoFrame.from_ndarray(result, format="rgb24")):
                    writer.mux(packet)
                frame_count += 1
            for packet in stream.encode():
                writer.mux(packet)
        if not frame_count:
            raise ValueError(f"Empty input video: {video}")
        os.replace(temporary, target)
        shutil.copyfile(data_path, target_data)
        metadata.update(
            alpha=alpha,
            vmax=config.vmax,
            font_path=font_path,
            font_family=font.getname()[0],
            resolved_font_path=str(font.path),
            colormap="blue_purple_red_yellow",
            restyle=signature,
            restyle_note="Source pixels are decoded from the original video; original patch probabilities are unchanged.",
        )
        target_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        temporary.unlink(missing_ok=True)
    return f"{frame_count} frames"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-path", default=os.environ.get("ATTENTION_FONT_PATH"))
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--vmax", type=float, help="Omit to retain each video's original probability scale")
    args = parser.parse_args()
    source, destination = args.input_dir.resolve(), args.output_dir.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("Input and output must be separate directories, not nested")
    load_attention_font(args.font_path)
    videos = sorted(source.rglob("*_attention.mp4"))
    if not videos:
        parser.error("No *_attention.mp4 videos found under --input-dir")
    for index, video in enumerate(videos, 1):
        target = destination / video.relative_to(source)
        status = replot_video(video, target, font_path=args.font_path, alpha=args.alpha, vmax=args.vmax)
        print(f"[{index}/{len(videos)}] {target}: {status}", flush=True)


if __name__ == "__main__":
    main()
