"""CPU-only constant-action probe for PI0.5's learned action-input time path.

All 50 input actions are *exactly identical*. This isolates the temporal
residual at the action expert input: no robot, dataset, Flow noise, optimizer,
or full VLA is needed. RoPE itself acts later on attention Q/K, so a flat
RoPE-only *input* does not imply a position-blind RoPE transformer.

Use --checkpoint for measurements from trained weights; --illustrative uses
deterministic, untrained weights for a clearly labelled schematic only.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.pi05.action_time import RelativeActionTimeEncoding


PROTOCOL = "pi05-task7-constant-action-50-v1"
HORIZON = 50
DEFAULT_ACTION = "0.10,-0.10,0.05,0.0,0.0,0.0,0.5"
DEFAULT_OUTPUT = Path(
    "/data1/wyn/chkpt/2601-lerobot/PromptVLA/analysis/task7-rope-time-input"
)


def parse_action(value: str) -> np.ndarray:
    """Seven normalized example action coordinates, not physical commands."""
    try:
        parts = [float(x.strip()) for x in value.split(",")]
    except ValueError as exc:
        raise ValueError("--action must contain seven comma-separated numbers") from exc
    if len(parts) != 7 or not np.isfinite(parts).all():
        raise ValueError("--action must contain exactly seven finite numbers")
    return np.asarray(parts, dtype=np.float32)


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def load_checkpoint(checkpoint: Path):
    """Read only three small tensors, never instantiate the 3B-parameter VLA."""
    from safetensors import safe_open

    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError("--checkpoint must be the pretrained_model directory")
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("action_time_encoding"):
        raise ValueError("Checkpoint has no enabled action-time encoder")
    if config.get("action_time_injection", "input") != "input":
        raise ValueError("This probe requires action_time_injection=input")
    if config.get("chunk_size") != HORIZON:
        raise ValueError("Expected the trained 50-step PI0.5 action chunk")

    names = (
        "model.action_in_proj.weight",
        "model.action_in_proj.bias",
        "model.action_time_encoder.weight",
    )
    with safe_open(str(weights_path), framework="pt", device="cpu") as reader:
        missing = [name for name in names if name not in reader.keys()]
        if missing:
            raise ValueError(f"Checkpoint is missing trained tensors: {missing}")
        projection, bias, time_weight = (reader.get_tensor(name).float() for name in names)

    if projection.ndim != 2 or bias.shape != (projection.shape[0],):
        raise ValueError("Invalid action input projection dimensions")
    width, max_action_dim = projection.shape
    encoding_dim = int(config.get("action_time_encoding_dim", 32))
    if time_weight.shape != (width, encoding_dim):
        raise ValueError("Action-time weight dimensions do not match the checkpoint config")
    if max_action_dim != int(config.get("max_action_dim", max_action_dim)):
        raise ValueError("Action input dimension disagrees with max_action_dim")
    if max_action_dim < 7:
        raise ValueError("Action input dimension is less than the seven LIBERO coordinates")
    if not torch.isfinite(projection).all() or not torch.isfinite(bias).all():
        raise ValueError("Nonfinite action projection in checkpoint")
    if not torch.isfinite(time_weight).all():
        raise ValueError("Nonfinite time projection in checkpoint")

    encoder = RelativeActionTimeEncoding(
        width,
        encoding_dim,
        float(config.get("action_time_encoding_fps", 10.0)),
        min_frequency_hz=float(config.get("action_time_min_frequency_hz", 0.1)),
        max_frequency_hz=float(config.get("action_time_max_frequency_hz", 10.0)),
        gain_mode=config.get("action_time_gain_mode", "none"),
    )
    with torch.no_grad():
        encoder.weight.copy_(time_weight)
    encoder.eval()
    basis = config.get("action_time_basis", "index")
    if basis not in ("index", "physical", "global_dt"):
        raise ValueError(f"Unsupported action-time basis: {basis}")
    source = {
        "kind": "trained_checkpoint",
        "checkpoint": str(checkpoint),
        "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "time_projection_sha256": _tensor_sha256(time_weight),
        "action_projection_sha256": _tensor_sha256(projection),
        "model_safetensors_size": weights_path.stat().st_size,
    }
    return encoder, projection, bias, basis, source


def illustrative_weights():
    """A reproducible drawing aid; emphatically not trained checkpoint data."""
    generator = torch.Generator(device="cpu").manual_seed(20261104)
    width, padded_actions = 128, 32
    projection = torch.randn(width, padded_actions, generator=generator) * 0.07
    bias = torch.zeros(width)
    encoder = RelativeActionTimeEncoding(width, 32, 10.0)
    with torch.no_grad():
        encoder.weight.copy_(torch.randn(width, 32, generator=generator) * 0.025)
    return encoder, projection, bias, "index", {
        "kind": "illustrative_untrained",
        "warning": "Random deterministic projection: not evidence from a learned policy",
    }


@torch.inference_mode()
def constant_action_probe(
    encoder: RelativeActionTimeEncoding,
    projection: torch.Tensor,
    bias: torch.Tensor,
    action: np.ndarray,
    *,
    basis: str = "index",
    delta_t: float | None = None,
):
    """Use the actual PI0.5 F.linear -> action_time_encoder path, with no Flow noise.

    With the fixed action input, zero and reversed act only on the time
    residual. Reversal is exactly equivalent to reversing its feature positions.
    """
    if projection.ndim != 2 or bias.shape != (projection.shape[0],):
        raise ValueError("Projection and bias shape mismatch")
    if encoder.weight.shape[0] != projection.shape[0]:
        raise ValueError("Action and time projection widths differ")
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Expected finite normalized seven-dimensional action")
    if projection.shape[1] < 7:
        raise ValueError("Padded action input cannot hold seven dimensions")
    if delta_t is not None and (not math.isfinite(delta_t) or delta_t <= 0):
        raise ValueError("--delta-t must be positive and finite")
    if basis == "index" and delta_t is not None:
        raise ValueError("--delta-t only applies to physical/global_dt time bases")

    padded = torch.zeros(1, HORIZON, projection.shape[1], dtype=torch.float32)
    padded[..., :7] = torch.as_tensor(action).view(1, 1, 7)
    before = F.linear(padded, projection, bias)
    if not torch.equal(before, before[:, :1].expand_as(before)):
        raise AssertionError("Constructed action inputs must be exactly constant")
    dt = None
    if basis != "index":
        dt = torch.tensor([1.0 / encoder.fps if delta_t is None else delta_t])
    after = encoder(before, basis=basis, delta_t=dt)

    # The zero intervention is identity. The reversed intervention uses the
    # same trained weight matrix and the same 50 Fourier features, but with
    # their positions reversed (RoPE downstream would remain untouched).
    zero = before.clone()
    features = encoder.features(HORIZON, before.device, basis=basis, delta_t=dt)
    correction = F.linear(features.flip(-2).to(encoder.weight.dtype), encoder.weight)
    if correction.ndim == 2:
        correction = correction.unsqueeze(0)
    if encoder.gain_mode == "relative_rms":
        token_rms = before.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        gains = (token_rms / token_rms.mean(dim=1, keepdim=True).clamp_min(1e-4)).clamp(0.5, 2.0)
        correction = correction * gains
    reversed_time = before + correction.to(before.dtype)

    return {
        "action": padded[0, :, :7].numpy(),
        "rope_input": before[0].numpy(),
        "correct": after[0].numpy(),
        "zero": zero[0].numpy(),
        "reversed": reversed_time[0].numpy(),
        "residual": (after - before)[0].numpy(),
        "time_features": features.squeeze(0).numpy(),
        "time_seconds": (np.arange(HORIZON) / encoder.fps if basis == "index"
                         else (np.full(HORIZON, float(dt[0])) if basis == "global_dt"
                               else np.arange(HORIZON) * float(dt[0]))),
    }


def summarize(data: dict) -> dict:
    raw, before, correct = data["action"], data["rope_input"], data["correct"]
    residual = data["residual"]
    distances = np.linalg.norm(correct - correct[:1], axis=-1)
    adjacent = np.linalg.norm(correct[1:] - correct[:-1], axis=-1)
    input_rms = float(np.sqrt(np.mean(before.astype(np.float64) ** 2)))
    residual_rms = float(np.sqrt(np.mean(residual.astype(np.float64) ** 2)))
    # A scale-aware threshold prevents interpreting float32 roundoff as time signal.
    threshold = 1e-6 * max(1.0, input_rms)
    return {
        "all_50_actions_identical": bool(np.array_equal(raw, np.broadcast_to(raw[0], raw.shape))),
        "rope_only_inputs_identical": bool(
            np.array_equal(before, np.broadcast_to(before[0], before.shape))
        ),
        "zero_equals_rope_input": bool(np.array_equal(before, data["zero"])),
        "correct_max_l2_difference_from_first": float(distances.max()),
        "correct_mean_adjacent_l2_difference": float(adjacent.mean()),
        "correct_vs_zero_rms": residual_rms,
        "reversed_vs_correct_rms": float(
            np.sqrt(np.mean((data["reversed"] - correct).astype(np.float64) ** 2))
        ),
        "time_residual_to_input_rms": residual_rms / max(input_rms, 1e-12),
        "position_signal_detected_at_input": bool(distances.max() > threshold),
        "detection_threshold_l2": threshold,
    }


def save_figure(data: dict, out: Path, *, source_kind: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(HORIZON)
    residual = data["residual"].astype(np.float64)
    centered = residual - residual.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T
    zero_xy = -residual.mean(axis=0) @ vt[:2].T
    norm = np.linalg.norm(residual, axis=-1)
    denom = norm[:, None] * norm[None, :]
    similarity = np.divide(residual @ residual.T, denom,
                           out=np.zeros((HORIZON, HORIZON)), where=denom > 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), layout="constrained")
    ax = axes[0, 0]
    im = ax.imshow(data["action"], aspect="auto", cmap="coolwarm")
    ax.set(title="A. The same 7-D action repeated 50 times", xlabel="Action dimension",
           ylabel="Position in action chunk")
    ax.set_xticks(range(7), [f"a{i}" for i in range(7)])
    fig.colorbar(im, ax=ax, shrink=0.82, label="Normalized action value")

    ax = axes[0, 1]
    ax.plot(xy[:, 0], xy[:, 1], color="#64748b", alpha=0.55, lw=1)
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=steps, cmap="viridis", s=28, zorder=3)
    ax.scatter([zero_xy[0]], [zero_xy[1]], c="#d33f49", marker="x", s=100,
               linewidths=2.2, label="RoPE-only / zero input")
    ax.annotate("t=0", xy[0], xytext=(5, 7), textcoords="offset points")
    ax.annotate("t=49", xy[-1], xytext=(5, 7), textcoords="offset points")
    ax.set(title="B. Input-embedding PCA (colored by position)",
           xlabel="Time-residual PC1", ylabel="Time-residual PC2")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(scatter, ax=ax, shrink=0.82, label="Action index")
    if singular[0] < 1e-10:
        ax.text(0.5, 0.06, "No position-varying residual", transform=ax.transAxes,
                ha="center", color="#b91c1c")

    ax = axes[1, 0]
    img = ax.imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.set(title="C. Cosine similarity of time-residual vectors",
           xlabel="Action index", ylabel="Action index")
    fig.colorbar(img, ax=ax, shrink=0.82, label="Cosine similarity")

    ax = axes[1, 1]
    for key, label, style in (
        ("rope_input", "RoPE-only input (= Zero)", "--"),
        ("correct", "Time-Input: correct order", "-"),
        ("reversed", "Time-Input: reversed order", ":"),
    ):
        embeddings = data[key]
        distances = np.linalg.norm(embeddings - embeddings[:1], axis=-1)
        ax.plot(steps, distances, style, label=label, lw=2)
    ax.set(title="D. Change from the first token at expert input",
           xlabel="Action position (0–49)", ylabel="L2 distance to position 0")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    caption = ("TRAINED checkpoint" if source_kind == "trained_checkpoint" else
               "ILLUSTRATIVE only — untrained deterministic weights")
    fig.suptitle(f"Constant-action control (H=50) | {caption}", fontsize=14)
    fig.text(0.5, 0.009,
             "The RoPE-only curve refers ONLY to expert INPUT embeddings; "
             "RoPE still encodes position later in Q/K attention. No action prediction is measured.",
             ha="center", va="bottom", fontsize=9)
    paths = []
    for suffix in ("png", "svg"):
        path = out / f"fig00_constant_action_50.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def save_results(out: Path, data: dict, summary: dict, source: dict, encoder, basis: str):
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "constant_action_50_embeddings.npz"
    np.savez_compressed(npz, **data)
    csv = out / "constant_action_50_positions.csv"
    columns = np.column_stack([
        np.arange(HORIZON), data["time_seconds"],
        np.linalg.norm(data["residual"], axis=-1),
        np.linalg.norm(data["correct"] - data["correct"][:1], axis=-1),
        np.linalg.norm(data["reversed"] - data["reversed"][:1], axis=-1),
    ])
    np.savetxt(csv, columns, delimiter=",", comments="", fmt="%.9g",
               header="position,time_seconds,residual_l2,correct_from_first_l2,reversed_from_first_l2")
    figures = save_figure(data, out, source_kind=source["kind"])
    report = {
        "protocol": PROTOCOL,
        "source": source,
        "action_vector_7d": data["action"][0].tolist(),
        "horizon": HORIZON,
        "fps": encoder.fps,
        "time_basis": basis,
        "time_encoding_dim": encoder.encoding_dim,
        "min_frequency_hz": encoder.min_frequency_hz,
        "max_frequency_hz": encoder.max_frequency_hz,
        "gain_mode": encoder.gain_mode,
        "metrics": summary,
        "figures": figures,
        "data_files": [str(csv), str(npz)],
        "interpretation_limit": (
            "RoPE-only has identical ACTION INPUTS, but still has positional "
            "Q/K rotations downstream. This probe checks time-path injection, "
            "not improved action prediction or task success. Use the held-out "
            "Correct/Zero/Reversed Flow probe for prediction sensitivity."
        ),
    }
    report_path = out / "constant_action_50_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path,
                       help="Trained Time-Input checkpoint's pretrained_model directory")
    group.add_argument("--from-task7-plan", action="store_true",
                       help="Locate the trained time_input checkpoint using the existing task7 plan")
    group.add_argument("--illustrative", action="store_true",
                       help="Make a labelled drawing with deterministic UNTRAINED weights")
    parser.add_argument("--directory", type=Path,
                        default=Path("/data1/wyn/prompt-work/gripper-event-task7-1200/base"),
                        help="Training plan for --from-task7-plan")
    parser.add_argument("--output-root", type=Path,
                        default=Path("/data1/wyn/chkpt/2601-lerobot/PromptVLA"),
                        help="Training run root for --from-task7-plan")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Timepath experiment repeat number for --from-task7-plan")
    parser.add_argument("--action", default=DEFAULT_ACTION,
                        help="Seven normalized, comma-separated values repeated 50 times")
    parser.add_argument("--delta-t", type=float, default=None,
                        help="Actual interval in seconds; only for physical/global_dt checkpoints")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        action = parse_action(args.action)
        if args.illustrative:
            encoder, projection, bias, basis, source = illustrative_weights()
        else:
            checkpoint = args.checkpoint
            if args.from_task7_plan:
                from lerobot.scripts import pi05_task7_timepath_experiment as timepath

                record = timepath.check_plan(args.directory)
                checkpoint = timepath.build_jobs(
                    record, args.directory, args.output_root,
                    selection="time_input", repeat=args.repeat,
                )[0]["checkpoint"]
            encoder, projection, bias, basis, source = load_checkpoint(checkpoint)
        data = constant_action_probe(encoder, projection, bias, action, basis=basis,
                                     delta_t=args.delta_t)
        summary = summarize(data)
        output = args.output or (DEFAULT_OUTPUT / "illustrative" if args.illustrative
                                 else DEFAULT_OUTPUT)
        report_path = save_results(output.expanduser().resolve(), data, summary,
                                   source, encoder, basis)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Constant-action probe failed: {exc}\n")
    print(f"Saved {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
