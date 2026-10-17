"""Replot saved NTK results above training loss, without loading a model or using a GPU."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from lerobot.scripts.plot_pi05_ntk_stages import COLORS, LABELS, METRICS, STAGE_LABELS

NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
LOG_METRIC = re.compile(rf"\bstep:\s*({NUMBER}[KMBTQ]?)\s+smpl:.*?\bloss:\s*({NUMBER})\b")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PHASE_COLORS = {"priming": "#557A95", "bridge": "#A67D3D", "flow": "#537F65", "transition": "#777777"}
PHASE_LABELS = {
    "priming": "Priming",
    "bridge": "Bridge",
    "flow": "Flow adaptation",
    "transition": "Mixed-objective log window",
}


def _display_step(step):
    for suffix in ("", "K", "M", "B", "T", "Q"):
        if abs(step) < 1000:
            return f"{step:.0f}{suffix}"
        step /= 1000
    raise ValueError("Training step is too large")


def _phase(start, end, priming, pretrain):
    if end <= priming:
        return "priming"
    if start >= pretrain:
        return "flow"
    if start >= priming and end <= pretrain:
        return "bridge"
    return "transition"


def load_training_log(path, priming=750, pretrain=1000, log_freq=None):
    """Recover exact local steps despite the trainer's rounded '1K'/'2K' display.

    A complete fresh integrated-training log and its fixed logging interval are required.
    Window means that straddle the objective switch are kept as separate points.
    """
    text = ANSI.sub("", Path(path).read_text(errors="replace"))
    start_marker = "Starting integrated PI0.5 Stage 1"
    flow_marker = "PI0.5 Stage 1 complete; rebuilding"
    if start_marker not in text:
        raise ValueError(
            "Need the complete integrated-training .log, including its Stage 1 start marker; or use --loss-csv"
        )
    frequencies = {int(value) for value in re.findall(r"['\"]log_freq['\"]\s*:\s*(\d+)", text)}
    if log_freq is None:
        if len(frequencies) != 1:
            raise ValueError("Cannot identify one logging interval; supply the actual --log-freq")
        log_freq = frequencies.pop()
    elif frequencies and frequencies != {log_freq}:
        raise ValueError("--log-freq disagrees with the saved training log")
    if log_freq <= 0:
        raise ValueError("Logging interval must be positive")
    phase, local_step, starts, records = None, 0, 0, []
    for line in text.splitlines():
        if start_marker in line:
            if records:
                raise ValueError("Multiple training attempts in one log; use one complete run per input")
            phase, local_step = "pretrain", 0
        if flow_marker in line and phase != "flow":
            phase, local_step, starts = "flow", 0, 0
        if "Start offline training on a fixed dataset" in line:
            starts += 1
            if starts > 1:
                raise ValueError(
                    "Unexpected training restart; use a complete fresh-run log or an exact-step CSV"
                )
        match = LOG_METRIC.search(line)
        if not match:
            continue
        if phase is None:
            raise ValueError("Loss record has no identifiable training stage")
        token, loss = match.groups()
        if not np.isfinite(float(loss)):
            raise ValueError("Training log contains a non-finite loss")
        expected = local_step + log_freq
        if token != _display_step(expected):
            raise ValueError(
                f"Non-contiguous {phase} log: expected step {expected} ({_display_step(expected)}), got {token}; exact steps cannot be recovered"
            )
        if phase == "pretrain" and expected > pretrain:
            raise ValueError("Missing transition from pretraining to formal flow")
        offset = pretrain if phase == "flow" else 0
        end = expected + offset
        records.append(
            {"step": end, "loss": float(loss), "phase": _phase(local_step + offset, end, priming, pretrain)}
        )
        local_step = expected
    if not records:
        raise ValueError(f"No training loss found in {path}")
    return records


def load_loss_csv(path, step_column=None, loss_column=None, step_offset=0, priming=750, pretrain=1000):
    """Read exact steps from a normalized CSV or a single-run W&B CSV export."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if step_column is None:
            step_column = next(
                (key for key in ("cumulative_step", "train/steps", "Step", "_step", "step") if key in fields),
                None,
            )
        if loss_column is None:
            candidates = [
                key for key in fields if key in ("train/loss", "loss") or key.endswith(" - train/loss")
            ]
            if len(candidates) != 1:
                raise ValueError(f"Select one run with --loss-column; available columns: {fields}")
            loss_column = candidates[0]
        if step_column not in fields or loss_column not in fields:
            raise ValueError(f"Unknown CSV columns; available columns: {fields}")
        if step_column == "cumulative_step" and step_offset:
            raise ValueError("cumulative_step is already global; use --step-offset=0")
        points = {}
        for row in reader:
            if not row[step_column] or not row[loss_column]:
                continue
            step, loss = float(row[step_column]) + step_offset, float(row[loss_column])
            if not np.isfinite(step) or step < 0 or step != int(step) or not np.isfinite(loss):
                raise ValueError("CSV must contain finite losses and exact nonnegative integer steps")
            if step in points:
                raise ValueError(f"Duplicate step {step} in CSV; select a single run")
            points[int(step)] = (loss, row.get("phase"))
    if not points:
        raise ValueError(f"No finite loss records in {path}")
    records, previous = [], step_offset
    for step, (loss, phase) in sorted(points.items()):
        phase = phase or _phase(previous, step, priming, pretrain)
        if phase not in PHASE_COLORS:
            raise ValueError(f"Unknown loss phase: {phase}")
        records.append({"step": step, "loss": loss, "phase": phase})
        previous = step
    return records


def load_ntk(path, scope):
    document = json.loads(Path(path).read_text())
    stages, seeds = document["manifest"]["stages"], document["manifest"]["seeds"]
    if [stage["name"] for stage in stages] != list(STAGE_LABELS):
        raise ValueError("This composite figure requires all four completed NTK stages")
    steps = [stage["total_step"] for stage in stages]
    if steps[:3] != [0, 750, 1000] or steps[3] <= 1000:
        raise ValueError("Expected cumulative NTK steps 0, 750, 1000, and a later final checkpoint")
    records = document["records"]
    lookup = {(record["stage"], record["seed"]): record for record in records}
    expected = {(stage["name"], seed) for stage in stages for seed in seeds}
    if not seeds or len(seeds) != len(set(seeds)) or len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError("NTK results contain incomplete or duplicate paired seed records")
    scopes = ("backbone", "prompts") if scope == "both" else (scope,)
    for record in records:
        for current in scopes:
            for module in COLORS:
                group = record["groups"].get(f"{current}/{module}")
                if group is None:
                    raise ValueError(f"Missing {current}/{module}; select --scope matching the saved results")
                for metric in METRICS:
                    if not np.isfinite(group[metric]) or group[metric] < 0:
                        raise ValueError(f"Invalid saved NTK metric: {metric}")
    return stages, seeds, lookup, scopes


def _smooth(values, window):
    # Trailing moving average; called separately for each objective segment.
    return np.asarray(
        [np.mean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]
    )


def make_figure(stages, seeds, lookup, scope, losses, smooth_window=1):
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    fig = plt.figure(figsize=(17, 9))
    grid = fig.add_gridspec(
        2, 4, height_ratios=(1, 1.15), left=0.09, right=0.975, bottom=0.11, top=0.9, hspace=0.65, wspace=0.12
    )
    axes = []
    for index in range(4):
        axes.append(
            fig.add_subplot(
                grid[0, index], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None
            )
        )
    energy_values = []
    for index, (stage, axis) in enumerate(zip(stages, axes, strict=True)):
        for module, color in COLORS.items():
            group = f"{scope}/{module}"
            x = np.asarray(
                [lookup[(stage["name"], seed)]["groups"][group]["effective_rank"] for seed in seeds]
            )
            y = np.asarray(
                [
                    lookup[(stage["name"], seed)]["groups"][group]["parameter_normalized_energy"]
                    for seed in seeds
                ]
            )
            energy_values.extend(y)
            axis.scatter(x, y, color=color, alpha=0.4, s=23, edgecolors="none")
            xq, yq = np.quantile(x, [0.25, 0.5, 0.75]), np.quantile(y, [0.25, 0.5, 0.75])
            axis.errorbar(
                xq[1],
                yq[1],
                xerr=[[xq[1] - xq[0]], [xq[2] - xq[1]]],
                yerr=[[yq[1] - yq[0]], [yq[2] - yq[1]]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=6,
                label=LABELS[module],
            )
        axis.set_title(f"{STAGE_LABELS[stage['name']]}\nStep {stage['total_step']}", fontsize=11, pad=10)
        axis.set_xlabel(METRICS["effective_rank"], fontsize=10)
        axis.tick_params(labelleft=index == 0)
        axis.grid(alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    if np.all(np.asarray(energy_values) > 0):
        axes[0].set_yscale("log")
    axes[0].set_ylabel("Parameter-normalized\ntangent energy", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    final_step = stages[-1]["total_step"]
    loss_axis = fig.add_subplot(grid[1, :])
    for name, begin, end in (("priming", 0, 750), ("bridge", 750, 1000), ("flow", 1000, final_step)):
        loss_axis.axvspan(begin, end, color=PHASE_COLORS[name], alpha=0.045, linewidth=0)
    # Never draw a connecting line or smooth through an objective switch.
    for phase, color in PHASE_COLORS.items():
        selected = [row for row in losses if row["phase"] == phase]
        if not selected:
            continue
        x, y = np.asarray([row["step"] for row in selected]), np.asarray([row["loss"] for row in selected])
        if phase == "transition":
            loss_axis.scatter(x, y, marker="x", color=color, s=30, label=PHASE_LABELS[phase], zorder=4)
        else:
            if smooth_window > 1:
                loss_axis.plot(x, y, color=color, alpha=0.25, linewidth=0.8)
            loss_axis.plot(
                x,
                _smooth(y, smooth_window),
                "o-",
                color=color,
                markersize=3,
                linewidth=1.5,
                label=PHASE_LABELS[phase],
            )
    loss_axis.set(
        xlabel="Cumulative training steps",
        ylabel="Training loss (logged mean)",
        xlim=(-final_step * 0.015, final_step * 1.015),
    )
    loss_axis.margins(y=0.18)
    ticks = sorted(set([0, 750, 1000, final_step] + list(range(2000, final_step, 1000))))
    loss_axis.set_xticks(ticks)
    loss_axis.grid(alpha=0.16)
    loss_axis.spines[["top", "right"]].set_visible(False)
    loss_axis.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    for stage, axis in zip(stages, axes, strict=True):
        step = stage["total_step"]
        loss_axis.axvline(step, color="#7D8790", linewidth=0.85, linestyle="--", alpha=0.65)
        arrow = ConnectionPatch(
            xyA=(0.5, -0.24),
            coordsA=axis.transAxes,
            xyB=(step, 1.02),
            coordsB=loss_axis.get_xaxis_transform(),
            arrowstyle="-|>",
            color="#616A73",
            linewidth=1.05,
            mutation_scale=12,
            clip_on=False,
        )
        fig.add_artist(arrow)
    title = "Backbone" if scope == "backbone" else "Prompts"
    fig.suptitle(f"{title}: NTK sensitivity across training", fontsize=15, y=0.985)
    note = "NTK: paired-seed points; median and IQR. Arrows mark checkpoint steps; unlogged losses are not interpolated."
    if smooth_window > 1:
        note += f" Loss smoothing: {smooth_window} logged points, within each objective."
    fig.text(0.09, 0.025, note, fontsize=8, color="#555555")
    return fig


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Existing ntk_stages/results.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--training-log", type=Path, help="Complete launcher .log containing pretrain and flow"
    )
    source.add_argument(
        "--loss-csv", type=Path, help="CSV exported from one W&B run, or cumulative_step/loss CSV"
    )
    parser.add_argument("--log-freq", type=int, help="Actual logging interval if absent from the .log config")
    parser.add_argument("--step-column", help="CSV x column; prefers train/steps over W&B internal _step")
    parser.add_argument("--loss-column", help="CSV loss column; mandatory if multiple runs were exported")
    parser.add_argument(
        "--step-offset", type=int, default=0, help="Add 1000 for a flow-only W&B CSV; 0 for cumulative steps"
    )
    parser.add_argument("--scope", choices=("backbone", "prompts", "both"), default="both")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Trailing mean over logged points, separately within each objective",
    )
    parser.add_argument("--output-dir", type=Path, help="Defaults to results.json's directory / with_loss")
    return parser


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = make_parser().parse_args()
    if args.smooth_window < 1 or args.step_offset < 0:
        raise ValueError("Smoothing window must be positive and step offset nonnegative")
    stages, seeds, lookup, scopes = load_ntk(args.results, args.scope)
    if args.training_log:
        if args.step_offset or args.step_column or args.loss_column:
            raise ValueError("CSV column/offset options cannot be combined with --training-log")
        losses = load_training_log(args.training_log, log_freq=args.log_freq)
    else:
        losses = load_loss_csv(args.loss_csv, args.step_column, args.loss_column, args.step_offset)
    if any(row["step"] > stages[-1]["total_step"] for row in losses):
        raise ValueError(
            "Loss extends past the final NTK checkpoint; select the matching run or trim the CSV explicitly"
        )
    if losses[-1]["step"] < stages[-1]["total_step"]:
        print(
            f"Loss logging ends at {losses[-1]['step']}; final checkpoint is {stages[-1]['total_step']}. No loss is extrapolated."
        )
    if min(row["step"] for row in losses) >= 1000:
        print("No recorded loss before cumulative step 1000; that part of the timeline will remain empty.")
    output = args.output_dir or args.results.parent / "with_loss"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "loss_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("cumulative_step", "loss", "phase"))
        writer.writeheader()
        writer.writerows(
            {"cumulative_step": row["step"], "loss": row["loss"], "phase": row["phase"]} for row in losses
        )
    with plt.rc_context(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}
    ):
        for scope in scopes:
            fig = make_figure(stages, seeds, lookup, scope, losses, args.smooth_window)
            for suffix in ("png", "pdf", "svg"):
                path = output / f"{scope}_ntk_with_loss.{suffix}"
                fig.savefig(path, dpi=300, bbox_inches="tight")
                print(f"Saved {path}")
            plt.close(fig)
    source = args.training_log or args.loss_csv
    metadata = {
        "ntk_results": str(args.results.resolve()),
        "ntk_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "loss_source": str(source.resolve()),
        "loss_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cumulative_checkpoint_steps": [stage["total_step"] for stage in stages],
        "loss_points": len(losses),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
    }
    (output / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
