"""Publication figures for completed stages of the PI05 NTK protocol (no GPU required)."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

COLORS = {"vlm": "#3265A8", "action": "#D96936"}
LABELS = {"vlm": "VLM", "action": "Action expert"}
METRICS = {
    "effective_rank": "Spectral effective rank",
    "parameter_normalized_energy": "Parameter-normalized tangent energy",
}
STAGE_LABELS = {
    "before": "Before training",
    "priming": "After priming",
    "stage2": "After stage 2",
    "final": "After adaptation",
}


def plot_results(results_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    results_path = Path(results_path)
    document = json.loads(results_path.read_text())
    records = document["records"]
    stages = document["manifest"]["stages"]
    seeds = document["manifest"]["seeds"]
    out = results_path.parent
    lookup = {(row["stage"], row["seed"]): row for row in records}
    names = [stage["name"] for stage in stages]
    if not names or names != list(STAGE_LABELS)[: len(names)] or not seeds:
        raise ValueError("Expected nonempty consecutive stages starting before training and paired seeds")
    expected = {(name, seed) for name in names for seed in seeds}
    if len(lookup) != len(records) or set(lookup) != expected or len(seeds) != len(set(seeds)):
        raise ValueError("Cannot draw stage trends from incomplete or duplicate seed records")
    groups = list(records[0]["groups"])
    if any(set(row["groups"]) != set(groups) for row in records):
        raise ValueError("Missing parameter group in a stage/seed")
    table = []
    for row in records:
        for group, values in row["groups"].items():
            table.append(
                {
                    "stage": row["stage"],
                    "total_step": row["total_step"],
                    "local_step": row["local_step"],
                    "seed": row["seed"],
                    "group": group,
                    **{
                        key: values[key]
                        for key in (
                            "parameter_count",
                            "effective_rank",
                            "participation_rank",
                            "parameter_normalized_energy",
                            "tangent_trace",
                            "sketch_trace_relative_error",
                        )
                    },
                }
            )
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    def values(scope, module, metric):
        return np.asarray(
            [
                [lookup[(stage["name"], seed)]["groups"][f"{scope}/{module}"][metric] for seed in seeds]
                for stage in stages
            ]
        )

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def style(axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.set_axisbelow(True)

    def energy_scale(axis, energies):
        if np.all(np.asarray(energies) > 0):
            axis.set_yscale("log")

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.titleweight": "semibold",
            "lines.linewidth": 1.8,
        }
    ):
        for scope in dict.fromkeys(group.split("/")[0] for group in groups):
            ranks = {module: values(scope, module, "effective_rank") for module in COLORS}
            energies = {module: values(scope, module, "parameter_normalized_energy") for module in COLORS}
            title = "Backbone sensitivity" if scope == "backbone" else "Prompt sensitivity"
            notes = "Points: paired probe seeds; bars/bands: median and IQR (not training-run uncertainty)."
            rows, columns = (2, 2) if len(stages) == 4 else (1, len(stages))
            fig, axes = plt.subplots(
                rows,
                columns,
                figsize=(5 * columns, 3.5 * rows + (1 if rows == 1 else 0)),
                sharex=True,
                sharey=True,
                squeeze=False,
                layout="constrained",
            )
            for index, (stage, axis) in enumerate(zip(stages, axes.flat, strict=True)):
                for module, color in COLORS.items():
                    x, y = ranks[module][index], energies[module][index]
                    axis.scatter(x, y, c=color, alpha=0.40, s=24, edgecolors="none")
                    xq, yq = np.quantile(x, [0.25, 0.5, 0.75]), np.quantile(y, [0.25, 0.5, 0.75])
                    axis.errorbar(
                        xq[1],
                        yq[1],
                        xerr=[[xq[1] - xq[0]], [xq[2] - xq[1]]],
                        yerr=[[yq[1] - yq[0]], [yq[2] - yq[1]]],
                        fmt="o",
                        color=color,
                        capsize=3,
                        markersize=7,
                        label=LABELS[module],
                    )
                axis.set_title(f"{STAGE_LABELS[stage['name']]} · step {stage['total_step']}")
                style(axis)
            energy_scale(axes[0, 0], list(energies.values()))
            axes[0, 0].legend(frameon=False)
            fig.supxlabel(METRICS["effective_rank"])
            fig.supylabel(METRICS["parameter_normalized_energy"])
            fig.suptitle(title + "\n" + notes, fontsize=11)
            save(fig, f"{scope}_stages")

            # Paired trajectories preserve the user's original x/y axes.
            fig, axis = plt.subplots(figsize=(7, 5), layout="constrained")
            markers = ("o", "s", "^", "D")[: len(stages)]
            for module, color in COLORS.items():
                x, y = np.median(ranks[module], axis=1), np.median(energies[module], axis=1)
                axis.plot(x, y, color=color, label=LABELS[module])
                for index, _stage in enumerate(stages):
                    axis.scatter(x[index], y[index], marker=markers[index], color=color, s=65, zorder=3)
                    if index:
                        axis.annotate(
                            "",
                            xy=(x[index], y[index]),
                            xytext=(x[index - 1], y[index - 1]),
                            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
                        )
            energy_scale(axis, list(energies.values()))
            style(axis)
            axis.set(
                xlabel=METRICS["effective_rank"],
                ylabel=METRICS["parameter_normalized_energy"],
                title=title + " across training (seed medians)",
            )
            handles, labels = axis.get_legend_handles_labels()
            handles += [
                Line2D([], [], color="#555555", marker=marker, linestyle="none") for marker in markers
            ]
            labels += [f"Step {stage['total_step']}" for stage in stages]
            axis.legend(handles, labels, frameon=False, fontsize=9, loc="best")
            save(fig, f"{scope}_trajectory")

            for metric, metric_label in METRICS.items():
                fig, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
                steps = [stage["total_step"] for stage in stages]
                for module, color in COLORS.items():
                    data = values(scope, module, metric)
                    q = np.quantile(data, [0.25, 0.5, 0.75], axis=1)
                    axis.plot(steps, data, color=color, alpha=0.10, linewidth=0.8)
                    axis.fill_between(steps, q[0], q[2], color=color, alpha=0.15)
                    axis.plot(steps, q[1], "o-", color=color, label=LABELS[module])
                if metric == "parameter_normalized_energy":
                    energy_scale(axis, list(energies.values()))
                # Keep 750 and 1000 readable at their actual (unequally spaced) positions.
                # Full stage names are supplied by the companion stage-panel figure.
                axis.set_xticks(steps)
                axis.set(xlabel="Cumulative optimizer updates", ylabel=metric_label, title=title)
                axis.tick_params(axis="x", labelsize=8)
                axis.legend(frameon=False)
                style(axis)
                save(fig, f"{scope}_{'effective_rank' if metric == 'effective_rank' else 'tangent_energy'}")
    print(f"Saved metrics.csv and PNG/PDF figures to {out}")
    # Original scalar-only exports remain usable. Current analysis saves all
    # kernels, so also provide stage similarity whenever comparisons are possible.
    if len(stages) >= 2 and all("kernel" in group for row in records for group in row["groups"].values()):
        from lerobot.scripts.plot_pi05_ntk_similarity import plot_similarity

        plot_similarity(results_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    plot_results(parser.parse_args().results)


if __name__ == "__main__":
    main()
