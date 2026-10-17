"""Replot prompt-rank histories, spectra and optional LIBERO/alignment results on CPU."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from lerobot.utils.prompt_rank_analysis import (
    correlations,
    metric_history,
    read_json,
    read_success,
    resolve_runs,
    snapshot,
    write_csv,
    write_json,
)

GEOMETRY = ("jacobian_erank", "ntk_erank", "rank95_energy", "tangent_trace")
LABELS = {
    "jacobian_erank": "Jacobian effective rank",
    "ntk_erank": "NTK effective rank",
    "rank95_energy": "Components for 95% energy",
    "tangent_trace": "Tangent energy (trace)",
    "top16_alignment": "Residual overlap with top 16 directions",
    "heldout_linear_reduction": "Held-out linear error reduction",
}


def make_report(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if not args.steps or len(set(args.steps)) != len(args.steps) or min(args.steps) < 0:
        raise ValueError("Expected distinct nonnegative spectrum steps")
    if args.font_path:
        font_manager.fontManager.addfont(str(args.font_path))
        font = font_manager.FontProperties(fname=args.font_path).get_name()
    else:
        font = "DejaVu Serif"
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    runs = resolve_runs(args.ckpt_root, args.tokens, args.seed, args.run_prefix, args.source)
    histories, spectra, final, alignment, spectrum_rows = [], {}, [], [], []
    eval_hash = None
    alignment_protocol = None
    for run in runs:
        history = metric_history(run)
        histories.append(history)
        lookup = {row["step"]: row for row in history}
        if args.final_step not in lookup or 0 not in lookup:
            raise ValueError(f"Missing initial/final metrics: {run['name']}")
        row = {"run": run["name"], "tokens": run["tokens"], "seed": run["seed"], **lookup[args.final_step]}
        for key in GEOMETRY:
            row[f"initial_{key}"] = lookup[0][key]
            row[f"delta_{key}"] = row[key] - lookup[0][key]
        for step in args.steps:
            saved = snapshot(run, step)
            lam, sigma = saved["eigenvalues"], saved["singular_values"]
            if (
                np.any(lam < 0)
                or np.any(np.diff(lam) > 0)
                or not np.allclose(sigma**2, lam, rtol=1e-8, atol=1e-15)
            ):
                raise ValueError(f"Invalid saved spectrum: {run['name']} step={step}")
            sigma_probability = sigma / sigma.sum() if sigma.sum() else np.zeros_like(sigma)
            energy = np.cumsum(lam) / lam.sum() if lam.sum() else np.zeros_like(lam)
            spectra[run["tokens"], step] = (sigma_probability, energy)
            for k, (value, cumulative) in enumerate(zip(sigma_probability, energy, strict=True), 1):
                spectrum_rows.append(
                    {
                        "tokens": run["tokens"],
                        "seed": run["seed"],
                        "step": step,
                        "component": k,
                        "singular_probability": float(value),
                        "cumulative_squared_energy": float(cumulative),
                    }
                )
        if args.eval_root:
            success, evaluator = read_success(
                args.eval_root, run, args.final_step, args.episodes_per_task, args.eval_seed
            )
            if eval_hash is not None and evaluator != eval_hash:
                raise ValueError("Evaluation code differs across selected runs")
            eval_hash = evaluator
            row.update(success)
        if args.include_alignment:
            directory = run["path"] / args.analysis_name
            protocol = read_json(directory / "protocol.json")
            manifest_path = run["path"] / "prompt_rank/manifest.json"
            if (
                protocol["run"] != run["name"]
                or protocol["probe_manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            ):
                raise ValueError("Alignment protocol does not match the recorded probes")
            shared = {
                key: protocol[key]
                for key in (
                    "schema_version",
                    "torch_version",
                    "device_type",
                    "heldout_probes",
                    "heldout_seed",
                    "ridge_relative",
                    "gram_rtol",
                    "scope",
                    "noise_projection_sha256",
                )
            }
            if alignment_protocol is not None and shared != alignment_protocol:
                raise ValueError("Alignment settings differ across selected runs")
            alignment_protocol = shared
            for step in sorted(set(args.steps) | {args.final_step}):
                scores = read_json(directory / f"step_{step:06d}.json")
                source = run["path"] / f"prompt_rank/step_{step:06d}.npz"
                if (
                    scores["run"] != run["name"]
                    or scores["step"] != step
                    or scores["snapshot_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
                ):
                    raise ValueError("Alignment does not match this prompt snapshot")
                alignment.append(scores)
                if step == args.final_step:
                    row.update(
                        {
                            key: value
                            for key, value in scores.items()
                            if key
                            not in {
                                "top_k",
                                "step",
                                "tokens",
                                "seed",
                                "run",
                                "rank90_energy",
                                "rank95_energy",
                            }
                        }
                    )
        final.append(row)

    # All input validation happens before creating or replacing report outputs.
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "final_summary.csv", final)
    write_csv(output / "spectra.csv", spectrum_rows)
    write_csv(
        output / "history.csv",
        [
            dict(run=run["name"], tokens=run["tokens"], seed=run["seed"], **row)
            for run, history in zip(runs, histories, strict=True)
            for row in history
        ],
    )
    if alignment:
        write_csv(
            output / "alignment_summary.csv",
            [{key: value for key, value in row.items() if key != "top_k"} for row in alignment],
        )
    metrics = list(GEOMETRY) + (["top16_alignment", "heldout_linear_reduction"] if alignment else [])
    corr = correlations(final, metrics) if args.eval_root else []
    write_csv(output / "correlations.csv", corr)
    write_json(
        output / "report.json",
        {
            "runs": [run["name"] for run in runs],
            "final_step": args.final_step,
            "spectrum_steps": args.steps,
            "probe_manifest": runs[0]["manifest"],
            "evaluator_sha256": eval_hash,
            "alignment_protocol": alignment_protocol,
            "final": final,
            "correlations": corr,
            "limitations": [
                "Output-projected local velocity Jacobian; not rollout Jacobian or global skill count",
                "Token settings at one training seed; correlations are descriptive, not causal",
                "Held-out alignment uses new output projections, NOT new observations/tasks",
                "Ridge scores predict linear corrections; weights are never updated",
                "Entropy rank is not the number of components covering 95% energy",
            ],
        },
    )
    colors = {run["tokens"]: plt.get_cmap("tab10")(index) for index, run in enumerate(runs)}

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def style(axis):
        axis.grid(alpha=0.2, linewidth=0.6)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    for axis, metric in zip(axes.flat, GEOMETRY, strict=True):
        for run, history in zip(runs, histories, strict=True):
            axis.plot(
                [row["step"] for row in history],
                [row[metric] for row in history],
                marker="o",
                markersize=3,
                color=colors[run["tokens"]],
                label=f"{run['tokens']} tokens",
            )
        axis.set(xlabel="Training step", ylabel=LABELS[metric])
        if metric == "tangent_trace" and all(row[metric] > 0 for history in histories for row in history):
            axis.set_yscale("log")
        style(axis)
    axes[0, 0].legend(ncol=2, fontsize=8)
    save(fig, "rank_dynamics")

    fig, axes = plt.subplots(
        2,
        len(args.steps),
        figsize=(4.1 * len(args.steps), 6.3),
        squeeze=False,
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    for column, step in enumerate(args.steps):
        for run in runs:
            probability, energy = spectra[run["tokens"], step]
            k = np.arange(1, len(probability) + 1)
            positive = probability > 0
            axes[0, column].semilogy(
                k[positive],
                probability[positive],
                color=colors[run["tokens"]],
                label=f"{run['tokens']} tokens",
            )
            axes[1, column].plot(k, energy, color=colors[run["tokens"]])
        axes[0, column].set(title=f"Step {step}", ylabel="Normalized singular value", xlabel="Component")
        axes[1, column].axhline(0.95, color="gray", ls="--", lw=1)
        axes[1, column].set(xlabel="Leading components", ylabel="Cumulative squared energy", ylim=(0, 1.03))
        style(axes[0, column])
        style(axes[1, column])
    axes[0, 0].legend(fontsize=8)
    save(fig, "spectra_and_energy")

    if alignment:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for run in runs:
            selected = sorted(
                [row for row in alignment if row["run"] == run["name"]], key=lambda row: row["step"]
            )
            last = next(row for row in selected if row["step"] == args.final_step)
            axes[0].plot(
                [row["k"] for row in last["top_k"]],
                [row["captured_fraction"] for row in last["top_k"]],
                color=colors[run["tokens"]],
                label=f"{run['tokens']} tokens",
            )
            axes[1].plot(
                [row["step"] for row in selected],
                [row["heldout_linear_reduction"] for row in selected],
                marker="o",
                color=colors[run["tokens"]],
            )
        ceiling = runs[0]["manifest"]["projected_rows"]
        axes[0].plot([0, ceiling], [0, 1], "--", color="gray", label="Isotropic reference k/M")
        axes[0].set(
            xlabel="Leading tangent directions",
            ylabel="Projected residual energy captured",
            title=f"Step {args.final_step}: top-k alignment",
            ylim=(0, 1.03),
        )
        axes[0].legend(fontsize=8)
        axes[1].axhline(0, color="gray", lw=1)
        axes[1].set(
            xlabel="Training step",
            ylabel="1 - corrected / original squared error",
            title="Independent output projections: linear prediction",
        )
        for axis in axes:
            style(axis)
        save(fig, "residual_alignment")

    if args.eval_root:
        plotted_metrics = [metric for metric in metrics if any(row.get(metric) is not None for row in final)]
        fig, axes = plt.subplots(
            2,
            int(np.ceil(len(plotted_metrics) / 2)),
            figsize=(5 * np.ceil(len(plotted_metrics) / 2), 7),
            squeeze=False,
            sharey=True,
            constrained_layout=True,
        )
        for axis, metric in zip(axes.flat, plotted_metrics, strict=False):
            for row in final:
                if row.get(metric) is not None:
                    axis.scatter(row[metric], row["overall_success_pct"], color=colors[row["tokens"]], s=45)
                    axis.annotate(
                        str(row["tokens"]),
                        (row[metric], row["overall_success_pct"]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=9,
                    )
            axis.set(xlabel=LABELS[metric], ylabel="LIBERO overall success (%)")
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.margins(0.12)
            style(axis)
        for axis in list(axes.flat)[len(plotted_metrics) :]:
            axis.set_axis_off()
        save(fig, "success_vs_geometry")
    print(f"Report saved: {output}")
    for row in final:
        success = f" success={row['overall_success_pct']:.2f}%" if args.eval_root else ""
        print(f"{row['run']}: J-rank={row['jacobian_erank']:.2f} NTK-rank={row['ntk_erank']:.2f}{success}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-root", type=Path, default=Path("/root/autodl-tmp/chkpt/2601-lerobot/vlm-only")
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source", choices=["base", "finetuned"], default="base")
    parser.add_argument("--run-prefix", default="pi05")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1500, 3000])
    parser.add_argument("--final-step", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-root", type=Path, help="Require complete, matched four-suite results when supplied"
    )
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument(
        "--include-alignment", action="store_true", help="Require all selected alignment records"
    )
    parser.add_argument("--analysis-name", default="prompt_alignment")
    parser.add_argument("--font-path", type=Path)
    make_report(parser.parse_args())


if __name__ == "__main__":
    main()
