"""Compare saved PI05 NTKs across checkpoints with paired-seed, orange-red CKA heatmaps.

Only NumPy and Matplotlib are needed; no model, dataset, training loss or GPU is loaded.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

STAGES = ("before", "priming", "stage2", "final")
MODULES = {"vlm": "VLM", "action": "Action head"}
# Low similarity: pale orange; high similarity: deep red. Shared by all modules.
HEATMAP_COLORS = ("#FFF4E8", "#FED7AA", "#FB923C", "#E34A33", "#A50F15")


def centered_unit_kernel(kernel):
    """Return HKH / ||HKH||_F, rejecting invalid or uninformative sample Grams.

    Scaling before centering avoids overflow/underflow and makes the degeneracy
    threshold relative to the input. A constant Gram has undefined centered CKA.
    """
    matrix = np.asarray(kernel, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("NTK kernel must be a square matrix with at least two samples")
    if not np.isfinite(matrix).all():
        raise ValueError("NTK kernel contains non-finite entries")
    scale = np.max(np.abs(matrix))
    if scale == 0:
        raise ValueError("Centered CKA is undefined for a zero kernel")
    matrix = matrix / scale
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-8):
        raise ValueError("NTK kernel must be symmetric")
    matrix = (matrix + matrix.T) / 2
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] < -1e-8 * np.max(np.abs(eigenvalues)):
        raise ValueError("NTK kernel must be positive semidefinite")
    centered = matrix - matrix.mean(axis=0)[None, :] - matrix.mean(axis=1)[:, None] + matrix.mean()
    norm = np.linalg.norm(centered)
    if norm <= 1e-12 * np.linalg.norm(matrix):
        raise ValueError("Centered CKA is undefined for a constant or numerically constant kernel")
    return centered / norm


def compute_similarity(document, scope="auto"):
    """Compute CKA within each probe seed first, then take cellwise median/IQR.

    All records must belong to the single shared-sample manifest in results.json.
    Comparing kernels from different seeds, or averaging kernels before CKA,
    would mix checkpoint effects with changes in the probing experiment.
    """
    manifest, records = document["manifest"], document["records"]
    stages, seeds = manifest["stages"], manifest["seeds"]
    names = [stage["name"] for stage in stages]
    if not 2 <= len(names) <= 4 or names != list(STAGES[: len(names)]):
        raise ValueError("Need two to four consecutive NTK stages starting before training")
    steps = [stage["total_step"] for stage in stages]
    if any(not isinstance(step, int) or step < 0 for step in steps) or steps[0] != 0:
        raise ValueError("Checkpoint steps must be nonnegative cumulative integers starting at zero")
    if any(right <= left for left, right in zip(steps, steps[1:], strict=False)):
        raise ValueError("Checkpoint cumulative steps must increase")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Expected distinct paired probe seeds")
    lookup = {(row["stage"], row["seed"]): row for row in records}
    expected = {(name, seed) for name in names for seed in seeds}
    if len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError("Incomplete or duplicate paired stage/seed records")
    indices = manifest.get("sample_indices")
    if not isinstance(indices, list) or len(indices) < 2:
        raise ValueError("Need manifest.sample_indices to identify the common NTK sample ordering")
    available = {group.split("/")[0] for group in records[0]["groups"]}
    if scope == "auto":
        scopes = [name for name in ("backbone", "prompts") if name in available]
    elif scope == "both":
        scopes = ["backbone", "prompts"]
    elif scope in ("backbone", "prompts"):
        scopes = [scope]
    else:
        raise ValueError(f"Unknown scope: {scope}")
    if not scopes:
        raise ValueError("No backbone or prompts NTK groups found")
    for stage in stages:
        for seed in seeds:
            if lookup[(stage["name"], seed)]["total_step"] != stage["total_step"]:
                raise ValueError(f"Record step disagrees with manifest: {stage['name']}, seed {seed}")

    summary = {
        "metric": "centered kernel alignment (biased HSIC normalization)",
        "formula": "<H K_s H, H K_t H>_F / (||H K_s H||_F ||H K_t H||_F)",
        "aggregation": "CKA per paired probe seed, then cellwise median and quartiles",
        "uncertainty": "Probe/flow randomness; not independent training-run uncertainty",
        "stages": [{"name": stage["name"], "total_step": stage["total_step"]} for stage in stages],
        "seeds": seeds,
        "sample_indices": indices,
        "sample_digest": manifest.get("sample_digest"),
        "protocol": manifest.get("protocol"),
        "groups": {},
    }
    pairs = []
    for current in scopes:
        for module in MODULES:
            group = f"{current}/{module}"
            similarities, reference_signature = [], None
            has_signatures = [group in row.get("parameter_groups", {}) for row in records]
            if any(has_signatures) and not all(has_signatures):
                raise ValueError(f"Incomplete parameter-group signatures for {group}")
            for seed in seeds:
                kernels = []
                for stage in stages:
                    row = lookup[(stage["name"], seed)]
                    values = row["groups"].get(group)
                    if values is None or "kernel" not in values:
                        raise ValueError(
                            f"Missing saved kernel for {group}, {stage['name']}, seed {seed}; "
                            "use the original analysis results.json, not metrics.csv"
                        )
                    signature = row.get("parameter_groups", {}).get(group)
                    if reference_signature is None:
                        reference_signature = signature
                    elif signature != reference_signature:
                        raise ValueError(f"Parameter groups differ across stages/seeds: {group}")
                    try:
                        kernel = centered_unit_kernel(values["kernel"])
                    except ValueError as error:
                        raise ValueError(f"{group}, {stage['name']}, seed {seed}: {error}") from error
                    if kernel.shape != (len(indices), len(indices)):
                        raise ValueError(f"Kernel shape disagrees with manifest.sample_indices: {group}")
                    kernels.append(kernel)
                matrix = np.einsum("sij,tij->st", kernels, kernels)
                # Valid PSD kernels yield [0, 1]; clip only floating-point roundoff.
                if matrix.min() < -1e-7 or matrix.max() > 1 + 1e-7:
                    raise ValueError(f"Invalid centered kernel alignment for {group}")
                matrix = np.clip(matrix, 0, 1)
                similarities.append(matrix)
                for right in range(1, len(stages)):
                    for left in range(right):
                        pairs.append(
                            {
                                "group": group,
                                "seed": seed,
                                "stage_from": names[left],
                                "step_from": steps[left],
                                "stage_to": names[right],
                                "step_to": steps[right],
                                "cka": float(matrix[left, right]),
                            }
                        )
            q25, median, q75 = np.quantile(similarities, [0.25, 0.5, 0.75], axis=0)
            summary["groups"][group] = {
                "median": median.tolist(),
                "q25": q25.tolist(),
                "q75": q75.tolist(),
                "per_seed": np.asarray(similarities).tolist(),
                "parameter_groups_checked": all(has_signatures),
            }
    return summary, pairs


def plot_similarity(results_path, output_dir=None, scope="auto", full_matrix=False):
    """Write separate module heatmaps, a two-panel figure per scope, and numeric results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    results_path = Path(results_path)
    source = results_path.read_bytes()
    summary, pairs = compute_similarity(json.loads(source), scope)
    summary["source"] = {"path": str(results_path.resolve()), "sha256": hashlib.sha256(source).hexdigest()}
    summary["plot"] = {
        "color_limits": [0, 1],
        "full_matrix": full_matrix,
        "annotation_decimals": 2,
        "colormap": "ntk_orange_red",
        "colors": list(HEATMAP_COLORS),
    }
    out = Path(output_dir) if output_dir is not None else results_path.parent / "similarity"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage_similarity.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (out / "stage_similarity_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)

    cmap = LinearSegmentedColormap.from_list("ntk_orange_red", HEATMAP_COLORS)
    cmap.set_bad("white")
    steps = [str(stage["total_step"]) for stage in summary["stages"]]
    count = len(steps)

    def draw(axis, group, title):
        data = np.asarray(summary["groups"][group]["median"])
        mask = np.triu(np.ones_like(data, dtype=bool), k=1) if not full_matrix else np.zeros_like(data, bool)
        # Keep the cells as vector objects in PDF/SVG, with no decorative grid.
        mesh = axis.pcolormesh(
            np.ma.array(data, mask=mask), cmap=cmap, vmin=0, vmax=1, edgecolors="none", rasterized=False
        )
        axis.set(xlim=(0, count), ylim=(count, 0), aspect="equal")
        axis.set_xticks(np.arange(count) + 0.5, steps)
        axis.set_yticks(np.arange(count) + 0.5, steps)
        axis.tick_params(length=0, pad=7, labelsize=10, colors="#555555")
        axis.spines[:].set_visible(False)
        axis.set_title(title, fontsize=13, fontweight="semibold", pad=15, color="#303030")
        for row in range(count):
            for column in range(count):
                if not mask[row, column]:
                    value = data[row, column]
                    # Choose black/white from the actual cell luminance, so
                    # orange midtones remain readable when the palette changes.
                    rgb = np.asarray(cmap(float(value))[:3])
                    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                    luminance = float(linear_rgb @ [0.2126, 0.7152, 0.0722])
                    axis.text(
                        column + 0.5,
                        row + 0.5,
                        "1" if row == column else f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="white" if luminance < 0.179 else "black",
                    )
        return mesh

    def finish(fig, mesh, axes, name):
        colorbar = fig.colorbar(mesh, ax=axes, fraction=0.04, pad=0.04, shrink=0.78, ticks=[0, 0.5, 1])
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0, labelsize=9, colors="#555555")
        colorbar.ax.set_title("CKA", fontsize=9, color="#555555", pad=9)
        fig.text(
            0.5,
            0.03,
            f"Median across {len(summary['seeds'])} paired probe seeds",
            ha="center",
            fontsize=9,
            color="#777777",
        )
        for extension in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{extension}", dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)

    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"}):
        for current in dict.fromkeys(group.split("/")[0] for group in summary["groups"]):
            fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1))
            fig.subplots_adjust(left=0.08, right=0.90, top=0.86, bottom=0.18, wspace=0.35)
            for axis, (module, title) in zip(axes, MODULES.items(), strict=True):
                mesh = draw(axis, f"{current}/{module}", title)
            finish(fig, mesh, axes, f"{current}_stage_similarity")
            for module, title in MODULES.items():
                fig, axis = plt.subplots(figsize=(4.5, 4.1))
                fig.subplots_adjust(left=0.15, right=0.86, top=0.86, bottom=0.18)
                mesh = draw(axis, f"{current}/{module}", title)
                finish(fig, mesh, axis, f"{current}_{module}_stage_similarity")
    print(f"Saved paired-seed CKA heatmaps (PNG/PDF/SVG) and numeric results to {out}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Saved four-stage or partial-stage results.json")
    parser.add_argument("--scope", choices=("auto", "backbone", "prompts", "both"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--full-matrix", action="store_true", help="Also draw the redundant upper triangle")
    args = parser.parse_args()
    plot_similarity(args.results, args.output_dir, args.scope, args.full_matrix)


if __name__ == "__main__":
    main()
