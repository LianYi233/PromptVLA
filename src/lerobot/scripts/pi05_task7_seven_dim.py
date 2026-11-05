"""CPU-only decomposition of saved, paired full-sampling action predictions.

This module reads completed ``pi05_task7_full_sampling`` artifacts. It never
loads a model, dataset, CUDA or simulator and never changes the source files.
The original full-sampling MSE already covers all seven action dimensions;
these outputs expose its components, not a new independent success metric.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path


PROTOCOL = "pi05-task7-seven-dim-v1"
SOURCE_PROTOCOL = "pi05-task7-full-sampling-v1"
MODES = ("correct", "zero", "reversed")
HORIZON = 50
ACTION_DIM = 7
GROUPS = {
    "all": list(range(7)),
    "translation": [0, 1, 2],
    "rotation": [3, 4, 5],
    "gripper": [6],
    **{f"dim{dimension}": [dimension] for dimension in range(7)},
}
SCHEMA_NOTE = (
    "Commands are in checkpoint-normalized coordinates, not measured robot poses. "
    "Translation (0..2), rotation (3..5) and gripper (6) groups assume the standard "
    "seven-command LIBERO ordering. The saved arrays do not establish physical "
    "units, reference frame, rotation parameterization or the open/close sign."
)
ARTIFACT_NAMES = (
    "seven_dim_evidence.json",
    "seven_dim_metrics.csv",
    "seven_dim_position_errors.csv.gz",
    "fig06_seven_dim_actions.png",
    "fig06_seven_dim_actions.svg",
    "fig07_seven_dim_error_changes.png",
    "fig07_seven_dim_error_changes.svg",
    "all_windows_seven_dim.pdf",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _window_groups(conditions):
    groups = {}
    for index, condition in enumerate(conditions):
        key = (condition["episode_index"], condition["global_frame_index"])
        groups.setdefault(key, []).append(index)
    return groups


def demo_means(values, conditions):
    """Average noise draws within a window, then windows within each demo."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or len(values) != len(conditions) or not len(conditions):
        raise ValueError("Nonempty condition metadata must match the measurement rows")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite measurements cannot be aggregated")
    groups = _window_groups(conditions)
    episodes = sorted({episode for episode, _ in groups})
    output = [np.stack([values[indices].mean(axis=0) for (episode, _), indices in groups.items()
                        if episode == target]).mean(axis=0) for target in episodes]
    return episodes, np.stack(output)


def _paired_interval(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return None
    indices = np.random.default_rng(20261105).integers(0, len(values), size=(4000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def summarize(predictions, targets, conditions, executed_steps):
    """Keep all seven components and the original equal-demo weighting."""
    import numpy as np

    _integer(executed_steps, "executed_steps", minimum=1)
    if executed_steps > HORIZON:
        raise ValueError("executed_steps cannot exceed the 50-command horizon")
    targets = np.asarray(targets, dtype=np.float64)
    expected = (len(conditions), HORIZON, ACTION_DIM)
    if not len(conditions) or targets.shape != expected or not np.isfinite(targets).all():
        raise ValueError("Targets must be finite [N, 50, 7] arrays matching the condition rows")
    if set(predictions) != set(MODES):
        raise ValueError("All three paired interventions are required")
    errors, output_changes = {}, {}
    for mode in MODES:
        predicted = np.asarray(predictions[mode], dtype=np.float64)
        if predicted.shape != expected or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid seven-dimensional predictions for {mode}")
        errors[mode] = (predicted - targets) ** 2
        if mode != "correct":
            output_changes[mode] = predicted - np.asarray(predictions["correct"], dtype=np.float64)

    regions = {"executed_prefix": [0, executed_steps], "full_chunk": [0, HORIZON]}
    metrics = {}
    for region, (start, stop) in regions.items():
        metrics[region] = {}
        for name, dimensions in GROUPS.items():
            per_demo, output_rms = {}, {}
            for mode in MODES:
                rows = errors[mode][:, start:stop, :][:, :, dimensions].mean(axis=(1, 2))
                episode_ids, per_demo[mode] = demo_means(rows, conditions)
            paired = {f"{mode}_minus_correct": per_demo[mode] - per_demo["correct"]
                      for mode in ("zero", "reversed")}
            for mode, changes in output_changes.items():
                rows = np.sqrt((changes[:, start:stop, :][:, :, dimensions] ** 2).mean(axis=(1, 2)))
                _, output_rms[mode] = demo_means(rows, conditions)
            metrics[region][name] = {
                "mean_mse": {mode: float(values.mean()) for mode, values in per_demo.items()},
                "per_demo_mse": {mode: values.tolist() for mode, values in per_demo.items()},
                "paired_deltas": {key: values.tolist() for key, values in paired.items()},
                "mean_paired_deltas": {key: float(values.mean()) for key, values in paired.items()},
                "paired_bootstrap_95pct": {key: _paired_interval(values) for key, values in paired.items()},
                "per_demo_output_change_rms": {mode: values.tolist() for mode, values in output_rms.items()},
                "mean_output_change_rms": {mode: float(values.mean()) for mode, values in output_rms.items()},
            }
    per_position = {mode: demo_means(values, conditions)[1] for mode, values in errors.items()}
    deltas = {mode: per_position[mode] - per_position["correct"] for mode in ("zero", "reversed")}
    # Ranking is an explicitly descriptive lookup, never a sample-selection rule.
    hotspots = []
    for mode, differences in deltas.items():
        for demo_index, episode in enumerate(episode_ids):
            for position in range(HORIZON):
                by_dimension = differences[demo_index, position]
                hotspots.append({
                    "intervention": mode, "episode_index": episode, "future_action_index": position,
                    "within_executed_prefix": position < executed_steps,
                    "all_dim_mse_difference": float(by_dimension.mean()),
                    "per_dimension_mse_difference": by_dimension.tolist(),
                    "per_dimension_contribution_to_all_dim_difference": (by_dimension / ACTION_DIM).tolist(),
                })
    hotspots.sort(key=lambda row: (-abs(row["all_dim_mse_difference"]), row["episode_index"],
                                  row["future_action_index"], row["intervention"]))
    return {
        "episode_ids": episode_ids, "number_of_demonstrations": len(episode_ids),
        "number_of_conditions": len(conditions), "number_of_windows": len(_window_groups(conditions)),
        "regions": regions, "dimension_groups": GROUPS, "action_schema_note": SCHEMA_NOTE,
        "physical_schema_verified_from_arrays": False, "metrics": metrics,
        "per_demo_position_dimension_mse": {mode: values.tolist() for mode, values in per_position.items()},
        "mean_position_dimension_mse_difference": {
            f"{mode}_minus_correct": values.mean(axis=0).T.tolist() for mode, values in deltas.items()
        },
        "largest_absolute_demo_position_differences": hotspots[:12],
        "hotspot_selection_note": (
            "Largest absolute demo-by-position differences are descriptive lookup aids only. "
            "All conditions, including these cells, remain in every primary summary and the full CSV."
        ),
        "aggregation": "Mean over noises per window, then windows per demo; equal-weight demos",
        "output_rms_aggregation": "Compute RMS across region/dimensions per condition, then the same hierarchy",
        "difference_sign": "intervention minus correct; positive means correct has lower recorded-action MSE",
        "confidence_interval": (
            "4000 deterministic paired bootstrap draws over demonstrations; conditional on this checkpoint. "
            "The components are descriptive multiple comparisons, not confirmatory significance tests."
        ),
    }


def load_sampling_outputs(directory):
    """Validate source identity and pairing before interpreting any saved arrays."""
    import numpy as np

    directory = Path(directory).expanduser().resolve()
    report_path = directory / "full_sampling_evidence.json"
    array_path = directory / "full_sampling_arrays.npz"
    report_bytes = report_path.read_bytes()
    source = json.loads(report_bytes)
    if source.get("protocol") != SOURCE_PROTOCOL or source.get("status") != "completed":
        raise ValueError("Input must be a completed pi05-task7-full-sampling-v1 run")
    if source.get("array_row_order") != "conditions" or not _digest(source.get("array_sha256")):
        raise ValueError("Source must declare the paired array order and its SHA256")
    if source.get("interventions") != list(MODES):
        raise ValueError("Source must contain Correct, Zero and Reversed from one checkpoint")
    hashes = {"report_sha256": hashlib.sha256(report_bytes).hexdigest(), "arrays_sha256": _sha256(array_path)}
    if hashes["arrays_sha256"] != source["array_sha256"]:
        raise ValueError("Saved NPZ SHA256 differs from the completed source report")
    conditions = source.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Source needs nonempty paired condition metadata")
    seen, frame_owners = set(), {}
    for condition in conditions:
        for key in ("episode_index", "global_frame_index", "frame_index", "noise_seed"):
            _integer(condition.get(key), f"conditions.{key}")
        fraction = condition.get("chunk_start_fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) \
                or not 0 <= fraction <= 1:
            raise ValueError("Every condition must declare its finite chunk_start_fraction in [0, 1]")
        identity = tuple(condition[key] for key in ("episode_index", "global_frame_index", "noise_seed"))
        if identity in seen:
            raise ValueError("Duplicate source condition")
        seen.add(identity)
        frame = condition["global_frame_index"]
        owner = (condition["episode_index"], condition["frame_index"], fraction)
        if frame in frame_owners and owner != frame_owners[frame]:
            raise ValueError("Conflicting identity for the same saved window")
        frame_owners[frame] = owner
    groups = _window_groups(conditions)
    episodes = sorted({episode for episode, _ in groups})
    if source.get("number_of_conditions") != len(conditions) or source.get("episode_ids") != episodes \
            or source.get("number_of_demonstrations") != len(episodes) \
            or source.get("number_of_anchors") != len(groups):
        raise ValueError("Source counts or demonstration IDs disagree with its condition rows")
    noise_seeds = source.get("noise_seeds")
    if not isinstance(noise_seeds, list) or not noise_seeds or len(set(noise_seeds)) != len(noise_seeds):
        raise ValueError("Source must declare distinct initial noise seeds")
    for value in noise_seeds:
        _integer(value, "noise_seeds")
    if any({conditions[index]["noise_seed"] for index in indices} != set(noise_seeds)
           for indices in groups.values()):
        raise ValueError("The saved window/noise grid is incomplete")
    sampling = source.get("sampling", {})
    executed_steps = _integer(sampling.get("executed_prefix_steps"), "executed_prefix_steps", minimum=1)
    native_steps = _integer(sampling.get("native_num_inference_steps"), "native_num_inference_steps", minimum=1)
    if executed_steps > HORIZON or source.get("regions") != {
        "executed_prefix": [0, executed_steps], "full_chunk": [0, HORIZON],
    }:
        raise ValueError("Source scoring regions differ from its declared sampling prefix/horizon")
    if sampling.get("targets_passed_to_inference") is not False \
            or sampling.get("initial_state") != "pure_gaussian_noise" \
            or sampling.get("one_checkpoint_loaded_for_all_interventions") is not True \
            or sampling.get("prediction_horizon") != HORIZON:
        raise ValueError("Source does not establish the paired pure-noise sampling contract")
    selected = _integer(source.get("representative_row"), "representative_row")
    if selected >= len(conditions) or source.get("representative_condition") != conditions[selected]:
        raise ValueError("Source representative condition does not identify a saved row")
    expected_selected = min(range(len(conditions)), key=lambda index: (
        conditions[index]["episode_index"], abs(conditions[index]["chunk_start_fraction"] - 0.5),
        conditions[index]["frame_index"], conditions[index]["noise_seed"],
    ))
    if selected != expected_selected:
        raise ValueError("Source representative condition violates its prespecified selection rule")
    with np.load(array_path, allow_pickle=False) as archive:
        required = {"clean_target", "initial_noise", *(f"{mode}_actions" for mode in MODES)}
        if set(archive.files) != required:
            raise ValueError("Saved NPZ must contain targets, initial noises and all three predictions")
        arrays = {name: archive[name].copy() for name in required}
    expected = (len(conditions), HORIZON, ACTION_DIM)
    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f"Saved {name} must contain finite floating-point values")
        if name != "initial_noise" and array.shape != expected:
            raise ValueError(f"Saved {name} must have shape {expected}")
    noise = arrays["initial_noise"]
    if noise.ndim != 3 or noise.shape[:2] != expected[:2] or noise.shape[2] < ACTION_DIM \
            or noise.dtype != np.dtype("float32"):
        raise ValueError("Saved initial noise must be float32 [N, 50, padded_action_dim >= 7]")
    audits = source.get("sampling_audits")
    if not isinstance(audits, list) or len(audits) != len(conditions):
        raise ValueError("Source is missing its per-condition native sampling audits")
    for index, row in enumerate(audits):
        if set(row) != set(MODES):
            raise ValueError("Sampling audit is missing an intervention")
        # Reproduce the existing tensor digest without importing Torch. Noise
        # is stored without its singleton batch axis, which the original hash includes.
        shape = (1, HORIZON, noise.shape[2])
        digest = hashlib.sha256(f"torch.float32:{shape}:".encode())
        digest.update(np.ascontiguousarray(noise[index]).tobytes())
        noise_hash = digest.hexdigest()
        observation_hashes = set()
        for mode in MODES:
            audit = row[mode]
            observation_hashes.add(audit.get("observation_sha256"))
            times = audit.get("flow_times")
            if audit.get("intervention") != mode or audit.get("initial_noise_sha256") != noise_hash \
                    or audit.get("starts_from_supplied_pure_noise") is not True \
                    or audit.get("denoising_calls") != native_steps:
                raise ValueError("Saved condition/noise order differs from its paired sampling audit")
            if not isinstance(times, list) or len(times) != native_steps or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not math.isclose(value, 1 - step / native_steps, abs_tol=1e-6)
                for step, value in enumerate(times)
            ):
                raise ValueError("Source audit does not contain the complete native denoising schedule")
        if len(observation_hashes) != 1 or not _digest(next(iter(observation_hashes))):
            raise ValueError("Intervention arms did not receive the same audited observation")
    for indices in groups.values():
        target = arrays["clean_target"][indices[0]]
        if any(not np.array_equal(target, arrays["clean_target"][index]) for index in indices[1:]):
            raise ValueError("The recorded target changed across initial noises of the same window")
    predictions = {mode: arrays[f"{mode}_actions"] for mode in MODES}
    summary = summarize(predictions, arrays["clean_target"], conditions, executed_steps)
    # This is a decomposition of the old seven-dimensional score, not a silently
    # changed normalization, row association or demonstration weighting.
    for region in summary["regions"]:
        for component in ("all", "gripper"):
            old = source.get("metrics", {}).get(region, {}).get(component, {})
            for mode in MODES:
                saved = old.get("mean_mse", {}).get(mode)
                calculated = summary["metrics"][region][component]["mean_mse"][mode]
                if not isinstance(saved, (int, float)) or isinstance(saved, bool) \
                        or not math.isfinite(saved) or not math.isclose(saved, calculated, rel_tol=1e-8, abs_tol=1e-12):
                    raise ValueError("Saved arrays do not reproduce the source whole-action/gripper MSE")
    return source, predictions, arrays["clean_target"], summary, {
        "report": str(report_path), "arrays": str(array_path), **hashes,
    }


def _action_figure(predictions, targets, condition, row_index, executed_steps, *, page_label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(ACTION_DIM, 1, figsize=(14, 16), sharex=True)
    positions = np.arange(HORIZON)
    styles = {"correct": ("#19835d", "-", "Correct time"),
              "zero": ("#7a8494", "--", "Time branch off (Zero)"),
              "reversed": ("#db6138", "-.", "Reversed time")}
    for dimension, axis in enumerate(axes):
        axis.axvspan(-0.5, executed_steps - 0.5, color="#dce7f3", alpha=0.7, zorder=0)
        axis.axvline(executed_steps - 0.5, color="#486581", linestyle="--", linewidth=1.0)
        axis.plot(positions, targets[row_index, :, dimension], color="#172a3a", linewidth=2.1,
                  label="Recorded action target")
        for mode, (color, line, label) in styles.items():
            axis.plot(positions, predictions[mode][row_index, :, dimension], color=color,
                      linestyle=line, linewidth=1.4, label=label)
        group = "translation*" if dimension < 3 else "rotation*" if dimension < 6 else "gripper*"
        axis.set_ylabel(f"Dim {dimension}\n{group}", fontsize=10)
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=10)
        axis.margins(y=0.12)
        axis.set_xlim(-0.5, HORIZON - 0.5)
    axes[-1].set_xlabel("Future action index (0–49); all seven dimensions share the same indices", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(color="#dce7f3"))
    labels.append(f"Executed prefix 0–{executed_steps - 1}")
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.936), ncol=3,
                  frameon=False, fontsize=11)
    figure.suptitle(
        f"Seven normalized action commands — {page_label}\n"
        f"Demo {condition['episode_index']}, frame {condition['frame_index']}, "
        f"same initial noise seed {condition['noise_seed']}", fontsize=17, weight="bold", y=0.987,
    )
    figure.text(0.012, 0.50, "Normalized commanded action (not measured robot state)",
                rotation="vertical", va="center", fontsize=11)
    figure.text(0.52, 0.028,
                "*Groups assume standard LIBERO command ordering; these axes are not meters or degrees. "
                "Gripper open/close sign is not inferred.\n"
                "All three conditions share a trained checkpoint and initial noise. "
                "The recorded future is a reference, not the only physically valid trajectory.",
                ha="center", fontsize=9.5)
    figure.subplots_adjust(left=0.105, right=0.985, top=0.89, bottom=0.080, hspace=0.14)
    return figure


def _save_figure(figure, output, stem):
    paths = []
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{stem}.", suffix=f".{extension}", dir=output)
        os.close(descriptor)
        try:
            figure.savefig(temporary, dpi=160)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        paths.append(str(path))
    return paths


def plot_outputs(source, summary, predictions, targets, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    conditions = source["conditions"]
    executed_steps = summary["regions"]["executed_prefix"][1]
    representative = source["representative_row"]
    figure = _action_figure(predictions, targets, conditions[representative], representative,
                            executed_steps, page_label="preselected example")
    try:
        paths = _save_figure(figure, output, "fig06_seven_dim_actions")
    finally:
        plt.close(figure)
    changes = summary["mean_position_dimension_mse_difference"]
    limit = max(abs(np.asarray(values)).max() for values in changes.values())
    # Use one unclipped symmetric scale for both panels, including all outliers.
    plotting_limit = float(limit) if limit > 0 else 1e-6
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    for axis, mode in zip(axes, ("zero", "reversed"), strict=True):
        values = np.asarray(changes[f"{mode}_minus_correct"])
        pixels = axis.imshow(values, origin="upper", aspect="auto", cmap="RdBu_r",
                             vmin=-plotting_limit, vmax=plotting_limit, interpolation="nearest")
        axis.axvline(executed_steps - 0.5, color="#172a3a", linestyle="--", linewidth=1.2)
        axis.set_yticks(range(ACTION_DIM), [f"dim {dimension}" for dimension in range(ACTION_DIM)])
        axis.set_ylabel("Command dimension")
        axis.set_title(f"{mode.title()} − Correct: recorded-action squared error, averaged within each demo")
        if limit == 0:
            axis.text(0.98, 0.90, "Every difference is exactly zero", transform=axis.transAxes, ha="right")
    axes[-1].set_xlabel("Future action index (0–49); dashed line ends the executed prefix")
    figure.colorbar(pixels, ax=axes, label="MSE difference: intervention − correct", shrink=0.90, pad=0.02)
    figure.suptitle("Where does the added time branch change seven-dimensional prediction error?", fontsize=17)
    figure.supxlabel(
        "Red: intervention has larger error. Blue: smaller error. White: similar errors, possibly equally wrong.\n"
        "Equal weight per demo; all recorded windows and noise draws included. Offline predictions, not robot rollouts.",
        fontsize=10,
    )
    try:
        paths += _save_figure(figure, output, "fig07_seven_dim_error_changes")
    finally:
        plt.close(figure)

    # Include every saved window in a deterministic collection. Selecting its
    # smallest fixed noise seed occurs before any errors are considered.
    pages = []
    for _, indices in sorted(_window_groups(conditions).items()):
        index = min(indices, key=lambda row: conditions[row]["noise_seed"])
        pages.append({"page": len(pages) + 1, "row_index": index, **conditions[index]})
    pdf = output / "all_windows_seven_dim.pdf"
    descriptor, temporary = tempfile.mkstemp(prefix=".all_windows.", suffix=".pdf", dir=output)
    os.close(descriptor)
    try:
        with PdfPages(temporary) as collection:
            collection.infodict()["Title"] = "Seven-dimensional predictions: every saved window"
            collection.infodict()["Subject"] = "Smallest fixed noise seed per window; no outcome filtering"
            for page in pages:
                index = page["row_index"]
                figure = _action_figure(predictions, targets, conditions[index], index, executed_steps,
                                        page_label=f"window {page['page']}/{len(pages)} (all windows)")
                try:
                    collection.savefig(figure)
                finally:
                    plt.close(figure)
        os.replace(temporary, pdf)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return paths, {"path": str(pdf), "pages": pages,
                   "selection": "Every window sorted by demo/frame; smallest fixed noise seed per window. "
                                "No outcome filtering. Metrics and CSV include every noise draw."}


def write_metrics_csv(summary, path):
    fields = ["region", "group", "dimensions", *(f"mse_{mode}" for mode in MODES)]
    for mode in ("zero", "reversed"):
        fields += [f"{mode}_minus_correct", f"{mode}_bootstrap_95pct_low", f"{mode}_bootstrap_95pct_high",
                   f"{mode}_output_change_rms"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for region, groups in summary["metrics"].items():
            for name, metric in groups.items():
                row = {"region": region, "group": name, "dimensions": ",".join(map(str, GROUPS[name])),
                       **{f"mse_{mode}": metric["mean_mse"][mode] for mode in MODES}}
                for mode in ("zero", "reversed"):
                    interval = metric["paired_bootstrap_95pct"][f"{mode}_minus_correct"]
                    row.update({f"{mode}_minus_correct": metric["mean_paired_deltas"][f"{mode}_minus_correct"],
                                f"{mode}_bootstrap_95pct_low": interval[0] if interval is not None else "",
                                f"{mode}_bootstrap_95pct_high": interval[1] if interval is not None else "",
                                f"{mode}_output_change_rms": metric["mean_output_change_rms"][mode]})
                writer.writerow(row)


def write_position_csv(predictions, targets, conditions, executed_steps, path):
    """Write every error contribution, including all noise draws and hotspots."""
    import numpy as np

    fields = ["row_index", "episode_index", "global_frame_index", "frame_index", "noise_seed",
              "future_action_index", "dimension", "intervention", "within_executed_prefix",
              "condition_mean_weight", "recorded_target", "correct_prediction", "intervention_prediction",
              "correct_squared_error", "intervention_squared_error", "error_difference", "output_difference",
              "contribution_to_full_chunk_all_mse_delta", "contribution_to_prefix_all_mse_delta"]
    groups = _window_groups(conditions)
    episodes = sorted({episode for episode, _ in groups})
    windows_per_demo = {episode: sum(key[0] == episode for key in groups) for episode in episodes}
    values = {mode: np.asarray(value, dtype=np.float64) for mode, value in predictions.items()}
    targets = np.asarray(targets, dtype=np.float64)
    # Deterministic gzip metadata helps make repeated postprocessing reviewable.
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed, \
            io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, condition in enumerate(conditions):
            key = (condition["episode_index"], condition["global_frame_index"])
            weight = 1.0 / (len(episodes) * windows_per_demo[key[0]] * len(groups[key]))
            for position in range(HORIZON):
                for dimension in range(ACTION_DIM):
                    target = float(targets[index, position, dimension])
                    correct = float(values["correct"][index, position, dimension])
                    correct_error = (correct - target) ** 2
                    for mode in ("zero", "reversed"):
                        predicted = float(values[mode][index, position, dimension])
                        error = (predicted - target) ** 2
                        difference = error - correct_error
                        writer.writerow({
                            "row_index": index,
                            **{name: condition[name] for name in
                               ("episode_index", "global_frame_index", "frame_index", "noise_seed")},
                            "future_action_index": position, "dimension": dimension, "intervention": mode,
                            "within_executed_prefix": position < executed_steps, "condition_mean_weight": weight,
                            "recorded_target": target, "correct_prediction": correct,
                            "intervention_prediction": predicted, "correct_squared_error": correct_error,
                            "intervention_squared_error": error, "error_difference": difference,
                            "output_difference": predicted - correct,
                            "contribution_to_full_chunk_all_mse_delta": weight * difference / (HORIZON * ACTION_DIM),
                            "contribution_to_prefix_all_mse_delta": (
                                weight * difference / (executed_steps * ACTION_DIM) if position < executed_steps else 0.0
                            ),
                        })


def _atomic_table(path, write):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        write(Path(temporary))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(args):
    from lerobot.scripts.pi05_task7_train_eval import atomic_json, output_lock

    directory = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else directory.parent / "seven_dim"
    if directory == output:
        raise ValueError("Use a separate output directory so the source full-sampling artifacts stay unchanged")
    report_path = output / "seven_dim_evidence.json"
    with output_lock(output):
        atomic_json(report_path, {"protocol": PROTOCOL, "status": "running", "input": str(directory)})
        try:
            source, predictions, targets, summary, provenance = load_sampling_outputs(directory)
            print(f"Seven-dimensional CPU analysis: {summary['number_of_demonstrations']} demos, "
                  f"{summary['number_of_windows']} windows, {summary['number_of_conditions']} paired conditions",
                  flush=True)
            figures, collection = plot_outputs(source, summary, predictions, targets, output)
            metrics_csv = output / "seven_dim_metrics.csv"
            position_csv = output / "seven_dim_position_errors.csv.gz"
            _atomic_table(metrics_csv, lambda path: write_metrics_csv(summary, path))
            _atomic_table(position_csv, lambda path: write_position_csv(
                predictions, targets, source["conditions"], summary["regions"]["executed_prefix"][1], path,
            ))
            if _sha256(provenance["report"]) != provenance["report_sha256"] \
                    or _sha256(provenance["arrays"]) != provenance["arrays_sha256"]:
                raise RuntimeError("Source full-sampling artifacts changed during postprocessing")
            summary.update({
                "protocol": PROTOCOL, "status": "completed", "source": provenance,
                "checkpoint": source["checkpoint"], "checkpoint_fingerprint": source["checkpoint_fingerprint"],
                "source_protocol": source["protocol"], "conditions": source["conditions"],
                "representative_row": source["representative_row"],
                "representative_condition": source["representative_condition"],
                "representative_selection": source["representative_selection"],
                "figure_paths": figures, "all_windows_collection": collection,
                "metrics_csv": str(metrics_csv), "position_errors_csv": str(position_csv),
                "position_errors_csv_note": (
                    "Every condition, future index, dimension and intervention is retained. "
                    "Sum the contribution columns per intervention to reproduce the pooled full/prefix MSE delta. "
                    "Filter episode_index/future_action_index to decompose any heatmap cell across dimensions, "
                    "windows and initial noise draws."
                ),
                "source_sha256": _sha256(Path(__file__)),
                "note": (
                    "No inference or robot execution is performed. The old full-sampling MSE already uses all seven "
                    "dimensions; this report decomposes that same score. A smaller recorded-action error does not "
                    "by itself establish coordinated motion or task success. Zero retains the same checkpoint and RoPE."
                ),
            })
            summary["artifact_sha256"] = {name: _sha256(output / name) for name in ARTIFACT_NAMES
                                           if name != report_path.name}
            atomic_json(report_path, summary)
        except BaseException as error:
            atomic_json(report_path, {"protocol": PROTOCOL, "status": "failed", "input": str(directory),
                                      "error": f"{type(error).__name__}: {error}"})
            raise
    print(f"Saved {report_path}", flush=True)
    print("No model/GPU/simulator was loaded; source predictions are unchanged.", flush=True)
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing completed full_sampling_evidence.json and full_sampling_arrays.npz")
    parser.add_argument("--output", type=Path, help="Default: sibling seven_dim directory")
    parser.add_argument("--dry-run", action="store_true", help="Describe outputs without imports, reads or writes")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        args.input = args.input.expanduser().resolve()
        args.output = args.output.expanduser().resolve() if args.output else args.input.parent / "seven_dim"
        if args.input == args.output:
            raise ValueError("Input and output directories must be different")
        if args.dry_run:
            print("DRY RUN: CPU postprocessing only; no model/dataset/GPU/simulator imports and no files written.")
            print(f"Input: {args.input}")
            print(f"Output: {args.output}")
            print("Figures: all seven normalized commands, two dimension-by-index error-change heatmaps, "
                  "and every window in a deterministic PDF collection.")
            print("Scores: prefix/full, per dimension and translation/rotation/gripper groups; "
                  "equal-demo paired MSE deltas and output-change RMS.")
            return None
        return run(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Task7 seven-dimensional analysis failed: {error}\n")


if __name__ == "__main__":
    main()
