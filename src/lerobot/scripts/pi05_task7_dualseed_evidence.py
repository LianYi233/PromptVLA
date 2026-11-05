"""Validate and visualize saved task7 comparisons without re-running a policy.

Only complete, paired scene x policy x fixed-init comparisons are accepted.
Calibration draws are never appended to held-out evaluation results. This
module validates the saved report; raw physics/checkpoint manifests are
audited by the evaluation producer before it writes that report.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


DUAL_PROTOCOL = "pi05-task7-dualseed-calibration-lock-v1"
FIXED_PROTOCOL = "pi05-task7-fixed-protocol-v1"
METHOD_LABELS = {
    "rope": "RoPE",
    "time_input": "Time-Input",
    "time_qk": "Time-QK",
    "time_v": "Time-V",
    "lowfreq": "Low frequency",
    "rms_gate": "RMS gain",
    "lowfreq_rms": "Low frequency + RMS",
}


def _seeds(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"Missing {label}")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"Invalid {label}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {label}")
    return values


def _pairs(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    return {
        "both_success": int(np.sum(a & b)),
        "both_failure": int(np.sum(~a & ~b)),
        "model_only_wins": int(np.sum(~a & b)),
        "rope_only_wins": int(np.sum(a & ~b)),
        "net_wins": int(np.sum(b) - np.sum(a)),
    }


def validate_comparison(report):
    """Return canonical paired data; reject partial or mixed-protocol reports."""
    protocol, stage = report.get("protocol"), report.get("stage")
    if protocol == DUAL_PROTOCOL:
        if stage not in ("calibrated_stress", "unselected_scenes"):
            raise ValueError("Expected a comparison stage, not calibration results")
        lock = report.get("lock", {})
        settings = lock.get("settings", {})
        if settings.get("protocol") != DUAL_PROTOCOL:
            raise ValueError("Missing dual-seed calibration provenance")
        if not lock.get("calibration", {}).get("in_target_band"):
            raise ValueError("Dual-seed report lacks its successful calibration lock")
        scene_key = "scene_seeds" if stage == "calibrated_stress" else "heldout_scene_seeds"
        scenes = _seeds(settings.get(scene_key), scene_key)
        policies = _seeds(settings.get("heldout_policy_seeds"), "heldout_policy_seeds")
        calibration = _seeds(settings.get("calibration_policy_seeds"), "calibration_policy_seeds")
        if set(calibration) & set(policies):
            raise ValueError("Calibration and test policy seeds overlap")
        if stage == "unselected_scenes" and set(scenes) & set(settings.get("scene_seeds", [])):
            raise ValueError("Unselected scenes overlap calibrated scenes")
    elif protocol == FIXED_PROTOCOL and stage == "fixed_protocol":
        settings = report.get("settings", {})
        scenes = _seeds(settings.get("scene_seeds"), "scene_seeds")
        policies = _seeds(settings.get("test_policy_seeds"), "test_policy_seeds")
        if settings.get("fixed_init_states") != list(range(10)):
            raise ValueError("Fixed protocol requires init states 0..9")
    else:
        raise ValueError("Unsupported comparison protocol/stage")
    cases = report.get("cases")
    if (
        not isinstance(cases, list) or len(set(cases)) != len(cases)
        or any(case not in METHOD_LABELS for case in cases)
        or "rope" not in cases or "time_input" not in cases
    ):
        raise ValueError("Comparison requires distinct known methods including rope and time_input")
    evaluations = report.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("No per-episode comparison results")
    cells, checkpoints = {}, {}
    for cell in evaluations:
        scene, policy = cell.get("scene_seed"), cell.get("policy_seed")
        if type(scene) is not int or type(policy) is not int:
            raise ValueError("Cell scene/policy seeds must be integers")
        if scene not in scenes or policy not in policies:
            raise ValueError("Unexpected scene/policy seed, including possible calibration leakage")
        key = (scene, policy)
        if key in cells:
            raise ValueError("Duplicate scene/policy cell")
        results = cell.get("results", {})
        if set(results) != set(cases):
            raise ValueError("Cell methods do not match declared cases")
        clean = {}
        for case in cases:
            row = results[case]
            if row.get("case") != case or row.get("scene_seed") != scene or row.get("policy_seed") != policy:
                raise ValueError("Per-method identity disagrees with its scene/policy cell")
            flags = row.get("per_episode")
            if (
                not isinstance(flags, list) or len(flags) != 10
                or any(type(x) not in (bool, int) or x not in (0, 1) for x in flags)
            ):
                raise ValueError("Expected ten boolean/0-1 per-episode outcomes")
            if type(row.get("episodes")) is not int or row["episodes"] != 10:
                raise ValueError("Expected ten episodes per cell")
            if type(row.get("successes")) is not int or row["successes"] != sum(flags):
                raise ValueError("Success count disagrees with raw episode flags")
            checkpoint = row.get("checkpoint")
            if not isinstance(checkpoint, str) or not checkpoint or not row.get("eval_info"):
                raise ValueError("Missing checkpoint/eval_info provenance")
            if case in checkpoints and checkpoints[case] != checkpoint:
                raise ValueError("A method uses different checkpoints across cells")
            checkpoints[case] = checkpoint
            clean[case] = list(map(bool, flags))
        cells[key] = clean
    expected = [(s, p) for s in scenes for p in policies]
    if set(cells) != set(expected):
        raise ValueError("Incomplete scene x policy comparison grid")
    matrix = np.asarray([
        [flag for key in expected for flag in cells[key][case]] for case in cases
    ], dtype=np.int64)
    baseline = matrix[cases.index("rope")]
    pairs = {case: _pairs(baseline, matrix[i]) for i, case in enumerate(cases) if case != "rope"}
    totals = {case: [int(matrix[i].sum()), int(matrix.shape[1])] for i, case in enumerate(cases)}
    supplied = report.get("summary", {})
    if "aggregate" in supplied and supplied["aggregate"] != totals:
        raise ValueError("Pooled summary disagrees with per-episode results")
    expected_pairs = {case: [value["model_only_wins"], value["rope_only_wins"]]
                      for case, value in pairs.items()}
    if "paired_counts" in supplied and supplied["paired_counts"] != expected_pairs:
        raise ValueError("Paired summary disagrees with per-episode results")
    by_scene = {}
    for scene in scenes:
        rows = {case: [flag for p in policies for flag in cells[(scene, p)][case]] for case in cases}
        by_scene[str(scene)] = {
            "aggregate": {case: [sum(flags), len(flags)] for case, flags in rows.items()},
            "paired_counts": {case: _pairs(rows["rope"], rows[case]) for case in cases if case != "rope"},
        }
    discordant = []
    for scene, policy in expected:
        for init, (a, b) in enumerate(zip(cells[(scene, policy)]["rope"],
                                          cells[(scene, policy)]["time_input"], strict=True)):
            if a != b:
                discordant.append({"scene_seed": scene, "policy_seed": policy, "init_state": init,
                                   "rope_success": a, "time_input_success": b})
    return {
        "source_protocol": protocol, "stage": stage,
        "cases": cases, "scene_seeds": scenes, "policy_seeds": policies,
        "cell_order": [list(key) for key in expected], "fixed_init_states": list(range(10)),
        "data": matrix.tolist(), "aggregate": totals, "paired_counts": pairs,
        "by_scene": by_scene, "checkpoints": checkpoints,
        "time_input_discordant_episodes": discordant,
        "note": "Repeated fixed initial states; policy seeds are inference draws, not training seeds.",
    }


def plot_comparison(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cases = data["cases"]
    matrix = np.asarray(data["data"])
    cells = data["cell_order"]
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(cells) * 1.65), 7.8),
                             gridspec_kw={"height_ratios": [1.35, 1]}, layout="constrained")
    ax = axes[0]
    ax.imshow(matrix, cmap=ListedColormap(["#e2e8f0", "#267c68"]), vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(cases)), [METHOD_LABELS[c] for c in cases])
    ax.set_xticks([i * 10 + 4.5 for i in range(len(cells))],
                  [f"Scene {s}\nPolicy {p}" for s, p in cells], fontsize=9)
    for i in range(1, len(cells)):
        ax.axvline(i * 10 - 0.5, color="#64748b", lw=1.2)
    for i, case in enumerate(cases):
        hits, n = data["aggregate"][case]
        ax.text(matrix.shape[1] + .4, i, f"{hits}/{n} ({100 * hits / n:.1f}%)", va="center", fontsize=9)
    ax.set_xlim(-.5, matrix.shape[1] + max(8, matrix.shape[1] * .15))
    ax.set_title("A. Actual paired outcomes — green: success; gray: failure")
    ax.set_xlabel("Within each cell: fixed initial states 0–9")
    alternatives = [case for case in cases if case != "rope"]
    scenes = data["scene_seeds"]
    width = .72 / max(1, len(alternatives))
    colors = ["#267c68", "#4c78a8", "#e39a35", "#9969a9", "#cf6d65", "#708690"]
    ax = axes[1]
    for k, case in enumerate(alternatives):
        deltas = [100 * data["by_scene"][str(s)]["paired_counts"][case]["net_wins"]
                  / data["by_scene"][str(s)]["aggregate"][case][1] for s in scenes]
        positions = np.arange(len(scenes)) + (k - (len(alternatives) - 1) / 2) * width
        ax.bar(positions, deltas, width=width, label=METHOD_LABELS[case], color=colors[k])
        for x, y in zip(positions, deltas, strict=True):
            ax.annotate(f"{y:+.1f}", (x, y), xytext=(0, 4 if y >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(0, color="#334155", lw=1)
    ax.set_xticks(range(len(scenes)), [str(s) for s in scenes])
    ax.set_xlabel("Scene seed")
    ax.set_ylabel("Success-rate change vs RoPE (percentage points)")
    ax.set_title("B. Scene-specific paired gain, including losses")
    ax.grid(axis="y", alpha=.2)
    ax.margins(y=.25)
    ax.legend(fontsize=9, ncol=min(3, len(alternatives)))
    titles = {"calibrated_stress": "Calibrated stress scenes; held-out policy seeds",
              "unselected_scenes": "Unselected scenes; held-out policy seeds",
              "fixed_protocol": "Fixed stress scenes; no recalibration for new checkpoints"}
    fig.suptitle(titles[data["stage"]], fontsize=14)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        path = output / f"fig01_paired_task7_outcomes.{ext}"
        fig.savefig(path, dpi=190, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def run_success_report(comparison_json, output):
    path = Path(comparison_json).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    report = validate_comparison(json.loads(raw))
    report["source_report"] = str(path)
    report["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    report["figure_paths"] = plot_comparison(report, output)
    report["validation_scope"] = "Saved report identities, complete paired grid, flags and counts; producer audits raw manifests."
    dest = Path(output) / "paired_success.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved dual-seed paired outcomes: {dest}", flush=True)
    print("Time-Input vs RoPE:", report["paired_counts"]["time_input"], flush=True)
    return report
