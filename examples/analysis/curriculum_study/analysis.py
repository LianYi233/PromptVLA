"""Empirical loss-space trajectories and paired-risk summaries, never toy evidence."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import META, SCHEMA, arrays_hash, digest, json_hash, load_plan, load_scores, read_json, write_json


def aggregate(rows: list[dict], *, weighting: str = "task") -> dict[str, float]:
    """Mask-aware SSE/count per sample, then sample/episode/task macro averages.

    Frames or flow-noise draws are NOT independent training-seed replicates.
    A task-balanced comparison requires the SAME tasks represented in both groups.
    """
    if weighting not in {"task", "micro"}:
        raise ValueError("Unknown aggregation")
    cells = defaultdict(lambda: [0., 0])
    for row in rows:
        count, error = row["valid_coordinates"], row["squared_error_sum"]
        if type(count) is not int or count <= 0 or not np.isfinite(error) or error < 0:
            raise ValueError("Invalid SSE/valid-coordinate count")
        key = (row["group"], int(row["task_index"]), int(row["episode_index"]),
               int(row.get("dataset_index", row.get("index", row["sample_index"]))))
        cells[key][0] += error
        cells[key][1] += count
    groups = sorted({key[0] for key in cells})
    if len(groups) != 2:
        raise ValueError("Need exactly two nonempty probe groups")
    if weighting == "micro":
        return {g: sum(v[0] for k, v in cells.items() if k[0] == g) /
                sum(v[1] for k, v in cells.items() if k[0] == g) for g in groups}
    tasks = {g: {key[1] for key in cells if key[0] == g} for g in groups}
    if tasks[groups[0]] != tasks[groups[1]]:
        raise ValueError("Probe groups do not cover the same tasks. Use balanced new probes, or "
                         "--weighting micro for an explicitly exploratory legacy plot.")
    episodes = defaultdict(list)
    for (group, task, episode, _), (error, count) in cells.items():
        episodes[group, task, episode].append(error / count)
    task_values = defaultdict(list)
    for (group, task, _), values in episodes.items():
        task_values[group, task].append(float(np.mean(values)))
    return {g: float(np.mean([np.mean(values) for (group, _), values in task_values.items() if group == g]))
            for g in groups}


def _validate_measurement(record, identities, timesteps, repeats=1):
    expected = {(i, float(t), r) for i in range(len(identities)) for t in timesteps for r in range(repeats)}
    seen = set()
    for row in record["samples"]:
        key = int(row["sample_index"]), float(row["time"]), int(row.get("noise_repeat", 0))
        if key in seen or key not in expected:
            raise ValueError("Duplicate/unexpected sample/time/noise-repeat in a probe record")
        seen.add(key)
        identity = identities[key[0]]
        if int(row["task_index"]) != int(identity["task_index"]):
            raise ValueError("Probe task changed")
        if row["valid_coordinates"] <= 0 or not np.isfinite(row["squared_error_sum"]):
            raise ValueError("Nonfinite/incomplete probe error")
    if seen != expected:
        raise ValueError("Missing sample/time/noise-repeat entries")


def read_new_run(run: str | Path, *, weighting="task") -> tuple[dict, list[dict]]:
    directory = Path(run) / "curriculum_probe"
    manifest = read_json(directory / "manifest.json")
    if manifest.get("protocol") != SCHEMA:
        raise ValueError("Unknown probe protocol")
    protocol = manifest["probe"]
    rows = []
    for path in sorted(directory.glob("step_*.json")):
        record = read_json(path)
        if (record["probe_id"], record["processed_inputs_id"]) != (
                protocol["probe_id"], manifest["processed_inputs_id"]):
            raise ValueError("Probe protocol changed during training")
        _validate_measurement(record, protocol["samples"], protocol["timesteps"], protocol["noise_repeats"])
        for sample in record["samples"]:
            identity = protocol["samples"][sample["sample_index"]]
            for key in ("dataset_index", "episode_index", "group", "index", "frame_index"):
                if sample[key] != identity[key]:
                    raise ValueError(f"Probe identity changed: {key}")
        rows.append({"step": record["step"], **aggregate(record["samples"], weighting=weighting)})
    if not rows or rows[0]["step"] != 0 or len({r["step"] for r in rows}) != len(rows):
        raise ValueError("Missing step zero or duplicate probe steps")
    return manifest, rows


def legacy_runs(runs: dict[str, Path], scores_path: Path, baseline_plan: Path,
                *, weighting="task", allow_unverified=False):
    """Read-only importer for old prompt_rank JSONL. Never relabel as held-out."""
    scores, score_manifest = load_scores(scores_path)
    base = load_plan(baseline_plan)
    if score_manifest["dataset_metadata_sha256"] != base["manifest"]["dataset_metadata_sha256"]:
        raise ValueError("Scores and baseline dataset differ")
    low = set()
    for task in np.unique(base["task_index"]):
        frames = np.unique(base["index"][base["task_index"] == task])
        ranked = frames[np.lexsort((frames, scores["score"][frames]))]
        low.update(map(int, ranked[:len(ranked)//2]))
    trajectories, reference, signatures, sources = {}, None, [], {}
    for label, run in runs.items():
        directory = Path(run) / "prompt_rank"
        manifest = read_json(directory / "manifest.json")
        identities = []
        for i, item in enumerate(manifest["samples"]):
            index = int(item.get("index", item["dataset_index"]))
            if not 0 <= index < len(scores["index"]):
                raise ValueError("Legacy probe index outside score archive")
            for key in ("task_index", "episode_index", "frame_index"):
                if key in item and int(item[key]) != int(scores[key][index]):
                    raise ValueError(f"Legacy identity mismatch: {key}")
            # Probes outside the candidate pool are classified using the same task
            # boundary, not arbitrarily assigned to high_score.
            task = int(scores["task_index"][index])
            candidates = np.unique(base["index"][base["task_index"] == task])
            ranked = candidates[np.lexsort((candidates, scores["score"][candidates]))]
            edge = int(ranked[len(ranked)//2 - 1])
            is_low = (float(scores["score"][index]), index) <= (float(scores["score"][edge]), edge)
            identities.append({"sample_index": i, "dataset_index": index,
                               **{key: int(scores[key][index]) for key in META},
                               "group": "low_score" if is_low else "high_score"})
        signature = {key: manifest.get(key) for key in ("sample_sha256", "options", "action_dim",
                    "chunk_size", "normalization_mapping", "pretrained_path", "forward_precision")}
        signature["identities"] = identities
        # A matching file hash establishes equal cached noise as well as images;
        # tensor-level hashing is optional below when torch is installed.
        cache = directory / "fixed_probe.pt"
        if cache.exists():
            import torch
            from .train import tensor_fingerprint
            data = torch.load(cache, map_location="cpu", weights_only=True)
            noises = data["noise"]
            if isinstance(noises, list):
                noises = torch.stack(noises)
            signature["processed_inputs"] = tensor_fingerprint(data["samples"], noises)
        elif not allow_unverified:
            raise ValueError(f"Missing fixed_probe.pt for {label}; cannot verify paired noise")
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Legacy runs do not share exactly the same probe protocol/cache")
        pdet = Path(run) / "pdet/manifest-rank0.json"
        if pdet.exists():
            pd = read_json(pdet)
            info = pd["order_plan"]
            signatures.append({key: info[key] for key in ("instance_pool_sha256", "task_schedule_sha256",
                               "dataset_metadata_sha256", "steps", "batch_size", "world_size")})
            if info["dataset_metadata_sha256"] != score_manifest["dataset_metadata_sha256"]:
                raise ValueError("Legacy training dataset differs from scores")
        elif not allow_unverified:
            raise ValueError("Missing PDET provenance; use --allow-unverified only for exploratory plots")
        losses, seen_steps = [], set()
        path = directory / "fixed_training_probe_loss.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)  # truncated/malformed records are NOT silently skipped
            if record["step"] in seen_steps:
                raise ValueError("Duplicate legacy step (possibly a mixed/resumed log)")
            seen_steps.add(record["step"])
            _validate_measurement(record, identities, manifest["options"]["timesteps"])
            rows = [{**sample, **identities[sample["sample_index"]]} for sample in record["samples"]]
            losses.append({"step": record["step"], **aggregate(rows, weighting=weighting)})
        losses.sort(key=lambda x: x["step"])
        if not losses or losses[0]["step"] != 0:
            raise ValueError("Legacy trajectory has no common initial probe")
        trajectories[label] = losses
        sources[label] = {"run": str(run), "loss_sha256": digest(path), "manifest_sha256": digest(directory / "manifest.json")}
    if signatures and any(s != signatures[0] for s in signatures):
        raise ValueError("Legacy runs have different exposure pools/budgets; compare A/B/C or D/E/F separately")
    return trajectories, {"scope": "Exploratory fixed TRAINING probes, not held-out validation or control risk",
                          "weighting": weighting, "probe_pairing_verified": not allow_unverified,
                          "sources": sources, "score_sha256": digest(scores_path),
                          "note": "Provenance/cache checks do not replace all-step PDET auditing"}


def export_trajectories(trajectories: dict[str, list[dict]], output: Path, metadata: dict):
    """One figure per LR setting/call, unsmoothed measured points and arrows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not trajectories:
        raise ValueError("No trajectories")
    keys = set(next(iter(trajectories.values()))[0]) - {"step"}
    if keys == {"low_score", "high_score"}:
        xkey, ykey = "low_score", "high_score"
        labels = ("Low-score probe flow MSE", "High-score probe flow MSE")
    elif keys == {"routine", "critical"}:
        xkey, ykey = "routine", "critical"
        labels = ("Routine-labelled probe flow MSE", "Critical-labelled probe flow MSE")
    else:
        raise ValueError(f"Unexpected group labels: {keys}")
    with (output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "step", xkey, ykey])
        writer.writeheader()
        for label, rows in trajectories.items():
            for row in rows:
                if set(row) != {"step", xkey, ykey} or not np.isfinite([row[xkey], row[ykey]]).all():
                    raise ValueError("Incompatible trajectory groups or nonfinite losses")
                writer.writerow({"run": label, **row})
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for label, rows in trajectories.items():
        x, y = [r[xkey] for r in rows], [r[ykey] for r in rows]
        ax.plot(x, y, marker=".", linewidth=1.4, label=label)
        ax.scatter(x[:1], y[:1], marker="o", facecolors="none", s=65)
        ax.scatter(x[-1:], y[-1:], marker="X", s=65)
        ax.annotate(f"{label}: {rows[-1]['step']}", (x[-1], y[-1]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
        for i in sorted(set(np.linspace(1, len(rows)-1, min(4, len(rows)-1), dtype=int))):
            ax.annotate("", (x[i], y[i]), (x[i-1], y[i-1]),
                        arrowprops={"arrowstyle": "->", "linewidth": .8})
    ax.set(xlabel=labels[0], ylabel=labels[1], title="Measured learning trajectories (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"trajectory.{ext}", dpi=220)
    plt.close(fig)
    write_json(output / "provenance.json", metadata, exclusive=True)


def plot_new(runs: dict[str, Path], output: Path, *, weighting="task"):
    trajectories, reference, schedules = {}, None, set()
    for label, path in runs.items():
        manifest, rows = read_new_run(path, weighting=weighting)
        signature = {k: manifest[k] for k in ("probe", "processed_inputs_id", "dtype", "horizon", "action_dim")}
        if reference is None:
            reference = signature
        elif reference != signature:
            raise ValueError("Different fixed probes/noise/precision; refuse overlay")
        spec = read_json(Path(path) / "study_spec.json")
        schedules.add(spec["lr_mode"])
        if spec["steps"] != rows[-1]["step"]:
            raise ValueError("Incomplete training trajectory; final probe is missing")
        trajectories[label] = rows
    if len(schedules) != 1:
        raise ValueError("Plot each LR schedule separately to avoid mixing the factorial")
    export_trajectories(trajectories, output, {"scope": reference["probe"]["scope"],
                        "weighting": weighting, "lr_mode": next(iter(schedules)),
                        "probe_id": reference["probe"]["probe_id"], "processed_inputs_id": reference["processed_inputs_id"],
                        "note": "Loss-space coordinates, not parameter directions, risk or a known optimum"})


def paired_summary(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    seen = set()
    for row in records:
        key = row["state_uid"], row["trial"]
        if key in seen:
            raise ValueError("Duplicate perturbation trial")
        seen.add(key)
        for branch in ("clean_success", "perturbed_success", "zero_success"):
            if type(row[branch]) is not bool:
                raise ValueError("Successes must be actual booleans")
        if row["clean_success"] != row["zero_success"] or row["zero_matches"] is not True:
            raise ValueError("Zero-perturbation control failed; risk estimate invalid")
        if row["replay_matches"] is not True:
            raise ValueError("Paired branches did not reconstruct the same starting state")
        groups[row["state_uid"]].append(row)
    result = []
    for uid, rows in sorted(groups.items()):
        effects = np.array([int(r["clean_success"]) - int(r["perturbed_success"]) for r in rows], dtype=float)
        first = rows[0]
        result.append({"state_uid": uid, "suite": first["suite"], "task_id": first["task_id"],
                       "episode": first["episode"], "anchor_step": first["anchor_step"],
                       "trials": len(rows), "p_clean": float(np.mean([r["clean_success"] for r in rows])),
                       "p_perturbed": float(np.mean([r["perturbed_success"] for r in rows])),
                       "control_sensitivity": float(effects.mean()),
                       "mc_standard_error": float(effects.std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None,
                       "all_clean_failed": not any(r["clean_success"] for r in rows)})
    return result


def risk_plots(summary: list[dict], features_path: Path, output: Path):
    """Join independently supplied difficulty/error/importance by exact state UID.

    No automatic invention of semantic difficulty or nearest-frame matching.
    Optional columns: semantic_difficulty, action_error, importance. At least
    one x-variable is required. Missing/extra states fail rather than disappear.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    with Path(features_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    features = {}
    for row in rows:
        uid = row["state_uid"]
        if uid in features:
            raise ValueError("Duplicate feature state_uid")
        features[uid] = row
    if set(features) != {r["state_uid"] for r in summary}:
        raise ValueError("Feature/risk identities differ; explicit one-to-one UID join required")
    columns = [c for c in ("semantic_difficulty", "action_error", "importance") if all(c in r and r[c] != "" for r in rows)]
    if not columns:
        raise ValueError("No complete measured feature column; fill the exported template first")
    output.mkdir(parents=True, exist_ok=False)
    joined = [{**r, **{c: float(features[r["state_uid"]][c]) for c in columns}} for r in summary]
    for c in columns:
        x = np.array([r[c] for r in joined])
        y = np.array([r["control_sensitivity"] for r in joined])
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite feature: {c}")
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        if "importance" in columns and c != "importance":
            points = ax.scatter(x, y, c=[r["importance"] for r in joined])
            fig.colorbar(points, ax=ax, label="Measured importance proxy")
        else:
            ax.scatter(x, y)
        ax.set(xlabel=c.replace("_", " "), ylabel="Paired success-probability drop",
               title="Control sensitivity at predeclared rollout states")
        ax.axhline(0, linestyle="--", linewidth=.7)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(output / f"risk_vs_{c}.{ext}", dpi=220)
        plt.close(fig)
    write_json(output / "joined.json", joined, exclusive=True)
    write_json(output / "provenance.json", {"feature_sha256": digest(features_path),
               "scope": "Associations in sampled rollout states, not causal proof of a curriculum mechanism",
               "floor_states_retained": sum(r["all_clean_failed"] for r in summary),
               "note": "MC standard errors condition on a state; frames are not independent episodes"}, exclusive=True)
