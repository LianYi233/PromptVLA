"""CPU-only paired success and measured execution traces for task7 interventions.

All three conditions use the same trained Time-Input checkpoint. Official task
success is the primary outcome; paths and joint/command plots diagnose behavior.
This reporter never integrates normalized actions into a supposed robot path,
infers grasp/contact from a gripper command, or treats inference seeds as
independent training runs. Inputs are completed, provenance-checked evaluations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


COMPARISON_PROTOCOL = "pi05-task7-closed-loop-comparison-v1"
TRACE_PROTOCOL = "pi05-task7-execution-trace-v1"
PROTOCOL = "pi05-task7-closed-loop-evidence-v1"
MODES = ("correct", "zero", "reversed")
LABELS = {"correct": "Correct time", "zero": "Time branch off", "reversed": "Reversed time"}
COLORS = {"correct": "#218564", "zero": "#818b9a", "reversed": "#d75e32"}
ARTIFACTS = (
    "closed_loop_evidence.json", "closed_loop_paired_outcomes.csv",
    "fig08_closed_loop_success.png", "fig08_closed_loop_success.svg",
    "fig09_execution_trajectories.png", "fig09_execution_trajectories.svg",
    "all_episode_execution_traces.pdf",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON number in {path}: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def _integers(values, label):
    if (not isinstance(values, list) or not values
            or any(type(value) is not int or value < 0 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"Expected distinct nonnegative integers for {label}")
    return values


def _hash(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"Missing/invalid SHA256: {label}")
    return value


def _numeric(value, shape, label):
    import numpy as np

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric array: {label}") from exc
    if array.shape != tuple(shape) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite {tuple(shape)} array: {label}")
    return array


def _validate_state(state, label):
    if not isinstance(state, dict):
        raise ValueError(f"Missing measured simulator state: {label}")
    _numeric(state.get("eef_position"), (3,), label + ".eef_position")
    _numeric(state.get("eef_quaternion"), (4,), label + ".eef_quaternion")
    _numeric(state.get("gripper_qpos"), (2,), label + ".gripper_qpos")
    if state.get("eef_rotation_matrix") is not None:
        _numeric(state["eef_rotation_matrix"], (3, 3), label + ".eef_rotation_matrix")
    if state.get("gripper_qvel") is not None:
        _numeric(state["gripper_qvel"], (2,), label + ".gripper_qvel")
    if state.get("sim_time") is not None:
        if isinstance(state["sim_time"], bool) or not math.isfinite(float(state["sim_time"])):
            raise ValueError(f"Invalid simulator time: {label}")
    objects = state.get("objects")
    if not isinstance(objects, dict):
        raise ValueError(f"Object state availability must be explicit: {label}")
    for name, pose in objects.items():
        if not isinstance(name, str) or not isinstance(pose, dict):
            raise ValueError(f"Invalid object pose: {label}")
        _numeric(pose.get("position"), (3,), f"{label}.objects.{name}.position")
        if pose.get("quaternion") is not None:
            _numeric(pose["quaternion"], (4,), f"{label}.objects.{name}.quaternion")


def validate_trace(trace, *, mode, scene, policy, initial, checkpoint, success, init_seed, num_inference_steps):
    """Validate saved action/state alignment without importing a policy or simulator."""
    import numpy as np

    expected = {
        "protocol": TRACE_PROTOCOL, "status": "completed", "intervention": mode,
        "scene_seed": scene, "policy_seed": policy, "init_state_id": initial,
        "checkpoint": checkpoint, "executed_steps": 10, "horizon": 50,
        "rollout_seed": init_seed + initial, "num_inference_steps": num_inference_steps,
    }
    if not isinstance(trace, dict) or any(trace.get(key) != value for key, value in expected.items()):
        raise ValueError("Trace identity/protocol/checkpoint does not match its comparison cell")
    if type(trace.get("success")) is not bool or trace["success"] is not success:
        raise ValueError("Official success disagrees between trace and evaluation")
    if type(trace.get("policy_rng_seed")) is not int or trace["policy_rng_seed"] < 0:
        raise ValueError("Missing episode policy RNG provenance")
    _validate_state(trace.get("initial_state"), "initial_state")
    _hash(trace["initial_state"].get("simulator_state_sha256"), "initial simulator state")
    steps, chunks = trace.get("steps"), trace.get("chunks")
    if not isinstance(steps, list) or not steps or not isinstance(chunks, list):
        raise ValueError("A completed trace needs nonempty executed steps and sampling chunks")
    if len(chunks) != math.ceil(len(steps) / 10):
        raise ValueError("Sampling chunk count does not match the executed prefix cadence")
    inference_steps = set()
    for index, chunk in enumerate(chunks):
        if chunk.get("chunk_index") != index or chunk.get("start_step") != index * 10:
            raise ValueError("Sampling chunks are not contiguous at the declared execution cadence")
        _hash(chunk.get("initial_noise_sha256"), "initial sampling noise")
        _hash(chunk.get("observation_sha256"), "sampling observation")
        count = chunk.get("denoising_calls")
        if type(count) is not int or count != num_inference_steps:
            raise ValueError("Native denoising count differs from the declared checkpoint configuration")
        if chunk.get("starts_from_native_pure_noise") is not True:
            raise ValueError("Missing audit of native pure-noise sampling")
        schedule = _numeric(chunk.get("flow_times"), (count,), "flow_times")
        if not np.allclose(schedule, 1 - np.arange(count) / count, rtol=0, atol=1e-6):
            raise ValueError("Trace does not contain the complete native denoising schedule")
        inference_steps.add(count)
        _numeric(chunk.get("actions"), (50, 7), "normalized generated chunk")
    if len(inference_steps) != 1:
        raise ValueError("Native inference step count changed within an episode")
    for index, step in enumerate(steps):
        if (step.get("step") != index or step.get("chunk_index") != index // 10
                or step.get("chunk_step") != index % 10):
            raise ValueError("Executed actions are not aligned to their generated chunks")
        _numeric(step.get("executed_action"), (7,), "postprocessed executed action")
        normalized = _numeric(step.get("normalized_action"), (7,), "normalized executed action")
        prediction = chunks[index // 10]["actions"][index % 10]
        if not np.allclose(normalized, prediction, rtol=0, atol=1e-7):
            raise ValueError("Recorded normalized action does not match the generated chunk prefix")
        _validate_state(step.get("state_after"), f"step {index} state_after")
        if any(type(step.get(key)) is not bool for key in ("success", "terminated", "truncated")):
            raise ValueError("Every executed step requires boolean success/termination flags")
        if index < len(steps) - 1 and (step["terminated"] or step["truncated"] or step["success"]):
            raise ValueError("Trace contains actions after episode termination or success")
    if any(step["success"] for step in steps) is not success:
        raise ValueError("Official episode success disagrees with executed-step flags")
    if not (steps[-1]["terminated"] or steps[-1]["truncated"]):
        raise ValueError("Completed trace must end in native termination or truncation")


def _paired_audit(traces):
    """Only the initial observation is shared; later closed-loop observations may diverge."""
    for label, values in (
        ("initial simulator states", [value["initial_state"]["simulator_state_sha256"] for value in traces.values()]),
        ("initial observations", [value["chunks"][0]["observation_sha256"] for value in traces.values()]),
        ("episode policy RNG seeds", [value["policy_rng_seed"] for value in traces.values()]),
    ):
        if len(set(values)) != 1:
            raise ValueError(f"Paired interventions do not share {label}")
    initial_states = [value["initial_state"] for value in traces.values()]
    for name in ("eef_position", "eef_quaternion", "eef_rotation_matrix", "gripper_qpos",
                 "gripper_qvel", "objects", "controller_state"):
        values = [state.get(name) for state in initial_states]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Paired initial measured/controller state differs: {name}")
    lengths = [len(value["chunks"]) for value in traces.values()]
    shared_chunks = 0
    for index in range(max(lengths)):
        rows = [value["chunks"][index] for value in traces.values() if index < len(value["chunks"])]
        if len({row["denoising_calls"] for row in rows}) != 1:
            raise ValueError("Native inference step count differs across interventions")
        if len({row["initial_noise_sha256"] for row in rows}) != 1:
            raise ValueError(f"Paired sampling noise differs at shared chunk {index}")
        if len(rows) > 1:
            shared_chunks += 1
    return {
        "initial_simulator_state_equal": True,
        "initial_measured_geometry_equal": True,
        "initial_controller_state_equal": (True if initial_states[0].get("controller_state") is not None else None),
        "initial_observation_equal": True,
        "episode_policy_rng_equal": True,
        "matched_shared_chunk_indices": shared_chunks,
        "later_observations_required_equal": False,
    }


def load_comparison(path):
    """Require a complete predeclared grid and audit every referenced episode trace."""
    path = Path(path).expanduser().resolve(strict=True)
    report = _read_json(path)
    if (report.get("protocol") != COMPARISON_PROTOCOL or report.get("status") != "completed"):
        raise ValueError("Expected a completed same-checkpoint closed-loop comparison")
    settings = report.get("settings", {})
    scenes = _integers(settings.get("scene_seeds"), "scene_seeds")
    policies = _integers(settings.get("policy_seeds", settings.get("test_policy_seeds")), "policy_seeds")
    if (settings.get("fixed_init_states") != list(range(10))
            or settings.get("executed_steps") != 10 or settings.get("horizon") != 50):
        raise ValueError("Comparison must use fixed initial states 0..9, chunk50 and execution prefix10")
    checkpoint = settings.get("checkpoint")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
        raise ValueError("Missing common absolute checkpoint path")
    init_seed = settings.get("init_seed")
    if type(init_seed) is not int or init_seed < 0:
        raise ValueError("Missing initial-state rollout seed origin")
    num_inference_steps = settings.get("num_inference_steps")
    if type(num_inference_steps) is not int or num_inference_steps < 1:
        raise ValueError("Missing native num_inference_steps from the checkpoint configuration")
    cells = report.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("No paired closed-loop cells")
    expected = [(scene, policy) for scene in scenes for policy in policies]
    by_identity = {}
    for cell in cells:
        key = (cell.get("scene_seed"), cell.get("policy_seed"))
        if (any(type(value) is not int for value in key) or key not in expected or key in by_identity):
            raise ValueError("Duplicate, invalid or undeclared scene/policy cell")
        if set(cell.get("results", {})) != set(MODES):
            raise ValueError("Each cell requires Correct, Zero and Reversed from one checkpoint")
        for mode, result in cell["results"].items():
            flags, traces = result.get("successes"), result.get("traces")
            if (not isinstance(flags, list) or len(flags) != 10
                    or any(type(flag) is not bool for flag in flags)
                    or not isinstance(traces, list) or len(traces) != 10
                    or not isinstance(result.get("eval_info"), str) or not result["eval_info"]):
                raise ValueError(f"Incomplete official outcome/trace provenance for {mode}")
        by_identity[key] = cell
    if set(by_identity) != set(expected):
        raise ValueError("Incomplete scene x policy grid")
    episodes, used_paths, trace_sources, source_fingerprints = [], set(), [], []
    for scene, policy in expected:
        cell = by_identity[(scene, policy)]
        for initial in range(10):
            traces, paths = {}, {}
            for mode in MODES:
                result = cell["results"][mode]
                source = result["traces"][initial]
                if not isinstance(source, str) or not Path(source).is_absolute():
                    raise ValueError("Episode trace paths must be explicit absolute paths")
                trace_path = Path(source).resolve(strict=True)
                if trace_path in used_paths:
                    raise ValueError("A saved trace was reused for more than one evaluation identity")
                used_paths.add(trace_path)
                trace = _read_json(trace_path)
                validate_trace(
                    trace, mode=mode, scene=scene, policy=policy, initial=initial,
                    checkpoint=checkpoint, success=result["successes"][initial], init_seed=init_seed,
                    num_inference_steps=num_inference_steps,
                )
                traces[mode], paths[mode] = trace, str(trace_path)
                trace_sources.append({"path": str(trace_path), "sha256": _sha256(trace_path)})
                source_fingerprints.append(trace.get("source_fingerprints"))
            episodes.append({
                "scene_seed": scene, "policy_seed": policy, "init_state_id": initial,
                "traces": traces, "trace_paths": paths, "audit": _paired_audit(traces),
            })
    if any(value != source_fingerprints[0] for value in source_fingerprints[1:]):
        raise ValueError("Source implementation fingerprints differ across saved traces")
    return {
        "comparison": str(path), "comparison_sha256": _sha256(path),
        "settings": settings, "episodes": episodes, "trace_sources": trace_sources,
        "cell_order": expected,
        "runtime_source_fingerprints": source_fingerprints[0],
    }


def summarize_outcomes(episodes):
    if not episodes:
        raise ValueError("Cannot summarize an empty episode grid")
    flags = {mode: [row["traces"][mode]["success"] for row in episodes] for mode in MODES}
    count = len(episodes)
    paired = {}
    for mode in ("zero", "reversed"):
        pairs = list(zip(flags["correct"], flags[mode], strict=True))
        correct_only = sum(a and not b for a, b in pairs)
        other_only = sum(b and not a for a, b in pairs)
        paired[f"correct_vs_{mode}"] = {
            "both_success": sum(a and b for a, b in pairs),
            "both_failure": sum(not a and not b for a, b in pairs),
            "correct_only_wins": correct_only, "intervention_only_wins": other_only,
            "net_correct_wins": correct_only - other_only,
            "success_rate_difference_pp": 100 * (correct_only - other_only) / count,
            "episodes": count,
        }
    return {
        "totals": {mode: {"successes": sum(values), "episodes": count,
                          "success_rate": sum(values) / count} for mode, values in flags.items()},
        "paired": paired,
    }


def _states(trace):
    return [trace["initial_state"], *(step["state_after"] for step in trace["steps"])]


def gripper_sign_transitions(trace):
    """Zero crossings of the executed command; neither physical closure nor contact."""
    previous = 0
    result = []
    for index, step in enumerate(trace["steps"]):
        value = step["executed_action"][6]
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if sign and previous and sign != previous:
            result.append(index)
        if sign:
            previous = sign
    return result


def summarize_trace(trace):
    import numpy as np

    states = _states(trace)
    position = np.asarray([state["eef_position"] for state in states], dtype=np.float64)
    gripper = np.asarray([state["gripper_qpos"] for state in states], dtype=np.float64)
    names = sorted({name for state in states for name in state["objects"]})
    rotation_sources = {}
    for state in states:
        if state.get("eef_rotation_matrix") is not None:
            source = state.get("eef_rotation_source", "unspecified in saved trace")
            rotation_sources[source] = rotation_sources.get(source, 0) + 1
    objects = {}
    for name in names:
        samples = [(index, state["objects"][name]["position"])
                   for index, state in enumerate(states) if name in state["objects"]]
        values = np.asarray([value for _, value in samples], dtype=np.float64)
        # A late-appearing object's first sample is not silently relabelled as initial state.
        initial = trace["initial_state"]["objects"].get(name)
        delta = None if initial is None else values - np.asarray(initial["position"])
        objects[name] = {
            "available_state_samples": len(samples), "first_sample_index": samples[0][0],
            "last_sample_index": samples[-1][0],
            "max_height_change_from_initial": None if delta is None else float(delta[:, 2].max()),
            "last_position_change_from_initial": None if delta is None else delta[-1].tolist(),
        }
    return {
        "executed_steps": len(trace["steps"]), "sampled_chunks": len(trace["chunks"]),
        "official_success": trace["success"],
        "gripper_command_sign_transition_steps": gripper_sign_transitions(trace),
        "eef_measured_path_length": float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()),
        "eef_final_minus_initial": (position[-1] - position[0]).tolist(),
        "gripper_joint_final_minus_initial": (gripper[-1] - gripper[0]).tolist(),
        "orientation_matrix_samples": sum(state.get("eef_rotation_matrix") is not None for state in states),
        "orientation_matrix_sources": rotation_sources,
        "contact_samples_available": sum(state.get("contact_pairs") is not None for state in states),
        "objects": objects,
        "state_conventions": trace.get("state_conventions", {}),
        "note": "Geometry and command transitions describe behavior; they are not task-quality or grasp labels.",
    }


def summarize_comparison(loaded):
    episodes, settings = loaded["episodes"], loaded["settings"]
    scenes = settings["scene_seeds"]
    policies = settings.get("policy_seeds", settings.get("test_policy_seeds"))
    summaries = []
    for row in episodes:
        summaries.append({
            **{key: row[key] for key in ("scene_seed", "policy_seed", "init_state_id")},
            "audit": row["audit"], "trace_paths": row["trace_paths"],
            "conditions": {mode: summarize_trace(row["traces"][mode]) for mode in MODES},
        })
    return {
        "protocol": PROTOCOL, "source_protocol": COMPARISON_PROTOCOL,
        "comparison": loaded["comparison"], "comparison_sha256": loaded["comparison_sha256"],
        "source_trace_hashes": loaded["trace_sources"], "settings": settings,
        "runtime_source_fingerprints": loaded["runtime_source_fingerprints"],
        "conditions": list(MODES), "primary_contrast": "correct_vs_zero",
        "auxiliary_contrast": "correct_vs_reversed", "summary": summarize_outcomes(episodes),
        "by_scene": {str(scene): summarize_outcomes([row for row in episodes if row["scene_seed"] == scene])
                     for scene in scenes},
        "by_policy_seed": {str(policy): summarize_outcomes([row for row in episodes if row["policy_seed"] == policy])
                           for policy in policies},
        "by_cell": [{"scene_seed": scene, "policy_seed": policy,
                     **summarize_outcomes([row for row in episodes
                                          if (row["scene_seed"], row["policy_seed"]) == (scene, policy)])}
                    for scene, policy in loaded["cell_order"]],
        "episodes": summaries,
        "representative_episode": {key: episodes[0][key] for key in ("scene_seed", "policy_seed", "init_state_id")},
        "representative_selection": "First declared scene, first declared policy seed, fixed init state 0; independent of outcomes.",
        "all_episode_page_order": [{"page": index + 1, **{key: row[key] for key in
                                     ("scene_seed", "policy_seed", "init_state_id")}}
                                   for index, row in enumerate(episodes)],
        "notes": [
            "Official closed-loop task success is primary. Paths use measured simulator states, never cumulative normalized actions.",
            "Three conditions share the checkpoint, initial simulator state and matched sampling-noise stream. Subsequent observations may diverge.",
            "Differences are descriptive for the fixed scene/policy/init grid; no interval assumes all episodes are independent training runs.",
            "Policy seeds are inference RNG draws. Repeat is a run namespace, not an independent training seed.",
            "Zero disables only the added time branch. Native RoPE remains present in all conditions.",
            "Command sign transitions are not labelled open/close, contact, grasp or release. Missing contact/object observations remain missing.",
            "A longer/shorter path or closer command match alone does not establish better coordination; inspect official outcomes and state/event traces together.",
        ],
    }


def _plot_modules():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, output, stem):
    paths = []
    for extension in ("png", "svg"):
        path = Path(output) / f"{stem}.{extension}"
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{stem}.", suffix=f".{extension}")
        os.close(descriptor)
        try:
            fig.savefig(temporary, format=extension, dpi=160, facecolor="white",
                        bbox_inches="tight", pad_inches=.12)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        paths.append(str(path))
    return paths


def plot_success(loaded, summary, output):
    import numpy as np
    from matplotlib.colors import ListedColormap

    plt = _plot_modules()
    episodes = loaded["episodes"]
    matrix = np.asarray([[row["traces"][mode]["success"] for row in episodes] for mode in MODES])
    fig = plt.figure(figsize=(15, 9.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.3))
    ax = fig.add_subplot(grid[0, :])
    ax.imshow(matrix, cmap=ListedColormap(["#e4e8ee", "#218564"]), vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    ax.set_yticks(range(3), [LABELS[mode] for mode in MODES])
    cells = loaded["cell_order"]
    ax.set_xticks([index * 10 + 4.5 for index in range(len(cells))],
                  [f"Scene {scene}\nPolicy {policy}" for scene, policy in cells], fontsize=9)
    for index in range(1, len(cells)):
        ax.axvline(index * 10 - .5, color="white", lw=2)
    for index, mode in enumerate(MODES):
        value = summary["summary"]["totals"][mode]
        ax.text(len(episodes) + .5, index,
                f"{value['successes']}/{value['episodes']} = {100 * value['success_rate']:.1f}%", va="center")
    ax.set_xlim(-.5, len(episodes) + max(8, len(episodes) * .2))
    ax.set_xlabel("Every cell contains fixed initial states 0–9; green = success, gray = failure")
    ax.set_title("A. Official task outcomes from three closed-loop rollouts per initial condition")
    ax = fig.add_subplot(grid[1, 0])
    other_modes = ("zero", "reversed")
    pairs = [summary["summary"]["paired"][f"correct_vs_{mode}"] for mode in other_modes]
    x = np.arange(2)
    wins = [row["correct_only_wins"] for row in pairs]
    losses = [-row["intervention_only_wins"] for row in pairs]
    ax.bar(x, wins, color=COLORS["correct"], label="Only Correct succeeds")
    ax.bar(x, losses, color="#b95b5b", label="Only intervention succeeds")
    for index, (win, loss, pair) in enumerate(zip(wins, losses, pairs, strict=True)):
        ax.annotate(f"+{win}", (index, win), xytext=(0, 5), textcoords="offset points", ha="center")
        ax.annotate(str(loss), (index, loss), xytext=(0, -14), textcoords="offset points", ha="center")
        ax.text(index, .98, f"Net Correct wins: {pair['net_correct_wins']:+d}",
                ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=10)
    ax.axhline(0, color="#334155", lw=1)
    ax.set_xticks(x, ["Correct vs Zero\n(primary)", "Correct vs Reversed\n(auxiliary)"])
    ax.set_ylabel("Paired episodes with different outcomes")
    ax.set_title("B. Gains and losses are both retained")
    ax.margins(y=.3)
    ax.legend(loc="lower left", fontsize=9)
    ax = fig.add_subplot(grid[1, 1])
    scenes = loaded["settings"]["scene_seeds"]
    positions = np.arange(len(scenes))
    for index, mode in enumerate(other_modes):
        values = [summary["by_scene"][str(scene)]["paired"][f"correct_vs_{mode}"]["success_rate_difference_pp"]
                  for scene in scenes]
        offset = positions + (index - .5) * .32
        ax.bar(offset, values, width=.30, color=COLORS[mode], label=f"Correct minus {mode.title()}")
        for xpos, value in zip(offset, values, strict=True):
            ax.annotate(f"{value:+.1f}", (xpos, value), xytext=(0, 5 if value >= 0 else -14),
                        textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(0, color="#334155", lw=1)
    ax.set_xticks(positions, [str(scene) for scene in scenes])
    ax.set_ylabel("Success-rate difference (percentage points)")
    ax.set_xlabel("Scene seed")
    ax.set_title("C. Positive values favor Correct time")
    ax.margins(y=.25)
    ax.legend(fontsize=9)
    title = "Does the added time branch improve actual task completion?"
    if loaded["settings"].get("synthetic_fixture"):
        title = "SYNTHETIC TEST FIXTURE — " + title
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.supxlabel("Same checkpoint and native sampler. Fixed-grid descriptive outcomes; inference seeds are not training seeds.", fontsize=10)
    paths = _save_figure(fig, output, "fig08_closed_loop_success")
    plt.close(fig)
    return paths


def trajectory_figure(episode, *, synthetic=False):
    import numpy as np
    from matplotlib.lines import Line2D

    plt = _plot_modules()
    fig = plt.figure(figsize=(17, 11), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.4, 1))
    all_positions = []
    for trace in episode["traces"].values():
        all_positions.extend(state["eef_position"] for state in _states(trace))
        all_positions.extend(pose["position"] for pose in trace["initial_state"]["objects"].values())
    points = np.asarray(all_positions)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) * .57, .025)
    for column, mode in enumerate(MODES):
        trace = episode["traces"][mode]
        states = _states(trace)
        position = np.asarray([state["eef_position"] for state in states])
        switches = gripper_sign_transitions(trace)
        ax = fig.add_subplot(grid[0, column], projection="3d")
        ax.plot(*position.T, color=COLORS[mode], lw=1.9)
        ax.scatter(*position[0], color=COLORS[mode], marker="o", s=42)
        ax.scatter(*position[-1], color=COLORS[mode], marker="x", s=52)
        if switches:
            # A command at step k is issued from state k, before its resulting state k+1.
            ax.scatter(*position[switches].T, color=COLORS[mode], marker="D", s=34)
        for name, pose in sorted(trace["initial_state"]["objects"].items()):
            point = np.asarray(pose["position"])
            ax.scatter(*point, color="#9399a2", marker=".", s=35)
            if column == 0:
                ax.text(*point, name, color="#5a6069", fontsize=7)
        keyframes = sorted({0, len(states) - 1, *switches[:3]})
        for index in keyframes:
            rotation = states[index].get("eef_rotation_matrix")
            if rotation is None:
                continue
            matrix = np.asarray(rotation)
            for axis, color in enumerate(("#c34848", "#3a9946", "#3674be")):
                direction = matrix[:, axis] * radius * .17
                ax.quiver(*position[index], *direction, color=color, lw=.9, arrow_length_ratio=.2)
        for setter, dimension in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setter(center[dimension] - radius, center[dimension] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("World x", labelpad=2)
        ax.set_ylabel("World y", labelpad=2)
        ax.set_zlabel("World z", labelpad=2)
        ax.tick_params(labelsize=8, pad=1)
        outcome = "SUCCESS" if trace["success"] else "FAILURE"
        ax.set_title(f"{LABELS[mode]}: {outcome}\n{len(trace['steps'])} actual executed steps", fontsize=12)
    command_ax = fig.add_subplot(grid[1, 0])
    joint_ax = fig.add_subplot(grid[1, 1])
    object_ax = fig.add_subplot(grid[1, 2])
    object_names = sorted({name for trace in episode["traces"].values()
                           for name in trace["initial_state"]["objects"]})
    styles = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1, 3, 1)))
    for mode in MODES:
        trace = episode["traces"][mode]
        states = _states(trace)
        commands = np.asarray([step["executed_action"] for step in trace["steps"]])
        joints = np.asarray([state["gripper_qpos"] for state in states])
        command_ax.step(np.arange(len(commands)), commands[:, 6], where="post", color=COLORS[mode],
                        label=LABELS[mode], lw=1.6)
        for step in gripper_sign_transitions(trace):
            command_ax.plot(step, commands[step, 6], marker="D", color=COLORS[mode], ms=4)
        joint_ax.plot(joints[:, 0], color=COLORS[mode], ls="-", lw=1.4)
        joint_ax.plot(joints[:, 1], color=COLORS[mode], ls="--", lw=1.4)
        for index, name in enumerate(object_names):
            initial_pose = trace["initial_state"]["objects"].get(name)
            if initial_pose is None:
                continue
            heights = [state["objects"][name]["position"][2] - initial_pose["position"][2]
                       if name in state["objects"] else np.nan for state in states]
            object_ax.plot(heights, color=COLORS[mode], ls=styles[index % len(styles)], lw=1.3)
    command_ax.set_title("Executed gripper command (dimension 6)")
    command_ax.set_ylabel("Postprocessed command; sign meaning not assigned")
    command_ax.legend(fontsize=8)
    joint_ax.set_title("Measured gripper joint state")
    joint_ax.set_ylabel("Joint position (simulator units)")
    joint_ax.legend(handles=[Line2D([], [], color="#334155", ls="-", label="joint 0"),
                             Line2D([], [], color="#334155", ls="--", label="joint 1")], fontsize=8)
    object_ax.set_title("Measured object height change")
    object_ax.set_ylabel("World z minus initial z (simulator units)")
    if object_names:
        object_ax.legend(handles=[Line2D([], [], color="#334155", ls=styles[index % len(styles)], label=name)
                                  for index, name in enumerate(object_names)], fontsize=7, loc="best")
    else:
        object_ax.text(.5, .5, "Object poses unavailable\nNo grasp/lift labels inferred", ha="center", va="center",
                       transform=object_ax.transAxes)
    for ax in (command_ax, joint_ax, object_ax):
        ax.set_xlabel("Executed control step (state curves include initial state 0)")
        ax.grid(alpha=.18)
    title = (f"Measured execution: scene {episode['scene_seed']}, policy {episode['policy_seed']}, "
             f"fixed init {episode['init_state_id']}")
    if synthetic:
        title = "SYNTHETIC TEST FIXTURE — " + title
    fig.suptitle(title, fontsize=16, fontweight="bold")
    cached_orientation = any("cached" in state.get("eef_rotation_source", "")
                             for trace in episode["traces"].values() for state in _states(trace))
    orientation_note = ("RGB arrows show logged EEF orientation; controller-cache fallback used (see JSON)."
                        if cached_orientation else "RGB arrows show measured EEF orientation at keyframes.")
    fig.supxlabel(
        "Top: measured EEF paths; circle = start, x = final, diamond = command sign transition; "
        + orientation_note + "\n"
        "Gray points mark initial object positions. Same initial state and sampling-noise schedule; later observations may diverge.\n"
        "A command sign transition does not establish physical closure, contact, grasp or release. Curves end when each rollout ends.",
        fontsize=9,
    )
    return fig


def plot_trajectories(loaded, output):
    from matplotlib.backends.backend_pdf import PdfPages

    plt = _plot_modules()
    output = Path(output)
    synthetic = bool(loaded["settings"].get("synthetic_fixture"))
    paths = []
    target = output / "all_episode_execution_traces.pdf"
    descriptor, temporary = tempfile.mkstemp(dir=output, prefix=".all-traces.", suffix=".pdf")
    os.close(descriptor)
    try:
        with PdfPages(temporary, metadata={"Title": "All predeclared task7 paired execution traces"}) as pages:
            for index, episode in enumerate(loaded["episodes"]):
                fig = trajectory_figure(episode, synthetic=synthetic)
                if index == 0:
                    paths.extend(_save_figure(fig, output, "fig09_execution_trajectories"))
                pages.savefig(fig, facecolor="white", bbox_inches="tight", pad_inches=.12)
                plt.close(fig)
                if (index + 1) % 10 == 0 or index + 1 == len(loaded["episodes"]):
                    print(f"Rendered paired execution trace pages: {index + 1}/{len(loaded['episodes'])}", flush=True)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return [*paths, str(target)]


def _write_outcomes_csv(episodes, output):
    target = Path(output) / "closed_loop_paired_outcomes.csv"
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".paired-outcomes.")
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "scene_seed", "policy_seed", "init_state_id", *[f"{mode}_success" for mode in MODES],
                "correct_minus_zero", "correct_minus_reversed", *[f"{mode}_executed_steps" for mode in MODES],
            ])
            for episode in episodes:
                traces = episode["traces"]
                values = {mode: int(traces[mode]["success"]) for mode in MODES}
                writer.writerow([
                    episode["scene_seed"], episode["policy_seed"], episode["init_state_id"],
                    *[values[mode] for mode in MODES], values["correct"] - values["zero"],
                    values["correct"] - values["reversed"], *[len(traces[mode]["steps"]) for mode in MODES],
                ])
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(target)


def run_report(comparison_path, output):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "closed_loop_evidence.json"
    state = {"protocol": PROTOCOL, "status": "running", "comparison": str(comparison_path)}
    _atomic_json(report_path, state)
    try:
        loaded = load_comparison(comparison_path)
        result = summarize_comparison(loaded)
        result["outcome_csv"] = _write_outcomes_csv(loaded["episodes"], output)
        result["figure_paths"] = [*plot_success(loaded, result, output), *plot_trajectories(loaded, output)]
        result["status"] = "completed"
        _atomic_json(report_path, result)
    except BaseException as exc:
        _atomic_json(report_path, {**state, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    print(f"Saved paired closed-loop evidence: {report_path}", flush=True)
    for key, value in result["summary"]["paired"].items():
        print(f"{key}: {value}", flush=True)
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print inputs/outputs without reading or writing files.")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        print(f"CPU-only report: {args.comparison_json}; output={args.output}")
        print("Require all three conditions on the declared grid, matched initial states and shared chunk noises.")
        print("Artifacts: " + ", ".join(ARTIFACTS))
        return
    try:
        run_report(args.comparison_json, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Task7 closed-loop report failed: {exc}\n")


if __name__ == "__main__":
    main()
