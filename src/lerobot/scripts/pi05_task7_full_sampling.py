"""Paired Correct/Zero/Reversed PI0.5 chunks generated from pure Gaussian noise.

One trained Time-Input checkpoint supplies all three arms. Recorded future
actions are normalized separately for scoring and never enter inference.
The native predict_action_chunk sampler and checkpoint denoising count are
preserved. This offline diagnostic does not execute actions in a simulator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


PROTOCOL = "pi05-task7-full-sampling-v1"
NOISE_PROTOCOL = "task7-full-sampling-gaussian-cpu-v1"
HORIZON = 50
ACTION_DIM = 7
MODES = ("correct", "zero", "reversed")
DEFAULT_ROOT = Path("/data1/wyn/chkpt/2601-lerobot/PromptVLA")
DEFAULT_PLAN = Path("/data1/wyn/prompt-work/gripper-event-task7-1200/base")


def tensor_sha256(value):
    """Hash dtype, shape and bytes, including bfloat16 tensors."""
    import torch

    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}:".encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def observation_sha256(batch):
    import torch

    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        digest.update(key.encode())
        digest.update((tensor_sha256(value) if torch.is_tensor(value)
                       else json.dumps(value, sort_keys=True, ensure_ascii=False)).encode())
    return digest.hexdigest()


def observation_only(item):
    """Whitelist causal deployment inputs before preprocessing any labels."""
    return {key: copy.deepcopy(value) for key, value in item.items()
            if key.startswith("observation.") or key == "task"}


def stable_sampling_noise(seed, frame_index, shape):
    import torch

    payload = f"{NOISE_PROTOCOL}:{int(seed)}:{int(frame_index)}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(derived)
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32, device="cpu")


def _field(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def validate_sampling_config(config):
    for key, expected in (("chunk_size", HORIZON), ("action_time_encoding", True),
                          ("action_time_injection", "input"), ("action_time_basis", "index")):
        if _field(config, key, "index" if key == "action_time_basis" else None) != expected:
            raise ValueError(f"Full sampling requires {key}={expected!r}")
    if _field(config, "training_stage", "flow") != "flow":
        raise ValueError("Full sampling requires a flow-stage checkpoint")
    steps = _field(config, "num_inference_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Checkpoint num_inference_steps must be a positive integer")
    width = _field(config, "max_action_dim")
    if isinstance(width, bool) or not isinstance(width, int) or width < ACTION_DIM:
        raise ValueError("Checkpoint max_action_dim must cover the seven action dimensions")
    feature = _field(config, "output_features", {}).get("action")
    if tuple(_field(feature, "shape", ())) != (ACTION_DIM,):
        raise ValueError("Full sampling expects seven native LIBERO action dimensions")
    rtc = _field(config, "rtc_config")
    if rtc is not None and _field(rtc, "enabled", False):
        raise ValueError("RTC must be disabled for independent pure-noise chunks")


def _sample_counterfactual(policy, observation_batch, noise, intervention):
    """Call the native sampler, auditing its initial state and complete schedule."""
    import torch
    from lerobot.scripts.pi05_task7_time_evidence import time_counterfactual

    if intervention not in MODES:
        raise ValueError(f"Unknown intervention: {intervention}")
    validate_sampling_config(policy.config)
    if any(not (key.startswith("observation.") or key == "task") for key in observation_batch):
        raise ValueError("Inference batch must contain only observations/task, never future action labels")
    if _field(policy.config, "compile_model", False):
        raise ValueError("Disable compile_model before applying temporary time interventions")
    if callable(getattr(policy, "_rtc_enabled", None)) and policy._rtc_enabled():
        raise ValueError("RTC must be disabled for independent pure-noise chunks")
    expected_shape = (1, HORIZON, policy.config.max_action_dim)
    if tuple(noise.shape) != expected_shape or not torch.isfinite(noise).all():
        raise ValueError(f"Initial noise must be finite with shape {expected_shape}")
    encoder = getattr(policy.model, "action_time_encoder", None)
    if encoder is None:
        raise ValueError("A trained action-time encoder is required in every arm")

    policy.eval()
    inputs = copy.deepcopy(observation_batch)
    initial = noise.detach().clone()
    noise_hash = tensor_sha256(initial)
    input_hash = observation_sha256(inputs)
    model = policy.model
    original = model.denoise_step
    had_override = "denoise_step" in model.__dict__
    times = []

    def audited_denoise(*call_args, **call_kwargs):
        x_t = call_kwargs.get("x_t")
        time = call_kwargs.get("timestep")
        if x_t is None or time is None:
            raise RuntimeError("Native denoise_step contract changed; cannot audit pure-noise sampling")
        if not times and not torch.equal(x_t, initial):
            raise RuntimeError("Native sampler did not start from the supplied pure noise")
        flat_time = time.detach().float().reshape(-1)
        if flat_time.numel() != 1:
            raise RuntimeError("Full sampling audit expects exactly one observation")
        times.append(float(flat_time[0]))
        return original(*call_args, **call_kwargs)

    try:
        model.denoise_step = audited_denoise
        with torch.inference_mode(), time_counterfactual(encoder, intervention):
            prediction = policy.predict_action_chunk(inputs, noise=initial.clone())
    finally:
        if had_override:
            model.denoise_step = original
        else:
            delattr(model, "denoise_step")
    steps = policy.config.num_inference_steps
    expected_times = [1.0 - index / steps for index in range(steps)]
    if len(times) != steps or any(not math.isclose(a, b, abs_tol=1e-6)
                                  for a, b in zip(times, expected_times, strict=True)):
        raise RuntimeError("Native full denoising schedule differs from checkpoint num_inference_steps")
    if tuple(prediction.shape) != (1, HORIZON, ACTION_DIM) or not torch.isfinite(prediction).all():
        raise ValueError("Native sampler must return finite normalized actions of shape [1, 50, 7]")
    if tensor_sha256(noise) != noise_hash or observation_sha256(observation_batch) != input_hash:
        raise RuntimeError("A paired sampling input was modified")
    return prediction.detach().clone(), {
        "intervention": intervention, "initial_noise_sha256": noise_hash,
        "observation_sha256": input_hash, "denoising_calls": len(times),
        "flow_times": times, "starts_from_supplied_pure_noise": True,
    }


def _make_batches(dataset, actions, frame_index, processor):
    """Preprocess observations alone, then separately normalize scoring targets."""
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate
    from lerobot.utils.collate import lerobot_collate_fn

    item = dict(dataset[int(frame_index)])
    if int(item["index"]) != int(frame_index):
        raise ValueError("Dataset frame identity changed")
    causal = observation_only(item)
    for key in dataset.meta.camera_keys:
        if key in causal and causal[key].dtype == torch.uint8:
            causal[key] = causal[key].float() / 255.0
    collate = lerobot_collate_fn if dataset.meta.has_language_columns else default_collate
    # No action, padding mask, episode clock or future label reaches this call.
    inputs = observation_only(processor(collate([copy.deepcopy(causal)])))
    targets = np.asarray(actions[frame_index:frame_index + HORIZON], dtype=np.float32)
    if targets.shape != (HORIZON, ACTION_DIM) or not np.isfinite(targets).all():
        raise ValueError("Scoring requires a complete finite 50 x 7 recorded action window")
    scoring = copy.deepcopy(causal)
    scoring["action"] = torch.from_numpy(targets.copy())
    scoring["action_is_pad"] = torch.zeros(HORIZON, dtype=torch.bool)
    normalized = processor(collate([scoring]))["action"]
    if tuple(normalized.shape[:2]) != (1, HORIZON) or normalized.shape[-1] < ACTION_DIM:
        raise ValueError("Checkpoint processor changed the target horizon or action dimensions")
    normalized = normalized[0, :, :ACTION_DIM].detach().float().cpu().numpy()
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized scoring targets are non-finite")
    return inputs, normalized


def _paired_interval(values):
    """Descriptive paired bootstrap over demonstrations, not noise repetitions."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return None
    indices = np.random.default_rng(20261105).integers(0, len(values), size=(4000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def summarize_predictions(predictions, targets, conditions, executed_steps=10):
    import numpy as np
    from lerobot.scripts.pi05_task7_time_evidence import episode_means

    targets = np.asarray(targets, dtype=np.float64)
    if not 1 <= executed_steps <= HORIZON:
        raise ValueError("executed_steps must be in [1, 50]")
    if not len(conditions) or targets.shape != (len(conditions), HORIZON, ACTION_DIM):
        raise ValueError("Targets must match all declared conditions with shape [N, 50, 7]")
    identities = [(row["episode_index"], row["global_frame_index"], row["noise_seed"])
                  for row in conditions]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate sampling condition")
    if set(predictions) != set(MODES) or not np.isfinite(targets).all():
        raise ValueError("Expected all three interventions and finite targets")
    squared = {}
    for mode in MODES:
        value = np.asarray(predictions[mode], dtype=np.float64)
        if value.shape != targets.shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid sampled actions for {mode}")
        squared[mode] = (value - targets) ** 2
    dimensions = {"all": list(range(ACTION_DIM)), "arm": list(range(6)), "gripper": [6]}
    regions = {"executed_prefix": [0, executed_steps], "full_chunk": [0, HORIZON]}
    metrics = {}
    episode_ids = None
    for region, (start, stop) in regions.items():
        metrics[region] = {}
        for name, dims in dimensions.items():
            per_demo = {}
            for mode in MODES:
                rows = squared[mode][:, start:stop, :][:, :, dims].mean(axis=(1, 2))
                episode_ids, values = episode_means(rows, conditions)
                per_demo[mode] = values
            paired = {f"{mode}_minus_correct": per_demo[mode] - per_demo["correct"]
                      for mode in ("zero", "reversed")}
            output_rms = {}
            for mode in ("zero", "reversed"):
                change = np.asarray(predictions[mode]) - np.asarray(predictions["correct"])
                rows = np.sqrt((change[:, start:stop, :][:, :, dims] ** 2).mean(axis=(1, 2)))
                _, means = episode_means(rows, conditions)
                output_rms[mode] = float(means.mean())
            metrics[region][name] = {
                "mean_mse": {mode: float(value.mean()) for mode, value in per_demo.items()},
                "per_demo_mse": {mode: value.tolist() for mode, value in per_demo.items()},
                "paired_deltas": {key: value.tolist() for key, value in paired.items()},
                "mean_paired_deltas": {key: float(value.mean()) for key, value in paired.items()},
                "paired_bootstrap_95pct": {key: _paired_interval(value) for key, value in paired.items()},
                "mean_output_change_rms": output_rms,
            }
    per_position = {}
    for name, dims in dimensions.items():
        per_position[name] = {}
        for mode in MODES:
            _, values = episode_means(squared[mode][:, :, dims].mean(axis=-1), conditions)
            per_position[name][mode] = values.tolist()
    return {
        "episode_ids": episode_ids, "number_of_demonstrations": len(episode_ids),
        "number_of_conditions": len(conditions), "regions": regions, "dimensions": dimensions,
        "metrics": metrics, "per_demo_position_mse": per_position,
        "aggregation": "Mean over noises per window, then windows per demo; equal-weight demos",
        "difference_sign": "intervention minus correct; positive means correct has lower error",
        "confidence_interval": "4000 deterministic paired bootstrap draws over demos; conditional on this checkpoint",
    }


def representative_index(conditions):
    """Choose before scoring: first demo, nearest middle window, smallest seed."""
    if not conditions:
        raise ValueError("No sampling conditions")
    return min(range(len(conditions)), key=lambda index: (
        conditions[index]["episode_index"],
        abs(conditions[index]["chunk_start_fraction"] - 0.5),
        conditions[index]["frame_index"], conditions[index]["noise_seed"],
    ))


def plot_full_sampling(summary, predictions, targets, conditions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = representative_index(conditions)
    row = conditions[selected]
    prefix = summary["regions"]["executed_prefix"][1]
    colors = {"correct": "#20805c", "zero": "#7d8796", "reversed": "#d96335"}
    labels = {"correct": "Correct time", "zero": "Time branch off", "reversed": "Reversed time"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.6), constrained_layout=True)
    fig.suptitle("Does correct time help full action generation from pure noise?", fontsize=19, weight="bold")
    ax = axes[0, 0]
    ax.axvspan(-0.5, prefix - 0.5, color="#e5edf6", alpha=0.8, label=f"Executed prefix (0-{prefix - 1})")
    ax.plot(targets[selected, :, 6], color="#17283a", lw=2.4, label="Recorded target")
    for mode in MODES:
        ax.plot(predictions[mode][selected, :, 6], color=colors[mode], lw=1.6, label=labels[mode])
    ax.set(title=f"A. Complete native sampling: demo {row['episode_index']}, frame {row['frame_index']}\n"
                 f"same initial noise seed {row['noise_seed']}",
           xlabel="Future action index (0-49)", ylabel="Normalized gripper action")
    ax.set_xlim(-0.5, HORIZON - 0.5)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.18)
    ax = axes[0, 1]
    for offset, region in enumerate(("executed_prefix", "full_chunk")):
        for index, mode in enumerate(MODES):
            x = offset * 4 + index
            values = np.asarray(summary["metrics"][region]["all"]["per_demo_mse"][mode])
            ax.scatter(x + np.linspace(-0.14, 0.14, len(values)), values, color=colors[mode], s=22)
            ax.plot([x - 0.28, x + 0.28], [values.mean()] * 2, color=colors[mode], lw=3)
    ax.set_xticks([0, 1, 2, 4, 5, 6], ["Correct", "Zero", "Reversed"] * 2)
    ax.set(title=f"B. Every dot is a demo; lines are means\nPrefix 0-{prefix - 1} (left) / full 0-49 (right)",
           ylabel="Normalized generated-action MSE (lower is better)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.18)
    ax = axes[1, 0]
    per_position = summary["per_demo_position_mse"]["all"]
    heat, row_labels = [], []
    for index, episode in enumerate(summary["episode_ids"]):
        for mode in ("zero", "reversed"):
            heat.append(np.asarray(per_position[mode][index]) - np.asarray(per_position["correct"][index]))
            row_labels.append(f"{episode} / {'Zero' if mode == 'zero' else 'Rev'}")
    heat = np.asarray(heat)
    limit = max(float(np.max(np.abs(heat))), 1e-12)
    artist = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                       interpolation="nearest")
    ax.axvline(prefix - 0.5, color="#17283a", lw=1.3, ls="--")
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    ax.set(title="C. Error change after full-sampling time interventions",
           xlabel=f"Future action index; dashed line ends prefix 0-{prefix - 1}", ylabel="Demonstration / intervention")
    fig.colorbar(artist, ax=ax, fraction=0.045, pad=0.02, label="MSE intervention - correct")
    ax = axes[1, 1]
    tick_labels = []
    for region_index, region in enumerate(("executed_prefix", "full_chunk")):
        metric = summary["metrics"][region]["all"]
        for mode_index, mode in enumerate(("zero", "reversed")):
            x = 2 * region_index + mode_index
            key = f"{mode}_minus_correct"
            values = np.asarray(metric["paired_deltas"][key])
            mean = float(values.mean())
            ax.scatter(x + np.linspace(-0.13, 0.13, len(values)), values, s=23, color=colors[mode])
            ax.plot([x - 0.3, x + 0.3], [mean] * 2, color="#17283a", lw=2.6)
            interval = metric["paired_bootstrap_95pct"][key]
            if interval is not None:
                ax.vlines(x, interval[0], interval[1], color="#17283a", lw=1.2)
            ax.annotate(f"{mean:+.4g}", (x, mean), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8)
            tick_labels.append(f"{'Zero' if mode == 'zero' else 'Reversed'} - Correct\n"
                               f"{'Prefix' if region_index == 0 else 'Full chunk'}")
    ax.axhline(0, color="#17283a", lw=1)
    ax.set_xticks(range(4), tick_labels, fontsize=9)
    ax.set(title="D. Paired demo differences; positive favors correct time",
           ylabel="Generated-action MSE difference")
    ax.grid(axis="y", alpha=0.18)
    fig.supxlabel("Same checkpoint, observations, initial noise and RoPE; only the added time branch changes.\n"
                  "Complete offline samples, not robot rollouts. Warm cells: intervention hurts; cool cells: helps.\n"
                  "Noise draws and windows are averaged within demo; vertical intervals in D are paired demo bootstrap 95%.",
                  fontsize=9)
    paths = []
    for extension in ("png", "svg"):
        path = output / f"fig05_full_sampling_time_alignment.{extension}"
        fig.savefig(path, dpi=160)
        paths.append(str(path))
    plt.close(fig)
    return paths, selected


def _atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path, **arrays):
    import numpy as np

    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".full-sampling-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(args):
    import numpy as np
    import torch
    from lerobot.scripts import pi05_task7_time_evidence as evidence
    from lerobot.scripts import pi05_task7_timepath_experiment as timepath
    from lerobot.scripts.pi05_task7_train_eval import file_sha256, output_lock

    record = timepath.check_plan(args.directory)
    if args.time_input_checkpoint is None:
        job = timepath.build_jobs(record, args.directory, args.output_root,
                                 selection="time_input", repeat=args.repeat)[0]
        checkpoint = Path(job["checkpoint"]).resolve()
    else:
        checkpoint = args.time_input_checkpoint.expanduser().resolve()
    for name in ("config.json", "model.safetensors"):
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise FileNotFoundError(f"Missing or empty trained checkpoint file: {checkpoint / name}")
    config = json.loads((checkpoint / "config.json").read_text())
    validate_sampling_config(config)
    provenance = evidence.validate_training_provenance({"time_input": checkpoint}, record)
    if not math.isclose(float(config.get("action_time_encoding_fps", 0)), float(record["fps"])):
        raise ValueError("Checkpoint time-encoding FPS differs from the recorded training plan")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("No CUDA available; --allow-cpu is for debugging only")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config_hash = file_sha256(checkpoint / "config.json")
    stat = (checkpoint / "model.safetensors").stat()
    fingerprint = {"config_sha256": config_hash, "weights_size": stat.st_size,
                   "weights_mtime_ns": stat.st_mtime_ns}
    with output_lock(args.output):
        report_path = args.output / "full_sampling_evidence.json"
        _atomic_json(report_path, {"protocol": PROTOCOL, "status": "running",
                                  "checkpoint": str(checkpoint)})
        try:
            anchors, dataset, actions, metadata = evidence._probe_data(args, record)
            descriptions = evidence.describe_anchors(metadata, anchors)
            policy, processor = evidence._load_policy_and_processor(checkpoint, args.tokenizer, device)
            validate_sampling_config(policy.config)
            encoder_hash = tensor_sha256(policy.model.action_time_encoder.weight)
            predictions = {mode: [] for mode in MODES}
            targets, conditions, audits, initial_noises = [], [], [], []
            print(f"Full sampling: {len(anchors)} windows x {len(args.noise_seeds)} noises x 3 interventions; "
                  f"{policy.config.num_inference_steps} native denoising steps; one Time-Input checkpoint", flush=True)
            for number, ((episode, frame), description) in enumerate(zip(anchors, descriptions, strict=True), 1):
                inputs, target = _make_batches(dataset, actions, frame, processor)
                for seed in args.noise_seeds:
                    noise = stable_sampling_noise(seed, frame, (1, HORIZON, policy.config.max_action_dim)).to(device)
                    row_audits = {}
                    for mode in MODES:
                        generated, audit = _sample_counterfactual(policy, inputs, noise, mode)
                        predictions[mode].append(generated[0].float().cpu().numpy())
                        row_audits[mode] = audit
                    if len({item["observation_sha256"] for item in row_audits.values()}) != 1 or len(
                        {item["initial_noise_sha256"] for item in row_audits.values()}
                    ) != 1:
                        raise RuntimeError("Intervention arms received different paired inputs")
                    targets.append(target.copy())
                    initial_noises.append(noise[0].cpu().numpy())
                    conditions.append({**description, "episode_index": int(episode),
                                       "global_frame_index": int(frame), "noise_seed": int(seed)})
                    audits.append(row_audits)
                print(f"  completed window {number}/{len(anchors)}: demo={episode}, frame={description['frame_index']}",
                      flush=True)
            if tensor_sha256(policy.model.action_time_encoder.weight) != encoder_hash:
                raise RuntimeError("Time encoder weights changed during sampling")
            inference_steps = int(policy.config.num_inference_steps)
            checkpoint_action_steps = int(policy.config.n_action_steps)
            del policy, processor
            if device.type == "cuda":
                torch.cuda.empty_cache()
            predictions = {mode: np.stack(values) for mode, values in predictions.items()}
            targets = np.stack(targets)
            if len(conditions) != len(anchors) * len(args.noise_seeds):
                raise RuntimeError("Incomplete predeclared window/noise grid")
            summary = summarize_predictions(predictions, targets, conditions, args.executed_steps)
            paths, selected = plot_full_sampling(summary, predictions, targets, conditions, args.output)
            array_path = args.output / "full_sampling_arrays.npz"
            _atomic_npz(array_path, clean_target=targets, initial_noise=np.stack(initial_noises),
                        **{f"{mode}_actions": values for mode, values in predictions.items()})
            after = (checkpoint / "model.safetensors").stat()
            if file_sha256(checkpoint / "config.json") != config_hash or (
                after.st_size, after.st_mtime_ns
            ) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError("Checkpoint files changed during the diagnostic")
            summary.update({
                "protocol": PROTOCOL, "status": "completed", "checkpoint": str(checkpoint),
                "checkpoint_fingerprint": fingerprint, "training_provenance": provenance,
                "source_dataset": record["dataset_root"],
                "training_episode_excluded": int(record["selection"]["episode_index"]),
                "number_of_anchors": len(anchors), "noise_seeds": list(args.noise_seeds),
                "anchor_selection": {
                    "probe_demos_requested": args.probe_demos,
                    "windows_per_demo_requested": args.windows_per_demo,
                    "total_window_cap": args.samples,
                    "rule": "Evenly spaced eligible demos and complete early-to-late windows; no outcome filtering",
                },
                "interventions": list(MODES), "conditions": conditions, "sampling_audits": audits,
                "sampling": {
                    "api": "PI05Policy.predict_action_chunk", "initial_state": "pure_gaussian_noise",
                    "native_num_inference_steps": inference_steps, "prediction_horizon": HORIZON,
                    "executed_prefix_steps": args.executed_steps,
                    "executed_prefix_source": "Explicit diagnostic setting; task7 closed-loop wrapper defaults to 10",
                    "checkpoint_n_action_steps": checkpoint_action_steps,
                    "targets_passed_to_inference": False, "noise_protocol": NOISE_PROTOCOL,
                    "one_checkpoint_loaded_for_all_interventions": True,
                },
                "representative_condition": conditions[selected], "representative_row": selected,
                "representative_selection": "First demo, nearest middle window, smallest fixed noise seed; independent of errors",
                "array_file": str(array_path), "array_row_order": "conditions",
                "array_sha256": file_sha256(array_path), "figure_paths": paths,
                "source_sha256": file_sha256(Path(__file__)), "time_encoder_sha256": encoder_hash,
                "implementation_sha256": {
                    "evidence_helpers": file_sha256(Path(evidence.__file__)),
                    **{name: file_sha256(Path(__file__).parents[1] / "policies" / "pi05" / name)
                       for name in ("modeling_pi05.py", "processor_pi05.py", "action_time.py")},
                },
                "runtime": {"torch_version": str(torch.__version__), "device": str(device),
                            "cuda_version": torch.version.cuda},
                "note": (
                    "Offline fully generated normalized actions, not closed-loop success or executed trajectories. "
                    "All arms retain native RoPE and share weights, observations and initial noise. "
                    "Positive intervention-minus-correct MSE favors correct time; no positive outcome is required. "
                    "Demonstrations exclude this fine-tuning episode; the pretrained source may have seen LIBERO. "
                    "One recorded future need not be the only valid action sequence."
                ),
            })
            _atomic_json(report_path, summary)
        except BaseException as error:
            _atomic_json(report_path, {"protocol": PROTOCOL, "status": "failed",
                                      "checkpoint": str(checkpoint), "error": f"{type(error).__name__}: {error}"})
            raise
    for region in ("executed_prefix", "full_chunk"):
        print(f"{region}: {summary['metrics'][region]['all']['mean_mse']}", flush=True)
        print(f"  paired MSE changes: {summary['metrics'][region]['all']['mean_paired_deltas']}", flush=True)
    print(f"Saved {report_path}", flush=True)
    print("Saved " + paths[0], flush=True)
    return summary


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--time-input-checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("/home/wyn/models/paligemma-3b-pt-224"))
    parser.add_argument("--dataset-repo-id", default="libero")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--probe-demos", type=int, default=8)
    parser.add_argument("--windows-per-demo", type=int, default=3)
    parser.add_argument("--samples", type=int, help="Optional total window cap for a smoke run")
    parser.add_argument("--noise-seeds", default="3000,4000,5000")
    parser.add_argument("--executed-steps", type=int, default=10,
                        help="Score the prefix actually executed by the eval wrapper; default 10, independent of saved config")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        args.noise_seeds = tuple(int(part.strip()) for part in args.noise_seeds.split(","))
        if not args.noise_seeds or len(set(args.noise_seeds)) != len(args.noise_seeds) or any(
            seed < 0 or seed >= 2**31 for seed in args.noise_seeds
        ):
            raise ValueError("Noise seeds must be distinct integers in [0, 2**31)")
        if args.gpu < 0 or args.repeat < 1 or args.probe_demos < 1 or args.windows_per_demo < 1:
            raise ValueError("Invalid GPU/repeat/demo/window count")
        if args.samples is not None and args.samples < 1:
            raise ValueError("--samples must be positive")
        if not 1 <= args.executed_steps <= HORIZON:
            raise ValueError("--executed-steps must be in [1, 50]")
        args.directory = args.directory.expanduser().resolve()
        args.output_root = args.output_root.expanduser().resolve()
        args.output = (args.output or args.output_root / "analysis" / "task7-train-eval"
                       / f"r{args.repeat}" / "full_sampling").expanduser().resolve()
        if args.dry_run:
            print("DRY RUN: no models/datasets/GPU are loaded and no files are written.")
            print(f"One Time-Input checkpoint; interventions={MODES}; pure Gaussian noise; native full denoising.")
            print(f"Requested: {args.probe_demos} demos x {args.windows_per_demo} windows x {len(args.noise_seeds)} noises.")
            print(f"Score prefix 0..{args.executed_steps - 1} and full chunk 0..49; labels used only for scoring.")
            print(f"Checkpoint: {args.time_input_checkpoint or f'from task7 plan, repeat {args.repeat}'}")
            print(f"Output: {args.output}")
            return None
        return run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Task7 full sampling failed: {error}\n")


if __name__ == "__main__":
    main()
