"""Empirical evidence for the RoPE + action-input temporal residual hypothesis.

Modes:
  success : CPU-only paired outcomes from a dual-seed comparison JSON, or the
            original four-arm task7 evaluations for legacy callers
  probe   : GPU offline velocity probe on held-out demonstration(s), comparing
            RoPE baseline and trained Time-Input with correct/zero/reversed
            time residuals; no simulator actions are executed
  all     : both

Counterfactuals on the SAME Time-Input weights are TRAIN/TEST mismatched;
they demonstrate reliance/sensitivity, not by themselves superior generalization.
The mechanism schematic is illustrative, while episode/velocity plots use
actual saved results and model predictions. All outputs identify the source.
"""

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.scripts import pi05_task7_timepath_experiment as timepath
from lerobot.scripts import pi05_gripper_event_experiment as previous

PROTOCOL = "pi05-task7-time-input-evidence-v2"
# Preserve the existing per-anchor random draws for the legacy singular flags.
NOISE_PROTOCOL = "pi05-task7-time-input-evidence-v1"
CASES = timepath.CASES
HORIZON = 50
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path("/data1/wyn/chkpt/2601-lerobot/PromptVLA")
DEFAULT_EVIDENCE = DEFAULT_OUTPUT / "analysis" / "task7-rope-time-input"
DEFAULT_FLOW_TIMES = (0.2, 0.5, 0.8)
DEFAULT_NOISE_SEEDS = (3000, 4000, 5000)


def parse_seeds(value):
    try:
        seeds = [int(token) for token in value.split(",")]
    except ValueError as error:
        raise ValueError("Expected comma-separated integer seeds") from error
    if not seeds or any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    return seeds


def parse_flow_times(value):
    try:
        values = [float(token) for token in value.split(",")]
    except ValueError as error:
        raise ValueError("Expected comma-separated flow times strictly between 0 and 1") from error
    if (not values or any(not math.isfinite(t) or not 0 < t < 1 for t in values)
            or len(set(values)) != len(values)):
        raise ValueError("Flow times must be distinct finite values strictly between 0 and 1")
    return values


def probe_grid(args):
    """The old singular flags explicitly select a one-point diagnostic grid."""
    singular_time = getattr(args, "flow_time", None)
    singular_seed = getattr(args, "noise_seed", None)
    plural_times = getattr(args, "flow_times", None)
    plural_seeds = getattr(args, "noise_seeds", None)
    if singular_time is not None and plural_times is not None:
        raise ValueError("Use either --flow-time or --flow-times")
    if singular_seed is not None and plural_seeds is not None:
        raise ValueError("Use either --noise-seed or --noise-seeds")
    times = ([float(singular_time)] if singular_time is not None else
             parse_flow_times(plural_times) if isinstance(plural_times, str) else
             list(plural_times) if plural_times is not None else list(DEFAULT_FLOW_TIMES))
    seeds = ([int(singular_seed)] if singular_seed is not None else
             parse_seeds(plural_seeds) if isinstance(plural_seeds, str) else
             list(plural_seeds) if plural_seeds is not None else list(DEFAULT_NOISE_SEEDS))
    # Validate programmatic calls as well as CLI input.
    times = parse_flow_times(",".join(map(str, times)))
    seeds = parse_seeds(",".join(map(str, seeds)))
    return times, seeds


def _json(path):
    return json.loads(Path(path).read_text())


def _task7_successes(info_path, expected=10):
    info = _json(info_path)
    tasks = info.get("per_task", info.get("Aggregated Metrics for per_task", []))
    if type(tasks) is not list or len(tasks) != 1:
        raise ValueError("Expected exactly one evaluated task, not all-suite aggregates")
    task = tasks[0]
    if task.get("task_group") != "libero_10" or task.get("task_id") != 7:
        raise ValueError("Not the official LIBERO-long task7")
    rows = task.get("metrics", task).get("successes")
    if (
        not isinstance(rows, list) or len(rows) != expected
        or any(type(x) is not bool for x in rows)
    ):
        raise ValueError("Missing or invalid boolean per-episode success flags")
    return np.asarray(rows, dtype=np.int64)


def paired_counts(rope, temporal):
    a = np.asarray(rope, dtype=bool).reshape(-1)
    b = np.asarray(temporal, dtype=bool).reshape(-1)
    if a.shape != b.shape or not len(a):
        raise ValueError("Paired success vectors must have equal nonzero length")
    return {
        "both_success": int(np.sum(a & b)),
        "both_failure": int(np.sum(~a & ~b)),
        "time_only_success": int(np.sum(~a & b)),
        "rope_only_success": int(np.sum(a & ~b)),
        "net_time_successes": int(np.sum(b)-np.sum(a)),
    }


def _timepath_args(args, seed):
    return SimpleNamespace(
        eval_root=args.eval_root, eval_seed=seed, shift_seed=args.shift_seed,
        shift_m=args.shift_m, episodes=10, repeat=args.repeat,
    )


def validated_success_matrix(args, record):
    """Read RAW eval_info.json and require matching physics + model fingerprints."""
    jobs = timepath.build_jobs(record, args.directory, args.output_root, repeat=args.repeat)
    matrix, manifests = [], []
    for seed in args.seeds:
        context = _timepath_args(args, seed)
        path = timepath.evaluation_config_path(context, record)
        shift = timepath.shift_config(context)
        if _json(path) != shift:
            raise ValueError(f"Seed {seed}: physics configuration changed")
        shift_sha = previous.file_hash(path)
        row = {}
        helper_hashes = set()
        for job in jobs:
            raw = timepath.result_path(job, record, context)
            manifest = _json(raw.parent / "resume_manifest.json")
            applied = manifest.get("task7_object_shift", {})
            if (
                applied.get("config") != shift or applied.get("config_sha256") != shift_sha
                or not applied.get("helper_sha256") or manifest.get("fixed_observation_noise")
            ):
                raise ValueError(f"Seed {seed}, {job['case']}: inconsistent shift/noise manifest")
            helper_hashes.add(applied["helper_sha256"])
            model = job["checkpoint"]
            current_files = [
                [str(p.relative_to(model)), p.stat().st_size, p.stat().st_mtime_ns]
                for p in sorted(model.rglob("*")) if p.is_file()
            ]
            if not current_files or current_files != manifest.get("model_files"):
                raise ValueError(f"Seed {seed}, {job['case']}: evaluated checkpoint changed")
            row[job["case"]] = _task7_successes(raw).tolist()
        if len(helper_hashes) != 1:
            raise ValueError(f"Seed {seed}: methods used different physics implementations")
        matrix.append(row)
        manifests.append({"seed": seed, "physics_helper_sha256": next(iter(helper_hashes))})
    return np.array(
        [[row[case] for row in matrix] for case in CASES], dtype=np.int64
    ).reshape(len(CASES), len(args.seeds)*10), manifests


def plot_success(matrix, seeds, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if matrix.shape != (len(CASES), 10*len(seeds)):
        raise ValueError("Expected 4 methods x (10 times number of seeds)")
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 6.2),
        gridspec_kw={"height_ratios": [2.1, 1.0]}, layout="constrained",
    )
    ax = axes[0]
    cmap = ListedColormap(["#DDE3EC", "#327A65"])
    ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(CASES)))
    ax.set_yticklabels(["RoPE-only", "Time-QK (layer 1)", "Time-V (layer 1)", "Time-Input"])
    ax.set_xticks([i*10+4.5 for i in range(len(seeds))])
    ax.set_xticklabels([f"eval seed {seed}" for seed in seeds])
    for pos in range(1, len(seeds)):
        ax.axvline(pos*10-0.5, linewidth=1.2, color="#111827")
    ax.set_title("Actual paired LIBERO-long task7 outcomes (green = success)")
    for k, case in enumerate(CASES):
        ax.text(matrix.shape[1]+0.3, k, f"{matrix[k].sum()}/{matrix.shape[1]}",
                va="center", fontsize=10)
    ax.set_xlim(-0.5, matrix.shape[1]+3.4)
    rope, ti = matrix[0].astype(bool), matrix[3].astype(bool)
    counts = paired_counts(rope, ti)
    bars = [
        counts["both_success"], counts["both_failure"],
        counts["time_only_success"], counts["rope_only_success"],
    ]
    labels = ["Both success", "Both fail", "Time only", "RoPE only"]
    axes[1].bar(labels, bars)
    for i, n in enumerate(bars):
        axes[1].text(i, n+.15, str(n), ha="center", va="bottom", fontsize=10)
    axes[1].set_ylim(0, max(bars+[1])*1.25+0.5)
    axes[1].set_ylabel("Paired episodes")
    axes[1].set_title(
        f"Time-Input vs RoPE-only: net {counts['net_time_successes']:+d} "
        "successes (descriptive, not independent train seeds)"
    )
    axes[1].grid(axis="y", alpha=0.2)
    return _save_figure(fig, out, "fig01_paired_task7_outcomes")


def _save_figure(fig, out, prefix):
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    paths=[]
    for ext in ("png", "svg"):
        path=out / f"{prefix}.{ext}"
        fig.savefig(path, dpi=190, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def stable_noise(seed, frame_index, shape):
    """CPU sample seeded by anchor ID, reused across ALL models/interventions."""
    message=f"{NOISE_PROTOCOL}:flow:{seed}:{frame_index}"
    generator=torch.Generator(device="cpu").manual_seed(
        int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little") & ((1<<63)-1)
    )
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def select_nontraining_anchors(frame_metadata, *, task_index, train_episode, count=None,
                               horizon=HORIZON, demos=8, windows_per_demo=3):
    """Preselect complete early/middle/late windows without inspecting outcomes.

    Episodes are evenly spaced in the sorted eligible episode IDs. Each chosen
    episode contributes evenly spaced *valid chunk starts*, including its first
    and last possible complete window. Short demos can have overlapping windows;
    all statistics therefore aggregate within episode before comparing demos.
    ``count`` is the legacy --samples total-window cap, applied by deterministic
    stratification over the full selection rather than taking only its prefix.
    """
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("count must be a positive integer or None")
    if any(type(value) is not int or value < 1 for value in (horizon, demos, windows_per_demo)):
        raise ValueError("horizon, demos and windows_per_demo must be positive integers")
    task = np.asarray(frame_metadata["task_index"])
    ep = np.asarray(frame_metadata["episode_index"])
    frame = np.asarray(frame_metadata["frame_index"])
    if task.ndim != 1 or ep.shape != task.shape or frame.shape != task.shape:
        raise ValueError("Frame metadata columns must be aligned one-dimensional arrays")
    available = []
    for episode in sorted(int(e) for e in np.unique(ep[task == task_index])
                          if int(e) != train_episode):
        idx = np.flatnonzero((task == task_index) & (ep == episode))
        if len(idx) < horizon:
            continue
        if not np.array_equal(frame[idx], np.arange(len(idx))):
            raise ValueError("Probe episode frame indices are not continuous")
        if not np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
            raise ValueError("Probe episode source data are not consecutive")
        available.append((episode, idx))
    if not available:
        raise ValueError(
            "No other complete task7 demonstration exists. Refusing to call "
            "training-demonstration reconstruction a heldout evaluation."
        )
    episode_slots = np.rint(np.linspace(0, len(available) - 1, min(demos, len(available)))).astype(int)
    selected = []
    for slot in episode_slots:
        episode, idx = available[int(slot)]
        last_start = len(idx) - horizon
        # With one requested window choose the middle, not always frame zero.
        offsets = (np.array([last_start // 2]) if windows_per_demo == 1 else
                   np.unique(np.rint(np.linspace(
                       0, last_start, min(windows_per_demo, last_start + 1)
                   )).astype(int)))
        selected.extend((int(episode), int(idx[offset])) for offset in offsets)
    if count is not None and len(selected) > count:
        slots = np.rint(np.linspace(0, len(selected) - 1, count)).astype(int)
        selected = [selected[int(slot)] for slot in slots]
    return selected


def describe_anchors(frame_metadata, anchors, horizon=HORIZON):
    """Persist the actual episode IDs and phase coverage, including overlap."""
    episodes = np.asarray(frame_metadata["episode_index"])
    frames = np.asarray(frame_metadata["frame_index"])
    descriptions = []
    intervals = {}
    for episode, idx in anchors:
        if idx < 0 or idx + horizon > len(episodes):
            raise ValueError("Anchor lies outside the source data")
        if not np.all(episodes[idx:idx + horizon] == episode):
            raise ValueError("Anchor crosses an episode boundary")
        start = int(frames[idx])
        if not np.array_equal(frames[idx:idx + horizon], np.arange(start, start + horizon)):
            raise ValueError("Anchor frames are not contiguous")
        length = int(np.sum(episodes == episode))
        last_start = length - horizon
        fraction = start / last_start if last_start > 0 else 0.0
        phase = "early" if fraction < 1/3 else "middle" if fraction < 2/3 else "late"
        overlap = any(start < old_end and start + horizon > old_start
                      for old_start, old_end in intervals.get(int(episode), []))
        intervals.setdefault(int(episode), []).append((start, start + horizon))
        descriptions.append({
            "episode_index": int(episode), "global_frame_index": int(idx),
            "frame_index": start, "end_frame_inclusive": start + horizon - 1,
            "episode_length": length, "chunk_start_fraction": fraction,
            "phase": phase, "overlaps_previous_selected_window": bool(overlap),
        })
    return descriptions


@contextmanager
def time_counterfactual(encoder, intervention):
    """Temporary, non-persistent intervention on the learned Time-Input only.

    The original Gemma RoPE is retained for all three arms.
    No weights are modified; no checkpoint is written.
    """
    if intervention == "correct":
        yield
        return
    if encoder is None:
        raise ValueError("Counterfactual requires the learned action-time encoder")
    if intervention == "zero":
        old=encoder.forward
        try:
            encoder.forward=lambda inputs, **kwargs: inputs
            yield
        finally:
            encoder.forward=old
        return
    if intervention == "reversed":
        old=encoder.features
        def reversed_features(horizon, device, **kwargs):
            result=old(horizon, device, **kwargs)
            return result.flip(dims=[-2])
        try:
            encoder.features=reversed_features
            yield
        finally:
            encoder.features=old
        return
    raise ValueError("Unknown intervention: " + str(intervention))


def _load_policy_and_processor(checkpoint, tokenizer, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    checkpoint=Path(checkpoint).resolve(strict=True)
    # Saved configs include the ChoiceRegistry discriminator ("type": "pi05").
    # Decode through the registry base, as PI05Policy.from_pretrained does;
    # decoding directly as PI05Config treats "type" as an unknown dataclass field.
    config=PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise ValueError(f"Expected a PI05 checkpoint at {checkpoint}, got {config.type!r}")
    config.device=str(device)
    config.tokenizer_name=str(Path(tokenizer).resolve(strict=True))
    config.compile_model=False
    config.gradient_checkpointing=False

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy=PI05Policy.from_pretrained(
        checkpoint, config=config, local_files_only=True, strict=True
    ).to(device).eval()
    preprocessor,_=make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor":{"device":str(device)},
            "tokenizer_processor":{"tokenizer_name":str(Path(tokenizer).resolve())},
        },
    )
    return policy, preprocessor


@torch.inference_mode()
def _probe_predict(policy, batch, noise, flow_time, intervention="correct"):
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    clean=policy.prepare_action(batch)
    if clean.shape!=noise.shape or clean.shape[0] != 1 or clean.shape[1] != HORIZON:
        raise ValueError("Probe action dimensions do not match the fixed Flow noise")
    if getattr(policy.config, "action_time_basis", "index") != "index":
        raise ValueError("This probe supports action_time_basis=index only; physical dt must be explicit")
    t=torch.full((clean.shape[0],), float(flow_time), device=clean.device)
    x_t=t[:,None,None]*noise + (1.0-t[:,None,None])*clean
    target=noise-clean
    model=policy.model
    encoder=model.action_time_encoder
    if intervention!="correct" and encoder is None:
        raise ValueError("Cannot run time ablation on a RoPE-only model")
    images,img_masks=policy._preprocess_images(batch)
    tokens,masks=batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    with torch.inference_mode(), time_counterfactual(encoder,intervention):
        pred=model.predict_velocity(images,img_masks,tokens,masks,x_t,t)
        action_embed=model.action_in_proj(x_t).detach()
    d=policy.config.output_features["action"].shape[0]
    if pred.shape != clean.shape or d != 7 or not torch.isfinite(pred).all():
        raise ValueError("Expected finite flow predictions with the native seven LIBERO action dimensions")
    per_step=((pred[:,:,:d]-target[:,:,:d])**2).mean(dim=-1)
    # This is a one-step estimate from a recorded-action interpolation point,
    # NOT an action chunk generated by the policy's full denoising rollout.
    clean_estimate = x_t[:, :, :d] - t[:, None, None] * pred[:, :, :d]
    return {
        "per_step_mse":per_step[0].detach().float().cpu().numpy(),
        "pred":pred[0,:,:d].detach().float().cpu().numpy(),
        "x_t":x_t,
        "embedding_before":action_embed,
        "clean_action":clean[0,:,:d].detach().float().cpu().numpy(),
        "clean_estimate":clean_estimate[0].detach().float().cpu().numpy(),
    }


def _record_time_residual(model, z):
    encoder=model.action_time_encoder
    with torch.inference_mode():
        time_emb=encoder(z)
        residual=(time_emb-z).detach().float().cpu().numpy()[0]
        base=z.detach().float().cpu().numpy()[0]
    rms=lambda x: np.sqrt(np.mean(x*x, axis=-1))
    ratio=rms(residual)/np.clip(rms(base),1e-8,None)
    return residual,ratio


def _probe_data(args, record):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.pdet_order import read_frame_metadata
    from lerobot.scripts.frameskip_scores import read_actions

    root=Path(record["dataset_root"]).resolve(strict=True)
    meta=read_frame_metadata(root)
    actions=read_actions(root)
    starts=select_nontraining_anchors(
        meta,task_index=int(record["selection"]["task_index"]),
        train_episode=int(record["selection"]["episode_index"]),count=args.samples,
        demos=args.probe_demos, windows_per_demo=args.windows_per_demo,
    )
    dataset=LeRobotDataset(
        args.dataset_repo_id,root=root,
        delta_timestamps={"action":[j/float(record["fps"]) for j in range(HORIZON)]},
        video_backend="pyav",return_uint8=True,
    )
    if tuple(actions.shape[1:]) != (7,):
        raise ValueError("Expected native LIBERO 7-D actions")
    return starts, dataset, actions, meta


def _batch_for_index(dataset, actions, frame_idx, *, processor):
    from lerobot.utils.collate import lerobot_collate_fn
    from torch.utils.data._utils.collate import default_collate

    item=dict(dataset[int(frame_idx)])
    if int(item["index"])!=int(frame_idx):
        raise ValueError("Dataset frame identity changed")
    # Use the RAW recorded ground-truth future 50 actions, no padding
    # and no synthetic cadence control assumptions.
    targets=np.asarray(actions[frame_idx:frame_idx+HORIZON],dtype=np.float32)
    item["action"]=torch.from_numpy(targets.copy())
    item["action_is_pad"]=torch.zeros(HORIZON,dtype=torch.bool)
    for key in dataset.meta.camera_keys:
        if key in item and item[key].dtype==torch.uint8:
            item[key]=item[key].float()/255.0
    fn=lerobot_collate_fn if dataset.meta.has_language_columns else default_collate
    batch=processor(fn([item]))
    if batch["action"].shape[1] != HORIZON:
        raise ValueError("Processor changed the 50-step action target horizon")
    return batch


def _compute_probe_for_model(
    label, checkpoint, dataset, actions, anchors, args, device,
    *, interventions,
):
    policy,processor=_load_policy_and_processor(checkpoint,args.tokenizer,device)
    results={mode:[] for mode in interventions}
    predictions={mode:[] for mode in interventions}
    clean_estimates={mode:[] for mode in interventions}
    residuals=[]
    ratios=[]
    normalized_targets=[]
    targets=[]
    conditions=[]
    flow_times, noise_seeds = probe_grid(args)
    for episode,idx in anchors:
        batch=_batch_for_index(dataset,actions,idx,processor=processor)
        with torch.inference_mode():
            clean=policy.prepare_action(batch)
        normalized_targets.append(clean.detach().float().cpu().numpy())
        first = len(conditions)
        for seed in noise_seeds:
            noise=stable_noise(seed,idx,tuple(clean.shape)).to(device)
            for flow_time in flow_times:
                conditions.append({
                    "episode_index":int(episode), "global_frame_index":int(idx),
                    "noise_seed":int(seed), "flow_time":float(flow_time),
                })
                for mode in interventions:
                    result=_probe_predict(policy,batch,noise,flow_time,intervention=mode)
                    results[mode].append(result["per_step_mse"].tolist())
                    predictions[mode].append(result["pred"])
                    clean_estimates[mode].append(result["clean_estimate"])
                    if mode == "correct":
                        targets.append(result["clean_action"])
                        if policy.model.action_time_encoder is not None:
                            residual,ratio=_record_time_residual(
                                policy.model,result["embedding_before"]
                            )
                            residuals.append(residual)
                            ratios.append(ratio)
        print(f"  {label}: episode={episode} anchor={idx} "
              f"correct-MSE={np.mean(results['correct'][first:]):.5f} "
              f"({len(flow_times)} times x {len(noise_seeds)} noises)",flush=True)
    out={
        "mse":{key:np.asarray(val,dtype=np.float64) for key,val in results.items()},
        "pred":{key:np.stack(val) for key,val in predictions.items()},
        "clean_estimate":{key:np.stack(val) for key,val in clean_estimates.items()},
        "clean_target":np.stack(targets),
        "ratio":np.stack(ratios) if ratios else None,
        "residual":np.stack(residuals) if residuals else None,
        "normalized_targets":np.concatenate(normalized_targets,axis=0),
        "conditions":conditions,
    }
    del policy,processor
    if device.type=="cuda":
        torch.cuda.empty_cache()
    return out


def _curve_mean_ci(values):
    mat=np.asarray(values,dtype=np.float64)
    return mat.mean(axis=0),mat.std(axis=0,ddof=1)/math.sqrt(len(mat)) if len(mat)>1 else np.zeros(mat.shape[1])


def episode_means(values, conditions):
    """Average repeats within anchor, then anchors within each demonstration.

    Each demonstration receives equal weight even if its number of complete
    windows differs. Noise draws and Flow times are not independent demos.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or len(values) != len(conditions) or not len(conditions):
        raise ValueError("Condition metadata must match the nonempty measurement array")
    if not np.isfinite(values).all():
        raise ValueError("Probe measurements contain nonfinite values")
    episodes = sorted({int(row["episode_index"]) for row in conditions})
    output = []
    for episode in episodes:
        frames = sorted({int(row["global_frame_index"]) for row in conditions
                         if int(row["episode_index"]) == episode})
        per_anchor = []
        for frame in frames:
            selected = [i for i, row in enumerate(conditions)
                        if int(row["episode_index"]) == episode
                        and int(row["global_frame_index"]) == frame]
            per_anchor.append(values[selected].mean(axis=0))
        output.append(np.stack(per_anchor).mean(axis=0))
    return episodes, np.stack(output)


def aggregate_probe(rope, temporal):
    """Convert raw paired condition rows into demonstration-level plot data."""
    conditions = temporal["conditions"]
    if rope["conditions"] != conditions:
        raise ValueError("RoPE and Time-Input condition IDs differ; cannot pair errors")
    plots = {}
    for name, model in (("rope", rope), ("time_input", temporal)):
        current = {"mse": {}, "pred": {}}
        for key, values in model["mse"].items():
            episodes, current["mse"][key] = episode_means(values, conditions)
        for key, values in model["pred"].items():
            _, current["pred"][key] = episode_means(values, conditions)
        for key in ("ratio", "residual"):
            current[key] = (episode_means(model[key], conditions)[1]
                            if model[key] is not None else None)
        plots[name] = current
    plots["episode_ids"] = episodes
    plots["output_change_rms"] = {}
    for mode in ("zero", "reversed"):
        # Compute output changes BEFORE averaging, to avoid cancellation.
        per_condition = np.sqrt(np.mean(
            (temporal["pred"][mode] - temporal["pred"]["correct"]) ** 2, axis=-1
        ))
        plots["output_change_rms"][mode] = episode_means(per_condition, conditions)[1]
    return plots


def alignment_summary(plots):
    """Describe measured directions without manufacturing a positive conclusion."""
    variants = {"rope": plots["rope"]["mse"]["correct"],
                **plots["time_input"]["mse"]}
    per_episode = {key: value.mean(axis=1) for key, value in variants.items()}
    contrasts = {
        "zero_minus_correct": per_episode["zero"] - per_episode["correct"],
        "reversed_minus_correct": per_episode["reversed"] - per_episode["correct"],
        "correct_minus_rope": per_episode["correct"] - per_episode["rope"],
    }
    tolerance = max(1e-10, abs(float(per_episode["correct"].mean())) * 1e-6)
    result = {
        "aggregation_unit": "demonstration",
        "aggregation_rule": "mean over noise/t per anchor, then anchors per demo, then equal-weight demos",
        "episode_ids": list(plots["episode_ids"]),
        "number_of_demonstrations": len(plots["episode_ids"]),
        "mse_by_variant": {key: float(value.mean()) for key, value in per_episode.items()},
        "paired_episode_mse": {key: value.tolist() for key, value in per_episode.items()},
        "paired_episode_deltas": {key: value.tolist() for key, value in contrasts.items()},
        "mean_paired_deltas": {key: float(value.mean()) for key, value in contrasts.items()},
        "numerical_tie_tolerance": tolerance,
        "delta_direction_counts": {
            key: {"positive": int(np.sum(value > tolerance)),
                  "negative": int(np.sum(value < -tolerance)),
                  "numerical_tie": int(np.sum(np.abs(value) <= tolerance))}
            for key, value in contrasts.items()
        },
    }
    off_delta = result["mean_paired_deltas"]["zero_minus_correct"]
    rev_delta = result["mean_paired_deltas"]["reversed_minus_correct"]
    if off_delta > tolerance and rev_delta > tolerance:
        result["observed_direction"] = "correct_has_lower_mean_error_than_both_interventions"
        result["interpretation"] = (
            "Correct alignment has lower mean offline error than both interventions on this probe set. "
            "This supports useful reliance on the trained time path here; it is not a significance "
            "test or proof of improved closed-loop generalization."
        )
    elif abs(off_delta) <= tolerance and abs(rev_delta) <= tolerance:
        result["observed_direction"] = "no_mean_error_difference_above_numerical_tolerance"
        result["interpretation"] = (
            "No mean error difference above numerical tolerance on this probe set; "
            "the experiment does not establish a useful alignment benefit."
        )
    else:
        result["observed_direction"] = "mixed_or_adverse_alignment_evidence"
        result["interpretation"] = (
            "Correct alignment does not beat both interventions in mean offline error. "
            "Report the negative and mixed results; this probe does not establish the proposed benefit."
        )
    return result


def representative_condition(conditions, anchor_metadata):
    """Choose by episode/phase/t/seed order only, never by prediction or error."""
    first_episode = min(row["episode_index"] for row in anchor_metadata)
    anchor = min((row for row in anchor_metadata if row["episode_index"] == first_episode),
                 key=lambda row: (abs(row["chunk_start_fraction"] - 0.5), row["frame_index"]))
    choices = [i for i, row in enumerate(conditions)
               if row["episode_index"] == first_episode
               and row["global_frame_index"] == anchor["global_frame_index"]]
    if not choices:
        raise ValueError("Representative anchor is absent from the probe conditions")
    # The original condition order breaks ties, choosing the first configured noise seed.
    index = min(choices, key=lambda i: (abs(conditions[i]["flow_time"] - 0.5), i))
    return index, {**anchor, **conditions[index], "action_dimension": 6,
                   "selection_rule": "first selected episode, nearest-middle window, nearest t=0.5, first noise seed"}


def plot_probe(probe, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rope=probe["rope"]["mse"]["correct"]
    tm=probe["time_input"]["mse"]
    orig=probe["time_input"]["pred"]["correct"]
    mask=probe["time_input"]["pred"]
    residual=probe["time_input"]["residual"]
    ratios=probe["time_input"]["ratio"]
    steps=np.arange(HORIZON)
    fig,axes=plt.subplots(2,2,figsize=(13.2,9.0),layout="constrained")
    ax=axes[0,0]
    for label,values in [
        ("RoPE-only",rope),
        ("Time-Input (trained)",tm["correct"]),
        ("Time-Input (zero residual)",tm["zero"]),
        ("Time-Input (reverse residual)",tm["reversed"]),
    ]:
        avg,stderr=_curve_mean_ci(values)
        ax.plot(steps,avg,label=label)
        ax.fill_between(steps,avg-stderr,avg+stderr,alpha=0.12)
    ax.set_title("A. Flow-velocity error (mean ± 1 SEM over demonstrations)")
    ax.set_xlabel("Future action index within 50-step chunk")
    ax.set_ylabel("Normalized velocity MSE (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax=axes[0,1]
    pairs={
        "Correct − RoPE": tm["correct"].mean(axis=1)-rope.mean(axis=1),
        "Zero − correct":tm["zero"].mean(axis=1)-tm["correct"].mean(axis=1),
        "Reversed − correct":tm["reversed"].mean(axis=1)-tm["correct"].mean(axis=1),
    }
    for i,(key,vals) in enumerate(pairs.items()):
        ax.scatter(np.full(len(vals),i)+np.linspace(-0.16,0.16,len(vals)),
                   vals,s=22,alpha=0.75)
        ax.plot([i-0.23,i+0.23],[vals.mean()]*2,linewidth=2.4)
    ax.axhline(0,color="black",linewidth=0.8,alpha=0.65)
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(pairs),rotation=11)
    ax.set_title("B. Per-demonstration paired MSE differences")
    ax.set_ylabel("MSE delta (positive = first label worse)")
    ax.grid(axis="y",alpha=0.2)
    ax=axes[1,0]
    ax.plot(steps,ratios.mean(axis=0))
    ax.fill_between(steps,ratios.mean(axis=0)-ratios.std(axis=0),
                    ratios.mean(axis=0)+ratios.std(axis=0),alpha=0.15)
    ax.set_title("C. Learned input-time residual strength")
    ax.set_xlabel("Future action index")
    ax.set_ylabel("RMS(time residual) / RMS(noisy-action embedding)")
    ax.grid(alpha=0.2)
    ax=axes[1,1]
    for name in ("zero","reversed"):
        dist = (probe["output_change_rms"][name] if "output_change_rms" in probe
                else np.sqrt(np.mean((mask[name]-orig)**2,axis=-1)))
        ax.plot(steps,dist.mean(axis=0),label=name+" vs correct")
    ax.set_title("D. Change in predicted velocity when time path is perturbed")
    ax.set_xlabel("Future action index")
    ax.set_ylabel("RMS velocity change (normalized action dimensions)")
    ax.legend()
    ax.grid(alpha=0.2)
    return _save_figure(fig,out,"fig03_time_input_counterfactual")


def plot_alignment_evidence(plots, raw_rope, raw_temporal, anchor_metadata, out):
    """A prediction example and paired errors; all plots retain unfavorable results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"rope": "#486FA4", "correct": "#237B59", "zero": "#7B8491", "reversed": "#C96334"}
    labels = {"rope": "RoPE (separate model)", "correct": "Correct time", "zero": "Time branch off",
              "reversed": "Reversed time"}
    steps = np.arange(HORIZON)
    index, selected = representative_condition(raw_temporal["conditions"], anchor_metadata)
    summary = alignment_summary(plots)
    episodes = summary["episode_ids"]
    tm = plots["time_input"]["mse"]
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.5), layout="constrained")
    ax = axes[0, 0]
    ax.plot(steps, raw_temporal["clean_target"][index, :, 6], color="#14202E", lw=2.5,
            label="Recorded target", zorder=5)
    for name in ("rope", "correct", "zero", "reversed"):
        values = (raw_rope["clean_estimate"]["correct"] if name == "rope"
                  else raw_temporal["clean_estimate"][name])
        ax.plot(steps, values[index, :, 6], color=colors[name],
                ls="--" if name == "rope" else "-", lw=1.6, alpha=.9, label=labels[name])
    ax.set_title(
        "A. Preselected window: one-step gripper estimate\n"
        f"demo {selected['episode_index']}, frame {selected['frame_index']}, "
        f"t={selected['flow_time']:g}, noise={selected['noise_seed']}", fontsize=11,
    )
    ax.set_xlabel("Future action index (0–49)")
    ax.set_ylabel("Normalized gripper coordinate")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.18)

    ax = axes[0, 1]
    order = ("rope", "correct", "zero", "reversed")
    for i, name in enumerate(order):
        values = np.asarray(summary["paired_episode_mse"][name])
        ax.bar(i, values.mean(), width=.56, color=colors[name], alpha=.18)
        ax.scatter(i + np.linspace(-.16, .16, len(values)), values, color=colors[name],
                   s=30, zorder=3, edgecolors="white", linewidths=.35)
        ax.plot([i-.23, i+.23], [values.mean()]*2, color=colors[name], lw=2.5)
    ax.set_xticks(range(4), ["RoPE\nseparate", "Correct", "Zero", "Reversed"])
    ax.set_ylabel("Normalized flow-velocity MSE (lower is better)")
    ax.set_title("B. Every dot is one demonstration; lines mark means", fontsize=11)
    ax.grid(axis="y", alpha=.18)

    ax = axes[1, 0]
    delta_rows, row_labels = [], []
    for i, episode in enumerate(episodes):
        for key, short in (("zero", "Zero"), ("reversed", "Rev")):
            delta_rows.append(tm[key][i] - tm["correct"][i])
            row_labels.append(f"{episode} · {short}")
    heat = np.stack(delta_rows)
    limit = max(float(np.abs(heat).max()), 1e-10)
    img = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                    interpolation="nearest")
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontsize=8)
    ax.set_xlabel("Future action index (0–49)")
    ax.set_ylabel("Demonstration · intervention")
    ax.set_title("C. Error change when correct time alignment is perturbed", fontsize=11)
    for y in range(2, len(row_labels), 2):
        ax.axhline(y-.5, color="white", linewidth=.6)
    fig.colorbar(img, ax=ax, shrink=.84, label="ΔMSE = intervention − Correct")

    ax = axes[1, 1]
    comparisons = (("zero_minus_correct", "Zero − Correct"),
                   ("reversed_minus_correct", "Reversed − Correct"),
                   ("correct_minus_rope", "Correct − RoPE"))
    for i, (key, _) in enumerate(comparisons):
        values = np.asarray(summary["paired_episode_deltas"][key])
        ax.scatter(i + np.linspace(-.16, .16, len(values)), values, color="#344C6A", s=31)
        ax.plot([i-.23, i+.23], [values.mean()]*2, color="#14202E", lw=2.4)
        ax.annotate(f"mean {values.mean():+.4g}", (i, values.mean()),
                    xytext=(0, 11), textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(0, color="#14202E", linewidth=1)
    ax.set_xticks(range(3), [label for _, label in comparisons], fontsize=9)
    ax.set_ylabel("Paired demonstration MSE difference")
    ax.set_title("D. Positive = the first named condition has larger error", fontsize=11)
    ax.grid(axis="y", alpha=.18)
    fig.suptitle("Does correct time alignment help prediction?", fontsize=16, weight="bold")
    fig.supxlabel(
        "Correct / Zero / Reversed share weights, observations, targets, noise and RoPE. "
        "Warm cells: intervention hurts; cool cells: intervention helps.\n"
        "A is an offline one-step estimate from a recorded-action interpolation point, not a rollout. "
        "Noise draws, Flow times and windows are averaged within each demonstration.",
        fontsize=9,
    )
    return _save_figure(fig, out, "fig04_time_alignment_evidence"), selected


def plot_residual(residuals, fps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix=np.mean(residuals,axis=0)
    norms=np.linalg.norm(matrix,axis=-1)
    directions=matrix/np.maximum(norms[:,None],1e-10)
    cosine=directions@directions.T
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12.3,4.6),layout="constrained")
    ax1.plot(np.arange(HORIZON)/fps, norms)
    ax1.set_title("Actual trained 32→1024 time-residual norm")
    ax1.set_xlabel("Seconds from start of action chunk")
    ax1.set_ylabel("L2 norm (Action Expert embedding units)")
    ax1.grid(alpha=0.2)
    heat=ax2.imshow(cosine,vmin=-1,vmax=1,origin="lower",aspect="auto",cmap="coolwarm")
    ax2.set_title("Cosine similarity: learned time residuals across action positions")
    ax2.set_xlabel("Future action index")
    ax2.set_ylabel("Future action index")
    fig.colorbar(heat,ax=ax2,shrink=.8,label="Cosine similarity")
    return _save_figure(fig,out,"fig02_learned_time_residual")


def resolve_probe_checkpoints(args, record):
    original_jobs={
        job["case"]:job for job in timepath.build_jobs(
            record,args.directory,args.output_root,repeat=args.repeat,
        )
    }
    return {
        "rope":Path(args.rope_checkpoint).expanduser().resolve()
        if getattr(args, "rope_checkpoint", None) is not None
        else original_jobs["rope"]["checkpoint"],
        "time_input":Path(args.time_input_checkpoint).expanduser().resolve()
        if args.time_input_checkpoint is not None
        else original_jobs["time_input"]["checkpoint"],
    }


def validate_training_provenance(checkpoints, record):
    """Verify which fine-tuning demo must be excluded before claiming held-out data."""
    sources = {}
    for case, checkpoint in checkpoints.items():
        checkpoint = Path(checkpoint)
        if len(checkpoint.parents) < 3:
            raise ValueError(f"{case}: checkpoint is not in a recorded training run")
        descriptor = checkpoint.parents[2] / "gripper_event_experiment.json"
        if not descriptor.is_file():
            raise ValueError(
                f"{case}: missing training provenance {descriptor}; "
                "cannot verify which demonstration to exclude"
            )
        saved = _json(descriptor)
        if saved.get("preparation") != record:
            raise ValueError(f"{case}: checkpoint training plan differs from the probe plan")
        sources[case] = {"descriptor":str(descriptor), "sha256":previous.file_hash(descriptor)}
    return sources


def validate_comparison_plan(report, directory):
    settings = report.get("lock", {}).get("settings") if "lock" in report else report.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Comparison report has no training-plan provenance")
    manifest = Path(directory) / "manifest.json"
    if settings.get("preparation_sha256") != previous.file_hash(manifest):
        raise ValueError("Comparison report training plan differs from --directory; cannot select held-out demos")


def probe(args, record):
    flow_times, noise_seeds = probe_grid(args)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("No CUDA; --allow-cpu is for small debugging only")
    device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoints = resolve_probe_checkpoints(args, record)
    for path in checkpoints.values():
        if not (path/"model.safetensors").is_file():
            raise FileNotFoundError(f"Missing trained 1000-step checkpoint: {path}")
    training_provenance = validate_training_provenance(checkpoints, record)
    # Check training recipe alignment even when custom Time-Input path supplied.
    source_config=_json(checkpoints["rope"]/"config.json")
    time_config=_json(checkpoints["time_input"]/"config.json")
    required={"train_action_expert":True,"num_vlm_prompt_tokens":2,
              "num_prompt_tokens":0,"chunk_size":HORIZON}
    for label,cfg in (("rope",source_config),("time_input",time_config)):
        for key,val in required.items():
            if cfg.get(key)!=val:
                raise ValueError(f"{label}: wrong {key}")
        if cfg.get("action_time_basis", "index") != "index":
            raise ValueError(
                f"{label}: offline probe supports action_time_basis=index only. "
                "Physical/global_dt checkpoints require an explicit timestamp-aware probe."
            )
    if source_config.get("action_time_encoding") is not False:
        raise ValueError("RoPE reference unexpectedly has a learned time encoder")
    if (
        time_config.get("action_time_encoding") is not True
        or time_config.get("action_time_injection")!="input"
    ):
        raise ValueError("Time-Input checkpoint is not an input-time model")
    for key in ("normalization_mapping","output_features","input_features"):
        if source_config.get(key)!=time_config.get(key):
            raise ValueError(f"Checkpoints differ in {key}; cannot compare offline MSE directly")
    anchors,dataset,actions,metadata=_probe_data(args,record)
    anchor_metadata = describe_anchors(metadata, anchors)
    n_episodes = len({row["episode_index"] for row in anchor_metadata})
    result={"protocol":PROTOCOL, "anchors":anchors, "anchor_metadata":anchor_metadata,
            "checkpoints":{k:str(v) for k,v in checkpoints.items()},
            "training_provenance":training_provenance}
    print(f"Offline probe: {n_episodes} non-training demos, {len(anchors)} complete windows, "
          f"{len(flow_times)} Flow times, {len(noise_seeds)} fixed noises",flush=True)
    rope=_compute_probe_for_model(
        "rope",checkpoints["rope"],dataset,actions,anchors,args,device,
        interventions=("correct",),
    )
    # Both policies must use EXACTLY the same normalized target and noise.
    # Different processors/stats could make their velocity MSE incomparable.
    tm=_compute_probe_for_model(
        "time_input",checkpoints["time_input"],dataset,actions,anchors,args,device,
        interventions=("correct","zero","reversed"),
    )
    if not np.allclose(
        rope["normalized_targets"],tm["normalized_targets"],rtol=1e-5,atol=1e-6,
    ):
        raise ValueError("RoPE and Time-Input normalized targets differ; MSE comparison invalid")
    if rope["conditions"] != tm["conditions"]:
        raise ValueError("RoPE and Time-Input condition identities differ")
    if len(tm["conditions"]) != len(anchors) * len(flow_times) * len(noise_seeds):
        raise ValueError("Incomplete predeclared Flow/noise/anchor grid")
    lookup = {(row["episode_index"], row["global_frame_index"]): row for row in anchor_metadata}
    for model in (rope, tm):
        model["conditions"] = [
            {**lookup[(row["episode_index"], row["global_frame_index"])], **row}
            for row in model["conditions"]
        ]
    plots = aggregate_probe(rope, tm)
    alignment = alignment_summary(plots)
    alignment.update({
        "protocol":PROTOCOL, "n_episodes":n_episodes, "n_anchors":len(anchors),
        "n_flow":len(flow_times), "n_noise":len(noise_seeds),
        "n_conditions":len(tm["conditions"]),
        "flow_times":flow_times, "noise_seeds":noise_seeds,
        "anchor_metadata":anchor_metadata,
    })
    result["mse_by_variant"] = alignment["mse_by_variant"]
    # Preserve raw rows for reproducibility, clearly named as conditions.
    result["paired_condition_mse"]={
        "rope":rope["mse"]["correct"].mean(axis=1).tolist(),
        **{key:val.mean(axis=1).tolist() for key,val in tm["mse"].items()},
    }
    result["paired_anchor_mse"]={
        name: [float(np.mean([
            result["paired_condition_mse"][name][i] for i, row in enumerate(tm["conditions"])
            if row["global_frame_index"] == idx and row["episode_index"] == episode
        ])) for episode, idx in anchors]
        for name in ("rope", "correct", "zero", "reversed")
    }
    result["per_step_mean_mse"]={
        "rope":plots["rope"]["mse"]["correct"].mean(axis=0).tolist(),
        **{key:val.mean(axis=0).tolist() for key,val in plots["time_input"]["mse"].items()},
    }
    result["time_signal_output_rms"]={
        name:float(plots["output_change_rms"][name].mean())
        for name in ("zero","reversed")
    }
    result["time_residual_relative_rms"]=float(plots["time_input"]["ratio"].mean())
    result["time_residual_nonzero"]=bool(np.linalg.norm(tm["residual"])>0)
    result["checkpoint_fingerprint"]={
        name:{
            "config_sha256":previous.file_hash(path/"config.json"),
            "weights_size":(path/"model.safetensors").stat().st_size,
            "weights_mtime_ns":(path/"model.safetensors").stat().st_mtime_ns,
        }
        for name,path in checkpoints.items()
    }
    result["probe_source_sha256"]=previous.file_hash(Path(__file__))
    result["noise_rng_protocol"]=NOISE_PROTOCOL
    result["number_of_anchors"]=len(anchors)
    result["number_of_demonstrations"]=n_episodes
    result["number_of_conditions"]=len(tm["conditions"])
    result["source_dataset"]=record["dataset_root"]
    result["training_episode_excluded"]=int(record["selection"]["episode_index"])
    result["flow_times"]=flow_times
    result["fixed_noise_seeds"]=noise_seeds
    result["conditions"]=tm["conditions"]
    result["flow_time"]=flow_times[0] if len(flow_times)==1 else None
    result["fixed_noise_seed"]=noise_seeds[0] if len(noise_seeds)==1 else None
    result["anchor_selection"]={
        "probe_demos_requested":args.probe_demos,
        "windows_per_demo_requested":args.windows_per_demo,
        "legacy_total_window_cap":args.samples,
        "rule":"Evenly spaced eligible demos; evenly spaced complete chunk starts from early to late",
        "overlapping_windows_are_clustered_by_demonstration":True,
    }
    result["note"]=(
        "Different demos from selected single training episode, but pretrained "
        "source has seen LIBERO. Zero/reversed time residuals induce train-test "
        "mismatch; inference-time effect is not causal proof of better generalization."
    )
    result["note"] += (
        " All windows/noises/t are averaged within demonstration before between-demo summaries. "
        "Recorded demos do not reproduce the shifted-scene rollouts."
    )
    args.output.mkdir(parents=True,exist_ok=True)
    fig2=plot_residual(plots["time_input"]["residual"],record["fps"],args.output)
    fig3=plot_probe(plots,args.output)
    fig4, selected = plot_alignment_evidence(plots,rope,tm,anchor_metadata,args.output)
    alignment["representative_condition"] = selected
    alignment["figure_paths"] = fig4
    alignment["note"] = result["note"]
    alignment["checkpoints"] = result["checkpoints"]
    alignment["checkpoint_fingerprint"] = result["checkpoint_fingerprint"]
    alignment["training_provenance"] = training_provenance
    alignment["probe_source_sha256"] = result["probe_source_sha256"]
    alignment["noise_rng_protocol"] = NOISE_PROTOCOL
    result["time_alignment_summary"] = alignment
    result["figure_paths"]=fig2+fig3+fig4
    arrays = {
        "rope_mse":rope["mse"]["correct"],
        "rope_velocity":rope["pred"]["correct"],
        "rope_clean_estimate":rope["clean_estimate"]["correct"],
        "clean_target":tm["clean_target"],
        "time_residual_relative_rms":tm["ratio"],
        "episode_mean_time_residual":plots["time_input"]["residual"],
        **{f"{key}_mse":val for key,val in tm["mse"].items()},
        **{f"{key}_velocity":val for key,val in tm["pred"].items()},
        **{f"{key}_clean_estimate":val for key,val in tm["clean_estimate"].items()},
    }
    arrays_path = args.output / "offline_probe_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    result["array_file"] = str(arrays_path)
    alignment["array_file"] = str(arrays_path)
    result["array_row_order"] = "Rows follow conditions; episode_mean_time_residual follows time_alignment_summary.episode_ids"
    (args.output/"time_alignment_evidence.json").write_text(
        json.dumps(alignment,indent=2)+"\n", encoding="utf-8",
    )
    with (args.output/"offline_probe.json").open("w") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    for name,value in result["mse_by_variant"].items():
        print(f"  {name:>12}: normalized flow MSE={value:.6f}")
    print("  Alignment evidence:", alignment["observed_direction"],flush=True)
    print("  Mean paired deltas:", alignment["mean_paired_deltas"],flush=True)
    print("  Saved:",*result["figure_paths"],sep="\n   ",flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("success","probe","all"),default="success")
    parser.add_argument("--directory",type=Path,default=timepath.DEFAULT_PLAN)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-root",type=Path,default=timepath.DEFAULT_EVAL)
    parser.add_argument("--output",type=Path,default=DEFAULT_EVIDENCE)
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--comparison-json",type=Path,default=None,
                        help="Saved dual-seed/fixed-protocol comparison for success/all")
    parser.add_argument("--seeds",default="1000,3000,4000,5000")
    parser.add_argument("--shift-m",type=float,default=0.12)
    parser.add_argument("--shift-seed",type=int,default=20261103)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--probe-demos",type=int,default=8,
                        help="Maximum non-training demonstrations, selected without looking at outcomes")
    parser.add_argument("--windows-per-demo",type=int,default=3,
                        help="Complete windows spread from early to late in each demonstration")
    parser.add_argument("--samples",type=int,default=None,
                        help="Legacy optional TOTAL window cap; default uses demos x windows-per-demo")
    times = parser.add_mutually_exclusive_group()
    times.add_argument("--flow-time",type=float,default=None,
                       help="Explicit single-time diagnostic instead of the default three-point grid")
    times.add_argument("--flow-times",default=None,
                       help="Comma-separated Flow times; default 0.2,0.5,0.8")
    noises = parser.add_mutually_exclusive_group()
    noises.add_argument("--noise-seed",type=int,default=None,
                        help="Explicit single-noise diagnostic instead of the default three seeds")
    noises.add_argument("--noise-seeds",default=None,
                        help="Comma-separated fixed noise seeds; default 3000,4000,5000")
    parser.add_argument("--tokenizer",type=Path,default=Path("/home/wyn/models/paligemma-3b-pt-224"))
    parser.add_argument("--dataset-repo-id",default="libero")
    parser.add_argument("--time-input-checkpoint",type=Path,default=None)
    parser.add_argument("--rope-checkpoint",type=Path,default=None)
    parser.add_argument("--allow-cpu",action="store_true")
    args=parser.parse_args(argv)
    try:
        if args.gpu<0 or args.repeat<1 or args.shift_m<0 or args.shift_m>0.14:
            raise ValueError("Invalid GPU, repeat or fixed shift")
        if args.probe_demos < 1 or args.windows_per_demo < 1 or (args.samples is not None and args.samples < 1):
            raise ValueError("Demo/window counts and an optional sample cap must be positive")
        probe_grid(args)
        if args.comparison_json is not None and args.mode == "probe":
            raise ValueError("--comparison-json applies to success/all; use --mode all or pass explicit checkpoints")
        args.seeds=parse_seeds(args.seeds)
        record = (None if args.mode == "success" and args.comparison_json is not None
                  else timepath.check_plan(args.directory))
        args.output=args.output.resolve()
        args.output.mkdir(parents=True,exist_ok=True)
        if args.mode in ("success","all") and args.comparison_json is not None:
            from lerobot.scripts.pi05_task7_dualseed_evidence import run_success_report, validate_comparison

            raw_comparison = _json(args.comparison_json)
            saved = validate_comparison(raw_comparison)
            if args.mode == "all":
                validate_comparison_plan(raw_comparison, args.directory)
                for case, option in (("rope", "rope_checkpoint"), ("time_input", "time_input_checkpoint")):
                    expected = Path(saved["checkpoints"][case]).expanduser().resolve()
                    explicit = getattr(args, option)
                    if explicit is not None and explicit.expanduser().resolve() != expected:
                        raise ValueError(f"{case}: probe checkpoint differs from the comparison report")
                    setattr(args, option, expected)
            run_success_report(args.comparison_json,args.output)
        elif args.mode in ("success","all"):
            matrix,manifests=validated_success_matrix(args,record)
            path=plot_success(matrix,args.seeds,args.output)
            outcome={
                "protocol":PROTOCOL,"seeds":args.seeds,
                "cases":list(CASES),"data":matrix.tolist(),
                "physics_manifests":manifests,
                "successes":{case:int(matrix[i].sum()) for i,case in enumerate(CASES)},
                "paired_rope_time":paired_counts(matrix[0],matrix[3]),
                "figure_paths":path,
                "caveat":"Original init state IDs reused; no independent training seeds.",
            }
            (args.output/"paired_success.json").write_text(json.dumps(outcome,indent=2)+"\n")
            print("Saved paired outcomes:",*path,sep="\n  ",flush=True)
            print("Time-Input vs RoPE pairs:",outcome["paired_rope_time"],flush=True)
        if args.mode in ("probe","all"):
            probe(args,record)
    except (OSError,ValueError,KeyError,RuntimeError) as err:
        parser.exit(1,f"Time-Input visualization failed: {err}\n")


if __name__ == "__main__":
    main()

