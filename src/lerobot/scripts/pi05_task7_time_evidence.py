"""Empirical evidence for the RoPE + action-input temporal residual hypothesis.

Modes:
  success : CPU-only matched 4-arm task7 rollout episode heatmap and paired wins
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

PROTOCOL = "pi05-task7-time-input-evidence-v1"
CASES = timepath.CASES
HORIZON = 50
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path("/data1/wyn/chkpt/2601-lerobot/PromptVLA")
DEFAULT_EVIDENCE = DEFAULT_OUTPUT / "analysis" / "task7-rope-time-input"


def parse_seeds(value):
    try:
        seeds = [int(token) for token in value.split(",")]
    except ValueError as error:
        raise ValueError("Expected comma-separated integer seeds") from error
    if not seeds or any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    return seeds


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
    jobs = timepath.build_jobs(record, args.directory, args.output_root)
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
    message=f"{PROTOCOL}:flow:{seed}:{frame_index}"
    generator=torch.Generator(device="cpu").manual_seed(
        int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little") & ((1<<63)-1)
    )
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def select_nontraining_anchors(frame_metadata, *, task_index, train_episode, count=8,
                               horizon=HORIZON):
    """Take anchors strictly from DIFFERENT demos of the exact task, no future labels leak."""
    if count < 1:
        raise ValueError("count must be positive")
    task=np.asarray(frame_metadata["task_index"])
    ep=np.asarray(frame_metadata["episode_index"])
    frame=np.asarray(frame_metadata["frame_index"])
    choices=[]
    for episode in sorted(int(e) for e in np.unique(ep[task==task_index])
                          if int(e) != train_episode):
        idx=np.flatnonzero((task==task_index)&(ep==episode))
        if len(idx)<horizon:
            continue
        if not np.array_equal(frame[idx], np.arange(len(idx))):
            raise ValueError("Probe episode frame indices are not continuous")
        if not np.array_equal(idx, np.arange(idx[0],idx[-1]+1)):
            raise ValueError("Probe episode source data are not consecutive")
        eligible=idx[:len(idx)-horizon+1]
        # Disjoint windows within an episode whenever possible: deterministic.
        for frame_id in eligible[::horizon]:
            choices.append((int(episode), int(frame_id)))
    if not choices:
        raise ValueError(
            "No other complete task7 demonstration exists. Refusing to call "
            "training-demonstration reconstruction a heldout evaluation."
        )
    # Deterministic stratification over eligible episodes; no training demo.
    available = sorted(set(e for e,_ in choices))
    queues={e:[start for ep_,start in choices if ep_==e] for e in available}
    picked=[]
    while len(picked)<count and any(queues.values()):
        for e in available:
            if queues[e] and len(picked)<count:
                picked.append((e,queues[e].pop(0)))
    return picked


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
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    checkpoint=Path(checkpoint).resolve(strict=True)
    config=PI05Config.from_pretrained(checkpoint)
    config.device=str(device)
    config.tokenizer_name=str(Path(tokenizer).resolve(strict=True))
    config.compile_model=False
    config.gradient_checkpointing=False
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
    if clean.shape!=noise.shape or clean.shape[1] != HORIZON:
        raise ValueError("Probe action dimensions do not match the fixed Flow noise")
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
    per_step=((pred[:,:,:d]-target[:,:,:d])**2).mean(dim=-1)
    return {
        "per_step_mse":per_step[0].detach().float().cpu().numpy(),
        "pred":pred[0,:,:d].detach().float().cpu().numpy(),
        "x_t":x_t,
        "embedding_before":action_embed,
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
    residuals=[]
    ratios=[]
    normalized_targets=[]
    for episode,idx in anchors:
        batch=_batch_for_index(dataset,actions,idx,processor=processor)
        with torch.inference_mode():
            clean=policy.prepare_action(batch)
        normalized_targets.append(clean.detach().float().cpu().numpy())
        noise=stable_noise(args.noise_seed,idx,tuple(clean.shape)).to(device)
        for mode in interventions:
            result=_probe_predict(policy,batch,noise,args.flow_time,intervention=mode)
            results[mode].append(result["per_step_mse"].tolist())
            predictions[mode].append(result["pred"])
            if mode=="correct" and policy.model.action_time_encoder is not None:
                residual,ratio=_record_time_residual(
                    policy.model,result["embedding_before"]
                )
                residuals.append(residual)
                ratios.append(ratio)
        print(f"  {label}: episode={episode} anchor={idx} "
              f"correct-MSE={np.mean(results['correct'][-1]):.5f}",flush=True)
    out={
        "mse":{key:np.asarray(val,dtype=np.float64) for key,val in results.items()},
        "pred":{key:np.stack(val) for key,val in predictions.items()},
        "ratio":np.stack(ratios) if ratios else None,
        "residual":np.stack(residuals) if residuals else None,
        "normalized_targets":np.concatenate(normalized_targets,axis=0),
    }
    del policy,processor
    if device.type=="cuda":
        torch.cuda.empty_cache()
    return out


def _curve_mean_ci(values):
    mat=np.asarray(values,dtype=np.float64)
    return mat.mean(axis=0),mat.std(axis=0,ddof=1)/math.sqrt(len(mat)) if len(mat)>1 else np.zeros(mat.shape[1])


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
    ax.set_title("A. Recorded-action flow-velocity error vs action position")
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
    ax.set_title("B. Per-anchor paired MSE differences")
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
        dist=np.sqrt(np.mean((mask[name]-orig)**2,axis=-1))
        ax.plot(steps,dist.mean(axis=0),label=name+" vs correct")
    ax.set_title("D. Change in predicted velocity when time path is perturbed")
    ax.set_xlabel("Future action index")
    ax.set_ylabel("RMS velocity change (normalized action dimensions)")
    ax.legend()
    ax.grid(alpha=0.2)
    return _save_figure(fig,out,"fig03_time_input_counterfactual")


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


def probe(args, record):
    if not 0<args.flow_time<1 or args.samples<1:
        raise ValueError("flow_time must be in (0,1) and samples>=1")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("No CUDA; --allow-cpu is for small debugging only")
    device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    original_jobs={
        job["case"]:job for job in timepath.build_jobs(
            record,args.directory,args.output_root
        )
    }
    checkpoints={
        "rope":original_jobs["rope"]["checkpoint"],
        "time_input":Path(args.time_input_checkpoint).resolve()
        if args.time_input_checkpoint is not None
        else original_jobs["time_input"]["checkpoint"],
    }
    for path in checkpoints.values():
        if not (path/"model.safetensors").is_file():
            raise FileNotFoundError(f"Missing trained 1000-step checkpoint: {path}")
    # Check training recipe alignment even when custom Time-Input path supplied.
    source_config=_json(checkpoints["rope"]/"config.json")
    time_config=_json(checkpoints["time_input"]/"config.json")
    required={"train_action_expert":True,"num_vlm_prompt_tokens":2,
              "num_prompt_tokens":0,"chunk_size":HORIZON}
    for label,cfg in (("rope",source_config),("time_input",time_config)):
        for key,val in required.items():
            if cfg.get(key)!=val:
                raise ValueError(f"{label}: wrong {key}")
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
    result={"anchors":anchors,"checkpoints":{k:str(v) for k,v in checkpoints.items()}}
    print(f"Offline probe on {len(anchors)} non-training demonstrations/anchors",flush=True)
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
    result["mse_by_variant"]={
        "rope":float(rope["mse"]["correct"].mean()),
        **{key:float(val.mean()) for key,val in tm["mse"].items()},
    }
    result["paired_anchor_mse"]={
        "rope":rope["mse"]["correct"].mean(axis=1).tolist(),
        **{key:val.mean(axis=1).tolist() for key,val in tm["mse"].items()},
    }
    result["per_step_mean_mse"]={
        "rope":rope["mse"]["correct"].mean(axis=0).tolist(),
        **{key:val.mean(axis=0).tolist() for key,val in tm["mse"].items()},
    }
    result["time_signal_output_rms"]={
        name:float(np.sqrt(np.mean((tm["pred"][name]-tm["pred"]["correct"])**2)))
        for name in ("zero","reversed")
    }
    result["time_residual_relative_rms"]=float(tm["ratio"].mean())
    result["time_residual_nonzero"]=bool(np.linalg.norm(tm["residual"])>0)
    result["checkpoint_fingerprint"]={
        name:{
            "config_sha256":previous.file_hash(path/"config.json"),
            "weights_size":(path/"model.safetensors").stat().st_size,
            "weights_mtime_ns":(path/"model.safetensors").stat().st_mtime_ns,
        }
        for name,path in checkpoints.items()
    }
    result["number_of_anchors"]=len(anchors)
    result["source_dataset"]=record["dataset_root"]
    result["flow_time"]=args.flow_time
    result["fixed_noise_seed"]=args.noise_seed
    result["note"]=(
        "Different demos from selected single training episode, but pretrained "
        "source has seen LIBERO. Zero/reversed time residuals induce train-test "
        "mismatch; inference-time effect is not causal proof of better generalization."
    )
    fig2=plot_residual(tm["residual"],record["fps"],args.output)
    fig3=plot_probe({"rope":rope,"time_input":tm},args.output)
    result["figure_paths"]=fig2+fig3
    with (args.output/"offline_probe.json").open("w") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    for name,value in result["mse_by_variant"].items():
        print(f"  {name:>12}: normalized flow MSE={value:.6f}")
    print("  Saved:",*result["figure_paths"],sep="\n   ",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("success","probe","all"),default="success")
    parser.add_argument("--directory",type=Path,default=timepath.DEFAULT_PLAN)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-root",type=Path,default=timepath.DEFAULT_EVAL)
    parser.add_argument("--output",type=Path,default=DEFAULT_EVIDENCE)
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--seeds",default="1000,3000,4000,5000")
    parser.add_argument("--shift-m",type=float,default=0.12)
    parser.add_argument("--shift-seed",type=int,default=20261103)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--samples",type=int,default=8)
    parser.add_argument("--flow-time",type=float,default=0.5)
    parser.add_argument("--noise-seed",type=int,default=3000)
    parser.add_argument("--tokenizer",type=Path,default=Path("/home/wyn/models/paligemma-3b-pt-224"))
    parser.add_argument("--dataset-repo-id",default="libero")
    parser.add_argument("--time-input-checkpoint",type=Path,default=None)
    parser.add_argument("--allow-cpu",action="store_true")
    args=parser.parse_args()
    try:
        if args.gpu<0 or args.repeat<1 or args.shift_m<0 or args.shift_m>0.14:
            raise ValueError("Invalid GPU, repeat or fixed shift")
        args.seeds=parse_seeds(args.seeds)
        record=timepath.check_plan(args.directory)
        args.output=args.output.resolve()
        args.output.mkdir(parents=True,exist_ok=True)
        if args.mode in ("success","all"):
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
