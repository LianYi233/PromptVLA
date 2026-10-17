#!/usr/bin/env python3
"""Compare PI05 actions with recorded Piper training labels, without robot hardware."""

import argparse
import csv
import hashlib
import json
import math
import random
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value):
    return as_numpy(value).item()


def select_indices(rows, stride, max_samples):
    """Select relative HF rows, never confuse them with absolute dataset indices."""
    candidates = sorted(
        (int(scalar(row["episode_index"])), int(scalar(row["frame_index"])), position)
        for position, row in enumerate(rows)
        if int(scalar(row["frame_index"])) % stride == 0
    )
    if not candidates:
        raise ValueError("没有可评估的样本")
    if max_samples and len(candidates) > max_samples:
        offsets = np.linspace(0, len(candidates) - 1, max_samples, dtype=int)
        candidates = [candidates[i] for i in offsets]
    return [position for _, _, position in candidates]


def parse_episodes(text, total, training_episodes=None):
    allowed = set(range(total)) if training_episodes is None else set(training_episodes)
    selected = sorted(allowed) if text == "all" else [int(s) for s in text.split(",")]
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= allowed:
        raise ValueError("episodes 必须是不重复的训练 episode 编号，或 all")
    if min(selected) < 0 or max(selected) >= total:
        raise ValueError("episode 编号超出数据集范围")
    return sorted(selected)


def recorded_targets(item, episode, chunk_size, dimension):
    target = as_numpy(item["action"]).astype(np.float32, copy=True)
    padding = as_numpy(item["action_is_pad"])
    if target.shape != (chunk_size, dimension) or padding.shape != (chunk_size,):
        raise ValueError(f"动作/掩码形状错误: {target.shape}, {padding.shape}")
    if padding.dtype != np.bool_:
        raise ValueError("action_is_pad 必须是布尔掩码")
    index, frame = int(scalar(item["index"])), int(scalar(item["frame_index"]))
    start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
    if index != start + frame or not start <= index < end:
        raise ValueError("episode/frame_index 与绝对 index 不一致，拒绝错位比较")
    expected_valid = index + np.arange(chunk_size) < end
    if not np.array_equal(~padding, expected_valid):
        raise ValueError("动作时间窗口与 episode 边界不符，不能跨 episode 比较")
    if not np.isfinite(target[expected_valid]).all():
        raise ValueError("有效 action 标签含 NaN/Inf")
    return target, expected_valid


def observation_inputs(item, input_keys):
    # Intentionally exclude action, future states, padding and all dataset bookkeeping.
    task = item["task"]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("数据集样本缺少 task 文本")
    return {**{key: deepcopy(item[key]) for key in input_keys}, "task": task}


def remove_empty_action(batch):
    # transition_to_batch always adds action=None, even for inference-only input.
    # Permit that placeholder, but reject any actual target tensor.
    if batch.pop("action", None) is not None:
        raise ValueError("推理输入不得包含 action 标签")
    return batch


def hold_baseline(state, state_names, action_names, chunk_size):
    state = as_numpy(state)
    if state.shape != (len(state_names),) or not np.isfinite(state).all():
        raise ValueError("当前 observation.state 形状错误或含 NaN/Inf")
    order = [state_names.index(name) for name in action_names]
    return np.repeat(state[order][None, :], chunk_size, axis=0)


def error_metrics(predicted, target, valid, names, horizon):
    error = (predicted[:, :horizon].astype(np.float64) - target[:, :horizon])[valid[:, :horizon]]
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("没有有效误差或预测含 NaN/Inf")
    per_dimension = {}
    for i, name in enumerate(names):
        absolute = np.abs(error[:, i])
        per_dimension[name] = {
            "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(error[:, i] ** 2))),
            "bias": float(error[:, i].mean()),
            "p95_absolute_error": float(np.quantile(absolute, 0.95)),
            "max_absolute_error": float(absolute.max()),
        }
    joints = [i for i, name in enumerate(names) if name.startswith("joint_")]
    grip = names.index("gripper.pos")
    return {
        "valid_action_pairs": len(error),
        "per_dimension_dataset_units": per_dimension,
        "joint_mae_deg": float(np.abs(error[:, joints]).mean() * 180 / math.pi),
        "joint_rmse_deg": float(np.sqrt(np.mean(error[:, joints] ** 2)) * 180 / math.pi),
        "gripper_mae_dataset_units": float(np.abs(error[:, grip]).mean()),
    }


def summarize(arrays, names, horizons):
    results = {}
    for horizon in sorted(set(horizons)):
        model = error_metrics(arrays["predicted"], arrays["target"], arrays["valid"], names, horizon)
        hold = error_metrics(arrays["hold"], arrays["target"], arrays["valid"], names, horizon)
        denom = hold["joint_mae_deg"]
        results[str(horizon)] = {
            "model": model,
            "hold_current_state": hold,
            "joint_mae_ratio_to_hold": model["joint_mae_deg"] / denom if denom > 1e-12 else None,
        }
    return results


def write_results(output, arrays, names, summary):
    np.savez_compressed(output / "predictions.npz", **arrays, action_names=np.asarray(names))
    fields = ["episode", "anchor_frame", "anchor_index", "offset", "label_frame", "label_index"]
    fields += [f"{kind}/{name}" for kind in ("predicted", "target", "hold") for name in names]
    with (output / "actions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for i, mask in enumerate(arrays["valid"]):
            for offset in np.flatnonzero(mask):
                frame, index = int(arrays["frame_index"][i]), int(arrays["index"][i])
                row = [
                    int(arrays["episode_index"][i]),
                    frame,
                    index,
                    int(offset),
                    frame + offset,
                    index + offset,
                ]
                for kind in ("predicted", "target", "hold"):
                    row.extend(arrays[kind][i, offset].tolist())
                writer.writerow(row)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["horizon", "predictor", "dimension", "unit", "count", "mae", "rmse", "bias", "p95", "max"]
        )
        for horizon, comparisons in summary["horizons"].items():
            for comparator in ("model", "hold_current_state"):
                metrics = comparisons[comparator]
                for name, values in metrics["per_dimension_dataset_units"].items():
                    unit = "rad" if name.startswith("joint_") else "dataset_gripper_unit"
                    writer.writerow(
                        [horizon, comparator, name, unit, metrics["valid_action_pairs"], *values.values()]
                    )
    # Write the completion marker last, so interrupted runs cannot look complete.
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def plot_sample(output, item, predicted, target, valid, names, cameras, fps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 15), layout="constrained")
    grid = fig.add_gridspec(8, max(2, len(cameras)), height_ratios=[2.3] + [1] * 7)
    for i, camera in enumerate(cameras):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(np.moveaxis(as_numpy(item[camera]), 0, -1))
        ax.set_title(camera.removeprefix("observation.images."))
        ax.axis("off")
    seconds = np.flatnonzero(valid) / fps
    for d, name in enumerate(names):
        ax = fig.add_subplot(grid[d + 1, :])
        scale = 180 / math.pi if name.startswith("joint_") else 1
        ax.plot(seconds, target[valid, d] * scale, label="teleoperation", color="#252525", linewidth=1.8)
        ax.plot(
            seconds, predicted[valid, d] * scale, label="action from model", color="#1261A0", linewidth=1.5
        )
        label = name.replace("joint_", "J").removesuffix(".pos")
        ax.set_ylabel(label + " (deg)" if scale != 1 else "Gripper\n(dataset unit)")
        ax.grid(alpha=0.2)
        if d == 0:
            ax.legend(loc="best", ncol=2)
        if d == 6:
            ax.set_xlabel("Seconds after recorded observation (offset 0 = current action label)")
        else:
            ax.tick_params(labelbottom=False)
    ep, frame = int(scalar(item["episode_index"])), int(scalar(item["frame_index"]))
    fig.suptitle(f"Recorded episode {ep}, frame {frame} | {item['task']}", fontsize=11)
    fig.savefig(output / f"episode_{ep:06d}_frame_{frame:06d}.png", dpi=150)
    plt.close(fig)


def evaluate(dataset, indices, predict, features, output, *, chunk_size, execution_steps, plots, manifest):
    names = features["action"]["names"]
    state_names = features["observation.state"]["names"]
    cameras = [key for key in features if key.startswith("observation.images.")]
    input_keys = ["observation.state", *cameras]
    plot_positions = set(np.linspace(0, len(indices) - 1, min(plots, len(indices)), dtype=int))
    values = {
        key: []
        for key in ("predicted", "target", "valid", "hold", "episode_index", "frame_index", "index", "task")
    }
    for position, relative_index in enumerate(indices):
        item = dataset[relative_index]
        episode_id = int(scalar(item["episode_index"]))
        target, valid = recorded_targets(item, dataset.meta.episodes[episode_id], chunk_size, len(names))
        # Independent check of offset 0 against the unexpanded parquet label.
        raw_label = as_numpy(dataset.get_raw_item(relative_index)["action"])
        if not np.array_equal(target[0], raw_label):
            raise ValueError("动作窗口 offset 0 与当前帧 action 不一致")
        hold = hold_baseline(item["observation.state"], state_names, names, chunk_size)
        observation = observation_inputs(item, input_keys)
        prediction = np.asarray(predict(observation, int(scalar(item["index"]))), dtype=np.float32)
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError(f"预测 action 形状错误或含 NaN/Inf: {prediction.shape}")
        for key, value in (("predicted", prediction), ("target", target), ("valid", valid), ("hold", hold)):
            values[key].append(value)
        for key in ("episode_index", "frame_index", "index"):
            values[key].append(int(scalar(item[key])))
        values["task"].append(item["task"])
        if position in plot_positions:
            plot_sample(output, item, prediction, target, valid, names, cameras, dataset.fps)
        if (position + 1) % 10 == 0 or position + 1 == len(indices):
            print(f"Offline samples: {position + 1}/{len(indices)}", flush=True)
    arrays = {key: np.asarray(value) for key, value in values.items()}
    horizons = [1, execution_steps, chunk_size]
    summary = {
        **manifest,
        "completed": True,
        "num_observations": len(indices),
        "action_names": names,
        "state_names": state_names,
        "fps": dataset.fps,
        "horizons": summarize(arrays, names, horizons),
        "per_episode_execution_horizon": {},
    }
    for ep in np.unique(arrays["episode_index"]):
        subset = {key: value[arrays["episode_index"] == ep] for key, value in arrays.items()}
        summary["per_episode_execution_horizon"][str(ep)] = summarize(subset, names, [execution_steps])[
            str(execution_steps)
        ]
    write_results(output, arrays, names, summary)
    print("horizon | joint MAE deg (model / hold) | gripper MAE dataset units (model / hold)")
    for horizon, result in summary["horizons"].items():
        a, b = result["model"], result["hold_current_state"]
        print(
            f"{horizon:>7} | {a['joint_mae_deg']:.4f} / {b['joint_mae_deg']:.4f} | "
            f"{a['gripper_mae_dataset_units']:.5f} / {b['gripper_mae_dataset_units']:.5f}"
        )
    return summary


def make_predictor(policy, preprocessor, postprocessor, config, device, seed):
    import torch
    from torch.utils.data import default_collate

    def predict(observation, index):
        for component in (policy, preprocessor, postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()
        batch = default_collate([observation])
        for key in config.image_features:
            if batch[key].dtype != torch.uint8:
                raise ValueError("数据读取应返回 uint8 RGB 图像，避免重复归一化")
            batch[key] = batch[key].float() / 255.0
        batch = remove_empty_action(preprocessor(batch))
        # Same anchor gets identical noise across 6000/9000/12000 comparisons,
        # even when a different episode subset or sampling stride is requested.
        generator = torch.Generator(device=device).manual_seed((seed + index) % (2**63 - 1))
        noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator, device=device)
        amp = (
            torch.autocast(device_type="cuda") if device.type == "cuda" and config.use_amp else nullcontext()
        )
        with torch.inference_mode(), amp:
            normalized = policy.predict_action_chunk(batch, noise=noise)
            expected = (1, config.chunk_size, config.output_features["action"].shape[0])
            if tuple(normalized.shape) != expected:
                raise ValueError(f"模型动作形状不符: {tuple(normalized.shape)}, expected={expected}")
            # Keep the postprocessor's normal [batch, action_dim] interface.
            predicted = torch.stack(
                [postprocessor(normalized[:, h]) for h in range(config.chunk_size)], dim=1
            )
        return predicted[0].detach().float().cpu().numpy()

    return predict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True, help="单个任务目录，含 data/meta/videos")
    parser.add_argument("--dataset_repo_id", help="默认读取 train_config.json，否则使用本地占位 ID")
    parser.add_argument("--tokenizer_path", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True, help="新的结果目录，不覆盖已有目录")
    parser.add_argument("--episodes", default="all", help="训练 episode 编号，如 0,1,2；默认 all")
    parser.add_argument("--stride", type=int, default=8, help="候选观测在每个 episode 内的帧间隔")
    parser.add_argument("--max_samples", type=int, default=256, help="均匀选取至多 N 个候选观测；0 表示全部")
    parser.add_argument("--execution_steps", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video_backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--plots", type=int, default=3, help="均匀选取观测生成图像及动作对比图；0 关闭")
    args = parser.parse_args(argv)
    if args.stride < 1 or args.max_samples < 0 or args.execution_steps < 1 or args.plots < 0:
        parser.error("stride/execution_steps 必须为正数；max_samples/plots 不得为负")
    if args.num_inference_steps is not None and args.num_inference_steps < 1:
        parser.error("num_inference_steps 必须为正数")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Share the deployment's strict checkpoint/source checks. These helpers never
    # import Piper SDK or RealSense; this script never constructs a robot.
    from deploy_piper_vlaa import (
        activate_checkout_source,
        check_checkpoint_files,
        check_prompt_weights,
        deployment_features,
        read_json,
        validate_pi05_source,
    )

    source = activate_checkout_source()
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    validate_pi05_source(PI05Config, source)
    policy_path, root, output = (
        p.expanduser().resolve() for p in (args.policy_path, args.dataset_root, args.output_dir)
    )
    if output.exists():
        raise ValueError(f"结果目录已存在，请换一个 --output_dir: {output}")
    check_checkpoint_files(policy_path)
    info = read_json(root / "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("此验证入口要求 LeRobot v3.0 数据集")
    config = PreTrainedConfig.from_pretrained(str(policy_path), local_files_only=True)
    if not isinstance(config, PI05Config) or config.training_stage != "flow" or config.use_peft:
        raise ValueError("请选择本分支正式 flow 阶段的完整 PI05 checkpoint")
    if not 1 <= args.execution_steps <= config.chunk_size:
        raise ValueError("execution_steps 不能超过 checkpoint 的 chunk_size")
    if config.action_delta_indices != list(range(config.chunk_size)):
        raise ValueError("要求标签时间窗口为 t..t+chunk_size-1")
    features = deployment_features(config, info)
    check_prompt_weights(policy_path, config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用；请在训练 GPU 环境运行")
    config.device, config.pretrained_path = str(device), policy_path
    config.compile_model, config.gradient_checkpointing, config.rtc_config = False, False, None
    if args.num_inference_steps is not None:
        config.num_inference_steps = args.num_inference_steps
    overrides = {"device_processor": {"device": str(device)}}
    if args.tokenizer_path:
        tokenizer = args.tokenizer_path.expanduser().resolve()
        if not tokenizer.is_dir():
            raise ValueError(f"tokenizer 目录不存在: {tokenizer}")
        config.tokenizer_name = str(tokenizer)
        overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer)}
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(policy_path),
        preprocessor_overrides=overrides,
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    for step in preprocessor.steps:
        if any(k != v for k, v in getattr(step, "rename_map", {}).items()):
            raise ValueError("checkpoint 使用了 observation rename，需要核对数据视角映射后再评估")
    train_path = policy_path / "train_config.json"
    training = read_json(train_path).get("dataset", {}) if train_path.is_file() else {}
    references = [policy_path.parent.parent.parent / "dataset_info.json"]
    if training.get("root"):
        references.append(Path(training["root"]).expanduser() / "meta/info.json")
    identity_reference = next((p for p in references if p.is_file()), None)
    if identity_reference is not None and read_json(identity_reference) != info:
        raise ValueError(f"数据集 meta/info.json 与训练时的元数据不一致: {identity_reference}")
    if identity_reference is None:
        print("未找到训练时的 dataset_info.json，无法自动核实数据集身份；请确认模型与任务对应。", flush=True)
    repo_id = args.dataset_repo_id or training.get("repo_id") or "local/piper-offline"
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    episodes = parse_episodes(args.episodes, metadata.total_episodes, training.get("episodes"))
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
        video_backend=args.video_backend,
        return_uint8=True,
        image_transforms=None,
    )
    indices = select_indices(
        dataset.select_columns(["episode_index", "frame_index"]), args.stride, args.max_samples
    )
    if args.plots:
        import matplotlib  # noqa: F401
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    print(
        f"离线验证: samples={len(indices)}, episodes={len(episodes)}, flow_steps={config.num_inference_steps}, "
        f"chunk={config.chunk_size}, execution={args.execution_steps}; 未连接机械臂。",
        flush=True,
    )
    policy = PI05Policy.from_pretrained(str(policy_path), config=config, local_files_only=True, strict=True)
    policy.to(device).eval()
    predict = make_predictor(policy, preprocessor, postprocessor, config, device, args.seed)
    manifest = {
        "policy_path": str(policy_path),
        "dataset_root": str(root),
        "dataset_repo_id": repo_id,
        "training_dataset_config": training,
        "requested_episodes": episodes,
        "dataset_identity_reference": str(identity_reference) if identity_reference else None,
        "dataset_info_sha256": hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest(),
        "stride": args.stride,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "num_inference_steps": config.num_inference_steps,
        "execution_steps": args.execution_steps,
        "chunk_size": config.chunk_size,
        "dtype": config.dtype,
        "use_amp": config.use_amp,
        "protocol": "Recorded current RGB views + state + task; labels t..t+H-1; padded tails excluded. "
        "One seeded noise draw per observation; checkpoint normalization; no action clipping. "
        "Hold baseline reorders current state into action-name order. "
        "Metrics weight valid anchor/offset pairs (overlapping labels may appear more than once). "
        "Joint degree conversion assumes the Piper dataset uses radians. "
        "Offline training-set fit, not closed-loop success or validation-set generalization.",
    }
    output.mkdir(parents=True, exist_ok=False)
    evaluate(
        dataset,
        indices,
        predict,
        features,
        output,
        chunk_size=config.chunk_size,
        execution_steps=args.execution_steps,
        plots=args.plots,
        manifest=manifest,
    )
    print(f"OFFLINE_EVAL_OK: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
