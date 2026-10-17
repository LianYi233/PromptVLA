#!/usr/bin/env python3
r"""Piper 真机部署：piper 分支基于 prompt-ablation 的 PI05 flow + 双 prompt。

从本仓库根目录运行；入口优先加载同目录 src/，复用原部署环境的 Piper 驱动接口。
环境检查（无需 checkpoint，不连接硬件）：python deploy_piper_vlaa.py --check_env
完整说明见 examples/piper/README.md。
--check_robot 只读 CAN/机械臂反馈；真实执行默认 --motion_speed 20。
故障锁定后停止策略并尝试固件急停；退出不主动失能，不保证故障时保持力矩。
训练所得 config、完整模型权重、pre/postprocessor JSON 及其引用的 safetensors
必须一起复制。不能仅复制两个 prompt tensor，也不能沿用旧 quantile regression 权重。

不连接硬件的完整推理测试（仍需要模型、tokenizer 和 LeRobot 推理依赖）：
  python deploy_piper_vlaa.py --policy_path /path/to/pretrained_model \
      --tokenizer_path /path/to/paligemma-3b-pt-224 --dry_run

真机运行（DATA 为该任务的 v3.0 根目录，只读取其中 meta/info.json）：
  python deploy_piper_vlaa.py --policy_path /path/to/pretrained_model \
      --tokenizer_path /path/to/paligemma-3b-pt-224 --dataset_root /path/to/DATA \
      --task "Put the apple on the yellow plate" --steps_per_chunk 8 --num_episodes 5

--dataset_info 可直接指定从训练数据复制的 meta/info.json；无需把整份视频复制到部署机。
状态与动作顺序使用该元信息的 names，动作沿用旧驱动约定：6 关节弧度 + 0..1 夹爪。
默认控制频率读取数据 fps；--control_hz 可显式覆盖，不改变动作定义或模型 chunk_size。
默认关闭 compile；--no_compile 保留为兼容参数。quantile_sampling 已移除。

相机名不同可用 --camera_map '{"observation.images.wrist":"wrist_camera"}'，
方向是「模型的相机 key -> robot.get_observation() 的相机 key」。不猜测相机对应关系。
--camera_serials '{"wrist_camera":"实际序列号"}' 可覆盖原 PiperConfig 的设备配置。
图像默认按 RGB 原样使用；若驱动实际返回 BGR，传 --camera_color_order bgr。

远端 prompt-ablation 未包含 src/lerobot/robots/piper/；实机需要从原部署环境保留/复制
完整 Piper 驱动目录及其本地依赖。--dry_run 不导入该驱动，不连接 CAN 或相机。
"""

import argparse
import importlib
import json
import logging
import math
import sys
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np

from piper_deploy_guard import GuardedPiperMixin as PiperConnectionMixin, check_robot_connection

logger = logging.getLogger(__name__)


def activate_checkout_source():
    """Use the adjacent checkout even if another LeRobot is installed editable."""
    source = Path(__file__).resolve().parent / "src"
    package = source / "lerobot"
    if not (package / "__init__.py").is_file():
        raise RuntimeError(
            "请在完整 piper 分支中运行 deploy_piper_vlaa.py；入口旁边必须有 src/lerobot/。"
            "不要把入口单独复制到旧版 LeRobot 中。"
        )
    # A caller may already have imported the other editable checkout. Replacing
    # sys.path cannot replace cached modules; fail instead of mixing versions.
    for name, module in tuple(sys.modules.items()):
        if name == "lerobot" or name.startswith("lerobot."):
            origin = getattr(module, "__file__", None)
            if origin is not None and not Path(origin).resolve().is_relative_to(package):
                raise RuntimeError(f"已加载其他目录的 {name}: {origin}；请用新 Python 进程运行此入口")
    sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    return source


def validate_pi05_source(config_class, source):
    """Check source identity and prompt support before reading checkpoint config."""
    module = sys.modules[config_class.__module__]
    actual = Path(module.__file__).resolve()
    expected = source / "lerobot/policies/pi05/configuration_pi05.py"
    if actual != expected.resolve():
        raise RuntimeError(f"PI05 配置来自其他源码: {actual}；应为 {expected}")
    required = {"num_vlm_prompt_tokens", "num_prompt_tokens", "training_stage"}
    missing = required - set(getattr(config_class, "__dataclass_fields__", {}))
    if missing:
        raise RuntimeError(
            f"PI05 源码版本不匹配，缺少 {sorted(missing)}：{actual}。"
            "请拉取完整 piper 分支，不要手动补配置字段。"
        )
    print(f"PI05 config source: {actual}")


# 关节安全限位（弧度），对应 joint_1 ~ joint_6
_DEG_LIMITS = [
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
]
JOINT_LIMITS_RAD = [(lo * np.pi / 180, hi * np.pi / 180) for lo, hi in _DEG_LIMITS]
GRIPPER_LIMIT = (0.0, 1.0)
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def clamp_action(action: dict) -> dict:
    action = dict(action)
    for key in [f"{name}.pos" for name in JOINT_NAMES] + ["gripper.pos"]:
        if not np.isfinite(action[key]):
            raise ValueError(f"模型输出了非有限动作: {key}={action[key]}")
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = JOINT_LIMITS_RAD[i]
        action[f"{name}.pos"] = float(np.clip(action[f"{name}.pos"], lo, hi))
    action["gripper.pos"] = float(np.clip(action["gripper.pos"], *GRIPPER_LIMIT))
    return action


def configure_deployment(policy_cfg, args, dataset_fps=None):
    """Only override inference settings; keep prompt counts, features and chunk size."""
    if policy_cfg.type != "pi05":
        raise ValueError("此脚本只支持 prompt-ablation 分支的 PI05")
    if policy_cfg.training_stage != "flow":
        raise ValueError("请选择正式 flow 阶段的 checkpoint，例如 checkpoints/003000/pretrained_model")
    if policy_cfg.use_peft:
        raise ValueError("需要完整 PI05 checkpoint，不支持仅包含 LoRA/PEFT adapter 的目录")
    steps_per_chunk = args.steps_per_chunk if args.steps_per_chunk is not None else policy_cfg.n_action_steps
    if not 1 <= steps_per_chunk <= policy_cfg.chunk_size:
        raise ValueError(f"steps_per_chunk 必须在 1..{policy_cfg.chunk_size} 之间")
    control_hz = args.control_hz if args.control_hz is not None else dataset_fps
    if control_hz is not None and (not math.isfinite(control_hz) or control_hz <= 0):
        raise ValueError("control_hz / 数据集 fps 必须为有限正数")
    if not args.dry_run and control_hz is None:
        raise ValueError("缺少训练数据 fps，请提供 --dataset_info / --dataset_root")
    if not 1 <= args.motion_speed <= 100:
        raise ValueError("motion_speed 必须为 1..100 的整数百分比")
    if args.num_episodes <= 0 or args.steps_per_episode <= 0:
        raise ValueError("num_episodes 和 steps_per_episode 必须大于 0")
    if not math.isfinite(args.inference_timeout) or args.inference_timeout <= 0:
        raise ValueError("inference_timeout 必须为有限正数")
    if not args.task.strip():
        raise ValueError("task 不能为空，请使用训练时相应任务的语言指令")
    if args.num_inference_steps is not None:
        if args.num_inference_steps <= 0:
            raise ValueError("num_inference_steps 必须大于 0")
        policy_cfg.num_inference_steps = args.num_inference_steps
    policy_cfg.compile_model = args.compile_model
    policy_cfg.gradient_checkpointing = False
    policy_cfg.n_action_steps = steps_per_chunk
    # This script executes discrete chunks, without RTC continuation/caches.
    policy_cfg.rtc_config = None
    if args.tokenizer_path:
        tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
        if not tokenizer_path.is_dir():
            raise ValueError(f"本地 tokenizer 路径不存在: {tokenizer_path}")
        policy_cfg.tokenizer_name = str(tokenizer_path)
    return control_hz, steps_per_chunk


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def check_checkpoint_files(policy_path):
    """Require all processor state files, including the training normalization stats."""
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (policy_path / name).is_file():
            raise ValueError(f"Checkpoint 文件缺失: {policy_path / name}；请复制完整 pretrained_model 目录")
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in read_json(policy_path / filename)["steps"]:
            if step.get("state_file") and not (policy_path / step["state_file"]).is_file():
                raise ValueError(f"Checkpoint 处理器状态缺失: {step['state_file']}；不能省略归一化统计文件")
    raw_config = read_json(policy_path / "config.json")
    if "action_generation_mode" in raw_config or any("quantile" in key for key in raw_config):
        raise ValueError("这是旧 quantile/regression 配置；请改用 prompt-ablation 训练得到的 flow checkpoint")


def check_prompt_weights(policy_path, policy_cfg):
    # The training loader initializes absent prompts for base models. Deployment
    # must reject that case before loading, so a missing learned prompt cannot be random.
    from safetensors import safe_open

    expected = {
        "model.vlm_prompt_tokens.weight": policy_cfg.num_vlm_prompt_tokens,
        "model.prompt_tokens.weight": policy_cfg.num_prompt_tokens,
    }
    with safe_open(str(policy_path / "model.safetensors"), framework="pt", device="cpu") as state:
        saved_keys = set(state.keys())
        for key, count in expected.items():
            if key not in saved_keys:
                raise ValueError(f"Checkpoint 缺少 {key}；需要本分支保存的完整模型，不能使用原始 base 权重")
            shape = state.get_slice(key).get_shape()
            if len(shape) != 2 or shape[0] != count:
                raise ValueError(f"Prompt 形状与 config 不符: {key}={shape}, tokens={count}")
            if not np.isfinite(state.get_tensor(key).float().numpy()).all():
                raise ValueError(f"Prompt 权重包含非有限值: {key}")
            print(f"Loaded prompt check: {key}, shape={shape}")


def json_mapping(value):
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in parsed.items()
    ):
        raise argparse.ArgumentTypeError("必须是字符串到字符串的 JSON 对象")
    return parsed


def deployment_features(policy_cfg, dataset_info=None):
    """Keep saved feature names and the dataset's component order, never overwrite config."""
    if not policy_cfg.input_features or not policy_cfg.output_features or not policy_cfg.image_features:
        raise ValueError("Checkpoint 缺少训练输入/输出特征；请选择真机训练得到的完整 checkpoint")
    expected_names = [f"{name}.pos" for name in JOINT_NAMES] + ["gripper.pos"]
    result = {}
    for key, config_features in (
        ("observation.state", policy_cfg.input_features),
        ("action", policy_cfg.output_features),
    ):
        feature = config_features.get(key)
        if feature is None or tuple(feature.shape) != (7,):
            raise ValueError(f"此 Piper 入口要求 {key} 为 7 维（6 关节+夹爪），checkpoint={feature}")
        if dataset_info is None:
            # Synthetic dry-run only: all zeros, so component order has no hardware effect.
            names = expected_names
        else:
            recorded = dataset_info["features"].get(key, {})
            names = recorded.get("names")
            if tuple(recorded.get("shape", ())) != tuple(feature.shape):
                raise ValueError(f"训练数据与 checkpoint 的 {key} 维度不符")
            if not isinstance(names, list) or len(names) != 7 or set(names) != set(expected_names):
                raise ValueError(
                    f"{key}.names 必须对应原 Piper 驱动的关节与夹爪名称，实际为 {names}；不能猜测顺序"
                )
        result[key] = {"dtype": "float32", "shape": (7,), "names": list(names)}
    for key, feature in policy_cfg.image_features.items():
        if not key.startswith("observation.images.") or len(feature.shape) != 3 or feature.shape[0] != 3:
            raise ValueError(f"不支持的相机特征: {key}={feature.shape}")
        if dataset_info is not None and key not in dataset_info["features"]:
            raise ValueError(f"Checkpoint 相机 {key} 不在指定训练数据中，请检查数据/模型是否属于同一任务")
        _, height, width = feature.shape
        result[key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return result


def resolve_camera_map(ds_features, explicit_map):
    expected = {key for key in ds_features if key.startswith("observation.images.")}
    unknown = set(explicit_map) - expected
    if unknown:
        raise ValueError(f"camera_map 左侧不是 checkpoint 中的相机 key: {sorted(unknown)}")
    result = {key: explicit_map.get(key, key.removeprefix("observation.images.")) for key in expected}
    if len(set(result.values())) != len(result):
        raise ValueError("不同模型视角不能映射到同一台相机")
    return result


def build_observation(obs_raw, ds_features, camera_map, color_order):
    names = ds_features["observation.state"]["names"]
    state = np.asarray([obs_raw[name] for name in names], dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError("机器人状态包含非有限值")
    result = {"observation.state": state}
    for model_key, robot_key in camera_map.items():
        frame = np.asarray(obs_raw[robot_key])
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"相机 {robot_key} 必须返回 uint8 HWC 三通道图像，实际 {frame.shape}/{frame.dtype}"
            )
        if color_order == "bgr":
            frame = frame[..., ::-1]
        result[model_key] = np.ascontiguousarray(frame)
    return result


def create_robot(args, ds_features, camera_map):
    try:
        from lerobot.robots.piper import PiperConfig
        from lerobot.robots.piper.piper import Piper
    except ModuleNotFoundError as exc:
        if exc.name not in {"lerobot.robots.piper", "lerobot.robots.piper.piper"}:
            # The driver may exist but have a missing SDK/local dependency.
            # Preserve that error instead of falsely reporting a missing driver.
            raise
        raise ImportError(
            "缺少自定义 Piper 驱动；请把原部署环境的 src/lerobot/robots/piper/ 完整目录复制到此分支，"
            "并保留其依赖。步骤见 examples/piper/README.md。--dry_run 不需要 Piper 驱动。"
        ) from exc

    class DeploymentPiper(PiperConnectionMixin, Piper):
        pass

    robot_cfg = PiperConfig(can_port=args.can_port)
    if args.camera_serials is not None:
        robot_cfg.camera_serials = args.camera_serials
    robot = DeploymentPiper(robot_cfg)
    robot._motion_speed = args.motion_speed
    # Validate before connect(), because the original connect routine enables the arm/gripper.
    if set(robot.action_features) != set(ds_features["action"]["names"]):
        raise ValueError("Piper 驱动 action_features 与训练动作名称不一致")
    required = set(ds_features["observation.state"]["names"]) | set(camera_map.values())
    missing = required - set(robot.observation_features)
    if missing:
        raise ValueError(f"Piper 缺少观测 {sorted(missing)}；相机别名请通过 --camera_map 显式指定")
    return robot


class ChunkInferenceWorker:
    """
    触发式后台推理线程。

    获取新观测后触发推理，等待完整动作段，再执行这一段的全部动作。
    """

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        policy_cfg,
        ds_features,
        device,
        task,
        robot_type,
        steps_per_chunk,
    ):
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._policy_cfg = policy_cfg
        self._ds_features = ds_features
        self._device = device
        self._task = task
        self._robot_type = robot_type
        self._steps_per_chunk = steps_per_chunk

        # 触发信号：主线程写入 obs，后台线程消费
        self._trigger_event = threading.Event()
        self._obs_to_infer = None
        self._obs_lock = threading.Lock()

        # 结果信号：后台线程写入 chunk，主线程消费
        self._ready_event = threading.Event()
        self._ready_chunk: list[dict] | None = None
        self._infer_time_ms = 0.0
        self._result_lock = threading.Lock()
        self._error: Exception | None = None

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._trigger_event.set()  # 唤醒线程让它退出
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def trigger(self, obs_frame: dict):
        """主线程调用：用最新观测触发下一次推理。"""
        with self._obs_lock:
            self._obs_to_infer = deepcopy(obs_frame)
        with self._result_lock:
            self._ready_chunk = None
            self._error = None
        self._ready_event.clear()
        self._trigger_event.set()

    def next_chunk(self, timeout: float = 10.0, abort_errors=None) -> tuple[list[dict], float]:
        """
        主线程调用：等待并取出推理好的 chunk。
        返回 (chunk, infer_time_ms)，chunk 是长度为 steps_per_chunk 的动作列表。
        """
        deadline = time.monotonic() + timeout
        while True:
            if abort_errors:
                raise RuntimeError("机械臂保位失败") from abort_errors[0]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"推理超时（>{timeout}s）")
            if self._ready_event.wait(timeout=min(0.05, remaining)):
                break
        with self._result_lock:
            if self._error is not None:
                raise RuntimeError("后台策略推理失败") from self._error
            return self._ready_chunk, self._infer_time_ms

    def _loop(self):
        while not self._stop_event.is_set():
            # 等待触发
            self._trigger_event.wait()
            self._trigger_event.clear()

            if self._stop_event.is_set():
                break

            with self._obs_lock:
                obs = deepcopy(self._obs_to_infer)

            if obs is None:
                continue

            try:
                chunk, infer_ms = self._infer_chunk(obs)
                with self._result_lock:
                    self._ready_chunk = chunk
                    self._infer_time_ms = infer_ms
            except Exception as exc:
                with self._result_lock:
                    self._error = exc
            finally:
                self._ready_event.set()

    def _infer_chunk(self, obs):
        import torch

        from lerobot.policies import make_robot_action, prepare_observation_for_inference

        t0 = time.perf_counter()
        amp = (
            torch.autocast(device_type="cuda")
            if self._device.type == "cuda" and self._policy_cfg.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            batch = prepare_observation_for_inference(
                deepcopy(obs), self._device, self._task, self._robot_type
            )
            batch = self._preprocessor(batch)
            # One multimodal forward/flow solve per observation. No quantile head,
            # private action queue, or repeated image/tokenizer preprocessing.
            actions = self._policy.predict_action_chunk(batch)
            expected_shape = (1, self._policy_cfg.chunk_size, len(self._ds_features["action"]["names"]))
            if tuple(actions.shape) != expected_shape:
                raise RuntimeError(f"动作段形状错误: {tuple(actions.shape)}, expected={expected_shape}")
            if not torch.isfinite(actions).all().item():
                raise ValueError("Flow 模型输出了非有限动作")
            chunk = []
            for index in range(self._steps_per_chunk):
                # Restore the training dataset's physical units with the saved stats.
                values = self._postprocessor(actions[:, index])
                chunk.append(clamp_action(make_robot_action(values, self._ds_features)))
        return chunk, (time.perf_counter() - t0) * 1000


_HOLD_INTERVAL = 0.01  # 100Hz 保活，对齐 Piper 官方 demo 的发送频率


def _hold_position(robot, action, stop_event, errors):
    """推理等待期间持续发送当前位置指令，防止固件超时失力矩。"""
    try:
        while not stop_event.is_set():
            robot.send_action(action)
            stop_event.wait(_HOLD_INTERVAL)
    except Exception as exc:
        errors.append(exc)
        stop_event.set()


def run_episode(
    robot,
    worker,
    ds_features,
    steps,
    control_hz,
    episode_idx,
    num_episodes,
    camera_map,
    color_order,
    inference_timeout,
):
    control_interval = 1.0 / control_hz

    total_infers = 0
    step = 0

    while step < steps:
        # ── 读取当前观测，触发后台推理下一个 chunk ───────────────
        obs_raw = robot.get_observation()
        obs_frame = build_observation(obs_raw, ds_features, camera_map, color_order)
        worker.trigger(obs_frame)

        # ── 推理等待期间用保活线程持续发送当前位置，防止固件超时软停 ──
        hold_action = {key: float(obs_raw[key]) for key in robot.action_features}
        if not all(math.isfinite(value) for value in hold_action.values()):
            raise ValueError("保位观测包含非有限值")
        if step == 0:
            joint_vals = {k: f"{v:.4f}" for k, v in hold_action.items() if "gripper" not in k}
            print(f"\n  [诊断] 首次保位关节值(rad): {joint_vals}")
            print(f"  [诊断] 夹爪值: {hold_action.get('gripper.pos', 'N/A'):.4f}")
        stop_hold = threading.Event()
        hold_errors = []
        hold_thread = threading.Thread(
            target=_hold_position,
            args=(robot, hold_action, stop_hold, hold_errors),
            daemon=True,
        )
        hold_thread.start()

        # ── 等待推理完成，取出 chunk ──────────────────────────────
        try:
            chunk, infer_ms = worker.next_chunk(timeout=inference_timeout, abort_errors=hold_errors)
        finally:
            stop_hold.set()
            hold_thread.join()
        if hold_errors:
            raise RuntimeError("机械臂保位失败") from hold_errors[0]
        total_infers += 1

        # ── 执行 chunk 里的每一步动作 ────────────────────────────
        for chunk_step, action in enumerate(chunk):
            if step >= steps:
                break

            loop_start = time.perf_counter()
            robot.send_action(action)

            elapsed = time.perf_counter() - loop_start
            sleep_t = control_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            actual_hz = 1.0 / (time.perf_counter() - loop_start)
            step += 1
            print(
                f"\r  [Ep {episode_idx}/{num_episodes}] "
                f"步数: {step}/{steps} | "
                f"chunk: {chunk_step + 1}/{len(chunk)} | "
                f"推理耗时: {infer_ms:.0f}ms | "
                f"控制频率: {actual_hz:.1f}Hz | "
                f"累计推理: {total_infers}次",
                end="",
                flush=True,
            )

    print(f"\n  完成 {steps} 步，共推理 {total_infers} 次。")


def reset_inference_state(policy, preprocessor, postprocessor):
    for component in (policy, preprocessor, postprocessor):
        reset = getattr(component, "reset", None)
        if callable(reset):
            reset()


def run_dry_run(worker, ds_features, policy, preprocessor, postprocessor):
    """Run full checkpoint inference on synthetic frames without importing Piper."""
    obs_frame = {
        key: np.zeros(feature["shape"], dtype=np.uint8 if feature["dtype"] == "video" else np.float32)
        for key, feature in ds_features.items()
        if key.startswith("observation.")
    }
    reset_inference_state(policy, preprocessor, postprocessor)
    forward_count = 0
    predict_chunk = policy.predict_action_chunk

    def counted_predict_chunk(*args, **kwargs):
        nonlocal forward_count
        forward_count += 1
        return predict_chunk(*args, **kwargs)

    policy.predict_action_chunk = counted_predict_chunk
    try:
        chunk, infer_ms = worker._infer_chunk(obs_frame)
    finally:
        policy.predict_action_chunk = predict_chunk
    if forward_count != 1 or len(chunk) != worker._steps_per_chunk:
        raise RuntimeError(f"动作段验证失败: forwards={forward_count}, steps={len(chunk)}")
    print(
        f"DRY_RUN_OK: forwards={forward_count}, predicted_steps={policy.config.chunk_size}, "
        f"execution_steps={len(chunk)}, inference_ms={infer_ms:.0f}, "
        f"compile_model={policy.config.compile_model}"
    )
    print("仅使用合成观测完成验证；未导入 Piper 驱动、未连接机械臂或相机。此结果不代表任务成功率。")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy_path", help="正式 flow 阶段完整 pretrained_model 目录")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check_env", action="store_true", help="检查本分支 PI05 导入，无需权重和硬件")
    mode.add_argument(
        "--check_robot", action="store_true", help="只读 CAN/机械臂状态，不使能、不发动作、不连接相机"
    )
    parser.add_argument("--tokenizer_path", help="本机 PaliGemma tokenizer 目录，覆盖训练时记录的路径")
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("--dataset_root", help="对应真机训练任务目录（只读取 meta/info.json）")
    dataset.add_argument("--dataset_info", help="直接指定对应训练数据的 meta/info.json 文件")
    parser.add_argument("--task", default="Put the apple on the yellow plate")
    parser.add_argument("--device", default="cuda", help="推理设备，例如 cuda、cuda:0 或 cpu")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--can_port", default="can0")
    parser.add_argument(
        "--motion_speed", type=int, default=20, help="MOVE J 速度百分比，默认 20；独立于动作 Hz"
    )
    parser.add_argument("--control_hz", type=float, help="默认读取训练数据 fps，覆盖后动作播放速度将改变")
    parser.add_argument(
        "--steps_per_chunk", type=int, help="每次预测后执行步数，默认 checkpoint.n_action_steps"
    )
    parser.add_argument("--num_inference_steps", type=int, help="Flow 求解步数，默认沿用 checkpoint")
    parser.add_argument("--inference_timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0, help="Flow 噪声随机种子")
    parser.add_argument("--camera_map", type=json_mapping, default={}, help="模型相机 key -> 驱动观测 key")
    parser.add_argument("--camera_serials", type=json_mapping, help="覆盖 PiperConfig.camera_serials")
    parser.add_argument("--camera_color_order", choices=("rgb", "bgr"), default="rgb")
    compilation = parser.add_mutually_exclusive_group()
    compilation.add_argument("--compile", dest="compile_model", action="store_true")
    compilation.add_argument("--no_compile", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=False)
    mode.add_argument("--dry_run", action="store_true", help="完整模型合成观测测试，不连接硬件")
    args = parser.parse_args(argv)
    if not (args.check_env or args.check_robot) and not args.policy_path:
        parser.error("需要 --policy_path；诊断可用 --check_env / --check_robot")
    if not (args.check_env or args.check_robot or args.dry_run) and not (
        args.dataset_root or args.dataset_info
    ):
        parser.error("实机运行需要 --dataset_root 或 --dataset_info，以核对训练时的关节顺序和 fps")
    return args


def main(argv=None):
    args = parse_args(argv)
    source = activate_checkout_source()
    if sys.version_info < (3, 12):  # noqa: UP036 -- also run directly in old deployment environments
        raise RuntimeError(f"piper 分支需要 Python >=3.12，当前为 {sys.version.split()[0]}")

    if args.check_robot:
        check_robot_connection(args.can_port)
        return

    # ML imports are deferred: --help and source-level tests need no robot SDK/GPU.
    import torch

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401 -- register checkpoint processor steps
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.device_utils import get_safe_torch_device

    validate_pi05_source(PI05Config, source)
    if args.check_env:
        import transformers

        print(f"Python: {sys.executable} ({sys.version.split()[0]})")
        print(f"torch: {torch.__version__}; transformers: {transformers.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        driver = source / "lerobot/robots/piper/piper.py"
        print(f"Piper driver file: {driver} (exists={driver.is_file()})")
        print("ENV_OK: 本分支双 prompt 配置、模型和处理器导入通过；未加载权重，未验证或连接驱动。")
        return

    policy_path = Path(args.policy_path).expanduser().resolve()
    check_checkpoint_files(policy_path)
    dataset_info = None
    if args.dataset_root or args.dataset_info:
        info_path = (
            Path(args.dataset_info).expanduser()
            if args.dataset_info
            else Path(args.dataset_root).expanduser() / "meta/info.json"
        )
        dataset_info = read_json(info_path)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_safe_torch_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path), local_files_only=True)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = str(device)
    control_hz, steps_per_chunk = configure_deployment(
        policy_cfg, args, dataset_info.get("fps") if dataset_info is not None else None
    )
    ds_features = deployment_features(policy_cfg, dataset_info)
    camera_map = resolve_camera_map(ds_features, args.camera_map)
    check_prompt_weights(policy_path, policy_cfg)

    print(f"Checkpoint: {policy_path}")
    print(f"State order: {ds_features['observation.state']['names']}")
    print(f"Action order: {ds_features['action']['names']}")
    print(f"Camera map: {camera_map}")
    print(f"Prompts: VLM={policy_cfg.num_vlm_prompt_tokens}, action={policy_cfg.num_prompt_tokens}")
    # Instantiate/validate the local driver only for real execution. No connection yet.
    robot = None if args.dry_run else create_robot(args, ds_features, camera_map)
    print("加载本分支 PI05 完整模型和处理器...")
    # Use native initialization/loading, avoiding old quantile-specific meta modules/buffers.
    policy = PI05Policy.from_pretrained(
        str(policy_path), config=policy_cfg, local_files_only=True, strict=True
    )
    policy.to(device)
    policy.eval()
    preprocessor_overrides = {"device_processor": {"device": str(device)}}
    if args.tokenizer_path:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": policy_cfg.tokenizer_name}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    # Frames already use checkpoint keys; applying a second rename could drop/swap views.
    for step in preprocessor.steps:
        rename_map = getattr(step, "rename_map", {})
        if any(source in ds_features and source != target for source, target in rename_map.items()):
            raise ValueError("Checkpoint 含额外 observation rename；请检查训练映射后再适配，不能自动覆盖")
    print(
        f"模型加载完成: device={device}, flow_steps={policy_cfg.num_inference_steps}, "
        f"预测={policy_cfg.chunk_size}, 执行={steps_per_chunk}, compile={policy_cfg.compile_model}"
    )
    if control_hz is not None:
        print(f"动作执行频率={control_hz}Hz；等待推理的时间会额外增加 chunk 之间的间隔")
    worker = ChunkInferenceWorker(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        policy_cfg=policy_cfg,
        ds_features=ds_features,
        device=device,
        task=args.task,
        robot_type=robot.robot_type if robot is not None else "piper",
        steps_per_chunk=steps_per_chunk,
    )
    if args.dry_run:
        run_dry_run(worker, ds_features, policy, preprocessor, postprocessor)
        return

    try:
        robot.connect()
        print(f"机器人已连接；MOVE J 速度参数={args.motion_speed}%，故障监测已启用。")
        worker.start()
        for ep in range(1, args.num_episodes + 1):
            input(f"\n[Episode {ep}/{args.num_episodes}] 按 Enter 开始推理...")
            reset_inference_state(policy, preprocessor, postprocessor)
            run_episode(
                robot=robot,
                worker=worker,
                ds_features=ds_features,
                steps=args.steps_per_episode,
                control_hz=control_hz,
                episode_idx=ep,
                num_episodes=args.num_episodes,
                camera_map=camera_map,
                color_order=args.camera_color_order,
                inference_timeout=args.inference_timeout,
            )
        print("\n所有 episode 执行完毕。")
    except KeyboardInterrupt:
        print("\n已中断。")
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            worker.stop()
        print("控制程序已结束；请现场确认机械臂状态。")


if __name__ == "__main__":
    main()
