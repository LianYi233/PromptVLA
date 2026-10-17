"""Deployment fault handling for the uploaded Piper driver and piper-sdk 0.6.1.

This is supervisory software, not a hardware emergency stop. A queued CAN frame
does not prove delivery; loss of communication/power can still release torque.
Never clear faults, reset the arm, reconnect, or re-enable it after a fault.
"""

import json
import logging
import math
import subprocess
import threading
import time
from copy import deepcopy

from piper_camera_check import (
    NamedCameraPipeline,
    PiperCameraError,
    color_stream_config,
    validate_camera_serials,
)

logger = logging.getLogger(__name__)


class PiperFaultError(RuntimeError):
    """A latched deployment fault; a new operator-controlled session is required."""


class FeedbackUnavailableError(PiperFaultError):
    """No recent feedback yet; only tolerated while establishing a connection."""


def read_can_link(port):
    """Read kernel CAN state, not python-can's cached Bus.state property."""
    try:
        result = subprocess.run(
            ["ip", "-json", "-details", "-statistics", "link", "show", "dev", port],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        links = json.loads(result.stdout)
        if len(links) != 1:
            raise ValueError("expected one CAN interface")
        return links[0]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise PiperFaultError(f"无法读取 {port} 的内核 CAN 状态: {exc}") from exc


def require_healthy_can(link):
    info = link.get("linkinfo", {})
    data = info.get("info_data", {})
    if (
        "UP" not in link.get("flags", [])
        or info.get("info_kind") != "can"
        or data.get("state") != "ERROR-ACTIVE"
        or data.get("bittiming", {}).get("bitrate") != 1000000
    ):
        raise PiperFaultError(
            f"CAN 未就绪: {json.dumps(link, ensure_ascii=False)}；"
            "需要 UP、ERROR-ACTIVE、1000000 bit/s。先排查连接/供电，不自动重置或恢复动作。"
        )


_MOTOR_FAULTS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "collision_status",
    "driver_error_status",
    "stall_status",
)
_GRIPPER_FAULTS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "sensor_status",
    "driver_error_status",
)


def feedback_snapshot(sdk, max_age=0.5):
    """Read the public SDK feedback, including units and controller fault bits."""
    messages = {
        "status": deepcopy(sdk.GetArmStatus()),
        "joints": deepcopy(sdk.GetArmJointMsgs()),
        "gripper": deepcopy(sdk.GetArmGripperMsgs()),
        "motors": deepcopy(sdk.GetArmLowSpdInfoMsgs()),
    }
    now = time.time()
    for name, message in messages.items():
        stamp = float(message.time_stamp)
        if not math.isfinite(stamp) or stamp <= 0 or not -0.1 <= now - stamp <= max_age:
            raise FeedbackUnavailableError(f"{name} 反馈未到达或已过期: age={now - stamp:.3f}s")
    status = messages["status"].arm_status
    grip = messages["gripper"].gripper_state
    motors = []
    for index in range(1, 7):
        motor = getattr(messages["motors"], f"motor_{index}")
        if motor.can_id != 0x260 + index:
            raise FeedbackUnavailableError(f"motor_{index} 反馈尚未到达")
        motors.append(
            {
                "joint": index,
                "voltage_V": motor.vol * 0.1,
                "enabled": bool(motor.foc_status.driver_enable_status),
                "faults": [name for name in _MOTOR_FAULTS if getattr(motor.foc_status, name)],
            }
        )
    joint_state = messages["joints"].joint_state
    return {
        "arm_status": int(status.arm_status),
        "err_code": int(status.err_code),
        "ctrl_mode": int(status.ctrl_mode),
        "motors": motors,
        "joint_rad": [getattr(joint_state, f"joint_{i}") * math.pi / 180000 for i in range(1, 7)],
        "gripper_raw": int(grip.grippers_angle),
        "gripper_enabled": bool(grip.foc_status.driver_enable_status),
        "gripper_faults": [name for name in _GRIPPER_FAULTS if getattr(grip.foc_status, name)],
    }


def require_healthy_feedback(snapshot, require_enabled=True):
    if (
        snapshot["arm_status"] != 0
        or snapshot["err_code"] != 0
        or snapshot["ctrl_mode"] not in (0, 1)
        or snapshot["gripper_faults"]
        or any(motor["faults"] for motor in snapshot["motors"])
    ):
        raise PiperFaultError(f"机械臂反馈异常: {json.dumps(snapshot, ensure_ascii=False)}")
    if require_enabled and not all_enabled(snapshot):
        raise PiperFaultError("运行中电机或夹爪已失能；停止策略，不自动重新使能")


def all_enabled(snapshot):
    return snapshot["gripper_enabled"] and all(motor["enabled"] for motor in snapshot["motors"])


class PiperGuard:
    """Latch TX/feedback faults and prevent the rest of a chunk from being sent."""

    def __init__(self, sdk, port):
        self.sdk = sdk
        self.port = port
        self.fault = None
        self.last_link = None
        self.last_feedback = None
        self._last_link_check = -math.inf
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._stop_attempted = False
        self._allow_stop = False
        self.transport = sdk.GetCanBus()
        # SDK 0.6.1 returns this enum from SendCanMessage, then only logs errors
        # in JointCtrl/MotionCtrl_2/GripperCtrl. Intercept below those log-only calls.
        self._success = self.transport.CAN_STATUS.SEND_MESSAGE_SUCCESS
        self._original_send = self.transport.SendCanMessage
        self.transport.SendCanMessage = self._send_checked

    def fail(self, reason):
        with self._lock:
            if self.fault is None:
                self.fault = str(reason)
                logger.critical(
                    "PIPER_FAULT: %s; can=%s; feedback=%s",
                    self.fault,
                    json.dumps(self.last_link, ensure_ascii=False),
                    json.dumps(self.last_feedback, ensure_ascii=False),
                )
            raise PiperFaultError(self.fault)

    def check_latch(self):
        if self.fault is not None:
            raise PiperFaultError(self.fault)

    def check_link(self, force=False):
        with self._lock:
            self.check_latch()
            now = time.monotonic()
            if force or now - self._last_link_check >= 0.1:
                try:
                    self.last_link = read_can_link(self.port)
                    require_healthy_can(self.last_link)
                    self._last_link_check = now
                except Exception as exc:
                    self.fail(exc)

    def check(self, require_enabled=True):
        self.check_link()
        try:
            snapshot = feedback_snapshot(self.sdk)
            self.last_feedback = snapshot
            require_healthy_feedback(snapshot, require_enabled)
        except Exception as exc:
            self.fail(exc)
        self.check_latch()
        return snapshot

    def wait_ready(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        while True:
            self.check_link()
            try:
                snapshot = feedback_snapshot(self.sdk)
            except FeedbackUnavailableError as exc:
                if time.monotonic() >= deadline:
                    self.fail(exc)
                time.sleep(0.02)
                continue
            self.last_feedback = snapshot
            try:
                require_healthy_feedback(snapshot, require_enabled=False)
            except Exception as exc:
                self.fail(exc)
            return snapshot

    def _send_checked(self, arbitration_id, data, *args, **kwargs):
        with self._lock:
            is_stop = self._allow_stop and arbitration_id == 0x150 and bytes(data) == bytes([1] + [0] * 7)
            if not is_stop:
                self.check_link()
                if self._stop_attempted:
                    self.fail("停止请求后禁止继续发送动作")
            try:
                result = self._original_send(arbitration_id, data, *args, **kwargs)
            except Exception as exc:
                self.fail(f"CAN TX exception id={arbitration_id:#x}: {exc}")
            if result != self._success:
                self.fail(f"CAN TX failed id={arbitration_id:#x}: {result}")
            return result

    def request_stop(self):
        """One best-effort firmware quick-stop; never DisableArm or ResetPiper."""
        with self._lock:
            if self._stop_attempted:
                return
            self._stop_attempted = True
            self._allow_stop = True
            try:
                self.sdk.MotionCtrl_1(0x01, 0x00, 0x00)
                logger.warning("已请求固件急停；CAN 入队成功不代表机械臂已收到，需现场确认。")
            except Exception as exc:
                logger.critical("固件急停请求发送失败，无法保证停止或保位；请执行现场停机流程: %s", exc)
            finally:
                self._allow_stop = False

    def start(self):
        def monitor():
            while not self._stop_event.wait(0.05):
                try:
                    self.check()
                except Exception:
                    self.request_stop()
                    return

        self._thread = threading.Thread(target=monitor, name="piper-feedback-monitor", daemon=True)
        self._thread.start()

    def shutdown(self):
        self._stop_event.set()
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def uninstall(self):
        # Only after closing the SDK port. No active controller may bypass the guard.
        self.transport.SendCanMessage = self._original_send


def check_robot_connection(port):
    """Read feedback only: no enable, movement, reset, or stop frames."""
    link = read_can_link(port)
    print("CAN:", json.dumps(link, ensure_ascii=False))
    require_healthy_can(link)
    from piper_sdk import C_PiperInterface_V2

    sdk = C_PiperInterface_V2(port)
    guard = None
    try:
        guard = PiperGuard(sdk, port)
        sdk.ConnectPort(piper_init=False)
        snapshot = guard.wait_ready()
        print("Feedback:", json.dumps(snapshot, ensure_ascii=False, indent=2))
        print("ROBOT_READ_OK: 仅检查 CAN 和状态反馈；未使能、未发送动作、未连接相机。")
    finally:
        sdk.DisconnectPort()
        if guard is not None:
            guard.uninstall()


class GuardedPiperMixin:
    """Override the supplied driver's lifecycle/TX while retaining its camera config."""

    def connect(self, calibrate=True):
        if self._connected:
            raise RuntimeError("Piper 已连接")
        # Refuse WARNING/PASSIVE/OFF before opening cameras or enabling any motor.
        require_healthy_can(read_can_link(self.config.can_port))
        validate_camera_serials(self.config.camera_serials)
        import pyrealsense2 as rs
        from piper_sdk import C_PiperInterface_V2

        self._guard = None
        try:
            self._sdk = C_PiperInterface_V2(self.config.can_port)
            self._guard = PiperGuard(self._sdk, self.config.can_port)
            self._sdk.ConnectPort(piper_init=False)
            self._guard.wait_ready()
            for name, serial in self.config.camera_serials.items():
                pipeline = NamedCameraPipeline(rs.pipeline(), name, serial)
                self._pipelines[name] = pipeline
                cfg = color_stream_config(rs, serial, self.config.image_width, self.config.image_height)
                pipeline.start(cfg)
            # start() alone does not prove frames arrive. Verify with all streams running,
            # before this session sends any arm/gripper enable command.
            for pipeline in self._pipelines.values():
                pipeline.wait_for_frames()
                print(f"CAMERA_READY: name={pipeline.name}, serial={pipeline.serial}", flush=True)
                self._guard.check(require_enabled=False)
            deadline = time.monotonic() + 5.0
            while True:
                feedback = self._guard.check(require_enabled=False)
                if all_enabled(feedback):
                    break
                if time.monotonic() >= deadline:
                    self._guard.fail("电机使能反馈超时")
                self._sdk.EnableArm(7)
                # Enable at the measured opening, not by commanding the gripper closed.
                self._sdk.GripperCtrl(max(0, min(70000, feedback["gripper_raw"])), 1000, 0x01, 0)
                time.sleep(0.1)
            self._connected = True
            self._guard.start()
        except BaseException:
            self.disconnect()
            raise

    def get_observation(self):
        self._guard.check()
        try:
            observation = super().get_observation()
        except PiperCameraError as exc:
            self._guard.fail(str(exc))
        self._guard.check()
        return observation

    def send_action(self, action):
        if not self._connected:
            raise PiperFaultError("Piper 未连接")
        self._guard.check()
        values = [float(action[f"joint_{i}.pos"]) for i in range(1, 7)]
        gripper = float(action["gripper.pos"])
        if not all(math.isfinite(value) for value in [*values, gripper]):
            self._guard.fail("动作含非有限值")
        # Same rad -> millidegree and normalized opening -> 0.001 mm conversion
        # as the uploaded driver. Configurable speed replaces its hardcoded 100%.
        self._sdk.MotionCtrl_2(0x01, 0x01, self._motion_speed, 0x00)
        self._sdk.JointCtrl(*(round(value * 180000 / math.pi) for value in values))
        self._sdk.GripperCtrl(round(max(0.0, min(1.0, gripper)) * 70000), 1000, 0x01, 0)
        self._guard.check_latch()
        return action

    def disconnect(self):
        # Do not invoke the original disconnect(): it unconditionally DisableArm(7).
        self._connected = False
        guard = getattr(self, "_guard", None)
        sdk = getattr(self, "_sdk", None)
        try:
            if guard is not None:
                guard.shutdown()
        finally:
            for pipeline in self._pipelines.values():
                try:
                    pipeline.stop()
                except Exception as exc:
                    logger.warning("相机关闭失败: %s", exc)
            self._pipelines.clear()
            try:
                if sdk is not None:
                    sdk.DisconnectPort()
            except Exception as exc:
                logger.warning("CAN 关闭失败: %s", exc)
            finally:
                # Retain the stopped guard on the SDK singleton even if Close()
                # fails internally. Restarting requires a new operator-run process.
                self._sdk = None
                self._guard = None
        logger.warning("已尝试关闭通信资源；未发送失能命令，不代表仍有力矩或已安全停稳。")
