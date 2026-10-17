"""Hardware-free fault injection; optionally exercise piper-sdk 0.6.1 encoders."""

# ruff: noqa: N801, N802 -- fake SDK preserves its public method/enum names.

import contextlib
import io
import math
import sys
import threading
import time
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as Namespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import piper_deploy_guard as guard_module  # noqa: E402
from piper_deploy_guard import (  # noqa: E402
    FeedbackUnavailableError,
    GuardedPiperMixin,
    PiperFaultError,
    PiperGuard,
    check_robot_connection,
    feedback_snapshot,
    require_healthy_can,
    require_healthy_feedback,
)


def can_link(state="ERROR-ACTIVE", bitrate=1000000):
    return {
        "flags": ["UP", "LOWER_UP"],
        "linkinfo": {"info_kind": "can", "info_data": {"state": state, "bittiming": {"bitrate": bitrate}}},
    }


class FakeTransport:
    class CAN_STATUS(Enum):
        SEND_MESSAGE_SUCCESS = 100016
        SEND_MESSAGE_FAILED = 100017

    def __init__(self):
        self.frames = []
        self.fail_id = None
        self.stopped = threading.Event()

    def SendCanMessage(self, arbitration_id, data, *args, **kwargs):
        self.frames.append((arbitration_id, bytes(data)))
        if arbitration_id == 0x150:
            self.stopped.set()
        if arbitration_id == self.fail_id:
            return self.CAN_STATUS.SEND_MESSAGE_FAILED
        return self.CAN_STATUS.SEND_MESSAGE_SUCCESS


def feedback_messages():
    def foc():
        return Namespace(
            **dict.fromkeys(set(guard_module._MOTOR_FAULTS + guard_module._GRIPPER_FAULTS), False),
            driver_enable_status=True,
        )

    return {
        "status": Namespace(arm_status=Namespace(arm_status=0, err_code=0, ctrl_mode=1)),
        "joints": Namespace(joint_state=Namespace(**{f"joint_{i}": 1000 * i for i in range(1, 7)})),
        "gripper": Namespace(gripper_state=Namespace(grippers_angle=35000, foc_status=foc())),
        "motors": Namespace(
            **{f"motor_{i}": Namespace(can_id=0x260 + i, vol=240, foc_status=foc()) for i in range(1, 7)}
        ),
    }


class FakeSDK:
    def __init__(self):
        self.transport = FakeTransport()
        self.messages = feedback_messages()
        self.age = 0
        self.ConnectPort = Mock()
        self.DisconnectPort = Mock()
        self.EnableArm = Mock()
        self.DisableArm = Mock()
        self.ResetPiper = Mock()
        self.GripperCtrl = Mock()

    def GetCanBus(self):
        return self.transport

    def message(self, name):
        self.messages[name].time_stamp = time.time() - self.age
        return self.messages[name]

    def GetArmStatus(self):
        return self.message("status")

    def GetArmJointMsgs(self):
        return self.message("joints")

    def GetArmGripperMsgs(self):
        return self.message("gripper")

    def GetArmLowSpdInfoMsgs(self):
        return self.message("motors")

    def MotionCtrl_1(self, emergency_stop, track_ctrl, teach_ctrl):
        self.transport.SendCanMessage(0x150, [emergency_stop, track_ctrl, teach_ctrl, 0, 0, 0, 0, 0])


def robot_for(sdk):
    robot = GuardedPiperMixin()
    robot._sdk = sdk
    robot._guard = PiperGuard(sdk, "can0")
    robot._connected = True
    robot._pipelines = {}
    robot._motion_speed = 20
    robot.config = Namespace(
        can_port="can0", camera_serials={"top": "123"}, image_width=640, image_height=480
    )
    return robot


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.link_patch = patch.object(guard_module, "read_can_link", return_value=can_link())
        self.read_link = self.link_patch.start()
        self.addCleanup(self.link_patch.stop)

    def test_kernel_fault_refuses_connection_before_sdk_import_or_enable(self):
        robot = GuardedPiperMixin()
        robot._connected = False
        robot.config = Namespace(can_port="can0")
        for state in ("ERROR-WARNING", "ERROR-PASSIVE", "BUS-OFF", "STOPPED", None):
            with self.subTest(state=state), patch.dict(sys.modules, {"piper_sdk": None}):
                self.read_link.return_value = can_link(state)
                with self.assertRaisesRegex(PiperFaultError, "CAN 未就绪"):
                    robot.connect()
        with self.assertRaises(PiperFaultError):
            require_healthy_can(can_link(bitrate=500000))

    def test_send_failure_latches_and_only_allows_one_quick_stop(self):
        sdk = FakeSDK()
        guard = PiperGuard(sdk, "can0")
        sdk.transport.fail_id = 0x159
        with self.assertRaisesRegex(PiperFaultError, "0x159"):
            sdk.transport.SendCanMessage(0x159, bytes(8))
        sdk.transport.fail_id = None
        with self.assertRaises(PiperFaultError):
            sdk.transport.SendCanMessage(0x151, bytes(8))
        guard.request_stop()
        guard.request_stop()
        with self.assertRaises(PiperFaultError):
            sdk.MotionCtrl_1(2, 0, 0)
        self.assertEqual(sdk.transport.frames, [(0x159, bytes(8)), (0x150, bytes([1] + [0] * 7))])
        sdk.EnableArm.assert_not_called()
        sdk.DisableArm.assert_not_called()
        sdk.ResetPiper.assert_not_called()

    def test_failed_stop_is_not_retried_and_never_clears_original_fault(self):
        sdk = FakeSDK()
        guard = PiperGuard(sdk, "can0")
        with self.assertRaises(PiperFaultError):
            guard.fail("initial fault")
        sdk.transport.fail_id = 0x150
        guard.shutdown()
        guard.shutdown()
        self.assertEqual(guard.fault, "initial fault")
        self.assertEqual(len(sdk.transport.frames), 1)

    def test_freshness_missing_motor_and_controller_faults(self):
        sdk = FakeSDK()
        sdk.age = 1
        with self.assertRaises(FeedbackUnavailableError):
            feedback_snapshot(sdk)
        sdk.age = 0
        sdk.messages["motors"].motor_6.can_id = 0
        with self.assertRaises(FeedbackUnavailableError):
            feedback_snapshot(sdk)
        sdk.messages["motors"].motor_6.can_id = 0x266
        snapshot = feedback_snapshot(sdk)
        self.assertEqual(snapshot["motors"][0]["voltage_V"], 24)
        self.assertAlmostEqual(snapshot["joint_rad"][0], math.pi / 180)
        require_healthy_feedback(snapshot)
        for fault in guard_module._MOTOR_FAULTS:
            with self.subTest(fault=fault):
                setattr(sdk.messages["motors"].motor_4.foc_status, fault, True)
                with self.assertRaisesRegex(PiperFaultError, fault):
                    require_healthy_feedback(feedback_snapshot(sdk))
                setattr(sdk.messages["motors"].motor_4.foc_status, fault, False)
        sdk.messages["status"].arm_status.arm_status = 7
        with self.assertRaisesRegex(PiperFaultError, "反馈异常"):
            require_healthy_feedback(feedback_snapshot(sdk))

    def test_disabled_feedback_allowed_only_before_enable(self):
        sdk = FakeSDK()
        sdk.messages["motors"].motor_2.foc_status.driver_enable_status = False
        snapshot = feedback_snapshot(sdk)
        require_healthy_feedback(snapshot, require_enabled=False)
        with self.assertRaisesRegex(PiperFaultError, "失能"):
            require_healthy_feedback(snapshot)

    def test_watchdog_stops_during_idle_without_action_calls(self):
        sdk = FakeSDK()
        guard = PiperGuard(sdk, "can0")
        guard.check()
        sdk.messages["motors"].motor_1.foc_status.collision_status = True
        guard.start()
        try:
            self.assertTrue(sdk.transport.stopped.wait(2), "idle watchdog did not issue stop")
            self.assertIsNotNone(guard.fault)
            sdk.EnableArm.assert_not_called()
        finally:
            guard.shutdown()

    def test_cleanup_attempts_all_resources_without_disabling_or_clearing_faults(self):
        sdk = FakeSDK()
        robot = robot_for(sdk)
        camera1, camera2 = Mock(), Mock()
        camera1.stop.side_effect = RuntimeError("camera failed")
        sdk.DisconnectPort.side_effect = RuntimeError("close failed")
        robot._pipelines = {"one": camera1, "two": camera2}
        robot.disconnect()
        robot.disconnect()
        camera1.stop.assert_called_once()
        camera2.stop.assert_called_once()
        sdk.DisconnectPort.assert_called_once()
        sdk.DisableArm.assert_not_called()
        sdk.GripperCtrl.assert_not_called()
        self.assertEqual(sdk.transport.frames, [(0x150, bytes([1] + [0] * 7))])
        with self.assertRaises(PiperFaultError):
            sdk.transport.SendCanMessage(0x151, bytes(8))

    def test_camera_start_failure_closes_port_without_enabling(self):
        sdk = FakeSDK()
        robot = robot_for(sdk)
        robot._guard.uninstall()
        robot._connected = False
        pipeline = Mock()
        pipeline.start.side_effect = RuntimeError("camera failed")
        rs = Namespace(
            pipeline=Mock(return_value=pipeline),
            config=Mock(),
            stream=Namespace(color=1),
            format=Namespace(bgr8=2),
        )
        with (
            patch.dict(
                sys.modules, {"piper_sdk": Namespace(C_PiperInterface_V2=lambda _: sdk), "pyrealsense2": rs}
            ),
            self.assertRaisesRegex(RuntimeError, "camera failed"),
        ):
            robot.connect()
        pipeline.stop.assert_called_once()
        sdk.DisconnectPort.assert_called_once()
        sdk.EnableArm.assert_not_called()
        sdk.GripperCtrl.assert_not_called()
        sdk.DisableArm.assert_not_called()
        self.assertFalse(robot._connected)

    def test_read_only_diagnostic_never_sends_even_when_motors_disabled(self):
        sdk = FakeSDK()
        for i in range(1, 7):
            getattr(sdk.messages["motors"], f"motor_{i}").foc_status.driver_enable_status = False
        sdk.messages["gripper"].gripper_state.foc_status.driver_enable_status = False
        with (
            patch.dict(
                sys.modules, {"piper_sdk": Namespace(C_PiperInterface_V2=lambda _: sdk), "pyrealsense2": None}
            ),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            check_robot_connection("can0")
        self.assertIn("ROBOT_READ_OK", output.getvalue())
        sdk.ConnectPort.assert_called_once_with(piper_init=False)
        sdk.DisconnectPort.assert_called_once()
        sdk.EnableArm.assert_not_called()
        sdk.GripperCtrl.assert_not_called()
        self.assertEqual(sdk.transport.frames, [])

    def test_camera_first_frame_failure_never_enables_arm_or_gripper(self):
        sdk = FakeSDK()
        for i in range(1, 7):
            getattr(sdk.messages["motors"], f"motor_{i}").foc_status.driver_enable_status = False
        robot = robot_for(sdk)
        robot._guard.uninstall()
        robot._connected = False
        pipeline = Mock()
        pipeline.wait_for_frames.side_effect = RuntimeError("Frame didn't arrive within 5000")
        rs = Namespace(
            pipeline=Mock(return_value=pipeline),
            config=Mock(),
            stream=Namespace(color=1),
            format=Namespace(bgr8=2),
        )
        with (
            patch.dict(
                sys.modules, {"piper_sdk": Namespace(C_PiperInterface_V2=lambda _: sdk), "pyrealsense2": rs}
            ),
            self.assertRaisesRegex(guard_module.PiperCameraError, "name=top, serial=123"),
        ):
            robot.connect()
        sdk.EnableArm.assert_not_called()
        sdk.GripperCtrl.assert_not_called()
        sdk.DisableArm.assert_not_called()
        pipeline.stop.assert_called_once()
        sdk.DisconnectPort.assert_called_once()
        self.assertFalse(robot._connected)

    def test_camera_runtime_failure_latches_future_actions(self):
        class OriginalDriver:
            def get_observation(self):
                raise guard_module.PiperCameraError("name=cam_wrist, serial=456: timeout")

        class Deployment(GuardedPiperMixin, OriginalDriver):
            pass

        sdk = FakeSDK()
        robot = Deployment()
        robot._guard = PiperGuard(sdk, "can0")
        with self.assertRaisesRegex(PiperFaultError, "cam_wrist"):
            robot.get_observation()
        with self.assertRaisesRegex(PiperFaultError, "cam_wrist"):
            sdk.transport.SendCanMessage(0x151, bytes(8))
        self.assertEqual(sdk.transport.frames, [])
        robot._guard.shutdown()

    def test_enable_waits_for_color_frames_from_both_cameras(self):
        sdk = FakeSDK()
        for i in range(1, 7):
            getattr(sdk.messages["motors"], f"motor_{i}").foc_status.driver_enable_status = False
        robot = robot_for(sdk)
        robot._guard.uninstall()
        robot._connected = False
        robot.config.camera_serials = {"high": "123", "wrist": "456"}
        first, second = Mock(), Mock()

        def enable(_):
            first.wait_for_frames.assert_called_once_with(5000)
            second.wait_for_frames.assert_called_once_with(5000)
            for i in range(1, 7):
                getattr(sdk.messages["motors"], f"motor_{i}").foc_status.driver_enable_status = True

        sdk.EnableArm.side_effect = enable
        rs = Namespace(
            pipeline=Mock(side_effect=[first, second]),
            config=Mock(),
            stream=Namespace(color=1),
            format=Namespace(bgr8=2),
        )
        with (
            patch.dict(
                sys.modules,
                {"piper_sdk": Namespace(C_PiperInterface_V2=lambda _: sdk), "pyrealsense2": rs},
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                robot.connect()
                self.assertTrue(robot._connected)
                sdk.EnableArm.assert_called_once_with(7)
            finally:
                robot.disconnect()
        first.stop.assert_called_once()
        second.stop.assert_called_once()


class ActualSDKTests(unittest.TestCase):
    """Use real SDK serialization, but never construct an SDK or open a CAN bus."""

    @classmethod
    def setUpClass(cls):
        try:
            from piper_sdk import C_PiperInterface_V2
            from piper_sdk.protocol.protocol_v2 import C_PiperParserV2
        except ImportError:
            raise unittest.SkipTest("Run with --with piper-sdk==0.6.1 for actual SDK tests") from None
        cls.sdk_class = C_PiperInterface_V2
        cls.parser_class = C_PiperParserV2

    def setUp(self):
        self.link_patch = patch.object(guard_module, "read_can_link", return_value=can_link())
        self.link_patch.start()
        self.addCleanup(self.link_patch.stop)
        sdk = object.__new__(self.sdk_class)
        self.transport = FakeTransport()
        # Inject only transport/parser/limit switches, bypassing the hardware constructor.
        sdk._C_PiperInterface_V2__arm_can = self.transport
        sdk._C_PiperInterface_V2__parser = self.parser_class()
        sdk._C_PiperInterface_V2__start_sdk_joint_limit = False
        sdk._C_PiperInterface_V2__start_sdk_gripper_limit = False
        source = FakeSDK()
        for name in ("GetArmStatus", "GetArmJointMsgs", "GetArmGripperMsgs", "GetArmLowSpdInfoMsgs"):
            setattr(sdk, name, getattr(source, name))
        self.sdk = sdk
        self.robot = robot_for(sdk)

    def action(self):
        return {**{f"joint_{i}.pos": math.pi / 180 * i for i in range(1, 7)}, "gripper.pos": 0.5}

    def test_action_encoding_speed_and_units(self):
        self.robot.send_action(self.action())
        frames = self.transport.frames
        self.assertEqual([frame[0] for frame in frames], [0x151, 0x155, 0x156, 0x157, 0x159])
        self.assertEqual(frames[0][1][:4], bytes([1, 1, 20, 0]))
        self.assertEqual(int.from_bytes(frames[1][1][:4], "big", signed=True), 1000)
        self.assertEqual(int.from_bytes(frames[3][1][4:], "big", signed=True), 6000)
        self.assertEqual(int.from_bytes(frames[4][1][:4], "big", signed=True), 35000)

    def test_each_sdk_tx_failure_aborts_remaining_frames_and_allows_quick_stop(self):
        for failed_id in (0x151, 0x155, 0x156, 0x157, 0x159):
            with self.subTest(failed_id=hex(failed_id)):
                self.robot._guard.uninstall()
                self.robot._guard = PiperGuard(self.sdk, "can0")
                self.transport.frames.clear()
                self.transport.fail_id = failed_id
                with self.assertRaisesRegex(PiperFaultError, hex(failed_id)):
                    self.robot.send_action(self.action())
                sent_ids = [frame[0] for frame in self.transport.frames]
                self.assertEqual(sent_ids[-1], failed_id)
                before = len(sent_ids)
                with self.assertRaises(PiperFaultError):
                    self.robot.send_action(self.action())
                self.assertEqual(len(self.transport.frames), before)
                self.robot._guard.request_stop()
                self.assertEqual(self.transport.frames[-1], (0x150, bytes([1] + [0] * 7)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
