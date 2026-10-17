"""Exercise camera diagnostics without installing RealSense or touching a robot."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as Namespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import piper_camera_check as cameras  # noqa: E402


def frameset(number=1):
    return Mock(get_color_frame=Mock(return_value=Mock(get_frame_number=Mock(return_value=number))))


def realsense(pipelines):
    return Namespace(
        pipeline=Mock(side_effect=pipelines),
        config=Mock(),
        stream=Namespace(color=1),
        format=Namespace(bgr8=2),
    )


class CameraTests(unittest.TestCase):
    def test_timeout_names_the_exact_camera_and_never_retries(self):
        raw = Mock()
        raw.wait_for_frames.side_effect = RuntimeError("Frame didn't arrive within 5000")
        pipeline = cameras.NamedCameraPipeline(raw, "cam_wrist", "456")
        with self.assertRaisesRegex(cameras.PiperCameraError, "cam_wrist.*456.*5000") as caught:
            pipeline.wait_for_frames()
        self.assertIs(caught.exception.__cause__, raw.wait_for_frames.side_effect)
        raw.wait_for_frames.assert_called_once_with(5000)
        raw.start.assert_not_called()

    def test_missing_color_is_a_failure_and_valid_frames_are_returned_unchanged(self):
        raw = Mock()
        raw.wait_for_frames.return_value = Mock(get_color_frame=Mock(return_value=None))
        pipeline = cameras.NamedCameraPipeline(raw, "cam_high", "123")
        with self.assertRaisesRegex(cameras.PiperCameraError, "cam_high.*color frame"):
            pipeline.wait_for_frames()
        actual = frameset()
        raw.wait_for_frames.return_value = actual
        self.assertIs(pipeline.wait_for_frames(), actual)

    def test_group_failure_closes_every_pipeline(self):
        first, second = Mock(), Mock()
        first.wait_for_frames.return_value = frameset()
        second.wait_for_frames.side_effect = RuntimeError("camera stalled")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            success = cameras.check_camera_group(
                realsense([first, second]), {"cam_high": "123", "cam_wrist": "456"}, frames=2
            )
        self.assertFalse(success)
        self.assertIn("cam_wrist, serial=456", errors.getvalue())
        first.stop.assert_called_once()
        second.stop.assert_called_once()

    def test_second_start_failure_closes_both_pipelines(self):
        first, second = Mock(), Mock()
        second.start.side_effect = RuntimeError("device busy")
        with contextlib.redirect_stderr(io.StringIO()):
            success = cameras.check_camera_group(
                realsense([first, second]), {"one": "123", "two": "456"}, frames=2
            )
        self.assertFalse(success)
        first.stop.assert_called_once()
        second.stop.assert_called_once()
        first.wait_for_frames.assert_not_called()

    def test_both_streams_are_running_before_first_frame_and_idle_is_tested(self):
        first, second = Mock(), Mock()
        seen = []
        first.start.side_effect = lambda _: seen.append("start_one")
        second.start.side_effect = lambda _: seen.append("start_two")

        def receive():
            self.assertEqual(seen[:2], ["start_one", "start_two"])
            seen.append("read")
            return frameset(len(seen))

        first.wait_for_frames.side_effect = lambda _: receive()
        second.wait_for_frames.side_effect = lambda _: receive()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(cameras.time, "sleep") as sleep:
            success = cameras.check_camera_group(
                realsense([first, second]), {"one": "123", "two": "456"}, frames=3, idle_seconds=10
            )
        self.assertTrue(success)
        sleep.assert_called_once_with(10)
        self.assertEqual(first.wait_for_frames.call_count, 4)
        self.assertEqual(second.wait_for_frames.call_count, 4)

    def test_duplicate_frame_numbers_fail_and_release_stream(self):
        pipeline = Mock()
        pipeline.wait_for_frames.return_value = frameset(42)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            success = cameras.check_camera_group(realsense([pipeline]), {"one": "123"}, frames=2)
        self.assertFalse(success)
        pipeline.stop.assert_called_once()

    def test_serial_mapping_rejects_empty_and_duplicate_devices(self):
        for mapping in ({}, [], {"high": 123}, {"": "123"}, {"high": "123", "wrist": "123"}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                cameras.validate_camera_serials(mapping)

    def test_cli_single_mode_uses_separate_groups_without_importing_robot(self):
        inventory = [{"serial_number": "123"}, {"serial_number": "456"}]
        with (
            patch.dict(sys.modules, {"pyrealsense2": Namespace(), "piper_sdk": None, "torch": None}),
            patch.object(cameras, "camera_inventory", return_value=inventory),
            patch.object(cameras, "check_camera_group", side_effect=[False, True]) as check,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            status = cameras.main(["--mode", "single", "--camera_serials", '{"high":"123","wrist":"456"}'])
        self.assertEqual(status, 1)
        self.assertEqual([call.args[1] for call in check.call_args_list], [{"high": "123"}, {"wrist": "456"}])
        self.assertNotIn("CAMERA_CHECK_OK", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
