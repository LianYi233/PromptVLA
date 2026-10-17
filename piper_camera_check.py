#!/usr/bin/env python3
"""Diagnose Piper's RealSense color streams without importing a robot or opening CAN."""

import argparse
import json
import sys
import time


class PiperCameraError(RuntimeError):
    """A camera failed to start or provide a color frame; no fallback image is used."""


def validate_camera_serials(camera_serials):
    if (
        not isinstance(camera_serials, dict)
        or not camera_serials
        or any(
            not isinstance(name, str) or not name.strip() or not isinstance(serial, str) or not serial.strip()
            for name, serial in camera_serials.items()
        )
    ):
        raise ValueError("camera_serials 必须是非空的 {相机名称: 序列号字符串} 字典")
    if len(set(camera_serials.values())) != len(camera_serials):
        raise ValueError("camera_serials 不能将两个相机名称绑定到同一个序列号")


class NamedCameraPipeline:
    """Preserve the driver's frameset interface, adding camera identity to failures."""

    def __init__(self, pipeline, name, serial):
        self.pipeline = pipeline
        self.name = name
        self.serial = serial

    def start(self, config):
        try:
            return self.pipeline.start(config)
        except RuntimeError as exc:
            raise PiperCameraError(f"相机启动失败: name={self.name}, serial={self.serial}: {exc}") from exc

    def wait_for_frames(self, timeout_ms=5000):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if not frames or not frames.get_color_frame():
                raise RuntimeError("未收到有效 color frame")
            return frames
        except RuntimeError as exc:
            raise PiperCameraError(
                f"相机取帧失败: name={self.name}, serial={self.serial}, timeout_ms={timeout_ms}: {exc}"
            ) from exc

    def stop(self):
        return self.pipeline.stop()


def color_stream_config(rs, serial, width, height, fps=30):
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    return config


def camera_inventory(rs):
    inventory = []
    for device in rs.context().query_devices():
        record = {}
        for name in ("name", "serial_number", "firmware_version", "usb_type_descriptor"):
            key = getattr(rs.camera_info, name)
            record[name] = device.get_info(key) if device.supports(key) else "unavailable"
        inventory.append(record)
    return inventory


def check_camera_group(rs, camera_serials, *, width=640, height=480, fps=30, frames=60, idle_seconds=0):
    """Start a group together, read first frames, optionally idle, then read each stream."""
    validate_camera_serials(camera_serials)
    pipelines = []
    success = True
    try:
        for name, serial in camera_serials.items():
            pipeline = NamedCameraPipeline(rs.pipeline(), name, serial)
            pipelines.append(pipeline)
            pipeline.start(color_stream_config(rs, serial, width, height, fps))
        for pipeline in pipelines:
            pipeline.wait_for_frames()
            print(f"CAMERA_FIRST_FRAME_OK: name={pipeline.name}, serial={pipeline.serial}", flush=True)
        if idle_seconds:
            print(f"暂停取帧 {idle_seconds}s，模拟等待 Enter；相机保持启动。", flush=True)
            time.sleep(idle_seconds)
        last_numbers = {}
        for _ in range(frames):
            for pipeline in pipelines:
                color = pipeline.wait_for_frames().get_color_frame()
                number = color.get_frame_number()
                if last_numbers.get(pipeline.name) == number:
                    raise PiperCameraError(f"相机帧号未更新: name={pipeline.name}, serial={pipeline.serial}")
                last_numbers[pipeline.name] = number
                del color
        for pipeline in pipelines:
            print(
                f"CAMERA_STREAM_OK: name={pipeline.name}, serial={pipeline.serial}, "
                f"frames={frames}, last_frame={last_numbers[pipeline.name]}, "
                f"stream={width}x{height}@{fps} BGR8",
                flush=True,
            )
    except RuntimeError as exc:
        success = False
        print(f"CAMERA_FAIL: {exc}", file=sys.stderr, flush=True)
    finally:
        for pipeline in reversed(pipelines):
            try:
                pipeline.stop()
            except RuntimeError as exc:
                success = False
                print(
                    f"CAMERA_CLOSE_FAIL: name={pipeline.name}, serial={pipeline.serial}: {exc}",
                    file=sys.stderr,
                )
    return success


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="仅列出 RealSense 序列号、固件和 USB 类型")
    parser.add_argument("--camera_serials", help='与部署相同的 JSON，例如 {"cam_high":"序列号"}')
    parser.add_argument("--mode", choices=("single", "together"), default="together")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="相机帧率，独立于机械臂 10Hz 控制频率")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--idle_seconds", type=int, default=0, help="首帧后暂停读取，再验证能否继续取帧")
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.fps, args.frames) <= 0 or args.idle_seconds < 0:
        parser.error("width/height/fps/frames 必须为正数，idle_seconds 不得为负")
    selected = json.loads(args.camera_serials) if args.camera_serials is not None else None
    if selected is not None:
        validate_camera_serials(selected)

    import pyrealsense2 as rs

    inventory = camera_inventory(rs)
    print("RealSense:", json.dumps(inventory, ensure_ascii=False, indent=2), flush=True)
    if not inventory:
        raise RuntimeError("未发现 RealSense 设备")
    if args.list:
        return 0
    if selected is None:
        # Device discovery cannot infer the training camera-to-view mapping.
        selected = {f"serial_{item['serial_number']}": item["serial_number"] for item in inventory}
        print("未指定 camera_serials：检查所有枚举设备，不验证 cam_high/cam_wrist 对应关系。", flush=True)
    available = {item["serial_number"] for item in inventory}
    if missing := set(selected.values()) - available:
        raise ValueError(f"指定相机未连接: {sorted(missing)}")
    groups = [{name: serial} for name, serial in selected.items()] if args.mode == "single" else [selected]
    all_ok = True
    for group in groups:
        ok = check_camera_group(
            rs,
            group,
            width=args.width,
            height=args.height,
            fps=args.fps,
            frames=args.frames,
            idle_seconds=args.idle_seconds,
        )
        all_ok = all_ok and ok
    if all_ok:
        print("CAMERA_CHECK_OK: 仅验证所选相机取帧；未加载模型、未导入 Piper SDK、未连接 CAN。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"CAMERA_CHECK_FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
