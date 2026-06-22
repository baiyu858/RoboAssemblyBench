#!/usr/bin/env python3
"""Record RGB and depth frames from an Intel RealSense camera."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


STOP_REQUESTED = False


def _handle_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record RealSense color frames and depth frames. Color is saved as "
            "standard image files; depth is saved as 16-bit PNG in millimeters."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: recordings/realsense_<timestamp>",
    )
    parser.add_argument("--serial", default=None, help="RealSense serial number to use.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to record. 0 means until Ctrl-C.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum saved frames. 0 means unlimited.")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS.")
    parser.add_argument("--width", type=int, default=640, help="Color stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color stream height.")
    parser.add_argument("--depth-width", type=int, default=640, help="Depth stream width.")
    parser.add_argument("--depth-height", type=int, default=480, help="Depth stream height.")
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to discard frames before recording.")
    parser.add_argument("--keep-every", type=int, default=1, help="Save every Nth captured frame.")
    parser.add_argument(
        "--rgb-format",
        choices=("png", "jpg"),
        default="png",
        help="Color image format.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when --rgb-format=jpg.")
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Do not align depth to color. By default depth is aligned to the color frame.",
    )
    parser.add_argument("--preview", action="store_true", help="Show a live RGB/depth preview window.")
    parser.add_argument(
        "--bag",
        action="store_true",
        help="Also save a raw RealSense recording as stream.db3.",
    )
    parser.add_argument(
        "--record-file",
        type=Path,
        default=None,
        help="Raw RealSense recording path. Must end with .db3 for this SDK.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before recording.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List connected RealSense devices and exit.")
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any]:
    missing: list[str] = []
    try:
        import numpy as np
    except ModuleNotFoundError:
        missing.append("numpy")
        np = None
    try:
        import cv2
    except ModuleNotFoundError:
        missing.append("opencv-python")
        cv2 = None
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError:
        missing.append("pyrealsense2")
        rs = None

    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "Missing Python package(s): "
            f"{', '.join(missing)}\nInstall with:\n  python -m pip install {packages}"
        )
    return np, cv2, rs


def device_info(device: Any, rs: Any) -> dict[str, str]:
    fields = {
        "name": rs.camera_info.name,
        "serial_number": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "product_id": rs.camera_info.product_id,
        "physical_port": rs.camera_info.physical_port,
    }
    info: dict[str, str] = {}
    for key, field in fields.items():
        try:
            info[key] = device.get_info(field) if device.supports(field) else ""
        except RuntimeError:
            info[key] = ""
    return info


def list_devices(rs: Any) -> None:
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return
    for index, device in enumerate(devices, start=1):
        info = device_info(device, rs)
        print(
            f"{index}) {info.get('name', '')} "
            f"serial={info.get('serial_number', '')} "
            f"fw={info.get('firmware_version', '')} "
            f"pid={info.get('product_id', '')}"
        )


def intrinsics_to_dict(intrinsics: Any) -> dict[str, Any]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def make_output_dir(path: Path | None, overwrite: bool) -> Path:
    if path is not None:
        output_dir = path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("recordings") / f"realsense_{stamp}"

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    existing_outputs = [
        output_dir / "metadata.csv",
        output_dir / "intrinsics.json",
        output_dir / "stream.db3",
    ]
    has_existing_frames = any(rgb_dir.iterdir()) or any(depth_dir.iterdir())
    has_existing_metadata = any(path.exists() for path in existing_outputs)
    if (has_existing_frames or has_existing_metadata) and not overwrite:
        raise SystemExit(
            f"Output directory is not empty: {output_dir}\n"
            "Use a new --output-dir or pass --overwrite."
        )

    return output_dir


def resolve_record_file(output_dir: Path, args: argparse.Namespace) -> Path | None:
    if args.record_file is None and not args.bag:
        return None

    record_file = args.record_file if args.record_file is not None else output_dir / "stream.db3"
    if record_file.suffix.lower() != ".db3":
        raise SystemExit(
            f"Raw RealSense recording must use .db3 with this SDK: {record_file}\n"
            "Example: --record-file recordings/test_rgbd1/stream.db3"
        )
    record_file.parent.mkdir(parents=True, exist_ok=True)
    return record_file


def write_intrinsics(
    output_dir: Path,
    args: argparse.Namespace,
    rs: Any,
    profile: Any,
    depth_scale: float,
    device: Any,
) -> None:
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    payload = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "device": device_info(device, rs),
        "sdk_version": getattr(rs, "__version__", ""),
        "depth_scale_m_per_unit": depth_scale,
        "depth_png_units": "millimeters",
        "depth_aligned_to_color": not args.no_align,
        "color_intrinsics": intrinsics_to_dict(color_profile.get_intrinsics()),
        "depth_intrinsics": intrinsics_to_dict(depth_profile.get_intrinsics()),
        "streams": {
            "color": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "format": "rgb8",
            },
            "depth": {
                "width": args.depth_width,
                "height": args.depth_height,
                "fps": args.fps,
                "format": "z16",
            },
        },
    }
    (output_dir / "intrinsics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_color(cv2: Any, path: Path, color_image: Any, args: argparse.Namespace) -> None:
    color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    if args.rgb_format == "jpg":
        ok = cv2.imwrite(str(path), color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
    else:
        ok = cv2.imwrite(str(path), color_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write color image: {path}")


def save_depth(cv2: Any, np: Any, path: Path, depth_raw: Any, depth_scale: float) -> None:
    depth_mm = np.rint(depth_raw.astype(np.float32) * depth_scale * 1000.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    ok = cv2.imwrite(str(path), depth_mm)
    if not ok:
        raise RuntimeError(f"Failed to write depth image: {path}")


def main() -> int:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    args = parse_args()
    if args.duration < 0:
        raise SystemExit("--duration must be >= 0")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")
    if args.keep_every < 1:
        raise SystemExit("--keep-every must be >= 1")

    np, cv2, rs = import_dependencies()

    if args.list_devices:
        list_devices(rs)
        return 0

    output_dir = make_output_dir(args.output_dir, args.overwrite)
    print(f"Recording to: {output_dir.resolve()}")
    record_file = resolve_record_file(output_dir, args)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)
    if record_file is not None:
        print(f"Raw RealSense recording: {record_file.resolve()}")
        config.enable_record_to_file(str(record_file))

    profile = pipeline.start(config)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    aligner = None if args.no_align else rs.align(rs.stream.color)
    write_intrinsics(output_dir, args, rs, profile, depth_scale, device)

    csv_path = output_dir / "metadata.csv"
    saved = 0
    captured = 0
    start_time = time.monotonic()
    record_start_time = start_time + max(args.warmup, 0.0)

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as metadata_file:
            writer = csv.writer(metadata_file)
            writer.writerow(
                [
                    "frame_index",
                    "captured_index",
                    "elapsed_s",
                    "color_timestamp_ms",
                    "depth_timestamp_ms",
                    "color_frame_number",
                    "depth_frame_number",
                    "rgb_path",
                    "depth_path",
                    "depth_scale_m_per_unit",
                ]
            )

            print("Warming up..." if args.warmup > 0 else "Recording...")
            while not STOP_REQUESTED:
                now = time.monotonic()
                if args.duration and now - record_start_time >= args.duration:
                    break
                if args.max_frames and saved >= args.max_frames:
                    break

                frames = pipeline.wait_for_frames(5000)
                if aligner is not None:
                    frames = aligner.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                captured += 1
                if now < record_start_time:
                    continue
                if captured % args.keep_every != 0:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())

                name = f"{saved:06d}"
                rgb_path = output_dir / "rgb" / f"{name}.{args.rgb_format}"
                depth_path = output_dir / "depth" / f"{name}.png"
                save_color(cv2, rgb_path, color_image, args)
                save_depth(cv2, np, depth_path, depth_raw, depth_scale)

                writer.writerow(
                    [
                        saved,
                        captured,
                        f"{now - record_start_time:.6f}",
                        f"{color_frame.get_timestamp():.3f}",
                        f"{depth_frame.get_timestamp():.3f}",
                        color_frame.get_frame_number(),
                        depth_frame.get_frame_number(),
                        rgb_path.relative_to(output_dir),
                        depth_path.relative_to(output_dir),
                        f"{depth_scale:.10f}",
                    ]
                )
                saved += 1

                if saved == 1:
                    print("Recording...")
                if saved % max(args.fps, 1) == 0:
                    print(f"Saved {saved} frames", end="\r", flush=True)

                if args.preview:
                    color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                    depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    preview = np.hstack((color_bgr, depth_vis))
                    cv2.imshow("RealSense RGB | Depth", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
    finally:
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"\nSaved {saved} RGB/depth frame pairs.")
    print(f"Metadata: {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
