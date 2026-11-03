#!/usr/bin/env python3
"""Record synchronized, color-aligned Intel RealSense RGB-D frames."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from dataset_config import dataset_path

try:
    import pyrealsense2 as rs
except ImportError as error:
    raise RuntimeError(
        "pyrealsense2 is required. Install it in this environment with "
        "`python -m pip install pyrealsense2`."
    ) from error


def dataset_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    return dataset_path(config["path"], config["dataset_id"])


def prepare_output(folder: Path, overwrite: bool) -> None:
    if folder.exists() and any(folder.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output folder is not empty: {folder}. Use --overwrite "
                "only if these frames may be replaced."
            )
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def check_output(folder: Path, overwrite: bool) -> None:
    if folder.exists() and any(folder.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output folder is not empty: {folder}. Use --overwrite "
            "only if these frames may be replaced."
        )


def camera_matrix(intrinsics) -> list[float]:
    return [
        float(intrinsics.fx), 0.0, float(intrinsics.ppx),
        0.0, float(intrinsics.fy), float(intrinsics.ppy),
        0.0, 0.0, 1.0,
    ]


def validate_camera_json(path: Path, width: int, height: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibrated camera file not found: {path}. Pass --camera-json."
        )
    data = json.loads(path.read_text())
    values = data.get("cam_K")
    if not isinstance(values, list) or len(values) != 9:
        raise ValueError(f"{path} must contain cam_K with nine values")
    calibrated_width = data.get("image_width")
    calibrated_height = data.get("image_height")
    if calibrated_width is not None and int(calibrated_width) != width:
        raise ValueError(
            f"Calibration width {calibrated_width} does not match capture width {width}"
        )
    if calibrated_height is not None and int(calibrated_height) != height:
        raise ValueError(
            f"Calibration height {calibrated_height} does not match capture height {height}"
        )
    # Saved depth PNGs are explicitly converted to integer millimetres.
    data["depth_scale"] = 1.0
    data["depth_unit"] = "millimetre"
    data["image_width"] = width
    data["image_height"] = height
    return data


def depth_colormap(depth_mm: np.ndarray, maximum_mm: int) -> np.ndarray:
    scaled = np.clip(depth_mm.astype(np.float32) / maximum_mm * 255.0, 0, 255)
    colored = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth_mm == 0] = 0
    return colored


def add_preview_bar(image: np.ndarray, status: str, controls: str) -> np.ndarray:
    result = cv2.copyMakeBorder(
        image, 0, 78, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 29)
    )
    cv2.putText(result, status, (16, image.shape[0] + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 220, 255), 2,
                cv2.LINE_AA)
    cv2.putText(result, controls, (16, image.shape[0] + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2,
                cv2.LINE_AA)
    return result


def write_rgbd_pair(rgb_path: Path, depth_path: Path,
                    color: np.ndarray, depth_mm: np.ndarray) -> None:
    """Encode the independent RGB and depth PNGs concurrently."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        rgb_future = executor.submit(cv2.imwrite, str(rgb_path), color)
        depth_future = executor.submit(cv2.imwrite, str(depth_path), depth_mm)
        if not rgb_future.result():
            raise RuntimeError(f"Failed to write {rgb_path}")
        if not depth_future.result():
            raise RuntimeError(f"Failed to write {depth_path}")


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=project / "scripts/configs/default.yaml",
        help="Project config used to resolve data/####.",
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--camera-json", type=Path,
        help=(
            "Source calibrated RGB camera.json; defaults to "
            "calibration/charuco/camera.json. A copy is placed in data/####."
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--depth-width", type=int, default=1280)
    parser.add_argument(
        "--depth-height", type=int, default=720,
        help=(
            "Native depth stream height before alignment (default: 720). "
            "It does not need to equal the RGB height."
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--target-frames", type=int, default=250,
        help="Stop after saving this many RGB-D pairs (default: 250).",
    )
    parser.add_argument(
        "--save-fps", type=float, default=5.0,
        help="PNG sampling rate; the optional bag still records at --fps.",
    )
    parser.add_argument(
        "--duration", type=float,
        help="Stop after this many seconds; otherwise press Q or Ctrl-C.",
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--serial", help="Optional RealSense device serial number")
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Keep distorted RGB/depth pixels instead of applying camera.json distortion.",
    )
    parser.add_argument(
        "--bag", type=Path,
        help="Optionally record the original synchronized streams to a .bag file.",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-max-depth-m", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.save_fps <= 0 or args.save_fps > args.fps:
        raise ValueError("--save-fps must be greater than zero and no larger than --fps")
    if args.no_preview and args.duration is None:
        raise ValueError("--no-preview requires --duration")
    if args.preview_max_depth_m <= 0:
        raise ValueError("Preview maximum depth must be positive")
    if args.target_frames <= 0:
        raise ValueError("--target-frames must be positive")

    config_path = args.config.expanduser().resolve()
    dataset = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir else dataset_from_config(config_path)
    )
    dataset.mkdir(parents=True, exist_ok=True)
    images_dir = dataset / "images"
    depth_dir = dataset / "depth_maps"
    check_output(images_dir, args.overwrite)
    check_output(depth_dir, args.overwrite)

    calibrated_path = (
        args.camera_json.expanduser().resolve()
        if args.camera_json else project / "calibration/charuco/camera.json"
    )
    calibrated = validate_camera_json(
        calibrated_path, args.width, args.height
    )
    dataset_camera_path = dataset / "camera.json"
    if calibrated_path != dataset_camera_path:
        shutil.copy2(calibrated_path, dataset_camera_path)
    print(f"Dataset directory: {dataset}")
    print(f"Copied camera calibration to: {dataset_camera_path}")
    calibrated_K = np.asarray(calibrated["cam_K"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(
        calibrated.get("distortion", [0.0] * 5), dtype=np.float64
    ).reshape(-1)
    rectify_maps = None
    if not args.no_undistort and np.any(np.abs(distortion) > 1e-12):
        rectify_maps = cv2.initUndistortRectifyMap(
            calibrated_K,
            distortion,
            None,
            calibrated_K,
            (args.width, args.height),
            cv2.CV_32FC1,
        )

    pipeline = rs.pipeline()
    stream_config = rs.config()
    if args.serial:
        stream_config.enable_device(args.serial)
    stream_config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    stream_config.enable_stream(
        rs.stream.depth,
        args.depth_width,
        args.depth_height,
        rs.format.z16,
        args.fps,
    )
    if args.bag:
        bag_path = args.bag.expanduser().resolve()
        bag_path.parent.mkdir(parents=True, exist_ok=True)
        if bag_path.exists() and not args.overwrite:
            raise FileExistsError(bag_path)
        if bag_path.exists():
            bag_path.unlink()
        stream_config.enable_record_to_file(str(bag_path))

    try:
        profile = pipeline.start(stream_config)
    except RuntimeError as error:
        requested = (
            f"RGB {args.width}x{args.height} {args.fps} FPS bgr8 and "
            f"depth {args.depth_width}x{args.depth_height} "
            f"{args.fps} FPS z16"
        )
        raise RuntimeError(
            f"RealSense could not start the requested profiles: {requested}. "
            "The RGB resolution must remain equal to the ChArUco calibration "
            f"({args.width}x{args.height}), but the native depth resolution may "
            "be different because it is aligned to RGB afterward. Try "
            "`--depth-width 848 --depth-height 480`, and verify that no other "
            "application is using the camera."
        ) from error
    align_to_color = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    native_depth_scale_m = float(depth_sensor.get_depth_scale())
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrinsics = color_profile.get_intrinsics()
    device = profile.get_device()
    device_name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)

    # Only replace the previous frames after the requested RealSense stream
    # profiles have started successfully.
    prepare_output(images_dir, args.overwrite)
    prepare_output(depth_dir, args.overwrite)

    # Write a capture-specific copy while retaining the user's calibrated K.
    camera_output = dict(calibrated)
    camera_output["source"] = "ChArUco RGB calibration"
    camera_output["depth_alignment"] = "depth_to_color"
    camera_output["realsense_native_depth_scale_m"] = native_depth_scale_m
    camera_output["realsense_factory_color_K"] = camera_matrix(color_intrinsics)
    camera_output["device_name"] = device_name
    camera_output["device_serial"] = serial
    camera_output["images_undistorted"] = rectify_maps is not None
    if rectify_maps is not None:
        # The output pinhole images use the same K selected for rectification.
        camera_output["source_distortion"] = distortion.tolist()
        camera_output["distortion"] = [0.0] * len(distortion)
    dataset_camera_path.write_text(
        json.dumps(camera_output, indent=2) + "\n"
    )

    records = []
    saved = 0
    received = 0
    capturing = bool(args.no_preview)
    capture_started_at = None
    capture_elapsed = 0.0
    next_save_time = 0.0
    print(f"Recording {device_name} ({serial})")
    print(
        f"Streams: {args.width}x{args.height}@{args.fps}; "
        f"saving {args.save_fps:g} RGB-D pairs/s"
    )
    print(f"Native depth scale: {native_depth_scale_m:.9f} m/unit")
    if not args.no_preview:
        print("OpenCV preview: SPACE start/pause capture | Q/ESC stop and save")

    try:
        while True:
            frames = pipeline.wait_for_frames(5000)
            aligned = align_to_color.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            received += 1
            if received <= args.warmup_frames:
                continue

            now = time.monotonic()
            if capturing and capture_started_at is None:
                capture_started_at = now
            elapsed = (
                capture_elapsed + now - capture_started_at
                if capturing and capture_started_at is not None
                else capture_elapsed
            )
            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_mm = np.rint(
                depth_raw.astype(np.float64) * native_depth_scale_m * 1000.0
            )
            depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(
                np.uint16
            )
            if rectify_maps is not None:
                map_x, map_y = rectify_maps
                color = cv2.remap(
                    color, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                depth_mm = cv2.remap(
                    depth_mm, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )

            if capturing and elapsed + 1e-9 >= next_save_time:
                rgb_path = images_dir / f"{saved:06d}.png"
                depth_path = depth_dir / f"depth_{saved:06d}.png"
                write_rgbd_pair(rgb_path, depth_path, color, depth_mm)
                records.append(
                    {
                        "frame_id": saved,
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_path),
                        "elapsed_s": elapsed,
                        "color_timestamp_ms": float(color_frame.get_timestamp()),
                        "depth_timestamp_ms": float(depth_frame.get_timestamp()),
                        "color_frame_number": int(color_frame.get_frame_number()),
                        "depth_frame_number": int(depth_frame.get_frame_number()),
                        "valid_depth_fraction": float(np.mean(depth_mm > 0)),
                    }
                )
                saved += 1
                next_save_time = elapsed + 1.0 / args.save_fps
                print(f"Saved pair {saved:06d}", end="\r", flush=True)

            if not args.no_preview:
                depth_color = depth_colormap(
                    depth_mm, int(args.preview_max_depth_m * 1000)
                )
                preview = np.hstack((color, depth_color))
                if preview.shape[1] > 1600:
                    scale = 1600.0 / preview.shape[1]
                    preview = cv2.resize(
                        preview, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )
                valid_fraction = float(np.mean(depth_mm > 0)) * 100.0
                preview = add_preview_bar(
                    preview,
                    f"{'CAPTURING' if capturing else 'PAUSED'}    RGB: left    "
                    "Aligned metric depth: right    "
                    f"Saved: {saved}/{args.target_frames}    "
                    f"Valid depth: {valid_fraction:.1f}%",
                    "SPACE: start / pause capture    Q / Esc: finish and save",
                )
                cv2.imshow("RealSense RGB-D recording", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if capturing:
                        capture_elapsed = elapsed
                        capture_started_at = None
                        capturing = False
                    else:
                        capture_started_at = time.monotonic()
                        capturing = True
            if saved >= args.target_frames:
                print(f"\nTarget of {args.target_frames} frames reached")
                break
            if args.duration is not None and capturing and elapsed >= args.duration:
                break
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    manifest = {
        "format": "realsense_rgbd_capture_v1",
        "device_name": device_name,
        "device_serial": serial,
        "width": args.width,
        "height": args.height,
        "native_depth_width": args.depth_width,
        "native_depth_height": args.depth_height,
        "stream_fps": args.fps,
        "saved_fps": args.save_fps,
        "target_frames": args.target_frames,
        "depth_format": "uint16 PNG",
        "depth_unit": "millimetre",
        "depth_aligned_to": "color",
        "images_undistorted": rectify_maps is not None,
        "native_depth_scale_m": native_depth_scale_m,
        "camera_json": str(dataset_camera_path),
        "bag_path": str(args.bag.expanduser().resolve()) if args.bag else None,
        "frames": records,
    }
    manifest_path = dataset / "rgbd_capture.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nSaved {saved} synchronized RGB-D pairs")
    print(f"Capture manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
