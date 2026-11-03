#!/usr/bin/env python3
"""Calibrate a USB RGB camera with a ChArUco board."""

from __future__ import annotations

import argparse
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
        "pyrealsense2 is required because camera calibration uses a RealSense "
        "color stream. Install it with `python -m pip install pyrealsense2`."
    ) from error


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def dictionary_names() -> list[str]:
    return sorted(
        name
        for name in dir(cv2.aruco)
        if name.startswith("DICT_") and isinstance(getattr(cv2.aruco, name), int)
    )


def make_board(args: argparse.Namespace):
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dictionary)
    )
    size = (args.squares_x, args.squares_y)
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            size, args.square_length, args.marker_length, dictionary
        )
    else:
        board = cv2.aruco.CharucoBoard_create(
            *size, args.square_length, args.marker_length, dictionary
        )
    return dictionary, board


def make_detector(dictionary, board):
    if hasattr(cv2.aruco, "CharucoDetector"):
        return cv2.aruco.CharucoDetector(board)
    return None


def normalize_charuco_observation(corners, ids):
    """Return matched OpenCV-shaped ChArUco corners and IDs.

    Some OpenCV 5 builds occasionally return differently shaped or partially
    populated arrays from ``CharucoDetector.detectBoard``.  Passing those
    arrays directly to ``drawDetectedCornersCharuco`` triggers an assertion
    before the frame can be displayed.
    """
    if corners is None or ids is None:
        return None, None

    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    count = min(len(corners), len(ids))
    if count == 0:
        return None, None

    corners = corners[:count]
    ids = ids[:count]
    valid = np.isfinite(corners).all(axis=1) & (ids >= 0)
    corners = corners[valid]
    ids = ids[valid]
    if len(ids) == 0:
        return None, None

    # These are the layouts expected by the ArUco drawing and calibration APIs.
    return corners.reshape(-1, 1, 2), ids.reshape(-1, 1)


def detect_charuco(gray: np.ndarray, dictionary, board, detector):
    if detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            detector.detectBoard(gray)
        )
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary
        )
        if marker_ids is None:
            return None, None, marker_corners, marker_ids
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
    charuco_corners, charuco_ids = normalize_charuco_observation(
        charuco_corners, charuco_ids
    )
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def draw_detection(image, corners, ids, marker_corners, marker_ids,
                   maximum_markers: int):
    """Draw a bounded preview without discarding calibration observations."""
    display = image.copy()
    if marker_ids is not None and len(marker_ids):
        displayed_count = min(len(marker_ids), maximum_markers)
        cv2.aruco.drawDetectedMarkers(
            display, marker_corners[:displayed_count], marker_ids[:displayed_count]
        )
    if corners is not None and ids is not None and len(ids):
        displayed_count = min(len(ids), maximum_markers)
        cv2.aruco.drawDetectedCornersCharuco(
            display, corners[:displayed_count], ids[:displayed_count]
        )
    return display


def add_calibration_help(display: np.ndarray) -> np.ndarray:
    result = cv2.copyMakeBorder(
        display, 0, 54, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 29)
    )
    cv2.putText(
        result,
        "SPACE: accept view    A: automatic capture    Q / Esc: finish",
        (15, display.shape[0] + 35), cv2.FONT_HERSHEY_SIMPLEX,
        0.68, (240, 240, 240), 2, cv2.LINE_AA,
    )
    return result


def board_points(board, corners, ids):
    if hasattr(board, "matchImagePoints"):
        object_points, image_points = board.matchImagePoints(corners, ids)
        return (
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
        )
    chessboard = np.asarray(board.chessboardCorners, dtype=np.float32)
    indices = np.asarray(ids, dtype=np.int32).reshape(-1)
    return chessboard[indices].reshape(-1, 1, 3), np.asarray(
        corners, dtype=np.float32
    ).reshape(-1, 1, 2)


def capture_images(args, dictionary, board, detector, capture_dir: Path):
    pipeline = rs.pipeline()
    stream_config = rs.config()
    if args.serial:
        stream_config.enable_device(args.serial)
    stream_config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    try:
        profile = pipeline.start(stream_config)
    except RuntimeError as error:
        selected = f" serial {args.serial}" if args.serial else ""
        raise RuntimeError(
            f"Could not start RealSense{selected} color stream at "
            f"{args.width}x{args.height}@{args.fps}. Select a supported preset "
            "in the Capture tab and ensure no other application uses the camera."
        ) from error
    device = profile.get_device()
    device_name = device.get_info(rs.camera_info.name)
    device_serial = device.get_info(rs.camera_info.serial_number)
    print(f"Calibrating {device_name} ({device_serial})")
    capture_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    last_capture = 0.0
    print("SPACE: accept view | A: toggle automatic capture | Q/ESC: finish")
    automatic = False
    try:
        while True:
            frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, marker_corners, marker_ids = detect_charuco(
                gray, dictionary, board, detector
            )
            count = 0 if ids is None else len(ids)
            display = draw_detection(
                image, corners, ids, marker_corners, marker_ids,
                args.max_displayed_markers,
            )
            marker_count = 0 if marker_ids is None else len(marker_ids)
            displayed_markers = min(marker_count, args.max_displayed_markers)
            displayed_corners = min(count, args.max_displayed_markers)
            status = (
                f"views {len(accepted)}/{args.views} | "
                f"corners shown {displayed_corners}/{count} | "
                f"markers {displayed_markers}/{marker_count} | "
                f"auto {'ON' if automatic else 'OFF'}"
            )
            cv2.putText(
                display, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0) if count >= args.minimum_corners else (0, 0, 255),
                2, cv2.LINE_AA,
            )
            display = add_calibration_help(display)
            cv2.imshow("ChArUco calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                automatic = not automatic
            enough = count >= args.minimum_corners
            requested = key == ord(" ")
            timed = automatic and time.monotonic() - last_capture >= args.interval
            if enough and (requested or timed):
                path = capture_dir / f"view_{len(accepted):03d}.png"
                cv2.imwrite(str(path), image)
                accepted.append(path)
                last_capture = time.monotonic()
                print(f"Accepted {path.name}: {count} corners")
                if len(accepted) >= args.views:
                    break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return accepted


def load_observations(paths, dictionary, board, detector, minimum_corners):
    object_points = []
    image_points = []
    used_paths = []
    image_size = None
    corner_counts = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: could not read {path}")
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(
                f"Calibration images have mixed sizes: {path} is {size}, "
                f"expected {image_size}"
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detect_charuco(gray, dictionary, board, detector)
        if ids is None or len(ids) < minimum_corners:
            print(f"Skipping {path.name}: insufficient ChArUco corners")
            continue
        obj, img = board_points(board, corners, ids)
        object_points.append(obj)
        image_points.append(img)
        used_paths.append(path)
        corner_counts.append(len(ids))
    if image_size is None:
        raise RuntimeError("No readable calibration images were supplied")
    return object_points, image_points, used_paths, image_size, corner_counts


def reprojection_errors(object_points, image_points, rvecs, tvecs, K, dist):
    errors = []
    for obj, observed, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        residual = projected.reshape(-1, 2) - observed.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    return errors


def write_outputs(args, result, config_path: Path | None):
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "charuco_calibration.json"
    calibration_path.write_text(json.dumps(result, indent=2) + "\n")

    K = result["camera_matrix"]
    camera_json = {
        "cam_K": [value for row in K for value in row],
        "distortion": result["distortion"],
        "distortion_names": result["distortion_names"],
        "image_width": result["image_width"],
        "image_height": result["image_height"],
        # SAM-6D expects uint16 depth PNG values in millimetres in this project.
        "depth_scale": 1.0,
    }
    (output_dir / "camera.json").write_text(
        json.dumps(camera_json, indent=2) + "\n"
    )

    if args.activate_dataset:
        if config_path is None:
            raise ValueError("--activate-dataset requires --config")
        config = yaml.safe_load(config_path.read_text()) or {}
        active_camera = dataset_path(
            config["path"], config["dataset_id"]
        ) / "camera.json"
        active_camera.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_dir / "camera.json", active_camera)
        print(f"Activated calibration for dataset: {active_camera}")

    if args.update_config:
        if config_path is None:
            raise ValueError("--update-config requires --config")
        config = yaml.safe_load(config_path.read_text())
        camera = config.setdefault("camera", {})
        camera.update(
            {
                "fx": K[0][0], "fy": K[1][1],
                "cx": K[0][2], "cy": K[1][2],
                "k1": result["distortion"][0],
                "k2": result["distortion"][1],
                "p1": result["distortion"][2],
                "p2": result["distortion"][3],
                "k3": result["distortion"][4],
            }
        )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Updated {config_path}")
    print(f"Saved calibration to {calibration_path}")
    print(f"Saved SAM-6D intrinsics to {output_dir / 'camera.json'}")


def parse_args():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="RealSense device serial number")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--views", type=int, default=25)
    parser.add_argument("--minimum-views", type=int, default=10)
    parser.add_argument("--minimum-corners", type=int, default=12)
    parser.add_argument(
        "--max-displayed-markers", type=int, default=30,
        help="Maximum detected ArUco markers drawn in each live preview frame.",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-length", type=float, default=0.030,
                        help="Printed chessboard square side in metres")
    parser.add_argument("--marker-length", type=float, default=0.023,
                        help="Printed ArUco marker side in metres")
    parser.add_argument("--dictionary", choices=dictionary_names(),
                        default="DICT_5X5_100")
    parser.add_argument("--input-dir", type=Path,
                        help="Calibrate from existing images instead of a camera")
    parser.add_argument("--generate-board", type=Path,
                        help="Generate a printable board PNG and exit")
    parser.add_argument("--board-width-px", type=int, default=2100)
    parser.add_argument("--margin-px", type=int, default=60)
    parser.add_argument("--output-dir", type=Path,
                        default=project / "calibration/charuco")
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--update-config", action="store_true")
    parser.add_argument(
        "--activate-dataset", action="store_true",
        help="Copy camera.json into the dataset selected by --config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_displayed_markers <= 0:
        raise ValueError("--max-displayed-markers must be positive")
    if args.marker_length >= args.square_length:
        raise ValueError("--marker-length must be smaller than --square-length")
    dictionary, board = make_board(args)
    if args.generate_board:
        aspect = (args.squares_y * args.square_length) / (
            args.squares_x * args.square_length
        )
        size = (args.board_width_px, int(round(args.board_width_px * aspect)))
        image = board.generateImage(size, marginSize=args.margin_px)
        destination = args.generate_board.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), image)
        print(f"Saved board to {destination}")
        print(
            f"Print with {args.squares_x} x {args.squares_y} squares; "
            f"each square must measure {args.square_length * 1000:.3f} mm."
        )
        return 0

    detector = make_detector(dictionary, board)
    capture_dir = args.output_dir.expanduser().resolve() / "captures"
    if args.input_dir:
        paths = sorted(
            path for path in args.input_dir.expanduser().resolve().iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        paths = capture_images(args, dictionary, board, detector, capture_dir)

    obj, img, used, image_size, counts = load_observations(
        paths, dictionary, board, detector, args.minimum_corners
    )
    if len(used) < args.minimum_views:
        raise RuntimeError(
            f"Only {len(used)} valid views; at least {args.minimum_views} required"
        )
    rms, K, distortion, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, image_size, None, None
    )
    per_view = reprojection_errors(obj, img, rvecs, tvecs, K, distortion)
    coefficients = np.zeros(5, dtype=float)
    flat_distortion = np.asarray(distortion).reshape(-1)
    coefficients[: min(5, len(flat_distortion))] = flat_distortion[:5]
    result = {
        "calibration_model": "OpenCV Brown-Conrady",
        "calibrated_stream": "RGB",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "rms_reprojection_error_px": float(rms),
        "mean_view_error_px": float(np.mean(per_view)),
        "camera_matrix": K.tolist(),
        "distortion": coefficients.tolist(),
        "distortion_names": ["k1", "k2", "p1", "p2", "k3"],
        "board": {
            "dictionary": args.dictionary,
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length_m": args.square_length,
            "marker_length_m": args.marker_length,
        },
        "views": [
            {"path": str(path), "corners": count, "error_px": error}
            for path, count, error in zip(used, counts, per_view)
        ],
        "depth_calibration": {
            "included": False,
            "note": "Depth scale and RGB-to-depth extrinsics require the camera SDK or a separate calibration.",
        },
    }
    write_outputs(args, result, args.config.expanduser().resolve())
    print(f"RMS reprojection error: {rms:.4f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
