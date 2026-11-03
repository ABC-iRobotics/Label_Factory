#!/usr/bin/env python3
"""Select RGB-D frames for SAM-6D independently of reconstruction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import stage_timing
from dataset_config import dataset_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def frame_id(path: Path) -> int:
    digits = "".join(filter(str.isdigit, path.stem))
    if not digits:
        raise ValueError(f"Image filename has no numeric frame ID: {path.name}")
    return int(digits)


def overlap_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 480.0 / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    return gray


def estimate_overlap(previous_gray: np.ndarray, image: np.ndarray) -> float:
    current_gray = overlap_gray(image)
    points = cv2.goodFeaturesToTrack(previous_gray, 500, 0.01, 8)
    if points is None or len(points) < 12:
        return 0.0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None, winSize=(21, 21), maxLevel=3
    )
    valid = status.reshape(-1).astype(bool)
    if valid.sum() < 12:
        return 0.0
    transform, inliers = cv2.estimateAffinePartial2D(
        points[valid], tracked[valid], method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if transform is None or inliers is None or int(inliers.sum()) < 8:
        return 0.0
    height, width = current_gray.shape
    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    warped = cv2.transform(corners, transform).reshape(-1, 2)
    intersection, _ = cv2.intersectConvexConvex(
        warped.astype(np.float32),
        np.float32([[0, 0], [width, 0], [width, height], [0, height]]),
    )
    return float(np.clip(intersection / float(width * height), 0.0, 1.0))


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--target-overlap", type=float, default=0.7)
    args = parser.parse_args()
    if not 0.0 <= args.target_overlap <= 1.0:
        raise ValueError("--target-overlap must be between 0 and 1")

    config = yaml.safe_load(args.config.expanduser().resolve().read_text())
    dataset = (args.dataset_dir.expanduser().resolve() if args.dataset_dir else
               dataset_path(config["path"], config["dataset_id"]))
    started = time.monotonic()
    rgb_dir = dataset / "images"
    depth_dir = dataset / "depth_maps"
    rgb = {frame_id(path): path for path in sorted(rgb_dir.iterdir())
           if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    depth_ids = {frame_id(path) for path in depth_dir.iterdir()
                 if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    image_paths = [rgb[index] for index in sorted(set(rgb) & depth_ids)]
    if not image_paths:
        raise RuntimeError("No matching RGB/depth frame IDs were found")

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read {image_paths[0]}")
    selected = [image_paths[0]]
    previous_gray = overlap_gray(first)
    observations = [{"image_name": image_paths[0].name, "overlap": None,
                     "selected": True}]
    print(f"Selecting SAM-6D frames at {args.target_overlap:.0%} target overlap")
    for position, path in enumerate(image_paths[1:], 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {path}")
        overlap = estimate_overlap(previous_gray, image)
        accepted = overlap <= args.target_overlap
        observations.append({"image_name": path.name, "overlap": overlap,
                             "selected": accepted})
        if accepted:
            selected.append(path)
            previous_gray = overlap_gray(image)
        print(f"Pairing [{position + 1}/{len(image_paths)}]", end="\r", flush=True)
    if len(image_paths) > 1 and selected[-1] != image_paths[-1]:
        selected.append(image_paths[-1])
        observations[-1]["selected"] = True

    output = dataset / "sam6d_frame_selection.json"
    output.write_text(json.dumps({
        "format": "label_factory_sam6d_selection_v1",
        "target_overlap": args.target_overlap,
        "total_frames": len(image_paths),
        "selected_frames": len(selected),
        "frame_ids": [frame_id(path) for path in selected],
        "image_names": [path.name for path in selected],
        "frames": observations,
    }, indent=2) + "\n")
    stage_timing.record(dataset, "frame_selection_sam6d", time.monotonic() - started,
                        input_frames=len(image_paths), selected_frames=len(selected))
    print(f"\nSelected {len(selected)}/{len(image_paths)} SAM-6D frames")
    print(f"Selection written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
