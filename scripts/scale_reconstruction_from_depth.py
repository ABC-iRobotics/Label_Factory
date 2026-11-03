#!/usr/bin/env python3
"""Scale an SfM/MVS reconstruction to metres using aligned RGB-D maps."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import yaml
from scipy.spatial.transform import Rotation

from dataset_config import dataset_path


def dataset_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    return dataset_path(config["path"], config["dataset_id"])


def frame_id(value: str | Path) -> int:
    numbers = re.findall(r"\d+", Path(value).stem)
    if not numbers:
        raise ValueError(f"No numeric frame ID in {value}")
    return int(numbers[-1])


def indexed_depth_maps(folder: Path) -> dict[int, Path]:
    result = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        index = frame_id(path)
        if index in result:
            raise ValueError(
                f"Duplicate depth frame {index}: {result[index]} and {path}"
            )
        result[index] = path
    return result


def camera_from_world_matrix(image) -> np.ndarray:
    pose = image.cam_from_world
    if callable(pose):
        pose = pose()
    raw = np.asarray(pose.matrix(), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :] = raw[:3, :]
    return matrix


def projected_depth_ratios(
    world_points: np.ndarray,
    camera_from_world: np.ndarray,
    K: np.ndarray,
    depth_m: np.ndarray,
    minimum_depth_m: float,
    maximum_depth_m: float,
    maximum_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = (
        camera_from_world[:3, :3] @ world_points.T
    ).T + camera_from_world[:3, 3]
    z = camera_points[:, 2]
    positive = np.isfinite(camera_points).all(axis=1) & (z > 1e-10)
    camera_points = camera_points[positive]
    z = z[positive]
    if not len(z):
        return np.empty(0), np.empty(0), np.empty(0)

    projected = (K @ camera_points.T).T
    u = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int64)
    v = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int64)
    height, width = depth_m.shape
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    if not len(z):
        return np.empty(0), np.empty(0), np.empty(0)

    # Keep only the closest reconstructed point at each image pixel.
    pixel = v * width + u
    order = np.lexsort((z, pixel))
    sorted_pixels = pixel[order]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    selected = order[first]
    u, v, z = u[selected], v[selected], z[selected]
    measured = depth_m[v, u]
    valid = (
        np.isfinite(measured)
        & (measured >= minimum_depth_m)
        & (measured <= maximum_depth_m)
    )
    measured, z = measured[valid], z[valid]
    ratios = measured / z
    valid_ratio = np.isfinite(ratios) & (ratios > 0)
    measured, z, ratios = measured[valid_ratio], z[valid_ratio], ratios[valid_ratio]
    if maximum_samples > 0 and len(ratios) > maximum_samples:
        indices = np.linspace(0, len(ratios) - 1, maximum_samples, dtype=int)
        measured, z, ratios = measured[indices], z[indices], ratios[indices]
    return ratios, z, measured


def process_depth_frame(job):
    (index, image_name, depth_path, camera_from_world, estimation_points, K,
     depth_scale_mm, minimum_depth_m, maximum_depth_m,
     maximum_samples_per_frame, minimum_correspondences) = job
    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw_depth is None or raw_depth.ndim != 2:
        return index, image_name, depth_path, None, "invalid depth map"
    depth_m = raw_depth.astype(np.float64) * depth_scale_mm / 1000.0
    values = projected_depth_ratios(
        estimation_points, camera_from_world, K, depth_m,
        minimum_depth_m, maximum_depth_m, maximum_samples_per_frame,
    )
    if len(values[0]) < minimum_correspondences:
        return index, image_name, depth_path, None, (
            f"only {len(values[0])} depth correspondences"
        )
    return index, image_name, depth_path, values, None


def robust_scale(frame_results: list[dict], mad_multiplier: float):
    frame_medians = np.asarray(
        [result["raw_median_scale"] for result in frame_results],
        dtype=np.float64,
    )
    center = float(np.median(frame_medians))
    mad = float(np.median(np.abs(frame_medians - center)))
    robust_sigma = 1.4826 * mad
    if robust_sigma > 0:
        accepted = np.abs(frame_medians - center) <= mad_multiplier * robust_sigma
    else:
        accepted = np.isclose(frame_medians, center, rtol=1e-6, atol=1e-12)
    if not np.any(accepted):
        raise RuntimeError("Metric scale estimation rejected every frame")
    scale = float(np.median(frame_medians[accepted]))
    return scale, accepted, center, robust_sigma


def write_metric_camera_poses(
    reconstruction,
    scale: float,
    alignment_rotation: np.ndarray,
    alignment_translation: np.ndarray,
    output_path: Path,
) -> list[dict]:
    """Transform COLMAP camera poses with the same Sim(3) as the cloud."""
    poses = []
    for image in sorted(
        reconstruction.images.values(), key=lambda item: item.name
    ):
        camera_from_world = camera_from_world_matrix(image)
        world_from_camera = np.linalg.inv(camera_from_world)
        rotation_wc = world_from_camera[:3, :3]
        center = world_from_camera[:3, 3]

        metric_rotation_wc = alignment_rotation @ rotation_wc
        metric_center = (
            scale * (alignment_rotation @ center) + alignment_translation
        )
        metric_world_from_camera = np.eye(4, dtype=np.float64)
        metric_world_from_camera[:3, :3] = metric_rotation_wc
        metric_world_from_camera[:3, 3] = metric_center
        metric_camera_from_world = np.linalg.inv(metric_world_from_camera)
        quaternion = Rotation.from_matrix(metric_rotation_wc).as_quat()
        poses.append(
            {
                "frame_id": frame_id(image.name),
                "image_name": image.name,
                "coordinate_system": "scaled_colmap_world",
                "translation_unit": "metre",
                "world_from_camera": metric_world_from_camera.tolist(),
                "camera_from_world": metric_camera_from_world.tolist(),
                "camera_center_m": metric_center.tolist(),
                "rotation_world_from_camera": metric_rotation_wc.tolist(),
                "quaternion_world_from_camera_xyzw": quaternion.tolist(),
            }
        )
    output_path.write_text(
        json.dumps(
            {
                "format": "metric_camera_poses_v1",
                "translation_unit": "metre",
                "scale_metres_per_reconstruction_unit": scale,
                "alignment_rotation": alignment_rotation.tolist(),
                "alignment_translation_m": alignment_translation.tolist(),
                "poses": poses,
            },
            indent=2,
        )
        + "\n"
    )
    return poses


def scale_reconstruction(
    cloud_path: Path,
    camera_model_path: Path,
    depth_dir: Path,
    camera_json_path: Path,
    output_cloud_path: Path,
    alignment_dir: Path,
    maximum_points: int = 1_000_000,
    maximum_samples_per_frame: int = 20_000,
    minimum_correspondences: int = 100,
    minimum_frames: int = 3,
    minimum_depth_m: float = 0.05,
    maximum_depth_m: float = 10.0,
    mad_multiplier: float = 3.5,
    workers: int = 1,
) -> dict:
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty: {cloud_path}")
    points = np.asarray(cloud.points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if maximum_points > 0 and len(points) > maximum_points:
        indices = np.linspace(0, len(points) - 1, maximum_points, dtype=int)
        estimation_points = points[indices]
    else:
        estimation_points = points

    camera_data = json.loads(camera_json_path.read_text())
    K = np.asarray(camera_data["cam_K"], dtype=np.float64).reshape(3, 3)
    depth_scale_mm = float(camera_data.get("depth_scale", 1.0))
    depths = indexed_depth_maps(depth_dir)
    reconstruction = pycolmap.Reconstruction(str(camera_model_path))
    images = sorted(reconstruction.images.values(), key=lambda image: image.name)

    jobs = []
    for image in images:
        index = frame_id(image.name)
        depth_path = depths.get(index)
        if depth_path is None:
            print(f"Warning: no depth map for {image.name}")
            continue
        jobs.append((
            index, image.name, depth_path, camera_from_world_matrix(image),
            estimation_points, K, depth_scale_mm, minimum_depth_m,
            maximum_depth_m, maximum_samples_per_frame,
            minimum_correspondences,
        ))

    frame_results = []
    raw_observations = []
    print(f"Metric depth comparison workers: {workers}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, image_name, depth_path, values, warning in executor.map(
            process_depth_frame, jobs
        ):
            if warning:
                print(f"Warning: frame {index}: {warning} ({depth_path})")
                continue
            ratios, reconstructed_z, measured_z = values
            median = float(np.median(ratios))
            frame_results.append({
                "frame_id": index,
                "image_name": image_name,
                "depth_path": str(depth_path),
                "correspondences": int(len(ratios)),
                "raw_median_scale": median,
            })
            raw_observations.append((reconstructed_z, measured_z))
            print(f"Frame {index}: {len(ratios)} pairs, scale={median:.9g}")

    if len(frame_results) < minimum_frames:
        raise RuntimeError(
            f"Only {len(frame_results)} usable RGB-D frames; "
            f"at least {minimum_frames} required"
        )
    scale, accepted, raw_center, frame_sigma = robust_scale(
        frame_results, mad_multiplier
    )
    accepted_errors = []
    for result, is_accepted, observation in zip(
        frame_results, accepted, raw_observations
    ):
        reconstructed_z, measured_z = observation
        errors = np.abs(scale * reconstructed_z - measured_z)
        result["accepted"] = bool(is_accepted)
        result["median_absolute_depth_error_m"] = float(np.median(errors))
        if is_accepted:
            accepted_errors.append(errors)

    scaled = o3d.io.read_point_cloud(str(cloud_path))
    scaled.points = o3d.utility.Vector3dVector(
        np.asarray(scaled.points, dtype=np.float64) * scale
    )
    output_cloud_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_cloud_path), scaled):
        raise RuntimeError(f"Failed to write {output_cloud_path}")

    alignment_dir.mkdir(parents=True, exist_ok=True)
    alignment_rotation = np.eye(3, dtype=np.float64)
    alignment_translation = np.zeros(3, dtype=np.float64)
    np.save(alignment_dir / "scale.npy", np.asarray(scale))
    np.save(alignment_dir / "rotation.npy", alignment_rotation)
    np.save(alignment_dir / "translation.npy", alignment_translation)
    camera_poses_path = alignment_dir / "camera_poses_metric.json"
    metric_camera_poses = write_metric_camera_poses(
        reconstruction,
        scale,
        alignment_rotation,
        alignment_translation,
        camera_poses_path,
    )
    all_errors = np.concatenate(accepted_errors)
    report = {
        "format": "depth_metric_scale_v1",
        "scale_metres_per_reconstruction_unit": scale,
        "rotation": np.eye(3).tolist(),
        "translation_m": [0.0, 0.0, 0.0],
        "note": (
            "Depth establishes global scale. Reconstruction origin and "
            "orientation are retained."
        ),
        "input_cloud": str(cloud_path),
        "scaled_cloud": str(output_cloud_path),
        "camera_model": str(camera_model_path),
        "camera_json": str(camera_json_path),
        "metric_camera_poses": str(camera_poses_path),
        "number_of_metric_camera_poses": len(metric_camera_poses),
        "depth_maps": str(depth_dir),
        "estimation_points": int(len(estimation_points)),
        "usable_frames": len(frame_results),
        "accepted_frames": int(np.sum(accepted)),
        "raw_frame_scale_median": raw_center,
        "raw_frame_scale_robust_sigma": frame_sigma,
        "median_absolute_depth_error_m": float(np.median(all_errors)),
        "frames": frame_results,
    }
    (alignment_dir / "depth_scale_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=project / "scripts/configs/default.yaml",
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--reconstruction-dir", type=Path)
    parser.add_argument("--cloud", type=Path)
    parser.add_argument("--camera-model", type=Path)
    parser.add_argument("--depth-dir", type=Path)
    parser.add_argument("--camera-json", type=Path)
    parser.add_argument("--output-cloud", type=Path)
    parser.add_argument("--alignment-dir", type=Path)
    parser.add_argument("--maximum-points", type=int, default=1_000_000)
    parser.add_argument("--samples-per-frame", type=int, default=20_000)
    parser.add_argument("--minimum-correspondences", type=int, default=100)
    parser.add_argument("--minimum-frames", type=int, default=3)
    parser.add_argument("--minimum-depth-m", type=float, default=0.05)
    parser.add_argument("--maximum-depth-m", type=float, default=10.0)
    parser.add_argument("--mad-multiplier", type=float, default=3.5)
    parser.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("METRIC_SCALE_WORKERS", "4")),
        help="Parallel RGB-D frame comparison workers (default: 4).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    config_path = args.config.expanduser().resolve()
    dataset = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir else dataset_from_config(config_path)
    )
    reconstruction_dir = (
        args.reconstruction_dir.expanduser().resolve()
        if args.reconstruction_dir else dataset / "reconstruction"
    )
    cloud = args.cloud.expanduser().resolve() if args.cloud else reconstruction_dir / "mvs_filtered.ply"
    camera_model = (
        args.camera_model.expanduser().resolve()
        if args.camera_model else reconstruction_dir / "sfm_filtered"
    )
    depth_dir = args.depth_dir.expanduser().resolve() if args.depth_dir else dataset / "depth_maps"
    camera_json = args.camera_json.expanduser().resolve() if args.camera_json else dataset / "camera.json"
    output_cloud = (
        args.output_cloud.expanduser().resolve()
        if args.output_cloud else reconstruction_dir / "mvs_filtered_metric.ply"
    )
    alignment_dir = (
        args.alignment_dir.expanduser().resolve()
        if args.alignment_dir else reconstruction_dir / "metric_alignment"
    )
    for required in (cloud, camera_model, depth_dir, camera_json):
        if not required.exists():
            raise FileNotFoundError(required)
    report = scale_reconstruction(
        cloud,
        camera_model,
        depth_dir,
        camera_json,
        output_cloud,
        alignment_dir,
        args.maximum_points,
        args.samples_per_frame,
        args.minimum_correspondences,
        args.minimum_frames,
        args.minimum_depth_m,
        args.maximum_depth_m,
        args.mad_multiplier,
        args.workers,
    )
    print(
        "\nMetric scale: "
        f"{report['scale_metres_per_reconstruction_unit']:.12g} m/unit"
    )
    print(
        "Median absolute depth error: "
        f"{report['median_absolute_depth_error_m'] * 1000:.3f} mm"
    )
    print(f"Scaled cloud: {output_cloud}")
    print(f"Alignment: {alignment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
