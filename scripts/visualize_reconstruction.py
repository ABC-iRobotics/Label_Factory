#!/usr/bin/env python3
"""Standalone interactive viewer for SfM and MVS reconstruction stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from dataset_config import dataset_path
from open3d_gui_window import HelpSceneWindow

try:
    import pycolmap
except ImportError as error:
    raise RuntimeError(
        "pycolmap is required to display reconstructed cameras. Run this "
        "script in the reconstruction environment."
    ) from error


STAGES = (
    ("sfm", "sfm.ply", (0.95, 0.25, 0.20)),
    ("sfm_filtered", "sfm_filtered.ply", (1.00, 0.72, 0.10)),
    ("mvs", "mvs.ply", (0.15, 0.75, 1.00)),
    ("mvs_filtered", "mvs_filtered.ply", (0.20, 0.95, 0.35)),
    ("mvs_metric", "mvs_filtered_metric.ply", (0.95, 0.20, 0.85)),
)


def reconstruction_from_config(config_path: Path) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    if "path" not in config or "dataset_id" not in config:
        raise ValueError(
            f"{config_path} must contain both 'path' and 'dataset_id'"
        )
    return dataset_path(config["path"], config["dataset_id"]) / "reconstruction"


def load_cloud(
    path: Path,
    maximum_points: int,
    preserve_colors: bool,
    display_color: tuple[float, float, float],
) -> tuple[o3d.geometry.PointCloud, np.ndarray | None]:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty: {path}")
    cloud.remove_non_finite_points()
    cloud.remove_duplicated_points()
    if maximum_points > 0 and len(cloud.points) > maximum_points:
        indices = np.linspace(
            0, len(cloud.points) - 1, maximum_points, dtype=np.int64
        )
        cloud = cloud.select_by_index(indices.tolist())
    original_colors = (
        np.asarray(cloud.colors).copy() if cloud.has_colors() else None
    )
    if not preserve_colors or original_colors is None:
        cloud.paint_uniform_color(display_color)
    return cloud, original_colors


def camera_from_world_matrix(image) -> np.ndarray:
    pose = image.cam_from_world
    if callable(pose):
        pose = pose()
    raw = np.asarray(pose.matrix(), dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :] = raw[:3, :]
    return result


def camera_frustums(
    model_path: Path,
    depth: float,
    color: tuple[float, float, float] = (0.8, 0.2, 1.0),
) -> tuple[o3d.geometry.LineSet, np.ndarray]:
    reconstruction = pycolmap.Reconstruction(str(model_path))
    points = []
    lines = []
    line_colors = []
    centers = []
    for image in sorted(
        reconstruction.images.values(), key=lambda item: item.name
    ):
        camera_from_world = camera_from_world_matrix(image)
        world_from_camera = np.linalg.inv(camera_from_world)
        center = world_from_camera[:3, 3]
        rotation = world_from_camera[:3, :3]
        centers.append(center)

        # OpenCV camera coordinates: +X right, +Y down, +Z forward.
        half_width = depth * 0.65
        half_height = depth * 0.42
        local = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [-half_width, -half_height, depth],
                [half_width, -half_height, depth],
                [half_width, half_height, depth],
                [-half_width, half_height, depth],
            ]
        )
        world = (rotation @ local.T).T + center
        offset = len(points)
        points.extend(world.tolist())
        local_lines = (
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (2, 3), (3, 4), (4, 1),
        )
        lines.extend((offset + first, offset + second) for first, second in local_lines)
        line_colors.extend([color] * len(local_lines))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors))
    return line_set, np.asarray(centers)


def metric_camera_frustums(
    poses_path: Path,
    depth: float,
    color: tuple[float, float, float] = (1.0, 0.35, 0.85),
) -> tuple[o3d.geometry.LineSet, np.ndarray]:
    data = json.loads(poses_path.read_text())
    points = []
    lines = []
    line_colors = []
    centers = []
    for pose in data.get("poses", []):
        world_from_camera = np.asarray(
            pose["world_from_camera"], dtype=np.float64
        )
        center = world_from_camera[:3, 3]
        rotation = world_from_camera[:3, :3]
        centers.append(center)
        half_width = depth * 0.65
        half_height = depth * 0.42
        local = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [-half_width, -half_height, depth],
                [half_width, -half_height, depth],
                [half_width, half_height, depth],
                [-half_width, half_height, depth],
            ]
        )
        world = (rotation @ local.T).T + center
        offset = len(points)
        points.extend(world.tolist())
        local_lines = (
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (2, 3), (3, 4), (4, 1),
        )
        lines.extend((offset + first, offset + second) for first, second in local_lines)
        line_colors.extend([color] * len(local_lines))
    if not centers:
        raise RuntimeError(f"No metric camera poses found in {poses_path}")
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors))
    return line_set, np.asarray(centers)


def print_cloud_summary(name: str, path: Path, cloud) -> None:
    bounds = cloud.get_axis_aligned_bounding_box()
    print(f"{name:14s}: {len(cloud.points):>10,d} displayed points | {path}")
    print(f"{'':16s} min={np.asarray(bounds.min_bound)}")
    print(f"{'':16s} max={np.asarray(bounds.max_bound)}")
    print(f"{'':16s} extent={np.asarray(bounds.get_extent())}")


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reconstruction_dir",
        type=Path,
        nargs="?",
        help=(
            "Optional reconstruction directory override. By default it is "
            "resolved as <config path>/data/<dataset_id>/reconstruction."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project / "scripts/configs/default.yaml",
        help="YAML file containing the project path and dataset_id.",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "sfm", "sfm_filtered", "mvs", "mvs_filtered", "mvs_metric"
        ),
        default="mvs_metric",
    )
    parser.add_argument(
        "--camera-model",
        type=Path,
        help="Defaults to <reconstruction>/sfm_filtered, then sfm.",
    )
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--preserve-colors", action="store_true")
    parser.add_argument(
        "--max-points",
        type=int,
        default=2_000_000,
        help="Maximum displayed points per cloud; zero keeps every point.",
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--reference-frame-scale", type=float, default=0.005,
        help="Reference-axis length as a fraction of scene diagonal (default: 0.005).",
    )
    parser.add_argument(
        "--frustum-scale",
        type=float,
        help="Camera-frustum depth; defaults to 3%% of cloud diagonal.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=Path("reconstruction_view.png"),
        help="Path used when S is pressed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reference_frame_scale <= 0:
        raise ValueError("--reference-frame-scale must be positive")
    config_path = args.config.expanduser().resolve()
    root = (
        args.reconstruction_dir.expanduser().resolve()
        if args.reconstruction_dir is not None
        else reconstruction_from_config(config_path)
    )
    print(f"Reconstruction directory: {root}")
    if args.reconstruction_dir is None:
        print(f"Resolved from config: {config_path}")
    if not root.is_dir():
        raise FileNotFoundError(
            f"Reconstruction directory not found: {root}. Run "
            "reconstruct_rgbd_scene.py for the configured dataset first."
        )

    clouds = {}
    original_colors = {}
    stage_colors = {}
    for name, filename, color in STAGES:
        path = root / filename
        if not path.is_file():
            print(f"Missing stage, skipping: {path}")
            continue
        cloud, source_colors = load_cloud(
            path, args.max_points, args.preserve_colors, color
        )
        clouds[name] = cloud
        original_colors[name] = source_colors
        stage_colors[name] = np.asarray(color, dtype=np.float64)
        print_cloud_summary(name, path, cloud)
    if not clouds:
        raise RuntimeError(f"No reconstruction PLY files found in {root}")

    combined_bounds = o3d.geometry.AxisAlignedBoundingBox()
    all_points = np.vstack(
        [np.asarray(cloud.points) for cloud in clouds.values()]
    )
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    diagonal = float(np.linalg.norm(maximum - minimum))
    axis_size = max(diagonal * args.reference_frame_scale, 1e-6)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)

    camera_geometries = {}
    if not args.no_cameras:
        model_path = (
            args.camera_model.expanduser().resolve()
            if args.camera_model
            else root / "sfm_filtered"
        )
        if not model_path.is_dir():
            fallback = root / "sfm"
            model_path = fallback if fallback.is_dir() else model_path
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"No COLMAP camera model found at {model_path}"
            )
        unscaled_cloud_points = [
            np.asarray(cloud.points)
            for name, cloud in clouds.items()
            if name != "mvs_metric"
        ]
        unscaled_points = np.vstack(
            unscaled_cloud_points
            if unscaled_cloud_points
            else [np.asarray(next(iter(clouds.values())).points)]
        )
        unscaled_diagonal = float(
            np.linalg.norm(unscaled_points.max(axis=0) - unscaled_points.min(axis=0))
        )
        unscaled_depth = args.frustum_scale or max(unscaled_diagonal * 0.03, 1e-6)
        unscaled_geometry, centers = camera_frustums(
            model_path, unscaled_depth
        )
        camera_geometries["unscaled"] = unscaled_geometry
        print(f"SfM cameras   : {len(centers)} from {model_path}")

        metric_poses_path = root / "metric_alignment/camera_poses_metric.json"
        if "mvs_metric" in clouds and metric_poses_path.is_file():
            metric_points = np.asarray(clouds["mvs_metric"].points)
            metric_diagonal = float(
                np.linalg.norm(
                    metric_points.max(axis=0) - metric_points.min(axis=0)
                )
            )
            metric_depth = (
                args.frustum_scale
                if args.frustum_scale is not None
                else max(metric_diagonal * 0.03, 1e-6)
            )
            metric_geometry, metric_centers = metric_camera_frustums(
                metric_poses_path, metric_depth
            )
            camera_geometries["metric"] = metric_geometry
            print(
                f"Metric cameras: {len(metric_centers)} from "
                f"{metric_poses_path}"
            )

    controls = (
        "Left drag: rotate   Right drag: pan   Wheel: zoom   "
        "1: SfM   2: filtered SfM   3: MVS   4: filtered MVS   "
        "5: metric MVS   C: cameras   F: frame   P: colors   "
        "R: fit view   S: screenshot   Q/Esc: close"
    )
    visualizer = HelpSceneWindow(
        "Reconstruction stages", controls, width=1440, height=900
    )
    point_material = visualizer.point_material(args.point_size)
    line_material = visualizer.line_material(1.5)
    mesh_material = visualizer.mesh_material()

    visible = {name: False for name in clouds}
    camera_visible = False
    shown_camera_key = None
    frame_visible = True
    use_original_colors = args.preserve_colors
    visualizer.add("reference_frame", frame, mesh_material)

    def visible_bounds() -> o3d.geometry.AxisAlignedBoundingBox:
        selected_points = [
            np.asarray(clouds[name].points)
            for name, is_visible in visible.items() if is_visible
        ]
        points = np.vstack(selected_points) if selected_points else all_points
        return o3d.geometry.AxisAlignedBoundingBox(
            points.min(axis=0), points.max(axis=0)
        )

    def fit_visible_view() -> None:
        visualizer.fit_view(visible_bounds())

    def desired_camera_key():
        if visible.get("mvs_metric", False) and "metric" in camera_geometries:
            return "metric"
        if "unscaled" in camera_geometries:
            return "unscaled"
        return None

    def refresh_camera_geometry():
        nonlocal shown_camera_key
        if not camera_visible:
            return
        desired = desired_camera_key()
        if desired == shown_camera_key:
            return
        if shown_camera_key is not None:
            visualizer.remove("cameras")
        shown_camera_key = desired
        if shown_camera_key is not None:
            visualizer.add("cameras", camera_geometries[shown_camera_key], line_material)
            print(
                "Camera coordinate system: "
                + ("metric" if shown_camera_key == "metric" else "unscaled SfM")
            )

    def show_clouds(selected: set[str]):
        for name, geometry in clouds.items():
            should_show = name in selected
            if should_show and not visible[name]:
                visualizer.add(f"cloud_{name}", geometry, point_material)
                visible[name] = True
            elif not should_show and visible[name]:
                visualizer.remove(f"cloud_{name}")
                visible[name] = False
        refresh_camera_geometry()

    def select_callback(name: str):
        def callback():
            show_clouds({name} if name in clouds else set())
            fit_visible_view()
            print(f"Showing: {name}")
        return callback

    def toggle_cameras():
        nonlocal camera_visible, shown_camera_key
        if not camera_geometries:
            return False
        if camera_visible:
            if shown_camera_key is not None:
                visualizer.remove("cameras")
            shown_camera_key = None
        else:
            camera_visible = True
            refresh_camera_geometry()
            print("Cameras: ON")
            return
        camera_visible = not camera_visible
        print(f"Cameras: {'ON' if camera_visible else 'OFF'}")

    def toggle_reference_frame():
        nonlocal frame_visible
        if frame_visible:
            visualizer.remove("reference_frame")
        else:
            visualizer.add("reference_frame", frame, mesh_material)
        frame_visible = not frame_visible
        print(f"World reference frame: {'ON' if frame_visible else 'OFF'}")

    def toggle_point_colors():
        nonlocal use_original_colors
        use_original_colors = not use_original_colors
        for name, cloud in clouds.items():
            source = original_colors[name]
            if use_original_colors and source is not None:
                cloud.colors = o3d.utility.Vector3dVector(source)
            else:
                cloud.paint_uniform_color(stage_colors[name])
            if visible[name]:
                visualizer.add(f"cloud_{name}", cloud, point_material)
        mode = "original RGB" if use_original_colors else "stage colors"
        unavailable = [
            name for name, colors in original_colors.items() if colors is None
        ]
        print(f"Point coloring: {mode}")
        if use_original_colors and unavailable:
            print(
                "No stored RGB for: " + ", ".join(unavailable)
                + "; keeping their stage colors"
            )

    screenshot_path = args.screenshot.expanduser().resolve()

    def save_screenshot():
        visualizer.save_image(screenshot_path)
        print(f"Saved screenshot: {screenshot_path}")

    def reset_view():
        fit_visible_view()

    print("\n" + controls)
    visualizer.register_key(ord("1"), select_callback("sfm"))
    visualizer.register_key(ord("2"), select_callback("sfm_filtered"))
    visualizer.register_key(ord("3"), select_callback("mvs"))
    visualizer.register_key(ord("4"), select_callback("mvs_filtered"))
    visualizer.register_key(ord("5"), select_callback("mvs_metric"))
    visualizer.register_key(ord("C"), toggle_cameras)
    visualizer.register_key(ord("F"), toggle_reference_frame)
    visualizer.register_key(ord("P"), toggle_point_colors)
    visualizer.register_key(ord("R"), reset_view)
    visualizer.register_key(ord("S"), save_screenshot)
    visualizer.register_key(ord("Q"), visualizer.close)
    visualizer.register_key(256, visualizer.close)

    if args.stage in clouds:
        initial = {args.stage}
    else:
        fallback = next(
            name
            for name in (
                "mvs_metric", "mvs_filtered", "mvs", "sfm_filtered", "sfm"
            )
            if name in clouds
        )
        print(f"Requested stage {args.stage} is unavailable; showing {fallback}")
        initial = {fallback}
    show_clouds(initial)
    if camera_geometries:
        camera_visible = True
        refresh_camera_geometry()
    fit_visible_view()
    visualizer.run()
    return 0


if __name__ == "__main__":
    import os
    os.environ["XDG_SESSION_TYPE"] = "x11"
    raise SystemExit(main())
