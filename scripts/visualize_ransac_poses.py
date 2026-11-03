#!/usr/bin/env python3
"""Visualize final multi-instance RANSAC positions and axis-angle orientations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

plot_cache = Path(tempfile.gettempdir()) / "label_factory_plot_cache"
plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(plot_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dataset_config import dataset_path


def dataset_from_config(config_path: Path) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    if "path" not in config or "dataset_id" not in config:
        raise ValueError(
            f"{config_path} must contain both 'path' and 'dataset_id'"
        )
    return dataset_path(config["path"], config["dataset_id"])


def equal_3d_axes(axis, points: np.ndarray, minimum_radius: float) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.6, minimum_radius)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def load_poses(sequence_dir: Path, object_name: str | None) -> list[dict]:
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    directories = (
        [sequence_dir / object_name]
        if object_name
        else sorted(
            (path for path in sequence_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name.lower(),
        )
    )
    poses = []
    for directory in directories:
        path = directory / f"{directory.name}_world_pose.json"
        if not path.is_file():
            print(f"Missing RANSAC result, skipping: {path}")
            continue
        payload = json.loads(path.read_text())
        instances = payload.get("instances") or [payload]
        for index, instance in enumerate(instances, 1):
            matrix = np.asarray(instance["world_from_object"], dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"Invalid world_from_object in {path}")
            rotation = Rotation.from_matrix(matrix[:3, :3])
            rotation_vector = rotation.as_rotvec()
            angle_rad = float(np.linalg.norm(rotation_vector))
            axis = (
                rotation_vector / angle_rad
                if angle_rad > 1e-12
                else np.asarray([1.0, 0.0, 0.0])
            )
            poses.append(
                {
                    "object_name": payload["object_name"],
                    "instance_id": instance.get(
                        "instance_id", f"{payload['object_name']}_{index:03d}"
                    ),
                    "translation_m": matrix[:3, 3],
                    "rotation_vector_rad": rotation_vector,
                    "axis": axis,
                    "angle_deg": float(np.degrees(angle_rad)),
                    "inliers": instance.get(
                        "number_of_inliers",
                        payload.get("ransac", {}).get("number_of_inliers"),
                    ),
                    "orientation_inliers": instance.get(
                        "number_of_orientation_inliers"
                    ),
                    "orientation_status": instance.get(
                        "orientation_status", "legacy"
                    ),
                    "source": str(path),
                }
            )
    if not poses:
        raise RuntimeError(f"No RANSAC world poses found under {sequence_dir}")
    return poses


def pose_colors(count: int):
    color_map = plt.get_cmap("tab20")
    return [color_map(index % 20) for index in range(count)]


def apply_plot_theme(figure, axis, dark: bool) -> str:
    background = "#0E1117" if dark else "white"
    foreground = "#FAFAFA" if dark else "black"
    grid_color = "#7B8494" if dark else "#666666"
    figure.patch.set_facecolor(background)
    axis.set_facecolor(background)
    axis.tick_params(colors=foreground)
    axis.xaxis.label.set_color(foreground)
    axis.yaxis.label.set_color(foreground)
    axis.zaxis.label.set_color(foreground)
    axis.title.set_color(foreground)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor(background)
        pane.set_edgecolor(grid_color)
    axis.grid(True, color=grid_color, alpha=0.35)
    return foreground


def draw_positions(poses: list[dict], output: Path, dpi: int,
                   dark: bool = False) -> None:
    positions = np.stack([pose["translation_m"] for pose in poses])
    colors = pose_colors(len(poses))
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    origin_color = "white" if dark else "black"
    axis.scatter(0, 0, 0, marker="+", s=130, c=origin_color, label="World origin")
    for pose, position, color in zip(poses, positions, colors):
        label = pose["instance_id"]
        if pose["inliers"] is not None:
            label += f" ({pose['inliers']} inliers)"
        axis.scatter(*position, s=75, color=color, depthshade=False, label=label)
        axis.text(*position, f"  {pose['instance_id']}", color=color)
        axis.plot(
            [0, position[0]], [0, position[1]], [0, position[2]],
            color=color, alpha=0.35, linewidth=1.2,
        )
    equal_3d_axes(axis, np.vstack((positions, np.zeros((1, 3)))), 0.02)
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Y (m)")
    axis.set_zlabel("World Z (m)")
    axis.set_title("RANSAC object-instance positions")
    foreground = apply_plot_theme(figure, axis, dark)
    legend = axis.legend(loc="upper left", fontsize=8)
    if dark:
        legend.get_frame().set_facecolor("#1A1D24")
        legend.get_frame().set_edgecolor("#7B8494")
        for label in legend.get_texts():
            label.set_color(foreground)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight",
                   facecolor=figure.get_facecolor())
    plt.close(figure)


def draw_orientations(poses: list[dict], output: Path, dpi: int,
                      dark: bool = False) -> None:
    vectors = np.stack([pose["rotation_vector_rad"] for pose in poses])
    colors = pose_colors(len(poses))
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(0, 0, 0, marker="+", s=130,
                 c="white" if dark else "black")
    for pose, vector, color in zip(poses, vectors, colors):
        axis.quiver(
            0, 0, 0, *vector, color=color, linewidth=2.5,
            arrow_length_ratio=0.08,
        )
        axis.scatter(*vector, s=55, color=color, depthshade=False)
        axis.text(*vector, f"  {pose['instance_id']}", color=color)
    equal_3d_axes(axis, np.vstack((vectors, np.zeros((1, 3)))), 0.25)
    axis.set_xlabel("Axis × angle X (rad)")
    axis.set_ylabel("Axis × angle Y (rad)")
    axis.set_zlabel("Axis × angle Z (rad)")
    axis.set_title("RANSAC orientations in axis-angle representation")
    apply_plot_theme(figure, axis, dark)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight",
                   facecolor=figure.get_facecolor())
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=project / "scripts/configs/default.yaml",
    )
    parser.add_argument("--object-name", help="Optional single-object filter")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--theme", choices=("light", "dark", "both"), default="both",
        help="Plot theme to generate. Both preserves light files and adds *_dark.png.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    dataset = dataset_from_config(config_path)
    sequences = dataset / "sam6d_sequences"
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset / "sam6d_scene" / "ransac_visualizations"
    )
    poses = load_poses(sequences, args.object_name)
    generated_paths = []
    if args.theme in ("light", "both"):
        positions_path = output / "ransac_positions.png"
        orientations_path = output / "ransac_orientations_axis_angle.png"
        draw_positions(poses, positions_path, args.dpi)
        draw_orientations(poses, orientations_path, args.dpi)
        generated_paths.extend((positions_path, orientations_path))
    if args.theme in ("dark", "both"):
        dark_positions = output / "ransac_positions_dark.png"
        dark_orientations = output / "ransac_orientations_axis_angle_dark.png"
        draw_positions(poses, dark_positions, args.dpi, dark=True)
        draw_orientations(poses, dark_orientations, args.dpi, dark=True)
        generated_paths.extend((dark_positions, dark_orientations))

    print(f"Visualized {len(poses)} RANSAC object instances")
    for pose in poses:
        translation_mm = pose["translation_m"] * 1000.0
        print(
            f"  {pose['instance_id']}: position_mm="
            f"{np.round(translation_mm, 3).tolist()}, "
            f"axis={np.round(pose['axis'], 5).tolist()}, "
            f"angle_deg={pose['angle_deg']:.3f}, "
            f"orientation_status={pose['orientation_status']}, "
            f"orientation_inliers={pose['orientation_inliers']}"
        )
    for path in generated_paths:
        print(f"Plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
