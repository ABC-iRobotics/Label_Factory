#!/usr/bin/env python3
"""Build a posed Blender scene and render annotations for every input image."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

import stage_timing

from RANSAC_object_pose import (
    dataset_from_config,
    load_world_from_camera,
    numeric_frame_id,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_sam6d_python(requested: Path | None) -> Path:
    if requested:
        result = requested.expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    candidates = (
        Path.home() / "anaconda3/envs/sam6d/bin/python",
        Path.home() / "anaconda3/envs/sam6d/bin/python3.9",
        Path.home() / "miniconda3/envs/sam6d/bin/python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "SAM-6D Python was not found; pass --python /path/to/sam6d/bin/python."
    )


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project / "scripts/configs/default.yaml",
    )
    parser.add_argument("--python", type=Path, help="SAM-6D Python interpreter")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reconstruction", type=Path)
    parser.add_argument("--alignment-dir", type=Path)
    parser.add_argument(
        "--model-scale",
        type=float,
        default=0.001,
        help="STL model-unit to metre scale (default: millimetres to metres).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    dataset = dataset_from_config(config_path)
    started = time.monotonic()
    sequences = dataset / "sam6d_sequences"
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset / "sam6d_scene"
    )
    reconstruction = (
        args.reconstruction.expanduser().resolve()
        if args.reconstruction
        else dataset / "reconstruction/sfm_filtered"
    )
    alignment_dir = (
        args.alignment_dir.expanduser().resolve()
        if args.alignment_dir
        else dataset / "reconstruction/metric_alignment"
    )
    for required in (
        reconstruction / "images.bin",
        reconstruction / "cameras.bin",
        alignment_dir / "scale.npy",
        alignment_dir / "rotation.npy",
        alignment_dir / "translation.npy",
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required metric reconstruction input not found: {required}. "
                "Run scripts/reconstruct_rgbd_scene.py first."
            )

    camera_data = json.loads((dataset / "camera.json").read_text())
    intrinsic = np.asarray(camera_data["cam_K"], dtype=float).reshape(3, 3)
    image_paths = sorted(
        path
        for path in (dataset / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"No input images found in {dataset / 'images'}")
    with Image.open(image_paths[0]) as image:
        width, height = image.size
    for path in image_paths[1:]:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"All images must have the same dimensions; {path} is "
                    f"{image.size}, expected {(width, height)}"
                )

    camera_poses = load_world_from_camera(reconstruction, alignment_dir)
    frames = []
    for path in image_paths:
        frame_id = numeric_frame_id(path)
        if frame_id not in camera_poses:
            print(f"Warning: skipping {path.name}; no COLMAP camera pose")
            continue
        frames.append(
            {
                "frame_id": frame_id,
                "image_path": str(path.resolve()),
                "world_from_camera_cv": camera_poses[frame_id].tolist(),
            }
        )
    if not frames:
        raise RuntimeError("No input images have corresponding camera poses")

    objects = []
    excluded_objects = []
    for object_dir in sorted(
        (path for path in sequences.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    ):
        pose_path = object_dir / f"{object_dir.name}_world_pose.json"
        if not pose_path.is_file():
            excluded_objects.append(
                {
                    "object_name": object_dir.name,
                    "reason": "missing valid RANSAC world pose",
                }
            )
            print(
                f"Excluding {object_dir.name} from Blender scene: "
                f"no valid pose at {pose_path}"
            )
            continue
        pose = json.loads(pose_path.read_text())
        model_path = Path(pose["object_path"]).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        instances = pose.get("instances") or [pose]
        for instance_index, instance in enumerate(instances, 1):
            objects.append(
                {
                    "object_id": len(objects) + 1,
                    "object_name": pose["object_name"],
                    "instance_name": instance.get(
                        "instance_id",
                        f"{pose['object_name']}_{instance_index:03d}",
                    ),
                    "model_path": str(model_path),
                    "world_from_object": instance["world_from_object"],
                }
            )
    if not objects:
        raise RuntimeError(f"No *_world_pose.json files found under {sequences}")

    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "scene_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset),
                "config_path": str(config_path),
                "output_dir": str(output),
                "image_width": width,
                "image_height": height,
                "cam_K": intrinsic.tolist(),
                "model_scale_to_metre": args.model_scale,
                "objects": objects,
                "excluded_objects": excluded_objects,
                "frames": frames,
            },
            indent=2,
        )
        + "\n"
    )

    sam6d_python = find_sam6d_python(args.python)
    blenderproc = sam6d_python.parent / "blenderproc"
    if not blenderproc.is_file():
        raise FileNotFoundError(blenderproc)
    blender_path = Path(
        os.environ.get(
            "SAM6D_BLENDER_PATH",
            str(Path.home() / "blender-3.3.1-linux-x64"),
        )
    ).expanduser()
    pipeline = (
        project
        / "external/SAM-6D/SAM-6D/Render/create_scene_annotations.py"
    )
    command = [
        blenderproc,
        "run",
        "--custom-blender-path",
        blender_path,
        pipeline,
        "--manifest",
        manifest_path,
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    print("$ " + " ".join(map(str, command)), flush=True)
    subprocess.run(
        list(map(str, command)),
        cwd=project,
        env=environment,
        check=True,
    )
    stage_timing.record(dataset, "annotation_rendering", time.monotonic() - started)
    print(f"Saved Blender scene and annotations to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
