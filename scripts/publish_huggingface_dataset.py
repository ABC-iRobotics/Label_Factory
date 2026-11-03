#!/usr/bin/env python3
"""Inventory and optionally upload a human-reviewed dataset to HF Hub."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from dataset_config import dataset_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TASK_CONFIGS = ("depth_estimation", "instance_segmentation", "object_pose_estimation")
UPLOAD_PATTERNS = [
    "camera.json",
    "images/**",
    "depth_maps/**",
    "objects/stl/**",
    "objects/textured/**",
    "reconstruction/mvs_filtered_metric.ply",
    "reconstruction/metric_alignment/**",
    "sam6d_scene/annotations.json",
    "sam6d_scene/scene_manifest.json",
    "sam6d_scene/masks/**",
    "sam6d_scene/visualizations/**",
    "sam6d_scene/segmentation_raw/**",
    "sam6d_scene/annotation_overlays.mp4",
    "sam6d_scene/ransac_visualizations/**",
    "sam6d_sequences/*/*_poses_per_frame.json",
    "experimental_multi_instance/pem/*/*_poses_per_frame.json",
    "sam6d_sequences/*/*_world_pose.json",
    "sam6d_sequences/ransac_run_summary.json",
]


def dataset_from_config(path: Path) -> Path:
    payload = yaml.safe_load(path.read_text()) or {}
    if "path" not in payload or "dataset_id" not in payload:
        raise ValueError(f"{path} must define path and dataset_id")
    return dataset_path(payload["path"], payload["dataset_id"])


def numeric_id(path: Path) -> int | None:
    values = re.findall(r"\d+", path.stem)
    return int(values[-1]) if values else None


def indexed_files(folder: Path) -> dict[int, Path]:
    result = {}
    if not folder.is_dir():
        return result
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index = numeric_id(path)
            if index is not None:
                result[index] = path
    return result


def selected_files(dataset: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in UPLOAD_PATTERNS:
        if pattern.endswith("/**"):
            root = dataset / pattern[:-3]
            if root.is_dir():
                files.update(path for path in root.rglob("*") if path.is_file())
        else:
            files.update(path for path in dataset.glob(pattern) if path.is_file())
    return sorted(files)


def stage_upload_tree(dataset: Path, task_root: Path, card_path: Path,
                      stage: Path) -> int:
    """Build one upload tree using hard links, avoiding a second data copy."""
    mappings: list[tuple[Path, Path]] = [(card_path, Path("README.md"))]
    mappings.extend(
        (source, Path(dataset.name) / source.relative_to(dataset))
        for source in selected_files(dataset)
    )
    for task in TASK_CONFIGS:
        task_dir = task_root / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Missing generated task dataset: {task_dir}")
        mappings.extend(
            (source, Path(dataset.name) / "data" / task / source.name)
            for source in sorted(task_dir.iterdir()) if source.is_file()
        )
    for source, relative_target in mappings:
        target = stage / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return len(mappings)


def inventory(dataset: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    camera_path = dataset / "camera.json"
    if not camera_path.is_file():
        errors.append("Missing camera.json")
    else:
        try:
            camera = json.loads(camera_path.read_text())
            for key in ("cam_K", "image_width", "image_height"):
                if key not in camera:
                    errors.append(f"camera.json is missing {key}")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"Invalid camera.json: {error}")

    images = indexed_files(dataset / "images")
    depths = indexed_files(dataset / "depth_maps")
    if not images:
        errors.append("No RGB images found")
    if not depths:
        errors.append("No depth maps found")
    missing_depth = sorted(set(images) - set(depths))
    missing_rgb = sorted(set(depths) - set(images))
    if missing_depth:
        errors.append(f"{len(missing_depth)} RGB frames have no depth map")
    if missing_rgb:
        errors.append(f"{len(missing_rgb)} depth maps have no RGB frame")

    required_outputs = [
        dataset / "reconstruction/mvs_filtered_metric.ply",
        dataset / "reconstruction/metric_alignment/camera_poses_metric.json",
        dataset / "sam6d_scene/annotations.json",
    ]
    for path in required_outputs:
        if not path.is_file():
            errors.append(f"Missing required output: {path.relative_to(dataset)}")
    camera_poses_path = (
        dataset / "reconstruction/metric_alignment/camera_poses_metric.json"
    )
    metric_camera_pose_count = 0
    if camera_poses_path.is_file():
        try:
            camera_poses = json.loads(camera_poses_path.read_text())
            metric_camera_pose_count = len(camera_poses.get("poses", []))
            if metric_camera_pose_count != len(images):
                errors.append(
                    f"Metric camera pose count ({metric_camera_pose_count}) does "
                    f"not match RGB frame count ({len(images)})"
                )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"Invalid metric camera pose JSON: {error}")
    annotation_path = dataset / "sam6d_scene/annotations.json"
    annotation_frames = annotation_objects = 0
    if annotation_path.is_file():
        try:
            annotations = json.loads(annotation_path.read_text())
            annotation_frames = len(annotations.get("frames", []))
            annotation_objects = len(annotations.get("objects", []))
            if annotation_frames == 0:
                errors.append("annotations.json contains no frames")
            if annotation_objects == 0:
                errors.append("annotations.json contains no objects")
            if annotation_frames != len(images):
                errors.append(
                    f"Annotation frame count ({annotation_frames}) does not "
                    f"match RGB frame count ({len(images)})"
                )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"Invalid annotations.json: {error}")

    stls = list((dataset / "objects/stl").glob("*.stl"))
    if not stls:
        warnings.append("No STL object models are included")
    masks = indexed_files(dataset / "sam6d_scene/masks")
    overlays = indexed_files(dataset / "sam6d_scene/visualizations")
    world_poses = list((dataset / "sam6d_sequences").glob("*/*_world_pose.json"))
    frame_poses = list((dataset / "sam6d_sequences").glob("*/*_poses_per_frame.json"))
    frame_poses.extend(
        (dataset / "experimental_multi_instance/pem").glob(
            "*/*_poses_per_frame.json"
        )
    )
    if not masks:
        errors.append("No final instance-mask images found in sam6d_scene/masks")
    if not overlays:
        errors.append("No annotation-overlay images found in sam6d_scene/visualizations")
    if not world_poses:
        errors.append("No world-coordinate object pose JSON files found")
    if not frame_poses:
        errors.append("No per-frame object pose JSON files found")
    files = selected_files(dataset)
    task_root = dataset / "publishing/task_parquet"
    task_files = sorted(
        path for task in TASK_CONFIGS
        for path in (task_root / task).glob("*") if path.is_file()
    )
    missing_task_configs = [task for task in TASK_CONFIGS
                            if not list((task_root / task).glob("train-*.parquet"))]
    if missing_task_configs:
        warnings.append(
            "Task Parquet files have not been generated yet for: " +
            ", ".join(missing_task_configs) + ". They are generated automatically before upload."
        )
    # The upload staging tree always contains the generated repository data card
    # in addition to the dataset assets and task Parquet shards.
    card_path = dataset / "publishing/README.md"
    card_bytes = card_path.stat().st_size if card_path.is_file() else 0
    total_bytes = sum(path.stat().st_size for path in files + task_files) + card_bytes
    return {
        "format": "label_factory_huggingface_inventory_v1",
        "dataset": str(dataset),
        "remote_dataset_directory": dataset.name,
        "technically_ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "statistics": {
            "rgb_frames": len(images), "depth_frames": len(depths),
            "annotation_frames": annotation_frames,
            "annotated_objects": annotation_objects,
            "stl_models": len(stls), "mask_images": len(masks),
            "annotation_images": len(overlays),
            "world_pose_files": len(world_poses),
            "per_frame_pose_files": len(frame_poses),
            "parquet_files": len(task_files),
            "upload_files": 1 + len(files) + len(task_files),
            "upload_bytes": total_bytes,
        },
        "included_files": (
            ["README.md"] +
            [str(path.relative_to(dataset)) for path in files] +
            [str(Path("data") / path.relative_to(task_root)) for path in task_files]
        ),
        "upload_patterns": UPLOAD_PATTERNS,
    }


def dataset_card(dataset: Path, repo_id: str, license_name: str,
                 report: dict) -> str:
    stats = report["statistics"]
    return f"""---
pretty_name: Label Factory Multi-Task RGB-D Collection
license: {license_name}
tags:
- image
- video
- 3d
- rgb-d
configs:
- config_name: depth_estimation
  data_files:
  - split: train
    path: "*/data/depth_estimation/train-*.parquet"
  - split: validation
    path: "*/data/depth_estimation/validation-*.parquet"
  - split: test
    path: "*/data/depth_estimation/test-*.parquet"
- config_name: instance_segmentation
  data_files:
  - split: train
    path: "*/data/instance_segmentation/train-*.parquet"
  - split: validation
    path: "*/data/instance_segmentation/validation-*.parquet"
  - split: test
    path: "*/data/instance_segmentation/test-*.parquet"
- config_name: object_pose_estimation
  data_files:
  - split: train
    path: "*/data/object_pose_estimation/train-*.parquet"
  - split: validation
    path: "*/data/object_pose_estimation/validation-*.parquet"
  - split: test
    path: "*/data/object_pose_estimation/test-*.parquet"
- config_name: object_pose_models
  data_files:
  - split: train
    path: "*/data/object_pose_estimation/models.parquet"
---

# {repo_id}

Growing multi-task RGB-D collection generated with the Label Factory workflow.
Every capture is stored in its own zero-padded directory (`0000`, `0001`,
`0002`, ...). Uploading a new capture appends it to the collection without
replacing earlier directories. Task configurations aggregate compatible
Parquet shards from every numbered directory.

## Training configurations

- `depth_estimation`: RGB input, metric depth target, intrinsics and depth units
- `instance_segmentation`: RGB input, instance-mask target, boxes and annotations
- `object_pose_estimation`: RGB-D input, masks, camera transforms and object poses
- `object_pose_models`: one STL payload per object type and numbered dataset

Every configuration provides deterministic `train`, `validation`, and `test`
splits. Adding another numbered capture appends rows through the wildcard paths;
it does not replace earlier datasets.

## Loading all or several numbered datasets

The configuration automatically combines every uploaded directory (`0000`,
`0001`, `0002`, ...). Filter `dataset_id` to use selected captures:

```python
from datasets import load_dataset

poses = load_dataset("{repo_id}", "object_pose_estimation")
selected_ids = set(["0000", "0001", "0004"])
for split_name in poses:
    poses[split_name] = poses[split_name].filter(
        lambda row: row["dataset_id"] in selected_ids
    )
```

## Depth-estimation example

```python
from datasets import load_dataset
import numpy as np

depth_dataset = load_dataset("{repo_id}", "depth_estimation")
sample = depth_dataset["train"][0]
rgb = sample["image"].convert("RGB")
depth_native = np.asarray(sample["depth"])
depth_mm = depth_native.astype(np.float32) * sample["depth_scale"]
depth_m = depth_mm / 1000.0
intrinsics_3x3 = np.asarray(sample["camera_intrinsics"]).reshape(3, 3)
```

## Instance-segmentation example

```python
from datasets import load_dataset
import json
import numpy as np

instances = load_dataset("{repo_id}", "instance_segmentation")
sample = instances["validation"][0]
rgb = sample["image"].convert("RGB")
instance_mask = np.asarray(sample["instance_mask"])
annotations = json.loads(sample["annotations_json"])
boxes_xywh = np.asarray(sample["boxes_xywh"])
instance_names = sample["instance_names"]
```

## Object-pose-estimation example

```python
from datasets import load_dataset
import numpy as np

poses = load_dataset("{repo_id}", "object_pose_estimation")
sample = poses["test"][0]
rgb = sample["image"].convert("RGB")
depth = np.asarray(sample["depth"])
instance_mask = np.asarray(sample["instance_mask"])
world_from_camera = np.asarray(sample["world_from_camera"]).reshape(4, 4)
camera_from_world = np.asarray(sample["camera_from_world"]).reshape(4, 4)
world_from_objects = np.asarray(sample["world_from_objects"]).reshape(-1, 4, 4)
camera_from_objects = np.asarray(sample["camera_from_objects"]).reshape(-1, 4, 4)
object_names = sample["object_names"]

# Resolve the STL models referenced by this pose sample.
models = load_dataset("{repo_id}", "object_pose_models")["train"]
required_keys = set(sample["object_model_keys"])
models = models.filter(lambda row: row["object_model_key"] in required_keys)
first_model = models[0]
with open(first_model["stl_file_name"], "wb") as file:
    file.write(first_model["stl"])
```

## Collection contents

Each numbered directory contributes:

- RGB images and aligned 16-bit metric depth maps
- camera intrinsics and metric world-coordinate camera poses
- instance masks, boxes, RLE annotations and visualization overlays
- per-frame and consolidated object poses
- metric scene reconstruction and object STL models
- task-specific train, validation and test Parquet shards

## Latest generated contribution

- Directory: `{dataset.name}/`
- RGB-D frames: {stats['rgb_frames']}
- Annotated object instances: {stats['annotated_objects']}
- Selected upload files: {stats['upload_files']}

Depth images are stored in millimetres. Pose conventions and coordinate-system
metadata are recorded in `{dataset.name}/sam6d_scene/annotations.json` and
`{dataset.name}/reconstruction/metric_alignment/camera_poses_metric.json`.

## Limitations

Annotations are automatically generated and must be independently reviewed for
the intended downstream use. Reconstruction, depth-alignment, segmentation and
pose-estimation errors may remain.
"""


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--repo-id", help="Hugging Face dataset ID: owner/name")
    parser.add_argument("--license", default="other")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--create-pr", action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    dataset = dataset_from_config(args.config.expanduser().resolve())
    task_root = dataset / "publishing/task_parquet"
    if args.upload:
        requested_splits = {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": 1.0 - args.train_ratio - args.validation_ratio,
        }
        if requested_splits["train"] <= 0 or requested_splits["validation"] < 0 or requested_splits["test"] <= 0:
            parser.error("split ratios must produce positive train/test and non-negative validation splits")
        def task_matches_settings(task: str) -> bool:
            info_path = task_root / task / "dataset_info.json"
            if not info_path.is_file():
                return False
            try:
                info = json.loads(info_path.read_text())
            except (OSError, json.JSONDecodeError):
                return False
            return (info.get("format") == "label_factory_task_parquet_v3_hf_images_models" and
                    info.get("split_seed") == args.split_seed and
                    all(abs(info.get("split_ratios", {}).get(name, -1) - ratio) < 1e-9
                        for name, ratio in requested_splits.items()) and
                    all(bool(list((task_root / task).glob(f"{name}-*.parquet")))
                        for name in ("train", "validation", "test")))
        task_parquet_ready = all(
            task_matches_settings(task) for task in TASK_CONFIGS
        )
        if task_parquet_ready:
            print(f"Reusing task-specific Parquet datasets: {task_root}")
        else:
            print("Preparing task-specific Parquet datasets")
            subprocess.run(
                [sys.executable, str(project / "scripts/create_huggingface_parquet.py"),
                 "--config", str(args.config.expanduser().resolve()), "--overwrite",
                 "--train-ratio", str(args.train_ratio),
                 "--validation-ratio", str(args.validation_ratio),
                 "--split-seed", str(args.split_seed)],
                check=True,
            )
    report = inventory(dataset)
    publishing = dataset / "publishing"
    publishing.mkdir(parents=True, exist_ok=True)
    report_path = publishing / "huggingface_inventory.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    stats = report["statistics"]
    print(f"Technical inventory: {'READY' if report['technically_ready'] else 'INCOMPLETE'}")
    print(f"RGB-D pairs: {stats['rgb_frames']}; objects: {stats['annotated_objects']}")
    print(f"Upload selection: {stats['upload_files']} files, "
          f"{stats['upload_bytes'] / 1_073_741_824:.2f} GiB")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    for warning in report["warnings"]:
        print(f"REVIEW: {warning}")
    print(f"Inventory report: {report_path}")
    if not args.upload:
        return 0 if report["technically_ready"] else 1
    if not report["technically_ready"]:
        raise RuntimeError("Technical dataset inputs are incomplete; upload was not started")
    if not args.repo_id or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo_id):
        raise ValueError("--repo-id must use owner/name format")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for upload")
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub in this environment") from error
    card_path = publishing / "README.md"
    card_path.write_text(dataset_card(dataset, args.repo_id, args.license, report))
    api = HfApi(token=token)
    repository_owner = args.repo_id.split("/", 1)[0]
    remote_dataset_dir = dataset.name
    identity_name = "the authenticated account"
    try:
        identity = api.whoami()
        identity_name = identity.get("name") or identity.get("fullname") or "unknown"
        print(f"Authenticated Hugging Face account: {identity_name}")
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private,
                        exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hf_upload_", dir=publishing) as temporary:
            stage = Path(temporary)
            staged_count = stage_upload_tree(dataset, task_root, card_path, stage)
            print(f"Submitting one Hugging Face upload containing {staged_count} files")
            upload_commit = api.upload_folder(
                folder_path=str(stage), repo_id=args.repo_id, repo_type="dataset",
                create_pr=True,
                commit_message=f"Add Label Factory dataset {dataset.name}",
            )
    except HfHubHTTPError as error:
        mode_hint = (
            "The pull-request upload was rejected. Ask an organization administrator "
            f"to allow {identity_name} to contribute to {repository_owner}, and verify "
            "that the token is authorized for this dataset repository."
        )
        raise RuntimeError(
            f"Hugging Face authorization failed for {args.repo_id}. {mode_hint}"
        ) from error
    pr_url = getattr(upload_commit, "pr_url", None)
    destination = pr_url or f"https://huggingface.co/datasets/{args.repo_id}/pulls"
    print("Submitted dataset pull request: " + destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
