#!/usr/bin/env python3
"""Create self-contained task-specific Parquet datasets from final outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from datasets import Dataset, Image

from dataset_config import dataset_path


TASKS = ("depth_estimation", "instance_segmentation", "object_pose_estimation")


def dataset_from_config(path: Path) -> Path:
    payload = yaml.safe_load(path.read_text()) or {}
    return dataset_path(payload["path"], payload["dataset_id"])


def json_compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def payload_size(value: object) -> int:
    if isinstance(value, (bytes, bytearray, str)):
        return len(value)
    if isinstance(value, dict):
        return sum(payload_size(item) for item in value.values())
    return 0


def frame_rows(dataset: Path):
    scene_path = dataset / "sam6d_scene/annotations.json"
    camera_path = dataset / "camera.json"
    if not scene_path.is_file() or not camera_path.is_file():
        raise FileNotFoundError("camera.json and sam6d_scene/annotations.json are required")
    scene = json.loads(scene_path.read_text())
    camera = json.loads(camera_path.read_text())
    intrinsic = camera["cam_K"]
    for frame in scene["frames"]:
        frame_id = int(frame["frame_id"])
        image_path = dataset / "images" / Path(frame["image_path"]).name
        depth_path = dataset / "depth_maps" / f"depth_{frame_id:06d}.png"
        mask_path = dataset / "sam6d_scene/masks" / f"{frame_id:06d}.png"
        missing = [path for path in (image_path, depth_path, mask_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing frame assets: " + ", ".join(map(str, missing)))
        common = {
            "dataset_id": dataset.name,
            "frame_id": frame_id,
            "image_file_name": image_path.name,
            "image": {"bytes": image_path.read_bytes(), "path": image_path.name},
            "width": int(camera["image_width"]),
            "height": int(camera["image_height"]),
            "camera_intrinsics": [float(value) for value in intrinsic],
        }
        yield common, frame, depth_path, mask_path, camera


def task_row(task: str, common: dict, frame: dict, depth_path: Path,
             mask_path: Path, camera: dict) -> dict:
    if task == "depth_estimation":
        return {
            **common,
            "depth_file_name": depth_path.name,
            "depth": {"bytes": depth_path.read_bytes(), "path": depth_path.name},
            "depth_unit": str(camera.get("depth_unit", "millimetre")),
            "depth_scale": float(camera.get("depth_scale", 1.0)),
        }
    annotations = frame.get("annotations", [])
    base = {
        **common,
        "instance_mask_file_name": mask_path.name,
        "instance_mask": {"bytes": mask_path.read_bytes(), "path": mask_path.name},
        "annotations_json": json_compact(annotations),
        "object_names": [str(item["object_name"]) for item in annotations],
        "object_model_keys": [f"{common['dataset_id']}:{item['object_name']}"
                              for item in annotations],
        "instance_names": [str(item["instance_name"]) for item in annotations],
        "boxes_xywh": [[int(value) for value in item["bbox_xywh"]]
                       for item in annotations],
    }
    if task == "instance_segmentation":
        return base
    return {
        **base,
        "depth_file_name": depth_path.name,
        "depth": {"bytes": depth_path.read_bytes(), "path": depth_path.name},
        "depth_unit": str(camera.get("depth_unit", "millimetre")),
        "depth_scale": float(camera.get("depth_scale", 1.0)),
        "world_from_camera": [float(value) for row in frame["world_from_camera"]
                              for value in row],
        "camera_from_world": [float(value) for row in frame["camera_from_world"]
                              for value in row],
        "world_from_objects": [
            [float(value) for row in item["world_from_object"] for value in row]
            for item in annotations
        ],
        "camera_from_objects": [
            [float(value) for row in item["camera_from_object"] for value in row]
            for item in annotations
        ],
    }


def split_for_frame(dataset_id: str, frame_id: int, train_ratio: float,
                    validation_ratio: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{dataset_id}:{frame_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def write_task(dataset: Path, output_root: Path, task: str, shard_bytes: int,
               overwrite: bool, train_ratio: float, validation_ratio: float,
               seed: int) -> dict:
    task_dir = output_root / task
    if task_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{task_dir} exists; pass --overwrite")
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)
    splits = ("train", "validation", "test")
    shard_paths: dict[str, list[Path]] = {split: [] for split in splits}
    rows: dict[str, list[dict]] = {split: [] for split in splits}
    estimated_bytes: dict[str, int] = {split: 0 for split in splits}

    def flush(split: str) -> None:
        if not rows[split]:
            return
        path = task_dir / f"{split}-{len(shard_paths[split]):05d}.parquet"
        hf_dataset = Dataset.from_list(rows[split])
        for column in ("image", "depth", "instance_mask"):
            if column in hf_dataset.column_names:
                hf_dataset = hf_dataset.cast_column(column, Image())
        hf_dataset.to_parquet(path)
        shard_paths[split].append(path)
        rows[split] = []
        estimated_bytes[split] = 0

    for common, frame, depth_path, mask_path, camera in frame_rows(dataset):
        row = task_row(task, common, frame, depth_path, mask_path, camera)
        row_bytes = sum(payload_size(value) for value in row.values())
        split = split_for_frame(dataset.name, common["frame_id"], train_ratio,
                                validation_ratio, seed)
        if rows[split] and estimated_bytes[split] + row_bytes > shard_bytes:
            flush(split)
        rows[split].append(row)
        estimated_bytes[split] += row_bytes
    for split in splits:
        flush(split)
    final_paths: dict[str, list[Path]] = {split: [] for split in splits}
    for split in splits:
        total = len(shard_paths[split])
        for index, path in enumerate(shard_paths[split]):
            final = path.with_name(f"{split}-{index:05d}-of-{total:05d}.parquet")
            path.rename(final)
            final_paths[split].append(final)
    all_paths = [path for split in splits for path in final_paths[split]]
    split_rows = {split: sum(pq.read_metadata(path).num_rows
                             for path in final_paths[split]) for split in splits}
    metadata = {
        "format": "label_factory_task_parquet_v3_hf_images_models",
        "dataset_id": dataset.name,
        "task": task,
        "split_seed": seed,
        "split_ratios": {"train": train_ratio, "validation": validation_ratio,
                         "test": 1.0 - train_ratio - validation_ratio},
        "rows": sum(split_rows.values()),
        "rows_per_split": split_rows,
        "shards": {split: [path.name for path in final_paths[split]]
                   for split in splits},
        "bytes": sum(path.stat().st_size for path in all_paths),
    }
    if task == "object_pose_estimation":
        scene = json.loads((dataset / "sam6d_scene/annotations.json").read_text())
        model_rows = []
        seen = set()
        for item in scene.get("objects", []):
            object_name = str(item["object_name"])
            if object_name in seen:
                continue
            seen.add(object_name)
            stl_path = dataset / "objects/stl" / Path(item["model_path"]).name
            if not stl_path.is_file():
                raise FileNotFoundError(stl_path)
            payload = stl_path.read_bytes()
            model_rows.append({
                "dataset_id": dataset.name,
                "object_name": object_name,
                "object_model_key": f"{dataset.name}:{object_name}",
                "stl_file_name": stl_path.name,
                "stl": payload,
                "stl_sha256": hashlib.sha256(payload).hexdigest(),
                "stl_size_bytes": len(payload),
                "model_scale_to_metre": float(scene.get("model_scale_to_metre", 0.001)),
            })
        models_path = task_dir / "models.parquet"
        Dataset.from_list(model_rows).to_parquet(models_path)
        metadata["object_models"] = len(model_rows)
        metadata["object_models_file"] = models_path.name
        metadata["bytes"] += models_path.stat().st_size
    (task_dir / "dataset_info.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--shard-size-mb", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be between 0 and 1")
    if not 0 <= args.validation_ratio < 1:
        parser.error("--validation-ratio must be between 0 and 1")
    if args.train_ratio + args.validation_ratio >= 1:
        parser.error("train + validation ratios must be less than 1")
    dataset = dataset_from_config(args.config.resolve())
    output_root = (args.output_dir.resolve() if args.output_dir else
                   dataset / "publishing/task_parquet")
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for task in args.tasks:
        print(f"Creating {task} Parquet dataset")
        summary = write_task(dataset, output_root, task,
                             args.shard_size_mb * 1024 * 1024, args.overwrite,
                             args.train_ratio, args.validation_ratio,
                             args.split_seed)
        summaries.append(summary)
        shard_count = sum(len(paths) for paths in summary["shards"].values())
        print(f"  {summary['rows']} rows {summary['rows_per_split']}, "
              f"{shard_count} shards, "
              f"{summary['bytes'] / 1_073_741_824:.2f} GiB")
    (output_root / "manifest.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Task datasets: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
