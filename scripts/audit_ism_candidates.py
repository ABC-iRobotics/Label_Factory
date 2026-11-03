#!/usr/bin/env python3
"""Audit all object-specific SAM-6D ISM candidates without changing the pipeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from dataset_config import dataset_path


COLORS = (
    (255, 70, 70), (70, 210, 90), (65, 135, 255), (245, 190, 55),
    (190, 85, 235), (35, 210, 210), (255, 120, 200), (170, 220, 55),
)


def dataset_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text()) or {}
    return dataset_path(config["path"], config["dataset_id"])


def decode_rle(rle: dict) -> np.ndarray:
    height, width = map(int, rle["size"])
    flat = np.zeros(height * width, dtype=bool)
    offset = 0
    foreground = False
    for raw_count in rle["counts"]:
        count = int(raw_count)
        if foreground:
            flat[offset:offset + count] = True
        offset += count
        foreground = not foreground
    if offset != height * width:
        raise ValueError(f"Invalid RLE length {offset}; expected {height * width}")
    return flat.reshape(width, height).T


def overlap_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(left & right))
    if intersection == 0:
        return 0.0, 0.0
    left_area = int(np.count_nonzero(left))
    right_area = int(np.count_nonzero(right))
    union = left_area + right_area - intersection
    return intersection / union, intersection / min(left_area, right_area)


def inspect_candidates(detections: list[dict], score_threshold: float,
                       iou_threshold: float,
                       containment_threshold: float) -> tuple[list[dict], list[dict]]:
    candidates = []
    for source_index, detection in enumerate(detections):
        score = float(detection.get("score", 0.0))
        if score < score_threshold:
            continue
        mask = decode_rle(detection["segmentation"])
        candidates.append({
            "source_index": source_index,
            "score": score,
            "bbox": [float(value) for value in detection.get("bbox", [])],
            "area_px": int(np.count_nonzero(mask)),
            "mask": mask,
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    retained: list[dict] = []
    rejected: list[dict] = []
    for candidate in candidates:
        duplicate = None
        for accepted_index, accepted in enumerate(retained):
            iou, containment = overlap_metrics(candidate["mask"], accepted["mask"])
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = {
                    "retained_index": accepted_index,
                    "iou": iou,
                    "containment": containment,
                }
                break
        if duplicate is None:
            retained.append(candidate)
        else:
            candidate["duplicate_of"] = duplicate
            rejected.append(candidate)
    return retained, rejected


def serializable(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if key != "mask"}


def visualize(image_path: Path, retained: list[dict], output_path: Path) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    for index, candidate in enumerate(retained):
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
        mask = candidate["mask"]
        image[mask] = image[mask] * 0.48 + color * 0.52
    rendered = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(rendered)
    for index, candidate in enumerate(retained):
        bbox = candidate["bbox"]
        if len(bbox) == 4:
            x, y, width, height = bbox
            color = COLORS[index % len(COLORS)]
            draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
            draw.text((x + 4, max(0, y - 15)),
                      f"#{index + 1} {candidate['score']:.3f}", fill=color,
                      stroke_width=2, stroke_fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)


def frame_id(path: Path) -> int:
    values = re.findall(r"\d+", path.name)
    if not values:
        raise ValueError(f"Frame directory has no numeric ID: {path}")
    return int(values[-1])


def indexed_images(folder: Path) -> dict[int, Path]:
    result = {}
    for path in folder.glob("*"):
        values = re.findall(r"\d+", path.stem)
        if path.is_file() and values and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            result[int(values[-1])] = path
    return result


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--object", action="append", dest="objects",
                        help="Object name to audit; repeat as needed. Default: all.")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--mask-iou-threshold", type=float, default=0.8)
    parser.add_argument("--containment-threshold", type=float, default=0.9)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--no-visualizations", action="store_true")
    args = parser.parse_args()
    for name, value in (("score", args.score_threshold),
                        ("IoU", args.mask_iou_threshold),
                        ("containment", args.containment_threshold)):
        if not 0 <= value <= 1:
            parser.error(f"{name} threshold must be between 0 and 1")

    dataset = dataset_from_config(args.config.expanduser().resolve())
    sequence_root = dataset / "sam6d_sequences"
    if not sequence_root.is_dir():
        raise FileNotFoundError(
            f"No SAM-6D sequences found in {sequence_root}. Run the existing SAM-6D "
            "stage first, or select a dataset that already contains ISM results."
        )
    object_names = args.objects or sorted(
        path.name for path in sequence_root.iterdir()
        if path.is_dir() and (dataset / "objects/stl" / f"{path.name}.stl").is_file()
    )
    images = indexed_images(dataset / "images")
    output_root = dataset / "experimental_multi_instance/ism_candidate_audit"
    summary_path = output_root / "summary.json"
    previous_objects = {}
    if args.objects and summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text())
            if previous.get("format") == "label_factory_multi_instance_ism_audit_v1":
                previous_objects = previous.get("objects", {})
        except (OSError, json.JSONDecodeError):
            pass
    summary = {
        "format": "label_factory_multi_instance_ism_audit_v1",
        "source_dataset": str(dataset),
        "experimental": True,
        "changes_current_pipeline": False,
        "settings": {
            "score_threshold": args.score_threshold,
            "mask_iou_threshold": args.mask_iou_threshold,
            "containment_threshold": args.containment_threshold,
        },
        "objects": previous_objects,
    }
    for object_name in object_names:
        frame_root = sequence_root / object_name / "frames"
        detection_paths = sorted(frame_root.glob("*/sam6d_results/detection_ism.json"))
        if args.start_frame is not None:
            detection_paths = [path for path in detection_paths
                               if frame_id(path.parents[1]) >= args.start_frame]
        if args.end_frame is not None:
            detection_paths = [path for path in detection_paths
                               if frame_id(path.parents[1]) <= args.end_frame]
        object_summary = {"frames": [], "totals": {
            "raw": 0, "above_score_threshold": 0, "retained": 0, "duplicates": 0,
        }}
        for detection_path in detection_paths:
            identifier = frame_id(detection_path.parents[1])
            detections = json.loads(detection_path.read_text())
            retained, rejected = inspect_candidates(
                detections, args.score_threshold, args.mask_iou_threshold,
                args.containment_threshold,
            )
            frame_summary = {
                "frame_id": identifier,
                "source": str(detection_path),
                "raw_count": len(detections),
                "above_score_threshold": len(retained) + len(rejected),
                "retained_count": len(retained),
                "duplicate_count": len(rejected),
                "retained": [serializable(item) for item in retained],
                "duplicates": [serializable(item) for item in rejected],
            }
            object_summary["frames"].append(frame_summary)
            totals = object_summary["totals"]
            totals["raw"] += len(detections)
            totals["above_score_threshold"] += len(retained) + len(rejected)
            totals["retained"] += len(retained)
            totals["duplicates"] += len(rejected)
            if not args.no_visualizations and identifier in images and retained:
                visualize(
                    images[identifier], retained,
                    output_root / object_name / "frames" / f"{identifier:06d}.png",
                )
        summary["objects"][object_name] = object_summary
        totals = object_summary["totals"]
        print(
            f"{object_name}: {len(detection_paths)} frames, {totals['raw']} raw, "
            f"{totals['above_score_threshold']} above threshold, "
            f"{totals['retained']} retained, {totals['duplicates']} duplicates"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Experimental audit: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
