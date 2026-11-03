#!/usr/bin/env python3
"""Remove reproducible pipeline intermediates while retaining final dataset assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from dataset_config import dataset_path


REVIEW_IMAGE_LIMIT = 10


def size_bytes(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def targets(dataset: Path) -> list[Path]:
    reconstruction = dataset / "reconstruction"
    result = [
        reconstruction / "mvs/stereo",
        reconstruction / "mvs/images",
        reconstruction / "mvs/sparse",
        reconstruction / ".sfm_candidates",
        reconstruction / ".mvs_fusion",
        reconstruction / ".selected_images",
        reconstruction / "database.db",
        reconstruction / "sfm",
        reconstruction / "sfm_filtered",
        reconstruction / "sfm.ply",
        reconstruction / "sfm_filtered.ply",
        reconstruction / "mvs.ply",
        reconstruction / "mvs_filtered.ply",
        dataset / "sam6d_shared",
        dataset / "experimental_multi_instance/ism_candidate_audit",
    ]
    sequences = dataset / "sam6d_sequences"
    if sequences.is_dir():
        for object_dir in sequences.iterdir():
            if object_dir.is_dir():
                result.extend((object_dir / "template", object_dir / "frames"))
    pem = dataset / "experimental_multi_instance/pem"
    if pem.is_dir():
        for object_dir in pem.iterdir():
            if object_dir.is_dir():
                result.extend((object_dir / "frames", object_dir / "visualizations"))
    return [path for path in result if path.exists()]


def validate_finals(dataset: Path) -> None:
    required = [
        dataset / "reconstruction/mvs_filtered_metric.ply",
        dataset / "reconstruction/metric_alignment/camera_poses_metric.json",
        dataset / "sam6d_scene/annotations.json",
    ]
    missing = [path for path in required if not path.is_file()]
    world_poses = list((dataset / "sam6d_sequences").glob("*/*_world_pose.json"))
    if not world_poses:
        missing.append(dataset / "sam6d_sequences/<object>/*_world_pose.json")
    if missing:
        raise RuntimeError(
            "Cleanup refused because final outputs are incomplete: "
            + ", ".join(str(path) for path in missing)
        )


def evenly_spaced(items: list[tuple[int, Path, Path]], limit: int):
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indices = {
        round(position * (len(items) - 1) / (limit - 1))
        for position in range(limit)
    }
    return [items[index] for index in sorted(indices)]


def preserve_sam6d_review_images(dataset: Path, limit: int) -> dict[str, int]:
    """Keep a small, evenly distributed ISM/PEM review set before cleanup."""
    preserved: dict[str, int] = {}
    sequences = dataset / "sam6d_sequences"
    review_root = dataset / "sam6d_scene/review/sam6d"
    if not sequences.is_dir():
        return preserved
    for object_dir in sorted(path for path in sequences.iterdir() if path.is_dir()):
        pairs: list[tuple[int, Path, Path]] = []
        for frame_dir in sorted((object_dir / "frames").glob("*")):
            if not frame_dir.is_dir() or not frame_dir.name.isdigit():
                continue
            results = frame_dir / "sam6d_results"
            ism = results / "vis_ism.png"
            pem = results / "vis_pem.png"
            if not pem.is_file():
                pem = (
                    dataset / "experimental_multi_instance/pem"
                    / object_dir.name / "frames" / frame_dir.name
                    / "sam6d_results" / "vis_pem.png"
                )
            if ism.is_file() and pem.is_file():
                pairs.append((int(frame_dir.name), ism, pem))
        if not pairs:
            continue
        destination = review_root / object_dir.name
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        selected = evenly_spaced(pairs, limit)
        for frame, ism, pem in selected:
            frame_dir = destination / f"{frame:06d}"
            frame_dir.mkdir()
            shutil.copy2(ism, frame_dir / "vis_ism.png")
            shutil.copy2(pem, frame_dir / "vis_pem.png")
        preserved[object_dir.name] = len(selected)
    return preserved


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--review-image-limit", type=int, default=REVIEW_IMAGE_LIMIT,
        help="Maximum number of retained ISM/PEM image pairs per object.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = args.config.expanduser().resolve()
    payload = yaml.safe_load(config.read_text()) or {}
    dataset = (args.dataset_dir.expanduser().resolve() if args.dataset_dir else
               dataset_path(payload["path"], payload["dataset_id"]))
    if args.review_image_limit < 1:
        raise ValueError("--review-image-limit must be at least 1")
    validate_finals(dataset)
    removable = targets(dataset)
    entries = [{"path": str(path.relative_to(dataset)), "bytes": size_bytes(path)}
               for path in removable]
    total = sum(entry["bytes"] for entry in entries)
    preserved_review_images = {}
    if args.apply:
        preserved_review_images = preserve_sam6d_review_images(
            dataset, args.review_image_limit
        )
        for path in removable:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    report = {
        "format": "label_factory_cleanup_v1",
        "dataset": str(dataset),
        "applied": bool(args.apply),
        "removed_bytes": total if args.apply else 0,
        "reclaimable_bytes": total,
        "entries": entries,
        "preserved_sam6d_review_pairs": preserved_review_images,
    }
    report_path = dataset / "publishing/cleanup_report.json"
    if args.apply:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    action = "Removed" if args.apply else "Can remove"
    print(f"{action} {len(entries)} intermediate paths ({total / 1073741824:.2f} GiB)")
    if args.apply:
        print(f"Cleanup report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
