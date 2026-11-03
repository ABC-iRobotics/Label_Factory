#!/usr/bin/env python3
"""Aggregate experimental per-frame PEM hypotheses for isolated pose fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dataset_config import dataset_path


def dataset_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text()) or {}
    return dataset_path(config["path"], config["dataset_id"])


def best_pose(candidates: list[dict]) -> dict | None:
    valid = [item for item in candidates if "R" in item and "t" in item]
    return max(valid, key=lambda item: float(item.get("score", 0.0))) if valid else None


def aggregate_object(dataset: Path, object_name: str,
                     require_poses: bool) -> tuple[Path, dict]:
    experiment_root = dataset / "experimental_multi_instance/pem" / object_name
    manifest_path = experiment_root / "experimental_pem_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run run_experimental_multi_instance_pem.py first."
        )
    manifest = json.loads(manifest_path.read_text())
    frames = []
    totals = {
        "frames": 0,
        "frames_with_candidates": 0,
        "frames_with_poses": 0,
        "input_candidates": 0,
        "pose_candidates": 0,
        "candidates_without_pose": 0,
    }
    for source_frame in manifest["frames"]:
        identifier = int(source_frame["frame_id"])
        results_dir = Path(source_frame["output_dir"]) / "sam6d_results"
        ism_path = Path(source_frame["ism_json"])
        final_pem_path = results_dir / "detection_pem_candidates.json"
        worker_pem_path = results_dir / "detection_pem.json"
        pem_path = final_pem_path if final_pem_path.is_file() else worker_pem_path
        ism_candidates = json.loads(ism_path.read_text()) if ism_path.is_file() else []
        pose_candidates = json.loads(pem_path.read_text()) if pem_path.is_file() else []
        pose_candidates = [
            item for item in pose_candidates
            if isinstance(item, dict) and "R" in item and "t" in item
        ]
        candidate_ids = [
            item.get("experimental_candidate_id") for item in pose_candidates
        ]
        status = (
            "ok" if pose_candidates else
            "pending_pem" if ism_candidates and not pem_path.is_file() else
            "no_depth_supported_pose" if ism_candidates else
            "no_ism_candidate"
        )
        record = {
            "frame_id": identifier,
            "image_path": source_frame["rgb_path"],
            "depth_path": source_frame["depth_path"],
            "status": status,
            "detection_count": len(pose_candidates),
            "input_candidate_count": len(ism_candidates),
            "pose_candidates": pose_candidates,
            "best_pose": best_pose(pose_candidates),
            "pem_result_path": str(pem_path) if pem_path.is_file() else None,
            "experimental_ism_path": str(ism_path),
            "experimental_candidate_ids": candidate_ids,
        }
        visualization = experiment_root / "visualizations" / f"{identifier:06d}.png"
        if visualization.is_file():
            record["pem_visualization"] = str(visualization)
        frames.append(record)
        totals["frames"] += 1
        totals["frames_with_candidates"] += int(bool(ism_candidates))
        totals["frames_with_poses"] += int(bool(pose_candidates))
        totals["input_candidates"] += len(ism_candidates)
        totals["pose_candidates"] += len(pose_candidates)
        totals["candidates_without_pose"] += max(
            0, len(ism_candidates) - len(pose_candidates)
        )

    if require_poses and totals["pose_candidates"] == 0:
        raise RuntimeError(
            f"No experimental PEM poses exist for {object_name}. Run the experimental "
            "PEM stage without --prepare-only first."
        )
    aggregate = {
        "format": "label_factory_experimental_multi_instance_poses_per_frame_v1",
        "experimental": True,
        "changes_current_pipeline": False,
        "object_name": object_name,
        "object_identity": manifest.get("object_identity"),
        "config_path": manifest.get("config_path"),
        "dataset_dir": str(dataset),
        "cad_path": manifest["cad_path"],
        "camera_path": next(
            (frame["camera_path"] for frame in manifest["frames"]), None
        ),
        "depth_unit": "millimetre",
        "source_audit": manifest.get("source_audit"),
        "experimental_pem_manifest": str(manifest_path),
        "statistics": totals,
        "frames": frames,
    }
    output_path = experiment_root / f"{object_name}_poses_per_frame.json"
    output_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    return output_path, totals


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--object", action="append", dest="objects",
                        help="Object name; repeat as needed. Default: all prepared objects.")
    parser.add_argument(
        "--require-poses", action="store_true",
        help="Fail instead of writing a pending aggregate when PEM has not run.",
    )
    args = parser.parse_args()
    dataset = dataset_from_config(args.config.expanduser().resolve())
    pem_root = dataset / "experimental_multi_instance/pem"
    if not pem_root.is_dir():
        raise FileNotFoundError(
            f"Missing {pem_root}. Run run_experimental_multi_instance_pem.py first."
        )
    object_names = args.objects or sorted(
        path.name for path in pem_root.iterdir()
        if path.is_dir() and (path / "experimental_pem_manifest.json").is_file()
    )
    if not object_names:
        raise RuntimeError("No prepared experimental PEM objects were found")
    for object_name in object_names:
        output_path, totals = aggregate_object(
            dataset, object_name, args.require_poses
        )
        print(
            f"{object_name}: {totals['pose_candidates']}/"
            f"{totals['input_candidates']} candidates have PEM poses across "
            f"{totals['frames_with_poses']}/{totals['frames']} frames"
        )
        print(f"Aggregate: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
