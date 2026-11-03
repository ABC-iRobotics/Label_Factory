#!/usr/bin/env python3
"""Fit world poses from experimental multi-instance PEM aggregates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

import stage_timing
from dataset_config import dataset_path


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--object-name")
    parser.add_argument("--translation-threshold", type=float, default=0.01)
    parser.add_argument("--rotation-threshold-deg", type=float, default=5.0)
    parser.add_argument("--maximum-rotation-threshold-deg", type=float, default=30.0)
    parser.add_argument("--minimum-score", type=float, default=0.0)
    parser.add_argument("--minimum-inliers", type=int, default=3)
    parser.add_argument("--minimum-instance-frame-fraction", type=float, default=0.05)
    parser.add_argument("--maximum-instances", type=int, default=0)
    parser.add_argument("--symmetry-mode", choices=("auto", "none"), default="auto")
    parser.add_argument("--min-silhouette-iou", type=float, default=None,
                        help="Reject pose clusters whose CAD silhouette overlaps the observed "
                             "masks with a median IoU below this value (0 = off; default: "
                             "LABEL_FACTORY_MIN_SILHOUETTE_IOU or 0).")
    parser.add_argument("--low-support-fallback", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Estimate a pose from the best cluster of objects seen in "
                             "too few frames instead of excluding them.")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text()) or {}
    dataset = dataset_path(config["path"], config["dataset_id"])
    started = time.monotonic()
    pem_root = dataset / "experimental_multi_instance/pem"
    object_names = [args.object_name] if args.object_name else sorted(
        path.name for path in pem_root.iterdir()
        if path.is_dir() and (path / f"{path.name}_poses_per_frame.json").is_file()
    )
    if not object_names:
        raise RuntimeError(f"No experimental pose aggregates found under {pem_root}")
    fitter = project / "scripts/RANSAC_object_pose.py"
    failures = []
    found = {}
    low_support = []
    for index, object_name in enumerate(object_names, 1):
        poses = pem_root / object_name / f"{object_name}_poses_per_frame.json"
        output = dataset / "sam6d_sequences" / object_name / f"{object_name}_world_pose.json"
        command = [
            sys.executable, fitter, "--config", config_path,
            "--object-name", object_name, "--poses", poses, "--output", output,
            "--translation-threshold", args.translation_threshold,
            "--rotation-threshold-deg", args.rotation_threshold_deg,
            "--maximum-rotation-threshold-deg", args.maximum_rotation_threshold_deg,
            "--minimum-score", args.minimum_score,
            "--minimum-inliers", args.minimum_inliers,
            "--minimum-instance-frame-fraction", args.minimum_instance_frame_fraction,
            "--maximum-instances", args.maximum_instances,
            "--symmetry-mode", args.symmetry_mode,
            "--low-support-fallback" if args.low_support_fallback
            else "--no-low-support-fallback",
        ]
        if args.min_silhouette_iou is not None:
            command += ["--min-silhouette-iou", args.min_silhouette_iou]
        print(f"=== Multi-instance pose fitting [{index}/{len(object_names)}]: "
              f"{object_name} ===", flush=True)
        try:
            subprocess.run(list(map(str, command)), cwd=project, check=True)
            payload = json.loads(output.read_text())
            found[object_name] = int(
                payload.get("instance_count", len(payload.get("instances", [])) or 1)
            )
            if payload.get("low_support"):
                low_support.append(object_name)
        except subprocess.CalledProcessError as error:
            failures.append((object_name, error.returncode))
    summary = {
        "format": "label_factory_multi_instance_pose_fitting_summary_v1",
        "objects_requested": object_names,
        "found_objects": found,
        "non_found_objects": [name for name, _ in failures],
        "objects_succeeded": list(found),
        "low_support_objects": low_support,
        "objects_excluded": [
            {"object_name": name, "reason": f"pose fitting exited with code {code}"}
            for name, code in failures
        ],
    }
    summary_path = dataset / "sam6d_sequences/ransac_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("\nFound objects:")
    if found:
        for name, count in found.items():
            print(f"{name}: {count}")
    else:
        print("None")
    if low_support:
        print("\nLow-support objects (seen in few frames; pose is less reliable):")
        for name in low_support:
            print(name)
    print("\nNon-found objects:")
    if failures:
        for name, _ in failures:
            print(name)
    else:
        print("None")
    stage_timing.record(dataset, "consolidation", time.monotonic() - started,
                        objects=len(object_names))
    print(f"\nPose fitting summary: {summary_path}")
    if failures and found:
        print(
            f"WARNING: Pose fitting found {len(found)}/{len(object_names)} object "
            "types. Scene annotation will exclude the non-found objects."
        )
    elif failures:
        names = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"Pose fitting found no objects: {names}")
    else:
        print("Multi-instance pose fitting completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
