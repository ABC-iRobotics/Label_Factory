#!/usr/bin/env python3
"""Run PEM on de-duplicated ISM candidates in an isolated experiment tree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from audit_ism_candidates import COLORS, decode_rle
from dataset_config import dataset_path


def dataset_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text()) or {}
    return dataset_path(config["path"], config["dataset_id"])


def prepare_object(dataset: Path, object_name: str, audit: dict,
                   overwrite: bool) -> tuple[Path, int]:
    source_manifest_path = (
        dataset / "sam6d_sequences" / object_name / f"{object_name}_batch_manifest.json"
    )
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_frames = {int(frame["frame_id"]): frame for frame in source_manifest["frames"]}
    experiment_root = dataset / "experimental_multi_instance/pem" / object_name
    frames = []
    candidate_count = 0
    for audited_frame in audit["frames"]:
        identifier = int(audited_frame["frame_id"])
        source_frame = source_frames.get(identifier)
        if source_frame is None:
            continue
        source_detections = json.loads(Path(audited_frame["source"]).read_text())
        filtered = []
        for retained_index, candidate in enumerate(audited_frame["retained"]):
            source_index = int(candidate["source_index"])
            detection = dict(source_detections[source_index])
            detection["experimental_candidate_id"] = (
                f"{object_name}:{identifier:06d}:{source_index:04d}"
            )
            detection["experimental_retained_index"] = retained_index
            detection["ism_score"] = float(detection["score"])
            filtered.append(detection)
        frame_root = experiment_root / "frames" / f"{identifier:06d}"
        results_root = frame_root / "sam6d_results"
        results_root.mkdir(parents=True, exist_ok=True)
        ism_path = results_root / "detection_ism_candidates.json"
        pem_path = results_root / "detection_pem_candidates.json"
        ism_path.write_text(json.dumps(filtered) + "\n")
        # SAM6D_batch_stage writes detection_pem.json. Preserve that convention
        # inside the isolated frame and rename it after the worker exits.
        worker_pem_path = results_root / "detection_pem.json"
        existing_candidate_ids = []
        if pem_path.is_file():
            try:
                existing_candidate_ids = [
                    item.get("experimental_candidate_id")
                    for item in json.loads(pem_path.read_text())
                ]
            except (OSError, json.JSONDecodeError):
                existing_candidate_ids = []
        requested_candidate_ids = [
            item["experimental_candidate_id"] for item in filtered
        ]
        pose_visualization_path = results_root / "vis_pem.png"
        run_pem = bool(filtered) and (
            overwrite or existing_candidate_ids != requested_candidate_ids
            or not pose_visualization_path.is_file()
        )
        if (run_pem or not filtered) and pem_path.is_file():
            pem_path.unlink()
        if not filtered and worker_pem_path.is_file():
            worker_pem_path.unlink()
        frame = dict(source_frame)
        frame.update({
            "output_dir": str(frame_root),
            "ism_json": str(ism_path),
            "pem_json": str(worker_pem_path),
            "run_ism": False,
            "run_pem": run_pem,
        })
        frames.append(frame)
        candidate_count += len(filtered)
    manifest = dict(source_manifest)
    manifest.update({
        "dataset_dir": str(dataset),
        "det_score_thresh": 0.0,
        # Produce the actual CAD pose projection for the best retained
        # candidate. The separate candidate diagnostic remains available in
        # <object>/visualizations, but must not be presented as vis_pem.
        "skip_visualizations": False,
        "experimental_multi_instance": True,
        "source_audit": str(
            dataset / "experimental_multi_instance/ism_candidate_audit/summary.json"
        ),
        "frames": frames,
    })
    manifest_path = experiment_root / "experimental_pem_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, candidate_count


def visualize_pose_candidates(rgb_path: Path, poses: list[dict],
                              output_path: Path) -> None:
    image = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
    for index, pose in enumerate(poses):
        mask = decode_rle(pose["segmentation"])
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
        image[mask] = image[mask] * 0.48 + color * 0.52
    rendered = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(rendered)
    for index, pose in enumerate(poses):
        bbox = pose.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, width, height = map(float, bbox)
        color = COLORS[index % len(COLORS)]
        candidate_id = pose.get("experimental_candidate_id", f"candidate-{index + 1}")
        short_id = candidate_id.rsplit(":", 1)[-1]
        label = f"#{short_id} pose={float(pose.get('score', 0.0)):.3f}"
        draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
        draw.text((x + 4, max(0, y - 15)), label, fill=color,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)


def finalize_outputs(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    frame_summaries = []
    totals = {"input_candidates": 0, "pose_hypotheses": 0, "skipped_candidates": 0}
    for frame in manifest["frames"]:
        results = Path(frame["output_dir"]) / "sam6d_results"
        ism_path = Path(frame["ism_json"])
        worker_path = results / "detection_pem.json"
        final_path = results / "detection_pem_candidates.json"
        if worker_path.is_file():
            worker_path.replace(final_path)
        inputs = json.loads(ism_path.read_text()) if ism_path.is_file() else []
        poses = json.loads(final_path.read_text()) if final_path.is_file() else []
        visualization_path = (
            manifest_path.parent / "visualizations" /
            f"{int(frame['frame_id']):06d}.png"
        )
        if poses:
            visualize_pose_candidates(
                Path(frame["rgb_path"]), poses, visualization_path
            )
        totals["input_candidates"] += len(inputs)
        totals["pose_hypotheses"] += len(poses)
        totals["skipped_candidates"] += max(0, len(inputs) - len(poses))
        frame_summaries.append({
            "frame_id": int(frame["frame_id"]),
            "input_candidates": len(inputs),
            "pose_hypotheses": len(poses),
            "skipped_candidates": max(0, len(inputs) - len(poses)),
            "ism_candidates": str(ism_path),
            "pem_candidates": str(final_path),
            "visualization": str(visualization_path) if poses else None,
            "candidate_ids": [pose.get("experimental_candidate_id") for pose in poses],
            "pose_scores": [float(pose.get("score", 0.0)) for pose in poses],
        })
    summary = {
        "format": "label_factory_experimental_multi_instance_pem_v1",
        "experimental": True,
        "changes_current_pipeline": False,
        "object_name": manifest["object_name"],
        "totals": totals,
        "frames": frame_summaries,
    }
    summary_path = manifest_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--object", action="append", dest="objects",
                        help="Object name; repeat as needed. Default: all audited objects.")
    parser.add_argument("--python", type=Path,
                        default=Path(os.environ.get("SAM6D_PYTHON", sys.executable)),
                        help="Python interpreter containing the SAM-6D PEM environment.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = dataset_from_config(args.config.expanduser().resolve())
    audit_path = dataset / "experimental_multi_instance/ism_candidate_audit/summary.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Missing {audit_path}. Run audit_ism_candidates.py first."
        )
    audit = json.loads(audit_path.read_text())
    object_names = args.objects or sorted(audit["objects"])
    worker = project / "scripts/SAM6D_batch_stage.py"
    model_dir = project / "external/SAM-6D/SAM-6D/Pose_Estimation_Model"
    for object_name in object_names:
        if object_name not in audit["objects"]:
            raise KeyError(f"Object {object_name!r} is not present in the ISM audit")
        manifest_path, candidates = prepare_object(
            dataset, object_name, audit["objects"][object_name], args.overwrite
        )
        print(f"{object_name}: prepared {candidates} retained ISM candidates")
        if args.prepare_only:
            print(f"Manifest: {manifest_path}")
            continue
        command = [
            str(args.python.expanduser().resolve()), str(worker), "--stage", "pem",
            "--manifest", str(manifest_path), "--model-dir", str(model_dir),
        ]
        print("Running one persistent PEM worker for " + object_name, flush=True)
        subprocess.run(command, cwd=project, check=True)
        summary = finalize_outputs(manifest_path)
        totals = summary["totals"]
        print(
            f"{object_name}: {totals['pose_hypotheses']}/"
            f"{totals['input_candidates']} candidates produced poses; "
            f"{totals['skipped_candidates']} lacked usable depth support"
        )
        print(f"Summary: {manifest_path.parent / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
