#!/usr/bin/env python3
"""Run the custom SAM-6D ISM and PEM pipelines over an RGB-D sequence.

RGB frames are paired with depth maps by the integer in their filenames, e.g.
``images/0007.png`` pairs with ``depth_maps/depth_000007.png``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import sys
from pathlib import Path
from typing import Iterable

import yaml

import stage_timing
from dataset_config import dataset_path
from textured_models import combined_sha256, find_textured_model


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def frame_id(path: Path) -> int:
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Filename does not contain a frame number: {path}")
    return int(numbers[-1])


def indexed_images(folder: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index = frame_id(path)
            if index in result:
                raise ValueError(
                    f"Duplicate frame ID {index}: {result[index]} and {path}"
                )
            result[index] = path
    return result


def safe_object_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    name = name.strip("._-")
    if not name:
        raise ValueError("Object name must contain at least one usable character")
    return name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_ism_signature(
    rgb_path: Path,
    segmentor_model: str,
    stability_score_thresh: float,
) -> dict:
    stat = rgb_path.stat()
    return {
        "version": 1,
        "rgb_path": str(rgb_path.resolve()),
        "rgb_size": stat.st_size,
        "rgb_mtime_ns": stat.st_mtime_ns,
        "segmentor_model": segmentor_model,
        "stability_score_thresh": stability_score_thresh,
    }


def valid_shared_ism_cache(cache_path: Path, metadata_path: Path, signature: dict) -> bool:
    if not cache_path.is_file() or not metadata_path.is_file():
        return False
    try:
        return json.loads(metadata_path.read_text()) == signature
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def templates_identity(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in path.iterdir()
        if item.is_file()
        and (
            item.name.startswith("rgb_")
            or item.name.startswith("mask_")
            or item.name.startswith("xyz_")
        )
    )
    if not files:
        raise RuntimeError(f"No rendered templates found in {path}")
    for item in files:
        stat = item.stat()
        digest.update(item.name.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def discover_stl_objects(dataset: Path) -> list[Path]:
    objects = sorted(
        (dataset / "objects" / "stl").glob("*.stl"),
        key=lambda path: path.stem.lower(),
    )
    if not objects:
        raise RuntimeError(
            f"No STL objects found in {dataset / 'objects' / 'stl'}"
        )
    return objects


def check_dataset_inputs(dataset: Path) -> None:
    rgb_dir = dataset / "images"
    depth_dir = dataset / "depth_maps"
    camera_path = dataset / "camera.json"
    for required in (rgb_dir, depth_dir, camera_path, dataset / "objects/stl"):
        if not required.exists():
            raise FileNotFoundError(required)
    camera = json.loads(camera_path.read_text())
    if not isinstance(camera.get("cam_K"), list) or len(camera["cam_K"]) != 9:
        raise ValueError(f"Invalid cam_K in {camera_path}")
    rgb = indexed_images(rgb_dir)
    depth = indexed_images(depth_dir)
    paired = sorted(set(rgb) & set(depth))
    if not paired:
        raise RuntimeError("No paired RGB/depth frames were found")
    objects = discover_stl_objects(dataset)
    print(f"Dataset: {dataset}")
    print(f"RGB frames: {len(rgb)} ({rgb_dir})")
    print(f"Depth maps: {len(depth)} ({depth_dir})")
    print(f"Paired RGB-D frames: {len(paired)}")
    print(f"Camera intrinsics: {camera_path}")
    print("CAD objects: " + ", ".join(path.name for path in objects))

    reconstruction = dataset / "reconstruction"
    world_inputs = (
        reconstruction / "sfm_filtered/images.bin",
        reconstruction / "sfm_filtered/cameras.bin",
        reconstruction / "metric_alignment/scale.npy",
        reconstruction / "metric_alignment/rotation.npy",
        reconstruction / "metric_alignment/translation.npy",
        reconstruction / "metric_alignment/camera_poses_metric.json",
        reconstruction / "mvs_filtered_metric.ply",
    )
    missing = [path for path in world_inputs if not path.is_file()]
    if missing:
        print("SAM-6D per-frame inference: READY")
        print("World-pose handoff: INCOMPLETE")
        for path in missing:
            print(f"  missing: {path}")
    else:
        print("SAM-6D per-frame inference: READY")
        print("World-pose handoff: READY")


def ensure_temporary_templates(
    project: Path,
    worker_python: Path,
    cad_path: Path,
    object_name: str,
    templates_dir: Path,
    render_model: Path | None = None,
    model_identity: str | None = None,
    timing_dataset: Path | None = None,
) -> None:
    # render_model is the textured OBJ when one exists, otherwise the STL; the
    # identity covers geometry and texture so adding a texture re-renders.
    render_model = render_model or cad_path
    cad_digest = model_identity or file_sha256(cad_path)
    example_output = templates_dir.parent
    marker_path = example_output / "template_object.json"
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        marker = {}
    if (
        marker.get("cad_sha256") == cad_digest
        and marker.get("object_name") == object_name
        and templates_dir.is_dir()
        and len(list(templates_dir.glob("xyz_*.npy"))) > 0
    ):
        print(f"Reusing object templates for {object_name}: {templates_dir}")
        return

    if templates_dir.exists():
        shutil.rmtree(templates_dir)
    blenderproc = worker_python.parent / "blenderproc"
    if not blenderproc.is_file():
        raise FileNotFoundError(
            f"blenderproc was not found in the sam6d environment: {blenderproc}"
        )
    blender_path = Path(
        os.environ.get(
            "SAM6D_BLENDER_PATH",
            str(Path.home() / "blender-3.3.1-linux-x64"),
        )
    ).expanduser()
    if not blender_path.is_dir():
        raise FileNotFoundError(
            f"Blender directory not found: {blender_path}. Set SAM6D_BLENDER_PATH."
        )
    render_dir = (
        project / "external" / "SAM-6D" / "SAM-6D" / "Render"
    )
    print(
        f"Rendering object templates for {object_name}: {templates_dir}",
        flush=True,
    )
    render_started = time.monotonic()
    run(
        [
            blenderproc,
            "run",
            "--custom-blender-path",
            blender_path,
            render_dir / "render_custom_templates.py",
            "--output_dir",
            example_output,
            "--templates_dir",
            templates_dir,
            "--cad_path",
            render_model,
        ],
        render_dir,
    )
    if timing_dataset is not None:
        stage_timing.record(
            timing_dataset, "sam6d_templates", time.monotonic() - render_started,
            item=object_name,
        )
    marker_path.write_text(
        json.dumps(
            {
                "object_name": object_name,
                "cad_path": str(cad_path),
                "cad_sha256": cad_digest,
                "render_model": str(render_model),
                "textured": render_model != cad_path,
            },
            indent=2,
        )
        + "\n"
    )


def dataset_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    return dataset_path(config["path"], config["dataset_id"])


def run(command: Iterable[str], cwd: Path) -> None:
    printable = " ".join(map(str, command))
    print(f"\n$ {printable}", flush=True)
    environment = os.environ.copy()
    # A stale PYTHONPATH can make a Conda interpreter import binary packages
    # from another virtual environment, which is unsafe for PyTorch.
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        list(map(str, command)),
        cwd=cwd,
        check=True,
        env=environment,
    )


def resolve_worker_python(requested: str | None) -> Path:
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            resolved = shutil.which(requested)
            if resolved is None:
                raise FileNotFoundError(f"Python interpreter not found: {requested}")
            path = Path(resolved)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()

    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix).name == "sam6d":
        candidates.append(Path(conda_prefix) / "bin" / "python")
    candidates.extend(
        [
            Path.home() / "anaconda3" / "envs" / "sam6d" / "bin" / "python",
            Path.home() / "miniconda3" / "envs" / "sam6d" / "bin" / "python",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "Could not find the sam6d Python environment. Activate it first or "
        "pass --python /path/to/sam6d/bin/python."
    )


def runtime_camera_file(camera_path: Path, output_dir: Path) -> Path:
    """Create the camera file expected for uint16 millimetre depth PNGs."""
    camera = json.loads(camera_path.read_text())
    camera["depth_scale"] = 1.0
    path = output_dir / "camera_depth_mm.json"
    path.write_text(json.dumps(camera, indent=2) + "\n")
    return path


def best_pose(detections: list[dict]) -> dict | None:
    valid = [
        detection
        for detection in detections
        if "R" in detection
        and "t" in detection
        and float(detection.get("score", 0.0)) > 0.0
    ]
    return max(valid, key=lambda item: float(item["score"])) if valid else None


def finalize_manifest(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    sequence: list[dict] = []
    for frame in manifest["frames"]:
        idx = frame["frame_id"]
        pem_json = Path(frame["pem_json"])
        results_dir = pem_json.parent
        if not pem_json.is_file():
            raise RuntimeError(f"PEM did not create expected result: {pem_json}")
        detections = json.loads(pem_json.read_text())
        pose = best_pose(detections)
        record = {
            "frame_id": idx,
            "image_path": frame["rgb_path"],
            "depth_path": frame["depth_path"],
            "status": "ok" if pose else "no_pose",
            "detection_count": len(detections),
            "pose_candidates": detections,
            "best_pose": pose,
            "pem_result_path": str(pem_json),
        }
        if not manifest["skip_visualizations"]:
            record["shared_ism_visualization"] = frame[
                "shared_ism_visualization"
            ]
            record["ism_visualization"] = str(results_dir / "vis_ism.png")
            record["pem_visualization"] = str(results_dir / "vis_pem.png")
        sequence.append(record)

    aggregate = {
        "object_name": manifest["object_name"],
        "object_identity": manifest["object_identity"],
        "config_path": manifest["config_path"],
        "dataset_dir": manifest["dataset_dir"],
        "cad_path": manifest["cad_path"],
        "camera_path": manifest["frames"][0]["camera_path"],
        "depth_unit": "millimetre",
        "frames": sequence,
    }
    output = manifest_path.parent
    poses_path = output / f"{manifest['object_name']}_poses_per_frame.json"
    poses_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(f"Saved {len(sequence)} frame results to {poses_path}")
    return poses_path


def valid_ism_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        detections = json.loads(path.read_text())
        return isinstance(detections, list) and all(
            len(detection["segmentation"]["size"]) == 2
            for detection in detections
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project / "scripts" / "configs" / "default.yaml",
        help="Project config containing path and dataset_id.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Optional override for the dataset derived from --config.",
    )
    parser.add_argument(
        "--cad-path",
        type=Path,
        help="Optional CAD override; defaults to objects/stl/<object-name>.stl.",
    )
    parser.add_argument(
        "--texture",
        choices=("auto", "off"),
        default=os.environ.get("LABEL_FACTORY_TEXTURE", "auto"),
        help=(
            "auto (default): render templates from objects/textured/<object>.obj "
            "when it exists and is complete, otherwise from the STL. off: always "
            "use the untextured STL (also settable with LABEL_FACTORY_TEXTURE=off)."
        ),
    )
    parser.add_argument(
        "--object-name",
        help="Output name; defaults to the CAD filename stem.",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        help=(
            "Rendered templates; defaults to "
            "data/####/sam6d_sequences/<object_name>/template."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--python",
        help="Python from the sam6d environment; detected automatically when omitted.",
    )
    parser.add_argument("--segmentor-model", choices=("sam", "fastsam"), default="sam")
    parser.add_argument("--stability-score-thresh", type=float, default=0.85)
    parser.add_argument("--det-score-thresh", type=float, default=0.2)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate resolved RGB-D, CAD, reconstruction, and alignment paths, then exit.",
    )
    parser.add_argument("--single-object", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--finalize-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--template-workers", type=int,
        default=int(os.environ.get("SAM6D_TEMPLATE_WORKERS", "1")),
        help="Concurrent per-object BlenderProc template jobs (default: 1).",
    )
    parser.add_argument(
        "--ism-only", action="store_true",
        help="Run object-specific ISM and stop before PEM/pose aggregation.",
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Delete neither files nor results; only suppress copying visualization paths into the index.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.template_workers < 1:
        raise ValueError("--template-workers must be at least 1")
    worker_python = resolve_worker_python(args.python)
    project = Path(__file__).resolve().parents[1]
    sam6d = project / "external" / "SAM-6D" / "SAM-6D"
    ism_dir = sam6d / "Instance_Segmentation_Model"
    pem_dir = sam6d / "Pose_Estimation_Model"

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    dataset = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir
        else dataset_from_config(config_path)
    )
    if args.check_inputs:
        check_dataset_inputs(dataset)
        return 0
    if not args.cad_path and not args.object_name and not args.single_object:
        if args.output_dir:
            raise ValueError(
                "--output-dir cannot be shared by all objects; omit it to use "
                "sam6d_sequences/<STL-name>."
            )
        objects = discover_stl_objects(dataset)
        print(
            "Discovered STL objects: "
            + ", ".join(path.stem for path in objects)
        )
        manifests = []
        preparation_jobs = []
        for number, object_path in enumerate(objects, 1):
            normalized_name = safe_object_name(object_path.stem)
            print(
                f"\n=== Object [{number}/{len(objects)}]: "
                f"{object_path.stem} ===",
                flush=True,
            )
            command = [
                sys.executable,
                Path(__file__).resolve(),
                "--config",
                config_path,
                "--object-name",
                normalized_name,
                "--python",
                worker_python,
                "--segmentor-model",
                args.segmentor_model,
                "--stability-score-thresh",
                str(args.stability_score_thresh),
                "--det-score-thresh",
                str(args.det_score_thresh),
                "--single-object",
                "--prepare-only",
            ]
            if args.dataset_dir:
                command.extend(["--dataset-dir", dataset])
            if args.templates_dir:
                command.extend(["--templates-dir", args.templates_dir])
            if args.start_frame is not None:
                command.extend(["--start-frame", str(args.start_frame)])
            if args.end_frame is not None:
                command.extend(["--end-frame", str(args.end_frame)])
            if args.overwrite:
                command.append("--overwrite")
            if args.skip_visualizations:
                command.append("--skip-visualizations")
            if args.ism_only:
                command.append("--ism-only")
            manifest_path = (
                dataset / "sam6d_sequences" / normalized_name
                / f"{normalized_name}_batch_manifest.json"
            )
            manifests.append(manifest_path)
            preparation_jobs.append((object_path.stem, command))

        print(
            f"Preparing templates with {min(args.template_workers, len(preparation_jobs))} "
            "parallel BlenderProc worker(s)", flush=True,
        )
        with ThreadPoolExecutor(
            max_workers=min(args.template_workers, len(preparation_jobs))
        ) as executor:
            futures = {
                executor.submit(run, command, project): name
                for name, command in preparation_jobs
            }
            for future in as_completed(futures):
                name = futures[future]
                future.result()
                print(f"Template preparation complete: {name}", flush=True)

        stage_worker = project / "scripts" / "SAM6D_batch_stage.py"
        ism_command = [
            worker_python, stage_worker, "--stage", "ism", "--model-dir", ism_dir
        ]
        for manifest_path in manifests:
            ism_command.extend(["--manifest", manifest_path])
        run(ism_command, project)
        if args.ism_only:
            print("All object-specific ISM sequences are complete")
            return 0

        pem_command = [
            worker_python, stage_worker, "--stage", "pem", "--model-dir", pem_dir
        ]
        for manifest_path in manifests:
            pem_command.extend(["--manifest", manifest_path])
        run(pem_command, project)
        for manifest_path in manifests:
            finalize_manifest(manifest_path)
        return 0

    rgb_dir = dataset / "images"
    depth_dir = dataset / "depth_maps"
    camera_path = dataset / "camera.json"
    reconstruction_dir = dataset / "reconstruction"
    sfm_model = reconstruction_dir / "sfm_filtered"
    metric_alignment = reconstruction_dir / "metric_alignment"
    metric_cloud = reconstruction_dir / "mvs_filtered_metric.ply"
    if args.cad_path:
        cad_path = args.cad_path.expanduser().resolve()
        object_name = safe_object_name(args.object_name or cad_path.stem)
    else:
        if not args.object_name:
            available = sorted(
                path.stem for path in (dataset / "objects" / "stl").glob("*.stl")
            )
            raise ValueError(
                "Specify --object-name. Available STL objects: "
                + ", ".join(available)
            )
        object_name = safe_object_name(args.object_name)
        cad_path = (
            dataset / "objects" / "stl" / f"{object_name}.stl"
        ).resolve()
    templates_dir = (
        args.templates_dir.expanduser().resolve()
        if args.templates_dir
        else (dataset / "sam6d_sequences" / object_name / "template").resolve()
    )
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else dataset / "sam6d_sequences" / object_name
    )

    manifest_path = output / f"{object_name}_batch_manifest.json"
    if args.finalize_only:
        finalize_manifest(manifest_path)
        return 0

    for required in (rgb_dir, depth_dir, camera_path, cad_path):
        if not required.exists():
            raise FileNotFoundError(required)

    rgb_frames = indexed_images(rgb_dir)
    depth_frames = indexed_images(depth_dir)
    common_ids = sorted(set(rgb_frames) & set(depth_frames))
    sam6d_selection_path = dataset / "sam6d_frame_selection.json"
    reconstruction_selection_path = dataset / "reconstruction_frame_selection.json"
    selection_path = (
        sam6d_selection_path
        if sam6d_selection_path.is_file()
        else reconstruction_selection_path
    )
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        selected_names = selection.get("image_names", [])
        selected_ids = {frame_id(Path(name)) for name in selected_names}
        common_ids = [idx for idx in common_ids if idx in selected_ids]
        print(
            f"SAM-6D keyframe selection: {len(common_ids)} frames from "
            f"{selection_path}"
        )
    if args.start_frame is not None:
        common_ids = [idx for idx in common_ids if idx >= args.start_frame]
    if args.end_frame is not None:
        common_ids = [idx for idx in common_ids if idx <= args.end_frame]
    if not common_ids:
        raise RuntimeError("No matching RGB/depth frame IDs were found")

    missing_depth = sorted(set(rgb_frames) - set(depth_frames))
    if missing_depth:
        print(f"Warning: {len(missing_depth)} RGB frames have no depth map")

    output.mkdir(parents=True, exist_ok=True)
    print(f"Using SAM-6D interpreter: {worker_python}")
    print(f"RGB frames: {rgb_dir}")
    print(f"Metric depth maps: {depth_dir}")
    print(f"Camera intrinsics: {camera_path}")
    if sfm_model.is_dir() and metric_alignment.is_dir():
        print(f"Filtered SfM cameras: {sfm_model}")
        print(f"Metric alignment: {metric_alignment}")
    else:
        print(
            "Warning: metric reconstruction handoff is incomplete. SAM-6D "
            "inference can run, but world-pose RANSAC will require "
            "scripts/reconstruct_rgbd_scene.py."
        )
    runtime_camera = runtime_camera_file(camera_path, output)
    textured_model, textured_files, texture_problem = (
        (None, [], None) if args.texture == "off"
        else find_textured_model(dataset, object_name)
    )
    if texture_problem:
        print(
            f"Warning: ignoring textured model for {object_name}: {texture_problem}. "
            "Using the untextured STL."
        )
    if textured_model:
        print(f"Texture: rendering {object_name} templates from {textured_model}")
    elif args.texture == "off":
        print(f"Texture: disabled (--texture off); using the untextured STL for {object_name}")
    else:
        print(
            f"Texture: none found at objects/textured/{object_name}.obj; "
            "using the untextured STL"
        )
    render_model = textured_model or cad_path
    # Rendered templates are deterministically generated from the CAD, and from
    # its texture when present. Without a texture this equals the plain STL hash,
    # so existing untextured results stay valid.
    templates_identity = combined_sha256(cad_path, textured_files)
    identity = {
        "object_name": object_name,
        "cad_path": str(cad_path),
        "cad_sha256": file_sha256(cad_path),
        "templates_identity": templates_identity,
    }
    if textured_model:
        identity["textured_model"] = str(textured_model)
    identity_changed = False
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
            previous_identity = previous.get("object_identity", {})
            identity_changed = any(
                previous_identity.get(key) != value
                for key, value in identity.items()
            )
        except (OSError, ValueError, json.JSONDecodeError):
            identity_changed = True
    if identity_changed:
        print(
            "Object CAD or templates changed; existing frame results will "
            "not be reused."
        )

    manifest_frames = []
    shared_ism_root = dataset / "sam6d_shared" / "ism" / "frames"
    for idx in common_ids:
        frame_name = f"{idx:06d}"
        frame_output = output / "frames" / frame_name
        results_dir = frame_output / "sam6d_results"
        ism_json = results_dir / "detection_ism.json"
        pem_json = results_dir / "detection_pem.json"
        shared_frame_dir = shared_ism_root / frame_name
        proposal_cache = shared_frame_dir / "sam_proposals.npz"
        proposal_metadata = shared_frame_dir / "cache.json"
        proposal_visualization = shared_frame_dir / "vis_ism_all.png"
        feature_cache = shared_frame_dir / "dino_query_features.pt"
        feature_metadata = shared_frame_dir / "dino_query_features.json"
        proposal_signature = shared_ism_signature(
            rgb_frames[idx],
            args.segmentor_model,
            args.stability_score_thresh,
        )
        shared_cache_missing = not valid_shared_ism_cache(
            proposal_cache, proposal_metadata, proposal_signature
        )
        frame_output.mkdir(parents=True, exist_ok=True)
        run_ism = (
            args.overwrite
            or identity_changed
            or shared_cache_missing
            or not valid_ism_result(ism_json)
        )
        run_pem = (
            args.overwrite
            or identity_changed
            or run_ism
            or not pem_json.is_file()
        )
        manifest_frames.append(
            {
                "frame_id": idx,
                "rgb_path": str(rgb_frames[idx]),
                "depth_path": str(depth_frames[idx]),
                "camera_path": str(runtime_camera),
                "cad_path": str(cad_path),
                "output_dir": str(frame_output),
                "ism_json": str(ism_json),
                "pem_json": str(pem_json),
                "shared_ism_cache": str(proposal_cache),
                "shared_ism_metadata": str(proposal_metadata),
                "shared_ism_visualization": str(proposal_visualization),
                "shared_ism_signature": proposal_signature,
                "shared_feature_cache": str(feature_cache),
                "shared_feature_metadata": str(feature_metadata),
                "shared_feature_signature": {
                    "version": 1,
                    "proposal_signature": proposal_signature,
                    "descriptor": "dinov2_vitl14",
                    "mask_post_processing": "sam6d_default",
                },
                "run_ism": run_ism,
                "run_pem": run_pem,
            }
        )

    ensure_temporary_templates(
        project,
        worker_python,
        cad_path,
        object_name,
        templates_dir,
        render_model=render_model,
        model_identity=templates_identity,
        timing_dataset=dataset,
    )
    if not templates_dir.is_dir():
        raise FileNotFoundError(templates_dir)
    if not any(
        frame["run_ism"] or frame["run_pem"] for frame in manifest_frames
    ):
        print(
            f"{object_name}: all inference frames are already complete; "
            "the object templates were verified."
        )

    manifest = {
        "version": 1,
        "config_path": str(config_path),
        "dataset_dir": str(dataset),
        "object_name": object_name,
        "object_identity": identity,
        "cad_path": str(cad_path),
        "textured_model": str(textured_model) if textured_model else None,
        "templates_dir": str(templates_dir),
        "segmentor_model": args.segmentor_model,
        "stability_score_thresh": args.stability_score_thresh,
        "det_score_thresh": args.det_score_thresh,
        "shared_ism_root": str(shared_ism_root),
        "skip_visualizations": args.skip_visualizations,
        "pem_config": str(pem_dir / "config" / "base.yaml"),
        "metric_reconstruction": {
            "sfm_model": str(sfm_model),
            "alignment_dir": str(metric_alignment),
            "metric_cloud": str(metric_cloud),
        },
        "frames": manifest_frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared SAM-6D manifest: {manifest_path}")
        return 0
    stage_worker = project / "scripts" / "SAM6D_batch_stage.py"

    run(
        [
            worker_python,
            stage_worker,
            "--stage",
            "ism",
            "--manifest",
            manifest_path,
            "--model-dir",
            ism_dir,
        ],
        project,
    )
    if args.ism_only:
        print(f"{object_name}: ISM complete; PEM deferred to multi-instance workflow")
        return 0
    run(
        [
            worker_python,
            stage_worker,
            "--stage",
            "pem",
            "--manifest",
            manifest_path,
            "--model-dir",
            pem_dir,
        ],
        project,
    )

    finalize_manifest(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
