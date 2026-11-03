"""Workflow discovery, validation, and command construction."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from scripts.dataset_config import dataset_path, normalize_dataset_name


@dataclass(frozen=True)
class WorkflowContext:
    project: Path
    config: Path
    dataset: Path
    dataset_id: str
    host_python: Path
    sam6d_python: Path


@dataclass(frozen=True)
class StageState:
    name: str
    ready: bool
    complete: bool
    detail: str


def load_context(config_path: str | Path, host_python: str | Path,
                 sam6d_python: str | Path) -> WorkflowContext:
    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    payload = yaml.safe_load(config.read_text()) or {}
    if "path" not in payload or "dataset_id" not in payload:
        raise ValueError("Configuration must define 'path' and 'dataset_id'")
    project = Path(payload["path"]).expanduser().resolve()
    dataset_id = normalize_dataset_name(payload["dataset_id"])
    return WorkflowContext(
        project=project,
        config=config,
        dataset=dataset_path(project, dataset_id),
        dataset_id=dataset_id,
        # Do not resolve interpreter symlinks. A venv's python executable is
        # commonly a symlink to /usr/bin/python; resolving it bypasses the
        # virtual environment and therefore its site-packages.
        host_python=Path(host_python).expanduser().absolute(),
        sam6d_python=Path(sam6d_python).expanduser().absolute(),
    )


def configured_dataset_name(config_path: str | Path) -> str:
    config = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(config.read_text()) or {}
    if "dataset_id" not in payload:
        raise ValueError("Configuration must define 'dataset_id'")
    return normalize_dataset_name(payload["dataset_id"])


def update_dataset_name(config_path: str | Path, value: str) -> str:
    """Update only the top-level dataset_id row, preserving the rest of YAML."""
    config = Path(config_path).expanduser().resolve()
    name = normalize_dataset_name(value)
    lines = config.read_text().splitlines(keepends=True)
    replacement = f"dataset_id: {json.dumps(name, ensure_ascii=False)}\n"
    found = False
    for index, line in enumerate(lines):
        if line.startswith("dataset_id:"):
            lines[index] = replacement
            found = True
            break
    if not found:
        lines.append(replacement)
    temporary = config.with_suffix(config.suffix + ".tmp")
    temporary.write_text("".join(lines))
    temporary.replace(config)
    return name


def count_files(path: Path, patterns: tuple[str, ...]) -> int:
    if not path.is_dir():
        return 0
    return len({item for pattern in patterns for item in path.glob(pattern)})


def discover_objects(context: WorkflowContext) -> list[str]:
    folder = context.dataset / "objects/stl"
    return sorted((item.stem for item in folder.glob("*.stl")), key=str.lower)


def textured_objects(context: WorkflowContext) -> set[str]:
    """Objects with a complete textured model in objects/textured/ (used automatically)."""
    from scripts.textured_models import find_textured_model
    return {name for name in discover_objects(context)
            if find_textured_model(context.dataset, name)[0] is not None}


def stage_states(context: WorkflowContext) -> list[StageState]:
    dataset = context.dataset
    images = count_files(dataset / "images", ("*.png", "*.jpg", "*.jpeg"))
    depths = count_files(dataset / "depth_maps", ("*.png",))
    objects = discover_objects(context)
    reconstruction = dataset / "reconstruction"
    sequence_root = dataset / "sam6d_sequences"
    multi_instance_root = dataset / "experimental_multi_instance" / "pem"
    def complete_multi_instance_aggregate(name: str) -> bool:
        path = multi_instance_root / name / f"{name}_poses_per_frame.json"
        if not path.is_file():
            return False
        try:
            aggregate = json.loads(path.read_text())
            return (
                int(aggregate.get("statistics", {}).get("pose_candidates", 0)) > 0
                and all(frame.get("status") != "pending_pem"
                        for frame in aggregate.get("frames", []))
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
    pose_inputs = sum(1 for name in objects
                      if complete_multi_instance_aggregate(name))
    world_poses = sum(1 for name in objects
                      if (sequence_root / name / f"{name}_world_pose.json").is_file())
    annotation_path = dataset / "sam6d_scene/annotations.json"
    return [
        StageState("Calibration", True, (dataset / "camera.json").is_file(),
                   "camera.json found" if (dataset / "camera.json").is_file()
                   else "camera.json has not been generated"),
        StageState("RGB-D recording", (dataset / "camera.json").is_file(),
                   images > 0 and depths > 0,
                   f"{images} RGB frames, {depths} depth frames"),
        StageState("Reconstruction", images >= 2 and (dataset / "camera.json").is_file(),
                   (reconstruction / "mvs_filtered_metric.ply").is_file() and
                   (reconstruction / "metric_alignment/camera_poses_metric.json").is_file(),
                   "metric cloud and poses found" if
                   (reconstruction / "mvs_filtered_metric.ply").is_file() else
                   "metric reconstruction is missing"),
        StageState("SAM-6D", bool(objects) and
                   (reconstruction / "metric_alignment/camera_poses_metric.json").is_file(),
                   bool(objects) and pose_inputs == len(objects),
                   f"{len(objects)} STL objects, {pose_inputs} pose sequences"),
        StageState("Pose Fitting", pose_inputs > 0,
                   bool(objects) and world_poses == len(objects),
                   f"{world_poses}/{len(objects)} object pose files"),
        StageState("Scene annotations", world_poses > 0, annotation_path.is_file(),
                   "annotations.json found" if annotation_path.is_file()
                   else "scene annotations are missing"),
    ]


def command_text(command: list[str | Path]) -> str:
    return shlex.join([str(item) for item in command])


def interpreter_modules(python: Path, modules: tuple[str, ...]) -> dict[str, bool]:
    """Check imports using the selected interpreter without importing locally."""
    result = {name: False for name in modules}
    if not python.is_file():
        return result
    program = (
        "import importlib.util; "
        "print(' '.join('1' if importlib.util.find_spec(x) else '0' "
        f"for x in {modules!r}))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", program], capture_output=True, text=True,
            timeout=10, check=True,
        )
        values = completed.stdout.strip().split()
        if len(values) == len(modules):
            result = dict(zip(modules, (value == "1" for value in values)))
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def pycolmap_cuda_available(python: Path) -> bool:
    if not python.is_file():
        return False
    try:
        completed = subprocess.run(
            [str(python), "-c",
             "import pycolmap; print('1' if pycolmap.has_cuda else '0')"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return completed.stdout.strip() == "1"
    except (OSError, subprocess.SubprocessError):
        return False


def aruco_dictionary_names(python: Path) -> list[str]:
    """Return the predefined marker dictionaries exposed by OpenCV ArUco."""
    if not python.is_file():
        return []
    program = """import cv2, json
names = sorted(n for n in dir(cv2.aruco)
               if n.startswith('DICT_') and isinstance(getattr(cv2.aruco, n), int))
unique = {}
for name in names:
    unique.setdefault(getattr(cv2.aruco, name), name)
print(json.dumps(sorted(unique.values())))
"""
    try:
        completed = subprocess.run(
            [str(python), "-c", program], capture_output=True, text=True,
            timeout=10, check=True,
        )
        values = json.loads(completed.stdout)
        return values if isinstance(values, list) else []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []


def realsense_devices(python: Path) -> list[dict[str, str]]:
    """Enumerate attached RealSense devices in the selected environment."""
    if not python.is_file():
        return []
    program = r'''import json
import pyrealsense2 as rs
result = []
for device in rs.context().query_devices():
    def info(field, fallback=""):
        try:
            return device.get_info(field) if device.supports(field) else fallback
        except RuntimeError:
            return fallback
    result.append({
        "name": info(rs.camera_info.name, "Intel RealSense"),
        "serial": info(rs.camera_info.serial_number),
        "product_line": info(rs.camera_info.product_line),
        "firmware": info(rs.camera_info.firmware_version),
    })
print(json.dumps(result))'''
    try:
        completed = subprocess.run(
            [str(python), "-c", program], capture_output=True, text=True,
            timeout=10, check=True,
        )
        devices = json.loads(completed.stdout)
        return devices if isinstance(devices, list) else []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []


def validate_stl_file(python: Path, project: Path, path: Path) -> dict:
    """Validate an uploaded STL with the pipeline's Open3D environment."""
    completed = subprocess.run(
        [str(python), str(project / "scripts/validate_stl_model.py"), str(path)],
        cwd=str(project), capture_output=True, text=True, timeout=120,
    )
    marker = "STL_REPORT_JSON="
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    detail = (completed.stderr or completed.stdout).strip()
    raise RuntimeError(detail or "STL validation did not return a report")


def script_command(context: WorkflowContext, script: str,
                   *arguments: str | Path, sam6d: bool = False) -> list[str]:
    python = context.sam6d_python if sam6d else context.host_python
    return [str(python), str(context.project / "scripts" / script),
            *map(str, arguments)]


def calibration_command(context: WorkflowContext, *, serial: str,
                        width: int, height: int, fps: int,
                        views: int, squares_x: int, squares_y: int,
                        square_length: float, marker_length: float,
                        dictionary: str,
                        max_displayed_markers: int = 30,
                        output_dir: Path | None = None) -> list[str]:
    command = script_command(
        context, "calibrate_charuco_camera.py", "--config", context.config,
        "--serial", serial, "--width", str(width), "--height", str(height),
        "--fps", str(fps), "--views", str(views),
        "--squares-x", str(squares_x), "--squares-y", str(squares_y),
        "--square-length", str(square_length), "--marker-length", str(marker_length),
        "--dictionary", dictionary,
        "--max-displayed-markers", str(max_displayed_markers),
        "--activate-dataset",
    )
    if output_dir is not None:
        command.extend(("--output-dir", str(output_dir)))
    return command


def recording_command(context: WorkflowContext, *, width: int, height: int,
                      depth_width: int, depth_height: int, fps: int,
                      overwrite: bool, serial: str = "",
                      target_frames: int = 250,
                      camera_json: Path | None = None) -> list[str]:
    command = script_command(
        context, "record_realsense_rgbd.py", "--config", context.config,
        "--width", str(width), "--height", str(height),
        "--depth-width", str(depth_width), "--depth-height", str(depth_height),
        "--fps", str(fps), "--target-frames", str(target_frames),
    )
    if overwrite:
        command.append("--overwrite")
    if serial:
        command.extend(("--serial", serial))
    if camera_json is not None:
        command.extend(("--camera-json", str(camera_json)))
    return command


def reconstruction_command(context: WorkflowContext, *, device: str,
                           overwrite: bool, max_image_size: int,
                           geometric_consistency: bool,
                           target_overlap: float = 0.7) -> list[str]:
    command = script_command(
        context, "reconstruct_rgbd_scene.py", "--config", context.config,
        "--device", device, "--max-image-size", str(max_image_size),
        "--target-overlap", str(target_overlap),
    )
    if overwrite:
        command.append("--overwrite")
    if not geometric_consistency:
        command.append("--no-geometric-consistency")
    return command


def sam6d_command(context: WorkflowContext, *, object_name: str | None,
                  segmentor: str, stability: float, detection: float,
                  start_frame: int | None, end_frame: int | None,
                  overwrite: bool, check_only: bool) -> list[str]:
    script = (
        "SAM6D_batch_inference.py"
        if check_only else "SAM6D_multi_instance_inference.py"
    )
    command = script_command(
        context, script, "--config", context.config,
        "--python", context.sam6d_python, "--segmentor-model", segmentor,
        "--stability-score-thresh", str(stability),
        "--det-score-thresh", str(detection),
    )
    if object_name:
        command.extend(("--object-name", object_name))
    if start_frame is not None:
        command.extend(("--start-frame", str(start_frame)))
    if end_frame is not None:
        command.extend(("--end-frame", str(end_frame)))
    if overwrite:
        command.append("--overwrite")
    if check_only:
        command.append("--check-inputs")
    return command


def sam6d_frame_selection_command(
    context: WorkflowContext, *, target_overlap: float
) -> list[str]:
    return script_command(
        context, "select_sam6d_frames.py", "--config", context.config,
        "--dataset-dir", context.dataset,
        "--target-overlap", str(target_overlap),
    )


def ransac_command(context: WorkflowContext, *, object_name: str | None,
                   translation: float, rotation: float, maximum_rotation: float,
                   minimum_inliers: int, symmetry: str,
                   min_silhouette_iou: float = 0.0) -> list[str]:
    command = script_command(
        context, "fit_multi_instance_poses.py", "--config", context.config,
        "--translation-threshold", str(translation),
        "--rotation-threshold-deg", str(rotation),
        "--maximum-rotation-threshold-deg", str(maximum_rotation),
        "--minimum-inliers", str(minimum_inliers), "--symmetry-mode", symmetry,
        "--min-silhouette-iou", str(min_silhouette_iou),
    )
    if object_name:
        command.extend(("--object-name", object_name))
    return command


def annotation_command(context: WorkflowContext, *, model_scale: float,
                       overwrite: bool) -> list[str]:
    command = script_command(
        context, "generate_scene_annotations.py", "--config", context.config,
        "--python", context.sam6d_python, "--model-scale", str(model_scale),
    )
    if overwrite:
        command.append("--overwrite")
    return command


def annotation_video_command(context: WorkflowContext, *, fps: float,
                             overwrite: bool) -> list[str]:
    command = script_command(
        context, "create_annotation_video.py", "--config", context.config,
        "--fps", str(fps),
    )
    if overwrite:
        command.append("--overwrite")
    return command


def huggingface_command(context: WorkflowContext, *, repo_id: str,
                        license_name: str, private: bool,
                        upload: bool, create_pr: bool = False,
                        train_ratio: float = 0.8,
                        validation_ratio: float = 0.1,
                        split_seed: int = 42) -> list[str]:
    command = script_command(
        context, "publish_huggingface_dataset.py", "--config", context.config,
        "--license", license_name, "--train-ratio", str(train_ratio),
        "--validation-ratio", str(validation_ratio),
        "--split-seed", str(split_seed),
    )
    if repo_id:
        command.extend(("--repo-id", repo_id))
    if private:
        command.append("--private")
    if upload:
        command.extend(("--upload", "--create-pr"))
    return command


def save_acknowledgement(context: WorkflowContext, content: str) -> Path:
    # Application acknowledgement: deliberately outside data/#### so it is
    # presented by the UI and never included in a dataset upload.
    path = context.project / "ui/ACKNOWLEDGEMENTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content.rstrip() + "\n")
    temporary.replace(path)
    return path


def read_annotation_summary(context: WorkflowContext) -> dict | None:
    path = context.dataset / "sam6d_scene/annotations.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


_PROJECT = Path(__file__).resolve().parents[1]
_PROJECT_VENV = _PROJECT / "LabelFactory/bin/python3"
DEFAULT_HOST_PYTHON = _PROJECT_VENV if _PROJECT_VENV.is_file() else Path(sys.executable)
DEFAULT_SAM6D_PYTHON = Path(
    os.environ.get("SAM6D_PYTHON", str(Path(sys.executable)))
).expanduser()
