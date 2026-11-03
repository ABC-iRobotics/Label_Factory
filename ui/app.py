#!/usr/bin/env python3
"""Streamlit workflow assistant for the Label Factory pipeline."""

from __future__ import annotations

import base64
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ui.pipeline import (DEFAULT_HOST_PYTHON, DEFAULT_SAM6D_PYTHON, annotation_command,
                         annotation_video_command,
                         aruco_dictionary_names, calibration_command, command_text, discover_objects,
                         textured_objects,
                         interpreter_modules, load_context,
                         update_dataset_name, ransac_command, recording_command,
                         reconstruction_command, realsense_devices, sam6d_command, script_command,
                         sam6d_frame_selection_command,
                         stage_states, huggingface_command, validate_stl_file)
from ui.process_runner import run_command


def runtime_pycolmap_cuda_available(python: Path) -> bool:
    """Probe on every Streamlit rerun; do not rely on a cached helper module."""
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


st.set_page_config(page_title="Label Factory", page_icon="🏭", layout="wide")
st.markdown(
    """
    <style>
    [role="tablist"] {
        gap: 1rem !important;
        width: 100% !important;
        min-height: 5.5rem !important;
    }
    [role="tab"] {
        min-height: 5.5rem !important;
        padding: 1.25rem 1.6rem !important;
        flex: 1 1 auto !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }
    [role="tab"] *,
    .stTabs [data-baseweb="tab"] *,
    div[data-testid="stTabs"] [data-baseweb="tab"] * {
        font-size: 1.55rem !important;
        line-height: 1.35 !important;
        font-weight: 700 !important;
        white-space: pre-line !important;
        text-align: center !important;
    }
    .label-factory-acknowledgement {
        opacity: 0.58;
        font-size: 0.78rem;
        line-height: 1.35;
        text-align: right;
        padding-top: 0.45rem;
    }
    .label-factory-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .label-factory-title h1 {
        margin: 0;
        padding: 0;
        font-size: 6.80rem;
        line-height: 1.08;
    }
    .label-factory-title img {
        width: 185px;
        height: 185px;
        object-fit: contain;
    }
    .label-factory-title .label-initial {
        color: #E9A85C;
    }
    .label-factory-title .factory-initial {
        color: #D65C43;
    }
    .workflow-stage-status {
        text-align: center;
        width: 100%;
        min-height: 7.5rem;
    }
    .progression-heading {
        display: inline-block;
        font-size: 1.5rem;
        font-weight: 600;
        line-height: 1.2;
        margin: -1rem 0 0.45rem -1rem;
        padding: 0.5rem 0.85rem;
        border-right: 1px solid rgba(128, 128, 128, 0.45);
        border-bottom: 1px solid rgba(128, 128, 128, 0.45);
        border-radius: 0 0 0.5rem 0;
        background: var(--background-color);
    }
    .workflow-stage-name {
        font-size: 0.9rem;
        line-height: 1.25;
        min-height: 1.5rem;
        margin-bottom: 0.3rem;
    }
    .workflow-stage-icon {
        font-size: 2.5rem;
        line-height: 1.2;
        margin-bottom: 0.55rem;
    }
    .workflow-stage-detail {
        font-size: 0.82rem;
        line-height: 1.3;
        opacity: 0.62;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
title_column, acknowledgement_column = st.columns((0.54, 0.46), gap="large")
with title_column:
    emoji_path = PROJECT / "Emoji.png"
    if emoji_path.is_file():
        emoji_data = base64.b64encode(emoji_path.read_bytes()).decode("ascii")
        st.markdown(
            '<div class="label-factory-title">'
            '<h1><span class="label-initial">L</span>abel '
            '<span class="factory-initial">F</span>actory</h1>'
            f'<img src="data:image/png;base64,{emoji_data}" alt="Label Factory icon">'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.title("Label Factory")
with acknowledgement_column:
    st.markdown(
        '<div class="label-factory-acknowledgement">'
        '<strong>Acknowledgement</strong><br>'
        'This work was supported in part by Project 2024-1.2.3-HU-RIZONT-00069 '
        '(MedLaBotX), implemented with support provided by the Ministry of Culture '
        'and Innovation of Hungary from the National Research, Development, and '
        'Innovation Fund, financed under the 2024-1.2.3-HU-RIZONT funding scheme; '
        'in part by the Consolidator Researcher Grant of Óbuda University under '
        'Grant OKPPKPC004'
        '</div>',
        unsafe_allow_html=True,
    )

default_config = PROJECT / "scripts/configs/default.yaml"

try:
    context = load_context(default_config, DEFAULT_HOST_PYTHON, DEFAULT_SAM6D_PYTHON)
except Exception as error:
    st.error(f"Configuration error: {error}")
    st.stop()

states = stage_states(context)
host_modules = interpreter_modules(
    context.host_python, ("open3d", "pycolmap", "cv2", "pyrealsense2")
)
reconstruction_runtime_ready = all(
    host_modules[name] for name in ("open3d", "pycolmap", "cv2")
)
pycolmap_cuda = runtime_pycolmap_cuda_available(context.host_python)
progress_box = st.container(border=True)
progress_box.markdown(
    '<div class="progression-heading">Progression bar</div>',
    unsafe_allow_html=True,
)
columns = progress_box.columns(len(states))
for column, state in zip(columns, states):
    icon = "✅" if state.complete else ("▶️" if state.ready else "⛔")
    column.markdown(
        '<div class="workflow-stage-status">'
        f'<div class="workflow-stage-name">{html.escape(state.name)}</div>'
        f'<div class="workflow-stage-icon">{icon}</div>'
        f'<div class="workflow-stage-detail">{html.escape(state.detail)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

def execute(command: list[str], key: str,
            environment_overrides: dict[str, str] | None = None) -> None:
    output = st.empty()
    with st.status("Running…", expanded=True) as status:
        return_code, final_output = run_command(
            command, context.project,
            lambda text: output.code(
                text, language=None, height=480, wrap_lines=False,
            ),
            environment_overrides=environment_overrides,
        )
        has_warnings = bool(re.search(
            r"(?im)^\s*(warning|warn|review|error)(?:\s|:)", final_output or ""
        ))
        if return_code == 0 and has_warnings:
            result_level = "warning"
            status.update(label="Completed with warnings", state="complete",
                          expanded=False)
            st.warning("Command completed with warnings. See the log at the bottom of the page.")
        elif return_code == 0:
            result_level = "success"
            status.update(label="Completed", state="complete", expanded=False)
            st.success("Command completed successfully.")
        else:
            result_level = "error"
            status.update(label=f"Failed with exit code {return_code}",
                          state="error", expanded=True)
            st.error("The command failed. See the log at the bottom of the page.")
    st.session_state["last_run"] = {
        "stage": key,
        "command": command_text(command),
        "return_code": return_code,
        "result_level": result_level,
        "output": final_output,
        "notification_pending": True,
    }
    tab_for_stage = {
        "calibration": "Capture",
        "recording": "Capture",
        "reconstruction": "Reconstruction",
        "reconstruction_visualizer": "Reconstruction",
        "sam_frame_selection": "Instance segmentation\n& Pose estimation",
        "sam_check": "Instance segmentation\n& Pose estimation",
        "sam6d": "Instance segmentation\n& Pose estimation",
        "ransac": "Pose\nFitting",
        "ransac_visualizer": "Pose\nFitting",
        "annotations": "Scene &\nannotation",
        "annotation_video": "Scene &\nannotation",
        "huggingface_inventory": "Review &\npublish",
        "dataset_cleanup": "Review &\npublish",
        "huggingface_upload": "Review &\npublish",
    }
    st.session_state["active_workflow_tab"] = tab_for_stage.get(key, "Setup")
    # Re-evaluate every stage prerequisite and discover newly generated files.
    # Persisting last_run keeps the result visible after this automatic rerun.
    st.rerun()


def action_completed(key: str, artifact_exists: bool = False) -> bool:
    """Return completion from durable outputs or the latest successful UI run."""
    last_run = st.session_state.get("last_run", {})
    if last_run.get("stage") == key:
        return last_run.get("return_code") == 0
    return artifact_exists


def completion_tick(key: str, artifact_exists: bool = False,
                    label: str = "Completed successfully", target=None) -> None:
    if target is None:
        target = st
    last_run = st.session_state.get("last_run", {})
    if last_run.get("stage") == key and last_run.get("result_level") == "error":
        target.caption("❌ Process failed — see the log at the bottom of the page")
    elif last_run.get("stage") == key and last_run.get("result_level") == "warning":
        target.caption("⚠️ Completed with warnings — see the log at the bottom of the page")
    elif action_completed(key, artifact_exists):
        target.caption(f"✅ {label}")


def indexed_images(folder: Path) -> dict[int, Path]:
    """Index image-like files by the last integer in their filename."""
    result: dict[int, Path] = {}
    if not folder.is_dir():
        return result
    for path in folder.iterdir():
        digits = "".join(character if character.isdigit() else " " for character in path.stem)
        values = digits.split()
        if path.is_file() and values:
            result[int(values[-1])] = path
    return result


def themed_file(light_path: Path, dark_path: Path) -> Path:
    """Choose a visualization matching Streamlit's active client theme."""
    preferred, fallback = (
        (dark_path, light_path)
        if st.context.theme.type == "dark"
        else (light_path, dark_path)
    )
    return preferred if preferred.is_file() else fallback


def depth_preview(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid = depth > 0
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], (1, 99))
        if high <= low:
            high = low + 1
        preview[valid] = np.clip((depth[valid] - low) * 255 / (high - low), 0, 255)
    return preview


def label_preview(path: Path) -> np.ndarray:
    labels = np.asarray(Image.open(path))
    if labels.ndim == 3:
        labels = labels[..., 0]
    colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    nonzero = labels > 0
    colors[..., 0][nonzero] = (labels[nonzero] * 67 + 29) % 255
    colors[..., 1][nonzero] = (labels[nonzero] * 131 + 71) % 255
    colors[..., 2][nonzero] = (labels[nonzero] * 197 + 113) % 255
    return colors


def realsense_presets(device_name: str) -> list[tuple[str, int, int, int, int, int]]:
    """Return label, RGB WxH, depth WxH, and FPS presets for a camera model."""
    model = device_name.upper()
    if "D455" in model or "D456" in model:
        return [
            ("Close range (recommended), ≈260–500 mm", 1280, 800, 640, 360, 30),
            ("Balanced range, ≈350–3000 mm", 1280, 800, 848, 480, 30),
            ("Long range, ≈520–6000 mm", 1280, 800, 1280, 720, 30),
        ]
    if "D435" in model:
        return [
            ("Balanced range (recommended), ≈195–3000 mm", 1280, 720, 848, 480, 30),
            ("Close range / lower bandwidth, ≈175–3000 mm", 640, 480, 640, 480, 30),
            ("Long range, ≈280–10000 mm", 1280, 720, 1280, 720, 30),
        ]
    if "D415" in model:
        return [
            ("Balanced range (recommended), ≈310–3000 mm", 1280, 720, 640, 480, 30),
            ("Lower bandwidth, ≈310–3000 mm", 640, 480, 640, 480, 30),
            ("Long range, ≈450–10000 mm", 1280, 720, 1280, 720, 30),
        ]
    return [
        ("Compatible default (recommended), model-specific range", 640, 480, 640, 480, 30),
        ("Higher depth detail, model-specific range", 1280, 720, 848, 480, 30),
    ]


workflow_tabs = [
    "Framework\nguide", "Setup", "Capture", "Reconstruction",
    "Instance segmentation\n& Pose estimation", "Pose\nFitting",
    "Scene &\nannotation", "Review &\npublish",
]


def workflow_tab_changed() -> None:
    selected_tab = st.session_state.get("workflow_tab_selection")
    if selected_tab in workflow_tabs:
        st.session_state["active_workflow_tab"] = selected_tab
    if selected_tab == "Review &\npublish":
        # Consumed inside the tab body. A new change away and back sets it again.
        st.session_state["refresh_inventory_on_open"] = True


active_workflow_tab = st.session_state.get("active_workflow_tab", "Setup")
if active_workflow_tab not in workflow_tabs:
    active_workflow_tab = "Setup"
(tab_guide, tab_setup, tab_capture, tab_reconstruction, tab_sam6d, tab_pose,
 tab_scene, tab_publish) = st.tabs(
    workflow_tabs, default=active_workflow_tab,
    key="workflow_tab_selection", on_change=workflow_tab_changed,
)

# Commands trigger an automatic rerun. Show their result once after that rerun so
# feedback appears on the restored workflow tab rather than disappearing with it.
notification_run = st.session_state.get("last_run", {})
if notification_run.pop("notification_pending", False):
    notification_level = notification_run.get("result_level", "success")
    notification_messages = {
        "success": "Process completed successfully.",
        "warning": "Process completed with warnings. See the log at the bottom of the page.",
        "error": "Process failed. See the log at the bottom of the page.",
    }
    notification_icons = {"success": "✅", "warning": "⚠️", "error": "❌"}
    st.toast(
        notification_messages[notification_level],
        icon=notification_icons[notification_level],
    )
stl_notification = st.session_state.pop("stl_notification_pending", None)
if stl_notification:
    st.toast(stl_notification["message"], icon=stl_notification["icon"])

with tab_guide:
    with st.expander("1. Setup — environment and object models", expanded=True):
        st.markdown(
            "**Purpose and preparation**\n\n"
            "Select the working dataset and verify both Python environments and CUDA. "
            "Upload one STL for every object type that must be recognized. Use a clear "
            "filename such as `Pliers.stl`, because its stem becomes the object identifier "
            "in every later result.\n\n"
            "**Model requirements**\n\n"
            "The mesh should represent the real object at its correct physical scale, have "
            "a useful local origin and orientation, and preferably be watertight. The setup "
            "validator rejects empty or unreadable geometry before it can cause harder-to-find "
            "errors in template rendering or pose estimation.\n\n"
            "**Continue when**\n\n"
            "All readiness indicators are green and the discovered-object list contains every "
            "expected object. If a model has the wrong dimensions, repair it before capture; "
            "its scale directly affects the estimated pose and rendered annotations."
        )
    with st.expander("2. Capture — calibration and synchronized RGB-D frames"):
        st.markdown(
            "**Calibrate the selected stream**\n\n"
            "Choose the connected RealSense and the exact resolution that will be recorded. "
            "Reuse a saved calibration only for the same camera and RGB resolution. For a new "
            "calibration, measure the printed ChArUco square and marker sizes accurately, then "
            "collect views near the image center and edges, at several distances and tilts. "
            "Many nearly identical views add little useful information.\n\n"
            "**Record the real scene**\n\n"
            "Arrange the objects in their final, stationary positions. Move the camera slowly "
            "around the scene, maintaining strong overlap between consecutive views and showing "
            "every object from several directions. Include textured surroundings to support "
            "camera tracking. About 250 useful frames are recommended. RGB and depth are saved "
            "synchronously, with depth aligned to RGB and expressed in millimetres.\n\n"
            "**Continue when**\n\n"
            "Object edges align in the RGB/depth preview, the images are sharp, and depth is "
            "present over the important surfaces. Re-record if objects moved, frames are blurred, "
            "or large areas of the target objects have invalid depth."
        )
    with st.expander("3. Reconstruction — sparse, dense, filtered, and metric scene"):
        st.markdown(
            "**How the scene is reconstructed**\n\n"
            "Structure from Motion matches visual features across the RGB sequence and jointly "
            "estimates a camera trajectory and sparse scene. Sparse filtering removes weakly "
            "supported points. Multi-View Stereo then adds dense surface samples to that same "
            "reconstruction, after which statistical filtering removes isolated dense noise.\n\n"
            "**How it becomes metric**\n\n"
            "Image-only reconstruction has arbitrary scale. The pipeline compares reconstructed "
            "geometry with the RealSense depth maps and estimates one similarity transform—scale, "
            "rotation, and translation. It applies this consistently to the point cloud and camera "
            "poses and records the transform in the alignment metadata.\n\n"
            "**Continue when**\n\n"
            "Use the reconstruction viewer to step through the sparse, filtered, dense, and metric "
            "results. Look for a smooth camera path, coherent surfaces, plausible dimensions, and "
            "no doubled scene. Floating regions, broken trajectories, or duplicated surfaces mean "
            "the capture or reconstruction settings need attention."
        )
    with st.expander("4. Instance segmentation & pose estimation — SAM-6D"):
        st.markdown(
            "**Find object candidates**\n\n"
            "SAM generates multiple object-shaped masks once for each image. These proposals are "
            "shared between all object types. Rendered STL templates provide appearance descriptors "
            "that match proposals to each known object. Retaining several candidates above the "
            "threshold allows repeated or visually similar instances to survive this stage.\n\n"
            "**Estimate a pose in every usable frame**\n\n"
            "For each matched mask, the pose network combines its RGB crop, aligned metric depth, "
            "camera intrinsics, and CAD geometry to estimate a six-degree-of-freedom object-to-camera "
            "transform. Scores and stable candidate identities are kept for later multi-view fitting.\n\n"
            "**Continue when**\n\n"
            "Inspect the mask previews and candidate counts. Masks should cover complete physical "
            "objects rather than only handles or fragments, and separate instances should remain "
            "distinct. Occasional bad poses are expected; consistently missing masks or empty depth "
            "inside masks must be corrected before pose fitting."
        )
    with st.expander("5. Pose fitting — robust scene-level object poses"):
        st.markdown(
            "**Combine observations across the scene**\n\n"
            "Each camera-relative estimate is transformed into the metric reconstruction frame. "
            "Position clustering runs first, allowing separate physical instances to emerge without "
            "requiring their count in advance. Orientation RANSAC then runs independently inside each "
            "position cluster, rejects outliers, relaxes its angular tolerance when supported, and "
            "compares equivalent rotations for symmetric meshes.\n\n"
            "**Understand the controls**\n\n"
            "The translation threshold decides how close observations must be to represent the same "
            "instance. Minimum inliers specifies how many unique frames must support a fitted pose. "
            "Thresholds that are too small fragment one object; thresholds that are too large may "
            "merge nearby copies.\n\n"
            "**Continue when**\n\n"
            "The position plot contains compact clusters at the visible object locations and the "
            "axis-angle plot shows consistent orientation support. The result panel lists the number "
            "of fitted instances per object. Missing types are warnings when other objects succeed."
        )
    with st.expander("6. Scene & annotation — per-frame ground truth"):
        st.markdown(
            "**Generate consistent frame annotations**\n\n"
            "The fitted pose places each accepted STL instance in the metric scene. For every RGB "
            "frame, the corresponding metric camera pose and calibrated intrinsics project those "
            "meshes into the image. Depth-aware rendering determines which portions are visible and "
            "assigns a stable instance identity and color across the full sequence. Objects that "
            "could not be fitted are excluded rather than stopping annotation of valid objects.\n\n"
            "**What is created**\n\n"
            "The stage writes machine-readable scene and per-frame metadata, instance-ID masks, raw "
            "segmentation data, colored overlays, and a review video. These are generated together so "
            "the JSON records, mask pixels, and visualization use the same poses and identities.\n\n"
            "**Continue when**\n\n"
            "Play the overlay video from beginning to end. Colored silhouettes should remain attached "
            "to the correct physical object, preserve identity between views, and respect occlusion. "
            "Systematic shifts indicate an incorrect pose, mesh scale, camera calibration, or camera "
            "coordinate convention and should not be accepted for publication."
        )
    with st.expander("7. Review & publish — dataset packaging"):
        st.markdown(
            "**Review before packaging**\n\n"
            "A person must inspect the reconstruction, pose-cluster plots, masks, and complete overlay "
            "video. The inventory then checks that paired RGB/depth frames, calibration, camera poses, "
            "object models, fitted poses, annotations, and visualizations are present and reports the "
            "exact files and size selected for upload.\n\n"
            "**Create training-ready subsets**\n\n"
            "Choose train, validation, and test ratios plus a deterministic seed. Frames receive stable "
            "assignments so rerunning with the same settings does not reshuffle existing data. Separate "
            "Parquet configurations expose image/depth pairs, instance-segmentation labels, and object-pose "
            "samples in forms that Hugging Face datasets can load directly. New dataset directories are "
            "appended as additional shards rather than replacing earlier captures.\n\n"
            "**Publish safely**\n\n"
            "Select a license that matches the rights to the images and models, provide a write-capable "
            "Hugging Face token, and submit the package. Publication always creates a pull request—never "
            "a direct commit—so the proposed files and generated data card can be reviewed on Hugging Face "
            "before merging."
        )

with tab_setup:
    with st.container(border=True):
        st.subheader("Dataset")
        st.write(
            "Select the dataset directory used by every stage of the workflow. "
            "Numeric names are normalized to four digits; descriptive text names are "
            "also supported."
        )
        dataset_columns = st.columns((3, 1), vertical_alignment="bottom")
        dataset_name = dataset_columns[0].text_input(
            "Dataset name", value=context.dataset_id,
            help="Examples: 0008, workshop_scene, or tool_capture_01. Changing this "
                 "updates the dataset_id entry in the workflow configuration.",
        )
        if dataset_columns[1].button(
            "Use dataset", type="primary", use_container_width=True,
            help="Activate this dataset name for capture and every subsequent stage.",
        ):
            try:
                selected_name = update_dataset_name(default_config, dataset_name)
            except (OSError, ValueError) as error:
                st.error(f"Could not change dataset: {error}")
            else:
                st.session_state.pop("last_run", None)
                st.session_state["active_workflow_tab"] = "Setup"
                st.session_state["dataset_notification"] = (
                    f"Dataset changed to {selected_name}"
                )
                st.rerun()
        dataset_notification = st.session_state.pop("dataset_notification", None)
        if dataset_notification:
            st.success(dataset_notification)

    environment_box = st.container(border=True)
    environment_box.subheader("Environment readiness")
    environment_box.write(("✅" if context.dataset.exists() else "ℹ️") +
             " Dataset directory " + ("found" if context.dataset.exists() else
                                       "will be created by capture"))
    environment_box.write(("✅" if context.host_python.is_file() else "❌") +
             " Reconstruction environment")
    if all(host_modules.values()):
        environment_box.write("✅ Python packages are found")
    else:
        missing_modules = [name for name, available in host_modules.items()
                           if not available]
        environment_box.write("❌ Python packages are missing")
        environment_box.caption("Missing: " + ", ".join(missing_modules))
    environment_box.write(("✅" if pycolmap_cuda else "❌") + " PyCOLMAP CUDA support")
    if not all(host_modules.values()):
        environment_box.error(
            "The selected Reconstruction Python is missing required modules. "
            "Install the required packages in the project environment."
        )
    environment_box.write(("✅" if context.sam6d_python.is_file() else "❌") +
             " SAM-6D environment")
    objects = discover_objects(context)
    with_texture = textured_objects(context)
    environment_box.write(
        "**Discovered objects:** " + (
            ", ".join(f"{name} (textured)" if name in with_texture else name
                      for name in objects) if objects else "none")
    )

    object_models_box = st.container(border=True)
    object_models_box.subheader("Object models")
    object_models_box.write(
        "Upload one or more STL object models. Files are validated before they are "
        "stored in the active dataset. To use a texture, also place `<name>.obj`, "
        "`<name>.mtl` and the texture image in the dataset's `objects/textured/` "
        "folder; it is used automatically for template rendering when complete, "
        "otherwise the STL is used."
    )
    previous_uploads = st.session_state.get("stl_upload_results", [])
    for result in previous_uploads:
        message = f"{result['filename']}: {result['message']}"
        if result["status"] == "saved":
            object_models_box.success(message)
        elif result["status"] == "warning":
            object_models_box.warning(message)
        else:
            object_models_box.error(message)
        report = result.get("report")
        if report:
            dimensions = report["dimensions_source_units"]
            object_models_box.caption(
                f"{report['triangle_count']:,} triangles; dimensions in STL source "
                f"units: {dimensions[0]:.3f} × {dimensions[1]:.3f} × "
                f"{dimensions[2]:.3f}; watertight: "
                f"{'yes' if report['watertight'] else 'no'}"
            )
            for warning in report["warnings"]:
                object_models_box.caption(f"Warning: {warning}")
    uploaded_stls = object_models_box.file_uploader(
        "STL files", type=["stl"], accept_multiple_files=True,
        key=f"stl_files_{st.session_state.get('stl_uploader_generation', 0)}",
        help="Select the CAD meshes for every manipulable object. Filenames become "
             "object identifiers and may contain letters, numbers, dots, underscores, "
             "and hyphens.",
    )
    replace_stls = object_models_box.checkbox(
        "Replace STL files with the same name", value=False,
        help="When disabled, an existing object model is protected from replacement.",
    )
    if object_models_box.button(
        "Validate and save STL files", disabled=not uploaded_stls,
        help="Checks geometry first, then stores only valid files in this dataset.",
    ):
        upload_results = []
        maximum_bytes = 200 * 1024 * 1024
        with tempfile.TemporaryDirectory(prefix="label_factory_stl_") as temporary:
            temporary_dir = Path(temporary)
            for uploaded in uploaded_stls:
                original_name = uploaded.name
                stem = Path(original_name).stem
                safe = (
                    Path(original_name).name == original_name
                    and Path(original_name).suffix.lower() == ".stl"
                    and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", stem))
                )
                if not safe:
                    upload_results.append({
                        "filename": original_name, "status": "error",
                        "message": "unsafe filename; use letters, numbers, dots, "
                                   "underscores, or hyphens",
                    })
                    continue
                content = uploaded.getvalue()
                if not content or len(content) > maximum_bytes:
                    upload_results.append({
                        "filename": original_name, "status": "error",
                        "message": "file is empty or exceeds the 200 MiB limit",
                    })
                    continue
                temporary_path = temporary_dir / f"{stem}.stl"
                temporary_path.write_bytes(content)
                try:
                    report = validate_stl_file(
                        context.host_python, context.project, temporary_path
                    )
                except Exception as error:
                    upload_results.append({
                        "filename": original_name, "status": "error",
                        "message": f"validation failed: {error}",
                    })
                    continue
                if not report["valid"]:
                    upload_results.append({
                        "filename": original_name, "status": "error",
                        "message": "; ".join(report["errors"]), "report": report,
                    })
                    continue
                stl_dir = context.dataset / "objects" / "stl"
                stl_dir.mkdir(parents=True, exist_ok=True)
                if not stl_dir.resolve().is_relative_to(context.dataset.resolve()):
                    raise RuntimeError("The STL directory resolves outside the dataset")
                destination = stl_dir / f"{stem}.stl"
                if destination.exists() and not replace_stls:
                    upload_results.append({
                        "filename": destination.name, "status": "warning",
                        "message": "already exists and was not replaced", "report": report,
                    })
                    continue
                pending = stl_dir / f".{stem}.stl.uploading"
                pending.write_bytes(content)
                pending.replace(destination)
                upload_results.append({
                    "filename": destination.name, "status": "saved",
                    "message": "validated and saved", "report": report,
                })
        st.session_state["stl_upload_results"] = upload_results
        upload_statuses = {result["status"] for result in upload_results}
        if "error" in upload_statuses:
            st.session_state["stl_upload_level"] = "error"
            st.session_state["stl_notification_pending"] = {
                "message": "One or more STL files failed validation. See the details in Setup.",
                "icon": "❌",
            }
        elif "warning" in upload_statuses:
            st.session_state["stl_upload_level"] = "warning"
            st.session_state["stl_notification_pending"] = {
                "message": "STL upload completed with warnings. See the details in Setup.",
                "icon": "⚠️",
            }
        else:
            st.session_state["stl_upload_level"] = "success"
            st.session_state["stl_notification_pending"] = {
                "message": "STL files were validated and saved successfully.",
                "icon": "✅",
            }
        # A file uploader retains its browser-side selection across reruns. Give it
        # a fresh widget key after processing so the completed upload is cleared.
        st.session_state["stl_uploader_generation"] = (
            st.session_state.get("stl_uploader_generation", 0) + 1
        )
        st.session_state["active_workflow_tab"] = "Setup"
        st.rerun()
    stl_upload_level = st.session_state.get("stl_upload_level")
    if stl_upload_level == "success":
        object_models_box.caption("✅ STL files validated and saved successfully")
    elif stl_upload_level == "warning":
        object_models_box.caption(
            "⚠️ STL upload completed with warnings — see the details above"
        )
    elif stl_upload_level == "error":
        object_models_box.caption("❌ STL upload failed — see the details above")

with tab_capture:
    with st.expander("Connected camera", expanded=True):
        if st.button(
            "Refresh connected cameras",
            help="Scan the RealSense SDK again after connecting, disconnecting, or "
                 "replacing a camera.",
        ):
            st.rerun()
        connected_realsense = realsense_devices(context.host_python)
        if connected_realsense:
            selected_device = st.selectbox(
                "RealSense camera",
                connected_realsense,
                format_func=lambda device: (
                    f"{device['name']} — {device['serial']}"
                    + (f" (firmware {device['firmware']})" if device.get("firmware") else "")
                ),
                help="Connected cameras are discovered through the RealSense SDK. The "
                     "serial number ensures calibration and recording use the same device.",
            )
            st.success(
                f"Connected: {selected_device['name']} "
                f"({selected_device['serial']})"
            )
        else:
            selected_device = {"name": "Intel RealSense", "serial": ""}
            st.warning(
                "No RealSense camera was discovered. Connect the camera and refresh "
                "the list. The controls remain available with a compatible fallback preset."
            )
    camera_serial = selected_device.get("serial", "")
    with st.expander("Camera parameters", expanded=True):
        available_presets = realsense_presets(selected_device.get("name", ""))
        selected_preset = st.selectbox(
            "Capture resolution preset",
            available_presets,
            format_func=lambda preset: (
                f"{preset[0]} — RGB {preset[1]}×{preset[2]}, "
                f"depth {preset[3]}×{preset[4]}"
            ),
            help="Suggested stream combinations depend on the selected RealSense model. "
                 "The millimetre values are approximate recommended working ranges, "
                 "not hard sensor cutoffs. Lower native depth resolutions generally "
                 "support a shorter minimum distance and require less USB bandwidth.",
        )
        _, width, height, depth_width, depth_height, _ = selected_preset
        fps = st.select_slider(
            "Capture FPS",
            options=list(range(6, 31, 2)),
            value=30,
            help="Select a frame rate from 6 to 30 FPS in steps of 2. A higher value "
                 "captures smoother motion but increases USB bandwidth and processing "
                 "load. 30 FPS is recommended when the selected stream profile supports it.",
        )
    with st.expander("ChArUco calibration", expanded=True):
        calibration_library = PROJECT / "calibration" / "cameras"
        previous_calibrations = sorted(
            calibration_library.glob("*/camera.json"),
            key=lambda path: path.parent.name.lower(),
        ) if calibration_library.is_dir() else []
        calibration_options: list[Path | None] = [None, *previous_calibrations]
        selected_calibration = st.selectbox(
            "Camera calibration",
            calibration_options,
            format_func=lambda path: (
                "Calibrate the selected camera" if path is None
                else path.parent.name.replace("_", " ")
            ),
            help="Create a new calibration or reuse a calibration previously saved "
                 "for the same camera type and RGB resolution.",
        )
        if selected_calibration is not None:
            try:
                selected_payload = json.loads(selected_calibration.read_text())
                selected_size = (
                    int(selected_payload.get("image_width", 0)),
                    int(selected_payload.get("image_height", 0)),
                )
                if selected_size != (int(width), int(height)):
                    st.warning(
                        f"This calibration is for {selected_size[0]}×{selected_size[1]}, "
                        f"but the selected RGB stream is {width}×{height}. Select the "
                        "matching resolution before using it."
                    )
                if st.button(
                    "Use selected calibration",
                    disabled=selected_size != (int(width), int(height)),
                    help="Copy this reusable calibration into the active dataset as camera.json.",
                ):
                    context.dataset.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(selected_calibration, context.dataset / "camera.json")
                    st.session_state["last_run"] = {
                        "stage": "calibration", "command": "Select saved calibration",
                        "return_code": 0, "result_level": "success",
                        "output": f"Activated {selected_calibration}",
                        "notification_pending": True,
                    }
                    st.rerun()
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                st.error(f"The selected calibration is invalid: {error}")

        dictionaries = aruco_dictionary_names(context.host_python)
        if not dictionaries:
            dictionaries = ["DICT_5X5_100"]
        preferred_dictionary = (
            dictionaries.index("DICT_5X5_100")
            if "DICT_5X5_100" in dictionaries else 0
        )
        dictionary = st.selectbox(
            "ChArUco marker dictionary", dictionaries, index=preferred_dictionary,
            help="Must exactly match the ArUco dictionary used to generate and print "
                 "the physical ChArUco board.",
        )
        c1, c2, c3 = st.columns(3)
        views = c1.number_input(
            "Calibration views", min_value=10, value=25,
            help="Target number of accepted board images. More varied views generally "
                 "improve calibration accuracy.")
        squares_x = c2.number_input(
            "Squares X", min_value=2, value=7,
            help="Number of chessboard squares across the printed board, not the number "
                 "of internal corners.")
        max_displayed_markers = c3.number_input(
            "Maximum displayed markers", min_value=1, value=30, step=1,
            help="Maximum number of ArUco outlines and ChArUco corner markers drawn "
                 "in the live preview. All detections are still used for calibration.",
        )
        squares_y = c1.number_input(
            "Squares Y", min_value=2, value=5,
            help="Number of chessboard squares vertically on the printed board.")
        square_length = c2.number_input(
            "Square length (m)", value=0.030, format="%.4f",
            help="Measured physical side length of one printed chessboard square in metres.")
        marker_length = c3.number_input(
            "Marker length (m)", value=0.023, format="%.4f",
            help="Measured physical side length of the ArUco marker in metres. It must "
                 "be smaller than the square length.")
        camera_type = re.sub(
            r"[^A-Za-z0-9]+", "_", selected_device.get("name", "RealSense")
        ).strip("_") or "RealSense"
        dated_calibration_dir = (
            calibration_library / f"{camera_type}_{date.today().isoformat()}"
        )
        calibration = calibration_command(
            context, serial=camera_serial, width=width, height=height, fps=fps,
            views=int(views),
            squares_x=int(squares_x), squares_y=int(squares_y),
            square_length=float(square_length), marker_length=float(marker_length),
            dictionary=dictionary,
            max_displayed_markers=int(max_displayed_markers),
            output_dir=dated_calibration_dir)
        if st.button(
            "Start calibration",
            disabled=(not context.host_python.is_file() or not host_modules["cv2"] or
                      not host_modules["pyrealsense2"]),
            help="Opens a native camera window for collecting ChArUco observations.",
        ):
            execute(calibration, "calibration")
        completion_tick("calibration", states[0].complete)
        st.info(
            "**Calibration window:** keep the ChArUco board fully visible and vary its "
            "distance, position, and tilt. Press **Space** to accept a view, **A** to "
            "toggle automatic capture, and **Q** or **Esc** to finish. Avoid collecting "
            "many nearly identical views."
        )
    with st.expander("RealSense RGB-D recording", expanded=True):
        st.caption(
            f"Selected streams: RGB {width}×{height}; native depth "
            f"{depth_width}×{depth_height}; {fps} FPS. Depth is aligned to RGB before saving."
        )
        recording_columns = st.columns(2)
        target_frames = recording_columns[0].number_input(
            "Target RGB-D frames", min_value=1, value=250, step=10,
            help="Approximately 250 varied, overlapping views are recommended for "
                 "reliable reconstruction. Recording stops automatically at this count.",
        )
        record_overwrite = recording_columns[1].checkbox(
            "Replace existing recording",
            help="Remove existing RGB and depth frames in this dataset before recording.")
        recording = recording_command(
            context, width=int(width), height=int(height), depth_width=int(depth_width),
            depth_height=int(depth_height), fps=int(fps), overwrite=record_overwrite,
            serial=camera_serial, target_frames=int(target_frames),
            camera_json=context.dataset / "camera.json")
        if st.button(
            "Open recording preview",
            help="Opens the aligned RGB and metric-depth preview in a paused state. "
                 "Press Space in that window to start capturing frames. If camera "
                 "calibration or RealSense support is missing, the recorder reports "
                 "the required input.",
        ):
            execute(recording, "recording")
        completion_tick("recording", states[1].complete)
        st.info(
            "**Recording window:** RGB and aligned metric depth are displayed "
            "side by side. The preview starts paused. Press **Space** to start or "
            "pause capture, and **Q** or **Esc** to finish early and save. The bottom "
            "bar shows the current state and saved-frame progress. Capture about "
            f"**{int(target_frames)} frames** (250 is recommended), move slowly, and "
            "maintain strong overlap between consecutive views."
        )

with tab_reconstruction:
    reconstruction_controls = st.container(border=True)
    reconstruction_controls.subheader("Reconstruction settings")
    if not pycolmap_cuda:
        reconstruction_controls.error(
            "Dense PatchMatch cannot run: the selected interpreter reports "
            "`pycolmap.has_cuda = False`. COLMAP PatchMatch has no CPU "
            "fallback. Select an environment containing a CUDA-enabled "
            "PyCOLMAP build."
        )
    c1, c2, c3 = reconstruction_controls.columns(3)
    device = c1.selectbox(
        "Compute device", ("auto", "cuda", "cpu"),
        help="Device used by supported reconstruction stages. Auto selects CUDA when "
             "available. Dense PatchMatch itself requires CUDA.")
    max_image_size = c2.number_input(
        "Maximum MVS image size", value=-1,
        help="Maximum image dimension used for dense stereo. -1 retains the original "
             "resolution; a smaller value reduces GPU memory use and processing time.")
    geometric = c3.checkbox(
        "Geometric consistency", value=True,
        help="Runs an additional consistency pass between views. It generally improves "
             "dense-cloud reliability but increases reconstruction time.")
    overlap_column, _ = reconstruction_controls.columns((1, 2))
    target_overlap = overlap_column.slider(
        "Target frame overlap",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.025,
        help="Target overlap between frames selected for dense reconstruction and "
             "SAM-6D. Higher values retain more similar views; lower values select "
             "fewer, more distinct views.",
    )
    reconstruction_overwrite = reconstruction_controls.checkbox(
        "Replace existing reconstruction",
        help="Allows existing reconstruction outputs for this dataset to be replaced."
    )
    reconstruction = reconstruction_command(
        context, device=device, overwrite=reconstruction_overwrite,
        max_image_size=int(max_image_size), geometric_consistency=geometric,
        target_overlap=float(target_overlap))
    if reconstruction_controls.button(
        "Run reconstruction",
                 disabled=(not states[2].ready or not reconstruction_runtime_ready or
                  not pycolmap_cuda),
        help="Runs SfM, filtering, dense MVS, and metric depth alignment.",
    ):
        execute(reconstruction, "reconstruction")
    completion_tick("reconstruction", states[2].complete, target=reconstruction_controls)
    reconstruction_visualization = st.container(border=True)
    reconstruction_visualization.subheader("Reconstruction visualization")
    visualizer = script_command(context, "visualize_reconstruction.py",
                                "--config", context.config)
    if reconstruction_visualization.button(
        "Open reconstruction visualizer",
        disabled=not states[2].complete or not host_modules["open3d"],
        help="Opens an interactive 3D comparison of the reconstruction stages.",
    ):
        execute(visualizer, "reconstruction_visualizer")
    completion_tick(
        "reconstruction_visualizer", label="Visualizer opened successfully",
        target=reconstruction_visualization,
    )
    reconstruction_visualization.info(
        "**Reconstruction visualization window:** **1** shows SfM; **2** filtered SfM; "
        "**3** MVS; **4** filtered MVS; and **5** metric MVS. **C** toggles camera "
        "poses; **F** toggles the reference frame; **P** switches "
        "point coloring; **R** fits the camera to the currently visible cloud; and "
        "**S** saves a screenshot. **Left-drag** rotates, **right-drag** pans, and the "
        "**wheel** zooms."
    )

with tab_sam6d:
    sam_controls = st.container(border=True)
    sam_controls.subheader("Segmentation and pose-estimation settings")
    objects = discover_objects(context)
    selected = sam_controls.selectbox(
        "Objects", ["All objects", *objects],
        help="Run pose estimation for every uploaded STL model or only the selected object.",
    )
    c1, c2, c3 = sam_controls.columns(3)
    segmentor = c1.selectbox(
        "Segmentor", ("sam", "fastsam"),
        help="SAM generally provides higher-quality masks; FastSAM is faster and uses less computation.",
    )
    stability = c2.slider(
        "Stability threshold", 0.0, 1.0, 0.85, 0.01,
        help="Minimum SAM mask-stability score. Higher values retain fewer, more stable proposals; lower values retain more candidate masks.",
    )
    detection = c3.slider(
        "Detection threshold", 0.0, 1.0, 0.20, 0.01,
        help="Minimum object-matching score passed to pose estimation. Lower values retain more candidates but may introduce false detections.",
    )
    sam_overlap_column, sam_pairing_button_column = sam_controls.columns((1, 2))
    sam_target_overlap = sam_overlap_column.slider(
        "SAM-6D frame overlap",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.025,
        help="Select a new set of RGB-D frames for ISM and PEM without rerunning or "
             "changing the reconstruction. Higher values generally retain more frames.",
    )
    sam_selection_path = context.dataset / "sam6d_frame_selection.json"
    if sam_pairing_button_column.button(
        "Recompute SAM-6D frame selection",
        disabled=not states[3].ready,
        help="Recalculate the image subset used by instance segmentation and pose "
             "estimation. The reconstructed point cloud remains unchanged.",
    ):
        execute(
            sam6d_frame_selection_command(
                context, target_overlap=float(sam_target_overlap)
            ),
            "sam_frame_selection",
        )
    completion_tick(
        "sam_frame_selection", sam_selection_path.is_file(),
        label="SAM-6D frame selection available", target=sam_controls,
    )
    use_range = sam_controls.checkbox(
        "Limit frame range",
        help="Process only a selected inclusive interval of frame identifiers instead of the complete sequence.",
    )
    start_frame = end_frame = None
    if use_range:
        c1, c2 = sam_controls.columns(2)
        start_frame = int(c1.number_input(
            "Start frame", min_value=0, value=0,
            help="First frame identifier included in this inference run."))
        end_frame = int(c2.number_input("End frame", min_value=start_frame,
                                        value=start_frame + 20,
                                        help="Last frame identifier included in this inference run."))
    sam_overwrite = sam_controls.checkbox(
        "Replace existing SAM-6D results",
        help="Recompute and replace existing per-object SAM-6D outputs. Leave disabled to reuse available results and shared segmentation proposals.",
    )
    object_name = None if selected == "All objects" else selected
    command = sam6d_command(
        context, object_name=object_name, segmentor=segmentor,
        stability=stability, detection=detection, start_frame=start_frame,
        end_frame=end_frame, overwrite=sam_overwrite, check_only=False)
    if sam_controls.button(
        "Run SAM-6D", disabled=not states[3].ready,
        help="Validate the RGB-D, camera, reconstruction, and STL inputs, then run "
             "shared instance-mask generation, duplicate-mask suppression, and "
             "multi-candidate pose estimation for the selected frames and objects.",
    ):
        execute(command, "sam6d")
    completion_tick("sam6d", states[3].complete, target=sam_controls)

    sam_visualization = st.container(border=True)
    sam_visualization.info(
        "**Instance annotation and pose estimation**\n\n"
        "Inspect the object-specific instance-segmentation result and pose-estimation "
        "result for the same frame."
    )
    visualization_objects = []
    for name in objects:
        frames_dir = context.dataset / "sam6d_sequences" / name / "frames"
        review_dir = context.dataset / "sam6d_scene" / "review" / "sam6d" / name
        if ((frames_dir.is_dir() and
             any(frames_dir.glob("*/sam6d_results/vis_ism.png"))) or
                (review_dir.is_dir() and any(review_dir.glob("*/vis_ism.png")))):
            visualization_objects.append(name)
    if visualization_objects:
        visualization_object = sam_visualization.selectbox(
            "Visualization object", visualization_objects,
            key="sam_visualization_object",
            help="Choose the object whose instance mask and estimated pose should be inspected.",
        )
        object_frames_dir = (
            context.dataset / "sam6d_sequences" / visualization_object / "frames"
        )
        visualization_frames: dict[int, tuple[Path, Path]] = {}
        if object_frames_dir.is_dir():
            for frame_dir in object_frames_dir.iterdir():
                if not frame_dir.is_dir() or not frame_dir.name.isdigit():
                    continue
                results_dir = frame_dir / "sam6d_results"
                ism_path = results_dir / "vis_ism.png"
                pem_path = results_dir / "vis_pem.png"
                if not pem_path.is_file():
                    pem_path = (
                        context.dataset / "experimental_multi_instance" / "pem"
                        / visualization_object / "frames" / frame_dir.name
                        / "sam6d_results" / "vis_pem.png"
                    )
                if ism_path.is_file() and pem_path.is_file():
                    visualization_frames[int(frame_dir.name)] = (ism_path, pem_path)
        if not visualization_frames:
            review_dir = (
                context.dataset / "sam6d_scene" / "review" / "sam6d"
                / visualization_object
            )
            if review_dir.is_dir():
                for frame_dir in review_dir.iterdir():
                    if not frame_dir.is_dir() or not frame_dir.name.isdigit():
                        continue
                    ism_path = frame_dir / "vis_ism.png"
                    pem_path = frame_dir / "vis_pem.png"
                    if ism_path.is_file() and pem_path.is_file():
                        visualization_frames[int(frame_dir.name)] = (ism_path, pem_path)
        if visualization_frames:
            visualization_frame = sam_visualization.selectbox(
                "Inspect frame", sorted(visualization_frames),
                format_func=lambda value: f"Frame {value:06d}",
                key=f"sam_visualization_frame_{visualization_object}",
                help="Choose a frame with both object-specific ISM and PEM visualizations.",
            )
            ism_path, pem_path = visualization_frames[visualization_frame]
            with Image.open(ism_path) as source:
                ism_composite = source.convert("RGB")
            with Image.open(pem_path) as source:
                pem_image = source.convert("RGB")
            if ism_composite.width > ism_composite.height * 2.3:
                split = ism_composite.width // 2
                rgb_image = ism_composite.crop((0, 0, split, ism_composite.height))
                ism_image = ism_composite.crop(
                    (split, 0, ism_composite.width, ism_composite.height)
                )
            else:
                ism_image = ism_composite
                rgb_image = None
                for candidate in (context.dataset / "images").iterdir():
                    digits = "".join(filter(str.isdigit, candidate.stem))
                    if (candidate.is_file() and digits and
                            int(digits) == visualization_frame):
                        with Image.open(candidate) as source:
                            rgb_image = source.convert("RGB")
                        break
                if rgb_image is None:
                    rgb_image = ism_image.copy()
            if pem_image.width > pem_image.height * 2.3:
                split = pem_image.width // 2
                pem_image = pem_image.crop(
                    (split, 0, pem_image.width, pem_image.height)
                )
            target_size = ism_image.size
            rgb_image = ImageOps.fit(rgb_image, target_size)
            pem_image = ImageOps.fit(pem_image, target_size)
            rgb_column, ism_column, pem_column = sam_visualization.columns(3)
            rgb_column.image(
                rgb_image, caption=f"{visualization_object} — RGB image",
                width="stretch",
            )
            ism_column.image(
                ism_image, caption=f"{visualization_object} — instance segmentation",
                width="stretch",
            )
            pem_column.image(
                pem_image, caption=f"{visualization_object} — pose estimation",
                width="stretch",
            )
        else:
            sam_visualization.warning(
                f"No frames contain both `vis_ism.png` and `vis_pem.png` for "
                f"{visualization_object}."
            )
    else:
        sam_visualization.warning(
            "No object-specific ISM and PEM visualizations are available yet. Run "
            "instance segmentation and pose estimation first."
        )

with tab_pose:
    pose_controls = st.container(border=True)
    pose_controls.subheader("Pose-fitting settings")
    objects = discover_objects(context)
    available_pose_files = [
        context.dataset / "sam6d_sequences" / name / f"{name}_world_pose.json"
        for name in objects
    ]
    has_visualizable_pose = any(path.is_file() for path in available_pose_files)
    selected = pose_controls.selectbox(
        "Objects", ["All objects", *objects],
        help="Fit poses for every object with per-frame pose estimates or only the selected object.",
    )
    c1, c2, c3 = pose_controls.columns(3)
    translation = c1.number_input(
        "Translation threshold (m)", value=0.01, min_value=0.0001,
        format="%.4f",
        help="Maximum position distance for per-frame estimates to belong to the same object-instance cluster.")
    rotation = c2.number_input(
        "Initial rotation threshold (deg)", value=5.0, min_value=0.1,
        help="Initial angular tolerance used when selecting a consistent orientation inside each position cluster.")
    maximum_rotation = c3.number_input(
        "Maximum rotation threshold (deg)", value=30.0,
        min_value=float(rotation),
        help="Largest angular tolerance allowed if orientation fitting must relax the initial threshold.")
    minimum_inliers = c1.number_input(
        "Minimum inliers", value=3, min_value=1,
        help="Minimum number of supporting estimates from unique frames required to accept an object pose.")
    symmetry = c2.selectbox(
        "CAD symmetry", ("auto", "none"),
        help="Auto detects rotational equivalences from the object mesh; none treats every orientation as distinct.")
    min_silhouette_iou = c3.number_input(
        "Min. silhouette IoU", value=float(os.environ.get("LABEL_FACTORY_MIN_SILHOUETTE_IOU", "0")),
        min_value=0.0, max_value=1.0, step=0.05,
        help="Rejects a pose cluster if the CAD silhouette at the fused pose overlaps the "
             "observed instance masks with a median IoU below this value. Removes objects that "
             "look like the model but have a different geometry (about 0.7). 0 keeps every "
             "cluster; the IoU is stored in the pose file either way.")
    command = ransac_command(
        context, object_name=None if selected == "All objects" else selected,
        translation=float(translation), rotation=float(rotation),
        maximum_rotation=float(maximum_rotation), minimum_inliers=int(minimum_inliers),
        symmetry=symmetry, min_silhouette_iou=float(min_silhouette_iou))
    if pose_controls.button(
        "Run pose fitting", disabled=not states[4].ready,
        help="Cluster position estimates first, then robustly fit an orientation for each detected instance.",
    ):
        execute(command, "ransac")
        # The command may have created only a subset of the requested poses.
        # Refresh availability immediately without requiring a page reload.
        has_visualizable_pose = any(
            path.is_file() for path in available_pose_files
        )
    completion_tick("ransac", states[4].complete, target=pose_controls)
    pose_results = st.container(border=True)
    pose_results.subheader("Fitted objects")
    pose_summary_path = context.dataset / "sam6d_sequences/ransac_run_summary.json"
    if pose_summary_path.is_file():
        try:
            pose_summary = json.loads(pose_summary_path.read_text())
            found_objects = pose_summary.get("found_objects")
            if found_objects is None:
                found_objects = {
                    name: 1 for name in pose_summary.get("objects_succeeded", [])
                }
            non_found_objects = pose_summary.get("non_found_objects")
            if non_found_objects is None:
                non_found_objects = [
                    item.get("object_name", "unknown")
                    for item in pose_summary.get("objects_excluded", [])
                ]
            summary_columns = pose_results.columns(2)
            if found_objects:
                summary_columns[0].success(
                    "**Found objects:**\n\n" + "\n".join(
                        f"- {name} : {count}"
                        for name, count in found_objects.items()
                    )
                )
            else:
                summary_columns[0].warning("**Found objects:**\n\nNone")
            if non_found_objects:
                summary_columns[1].error(
                    "**Objects not found:**\n\n" + "\n".join(
                        f"- {name}" for name in non_found_objects
                    )
                )
            else:
                summary_columns[1].success("**Objects not found:**\n\nNone")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            pose_results.warning(f"Could not read the pose-fitting summary: {error}")
    pose_visualization = st.container(border=True)
    pose_visualization.subheader("Pose-fitting visualization")
    pose_visualizer = script_command(context, "visualize_ransac_poses.py",
                                     "--config", context.config)
    pose_visualization.info(
        "**Pose-fitting visualization:** the position plot shows spatial clusters and "
        "their accepted instance centers. The orientation plot shows the corresponding "
        "rotations in axis-angle space, making orientation spread and outliers visible."
    )
    if pose_visualization.button(
        "Generate pose-fitting visualization", disabled=not has_visualizable_pose,
        help="Create position-cluster and axis-angle orientation plots from the accepted fitting outputs.",
    ):
        execute(pose_visualizer, "ransac_visualizer")
    ransac_visualization_dir = (
        context.dataset / "sam6d_scene/ransac_visualizations"
    )
    light_position_plot = ransac_visualization_dir / "ransac_positions.png"
    dark_position_plot = ransac_visualization_dir / "ransac_positions_dark.png"
    light_orientation_plot = (
        ransac_visualization_dir / "ransac_orientations_axis_angle.png"
    )
    dark_orientation_plot = (
        ransac_visualization_dir / "ransac_orientations_axis_angle_dark.png"
    )
    position_plot = themed_file(light_position_plot, dark_position_plot)
    orientation_plot = themed_file(light_orientation_plot, dark_orientation_plot)
    completion_tick(
        "ransac_visualizer",
        position_plot.is_file() and orientation_plot.is_file(),
        label="Visualization generated",
        target=pose_visualization,
    )
    if position_plot.is_file() or orientation_plot.is_file():
        position_column, orientation_column = pose_visualization.columns(2)
        if position_plot.is_file():
            position_column.image(
                position_plot.read_bytes(),
                caption="World-coordinate object positions",
                width="stretch",
            )
        if orientation_plot.is_file():
            orientation_column.image(
                orientation_plot.read_bytes(),
                caption="Object orientations in axis-angle representation",
                width="stretch",
            )

with tab_scene:
    annotation_box = st.container(border=True)
    annotation_box.subheader("Scene annotation generation")
    c1, c2 = annotation_box.columns(2)
    model_scale = c1.number_input(
        "STL unit to metre scale", value=0.001, min_value=0.000001,
        format="%.6f",
        help="Multiplier that converts STL coordinates to metres. Use 0.001 when the "
             "object meshes are stored in millimetres.")
    annotation_overwrite = c2.checkbox(
        "Replace scene annotations",
        help="Regenerate and replace existing scene masks, poses, and annotation files."
    )
    command = annotation_command(context, model_scale=float(model_scale),
                                 overwrite=annotation_overwrite)
    visualization_dir = context.dataset / "sam6d_scene/visualizations"
    images = sorted(visualization_dir.glob("*.png")) if visualization_dir.is_dir() else []
    if annotation_box.button(
        "Generate scene annotations", disabled=not states[5].ready,
        help="Place fitted object meshes in the metric reconstruction and render "
             "per-frame instance masks using the known camera poses, then generate "
             "colored overlays on the original RGB images.",
    ):
        execute(command, "annotations")
    completion_tick(
        "annotations", states[5].complete and bool(images),
        label="Scene annotations and overlays generated",
        target=annotation_box,
    )
    video_box = st.container(border=True)
    video_box.subheader("Overlay video")
    c1, c2 = video_box.columns(2)
    video_fps = c1.number_input(
        "Video FPS", min_value=1.0, max_value=120.0, value=20.0, step=1.0,
        help="Playback rate of the review video. It does not change annotation timing "
             "or the original capture data.")
    video_overwrite = c2.checkbox(
        "Replace existing overlay video", value=True,
        help="Allow a previously generated annotation review video to be replaced."
    )
    video_command = annotation_video_command(
        context, fps=float(video_fps), overwrite=video_overwrite)
    video_path = context.dataset / "sam6d_scene/annotation_overlays.mp4"
    if video_box.button(
        "Create overlay video", disabled=not bool(images),
        help="Encode the ordered overlay frames into a browser-playable review video."
    ):
        execute(video_command, "annotation_video")
    completion_tick("annotation_video", video_path.is_file(), target=video_box)
    if video_path.is_file():
        video_box.video(video_path.read_bytes(), format="video/mp4", autoplay=True,
                        muted=True, loop=True)

with tab_publish:
    review_section = st.container(border=True)
    review_section.subheader("Dataset content")
    review_section.write(
        "Inspect the generated RGB-D frames, masks, object poses, and visual evidence "
        "below. Approval is always made by a person."
    )
    review_section.info(
        "**Included dataset content**\n\n"
        "- RGB images and metric depth maps\n"
        "- Camera calibration (`camera.json`) and metric camera poses\n"
        "- Scene annotations (`annotations.json`) and the scene manifest\n"
        "- Instance masks, raw segmentations, annotation overlays, and the review video\n"
        "- Per-frame object-pose JSON files, fitted world-pose JSON files, and the "
        "pose-fitting run summary\n"
        "- Metric reconstructed point cloud and alignment metadata\n"
        "- STL object models and task-specific Parquet datasets"
    )
    split_section = st.container(border=True)
    split_section.subheader("Dataset splitting")
    dataset_licenses = {
        "other": "Other / custom license",
        "unknown": "Unknown license",
        "cc0-1.0": "Creative Commons Zero 1.0",
        "cc-by-4.0": "Creative Commons Attribution 4.0",
        "cc-by-sa-4.0": "Creative Commons Attribution-ShareAlike 4.0",
        "cc-by-nc-4.0": "Creative Commons Attribution-NonCommercial 4.0",
        "cc-by-nc-sa-4.0": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
        "cc-by-nd-4.0": "Creative Commons Attribution-NoDerivatives 4.0",
        "cc-by-nc-nd-4.0": "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0",
        "cdla-permissive-2.0": "Community Data License Agreement – Permissive 2.0",
        "cdla-permissive-1.0": "Community Data License Agreement – Permissive 1.0",
        "cdla-sharing-1.0": "Community Data License Agreement – Sharing 1.0",
        "odc-by": "Open Data Commons Attribution",
        "odbl": "Open Database License",
        "pddl": "Open Data Commons Public Domain Dedication and License",
        "apache-2.0": "Apache License 2.0",
        "mit": "MIT License",
        "bsd-3-clause": "BSD 3-Clause License",
        "gpl-3.0": "GNU General Public License 3.0",
    }
    split_columns = split_section.columns(3)
    train_ratio = split_columns[0].number_input(
        "Training ratio", min_value=0.05, max_value=0.95, value=0.80, step=0.05,
        help="Fraction of frames assigned to the training split.")
    validation_ratio = split_columns[1].number_input(
        "Validation ratio", min_value=0.0, max_value=0.90, value=0.10, step=0.05,
        help="Fraction of frames assigned to the validation split.")
    test_ratio = 1.0 - float(train_ratio) - float(validation_ratio)
    split_columns[2].text_input(
        "Testing ratio", value=f"{max(0.0, test_ratio):.2f}", disabled=True,
        help="Computed remainder after the training and validation ratios."
    )
    split_seed = split_section.number_input(
        "Split seed", min_value=0, value=42, step=1,
        help="Controls the deterministic frame assignment. Reusing the same seed "
             "and ratios preserves the same train, validation, and test split."
    )
    split_valid = test_ratio > 0
    if not split_valid:
        split_section.error(
            "Training and validation ratios must leave a positive testing ratio."
        )
    split_section.caption(
        "The same dataset ID, frame ID, and seed always produce the same split. "
        "New numbered datasets are appended without changing earlier assignments."
    )
    inventory_command = huggingface_command(
        context, repo_id="", license_name="other",
        private=True, upload=False, train_ratio=float(train_ratio),
        validation_ratio=float(validation_ratio), split_seed=int(split_seed),
    )
    cleanup_command = script_command(
        context, "cleanup_dataset_intermediates.py",
        "--config", context.config, "--dataset-dir", context.dataset, "--apply",
    )
    review_section.warning(
        "Cleanup permanently removes reproducible reconstruction work files, "
        "SAM/DINO caches, rendered templates, and per-frame diagnostic folders. "
        "RGB-D inputs, STL models, metric reconstruction, camera poses, fitted "
        "object poses, final annotations, masks, overlays, and publication files "
        "are retained. Run it only after pose fitting and scene annotation finish."
    )
    if review_section.button(
        "Cleanup the unnecessary files",
        help="Free storage by deleting only reproducible intermediate files. "
             "The operation refuses to run while required final outputs are missing.",
        type="secondary",
    ):
        execute(cleanup_command, "dataset_cleanup")
    cleanup_report_path = context.dataset / "publishing/cleanup_report.json"
    completion_tick(
        "dataset_cleanup", cleanup_report_path.is_file(),
        label="Intermediate files cleaned", target=review_section,
    )
    if st.session_state.pop("refresh_inventory_on_open", False):
        with review_section.spinner("Refreshing dataset content inventory…"):
            automatic_inventory = subprocess.run(
                inventory_command, cwd=str(context.project),
                capture_output=True, text=True,
            )
        st.session_state["automatic_inventory_log"] = (
            automatic_inventory.stdout + automatic_inventory.stderr
        )
        if automatic_inventory.returncode != 0:
            review_section.warning(
                "Inventory refreshed, but required dataset content is incomplete. "
                "Review the reported items below."
            )
    if review_section.button(
        "Refresh dataset content inventory",
        help="Scan the current dataset, validate required outputs, calculate its "
             "size, and list every file selected for publication."
    ):
        execute(inventory_command, "huggingface_inventory")

    report_path = context.dataset / "publishing/huggingface_inventory.json"
    completion_tick(
        "huggingface_inventory", report_path.is_file(),
        label="Dataset inventory generated",
        target=review_section,
    )
    report = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Could not read inventory report: {error}")
    if report:
        stats = report["statistics"]
        if report["technically_ready"]:
            review_section.success("Dataset content inventory refreshed.")
        else:
            review_section.error("Required technical inputs are incomplete.")
        c1, c2, c3, c4 = review_section.columns(4)
        c1.metric(
            "RGB-D frames", stats["rgb_frames"],
            help="Number of matched RGB images and metric depth maps."
        )
        c2.metric(
            "Objects", stats["annotated_objects"],
            help="Number of object instances recorded in the scene annotations."
        )
        c3.metric(
            "Upload files", stats["upload_files"],
            help="Total number of source assets and generated Parquet files selected."
        )
        c4.metric(
            "Upload size", f"{stats['upload_bytes'] / 1_073_741_824:.2f} GiB",
            help="Approximate combined size of all files selected for publication."
        )
        review_section.caption(
            "Hugging Face destination directory: "
            f"`{report.get('remote_dataset_directory', context.dataset_id)}/`"
        )
        for error in report["errors"]:
            review_section.error(error)
        for warning in report["warnings"]:
            # Older inventory reports may still contain the review reminders that
            # were removed from newly generated reports. Do not render those stale
            # messages while waiting for the next inventory refresh.
            if not warning.startswith("Human review required:"):
                review_section.warning(warning)
        with review_section.expander("Files selected for upload"):
            included = report["included_files"]
            st.code("\n".join(included[:500]), language=None, height=480)
            if len(included) > 500:
                st.caption(f"Showing 500 of {len(included)} files; download the report for the complete list.")
            st.download_button(
                "Download inventory report", data=report_path.read_bytes(),
                file_name=f"{context.dataset_id}_huggingface_inventory.json",
                mime="application/json",
                help="Download the complete machine-readable validation results and "
                     "publication file list.",
            )

    ransac_dir = context.dataset / "sam6d_scene/ransac_visualizations"
    dark_position_plot = ransac_dir / "ransac_positions_dark.png"
    dark_orientation_plot = ransac_dir / "ransac_orientations_axis_angle_dark.png"
    position_plot = themed_file(
        ransac_dir / "ransac_positions.png", dark_position_plot
    )
    orientation_plot = themed_file(
        ransac_dir / "ransac_orientations_axis_angle.png", dark_orientation_plot
    )
    plot_columns = review_section.columns(2)
    if position_plot.is_file():
        plot_columns[0].image(str(position_plot), caption="Pose-fitting position clusters",
                              width="stretch")
    else:
        plot_columns[0].warning("Pose-fitting position visualization is missing.")
    if orientation_plot.is_file():
        plot_columns[1].image(str(orientation_plot), caption="Pose-fitting orientations",
                              width="stretch")
    else:
        plot_columns[1].warning("Pose-fitting orientation visualization is missing.")

    review_video = context.dataset / "sam6d_scene/annotation_overlays.mp4"
    if review_video.is_file():
        review_section.video(review_video.read_bytes(), format="video/mp4")
    else:
        review_section.warning(
            "Annotation review video is missing. Create it in Scene & annotation."
        )

    rgb_files = indexed_images(context.dataset / "images")
    depth_files = indexed_images(context.dataset / "depth_maps")
    mask_files = indexed_images(context.dataset / "sam6d_scene/masks")
    overlay_files = indexed_images(context.dataset / "sam6d_scene/visualizations")
    review_frames = sorted(set(rgb_files) & set(depth_files) & set(mask_files))
    if review_frames:
        frame_id = review_section.selectbox(
            "Inspect frame", review_frames,
            format_func=lambda value: f"Frame {value:06d}",
            help="Choose one synchronized frame to compare its RGB image, metric "
                 "depth, instance mask, and rendered annotation overlay."
        )
        frame_columns = review_section.columns(4)
        frame_columns[0].image(str(rgb_files[frame_id]), caption="RGB",
                               width="stretch")
        frame_columns[1].image(depth_preview(depth_files[frame_id]), caption="Metric depth",
                               clamp=True, width="stretch")
        frame_columns[2].image(label_preview(mask_files[frame_id]), caption="Instance mask",
                               width="stretch")
        if frame_id in overlay_files:
            frame_columns[3].image(str(overlay_files[frame_id]), caption="Overlay",
                                   width="stretch")
        else:
            frame_columns[3].warning("Overlay unavailable for this frame.")
    else:
        review_section.warning(
            "No matching RGB, depth, and mask frame is available for inspection."
        )

    publishing_section = st.container(border=True)
    publishing_section.subheader("Hugging Face publishing")
    publishing_section.caption(
        "Upload includes curated final dataset assets, not COLMAP databases, "
        "MVS stereo caches, templates, or per-frame inference intermediates."
    )
    license_name = publishing_section.selectbox(
        "Dataset license", tuple(dataset_licenses),
        format_func=lambda identifier: (
            f"{dataset_licenses[identifier]} ({identifier})"
        ),
        help="Choose a Hugging Face-supported identifier that is compatible with "
             "every image, annotation, object model, and other included asset. This "
             "selection is not legal advice.",
    )
    repo_id = publishing_section.text_input(
        "Dataset repository", placeholder="username/dataset-name",
        help="Target Hugging Face dataset repository in owner/name format. The "
             "authenticated account must be allowed to open a pull request there."
    )
    private_repo = publishing_section.checkbox(
        "Private Hugging Face repository", value=True,
        help="Create a new destination repository as private. This does not change "
             "the visibility of an existing repository."
    )
    hf_token = publishing_section.text_input(
        "Hugging Face write token", type="password",
        help="Create a write token at huggingface.co/settings/tokens. "
             "It is passed only as HF_TOKEN to the upload process."
    )
    publication_modules = interpreter_modules(
        context.host_python, ("huggingface_hub", "pyarrow", "datasets")
    )
    hf_available = publication_modules["huggingface_hub"]
    pyarrow_available = publication_modules["pyarrow"]
    datasets_available = publication_modules["datasets"]
    if not hf_available or not pyarrow_available or not datasets_available:
        missing_publication_modules = [
            name for name, available in publication_modules.items() if not available
        ]
        publishing_section.error(
            "Publication dependencies are missing from Reconstruction Python: "
            f"`{', '.join(missing_publication_modules)}`. Run "
            "`LabelFactory/bin/python3 -m pip install -r ui/requirements.txt`."
        )
    upload_blockers = []
    if not report:
        upload_blockers.append("Run **Refresh dataset content inventory** first.")
    elif not report.get("technically_ready"):
        upload_blockers.append("Resolve the errors reported by the technical inventory.")
    if not repo_id.strip():
        upload_blockers.append("Enter a Hugging Face dataset repository as `owner/name`.")
    elif repo_id.count("/") != 1:
        upload_blockers.append("The repository must use the `owner/name` format.")
    if not hf_token.strip():
        upload_blockers.append("Enter a Hugging Face token with write permission.")
    if not hf_available:
        upload_blockers.append("Install `huggingface_hub` in Reconstruction Python.")
    if not pyarrow_available:
        upload_blockers.append("Install `pyarrow` in Reconstruction Python.")
    if not datasets_available:
        upload_blockers.append("Install `datasets` in Reconstruction Python.")
    if not split_valid:
        upload_blockers.append("Choose split ratios that leave a positive testing split.")
    if upload_blockers:
        publishing_section.info(
            "Upload requirements:\n\n- " + "\n- ".join(upload_blockers)
        )
    upload_command = huggingface_command(
        context, repo_id=repo_id.strip(), license_name=license_name,
        private=private_repo, upload=True,
        create_pr=True,
        train_ratio=float(train_ratio), validation_ratio=float(validation_ratio),
        split_seed=int(split_seed),
    )
    if publishing_section.button(
        "Upload reviewed dataset to Hugging Face", type="primary",
        help="Generate the task-specific Parquet files, stage the selected dataset "
             "content, and submit it through a Hugging Face pull request."
    ):
        if upload_blockers:
            publishing_section.error(
                "The upload cannot start yet. Complete the requirements shown above."
            )
        else:
            execute(upload_command, "huggingface_upload",
                    environment_overrides={"HF_TOKEN": hf_token.strip()})
    completion_tick(
        "huggingface_upload", label="Pull request submitted successfully",
        target=publishing_section,
    )


if "last_run" in st.session_state:
    st.divider()
    last_run = st.session_state["last_run"]
    result_level = last_run.get(
        "result_level", "success" if last_run["return_code"] == 0 else "error"
    )
    if result_level == "warning":
        st.warning(f"Last process completed with warnings: {last_run['stage']}")
    elif result_level == "error":
        st.error(
            f"Last process failed ({last_run['return_code']}): {last_run['stage']}"
        )
    else:
        st.success(f"Last process completed successfully: {last_run['stage']}")
    with st.expander("Last process log", expanded=result_level != "success"):
        st.code(last_run["output"] or "(no output)", language=None,
                height=480, wrap_lines=False)
        if st.button("Clear process log"):
            del st.session_state["last_run"]
            st.rerun()
