# Usage guide (command line)

This guide describes the complete Label Factory workflow from the command line.
The same stages are available in the graphical interface, which is described
step by step in the [user interface guide](UI_GUIDE.md). Installation is
described in [`INSTALL_DOCKER.md`](INSTALL_DOCKER.md) and
[`INSTALL_NATIVE.md`](INSTALL_NATIVE.md).

**Contents**

- [Configuration](#configuration)
- [Input and output layout](#input-and-output-layout)
- [1. Camera calibration](#1-camera-calibration)
- [2. Record an aligned RGB-D sequence](#2-record-an-aligned-rgb-d-sequence)
- [3. Reconstruct and scale the scene](#3-reconstruct-and-scale-the-scene)
- [4. Known-object poses with SAM-6D](#4-known-object-poses-with-sam-6d)
- [5. Create the posed scene and frame annotations](#5-create-the-posed-scene-and-frame-annotations)
- [6. Review and publish to Hugging Face](#6-review-and-publish-to-hugging-face)
- [7. Evaluate on YCB-Video (BOP19)](#7-evaluate-on-ycb-video-bop19)
- [8. Measure the runtime](#8-measure-the-runtime)
- [Troubleshooting](#troubleshooting)
- [Reproducibility checklist](#reproducibility-checklist)

## Configuration

Edit `scripts/configs/default.yaml`:

```yaml
path: .
dataset_id: "0000"
fps: 5
n_points: 20000
```

`dataset_id: 0` resolves to `data/0000`; `dataset_id: 3` resolves to
`data/0003`. The interface changes this entry when you select a dataset.

The camera intrinsics are not configured here. They come from the dataset's `camera.json`, see [Input and output layout](#input-and-output-layout).

## Input and output layout

The inputs you provide are the **CAD models** of the objects and an **RGB-D
recording** of the scene. The camera calibration is **not** an input you have to
supply: the framework creates it (step 1) and reuses it for every later recording
with the same camera and resolution. If you already have a calibration, it can be
used instead (see step 1). Only a sequence recorded outside the framework must be
imported together with its `camera.json`.

```text
data/0000/
├── camera.json                 intrinsics; written by the calibration/recording step
├── images/000000.png ...       RGB, written by the recording step
├── depth_maps/depth_000000.png ...   depth in millimetres (uint16), aligned to RGB
└── objects/
    ├── stl/Pliers.stl          CAD model (you provide this)
    └── textured/               optional textured model
        ├── Pliers.obj
        ├── Pliers.mtl
        └── Pliers.png
```

RGB and depth files are paired by numeric frame ID. STL files are required for
the known-object SAM-6D workflow; their file names become the object names. If a
complete textured OBJ exists, it is used for the template rendering.

The main outputs, created by the steps below:

```text
data/0000/
├── reconstruction/             sparse model, metric dense cloud, camera poses
├── sam6d_sequences/<object>/   per-frame pose candidates and <object>_world_pose.json
├── sam6d_scene/                annotations.json, instance masks, overlays, video
└── publishing/                 file inventory and task-specific Parquet tables
```

## 1. Camera calibration

The framework creates the calibration itself, so it is not a required input. Do it
**once per camera and RGB resolution**; the result is stored in
`calibration/cameras/<camera>_<date>/camera.json` and reused afterwards. If you
already have a calibration for the camera, you can skip this step and use it: copy
its `camera.json` into the dataset folder (`data/0000/camera.json`), or activate it
in the interface (**Capture** tab, *Use selected calibration*).

Capture ChArUco views and estimate the RGB intrinsics:

```bash
conda activate label-factory
python scripts/calibrate_charuco_camera.py \
  --config scripts/configs/default.yaml \
  --serial CAMERA_SERIAL \
  --width 1280 --height 800 \
  --squares-x 7 --squares-y 5 \
  --square-length 0.030 \
  --marker-length 0.023 \
  --views 25 \
  --output-dir calibration/cameras/Intel_RealSense_D455_2026-09-03 \
  --activate-dataset
```

Use `--input-dir` to calibrate from existing images and `--generate-board` to
write a printable board. Incorrect physical board dimensions produce an
incorrect metric calibration. The Capture tab can create dated, reusable
calibrations or activate a previous calibration. Activating one copies its
`camera.json` into the current dataset, which keeps reconstruction and SAM-6D
independent of the calibration-library layout.

## 2. Record an aligned RGB-D sequence

```bash
conda activate label-factory
python scripts/record_realsense_rgbd.py \
  --config scripts/configs/default.yaml \
  --width 1280 --height 800 \
  --depth-width 1280 --depth-height 720 \
  --fps 30
```

Any camera supported by the RealSense SDK (`pyrealsense2`) can be used. The requested stream combination must be supported by the connected camera. If
`Couldn't resolve requests` occurs, lower the resolution or FPS. Keep enough
texture, avoid motion blur, and observe every object from several angles.

## 3. Reconstruct and scale the scene

The reconstruction script performs feature extraction/matching, sparse SfM,
sparse filtering, MVS densification, dense filtering, and optional RGB-D metric
alignment:

```bash
conda activate label-factory
python scripts/reconstruct_rgbd_scene.py \
  --config scripts/configs/default.yaml
```

Use `--overwrite` only when the existing reconstruction may be replaced. To
perform metric scaling separately:

```bash
python scripts/scale_reconstruction_from_depth.py \
  --config scripts/configs/default.yaml
```

Important outputs:

```text
data/0000/reconstruction/
├── sfm/                              Sparse COLMAP model
├── sfm_filtered/                     Filtered sparse model
├── mvs/                              MVS workspace
├── mvs_filtered.ply                  Dense cloud, SfM scale
├── mvs_filtered_metric.ply           Dense cloud, metres
└── metric_alignment/
    ├── camera_poses_metric.json       Metric camera-to-world poses
    ├── scale.npy
    ├── rotation.npy
    ├── translation.npy
    └── depth_scale_report.json
```

Visualize the configured reconstruction:

```bash
python scripts/visualize_reconstruction.py \
  --config scripts/configs/default.yaml
```

## 4. Known-object poses with SAM-6D

Place known CAD models under `data/####/objects/stl/`. Each filename stem is an
object name.

**Optional textures.** If an object has a textured model, add it to
`data/####/objects/textured/` as `<name>.obj` + `<name>.mtl` + the texture image
(e.g. `<name>.png`) that the MTL references. The STL stays required: it provides
the geometry for every stage. The textured model must use the **same coordinates
and units as the STL** (the OBJ is imported unchanged, without Blender's usual
Y-up conversion). It is used automatically, and only, for rendering the SAM-6D
templates, which lets the segmentation and pose networks match appearance as well
as shape. If the textured model is missing or incomplete (no `mtllib`, no texture
image, missing file) the pipeline prints a warning and uses the STL, so existing
datasets behave as before. Adding, changing or removing a texture re-renders that
object's templates and re-runs its SAM-6D stage. Use `--texture off` (or
`LABEL_FACTORY_TEXTURE=off`) to ignore textures without deleting them, for example
to compare textured and untextured runs.

Run the sequence:

```bash
python scripts/SAM6D_batch_inference.py \
  --config scripts/configs/default.yaml \
  --python /path/to/conda/envs/sam6d/bin/python \
  --segmentor-model sam \
  --stability-score-thresh 0.85 \
  --det-score-thresh 0.2
```

Before a long run, validate resolved inputs:

```bash
python scripts/SAM6D_batch_inference.py \
  --config scripts/configs/default.yaml \
  --python /path/to/conda/envs/sam6d/bin/python \
  --check-inputs
```

Templates and results are isolated under
`data/####/sam6d_sequences/<object_name>/`. Raw SAM proposals are shared under:

```text
data/0000/sam6d_shared/ism/frames/<frame_id>/sam_proposals.npz
```

This prevents rerunning SAM for every CAD object. PEM remains object-specific.

Estimate stable world-coordinate poses and visualize them:

```bash
python scripts/RANSAC_object_pose.py --config scripts/configs/default.yaml
python scripts/visualize_ransac_poses.py --config scripts/configs/default.yaml
```

In the UI, **Generate RANSAC visualization** embeds the position and axis-angle
plots directly in the Pose RANSAC tab. The PNG files are also stored under
`data/####/sam6d_scene/ransac_visualizations/`.

RANSAC is evaluated independently for each object. If some objects do not
reach the required unique-frame support, their stale world-pose files are
archived and the remaining successful objects continue through the workflow.
The run summary is written to
`data/####/sam6d_sequences/ransac_run_summary.json`. The command fails only
when no requested object produces a usable pose.

**Rejecting look-alike objects (silhouette check).** Consistency between frames does
not exclude a different object that merely resembles the CAD model: it produces a
stable position cluster, sometimes the strongest one. After consolidation every
cluster is therefore also checked against the CAD model itself. The CAD silhouette is
projected at the fused pose into each supporting frame and compared with the instance
mask the pose estimator used (median IoU over up to `--verification-max-frames`
frames). Correctly matched objects typically reach 0.8-0.95; an object with a different
geometry clearly less (for example 0.6 in the lab-ware scene that motivated the check).
The IoU is always stored per instance as `silhouette_iou_median`. Clusters are removed
only when a threshold is set:

```bash
python scripts/fit_multi_instance_poses.py \
  --config scripts/configs/default.yaml --min-silhouette-iou 0.7
```

`0` (the default) keeps every cluster. The default can be set once with the
environment variable `LABEL_FACTORY_MIN_SILHOUETTE_IOU` (for example in
`compose.yaml`), and the UI offers it as **Min. silhouette IoU**. Rejected clusters are
listed under `ransac.rejected_pose_clusters` with reason
`silhouette_iou_below_threshold`; if all clusters of an object are rejected, the object
is excluded like any other object without a usable pose, and the low-support fallback
cannot bring it back. Thin objects and masks of poor quality lower the IoU, so choose
the threshold from the values stored for the objects that you know are correct. The
check uses surface samples and a z-buffer instead of a renderer, so it needs no extra
dependencies.

## 5. Create the posed scene and frame annotations

After RANSAC has produced stable world-coordinate object poses, construct the
scene and render masks for every registered camera:

```bash
python scripts/create_blender_scene_annotations.py \
  --config scripts/configs/default.yaml \
  --python /path/to/conda/envs/sam6d/bin/python
```

Objects without a valid current `*_world_pose.json` are reported and excluded
from the Blender scene. They are also recorded under `excluded_objects` in the
scene manifest.

By default, STL coordinates are interpreted as millimetres and converted to
metres with `--model-scale 0.001`. Override this only when the CAD models use a
different unit. Generated scene data is stored under:

```text
data/0000/sam6d_scene/
├── annotations.json
├── masks/
└── segmentation_raw/
```

Create RGB overlays with a stable, visible color for each object instance:

```bash
python scripts/visualize_scene_annotations.py \
  --config scripts/configs/default.yaml \
  --overwrite
```

The overlays are written to `data/0000/sam6d_scene/visualizations/`.

Encode the overlays as a 20 FPS MP4:

```bash
python scripts/create_annotation_video.py \
  --config scripts/configs/default.yaml \
  --fps 20 \
  --overwrite
```

The UI exposes the same action and plays
`data/####/sam6d_scene/annotation_overlays.mp4` directly in the browser.

## 6. Review and publish to Hugging Face

The **Review & publish** tab provides visual review evidence, a technical file
inventory and a guarded upload. A dataset cannot be uploaded until

1. the technical inventory confirms that the required files are present,
2. a person has reviewed the reconstruction, the pose-fitting plots, the
   RGB/depth/mask frames and the overlay video, and
3. a valid `owner/dataset-name` repository ID and a write token are provided.

Human review is intentionally not a checkbox: the person inspects the
visualizations directly. The upload always creates a **pull request** on the
repository, so the files and the generated dataset card can be inspected on
Hugging Face before merging. A Hugging Face `403` response means that the token
has no write access to the repository; it is unrelated to the local inventory.

The format of the uploaded dataset (configurations, repository layout, splits)
is described in [`DATASETS.md`](DATASETS.md).

The technical inventory can also be generated from the command line. It checks
file presence and consistency; it is not a substitute for human validation:

```bash
python scripts/publish_huggingface_dataset.py \
  --config scripts/configs/default.yaml
```

For a reviewed upload, keep the token out of command history:

```bash
export HF_TOKEN=hf_...your_write_token...
python scripts/publish_huggingface_dataset.py \
  --config scripts/configs/default.yaml \
  --repo-id owner/dataset-name \
  --license cc-by-4.0 \
  --private \
  --upload
```

The script refreshes the technical inventory immediately before creating or updating the dataset
repository. Hugging Face renders the generated `README.md` as the dataset card.

## 7. Evaluate on YCB-Video (BOP19)

The scripts used for the paper evaluate the framework on the 12 YCB-Video test
sequences of the BOP19 benchmark. They are optional and not needed to annotate
your own scenes.

1. Download the BOP `ycbv` dataset (test images, models and `test_targets_bop19.json`).
2. Convert the scenes into Label Factory datasets. Each scene becomes
   `data/comparison_NN/` together with `scripts/configs/comparison_NN.yaml`:
   ```bash
   python scripts/convert_ycbv_bop19.py --bop-root /path/to/BOP/ycbv
   ```
   Use `--scenes 48 49` to convert a subset and `--overwrite` to replace existing output.
3. Run the pipeline for every `comparison_NN` configuration as described above
   (calibration and capture are not needed).
4. Evaluate ADD-S, ADD(-S), the BOP recall scores and the mask IoU:
   ```bash
   python scripts/evaluate_ycbv.py
   ```
   Results are written to `evaluation/ycbv/` (`summary.json`, `per_object.csv`,
   `per_target.csv`, tables in Markdown and LaTeX, and BOP-format result CSVs).
5. Optional: compute the official BOP recall scores (AR$_{VSD}$, AR$_{MSSD}$, AR$_{MSPD}$)
   with the `bop_toolkit`, which is cloned separately and not part of this repository:
   ```bash
   git clone https://github.com/thodan/bop_toolkit external/bop_toolkit
   python3.9 -m venv external/bop_toolkit/.venv
   external/bop_toolkit/.venv/bin/python -m pip install -e external/bop_toolkit
   scripts/run_bop19_eval.sh /path/to/BOP_root evaluation/ycbv/bop_results_labelfactory_all.csv
   ```
   The wrapper renders headlessly with EGL; set `EGL_VENDOR_FILE` if your NVIDIA
   vendor file is in another location.

The result tables of the paper are included in [`evaluation/ycbv/`](../evaluation/ycbv).

## 8. Measure the runtime

Every pipeline stage records its wall-clock time in `data/####/pipeline_timings.json`
(see `scripts/stage_timing.py`). To aggregate several scenes:

```bash
python scripts/summarize_runtime.py --datasets 0000 0001 0002
```

The summary tables, the scaling plot and a draft results paragraph are written to
`evaluation/runtime/`. Measure on clean scenes: stages that resume from earlier
results report only the time of the work that was actually repeated. The script
`scripts/cleanup_dataset_intermediates.py` removes reproducible intermediates
(including the shared SAM cache) so that a scene can be timed again from scratch.

## Troubleshooting

### CUDA out of memory

- Lower the input image resolution or SAM proposal count.
- Process a shorter frame range while validating the setup.
- Stop unrelated GPU processes.
- Keep reconstruction and SAM-6D environments separate.

SAM-6D workers use `data/####/.sam6d_gpu.lock` to prevent two UI sessions from
loading ISM/PEM networks simultaneously. A worker prints free/total CUDA memory
before initialization. If DINOv2-L still cannot load, close external CUDA
applications (including old Python, Blender, and visualization processes) and
retry; the failed initialization does not consume a frame.

### Missing `open3d`, `cv2`, or `plyfile`

The wrong interpreter is usually active. Use `label-factory` for reconstruction
and the dedicated `sam6d` environment for SAM-6D inference.

### RealSense `Couldn't resolve requests`

The RGB/depth resolution and FPS combination is unsupported. Reduce FPS or
choose a stream mode supported by your camera.

### BlenderProc `StructRNA ... Material has been removed`

Use the tested Blender 3.3.1 path and the patched template renderer. This error
usually indicates Blender/BlenderProc incompatibility or a deleted material
still referenced by BlenderProc.

### PEM `stack expects a non-empty TensorList`

No detection had enough valid metric depth points. Check ISM masks, RGB/depth
alignment, depth range, camera intrinsics, and detection threshold. Invalid
frames should be skipped instead of becoming pose estimates.

## Reproducibility checklist

- The YAML selects the intended dataset.
- RGB/depth frame IDs match.
- `camera.json` exists (written by the calibration or recording step, or supplied for an imported sequence), with correct intrinsics and millimetre depth units.
- `mvs_filtered_metric.ply` and `camera_poses_metric.json` exist.
- Every STL filename matches its intended object name.
- SAM-6D templates were generated with the correct CAD scale.
- RANSAC outputs were visually checked before annotation generation.
- Annotation masks align with the source RGB frames.
- Every command runs in its intended Conda environment.

Every script exposes additional parameters through `--help`.
