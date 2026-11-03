# Usage guide

This guide describes the complete Label Factory workflow from the command line.
The same stages are available in the Streamlit user interface. Installation is
described in the [README](../README.md#installation); for the Docker-specific
details see [`DOCKER.md`](DOCKER.md).

**Contents**

- [Configuration](#configuration)
- [Guided user interface](#guided-user-interface)
- [Expected dataset layout](#expected-dataset-layout)
- [1. Camera calibration](#1-camera-calibration)
- [2. Record an aligned RealSense RGB-D sequence](#2-record-an-aligned-realsense-rgb-d-sequence)
- [3. Reconstruct and scale the scene](#3-reconstruct-and-scale-the-scene)
- [4. Known-object poses with SAM-6D](#4-known-object-poses-with-sam-6d)
- [5. Create the posed scene and frame annotations](#5-create-the-posed-scene-and-frame-annotations)
- [6. Review and publish to Hugging Face](#6-review-and-publish-to-hugging-face)
- [7. Evaluate on YCB-Video (BOP19)](#7-evaluate-on-ycb-video-bop19)
- [8. Measure the runtime](#8-measure-the-runtime)
- [Important limitations](#important-limitations)
- [Troubleshooting](#troubleshooting)
- [Reproducibility checklist](#reproducibility-checklist)

## Configuration

Edit `scripts/configs/default.yaml`:

```yaml
path: .
dataset_id: 0
fps: 5
n_points: 20000
camera:
  camera_id: 0
  fx: 627.28
  fy: 627.23
  cx: 629.34
  cy: 397.82
```

`dataset_id: 0` resolves to `data/0000`; `dataset_id: 3` resolves to
`data/0003`.

The camera calibration is part of the framework: calibrate once per camera and resolution (step 1), and the recording step copies the saved calibration into the dataset as `data/####/camera.json`. You do not need to write this file yourself, except when you import a sequence recorded elsewhere. Its intrinsic matrix,
image dimensions, and depth unit are used by reconstruction and projection.

## Guided user interface

> A step-by-step walkthrough with screenshots of every tab is in
> [`UI_GUIDE.md`](UI_GUIDE.md).

The Streamlit UI guides the user through the same command-line stages while
preserving the exact command used for every operation. Install its small UI
dependency set in the reconstruction environment:

```bash
conda activate label-factory
python -m pip install -r ui/requirements.txt
```

Start the application from the repository root:

```bash
streamlit run ui/app.py
```

For an NVIDIA GPU and RealSense-enabled container deployment, see
[`DOCKER.md`](DOCKER.md). The Docker setup keeps datasets, camera calibrations,
SAM-6D checkpoints, and Hugging Face credentials outside the image.

Then open `http://localhost:8501`. The sidebar selects the project YAML and the
Python interpreters used for reconstruction and SAM-6D. The stage cards show:

- `✅` when the expected outputs exist;
- `▶️` when prerequisites are ready;
- `⛔` when required inputs are missing.

The UI includes setup validation, camera calibration, RealSense recording,
reconstruction, SAM-6D, RANSAC, scene generation, log streaming, command
previews, annotation video, and reviewed Hugging Face publication. Calibration, capture, and Open3D
visualizers currently open native desktop windows; browser-native camera and 3D
views are planned for the Docker deployment.

## Expected dataset layout

```text
data/0000/
├── camera.json
├── images/
│   ├── 000000.png
│   └── ...
├── depth_maps/
│   ├── depth_000000.png
│   └── ...
└── objects/
    ├── stl/
    │   ├── Pliers.stl
    │   └── Screwdriver.stl
    └── textured/                 optional
        ├── Pliers.obj
        ├── Pliers.mtl
        └── Pliers.png
```

RGB and depth files are paired by numeric frame ID. Depth PNGs must be aligned
to RGB and contain metric depth in millimetres (`uint16`). STL files are only
required for the known-object SAM-6D workflow.

## 1. Camera calibration

Capture ChArUco views and estimate RGB intrinsics:

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

## 2. Record an aligned RealSense RGB-D sequence

```bash
conda activate label-factory
python scripts/record_realsense_rgbd.py \
  --config scripts/configs/default.yaml \
  --width 1280 --height 800 \
  --depth-width 1280 --depth-height 720 \
  --fps 30
```

The requested stream combination must be supported by the connected camera. If
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
inventory, and a guarded publication workflow. A dataset cannot be uploaded
until:

1. The technical inventory confirms that required files are present.
2. A person uses the provided reconstruction, RANSAC, RGB/depth/mask, and video
   visualizations to review the result.
3. A valid `owner/dataset-name` repository ID and write token are provided.

For organization repositories, the token owner must have direct write access
to the dataset repository. If the organization only permits contributions via
review, select **Pull request** as the upload method. A Hugging Face `403`
response means repository authorization failed; it is unrelated to the local
dataset inventory.

The fixed acknowledgement is always visible in the left sidebar. It cannot be
edited through the UI and is never copied into an uploaded dataset or dataset
card. Human review is intentionally not represented by checkboxes; the person
performs the review directly using the visualizations.

The curated upload includes RGB/depth frames, calibration, the metric cloud and
camera poses, STL files, per-frame and final world object poses, annotations,
raw segmentations, instance-mask images, annotation-overlay images, RANSAC
visualizations, the overlay video, and a generated dataset card. It excludes acknowledgements, the local technical
inventory, COLMAP databases, PatchMatch
caches, rendered templates, and inference intermediates.

Each configured dataset is uploaded below its zero-padded ID. This prevents a
later capture from replacing files belonging to an earlier capture:

```text
owner/repository
├── 0001/
│   ├── images/
│   ├── depth_maps/
│   ├── reconstruction/
│   ├── sam6d_scene/
│   ├── sam6d_sequences/
│   └── data/
│       ├── depth_estimation/*.parquet
│       ├── instance_segmentation/*.parquet
│       └── object_pose_estimation/
│           ├── train-*.parquet
│           ├── validation-*.parquet
│           ├── test-*.parquet
│           └── models.parquet
├── 0002/
│   └── ...
└── README.md
```

The training tasks and normalized pose-model table are Hugging Face dataset configurations, rather than Git
branches. This keeps shared files and task tables on the same reviewed revision,
works with direct and pull-request uploads, and is directly selectable by
training code:

```python
from datasets import load_dataset

depth = load_dataset("owner/repository", "depth_estimation")
instances = load_dataset("owner/repository", "instance_segmentation")
poses = load_dataset("owner/repository", "object_pose_estimation")
models = load_dataset("owner/repository", "object_pose_models")

train = poses["train"]
validation = poses["validation"]
test = poses["test"]
```

The Parquet rows embed the required PNG bytes using Hugging Face `Image`
features, so RGB, depth, and masks render in the Hub dataset viewer. They also contain frame IDs,
camera intrinsics, units, masks, boxes, instance identities, camera transforms,
and object transforms as appropriate for each task.

Pose rows contain `object_model_keys`. The `object_pose_models` configuration
stores each referenced STL once per numbered dataset, together with its file
name, SHA-256 digest, byte size, model scale and binary payload. This avoids
duplicating multi-megabyte meshes in every frame row while keeping them directly
available to training code.

All numbered captures are loaded automatically. To train with several selected
captures, filter the `dataset_id` column:

```python
from datasets import load_dataset

poses = load_dataset("owner/repository", "object_pose_estimation")
selected_ids = set(["0000", "0001", "0004"])
for split_name in poses:
    poses[split_name] = poses[split_name].filter(
        lambda row: row["dataset_id"] in selected_ids
    )
```

The generated Hugging Face card also contains complete examples for decoding
RGB, metric depth and instance-mask PNG payloads, reading boxes and annotation
JSON, and reshaping camera/object transformations into 4×4 matrices.

Training and validation ratios and the deterministic split seed are configured
in the UI; the remaining fraction becomes the test split. Assignment hashes the
dataset ID, frame ID, and seed, so uploading a new numbered dataset appends new
samples without moving existing samples between splits. The generated dataset
card declares all three configurations and all three splits using wildcard
paths across the numbered dataset directories.

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

## Important limitations

- Every object must have a known STL model and rendered SAM-6D templates.
- RGB/depth misalignment, incorrect intrinsics, missing viewpoints, and poor
  reconstruction poses propagate into the estimated world-coordinate poses.
- SAM-6D may return several candidates or only a partial mask. Multi-frame
  RANSAC rejects inconsistent estimates but cannot recover an object that is
  never detected reliably.
- Multiple identical objects require distinct position clusters. Their CAD
  symmetries are handled during orientation consensus when symmetry detection
  is enabled.

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
choose a stream mode supported by the D455.

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
- `camera.json` contains correct intrinsics and millimetre depth units.
- `mvs_filtered_metric.ply` and `camera_poses_metric.json` exist.
- Every STL filename matches its intended object name.
- SAM-6D templates were generated with the correct CAD scale.
- RANSAC outputs were visually checked before annotation generation.
- Annotation masks align with the source RGB frames.
- Every command runs in its intended Conda environment.

Every script exposes additional parameters through `--help`.
