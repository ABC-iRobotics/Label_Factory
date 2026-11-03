#!/usr/bin/env python3
"""Run sparse SfM, filtering, dense MVS, and dense filtering in one command."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d
import pycolmap
import yaml
from PIL import Image

import stage_timing
from dataset_config import dataset_path
from scale_reconstruction_from_depth import scale_reconstruction


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def dataset_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    return dataset_path(config["path"], config["dataset_id"])


def load_camera(camera_path: Path, image_size: tuple[int, int]):
    camera = json.loads(camera_path.read_text())
    K = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    width, height = image_size
    stored_width = camera.get("image_width")
    stored_height = camera.get("image_height")
    if stored_width is not None and int(stored_width) != width:
        raise ValueError(
            f"camera.json width {stored_width} does not match images ({width})"
        )
    if stored_height is not None and int(stored_height) != height:
        raise ValueError(
            f"camera.json height {stored_height} does not match images ({height})"
        )
    distortion = np.asarray(
        camera.get("distortion", [0.0] * 5), dtype=np.float64
    ).reshape(-1)
    if camera.get("images_undistorted", False):
        distortion = np.zeros_like(distortion)
    if np.any(np.abs(distortion[:4]) > 1e-12):
        model = "OPENCV"
        parameters = [
            K[0, 0], K[1, 1], K[0, 2], K[1, 2],
            distortion[0], distortion[1], distortion[2], distortion[3],
        ]
    else:
        model = "PINHOLE"
        parameters = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    return camera, model, parameters


def stage(message: str):
    class Timer:
        def __enter__(self):
            print(f"\n{'=' * 72}\n{message}\n{'=' * 72}", flush=True)
            self.started = time.monotonic()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.seconds = time.monotonic() - self.started
            if exc_type is None:
                print(f"Completed in {self.seconds:.1f} seconds", flush=True)
    return Timer()


def model_stats(reconstruction) -> dict:
    return {
        "registered_images": int(reconstruction.num_reg_images()),
        "points3D": int(reconstruction.num_points3D()),
        "cameras": int(reconstruction.num_cameras()),
    }


def validate_model(path: Path) -> None:
    missing = [name for name in MODEL_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Invalid COLMAP model at {path}; missing {missing}")


def write_keyframe_patch_match_config(
    mvs_dir: Path, image_names: list[str], maximum_sources: int
) -> Path:
    """Use only existing keyframes as explicit PatchMatch dependencies."""
    images_dir = mvs_dir / "images"
    available = [name for name in image_names if (images_dir / name).is_file()]
    missing = sorted(set(image_names) - set(available))
    if missing:
        raise RuntimeError(
            "Undistortion did not create selected MVS images: "
            + ", ".join(missing[:20])
        )
    if len(available) < 2:
        raise RuntimeError("PatchMatch requires at least two undistorted keyframes")
    lines = []
    positions = {name: index for index, name in enumerate(available)}
    for index, reference in enumerate(available):
        candidates = sorted(
            (name for name in available if name != reference),
            key=lambda name: abs(positions[name] - index),
        )[: min(maximum_sources, len(available) - 1)]
        lines.extend((reference, ", ".join(candidates)))
    config_path = mvs_dir / "stereo/patch-match.cfg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines) + "\n")
    referenced = set(lines[0::2]) | {
        name.strip() for line in lines[1::2] for name in line.split(",")
        if name.strip()
    }
    unavailable = sorted(
        name for name in referenced if not (images_dir / name).is_file()
    )
    if unavailable:
        raise RuntimeError(
            "PatchMatch configuration references missing images: "
            + ", ".join(unavailable[:20])
        )
    print(
        f"PatchMatch configuration: {len(available)} keyframes, up to "
        f"{min(maximum_sources, len(available) - 1)} valid sources per view"
    )
    return config_path


def overlap_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 480.0 / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    return gray


def estimate_frame_overlap(previous_gray: np.ndarray, image: np.ndarray) -> float:
    current_gray = overlap_gray(image)
    points = cv2.goodFeaturesToTrack(previous_gray, 500, 0.01, 8)
    if points is None or len(points) < 12:
        return 0.0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None, winSize=(21, 21), maxLevel=3
    )
    valid = status.reshape(-1).astype(bool)
    if valid.sum() < 12:
        return 0.0
    transform, inliers = cv2.estimateAffinePartial2D(
        points[valid], tracked[valid], method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if transform is None or inliers is None or int(inliers.sum()) < 8:
        return 0.0
    height, width = current_gray.shape
    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    warped = cv2.transform(corners, transform).reshape(-1, 2)
    intersection, _ = cv2.intersectConvexConvex(
        warped.astype(np.float32),
        np.float32([[0, 0], [width, 0], [width, height], [0, height]]),
    )
    return float(np.clip(intersection / float(width * height), 0.0, 1.0))


def select_reconstruction_frames(
    image_paths: list[Path], target_overlap: float, output_path: Path
) -> list[Path]:
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read {image_paths[0]} during frame pairing")
    selected = [image_paths[0]]
    previous_gray = overlap_gray(first)
    observations = [{"image_name": image_paths[0].name, "overlap": None,
                     "selected": True}]
    print(f"Pairing RGB frames at {target_overlap:.0%} target overlap")
    for position, path in enumerate(image_paths[1:], 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {path} during frame pairing")
        overlap = estimate_frame_overlap(previous_gray, image)
        accepted = overlap <= target_overlap
        observations.append({"image_name": path.name, "overlap": overlap,
                             "selected": accepted})
        if accepted:
            selected.append(path)
            previous_gray = overlap_gray(image)
        print(f"Pairing [{position + 1}/{len(image_paths)}]", end="\r", flush=True)
    if len(image_paths) > 1 and selected[-1] != image_paths[-1]:
        selected.append(image_paths[-1])
        observations[-1]["selected"] = True
    output_path.write_text(json.dumps({
        "format": "label_factory_reconstruction_selection_v1",
        "target_overlap": target_overlap,
        "total_frames": len(image_paths),
        "selected_frames": len(selected),
        "frame_ids": [int(''.join(filter(str.isdigit, path.stem))) for path in selected],
        "image_names": [path.name for path in selected],
        "frames": observations,
    }, indent=2) + "\n")
    print(f"\nSelected {len(selected)}/{len(image_paths)} MVS/SAM-6D keyframes")
    return selected


def select_largest_model(models: dict):
    if not models:
        raise RuntimeError("COLMAP did not produce a sparse reconstruction")
    return max(
        models.values(),
        key=lambda rec: (rec.num_reg_images(), rec.num_points3D()),
    )


def filter_sparse(reconstruction, maximum_error: float, minimum_track: int):
    filtered = pycolmap.Reconstruction(reconstruction)
    remove = [
        point_id
        for point_id, point in filtered.points3D.items()
        if not np.isfinite(point.xyz).all()
        or float(point.error) > maximum_error
        or int(point.track.length()) < minimum_track
    ]
    for point_id in remove:
        filtered.delete_point3D(point_id)
    print(
        f"Sparse filter removed {len(remove)} points; "
        f"{filtered.num_points3D()} remain"
    )
    return filtered


def filter_dense(reconstruction, neighbors: int, std_ratio: float):
    filtered = pycolmap.Reconstruction(reconstruction)
    point_ids = list(filtered.points3D.keys())
    if not point_ids:
        raise RuntimeError("The fused MVS reconstruction contains no points")
    points = np.asarray(
        [filtered.points3D[point_id].xyz for point_id in point_ids],
        dtype=np.float64,
    )
    finite = np.isfinite(points).all(axis=1)
    finite_indices = np.flatnonzero(finite)
    keep = np.zeros(len(points), dtype=bool)
    if len(finite_indices) <= 2:
        keep[finite_indices] = True
    else:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[finite])
        effective_neighbors = min(neighbors, len(finite_indices) - 1)
        _, retained = cloud.remove_statistical_outlier(
            nb_neighbors=effective_neighbors,
            std_ratio=std_ratio,
        )
        keep[finite_indices[np.asarray(retained, dtype=np.int64)]] = True
    remove_ids = [
        point_id for index, point_id in enumerate(point_ids) if not keep[index]
    ]
    for point_id in remove_ids:
        filtered.delete_point3D(point_id)
    print(
        f"Dense filter removed {len(remove_ids)} points; "
        f"{filtered.num_points3D()} remain"
    )
    return filtered


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=project / "scripts/configs/default.yaml",
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--camera-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-features", type=int, default=16384)
    parser.add_argument("--sift-peak-threshold", type=float, default=0.005)
    parser.add_argument(
        "--target-overlap", type=float,
        default=float(os.environ.get(
            "FRAME_SELECTION_TARGET_OVERLAP",
            os.environ.get("CAPTURE_TARGET_OVERLAP", "0.70"),
        )),
        help="Target overlap used to select MVS and SAM-6D keyframes.",
    )
    parser.add_argument(
        "--matching-mode",
        choices=("sequential", "exhaustive"),
        default=os.environ.get("MATCHING_MODE", "sequential"),
        help="Image-pairing strategy used for SfM (default: sequential).",
    )
    parser.add_argument(
        "--sequential-overlap",
        type=int,
        default=int(os.environ.get("SEQUENTIAL_OVERLAP", "15")),
        help="Number of neighboring frames paired in sequential mode (default: 15).",
    )
    parser.add_argument(
        "--sequential-linear-only",
        action="store_true",
        help="Disable additional quadratic-distance neighbors.",
    )
    parser.add_argument("--sparse-max-error", type=float, default=2.0)
    parser.add_argument("--sparse-min-track", type=int, default=3)
    parser.add_argument(
        "--sfm-threads",
        type=int,
        default=int(os.environ.get("SFM_THREADS", "16")),
        help="Number of CPU threads used by incremental SfM (default: 30).",
    )
    parser.add_argument(
        "--dense-neighbors", type=int,
        default=int(os.environ.get("DENSE_NEIGHBORS", "10")),
    )
    parser.add_argument("--dense-std-ratio", type=float, default=2.0)
    parser.add_argument(
        "--mvs-gpu-indices",
        default=os.environ.get("MVS_GPU_INDICES", "0"),
        help=(
            "Comma-separated PatchMatch GPU worker assignment. Repeating an "
            "index runs multiple workers on that GPU (default: 0,0,0,0)."
        ),
    )
    parser.add_argument(
        "--mvs-threads",
        type=int,
        default=int(os.environ.get("MVS_THREADS", "16")),
        help="Worker threads used by PatchMatch and stereo fusion (default: 30).",
    )
    parser.add_argument(
        "--mvs-cache-size-gb",
        type=float,
        default=float(os.environ.get("MVS_CACHE_SIZE_GB", "16")),
        help="Host-memory cache for PatchMatch and stereo fusion (default: 64 GiB).",
    )
    parser.add_argument(
        "--mvs-source-images",
        type=int,
        default=int(os.environ.get("MVS_SOURCE_IMAGES", "15")),
        help="Maximum automatically selected PatchMatch source views (default: 15).",
    )
    parser.add_argument(
        "--mvs-window-radius", type=int,
        default=int(os.environ.get("MVS_WINDOW_RADIUS", "4")),
    )
    parser.add_argument(
        "--mvs-window-step", type=int,
        default=int(os.environ.get("MVS_WINDOW_STEP", "1")),
    )
    parser.add_argument(
        "--mvs-num-samples", type=int,
        default=int(os.environ.get("MVS_NUM_SAMPLES", "10")),
    )
    parser.add_argument(
        "--mvs-num-iterations", type=int,
        default=int(os.environ.get("MVS_NUM_ITERATIONS", "3")),
    )
    parser.add_argument(
        "--skip-metric-scaling",
        action="store_true",
        help="Do not scale the final MVS cloud from data/####/depth_maps.",
    )
    parser.add_argument("--scale-maximum-points", type=int, default=1_000_000)
    parser.add_argument("--scale-samples-per-frame", type=int, default=20_000)
    parser.add_argument(
        "--metric-scale-workers", type=int,
        default=int(os.environ.get("METRIC_SCALE_WORKERS", "4")),
        help="Parallel workers for depth-based metric scaling (default: 4).",
    )
    parser.add_argument("--max-image-size", type=int, default=-1)
    parser.add_argument(
        "--no-geometric-consistency",
        action="store_true",
        help=(
            "Run only the photometric PatchMatch sweep. This is faster, but "
            "usually produces a less reliable dense cloud."
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sfm_threads < 1:
        raise ValueError("--sfm-threads must be at least 1")
    if not 0.0 <= args.target_overlap <= 1.0:
        raise ValueError("--target-overlap must be between 0 and 1")
    if args.sequential_overlap < 1:
        raise ValueError("--sequential-overlap must be at least 1")
    if args.mvs_threads < 1:
        raise ValueError("--mvs-threads must be at least 1")
    if args.metric_scale_workers < 1:
        raise ValueError("--metric-scale-workers must be at least 1")
    if args.mvs_cache_size_gb <= 0:
        raise ValueError("--mvs-cache-size-gb must be greater than 0")
    if args.mvs_source_images < 1:
        raise ValueError("--mvs-source-images must be at least 1")
    if args.mvs_window_radius < 1:
        raise ValueError("--mvs-window-radius must be at least 1")
    if args.mvs_window_step < 1:
        raise ValueError("--mvs-window-step must be at least 1")
    if args.mvs_num_samples < 1:
        raise ValueError("--mvs-num-samples must be at least 1")
    if args.mvs_num_iterations < 1:
        raise ValueError("--mvs-num-iterations must be at least 1")
    gpu_indices = [item.strip() for item in args.mvs_gpu_indices.split(",")]
    if not gpu_indices or any(not item.isdigit() for item in gpu_indices):
        raise ValueError(
            "--mvs-gpu-indices must contain comma-separated non-negative "
            "GPU indices, for example 0 or 0,0"
        )
    args.mvs_gpu_indices = ",".join(gpu_indices)
    config_path = args.config.expanduser().resolve()
    dataset = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir else dataset_from_config(config_path)
    )
    images_dir = dataset / "images"
    camera_path = (
        args.camera_json.expanduser().resolve()
        if args.camera_json else dataset / "camera.json"
    )
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else dataset / "reconstruction"
    )
    all_image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(all_image_paths) < 2:
        raise RuntimeError(f"At least two RGB images are required in {images_dir}")
    selection_path = dataset / "reconstruction_frame_selection.json"
    selection_started = time.monotonic()
    mvs_image_paths = select_reconstruction_frames(
        all_image_paths, args.target_overlap, selection_path
    )
    stage_timing.record(
        dataset, "frame_selection_reconstruction", time.monotonic() - selection_started,
        input_frames=len(all_image_paths), selected_frames=len(mvs_image_paths),
    )
    image_paths = all_image_paths
    if len(image_paths) < 2 or len(mvs_image_paths) < 2:
        raise RuntimeError(f"At least two RGB images are required in {images_dir}")
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    with Image.open(image_paths[0]) as image:
        image_size = image.size
    for path in image_paths[1:]:
        with Image.open(path) as image:
            if image.size != image_size:
                raise ValueError(f"Mixed image dimensions: {path} is {image.size}")
    camera, camera_model, camera_parameters = load_camera(
        camera_path, image_size
    )

    # COLMAP PatchMatch Stereo is CUDA-only. Device=cpu is valid for SIFT but
    # cannot provide a CPU fallback for stage 5. Check before replacing any
    # existing reconstruction or spending time on SfM.
    if not bool(getattr(pycolmap, "has_cuda", False)):
        raise RuntimeError(
            "Dense PatchMatch MVS requires a CUDA-enabled PyCOLMAP/COLMAP "
            "build, but the selected Python reports pycolmap.has_cuda=False. "
            "Installing a CUDA-capable PyTorch does not change PyCOLMAP. "
            "Build COLMAP and its Python bindings with CUDA enabled, then "
            "select that environment as Reconstruction Python. Existing "
            "reconstruction outputs were not modified."
        )

    if output.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Reconstruction output already exists: {output}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)
    sfm_dir = output / "sfm"
    sfm_filtered_dir = output / "sfm_filtered"
    mvs_dir = output / "mvs"
    candidates_dir = output / ".sfm_candidates"
    fusion_dir = output / ".mvs_fusion"
    database_path = output / "database.db"
    # pycolmap.Reconstruction.write() requires its destination to exist.
    # Create every final/staging model directory before the corresponding
    # stage writes COLMAP binary files into it.
    for model_dir in (
        sfm_dir,
        sfm_filtered_dir,
        mvs_dir,
        candidates_dir,
        fusion_dir,
    ):
        model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset}")
    print(
        f"SfM images: {len(image_paths)}; MVS keyframes: {len(mvs_image_paths)} "
        f"at {image_size[0]}x{image_size[1]}"
    )
    print(f"Camera: {camera_model} {camera_parameters}")
    print(f"Output: {output}")

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    reader.camera_params = ",".join(f"{value:.17g}" for value in camera_parameters)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = args.max_features
    extraction.sift.estimate_affine_shape = True
    extraction.sift.domain_size_pooling = True
    extraction.sift.peak_threshold = args.sift_peak_threshold
    device = {
        "auto": pycolmap.Device.auto,
        "cpu": pycolmap.Device.cpu,
        "cuda": pycolmap.Device.cuda,
    }[args.device]

    timings = {}
    with stage("1/7 Extract SIFT features") as timer:
        pycolmap.extract_features(
            database_path,
            images_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader,
            extraction_options=extraction,
            device=device,
        )
    timings["feature_extraction_s"] = timer.seconds

    matching = pycolmap.FeatureMatchingOptions()
    matching.guided_matching = True
    matching.max_num_matches = 65536
    matching.num_threads = args.sfm_threads
    matching_label = (
        "Sequential feature matching"
        if args.matching_mode == "sequential"
        else "Exhaustive feature matching"
    )
    with stage(f"2/7 {matching_label}") as timer:
        if args.matching_mode == "sequential":
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = args.sequential_overlap
            pairing.quadratic_overlap = not args.sequential_linear_only
            pairing.loop_detection = False
            print(
                f"Sequential pairing: overlap={pairing.overlap}; "
                f"quadratic_overlap={pairing.quadratic_overlap}; "
                "loop_detection=False",
                flush=True,
            )
            pycolmap.match_sequential(
                database_path,
                matching_options=matching,
                pairing_options=pairing,
                device=device,
            )
        else:
            pycolmap.match_exhaustive(
                database_path, matching_options=matching, device=device
            )
    timings["feature_matching_s"] = timer.seconds

    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.ba_refine_focal_length = False
    mapping.ba_refine_principal_point = False
    mapping.ba_refine_extra_params = False
    mapping.num_threads = args.sfm_threads
    # The published pycolmap-cuda12 wheel accelerates CUDA features such as
    # SIFT and PatchMatch, but its Ceres backend is not built with CUDA/cuDSS.
    # Requesting GPU BA only emits warnings and falls back to these same CPU
    # solvers, so select the real backend explicitly.
    mapping.ba_use_gpu = False
    mapping.triangulation.ignore_two_view_tracks = False
    mapping.triangulation.min_angle = 1.0
    with stage("3/7 Sparse SfM reconstruction") as timer:
        print(
            f"Incremental SfM: {args.sfm_threads} CPU threads; "
            "Ceres bundle adjustment on CPU",
            flush=True,
        )
        models = pycolmap.incremental_mapping(
            database_path, images_dir, candidates_dir, options=mapping
        )
        sparse = select_largest_model(models)
        registered_names = {
            sparse.images[image_id].name for image_id in sparse.reg_image_ids()
        }
        missing_camera_poses = sorted(
            path.name for path in image_paths if path.name not in registered_names
        )
        if missing_camera_poses:
            preview = ", ".join(missing_camera_poses[:20])
            suffix = " ..." if len(missing_camera_poses) > 20 else ""
            raise RuntimeError(
                f"COLMAP registered {len(registered_names)}/{len(image_paths)} "
                "RGB frames. Every training frame requires a camera pose. "
                f"Unregistered frames: {preview}{suffix}"
            )
        sparse.write(sfm_dir)
        sparse.export_PLY(output / "sfm.ply")
        validate_model(sfm_dir)
        shutil.rmtree(candidates_dir, ignore_errors=True)
        print(sparse.summary())
    timings["sparse_sfm_s"] = timer.seconds

    with stage("4/7 Filter sparse SfM") as timer:
        sparse_filtered = filter_sparse(
            sparse, args.sparse_max_error, args.sparse_min_track
        )
        sparse_filtered.write(sfm_filtered_dir)
        sparse_filtered.export_PLY(output / "sfm_filtered.ply")
        validate_model(sfm_filtered_dir)
        print(sparse_filtered.summary())
    timings["sparse_filter_s"] = timer.seconds

    with stage("5/7 Dense PatchMatch MVS and fusion") as timer:
        pycolmap.undistort_images(
            mvs_dir,
            sfm_filtered_dir,
            images_dir,
            image_names=[path.name for path in mvs_image_paths],
            output_type="COLMAP",
            num_patch_match_src_images=args.mvs_source_images,
        )
        write_keyframe_patch_match_config(
            mvs_dir,
            [path.name for path in mvs_image_paths],
            args.mvs_source_images,
        )
        patch_match = pycolmap.PatchMatchOptions()
        patch_match.geom_consistency = not args.no_geometric_consistency
        patch_match.filter = True
        patch_match.max_image_size = args.max_image_size
        patch_match.gpu_index = args.mvs_gpu_indices
        patch_match.num_threads = args.mvs_threads
        patch_match.cache_size = args.mvs_cache_size_gb
        patch_match.window_radius = args.mvs_window_radius
        patch_match.window_step = args.mvs_window_step
        patch_match.num_samples = args.mvs_num_samples
        patch_match.num_iterations = args.mvs_num_iterations
        print(
            f"PatchMatch GPU workers: {args.mvs_gpu_indices}; "
            f"threads: {args.mvs_threads}; cache: "
            f"{args.mvs_cache_size_gb:g} GiB; source images: "
            f"{args.mvs_source_images}; radius/step: "
            f"{args.mvs_window_radius}/{args.mvs_window_step}; "
            f"samples/iterations: "
            f"{args.mvs_num_samples}/{args.mvs_num_iterations}"
            "\nPatchMatch mode: "
            + (
                "photometric sweep only"
                if args.no_geometric_consistency
                else "photometric + geometric-consistency sweeps"
            )
        )
        pycolmap.patch_match_stereo(mvs_dir, options=patch_match)
        fusion = pycolmap.StereoFusionOptions()
        fusion.max_image_size = args.max_image_size
        fusion.num_threads = args.mvs_threads
        fusion.cache_size = args.mvs_cache_size_gb
        dense = pycolmap.stereo_fusion(
            output_path=fusion_dir,
            workspace_path=mvs_dir,
            input_type=(
                "photometric" if args.no_geometric_consistency else "geometric"
            ),
            options=fusion,
            output_type="bin",
        )
        # MVS enriches the filtered sparse reconstruction with a dense point
        # cloud. Keep camera poses authoritative in sfm_filtered; the fused
        # pycolmap model is only a transient container used to export the PLY.
        dense.export_PLY(output / "mvs.ply")
        shutil.rmtree(fusion_dir, ignore_errors=True)
        print(dense.summary())
    timings["dense_mvs_s"] = timer.seconds

    with stage("6/7 Filter fused MVS cloud") as timer:
        dense_filtered = filter_dense(
            dense, args.dense_neighbors, args.dense_std_ratio
        )
        dense_filtered.export_PLY(output / "mvs_filtered.ply")
        print(dense_filtered.summary())
    timings["dense_filter_s"] = timer.seconds

    metric_report = None
    metric_cloud_path = output / "mvs_filtered_metric.ply"
    metric_alignment_dir = output / "metric_alignment"
    if args.skip_metric_scaling:
        print("\n7/7 Metric depth scaling skipped by request")
    else:
        depth_dir = dataset / "depth_maps"
        if not depth_dir.is_dir():
            raise FileNotFoundError(
                f"Metric depth folder was not found: {depth_dir}. "
                "Use --skip-metric-scaling only when metric output is not required."
            )
        with stage("7/7 Scale filtered MVS cloud from RGB-D depth") as timer:
            metric_report = scale_reconstruction(
                output / "mvs_filtered.ply",
                sfm_filtered_dir,
                depth_dir,
                camera_path,
                metric_cloud_path,
                metric_alignment_dir,
                maximum_points=args.scale_maximum_points,
                maximum_samples_per_frame=args.scale_samples_per_frame,
                workers=args.metric_scale_workers,
            )
            print(
                "Metric scale: "
                f"{metric_report['scale_metres_per_reconstruction_unit']:.12g} "
                "m/reconstruction-unit"
            )
            print(
                "Median depth residual: "
                f"{metric_report['median_absolute_depth_error_m'] * 1000:.3f} mm"
            )
        timings["metric_scaling_s"] = timer.seconds

    stage_timing.record(
        dataset, "reconstruction", sum(timings.values()),
        registered_frames=len(image_paths), mvs_frames=len(mvs_image_paths),
        parts={key: round(value, 3) for key, value in timings.items()},
    )
    summary = {
        "format": "rgbd_colmap_reconstruction_v2",
        "input_rgb_frames": len(all_image_paths),
        "registered_camera_pose_frames": len(image_paths),
        "selected_mvs_frames": len(mvs_image_paths),
        "reconstruction_frame_selection": (
            str(selection_path) if selection_path.is_file() else None
        ),
        "dataset_dir": str(dataset),
        "images_dir": str(images_dir),
        "camera_json": str(camera_path),
        "camera_model": camera_model,
        "camera_parameters": camera_parameters,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "number_of_images": len(image_paths),
        "stages": {
            "sfm": model_stats(sparse),
            "sfm_filtered": model_stats(sparse_filtered),
            "mvs_dense_cloud": {
                "points3D": int(dense.num_points3D()),
                "source_camera_model": "sfm_filtered",
            },
            "mvs_filtered_dense_cloud": {
                "points3D": int(dense_filtered.num_points3D()),
                "source_camera_model": "sfm_filtered",
            },
        },
        "timings": timings,
        "paths": {
            "sfm": str(sfm_dir),
            "sfm_filtered": str(sfm_filtered_dir),
            "mvs_workspace": str(mvs_dir),
            "sfm_ply": str(output / "sfm.ply"),
            "sfm_filtered_ply": str(output / "sfm_filtered.ply"),
            "mvs_ply": str(output / "mvs.ply"),
            "mvs_filtered_ply": str(output / "mvs_filtered.ply"),
        },
        "camera_metadata": camera,
        "metric_scaling": metric_report,
    }
    if metric_report is not None:
        summary["paths"]["mvs_filtered_metric_ply"] = str(metric_cloud_path)
        summary["paths"]["metric_alignment"] = str(metric_alignment_dir)
    summary_path = output / "reconstruction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nComplete reconstruction saved to {output}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
