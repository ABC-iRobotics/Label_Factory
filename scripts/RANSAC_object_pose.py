#!/usr/bin/env python3
"""Estimate robust, possibly multi-instance, SAM-6D poses in world coordinates."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import re
import struct
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dataset_config import dataset_path
from silhouette_verification import sample_model_points, verify_instance


def numeric_frame_id(value: str | Path) -> int:
    numbers = re.findall(r"\d+", Path(value).stem)
    if not numbers:
        raise ValueError(f"No numeric frame ID in {value!s}")
    return int(numbers[-1])


def dataset_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    return dataset_path(config["path"], config["dataset_id"])


def safe_object_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    name = name.strip("._-")
    if not name:
        raise ValueError("Invalid empty object name")
    return name


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def read_colmap_images_binary(
    path: Path,
) -> list[tuple[str, np.ndarray]]:
    """Read image names and world-to-camera poses without pycolmap."""

    records: list[tuple[str, np.ndarray]] = []
    with path.open("rb") as handle:
        raw_count = handle.read(8)
        if len(raw_count) != 8:
            raise ValueError(f"Invalid COLMAP images file: {path}")
        number_of_images = struct.unpack("<Q", raw_count)[0]
        for _ in range(number_of_images):
            # image_id, quaternion (qw, qx, qy, qz), translation, camera_id
            fixed = handle.read(64)
            if len(fixed) != 64:
                raise ValueError(f"Truncated COLMAP images file: {path}")
            unpacked = struct.unpack("<idddddddi", fixed)
            qw, qx, qy, qz = unpacked[1:5]
            translation = np.asarray(unpacked[5:8], dtype=np.float64)

            name_bytes = bytearray()
            while True:
                character = handle.read(1)
                if not character:
                    raise ValueError(
                        f"Truncated image name in COLMAP file: {path}"
                    )
                if character == b"\0":
                    break
                name_bytes.extend(character)
            name = name_bytes.decode("utf-8")

            raw_points = handle.read(8)
            if len(raw_points) != 8:
                raise ValueError(f"Truncated COLMAP images file: {path}")
            number_of_points = struct.unpack("<Q", raw_points)[0]
            handle.seek(number_of_points * 24, 1)

            rotation_cw = Rotation.from_quat(
                [qx, qy, qz, qw]
            ).as_matrix()
            camera_from_world = transform(rotation_cw, translation)
            records.append((name, camera_from_world))
    return records


def read_colmap_camera_from_world(
    reconstruction_path: Path,
) -> list[tuple[str, np.ndarray]]:
    """Use pycolmap when available, otherwise read images.bin directly."""

    try:
        import pycolmap
    except ImportError:
        images_path = reconstruction_path / "images.bin"
        if not images_path.is_file():
            raise FileNotFoundError(
                f"COLMAP images.bin was not found: {images_path}"
            )
        print(
            "pycolmap is unavailable; reading COLMAP images.bin directly."
        )
        return read_colmap_images_binary(images_path)

    reconstruction = pycolmap.Reconstruction(str(reconstruction_path))
    records = []
    for image in reconstruction.images.values():
        cam_from_world = image.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        raw = np.asarray(cam_from_world.matrix(), dtype=np.float64)
        camera_from_world = np.eye(4, dtype=np.float64)
        camera_from_world[:3, :] = raw[:3, :]
        records.append((image.name, camera_from_world))
    return records


def load_world_from_camera(
    reconstruction_path: Path,
    alignment_dir: Path,
) -> dict[int, np.ndarray]:
    scale = float(np.asarray(np.load(alignment_dir / "scale.npy")).squeeze())
    align_rotation = np.asarray(
        np.load(alignment_dir / "rotation.npy"), dtype=np.float64
    )
    align_translation = np.asarray(
        np.load(alignment_dir / "translation.npy"), dtype=np.float64
    ).reshape(3)

    images = sorted(
        read_colmap_camera_from_world(reconstruction_path),
        key=lambda item: item[0],
    )
    poses: dict[int, np.ndarray] = {}
    for image_name, camera_from_world in images:
        # This is the same Sim(3) pose transformation used by
        # Depth_maps.transform_camera_poses_sim3.
        rotation_cw = camera_from_world[:3, :3]
        translation_cw = camera_from_world[:3, 3]
        rotation_wc = rotation_cw.T
        camera_center = -rotation_wc @ translation_cw
        metric_center = (
            scale * (align_rotation @ camera_center) + align_translation
        )
        metric_rotation_wc = align_rotation @ rotation_wc

        poses[numeric_frame_id(image_name)] = transform(
            metric_rotation_wc,
            metric_center,
        )
    if not poses:
        raise ValueError(f"No camera poses found in {reconstruction_path}")
    return poses


def raw_rotation_errors_deg(
    reference: np.ndarray, rotations: np.ndarray
) -> np.ndarray:
    relative = np.einsum("ij,njk->nik", reference.T, rotations)
    cosines = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0
    )
    return np.degrees(np.arccos(cosines))


def rotation_errors_deg(
    reference: np.ndarray,
    rotations: np.ndarray,
    symmetries: np.ndarray,
) -> np.ndarray:
    errors = np.full(len(rotations), np.inf)
    for symmetry in symmetries:
        equivalent = np.einsum("nij,jk->nik", rotations, symmetry)
        errors = np.minimum(
            errors, raw_rotation_errors_deg(reference, equivalent)
        )
    return errors


def candidate_axis_symmetries() -> np.ndarray:
    """Return the 24 proper signed-permutation rotations."""
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    return np.stack(rotations)


def detect_model_symmetries(
    object_path: Path,
    tolerance_fraction: float,
    maximum_points: int = 20_000,
) -> np.ndarray:
    """Detect finite axis-aligned rotations that preserve the CAD geometry."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError(
            "Automatic symmetry detection requires scipy."
        ) from error

    try:
        import trimesh

        geometry = trimesh.load(object_path, force="mesh", process=False)
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
    except ImportError:
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError(
                "Automatic symmetry detection requires either trimesh "
                "or open3d to load the CAD model."
            ) from error
        geometry = o3d.io.read_triangle_mesh(str(object_path))
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
    vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if len(vertices) < 4:
        raise ValueError(f"Object model has too few vertices: {object_path}")
    if len(vertices) > maximum_points:
        indices = np.linspace(0, len(vertices) - 1, maximum_points, dtype=int)
        vertices = vertices[indices]

    diameter = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if diameter <= 0:
        raise ValueError(f"Object model has zero diameter: {object_path}")
    tolerance = tolerance_fraction * diameter
    tree = cKDTree(vertices)
    accepted = []
    for symmetry in candidate_axis_symmetries():
        transformed = vertices @ symmetry.T
        distances, _ = tree.query(transformed, k=1)
        # The high quantile tolerates uneven mesh tessellation without allowing
        # a small symmetric part to dominate an asymmetric complete object.
        if float(np.quantile(distances, 0.95)) <= tolerance:
            accepted.append(symmetry)

    identity = np.eye(3)
    if not any(np.allclose(item, identity) for item in accepted):
        accepted.insert(0, identity)
    accepted.sort(key=lambda item: np.linalg.norm(item - identity))
    return np.stack(accepted)


def ransac_pose(
    matrices: np.ndarray,
    scores: np.ndarray,
    frame_ids: np.ndarray,
    translation_threshold: float,
    rotation_threshold_deg: float,
    minimum_inliers: int,
    symmetries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotations = matrices[:, :3, :3]
    translations = matrices[:, :3, 3]
    best_inliers: np.ndarray | None = None
    best_key: tuple[int, float, float] | None = None

    # One pose defines an SE(3) hypothesis, so testing every observation is a
    # deterministic form of one-sample RANSAC and avoids random-run variance.
    for candidate_rotation, candidate_translation in zip(rotations, translations):
        translation_errors = np.linalg.norm(
            translations - candidate_translation[None, :], axis=1
        )
        rotation_errors = rotation_errors_deg(
            candidate_rotation, rotations, symmetries
        )
        compatible = (
            (translation_errors <= translation_threshold)
            & (rotation_errors <= rotation_threshold_deg)
        )
        # Multiple SAM masks may describe the same physical object in one
        # image. Count at most one proposal per frame so duplicate masks cannot
        # manufacture RANSAC support for an instance.
        inliers = np.zeros(len(matrices), dtype=bool)
        for frame_id in np.unique(frame_ids[compatible]):
            candidates = np.flatnonzero(compatible & (frame_ids == frame_id))
            combined_error = (
                translation_errors[candidates]
                + np.deg2rad(rotation_errors[candidates])
            )
            selected = max(
                range(len(candidates)),
                key=lambda index: (
                    float(scores[candidates[index]]),
                    -float(combined_error[index]),
                ),
            )
            inliers[candidates[selected]] = True
        key = (
            int(inliers.sum()),
            float(scores[inliers].sum()),
            -float(
                np.median(
                    translation_errors[inliers]
                    + np.deg2rad(rotation_errors[inliers])
                )
            ),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_inliers = inliers

    assert best_inliers is not None
    if int(best_inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            f"RANSAC found only {int(best_inliers.sum())} inliers; "
            f"minimum requested is {minimum_inliers}"
        )

    weights = np.maximum(scores[best_inliers], 1e-8)
    weights /= weights.sum()
    refined_translation = np.average(
        translations[best_inliers], axis=0, weights=weights
    )
    aligned_rotations = rotations[best_inliers].copy()
    reference = rotations[np.flatnonzero(best_inliers)[0]]
    for index, observed in enumerate(aligned_rotations):
        equivalents = np.einsum("ij,kjl->kil", observed, symmetries)
        errors = raw_rotation_errors_deg(reference, equivalents)
        aligned_rotations[index] = equivalents[int(np.argmin(errors))]
    refined_rotation = Rotation.from_matrix(aligned_rotations).mean(
        weights=weights
    ).as_matrix()
    refined = transform(refined_rotation, refined_translation)

    translation_errors = np.linalg.norm(
        translations - refined_translation[None, :], axis=1
    )
    rotation_errors = rotation_errors_deg(
        refined_rotation, rotations, symmetries
    )
    compatible = (
        (translation_errors <= translation_threshold)
        & (rotation_errors <= rotation_threshold_deg)
    )
    final_inliers = np.zeros(len(matrices), dtype=bool)
    for frame_id in np.unique(frame_ids[compatible]):
        candidates = np.flatnonzero(compatible & (frame_ids == frame_id))
        combined_error = (
            translation_errors[candidates]
            + np.deg2rad(rotation_errors[candidates])
        )
        selected = max(
            range(len(candidates)),
            key=lambda index: (
                float(scores[candidates[index]]),
                -float(combined_error[index]),
            ),
        )
        final_inliers[candidates[selected]] = True
    if int(final_inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            f"Refined RANSAC pose has only {int(final_inliers.sum())} "
            f"unique-frame inliers; minimum requested is {minimum_inliers}"
        )
    return (
        refined,
        final_inliers,
        compatible,
        translation_errors,
        rotation_errors,
    )


def ransac_position(
    matrices: np.ndarray,
    scores: np.ndarray,
    frame_ids: np.ndarray,
    translation_threshold: float,
    minimum_inliers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find one translation cluster, counting at most one vote per frame."""
    translations = matrices[:, :3, 3]
    best_inliers = None
    best_key = None
    for candidate in translations:
        errors = np.linalg.norm(translations - candidate[None, :], axis=1)
        compatible = errors <= translation_threshold
        selected = np.zeros(len(matrices), dtype=bool)
        for frame_id in np.unique(frame_ids[compatible]):
            candidates = np.flatnonzero(compatible & (frame_ids == frame_id))
            winner = max(
                candidates,
                key=lambda index: (float(scores[index]), -float(errors[index])),
            )
            selected[winner] = True
        key = (
            int(selected.sum()),
            float(scores[selected].sum()),
            -float(np.median(errors[selected])),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_inliers = selected
    assert best_inliers is not None
    if int(best_inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            f"Position RANSAC found only {int(best_inliers.sum())} inliers; "
            f"minimum requested is {minimum_inliers}"
        )

    weights = np.maximum(scores[best_inliers], 1e-8)
    refined = np.average(
        translations[best_inliers], axis=0, weights=weights / weights.sum()
    )
    errors = np.linalg.norm(translations - refined[None, :], axis=1)
    compatible = errors <= translation_threshold
    final_inliers = np.zeros(len(matrices), dtype=bool)
    for frame_id in np.unique(frame_ids[compatible]):
        candidates = np.flatnonzero(compatible & (frame_ids == frame_id))
        winner = max(
            candidates,
            key=lambda index: (float(scores[index]), -float(errors[index])),
        )
        final_inliers[winner] = True
    if int(final_inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            f"Refined position has only {int(final_inliers.sum())} inliers"
        )
    return refined, final_inliers, compatible, errors


def ransac_orientation(
    rotations: np.ndarray,
    scores: np.ndarray,
    frame_ids: np.ndarray,
    rotation_threshold_deg: float,
    minimum_inliers: int,
    symmetries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the dominant orientation inside one position cluster."""
    best_inliers = None
    best_key = None
    for candidate in rotations:
        errors = rotation_errors_deg(candidate, rotations, symmetries)
        compatible = errors <= rotation_threshold_deg
        selected = np.zeros(len(rotations), dtype=bool)
        for frame_id in np.unique(frame_ids[compatible]):
            candidates = np.flatnonzero(compatible & (frame_ids == frame_id))
            winner = max(
                candidates,
                key=lambda index: (float(scores[index]), -float(errors[index])),
            )
            selected[winner] = True
        key = (
            int(selected.sum()),
            float(scores[selected].sum()),
            -float(np.median(errors[selected])),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_inliers = selected
    assert best_inliers is not None
    if int(best_inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            f"Orientation RANSAC found only {int(best_inliers.sum())} inliers; "
            f"minimum requested is {minimum_inliers}"
        )

    weights = np.maximum(scores[best_inliers], 1e-8)
    weights /= weights.sum()
    reference = rotations[np.flatnonzero(best_inliers)[0]]
    aligned = rotations[best_inliers].copy()
    for index, observed in enumerate(aligned):
        equivalents = np.einsum("ij,kjl->kil", observed, symmetries)
        errors = raw_rotation_errors_deg(reference, equivalents)
        aligned[index] = equivalents[int(np.argmin(errors))]
    refined = Rotation.from_matrix(aligned).mean(weights=weights).as_matrix()
    errors = rotation_errors_deg(refined, rotations, symmetries)
    compatible = errors <= rotation_threshold_deg
    final_inliers = np.zeros(len(rotations), dtype=bool)
    for frame_id in np.unique(frame_ids[compatible]):
        candidates = np.flatnonzero(compatible & (frame_ids == frame_id))
        winner = max(
            candidates,
            key=lambda index: (float(scores[index]), -float(errors[index])),
        )
        final_inliers[winner] = True
    if int(final_inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            f"Refined orientation has only {int(final_inliers.sum())} inliers; "
            f"minimum requested is {minimum_inliers}"
        )
    return refined, final_inliers, errors


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project / "scripts/configs/default.yaml",
        help="Project config containing path and dataset_id.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Optional explicit dataset directory; overrides --config path resolution.",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        help="Optional override for the object-specific batch pose JSON.",
    )
    parser.add_argument(
        "--reconstruction",
        type=Path,
        help="Optional override for the dataset COLMAP reconstruction.",
    )
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        help="Optional override for the dataset Sim(3) alignment directory.",
    )
    parser.add_argument(
        "--object-name",
        help=(
            "Optional single-object filter. When omitted, process every "
            "object directory under data/####/sam6d_sequences."
        ),
    )
    parser.add_argument(
        "--object-path",
        type=Path,
        help="Optional CAD override; defaults to objects/stl/<object-name>.stl.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--translation-threshold", type=float, default=0.01)
    parser.add_argument("--rotation-threshold-deg", type=float, default=5.0)
    parser.add_argument(
        "--maximum-rotation-threshold-deg",
        type=float,
        default=30.0,
        help=(
            "Maximum adaptive orientation-consensus threshold. If no "
            "multi-frame consensus exists at this value, retain the highest-"
            "scoring orientation and mark it as a low-confidence fallback."
        ),
    )
    parser.add_argument("--minimum-score", type=float, default=0.0)
    parser.add_argument("--minimum-inliers", type=int, default=3)
    parser.add_argument(
        "--minimum-instance-frame-fraction",
        type=float,
        default=0.05,
        help=(
            "Automatically reject pose clusters supported by fewer than this "
            "fraction of frames containing candidates (default: 0.05). Set "
            "to zero to use only --minimum-inliers."
        ),
    )
    parser.add_argument(
        "--min-silhouette-iou",
        type=float,
        default=float(os.environ.get("LABEL_FACTORY_MIN_SILHOUETTE_IOU", "0")),
        help=(
            "Reject a pose cluster whose CAD silhouette overlaps the observed instance "
            "masks with a median IoU below this value (0 disables the check). The IoU is "
            "low for a correct pose when the segmenter covers only part of the object, so "
            "prefer --min-surface-explained. Default from LABEL_FACTORY_MIN_SILHOUETTE_IOU, "
            "else 0."
        ),
    )
    parser.add_argument(
        "--min-surface-explained",
        type=float,
        default=float(os.environ.get("LABEL_FACTORY_MIN_SURFACE_EXPLAINED", "0")),
        help=(
            "Reject a pose cluster when less than this share of the visible CAD surface "
            "(median over supporting frames) is explained by the instance mask or by a "
            "measured depth that agrees with the CAD; points hidden by a closer surface are "
            "ignored. Robust to partial masks; about 0.9 removes similar-looking objects "
            "with a different geometry (0 disables). Default from "
            "LABEL_FACTORY_MIN_SURFACE_EXPLAINED, else 0."
        ),
    )
    parser.add_argument(
        "--min-mask-containment",
        type=float,
        default=float(os.environ.get("LABEL_FACTORY_MIN_MASK_CONTAINMENT", "0")),
        help=(
            "Reject a pose cluster when less than this share of the observed mask lies "
            "inside the CAD silhouette (catches objects larger than the model or masks "
            "elsewhere; a partial mask still passes). About 0.8 (0 disables). Default from "
            "LABEL_FACTORY_MIN_MASK_CONTAINMENT, else 0."
        ),
    )
    parser.add_argument(
        "--model-scale",
        type=float,
        default=0.001,
        help="CAD model unit to metre (default: millimetres), used for the silhouette check.",
    )
    parser.add_argument(
        "--verification-max-frames",
        type=int,
        default=40,
        help="Maximum supporting frames per cluster used for the silhouette check.",
    )
    parser.add_argument(
        "--low-support-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When an object cannot reach --minimum-inliers or the frame-support "
            "threshold, estimate one pose from its best-supported cluster instead "
            "of excluding the object. Such poses are marked low_support in the "
            "output. Objects that pass the normal thresholds are unaffected."
        ),
    )
    parser.add_argument(
        "--fallback-minimum-inliers",
        type=int,
        default=1,
        help="Minimum unique-frame inliers for the low-support fallback (default: 1).",
    )
    parser.add_argument(
        "--maximum-instances",
        type=int,
        default=0,
        help=(
            "Optional hard maximum physical instances per CAD object. Zero "
            "uses automatic frame-support filtering without a fixed count."
        ),
    )
    parser.add_argument(
        "--symmetry-mode",
        choices=("auto", "none"),
        default="auto",
        help="Detect finite CAD symmetries automatically, or disable symmetry.",
    )
    parser.add_argument(
        "--symmetry-tolerance",
        type=float,
        default=0.01,
        help="Maximum CAD mismatch as a fraction of model diameter.",
    )
    return parser.parse_args()


def run_object(
    args: argparse.Namespace,
    config_path: Path,
    dataset: Path,
    reconstruction: Path,
    alignment_dir: Path,
    camera_poses: dict[int, np.ndarray],
    object_name: str,
    low_support: bool = False,
) -> None:
    if not 0.0 <= args.minimum_instance_frame_fraction <= 1.0:
        raise ValueError("--minimum-instance-frame-fraction must be in [0, 1]")
    if args.maximum_rotation_threshold_deg < args.rotation_threshold_deg:
        raise ValueError(
            "--maximum-rotation-threshold-deg must be at least "
            "--rotation-threshold-deg"
        )
    object_path = (
        args.object_path.expanduser().resolve()
        if args.object_path
        else (dataset / "objects" / "stl" / f"{object_name}.stl").resolve()
    )
    if not object_path.is_file():
        raise FileNotFoundError(object_path)
    object_output = dataset / "sam6d_sequences" / object_name
    poses_path = (
        args.poses.expanduser().resolve()
        if args.poses
        else object_output / f"{object_name}_poses_per_frame.json"
    )
    aggregate = json.loads(poses_path.read_text())

    frame_ids: list[int] = []
    detection_indices: list[int] = []
    bounding_boxes: list[list[float] | None] = []
    scores: list[float] = []
    world_from_objects: list[np.ndarray] = []
    skipped: list[dict] = []
    segmentations: dict[tuple[int, int], dict | None] = {}

    for frame in aggregate.get("frames", []):
        idx = int(frame["frame_id"])
        candidates = frame.get("pose_candidates")
        if candidates is None:
            pem_path = Path(frame.get("pem_result_path", ""))
            if pem_path.is_file():
                candidates = json.loads(pem_path.read_text())
            else:
                best = frame.get("best_pose")
                candidates = [] if best is None else [best]
        candidates = [
            pose for pose in candidates
            if isinstance(pose, dict) and "R" in pose and "t" in pose
        ]
        if not candidates:
            skipped.append({"frame_id": idx, "reason": "no_object_pose"})
            continue
        if idx not in camera_poses:
            skipped.append({"frame_id": idx, "reason": "no_camera_pose"})
            continue

        accepted_in_frame = 0
        for detection_index, pose in enumerate(candidates):
            score = float(pose.get("score", 0.0))
            if score < args.minimum_score:
                continue
            camera_from_object = transform(
                np.asarray(pose["R"], dtype=np.float64),
                np.asarray(pose["t"], dtype=np.float64) / 1000.0,
            )
            world_from_object = camera_poses[idx] @ camera_from_object
            frame_ids.append(idx)
            detection_indices.append(detection_index)
            segmentations[(idx, detection_index)] = pose.get("segmentation")
            bounding_boxes.append(pose.get("bbox"))
            scores.append(score)
            world_from_objects.append(world_from_object)
            accepted_in_frame += 1
        if accepted_in_frame == 0:
            skipped.append({"frame_id": idx, "reason": "score_below_threshold"})

    if not world_from_objects:
        raise RuntimeError("No usable frame poses were found")

    matrices = np.stack(world_from_objects)
    score_array = np.asarray(scores)
    symmetries = (
        detect_model_symmetries(
            object_path,
            args.symmetry_tolerance,
        )
        if args.symmetry_mode == "auto"
        else np.eye(3)[None, :, :]
    )
    print(f"Using {len(symmetries)} rotational object symmetries")

    frame_id_array = np.asarray(frame_ids, dtype=np.int64)
    remaining = np.ones(len(matrices), dtype=bool)
    instance_results = []
    while int(remaining.sum()) >= args.minimum_inliers:
        if args.maximum_instances and len(instance_results) >= args.maximum_instances:
            break
        indices = np.flatnonzero(remaining)
        try:
            (
                refined_translation,
                local_position_inliers,
                local_position_compatible,
                local_translation_errors,
            ) = ransac_position(
                matrices[indices],
                score_array[indices],
                frame_id_array[indices],
                args.translation_threshold,
                args.minimum_inliers,
            )
        except RuntimeError:
            break

        position_local_indices = np.flatnonzero(local_position_compatible)
        compatible_indices = indices[position_local_indices]
        orientation_status = "consensus"
        orientation_threshold_used = args.rotation_threshold_deg
        orientation_error = None
        while orientation_threshold_used <= args.maximum_rotation_threshold_deg + 1e-9:
            try:
                (
                    refined_rotation,
                    position_cluster_orientation_inliers,
                    position_cluster_rotation_errors,
                ) = ransac_orientation(
                    matrices[compatible_indices, :3, :3],
                    score_array[compatible_indices],
                    frame_id_array[compatible_indices],
                    orientation_threshold_used,
                    args.minimum_inliers,
                    symmetries,
                )
                if orientation_threshold_used > args.rotation_threshold_deg:
                    orientation_status = "adaptive_consensus"
                    print(
                        "Orientation consensus found after relaxing threshold "
                        f"to {orientation_threshold_used:g} degrees"
                    )
                break
            except RuntimeError as error:
                orientation_error = error
                orientation_threshold_used += 5.0
        else:
            # Position is independently reliable. Do not delete the physical
            # instance merely because PEM rotations are dispersed; preserve
            # the highest-confidence observed orientation and flag it clearly.
            local_winner = int(np.argmax(score_array[compatible_indices]))
            refined_rotation = matrices[
                compatible_indices[local_winner], :3, :3
            ]
            position_cluster_orientation_inliers = np.zeros(
                len(compatible_indices), dtype=bool
            )
            position_cluster_orientation_inliers[local_winner] = True
            position_cluster_rotation_errors = rotation_errors_deg(
                refined_rotation,
                matrices[compatible_indices, :3, :3],
                symmetries,
            )
            orientation_status = "fallback_best_score"
            orientation_threshold_used = None
            print(
                "WARNING: no stable multi-frame orientation was found; "
                "retaining the highest-scoring PEM orientation as a "
                f"low-confidence fallback ({orientation_error})"
            )

        result = transform(refined_rotation, refined_translation)
        orientation_inlier_indices = compatible_indices[
            position_cluster_orientation_inliers
        ]
        instance_number = len(instance_results) + 1
        quaternion = Rotation.from_matrix(result[:3, :3]).as_quat()
        observations = []
        orientation_lookup = {
            int(global_index): local_index
            for local_index, global_index in enumerate(compatible_indices)
        }
        for global_index in compatible_indices:
            local_index = int(np.flatnonzero(indices == global_index)[0])
            orientation_index = orientation_lookup[int(global_index)]
            observations.append(
                {
                    "frame_id": frame_ids[global_index],
                    "detection_index": detection_indices[global_index],
                    "bbox": bounding_boxes[global_index],
                    "score": scores[global_index],
                    "selected_position_inlier": bool(
                        local_position_inliers[local_index]
                    ),
                    "selected_orientation_inlier": bool(
                        position_cluster_orientation_inliers[orientation_index]
                    ),
                    "translation_error_m": float(
                        local_translation_errors[local_index]
                    ),
                    "rotation_error_deg": float(
                        position_cluster_rotation_errors[orientation_index]
                    ),
                }
            )
        instance_results.append(
            {
                "instance_id": f"{object_name}_{instance_number:03d}",
                "instance_index": instance_number,
                "R": result[:3, :3].tolist(),
                "quaternion_xyzw": quaternion.tolist(),
                "translation": result[:3, 3].tolist(),
                "world_from_object": result.tolist(),
                "number_of_inliers": int(local_position_inliers.sum()),
                "number_of_position_inliers": int(local_position_inliers.sum()),
                "number_of_orientation_inliers": int(
                    position_cluster_orientation_inliers.sum()
                ),
                "orientation_status": orientation_status,
                "orientation_threshold_deg": orientation_threshold_used,
                "position_inlier_frame_ids": sorted(
                    {
                        frame_ids[index]
                        for index in indices[local_position_inliers]
                    }
                ),
                "orientation_inlier_frame_ids": sorted(
                    {
                        frame_ids[index]
                        for index in orientation_inlier_indices
                    }
                ),
                "inlier_frame_ids": sorted(
                    {frame_ids[index] for index in orientation_inlier_indices}
                ),
                "observations": observations,
            }
        )
        # Suppress every proposal compatible with this refined physical pose,
        # including duplicate SAM masks that were not selected as the one
        # representative for their frame.
        remaining[compatible_indices] = False

    if not instance_results:
        raise RuntimeError(
            "No pose cluster reached the requested number of unique-frame inliers"
        )

    candidate_frame_count = len(np.unique(frame_id_array))
    automatic_support_threshold = max(
        args.minimum_inliers,
        int(
            np.ceil(
                candidate_frame_count * args.minimum_instance_frame_fraction
            )
        ),
    )
    raw_instance_results = instance_results
    support_candidates = sorted(
        raw_instance_results,
        key=lambda instance: instance["number_of_position_inliers"],
        reverse=True,
    )
    instance_results = []
    rejected_clusters = []
    position_suppression_threshold = args.translation_threshold * 2.0
    for instance in support_candidates:
        if instance["number_of_position_inliers"] < automatic_support_threshold:
            rejected_clusters.append(
                {
                    "number_of_inliers": instance["number_of_position_inliers"],
                    "translation": instance["translation"],
                    "reason": "insufficient_unique_frame_support",
                }
            )
            continue
        translation = np.asarray(instance["translation"], dtype=np.float64)
        duplicate_of = next(
            (
                accepted
                for accepted in instance_results
                if np.linalg.norm(
                    translation
                    - np.asarray(accepted["translation"], dtype=np.float64)
                ) <= position_suppression_threshold
            ),
            None,
        )
        if duplicate_of is not None:
            rejected_clusters.append(
                {
                    "number_of_inliers": instance["number_of_position_inliers"],
                    "translation": instance["translation"],
                    "reason": "duplicate_position_cluster",
                    "duplicate_of": duplicate_of["instance_id"],
                }
            )
            continue
        instance_results.append(instance)
    print(
        "Automatic instance support threshold: "
        f"{automatic_support_threshold}/{candidate_frame_count} candidate frames"
    )
    if rejected_clusters:
        print(
            "Rejected outlier pose clusters with support: "
            + ", ".join(
                str(cluster["number_of_inliers"])
                for cluster in rejected_clusters
            )
        )
    if not instance_results:
        raise RuntimeError(
            "All pose clusters were rejected by automatic frame-support "
            "filtering. Lower --minimum-instance-frame-fraction if the object "
            "is visible in very few frames."
        )
    # Check each fused pose against the CAD model: consistency between frames does not
    # exclude a different object with a similar appearance.
    try:
        camera_json = json.loads((dataset / "camera.json").read_text())
        camera_matrix = np.asarray(camera_json["cam_K"], dtype=np.float64).reshape(3, 3)
        depth_scale = float(camera_json.get("depth_scale", 1.0))  # raw unit = millimetre
        model_points = sample_model_points(object_path, args.model_scale)
    except Exception as error:  # noqa: BLE001 - the check is optional
        camera_matrix, model_points = None, None
        print(f"Geometric verification skipped: {error}")
    if model_points is not None:
        depth_files = {
            numeric_frame_id(path): path
            for path in (dataset / "depth_maps").glob("*.png")
        } if (dataset / "depth_maps").is_dir() else {}

        def load_depth(frame_id: int):
            path = depth_files.get(frame_id)
            if path is None:
                return None
            import cv2
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.ndim != 2:
                return None
            return raw.astype(np.float64) * depth_scale / 1000.0

        thresholds = (
            ("silhouette_iou_median", args.min_silhouette_iou, "silhouette_iou_below_threshold"),
            ("surface_explained_median", args.min_surface_explained,
             "surface_explained_below_threshold"),
            ("mask_containment_median", args.min_mask_containment,
             "mask_containment_below_threshold"),
        )
        verified = []
        for instance in instance_results:
            instance.update(verify_instance(
                instance, model_points, camera_poses, camera_matrix, segmentations,
                args.verification_max_frames, load_depth,
            ))

            def show(key):
                value = instance[key]
                return "n/a" if value is None else f"{value:.2f}"
            print(
                f"{object_name}: cluster with {instance['number_of_position_inliers']} "
                f"position inliers, silhouette IoU {show('silhouette_iou_median')}, "
                f"mask containment {show('mask_containment_median')}, "
                f"surface explained {show('surface_explained_median')} "
                f"over {instance['silhouette_iou_frames']} frames"
            )
            failed = next(
                (reason for key, limit, reason in thresholds
                 if limit > 0 and instance[key] is not None and instance[key] < limit),
                None,
            )
            if failed:
                rejected_clusters.append({
                    "number_of_inliers": instance["number_of_position_inliers"],
                    "translation": instance["translation"],
                    "silhouette_iou_median": instance["silhouette_iou_median"],
                    "mask_containment_median": instance["mask_containment_median"],
                    "surface_explained_median": instance["surface_explained_median"],
                    "reason": failed,
                })
                continue
            verified.append(instance)
        instance_results = verified
        if not instance_results:
            raise RuntimeError(
                "All pose clusters were rejected by the geometric verification; the "
                "object may not be present, or only objects with a different geometry "
                "were matched."
            )
    for instance_number, instance in enumerate(instance_results, 1):
        instance["instance_id"] = f"{object_name}_{instance_number:03d}"
        instance["instance_index"] = instance_number

    primary = instance_results[0]

    payload = {
        "object_name": object_name,
        "object_path": str(object_path),
        "coordinate_system": "world",
        "translation_unit": "metre",
        "instance_count": len(instance_results),
        "low_support": low_support,
        "instances": instance_results,
        # Keep the strongest instance at the legacy top-level fields so older
        # scene tools remain usable while they migrate to the instances list.
        "R": primary["R"],
        "quaternion_xyzw": primary["quaternion_xyzw"],
        "translation": primary["translation"],
        "world_from_object": primary["world_from_object"],
        "ransac": {
            "estimation_mode": "position_then_orientation",
            "translation_threshold_m": args.translation_threshold,
            "rotation_threshold_deg": args.rotation_threshold_deg,
            "maximum_rotation_threshold_deg": (
                args.maximum_rotation_threshold_deg
            ),
            "minimum_score": args.minimum_score,
            "symmetry_mode": args.symmetry_mode,
            "symmetry_tolerance_fraction": args.symmetry_tolerance,
            "number_of_rotational_symmetries": len(symmetries),
            "maximum_instances": args.maximum_instances,
            "minimum_instance_frame_fraction": (
                args.minimum_instance_frame_fraction
            ),
            "low_support_fallback_used": low_support,
            "min_silhouette_iou": args.min_silhouette_iou,
            "min_surface_explained": args.min_surface_explained,
            "min_mask_containment": args.min_mask_containment,
            "automatic_support_threshold_frames": automatic_support_threshold,
            "position_suppression_threshold_m": position_suppression_threshold,
            "number_of_candidate_frames": candidate_frame_count,
            "number_of_candidate_observations": len(frame_ids),
            "number_of_raw_pose_clusters": len(raw_instance_results),
            "number_of_instances": len(instance_results),
            "remaining_unclustered_observations": int(remaining.sum()),
            "rejected_pose_clusters": rejected_clusters,
        },
        "skipped_frames": skipped,
        "sources": {
            "config_path": str(config_path),
            "object_poses": str(poses_path),
            "colmap_reconstruction": str(reconstruction),
            "sim3_alignment": str(alignment_dir),
        },
    }
    output = (
        args.output.expanduser().resolve()
        if args.output
        else object_output / f"{object_name}_world_pose.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Saved {len(instance_results)} {object_name} instance pose(s) to {output}: "
        + ", ".join(
            f"{item['instance_id']}="
            f"{item['number_of_position_inliers']} position/"
            f"{item['number_of_orientation_inliers']} orientation inliers"
            for item in instance_results
        )
    )


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    dataset = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir
        else dataset_from_config(config_path)
    )
    sequences_dir = dataset / "sam6d_sequences"
    if not sequences_dir.is_dir():
        raise FileNotFoundError(sequences_dir)

    reconstruction = (
        args.reconstruction.expanduser().resolve()
        if args.reconstruction
        else dataset / "reconstruction" / "sfm_filtered"
    )
    alignment_dir = (
        args.alignment_dir.expanduser().resolve()
        if args.alignment_dir
        else dataset / "reconstruction" / "metric_alignment"
    )
    for required in (
        reconstruction / "images.bin",
        reconstruction / "cameras.bin",
        alignment_dir / "scale.npy",
        alignment_dir / "rotation.npy",
        alignment_dir / "translation.npy",
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required metric reconstruction input not found: {required}. "
                "Run scripts/reconstruct_rgbd_scene.py first."
            )
    camera_poses = load_world_from_camera(reconstruction, alignment_dir)

    if args.object_name:
        object_names = [safe_object_name(args.object_name)]
    else:
        object_specific_overrides = {
            "--poses": args.poses,
            "--object-path": args.object_path,
            "--output": args.output,
        }
        supplied = [
            option
            for option, value in object_specific_overrides.items()
            if value is not None
        ]
        if supplied:
            raise ValueError(
                f"{', '.join(supplied)} can only be used together with "
                "--object-name"
            )
        object_names = []
        ignored_directories = []
        for path in sequences_dir.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            name = safe_object_name(path.name)
            poses_file = path / f"{name}_poses_per_frame.json"
            cad_file = dataset / "objects" / "stl" / f"{name}.stl"
            if poses_file.is_file() and cad_file.is_file():
                object_names.append(name)
            else:
                ignored_directories.append(path.name)
        object_names.sort(key=str.lower)
        if ignored_directories:
            print(
                "Ignoring non-object sequence directories: "
                + ", ".join(sorted(ignored_directories, key=str.lower))
            )
        if not object_names:
            raise RuntimeError(
                f"No object directories found under {sequences_dir}"
            )

    print(
        f"RANSAC objects ({len(object_names)}): "
        + ", ".join(object_names)
    )
    failures: list[tuple[str, str]] = []
    successes: list[str] = []
    low_support: list[str] = []
    for index, object_name in enumerate(object_names, 1):
        print(
            f"\n=== RANSAC object [{index}/{len(object_names)}]: "
            f"{object_name} ==="
        )
        try:
            try:
                run_object(
                    args,
                    config_path,
                    dataset,
                    reconstruction,
                    alignment_dir,
                    camera_poses,
                    object_name,
                )
            except RuntimeError as strict_error:
                if not args.low_support_fallback:
                    raise
                # The object was detected too rarely for the normal thresholds.
                # Keep only its single best-supported cluster rather than
                # dropping the object; downstream tools see low_support=true.
                print(
                    f"{object_name}: {strict_error}\n"
                    f"{object_name}: retrying with the low-support fallback "
                    f"(minimum {args.fallback_minimum_inliers} unique-frame "
                    "inliers, best cluster only)"
                )
                relaxed = copy.copy(args)
                relaxed.minimum_inliers = args.fallback_minimum_inliers
                relaxed.minimum_instance_frame_fraction = 0.0
                relaxed.maximum_instances = 1
                try:
                    run_object(
                        relaxed,
                        config_path,
                        dataset,
                        reconstruction,
                        alignment_dir,
                        camera_poses,
                        object_name,
                        low_support=True,
                    )
                except Exception:
                    # Report the original reason; the fallback had no data either.
                    raise strict_error
                low_support.append(object_name)
            successes.append(object_name)
        except Exception as error:
            failures.append((object_name, str(error)))
            print(f"ERROR: {object_name}: {error}")
            # Never let scene generation consume a pose left by an older,
            # successful run when the current observations rejected it.
            stale_pose = (
                args.output.expanduser().resolve()
                if args.output and args.object_name
                else sequences_dir / object_name / f"{object_name}_world_pose.json"
            )
            if stale_pose.is_file():
                archived = stale_pose.with_suffix(stale_pose.suffix + ".failed")
                suffix = 1
                while archived.exists():
                    archived = stale_pose.with_suffix(
                        stale_pose.suffix + f".failed.{suffix}"
                    )
                    suffix += 1
                stale_pose.replace(archived)
                print(
                    "Archived stale world pose so scene creation excludes "
                    f"{object_name}: {archived}"
                )

    summary_path = sequences_dir / "ransac_run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "objects_requested": object_names,
                "objects_succeeded": successes,
                "objects_low_support": low_support,
                "objects_excluded": [
                    {"object_name": name, "reason": message}
                    for name, message in failures
                ],
            },
            indent=2,
        )
        + "\n"
    )

    if low_support:
        print(
            "WARNING: low-support poses (few observations, less reliable): "
            + ", ".join(low_support)
        )
    if failures:
        details = "; ".join(
            f"{name}: {message}" for name, message in failures
        )
        if not successes:
            raise RuntimeError(
                f"RANSAC failed for all {len(object_names)} objects: {details}"
            )
        print(
            f"\nWARNING: RANSAC excluded {len(failures)}/{len(object_names)} "
            f"objects: {details}"
        )
        print(
            "Scene creation will use only: " + ", ".join(successes)
        )
    else:
        print(f"\nCompleted RANSAC for all {len(object_names)} objects.")
    print(f"RANSAC run summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
