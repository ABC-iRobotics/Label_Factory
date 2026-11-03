"""Geometric verification of consolidated object poses.

Multi-frame consolidation only checks that the per-frame estimates agree with each other.
An object whose shape merely resembles the CAD model can still produce a consistent but
wrong cluster. This module checks the fused pose against the CAD model itself: the CAD
silhouette at the fused pose is compared with the instance mask that the pose estimator
used in every supporting frame. A correct pose gives a high median IoU (about 0.8-0.95 on
well-matched objects); a similar-looking object with different geometry gives a clearly
lower one.

The silhouette is approximated by projecting surface samples of the CAD model with a
z-buffer (no renderer needed), so it runs in the plain reconstruction environment.

The instance segmenter does not always cover a whole object, so a correct pose can have a
low IoU with a partial mask. Three measures are therefore computed per frame:

  silhouette_iou     IoU of CAD silhouette and mask (low for partial masks, kept as information)
  mask_containment   share of the mask inside the CAD silhouette; a mask that is larger than
                     the CAD, or lies elsewhere, scores low, a partial mask still scores high
  surface_explained  share of the visible CAD surface that is covered by the mask OR whose
                     measured depth agrees with the CAD depth (so unsegmented parts of the
                     object still count); points occluded by something closer, and points
                     without any depth reading outside the mask, are ignored.
                     CAD surface over empty background or other objects is not explained.

A different object with a similar look fails mainly on surface_explained (its CAD
silhouette extends over background); an object larger than the CAD fails on containment.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # scipy is always available in the pipeline environment
    cv2 = None


def decode_segmentation(segmentation):
    """COCO run-length mask (uncompressed list or compressed string) -> boolean array."""
    if not isinstance(segmentation, dict) or "counts" not in segmentation:
        return None
    height, width = segmentation["size"]
    counts = segmentation["counts"]
    if isinstance(counts, (list, tuple)):
        flat = np.zeros(height * width, dtype=bool)
        position, value = 0, False
        for run in counts:
            if value:
                flat[position:position + run] = True
            position += run
            value = not value
        return flat.reshape((width, height)).T  # RLE is column-major
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None
    encoded = counts.encode() if isinstance(counts, str) else counts
    return mask_utils.decode({"size": [height, width], "counts": encoded}).astype(bool)


def sample_model_points(model_path, model_scale: float, count: int = 20000) -> np.ndarray:
    """Surface samples of the CAD model in metres (deterministic)."""
    import trimesh
    mesh = trimesh.load(model_path, force="mesh", process=False)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count, seed=0)
    except TypeError:  # older trimesh without the seed argument
        state = np.random.get_state()
        np.random.seed(0)
        points, _ = trimesh.sample.sample_surface(mesh, count)
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64) * model_scale


def _close(binary: np.ndarray, size: int = 9) -> np.ndarray:
    if cv2 is not None:
        return cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((size, size), np.uint8)) > 0
    from scipy.ndimage import binary_closing
    return binary_closing(binary, structure=np.ones((size, size), bool))


Z_BUFFER_TOLERANCE = 0.004   # m: points within this of the nearest depth are visible
OCCLUSION_MARGIN = 0.020     # m: measured depth this much closer than the CAD = occluded
DEPTH_AGREEMENT = 0.015      # m: measured depth this close to the CAD = surface present


def frame_metrics(points, world_from_object, world_from_camera, K, mask,
                  depth_m=None) -> dict | None:
    """Silhouette IoU, mask containment and explained surface of one pose in one frame."""
    camera_from_object = np.linalg.inv(world_from_camera) @ world_from_object
    camera_points = points @ camera_from_object[:3, :3].T + camera_from_object[:3, 3]
    z = camera_points[:, 2]
    keep = z > 0.05
    camera_points, z = camera_points[keep], z[keep]
    if not len(z):
        return None
    projected = camera_points @ K.T
    u = np.rint(projected[:, 0] / projected[:, 2]).astype(int)
    v = np.rint(projected[:, 1] / projected[:, 2]).astype(int)
    height, width = mask.shape
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    if len(z) < 50:
        return 0.0
    z_buffer = np.full((height, width), np.inf)
    np.minimum.at(z_buffer, (v, u), z)
    visible = z <= z_buffer[v, u] + Z_BUFFER_TOLERANCE  # only the visible surface
    silhouette = np.zeros((height, width), dtype=bool)
    silhouette[v[visible], u[visible]] = True
    silhouette = _close(silhouette)
    union = np.logical_or(silhouette, mask).sum()
    if not union:
        return None
    intersection = np.logical_and(silhouette, mask).sum()
    result = {
        "iou": float(intersection / union),
        "containment": float(intersection / mask.sum()) if mask.sum() else None,
        "explained": None,
    }
    if depth_m is not None and depth_m.shape == mask.shape:
        measured = depth_m[v[visible], u[visible]]
        model = z[visible]
        occluded = (measured > 0) & (measured < model - OCCLUSION_MARGIN)
        present = (measured > 0) & (np.abs(measured - model) < DEPTH_AGREEMENT)
        in_mask = mask[v[visible], u[visible]]
        # A pixel without a depth reading says nothing (dark or shiny surfaces, sensor
        # holes), so it counts only where the mask covers it; occluded points never count.
        considered = ~occluded & (in_mask | (measured > 0))
        if considered.sum() >= 20:
            result["explained"] = float(((in_mask | present) & considered).sum()
                                        / considered.sum())
    return result


def silhouette_iou(points, world_from_object, world_from_camera, K, mask) -> float | None:
    """IoU between the CAD silhouette at the given pose and an observed mask."""
    metrics = frame_metrics(points, world_from_object, world_from_camera, K, mask)
    return None if metrics is None else metrics["iou"]


def verify_instance(instance: dict, points, camera_poses: dict, K, segmentations: dict,
                    max_frames: int = 40, depth_loader=None) -> dict:
    """Median verification measures of a fused instance over its position-inlier frames.

    segmentations maps (frame_id, detection_index) -> RLE mask of the pose candidate;
    depth_loader(frame_id) returns the metric depth map in metres, or None.
    """
    world_from_object = np.asarray(instance["world_from_object"], dtype=np.float64)
    observations = [o for o in instance.get("observations", [])
                    if o.get("selected_position_inlier")]
    if len(observations) > max_frames:
        picks = np.linspace(0, len(observations) - 1, max_frames).astype(int)
        observations = [observations[i] for i in picks]
    values = {"iou": [], "containment": [], "explained": []}
    for observation in observations:
        frame_id = int(observation["frame_id"])
        segmentation = segmentations.get((frame_id, int(observation["detection_index"])))
        mask = decode_segmentation(segmentation) if segmentation else None
        if mask is None or frame_id not in camera_poses:
            continue
        depth = depth_loader(frame_id) if depth_loader else None
        metrics = frame_metrics(points, world_from_object, camera_poses[frame_id], K, mask, depth)
        if metrics is None:
            continue
        for key in values:
            if metrics[key] is not None:
                values[key].append(metrics[key])

    def median(key):
        return float(np.median(values[key])) if values[key] else None
    return {
        "silhouette_iou_median": median("iou"),
        "silhouette_iou_frames": len(values["iou"]),
        "mask_containment_median": median("containment"),
        "surface_explained_median": median("explained"),
    }
