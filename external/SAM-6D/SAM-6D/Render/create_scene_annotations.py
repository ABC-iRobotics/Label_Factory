import blenderproc as bproc

"""BlenderProc pipeline for posed-object scene and per-camera annotations."""

import argparse
import json
import os
from pathlib import Path

import bpy
import cv2
import numpy as np


def coco_rle(mask):
    """Return uncompressed COCO RLE (column-major order)."""
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts = []
    previous = 0
    run = 0
    for pixel in pixels:
        value = int(pixel)
        if value == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = value
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def bbox_xywh(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
output = Path(manifest["output_dir"])
masks_dir = output / "masks"
segmentation_raw_dir = output / "segmentation_raw"
masks_dir.mkdir(parents=True, exist_ok=True)
segmentation_raw_dir.mkdir(parents=True, exist_ok=True)

bproc.init()
bproc.camera.set_intrinsics_from_K_matrix(
    np.asarray(manifest["cam_K"], dtype=np.float64),
    int(manifest["image_width"]),
    int(manifest["image_height"]),
)

scale = float(manifest["model_scale_to_metre"])
scale_matrix = np.diag([scale, scale, scale, 1.0])
object_poses = {}
for item in manifest["objects"]:
    loaded = bproc.loader.load_obj(item["model_path"])
    if len(loaded) != 1:
        raise RuntimeError(
            f"Expected one mesh in {item['model_path']}, got {len(loaded)}"
    )
    obj = loaded[0]
    instance_name = item.get("instance_name", item["object_name"])
    obj.set_name(instance_name)
    obj.set_cp("object_name", item["object_name"])
    obj.set_cp("instance_name", instance_name)
    obj.set_cp("category_id", int(item["object_id"]))
    world_from_object = np.asarray(
        item["world_from_object"], dtype=np.float64
    )
    obj.set_local2world_mat(world_from_object @ scale_matrix)
    object_poses[instance_name] = world_from_object

# COLMAP/OpenCV cameras use +X right, +Y down, +Z forward. Blender cameras use
# +X right, +Y up, -Z forward.
cv_camera_from_blender_camera = np.diag([1.0, -1.0, -1.0, 1.0])
for frame_index, frame in enumerate(manifest["frames"]):
    world_from_camera_cv = np.asarray(
        frame["world_from_camera_cv"], dtype=np.float64
    )
    world_from_camera_blender = (
        world_from_camera_cv @ cv_camera_from_blender_camera
    )
    bproc.camera.add_camera_pose(
        world_from_camera_blender, frame=frame_index
    )

# Save before the segmentation renderer temporarily replaces materials.
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = len(manifest["frames"])
bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"))

segmentation = bproc.renderer.render_segmap(
    output_dir=str(segmentation_raw_dir),
    map_by=[
        "instance", "cp_object_name", "cp_instance_name", "cp_category_id"
    ],
    default_values={"object_name": "", "instance_name": "", "category_id": 0},
)
segmaps = segmentation["instance_segmaps"]
attribute_maps = segmentation["instance_attribute_maps"]
if len(segmaps) != len(manifest["frames"]):
    raise RuntimeError(
        f"Expected {len(manifest['frames'])} masks, got {len(segmaps)}"
    )

annotations = {
    "format": "sam6d_world_scene_annotations_v1",
    "coordinate_conventions": {
        "world_from_camera": "OpenCV camera: +X right, +Y down, +Z forward",
        "world_from_object": "Object CAD frame, translation in metres",
        "quaternion_order": "xyzw",
    },
    "image_width": manifest["image_width"],
    "image_height": manifest["image_height"],
    "cam_K": manifest["cam_K"],
    "model_scale_to_metre": scale,
    "scene_path": str(output / "scene.blend"),
    "objects": manifest["objects"],
    "frames": [],
}

for frame, segmap, mapping in zip(
    manifest["frames"], segmaps, attribute_maps
):
    frame_id = int(frame["frame_id"])
    world_from_camera = np.asarray(
        frame["world_from_camera_cv"], dtype=np.float64
    )
    camera_from_world = np.linalg.inv(world_from_camera)
    id_to_object = {
        int(entry["idx"]): entry
        for entry in mapping
        if entry.get("object_name")
    }
    visible = []
    for instance_id, entry in sorted(id_to_object.items()):
        object_name = entry["object_name"]
        instance_name = entry.get("instance_name", object_name)
        mask = np.asarray(segmap == instance_id)
        bbox = bbox_xywh(mask)
        if bbox is None:
            continue
        camera_from_object = (
            camera_from_world @ object_poses[instance_name]
        )
        visible.append(
            {
                "object_name": object_name,
                "instance_name": instance_name,
                "object_id": int(entry["category_id"]),
                "instance_id": instance_id,
                "bbox_xywh": bbox,
                "visible_area_px": int(mask.sum()),
                "segmentation_rle": coco_rle(mask),
                "world_from_object": object_poses[instance_name].tolist(),
                "camera_from_object": camera_from_object.tolist(),
            }
        )

    mask_path = masks_dir / f"{frame_id:06d}.png"
    cv2.imwrite(str(mask_path), np.asarray(segmap, dtype=np.uint16))
    annotations["frames"].append(
        {
            "frame_id": frame_id,
            "image_path": frame["image_path"],
            "instance_mask_path": str(mask_path),
            "world_from_camera": world_from_camera.tolist(),
            "camera_from_world": camera_from_world.tolist(),
            "annotations": visible,
        }
    )

(output / "annotations.json").write_text(
    json.dumps(annotations, indent=2) + "\n"
)
print(
    f"Saved {len(manifest['objects'])} objects and "
    f"{len(manifest['frames'])} annotated frames to {output}"
)
