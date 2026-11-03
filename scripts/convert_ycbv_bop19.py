#!/usr/bin/env python3
"""Convert the BOP YCB-Video test_bop19 scenes into Label Factory datasets.

Each scene becomes data/comparison_NN/ (NN counts scenes in ascending BOP order):

    images/000000.png ...            RGB, renumbered contiguously
    depth_maps/depth_000000.png ...  uint16 millimetres (BOP stores 0.1 mm units)
    camera.json                      intrinsics (constant within a YCB-V scene)
    objects/stl/obj_0000NN.stl       CAD models of the objects present, in mm
    objects/textured/obj_0000NN.*    optional textured model (.obj/.mtl/.png), same
                                     coordinates; the pipeline uses it automatically
    bop_gt/                          ground truth incl. visible masks (see source.json)

A config scripts/configs/comparison_NN.yaml is written next to it.
Run with the sam6d conda env (needs cv2, trimesh, pyyaml).
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh

from textured_models import export_textured_obj
import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bop-root", type=Path, default=ROOT / "BOP/ycbv",
                   help="Folder with ycbv_test_bop19/ and ycbv_models/")
    p.add_argument("--project-root", type=Path, default=ROOT,
                   help="Label Factory checkout to write data/ and configs into")
    p.add_argument("--scenes", type=int, nargs="*", default=None,
                   help="BOP scene ids to convert (default: all, e.g. 48 49 ...)")
    p.add_argument("--model-set", default="models", choices=["models", "models_fine"],
                   help="models = simplified meshes, models_fine = full meshes")
    p.add_argument("--no-textures", action="store_true",
                   help="Do not export the textured models (STL only)")
    p.add_argument("--textures-only", action="store_true",
                   help="Only add objects/textured/ to datasets that already exist "
                        "(keeps images, reconstruction and results untouched)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def add_textures(a, name):
    """Export the textured models for the objects already present in a dataset."""
    out = a.project_root / "data" / name
    if not (out / "objects/stl").is_dir():
        print(f"skip {name}: dataset not found")
        return None
    model_dir = a.bop_root / "ycbv_models" / a.model_set
    added = []
    for stl in sorted((out / "objects/stl").glob("obj_*.stl")):
        oid = int(stl.stem.split("_")[1])
        texture = model_dir / f"obj_{oid:06d}.png"
        mesh = trimesh.load(model_dir / f"obj_{oid:06d}.ply", process=False)
        if texture.is_file() and mesh.visual.kind == "texture":
            export_textured_obj(mesh.vertices, mesh.faces, mesh.visual.uv, mesh.vertex_normals,
                                texture, out / "objects/textured", stl.stem)
            added.append(oid)
    return added


def convert_scene(a, scene, name, base_config):
    src = a.bop_root / "ycbv_test_bop19/test" / f"{scene:06d}"
    out = a.project_root / "data" / name
    if out.exists():
        if not a.overwrite:
            print(f"skip {name}: exists (use --overwrite)")
            return None
        shutil.rmtree(out)

    cams = json.load(open(src / "scene_camera.json"))
    gts = json.load(open(src / "scene_gt.json"))
    infos = json.load(open(src / "scene_gt_info.json"))
    frame_ids = sorted(cams, key=int)

    ks = np.array([cams[f]["cam_K"] for f in frame_ids])
    if np.ptp(ks, axis=0).max() > 1e-6:
        raise SystemExit(f"Scene {scene}: intrinsics vary between frames")
    K = ks[0].reshape(3, 3)

    for sub in ("images", "depth_maps", "objects/stl", "bop_gt"):
        (out / sub).mkdir(parents=True)

    poses, frame_map = {}, {}
    for n, f in enumerate(frame_ids):
        file = f"{int(f):06d}.png"
        rgb = cv2.imread(str(src / "rgb" / file), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(src / "depth" / file), cv2.IMREAD_UNCHANGED)
        if rgb is None or raw is None:
            raise SystemExit(f"Scene {scene}: missing RGB/depth for frame {f}")
        h, w = rgb.shape[:2]
        mm = np.clip(np.round(raw.astype(np.float32) * cams[f]["depth_scale"]), 0, 65535)
        cv2.imwrite(str(out / "images" / f"{n:06d}.png"), rgb)
        cv2.imwrite(str(out / "depth_maps" / f"depth_{n:06d}.png"), mm.astype(np.uint16))
        R = np.array(cams[f]["cam_R_w2c"]).reshape(3, 3)
        t = np.array(cams[f]["cam_t_w2c"])
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
        poses[f"{n:06d}"] = {"camera_to_world_mm": c2w.tolist()}
        frame_map[f"{n:06d}"] = int(f)

    json.dump({
        "cam_K": K.flatten().tolist(),
        "distortion": [0.0] * 5,
        "distortion_names": ["k1", "k2", "p1", "p2", "k3"],
        "image_width": w, "image_height": h,
        "depth_scale": 1.0, "depth_unit": "millimetre",
        "source": f"BOP YCB-Video test_bop19 scene {scene:06d}",
        "depth_alignment": "depth_to_color",
        "images_undistorted": True,
    }, open(out / "camera.json", "w"), indent=2)

    # Ground truth re-indexed to the new frame ids.
    for fname, data in (("scene_camera.json", cams), ("scene_gt.json", gts),
                        ("scene_gt_info.json", infos)):
        json.dump({str(n): data[f] for n, f in enumerate(frame_ids)},
                  open(out / "bop_gt" / fname, "w"))
    # Visible-instance masks (one per ground-truth instance) for the IoU evaluation.
    mask_dir = out / "bop_gt/mask_visib"
    mask_dir.mkdir()
    for n, f in enumerate(frame_ids):
        for index in range(len(gts[f])):
            mask = src / "mask_visib" / f"{int(f):06d}_{index:06d}.png"
            if mask.is_file():
                shutil.copy2(mask, mask_dir / f"{n:06d}_{index:06d}.png")
    json.dump(frame_map, open(out / "bop_gt/frame_map.json", "w"), indent=1)
    json.dump({"units": "millimetre", "frames": poses},
              open(out / "bop_gt/gt_camera_poses.json", "w"))
    json.dump({"dataset": "ycbv", "split": "test_bop19", "scene_id": scene},
              open(out / "bop_gt/source.json", "w"), indent=1)

    # BOP YCB-V models are already in millimetres (matches --model-scale 0.001).
    model_dir = a.bop_root / "ycbv_models" / a.model_set
    info = json.load(open(model_dir / "models_info.json"))
    present = sorted({o["obj_id"] for f in frame_ids for o in gts[f]})
    for oid in present:
        mesh = trimesh.load(model_dir / f"obj_{oid:06d}.ply", process=False)
        mesh.export(out / "objects/stl" / f"obj_{oid:06d}.stl")
        texture = model_dir / f"obj_{oid:06d}.png"
        if not a.no_textures and texture.is_file() and mesh.visual.kind == "texture":
            export_textured_obj(mesh.vertices, mesh.faces, mesh.visual.uv, mesh.vertex_normals,
                                texture, out / "objects/textured", f"obj_{oid:06d}")
    json.dump({str(o): info[str(o)] for o in present},
              open(out / "bop_gt/models_info.json", "w"), indent=1)

    cfg = dict(base_config)
    cfg["dataset_id"] = name
    cam = dict(cfg.get("camera", {}))
    cam.update(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]))
    for k in ("k1", "k2", "p1", "p2", "k3"):
        if k in cam:
            cam[k] = 0.0
    cfg["camera"] = cam
    cfg_path = a.project_root / "scripts/configs" / f"{name}.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    return len(frame_ids), f"{w}x{h}", present


def main():
    a = parse_args()
    a.project_root = a.project_root.resolve()
    test_dir = a.bop_root / "ycbv_test_bop19/test"
    if not test_dir.is_dir():
        raise SystemExit(
            f"BOP YCB-V data not found: {test_dir}\n"
            "Pass the folder that contains ycbv_test_bop19/ and ycbv_models/ with "
            "--bop-root (default: <project-root>/BOP/ycbv)."
        )
    all_scenes = sorted(int(p.name) for p in test_dir.iterdir() if p.is_dir())
    if not all_scenes:
        raise SystemExit(f"No scenes in {test_dir}")
    # comparison_NN is tied to the scene's position in the full list, so a
    # partial run never renames or collides with another scene's dataset.
    numbering = {s: f"comparison_{i:02d}" for i, s in enumerate(all_scenes, 1)}
    wanted = a.scenes or all_scenes
    unknown = [s for s in wanted if s not in numbering]
    if unknown:
        raise SystemExit(f"Unknown scenes {unknown}; available: {all_scenes}")

    base = yaml.safe_load(open(a.project_root / "scripts/configs/default.yaml"))
    base["path"] = "." if str(base.get("path", ".")) == "." else str(a.project_root)
    for s in wanted:
        if a.textures_only:
            added = add_textures(a, numbering[s])
            if added is not None:
                print(f"{numbering[s]}: textured models for objects {added}")
            continue
        result = convert_scene(a, s, numbering[s], base)
        if result:
            n, size, objs = result
            print(f"{numbering[s]}  <- scene {s:06d}: {n} frames, {size}, objects {objs}")


if __name__ == "__main__":
    main()
