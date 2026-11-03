#!/usr/bin/env python3
"""Evaluate Label Factory results on the YCB-Video BOP19 comparison datasets.

Run after the pipeline has produced object poses (and, for IoU, annotation masks)
in data/comparison_NN. For every dataset the script

  1. converts the world-frame object poses into camera-frame poses per frame:
         cam_from_obj = inverse(world_from_camera) @ world_from_object
     (directly comparable to the BOP ground truth, no world alignment needed);
  2. scores them on the BOP19 targets (ground-truth instances >= 10 % visible),
     counting objects without a pose as failures;
  3. compares your rendered annotation masks (sam6d_scene/masks) with the BOP
     visible-instance masks;
  4. writes BOP-format result CSVs for the official bop_toolkit and a table that
     merges the results with literature numbers from a CSV you fill in.

Metrics
  ADD-S AUC, ADD(-S) AUC   PoseCNN protocol, AUC of the accuracy curve up to 10 cm
  ADD(-S) < 0.1d           share of targets with ADD(-S) below 10 % of the diameter
  AR_MSSD, AR_MSPD         BOP recall (mean over the BOP correctness thresholds);
                           AR_VSD needs depth rendering, use the bop_toolkit CSV for it
  mask IoU                 mean IoU of annotation mask vs BOP visible mask
  median translation error, and how many targets received a pose
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
# PoseCNN convention for ADD(-S): bowl, wood block, large clamp, extra large clamp,
# foam brick (BOP ids 13, 16, 19, 20, 21) are treated as symmetric.
SYMMETRIC_IDS = {13, 16, 19, 20, 21}
YCBV_NAMES = {
    1: "master_chef_can", 2: "cracker_box", 3: "sugar_box", 4: "tomato_soup_can",
    5: "mustard_bottle", 6: "tuna_fish_can", 7: "pudding_box", 8: "gelatin_box",
    9: "potted_meat_can", 10: "banana", 11: "pitcher_base", 12: "bleach_cleanser",
    13: "bowl", 14: "mug", 15: "power_drill", 16: "wood_block", 17: "scissors",
    18: "large_marker", 19: "large_clamp", 20: "extra_large_clamp", 21: "foam_brick",
}
MSSD_THRESHOLDS = np.arange(0.05, 0.5001, 0.05)   # fractions of the object diameter
MSPD_THRESHOLDS = np.arange(5.0, 50.001, 5.0)     # pixels at 640 px width


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Dataset names (default: every data/comparison_NN)")
    p.add_argument("--bop-root", type=Path, default=None,
                   help="BOP/ycbv folder (needed for IoU only if a dataset has no "
                        "bop_gt/mask_visib copy)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <project-root>/evaluation/ycbv")
    p.add_argument("--method-name", default="Label Factory (ours)")
    p.add_argument("--literature", type=Path, default=None,
                   help="CSV of literature results (default: <output-dir>/literature_ycbv.csv)")
    p.add_argument("--points", type=int, default=5000, help="Model points per object")
    p.add_argument("--auc-max-m", type=float, default=0.1)
    return p.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def se3(rotation, translation):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rotation, translation
    return m


_cache = {}


def model_points(dataset, obj_id, count):
    key = (dataset, obj_id)
    if key not in _cache:
        mesh = trimesh.load(dataset / "objects/stl" / f"obj_{obj_id:06d}.stl", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) / 1000.0  # mm -> m
        if len(vertices) > count:
            vertices = vertices[np.random.default_rng(0).choice(len(vertices), count, False)]
        _cache[key] = vertices
    return _cache[key]


def symmetry_transforms(model_info, step=0.01):
    """Global symmetries (4x4, translation in metres), as in the BOP toolkit."""
    discrete = [np.eye(4)]
    if "symmetries_discrete" in model_info:
        discrete = [np.array(m).reshape(4, 4) for m in model_info["symmetries_discrete"]]
        discrete = [np.eye(4)] + discrete
        for m in discrete:
            m[:3, 3] /= 1000.0
    continuous = [np.eye(4)]
    for sym in model_info.get("symmetries_continuous", []):
        axis = np.array(sym["axis"], dtype=float)
        axis /= np.linalg.norm(axis)
        offset = np.array(sym["offset"], dtype=float) / 1000.0
        for angle in np.arange(step, 2 * np.pi, step):
            rotation = cv2.Rodrigues(axis * angle)[0]
            continuous.append(se3(rotation, offset - rotation @ offset))
    return [d @ c for d in discrete for c in continuous]


def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def add_error(points, est, gt):
    return float(np.linalg.norm(transform(points, est) - transform(points, gt), axis=1).mean())


def adds_error(points, est, gt):
    distances, _ = cKDTree(transform(points, gt)).query(transform(points, est))
    return float(distances.mean())


def mssd_error(points, est, gt, symmetries):
    est_points = transform(points, est)
    return min(float(np.linalg.norm(est_points - transform(points, gt @ s), axis=1).max())
               for s in symmetries)


def project(points, K):
    z = np.maximum(points[:, 2:3], 1e-9)
    return (points @ K.T)[:, :2] / z


def mspd_error(points, est, gt, symmetries, K):
    est_px = project(transform(points, est), K)
    return min(float(np.linalg.norm(est_px - project(transform(points, gt @ s), K), axis=1).max())
               for s in symmetries)


def auc(errors, maximum, step=1e-3):
    """Area under the accuracy-threshold curve on [0, maximum], normalised to 1."""
    if not len(errors):
        return float("nan")
    errors = np.sort(np.asarray(errors, dtype=np.float64))  # inf = missing pose
    thresholds = np.arange(0.0, maximum + step, step)
    accuracy = np.searchsorted(errors, thresholds, side="right") / len(errors)
    return float(np.trapz(accuracy, thresholds) / maximum)


def recall(errors, thresholds):
    errors = np.asarray(errors, dtype=np.float64)
    if not len(errors):
        return float("nan")
    return float(np.mean([(errors < t).mean() for t in thresholds]))


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def load_predictions(dataset):
    cams = {p["frame_id"]: np.array(p["world_from_camera"]) for p in load_json(
        dataset / "reconstruction/metric_alignment/camera_poses_metric.json")["poses"]}
    predictions, low_support = {}, set()
    for obj_dir in sorted((dataset / "sam6d_sequences").glob("obj_*")):
        pose_file = obj_dir / f"{obj_dir.name}_world_pose.json"
        if not pose_file.is_file():
            continue
        payload = load_json(pose_file)
        obj_id = int(obj_dir.name.split("_")[1])
        if payload.get("low_support"):
            low_support.add(obj_id)
        predictions[obj_id] = [
            (float(inst["number_of_position_inliers"]), np.array(inst["world_from_object"]))
            for inst in payload["instances"]
        ]
    return cams, predictions, low_support


def decode_rle(segmentation):
    """COCO RLE (compressed or uncompressed) -> boolean mask."""
    from pycocotools import mask as mask_utils
    height, width = segmentation["size"]
    counts = segmentation["counts"]
    if isinstance(counts, list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = {"size": [height, width],
               "counts": counts.encode() if isinstance(counts, str) else counts}
    return mask_utils.decode(rle).astype(bool)


def load_single_frame(dataset):
    """Raw SAM-6D poses, one per (object, frame), without any multi-frame fusion.

    Returns ({(obj_id, frame): (score, cam_from_obj, segmentation)}, processed_frames). A frame counts
    as processed when SAM-6D was run on it, even if it found nothing (a failure).
    """
    poses, processed = {}, set()
    for path in sorted((dataset / "experimental_multi_instance/pem").glob(
            "obj_*/obj_*_poses_per_frame.json")):
        obj_id = int(path.parent.name.split("_")[1])
        for frame in load_json(path)["frames"]:
            fid = int(frame["frame_id"])
            processed.add(fid)
            candidates = [c for c in (frame.get("pose_candidates") or [])
                          if "R" in c and "t" in c]
            if not candidates and frame.get("best_pose"):
                candidates = [frame["best_pose"]]
            if candidates:
                best = max(candidates, key=lambda c: float(c.get("score", 0.0)))
                poses[(obj_id, fid)] = (float(best.get("score", 0.0)), se3(
                    np.array(best["R"], dtype=np.float64).reshape(3, 3),
                    np.array(best["t"], dtype=np.float64).reshape(3) / 1000.0),
                    best.get("segmentation"))
    return poses, processed


def annotation_lookup(dataset):
    """Return (masks dir, {mask value: obj_id}, primary mask values) or (None, {}, set())."""
    manifest = dataset / "sam6d_scene/scene_manifest.json"
    masks = dataset / "sam6d_scene/masks"
    if not (manifest.is_file() and masks.is_dir()):
        return None, {}, set()
    objects = load_json(manifest)["objects"]
    ids = {int(o["object_id"]): int(o["object_name"].split("_")[1]) for o in objects}
    # <object>_001 is the best-supported instance; extra instances are weaker clusters.
    primary = {int(o["object_id"]) for o in objects if o["instance_name"].endswith("_001")}
    return masks, ids, primary


def gt_visible_mask(dataset, bop_root, scene_id, frame, original_frame, index):
    local = dataset / "bop_gt/mask_visib" / f"{frame:06d}_{index:06d}.png"
    path = local if local.is_file() else (
        bop_root / "ycbv_test_bop19/test" / f"{scene_id:06d}/mask_visib"
        / f"{original_frame:06d}_{index:06d}.png" if bop_root else None)
    if path is None or not path.is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) > 0


def evaluate_dataset(dataset, args, exclude_low_support, single=None, frame_filter=None):
    """single: raw per-frame SAM-6D poses to score instead of the fused world poses;
    frame_filter: only score targets in these frames."""
    gt = load_json(dataset / "bop_gt/scene_gt.json")
    info = load_json(dataset / "bop_gt/scene_gt_info.json")
    models_info = load_json(dataset / "bop_gt/models_info.json")
    frame_map = load_json(dataset / "bop_gt/frame_map.json")
    scene_id = load_json(dataset / "bop_gt/source.json")["scene_id"]
    K = np.array(load_json(dataset / "camera.json")["cam_K"]).reshape(3, 3)
    cams, predictions, low_support = load_predictions(dataset)
    masks_dir, mask_ids, primary_ids = annotation_lookup(dataset)
    symmetries = {}

    records, csv_rows = [], []
    for key in sorted(gt, key=int):
        frame = int(key)
        if frame_filter is not None and frame not in frame_filter:
            continue
        pred_mask = None
        if masks_dir is not None and single is None and (masks_dir / f"{frame:06d}.png").is_file():
            pred_mask = cv2.imread(str(masks_dir / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        for index, (gt_obj, gt_info) in enumerate(zip(gt[key], info[key])):
            if gt_info["visib_fract"] < 0.1:  # BOP19 target rule
                continue
            obj_id = gt_obj["obj_id"]
            gt_pose = se3(np.array(gt_obj["cam_R_m2c"]).reshape(3, 3),
                          np.array(gt_obj["cam_t_m2c"]) / 1000.0)
            inf = np.inf
            record = {"scene": scene_id, "frame": frame, "obj_id": obj_id, "found": False,
                      "low_support": obj_id in low_support, "adds": inf, "add": inf,
                      "add_or_adds": inf, "mssd": inf, "mspd": inf, "t_err": inf,
                      "diameter": models_info[str(obj_id)]["diameter"] / 1000.0, "iou": None,
                      "iou_primary": None}
            use = not (exclude_low_support and obj_id in low_support)
            candidates = predictions.get(obj_id, []) if use else []
            est, single_segmentation = None, None
            if single is not None:
                if (obj_id, frame) in single:
                    score, est, single_segmentation = single[(obj_id, frame)]
            elif candidates and frame in cams:
                score, world_from_obj = max(candidates, key=lambda c: c[0])
                est = np.linalg.inv(cams[frame]) @ world_from_obj
            if est is not None:
                pts = model_points(dataset, obj_id, args.points)
                if obj_id not in symmetries:
                    symmetries[obj_id] = symmetry_transforms(models_info[str(obj_id)])
                adds, add = adds_error(pts, est, gt_pose), add_error(pts, est, gt_pose)
                record.update(
                    found=True, adds=adds, add=add,
                    add_or_adds=adds if obj_id in SYMMETRIC_IDS else add,
                    mssd=mssd_error(pts, est, gt_pose, symmetries[obj_id]),
                    mspd=mspd_error(pts, est, gt_pose, symmetries[obj_id], K),
                    t_err=float(np.linalg.norm(est[:3, 3] - gt_pose[:3, 3])))
                csv_rows.append([scene_id, frame_map[f"{frame:06d}"], obj_id, score,
                                 " ".join(f"{v:.8f}" for v in est[:3, :3].flatten()),
                                 " ".join(f"{v * 1000:.6f}" for v in est[:3, 3]), -1])
            if single is not None:
                # Raw SAM-6D instance mask of the selected candidate vs BOP visible mask;
                # a frame without a detection scores IoU 0.
                gt_mask = gt_visible_mask(dataset, args.bop_root, scene_id, frame,
                                          frame_map[f"{frame:06d}"], index)
                if gt_mask is not None:
                    predicted = (decode_rle(single_segmentation)
                                 if single_segmentation else np.zeros_like(gt_mask))
                    record["iou"] = mask_iou(predicted, gt_mask)
            if pred_mask is not None:
                gt_mask = gt_visible_mask(dataset, args.bop_root, scene_id, frame,
                                          frame_map[f"{frame:06d}"], index)
                if gt_mask is not None:
                    values = [v for v, o in mask_ids.items()
                              if o == obj_id and use]
                    predicted = np.isin(pred_mask, values) if values else np.zeros_like(gt_mask)
                    record["iou"] = mask_iou(predicted, gt_mask)
                    primary = [v for v in values if v in primary_ids]
                    record["iou_primary"] = mask_iou(
                        np.isin(pred_mask, primary) if primary else np.zeros_like(gt_mask),
                        gt_mask)
            records.append(record)
    return records, csv_rows


def summarise(records, maximum):
    def block(rows):
        found = [r for r in rows if r["found"]]
        ious = [r["iou"] for r in rows if r["iou"] is not None and not np.isnan(r["iou"])]
        ious_primary = [r["iou_primary"] for r in rows if r.get("iou_primary") is not None
                        and not np.isnan(r["iou_primary"])]
        rel = [r["add_or_adds"] / r["diameter"] for r in rows]
        return {
            "targets": len(rows),
            "found": len(found),
            "adds_auc": auc([r["adds"] for r in rows], maximum) * 100,
            "add_or_adds_auc": auc([r["add_or_adds"] for r in rows], maximum) * 100,
            "add_or_adds_below_0.1d": float(np.mean([e < 0.1 for e in rel])) * 100,
            "ar_mssd": recall([r["mssd"] / r["diameter"] for r in rows], MSSD_THRESHOLDS) * 100,
            "ar_mspd": recall([r["mspd"] for r in rows], MSPD_THRESHOLDS) * 100,
            "mean_mask_iou": float(np.mean(ious)) * 100 if ious else float("nan"),
            "mean_mask_iou_primary_instance": float(np.mean(ious_primary)) * 100
            if ious_primary else float("nan"),
            "mask_iou_targets": len(ious),
            "median_mask_iou": float(np.median(ious)) * 100 if ious else float("nan"),
            "mask_iou_ge_0.5": float(np.mean([v >= 0.5 for v in ious])) * 100
            if ious else float("nan"),
            "mask_iou_ge_0.75": float(np.mean([v >= 0.75 for v in ious])) * 100
            if ious else float("nan"),
            "median_adds_error_mm": float(np.median(
                [r["adds"] * 1000 for r in found])) if found else float("nan"),
            "median_translation_error_mm": float(np.median(
                [r["t_err"] * 1000 for r in found])) if found else float("nan"),
        }
    per_object = {o: block([r for r in records if r["obj_id"] == o])
                  for o in sorted({r["obj_id"] for r in records})}
    keys = ("adds_auc", "add_or_adds_auc", "add_or_adds_below_0.1d", "ar_mssd", "ar_mspd",
            "mean_mask_iou", "mean_mask_iou_primary_instance", "median_mask_iou",
            "mask_iou_ge_0.5", "mask_iou_ge_0.75", "median_translation_error_mm",
            "median_adds_error_mm")
    return {"pooled": block(records), "per_object": per_object,
            "mean_over_objects": {k: float(np.nanmean([v[k] for v in per_object.values()]))
                                  for k in keys}}


ACCURACY_COLUMNS = [  # key, CSV/markdown header, LaTeX header (None = not in the paper table)
    ("targets", "Targets", "Targets"),
    ("mean_mask_iou", "Mask IoU (all instances)", None),
    ("mean_mask_iou_primary_instance", "Mask IoU", "IoU"),
    ("median_mask_iou", "Median IoU", None),
    ("mask_iou_ge_0.5", "IoU >= 0.5 (%)", None),
    ("mask_iou_ge_0.75", "IoU >= 0.75 (%)", None),
    ("adds_auc", "ADD-S AUC", "ADD-S AUC"),
    ("add_or_adds_auc", "ADD(-S) AUC", "ADD(-S) AUC"),
    ("ar_mssd", "AR MSSD", "AR$_{MSSD}$"),
    ("ar_mspd", "AR MSPD", "AR$_{MSPD}$"),
    ("median_translation_error_mm", "Median transl. error (mm)", "Transl. err. (mm)"),
]


SEQUENCE_COLUMNS = [  # key, CSV/markdown header, LaTeX header (None = CSV/markdown only)
    ("objects", "Objects", "Obj."),
    ("targets", "Targets", "Targets"),
    ("mean_mask_iou_primary_instance", "Mask IoU", "IoU"),
    ("mean_mask_iou", "Mask IoU (all instances)", None),
    ("median_mask_iou", "Median IoU", None),
    ("mask_iou_ge_0.5", "IoU >= 0.5 (%)", None),
    ("mask_iou_ge_0.75", "IoU >= 0.75 (%)", None),
    ("adds_auc", "ADD-S AUC", "ADD-S AUC"),
    ("add_or_adds_auc", "ADD(-S) AUC", "ADD(-S) AUC"),
    ("ar_mssd", "AR MSSD", "AR$_{MSSD}$"),
    ("ar_mspd", "AR MSPD", "AR$_{MSPD}$"),
    ("median_translation_error_mm", "Median transl. error (mm)", "Transl. err. (mm)"),
]


def write_sequence_table(out, summary, per_scene, scene_dataset):
    """One table with a row per comparison sequence and overall rows."""
    rows = [(scene_dataset.get(sc, str(sc)), sc, v) for sc, v in per_scene.items()]
    rows.sort(key=lambda r: r[0])
    mean = {k: float(np.nanmean([v[k] for _, _, v in rows])) for k, _, _ in SEQUENCE_COLUMNS
            if k not in ("objects", "targets")}
    mean["objects"] = float(np.mean([v["objects"] for _, _, v in rows]))
    mean["targets"] = float(np.mean([v["targets"] for _, _, v in rows]))
    pooled = dict(summary["pooled"])
    pooled["objects"] = len(summary["per_object"])

    def cell(v, key):
        value = v.get(key)
        if key in ("targets", "objects"):
            return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"
        return fmt(value, 1)

    def line(name, scene, v, columns):
        return [name, str(scene)] + [cell(v, k) for k, *_ in columns]
    header = ["Sequence", "BOP scene"] + [h for _, h, _ in SEQUENCE_COLUMNS]
    table = [line(n, sc, v, SEQUENCE_COLUMNS) for n, sc, v in rows]
    table.append(line("Mean over sequences", "", mean, SEQUENCE_COLUMNS))
    table.append(line("All sequences (pooled)", "", pooled, SEQUENCE_COLUMNS))
    with open(out / "annotation_accuracy_per_dataset.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in table]
    (out / "annotation_accuracy_per_dataset.md").write_text("\n".join(md) + "\n")

    paper = [(k, t) for k, _, t in SEQUENCE_COLUMNS if t]
    paper_idx = [i for i, (_, _, t) in enumerate(SEQUENCE_COLUMNS) if t]
    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{Annotation accuracy of the full pipeline for each of the 12 YCB-Video "
           "BOP19 test sequences. IoU: mean mask IoU against the visible-instance masks of the "
           "best-supported instance of each object; ADD-S and ADD(-S) AUC (10\\,cm), "
           "AR$_{MSSD}$ and AR$_{MSPD}$ in \\%; objects without an annotation count as "
           "failures. The last row pools all targets.}",
           "\\label{tab:annotation_accuracy_sequences}",
           "\\begin{tabular}{l" + "r" * 2 + "c" * (len(paper) - 2) + "}", "\\hline",
           "Sequence & " + " & ".join(t for _, t in paper) + " \\\\", "\\hline"]
    for r in table:
        cells = [r[0].replace("_", "\\_")] + [r[2 + i] for i in paper_idx]
        if r[0] == "Mean over sequences":
            tex.append("\\hline")
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (out / "annotation_accuracy_per_dataset.tex").write_text("\n".join(tex) + "\n")


def write_annotation_accuracy(out, summary, per_scene, scene_dataset):
    """Annotation-accuracy table of the full pipeline: object-wise, mean and pooled rows."""
    rows = [(YCBV_NAMES.get(o, str(o)).replace("_", " "), v)
            for o, v in summary["per_object"].items()]
    rows.append(("Mean over objects", {**summary["mean_over_objects"],
                                       "targets": summary["pooled"]["targets"]}))
    rows.append(("Pooled over targets", summary["pooled"]))

    def cell(v, key):
        value = v.get(key)
        if key == "targets":
            return str(int(value))
        return fmt(value, 1)
    header = ["Object"] + [h for _, h, _ in ACCURACY_COLUMNS]
    table = [[name] + [cell(v, k) for k, _, _ in ACCURACY_COLUMNS] for name, v in rows]
    with open(out / "annotation_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in table]
    (out / "annotation_accuracy.md").write_text("\n".join(md) + "\n")

    paper = [(k, t) for k, _, t in ACCURACY_COLUMNS if t]
    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{Annotation accuracy of the full pipeline on the YCB-Video BOP19 test "
           f"scenes ({len(per_scene)} sequences). IoU: mean mask IoU against the visible-instance "
           "masks of the best-supported instance of each object; ADD-S and ADD(-S) AUC "
           "(10\\,cm), AR$_{MSSD}$ and AR$_{MSPD}$ in \\%; objects without an annotation "
           "count as failures.}",
           "\\label{tab:annotation_accuracy}",
           "\\begin{tabular}{l" + "r" + "c" * (len(paper) - 1) + "}", "\\hline",
           "Object & " + " & ".join(t for _, t in paper) + " \\\\", "\\hline"]
    for name, v in rows:
        if name == "Mean over objects":
            tex.append("\\hline")
        tex.append(name + " & " + " & ".join(cell(v, k) for k, _ in paper) + " \\\\")
    tex += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (out / "annotation_accuracy.tex").write_text("\n".join(tex) + "\n")

    write_sequence_table(out, summary, per_scene, scene_dataset)


def fmt(x, digits=1):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def main():
    args = parse_args()
    project = args.project_root.resolve()
    out = (args.output_dir or project / "evaluation/ycbv").resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = args.datasets or sorted(p.name for p in (project / "data").glob("comparison_*"))
    datasets = [project / "data" / n for n in names]

    def processed(ds):
        sequences = ds / "sam6d_sequences"
        # "Processed" = pose fitting has run; otherwise it is not-yet-run, not failed.
        return (ds / "reconstruction/metric_alignment/camera_poses_metric.json").is_file() \
            and sequences.is_dir() and (
                (sequences / "ransac_run_summary.json").is_file()
                or any(sequences.glob("obj_*/obj_*_world_pose.json")))

    todo = [d for d in datasets if processed(d)]
    skipped = [d.name for d in datasets if not processed(d)]
    if not todo:
        raise SystemExit("No evaluable datasets (need reconstruction and pose fitting outputs).")

    results = {}
    for variant, exclude in (("all", False), ("no_low_support", True)):
        records, rows = [], []
        for ds in todo:
            r, c = evaluate_dataset(ds, args, exclude)
            records += r
            rows += c
        results[variant] = summarise(records, args.auc_max_m)
        with open(out / f"bop_results_labelfactory_{variant}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
            w.writerows(rows)
        if variant == "all":
            with open(out / "per_target.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["dataset_scene", "frame", "obj_id", "found", "low_support",
                            "adds_m", "add_m", "mssd_m", "mspd_px", "translation_error_m",
                            "mask_iou", "mask_iou_primary_instance"])
                for r in records:
                    w.writerow([r["scene"], r["frame"], r["obj_id"], int(r["found"]),
                                int(r["low_support"]), r["adds"], r["add"], r["mssd"],
                                r["mspd"], r["t_err"],
                                "" if r["iou"] is None else r["iou"],
                                "" if r.get("iou_primary") is None else r["iou_primary"]])
            per_scene = {sc: summarise([r for r in records if r["scene"] == sc],
                                       args.auc_max_m)["pooled"]
                         for sc in sorted({r["scene"] for r in records})}
            for sc, block in per_scene.items():
                block["objects"] = len({r["obj_id"] for r in records if r["scene"] == sc})
            scene_dataset = {load_json(d / "bop_gt/source.json")["scene_id"]: d.name
                             for d in todo}
    results["datasets_evaluated"] = [d.name for d in todo]
    results["datasets_skipped_no_pose_outputs"] = skipped
    json.dump(results, open(out / "summary.json", "w"), indent=2)
    write_annotation_accuracy(out, results["all"], per_scene, scene_dataset)

    # Single-frame SAM-6D vs the fused result, on exactly the same targets: only frames
    # where SAM-6D was actually run, so datasets processed on fewer frames are fair.
    single_records, fused_records, per_dataset = [], [], []
    for ds in todo:
        single, processed = load_single_frame(ds)
        if not processed:
            continue
        r_single, _ = evaluate_dataset(ds, args, False, single=single, frame_filter=processed)
        r_fused, _ = evaluate_dataset(ds, args, False, frame_filter=processed)
        single_records += r_single
        fused_records += r_fused
        a_, b_ = summarise(r_single, args.auc_max_m)["pooled"], summarise(r_fused, args.auc_max_m)["pooled"]
        per_dataset.append([ds.name, len(processed), a_["targets"], a_["found"], b_["found"],
                            fmt(a_["adds_auc"]), fmt(b_["adds_auc"]),
                            fmt(a_["ar_mssd"]), fmt(b_["ar_mssd"]),
                            fmt(a_["ar_mspd"]), fmt(b_["ar_mspd"]),
                            fmt(a_["mean_mask_iou"]), fmt(b_["mean_mask_iou"])])
    if single_records:
        results["single_frame_comparison"] = {
            "sam6d_single_frame": summarise(single_records, args.auc_max_m),
            "label_factory_same_targets": summarise(fused_records, args.auc_max_m),
            "note": "Targets restricted to frames on which SAM-6D was run; frames where it "
                    "returned nothing count as failures for the single-frame result.",
        }
        json.dump(results, open(out / "summary.json", "w"), indent=2)
        with open(out / "single_frame_comparison_per_dataset.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "frames_processed", "targets", "found_single", "found_fused",
                        "adds_auc_single", "adds_auc_fused", "ar_mssd_single",
                        "ar_mssd_fused", "ar_mspd_single", "ar_mspd_fused",
                        "mask_iou_single", "mask_iou_fused"])
            w.writerows(per_dataset)
        with open(out / "single_frame_comparison_per_object.csv", "w", newline="") as f:
            w = csv.writer(f)
            sf = results["single_frame_comparison"]["sam6d_single_frame"]["per_object"]
            fu = results["single_frame_comparison"]["label_factory_same_targets"]["per_object"]
            w.writerow(["obj_id", "name", "targets", "found_single", "found_fused",
                        "adds_auc_single", "adds_auc_fused", "add_or_adds_auc_single",
                        "add_or_adds_auc_fused", "ar_mssd_single", "ar_mssd_fused",
                        "ar_mspd_single", "ar_mspd_fused", "mask_iou_single", "mask_iou_fused"])
            for obj_id in sf:
                a_, b_ = sf[obj_id], fu[obj_id]
                w.writerow([obj_id, YCBV_NAMES.get(obj_id, ""), a_["targets"], a_["found"],
                            b_["found"], fmt(a_["adds_auc"]), fmt(b_["adds_auc"]),
                            fmt(a_["add_or_adds_auc"]), fmt(b_["add_or_adds_auc"]),
                            fmt(a_["ar_mssd"]), fmt(b_["ar_mssd"]),
                            fmt(a_["ar_mspd"]), fmt(b_["ar_mspd"]),
                            fmt(a_["mean_mask_iou"]), fmt(b_["mean_mask_iou"])])

    with open(out / "per_object.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obj_id", "name", "targets", "found", "adds_auc", "add_or_adds_auc",
                    "add_or_adds_below_0.1d", "ar_mssd", "ar_mspd", "mean_mask_iou",
                    "median_translation_error_mm"])
        for obj_id, s in results["all"]["per_object"].items():
            w.writerow([obj_id, YCBV_NAMES.get(obj_id, ""), s["targets"], s["found"],
                        fmt(s["adds_auc"]), fmt(s["add_or_adds_auc"]),
                        fmt(s["add_or_adds_below_0.1d"]), fmt(s["ar_mssd"]),
                        fmt(s["ar_mspd"]), fmt(s["mean_mask_iou"]),
                        fmt(s["median_translation_error_mm"])])

    lit_path = args.literature or out / "literature_ycbv.csv"
    if not lit_path.is_file():
        with open(lit_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "adds_auc", "add_or_adds_auc", "bop_ar", "mask_iou",
                        "input", "uses_cad", "needs_training", "source"])
            for m in ("PoseCNN", "DenseFusion", "PVN3D", "FFB6D", "GDR-Net",
                      "MegaPose", "FoundationPose", "SAM-6D"):
                w.writerow([m] + [""] * 8)
        print(f"Created literature template {lit_path}: fill in numbers and sources.")
    literature = list(csv.DictReader(open(lit_path)))
    header = ["Method", "ADD-S AUC", "ADD(-S) AUC", "AR MSSD", "AR MSPD", "BOP AR", "mIoU",
              "Input", "CAD", "Training", "Source"]

    # Official BOP19 scores (VSD needs depth rendering) from run_bop19_eval.sh, if available.
    bop_scores = load_json(out.parent / "bop_toolkit/eval/labelfactory_ycbv-test/scores_bop19.json") \
        if (out.parent / "bop_toolkit/eval/labelfactory_ycbv-test/scores_bop19.json").is_file() else None
    if bop_scores:
        results["bop19_official"] = bop_scores
        json.dump(results, open(out / "summary.json", "w"), indent=2)

    def ours(label, s, bop=None):
        m = s["mean_over_objects"]
        return [label, fmt(m["adds_auc"]), fmt(m["add_or_adds_auc"]), fmt(s["pooled"]["ar_mssd"]),
                fmt(s["pooled"]["ar_mspd"]),
                fmt(bop["bop19_average_recall"] * 100) if bop else "", fmt(m["mean_mask_iou"]),
                "RGB-D seq.", "yes", "no", "this work"]
    table = [ours(args.method_name, results["all"], bop_scores),
             ours(args.method_name + ", low-support excluded", results["no_low_support"])]
    if single_records:
        sc = results["single_frame_comparison"]
        for label, key in (("SAM-6D single-frame (our run)", "sam6d_single_frame"),
                           ("Label Factory, same targets", "label_factory_same_targets")):
            m, pl = sc[key]["mean_over_objects"], sc[key]["pooled"]
            table.append([label, fmt(m["adds_auc"]), fmt(m["add_or_adds_auc"]),
                          fmt(pl["ar_mssd"]), fmt(pl["ar_mspd"]), "", fmt(m["mean_mask_iou"]), "RGB-D",
                          "yes", "no", "this work"])
    for r in literature:
        table.append([r.get("method", ""), r.get("adds_auc", ""), r.get("add_or_adds_auc", ""),
                      "", "", r.get("bop_ar", ""), r.get("mask_iou", ""), r.get("input", ""),
                      r.get("uses_cad", ""), r.get("needs_training", ""), r.get("source", "")])
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(row) + " |" for row in table]
    (out / "comparison_table.md").write_text("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{l" + "c" * 6 + "lccl}", "\\hline",
           " & ".join(header) + " \\\\", "\\hline"]
    tex += [" & ".join(c.replace("_", "\\_").replace("%", "\\%") for c in row) + " \\\\"
            for row in table]
    tex += ["\\hline", "\\end{tabular}"]
    (out / "comparison_table.tex").write_text("\n".join(tex) + "\n")

    # Literature-only table (the same numbers as the rows above, without ours).
    lit_header = ["Method", "Input", "CAD", "Training on YCB-V", "ADD-S AUC", "ADD(-S) AUC",
                  "BOP AR", "Source"]
    lit_rows = [[r["method"], r.get("input", ""), r.get("uses_cad", ""),
                 r.get("needs_training", ""), r.get("adds_auc", ""),
                 r.get("add_or_adds_auc", ""), r.get("bop_ar", ""), r.get("source", "")]
                for r in literature]
    lit_md = ["| " + " | ".join(lit_header) + " |", "|" + "---|" * len(lit_header)]
    lit_md += ["| " + " | ".join(r) + " |" for r in lit_rows]
    (out / "literature_table.md").write_text("\n".join(lit_md) + "\n")
    lit_tex = ["\\begin{tabular}{llccccc}", "\\hline",
               " & ".join(lit_header[:-1]) + " \\\\", "\\hline"]
    lit_tex += [" & ".join(c.replace("_", "\\_").replace("%", "\\%").replace("*", "$^{*}$")
                           for c in r[:-1]) + " \\\\" for r in lit_rows]
    lit_tex += ["\\hline", "\\end{tabular}"]
    (out / "literature_table.tex").write_text("\n".join(lit_tex) + "\n")

    pooled = results["all"]["pooled"]
    a, n = results["all"], results["no_low_support"]
    print(f"Datasets evaluated: {results['datasets_evaluated']}")
    if skipped:
        print(f"Skipped (no pose-fitting output yet): {skipped}")
    print(f"Targets: {pooled['targets']}, found: {pooled['found']}, "
          f"mask-IoU targets: {pooled['mask_iou_targets']}")
    for label, key in (("ADD-S AUC", "adds_auc"), ("ADD(-S) AUC", "add_or_adds_auc"),
                       ("ADD(-S)<0.1d", "add_or_adds_below_0.1d"), ("mask IoU", "mean_mask_iou")):
        print(f"{label:14s} (mean over objects): all {fmt(a['mean_over_objects'][key])}, "
              f"low-support excluded {fmt(n['mean_over_objects'][key])}")
    for label, key in (("AR MSSD", "ar_mssd"), ("AR MSPD", "ar_mspd")):
        print(f"{label:14s} (pooled): all {fmt(a['pooled'][key])}, "
              f"low-support excluded {fmt(n['pooled'][key])}")
    if single_records:
        sc = results["single_frame_comparison"]
        print("\nSingle-frame SAM-6D vs Label Factory on the same targets "
              f"({sc['sam6d_single_frame']['pooled']['targets']} targets):")
        for label, key in (("SAM-6D single-frame", "sam6d_single_frame"),
                           ("Label Factory", "label_factory_same_targets")):
            m, pl = sc[key]["mean_over_objects"], sc[key]["pooled"]
            print(f"  {label:20s} found {pl['found']:5d}  ADD-S AUC {fmt(m['adds_auc']):>5s}  "
                  f"ADD(-S) AUC {fmt(m['add_or_adds_auc']):>5s}  AR MSSD {fmt(pl['ar_mssd']):>5s}  "
                  f"AR MSPD {fmt(pl['ar_mspd']):>5s}  mask IoU {fmt(m['mean_mask_iou']):>5s}"
                  + (f" (primary instance only {fmt(m['mean_mask_iou_primary_instance'])})"
                     if key != "sam6d_single_frame" else ""))
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
