#!/usr/bin/env python3
"""Aggregate per-stage wall-clock times of the Label Factory pipeline over many scenes.

Reads data/<scene>/pipeline_timings.json (written by the pipeline scripts, see
stage_timing.py) plus the frame counts of each scene, and writes to evaluation/runtime/:

  runtime_per_scene.csv   per scene: frame counts and the six stage times (seconds)
  runtime_summary.md/.tex mean +- std per stage over the complete scenes
  runtime_scaling.png     total / SAM-6D time against subset size, with fit statistics
  runtime_scaling.json    the fit statistics (Pearson r, slope, R^2)
  runtime_section.tex     draft results paragraph filled from the numbers above

The six stages: frame selection (reconstruction + SAM-6D keyframes), SfM/MVS
reconstruction, SAM-6D template generation, SAM-6D instance segmentation + pose
estimation, multi-frame consolidation (RANSAC), BlenderProc annotation rendering.

Measure on clean scenes: stages that resume from earlier results (existing templates,
frames, reconstruction) report only the time of the work actually repeated.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGES = [  # key, label, LaTeX label
    ("frame_selection", "Frame selection", "Frame selection"),
    ("reconstruction", "SfM/MVS reconstruction", "SfM/MVS reconstruction"),
    ("sam6d_templates", "SAM-6D template generation", "SAM-6D templates"),
    ("sam6d_inference", "SAM-6D segmentation + pose estimation", "SAM-6D seg.\\ + pose"),
    ("consolidation", "Multi-frame consolidation", "Consolidation"),
    ("annotation_rendering", "Annotation rendering", "Annotation rendering"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Scene folder names (default: every data/* except comparison_*)")
    p.add_argument("--include-comparison", action="store_true",
                   help="Also include the comparison_* (benchmark) datasets")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <project-root>/evaluation/runtime")
    p.add_argument("--allow-partial", action="store_true",
                   help="Compute statistics for scenes that lack some stages (per-stage n differs)")
    return p.parse_args()


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def stage_seconds(timings, key, objects=(), has_sam_selection=False):
    """Total seconds of a logical stage for one scene, or None if not measured.

    - frame selection: the SAM-6D keyframe selection is an optional step. Without a
      sam6d_frame_selection.json the pipeline reuses the reconstruction subset, so the step
      did not run and costs 0 s; with one, its duration must have been recorded.
    - SAM-6D templates: only objects that are in the scene count (the timing file may also
      hold templates rendered for objects of earlier experiments).
    """
    from stage_timing import stage_total
    stages = (timings or {}).get("stages", {})
    if key == "frame_selection":
        reconstruction = stages.get("frame_selection_reconstruction")
        if reconstruction is None:
            return None
        sam6d = stages.get("frame_selection_sam6d")
        if sam6d is not None:
            return stage_total(reconstruction) + stage_total(sam6d)
        return None if has_sam_selection else stage_total(reconstruction)
    entry = stages.get(key)
    if entry is None:
        return None
    if key == "sam6d_templates" and entry.get("items") and objects:
        scene_items = {k: v for k, v in entry["items"].items() if k in objects}
        if scene_items:
            return float(sum(scene_items.values()))
    return stage_total(entry)


def inference_reruns(timings):
    """Seconds of per-object SAM-6D re-runs recorded next to a full run (not in the total)."""
    entry = ((timings or {}).get("stages", {}).get("sam6d_inference") or {})
    items = entry.get("items") or {}
    if "__all__" not in items:
        return 0.0
    return float(sum(v for k, v in items.items() if k != "__all__"))


def scene_record(scene: Path):
    timings = load_json(scene / "pipeline_timings.json")
    recon = load_json(scene / "reconstruction/reconstruction_summary.json") or {}
    sam_sel = load_json(scene / "sam6d_frame_selection.json") or {}
    rec_sel = load_json(scene / "reconstruction_frame_selection.json") or {}
    images = [p for p in (scene / "images").glob("*") if p.suffix.lower() in
              {".png", ".jpg", ".jpeg"}] if (scene / "images").is_dir() else []
    registered = recon.get("registered_camera_pose_frames") or recon.get("number_of_images")
    reconstruction_subset = recon.get("selected_mvs_frames") or rec_sel.get("selected_frames")
    sequences = scene / "sam6d_sequences"
    objects = sorted(d.name for d in sequences.iterdir() if d.is_dir()) \
        if sequences.is_dir() else []
    # The pipeline falls back to the reconstruction subset when there is no SAM-6D selection.
    sam6d_subset = sam_sel.get("selected_frames") or reconstruction_subset
    row = {
        "scene": scene.name,
        "frames_total": len(images),
        "frames_registered": registered,
        "frames_reconstruction_subset": reconstruction_subset,
        "frames_sam6d_subset": sam6d_subset,
        "objects": len(objects),
        "frame_object_tasks": sam6d_subset * len(objects) if sam6d_subset and objects else None,
        "reconstruction_target_overlap": rec_sel.get("target_overlap"),
        # Without a SAM-6D selection the reconstruction subset (and its overlap) is used.
        "sam6d_target_overlap": sam_sel.get("target_overlap", rec_sel.get("target_overlap")),
        "reconstruction_subset_pct": 100 * reconstruction_subset / len(images)
        if reconstruction_subset and images else None,
        "sam6d_subset_pct": 100 * sam6d_subset / len(images) if sam6d_subset and images else None,
    }
    for key, _, _ in STAGES:
        row[key] = stage_seconds(timings, key, objects, bool(sam_sel))
    row["sam6d_inference_reruns"] = inference_reruns(timings)
    row["_hardware"] = (timings or {}).get("hardware")
    return row


def mean_std(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"n": int(len(x)), "pearson_r": r, "r_squared": r * r,
            "slope_s_per_frame": float(slope), "intercept_s": float(intercept)}


def main():
    a = parse_args()
    project = a.project_root.resolve()
    import sys
    sys.path.insert(0, str(project / "scripts"))
    out = (a.output_dir or project / "evaluation/runtime").resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = a.datasets or sorted(
        p.name for p in (project / "data").iterdir()
        if p.is_dir() and (a.include_comparison or not p.name.startswith("comparison_")))
    rows = [scene_record(project / "data" / n) for n in names]
    hardware = next((r["_hardware"] for r in rows if r["_hardware"]), None)
    measured = [r for r in rows if any(r[k] is not None for k, _, _ in STAGES)]
    complete = [r for r in measured if all(r[k] is not None for k, _, _ in STAGES)]
    for r in rows:
        r["total"] = sum(r[k] for k, _, _ in STAGES) if r in complete else None
    used = measured if a.allow_partial else complete

    # --- per-scene CSV
    columns = ["scene", "frames_total", "frames_registered", "reconstruction_target_overlap",
               "frames_reconstruction_subset", "reconstruction_subset_pct",
               "sam6d_target_overlap", "frames_sam6d_subset", "sam6d_subset_pct",
               "objects", "frame_object_tasks"] \
        + [k for k, _, _ in STAGES] + ["total", "sam6d_inference_reruns"]
    with open(out / "runtime_per_scene.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow(["" if r[c] is None else
                        (round(r[c], 2 if c.endswith("_overlap") else 1)
                         if isinstance(r[c], float) else r[c]) for c in columns])

    missing = {r["scene"]: [k for k, _, _ in STAGES if r[k] is None] for r in rows
               if r not in complete}
    if not used:
        print("No scene has timing data yet. Run the pipeline stages (they now record their "
              "duration into data/<scene>/pipeline_timings.json) and rerun this script.")
        for scene, miss in missing.items():
            print(f"  {scene}: missing {', '.join(miss) if miss else 'everything'}")
        return 1

    # --- summary table
    stats = {k: mean_std([r[k] for r in used]) for k, _, _ in STAGES}
    stats["total"] = mean_std([r["total"] for r in used if r["total"] is not None]
                              if not a.allow_partial else
                              [sum(r[k] or 0 for k, _, _ in STAGES) for r in used])
    total_mean = stats["total"][0]
    lines = ["| Stage | Mean (min) | Std (min) | Share of total | n |", "|---|---|---|---|---|"]
    tex = ["\\begin{table}[t]", "\\centering",
           "\\caption{Per-stage wall-clock time of the pipeline (mean $\\pm$ sample std over "
           f"{len(used)} scenes).}}", "\\label{tab:runtime}", "\\begin{tabular}{lcc}", "\\hline",
           "Stage & Time (min) & Share \\\\", "\\hline"]
    for key, label, tex_label in STAGES:
        m, s, n = stats[key]
        share = 100 * m / total_mean if total_mean else float("nan")
        lines.append(f"| {label} | {m / 60:.1f} | {s / 60:.1f} | {share:.0f}% | {n} |")
        tex.append(f"{tex_label} & ${m / 60:.1f} \\pm {s / 60:.1f}$ & {share:.0f}\\% \\\\")
    m, s, n = stats["total"]
    lines.append(f"| **Total per scene** | {m / 60:.1f} | {s / 60:.1f} | 100% | {n} |")
    tex += ["\\hline", f"Total per scene & ${m / 60:.1f} \\pm {s / 60:.1f}$ & 100\\% \\\\",
            "\\hline", "\\end{tabular}", "\\end{table}"]
    (out / "runtime_summary.md").write_text("\n".join(lines) + "\n")
    (out / "runtime_summary.tex").write_text("\n".join(tex) + "\n")

    # --- per-scene LaTeX table (target overlap, frames and share, stage times in minutes)
    def pct(x):
        return "--" if x is None else f"{100 * x:.0f}"
    scene_tex = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\caption{Processing time per scene. Overlap: target overlap of the keyframe "
        "selection for the reconstruction (Rec.) and for SAM-6D; Frames: number of selected "
        "frames and their share of the recorded sequence; Obj.: number of annotated objects. "
        "Times in minutes: frame selection (Sel.), SfM/MVS reconstruction (Recon.), SAM-6D "
        "template generation (Templ.), SAM-6D segmentation and pose estimation (SAM-6D), "
        "multi-frame consolidation (Cons.) and annotation rendering (Render).}",
        "\\label{tab:runtime_scenes}", "\\begin{tabular}{lcccccrrrrrrr}", "\\hline",
        " & & \\multicolumn{2}{c}{Rec.} & \\multicolumn{2}{c}{SAM-6D} & & & & & & & \\\\",
        "Scene & Obj. & Overlap (\\%) & Frames & Overlap (\\%) & Frames & Sel. & Recon. & "
        "Templ. & SAM-6D & Cons. & Render & Total \\\\", "\\hline"]
    for r in used:
        scene_tex.append(
            f"{r['scene'].replace('_', chr(92) + '_')} & {r['objects']} & "
            f"{pct(r['reconstruction_target_overlap'])} & {r['frames_reconstruction_subset']} "
            f"({r['reconstruction_subset_pct']:.0f}\\%) & {pct(r['sam6d_target_overlap'])} & "
            f"{r['frames_sam6d_subset']} ({r['sam6d_subset_pct']:.0f}\\%) & "
            f"{r['frame_selection'] / 60:.1f} & {r['reconstruction'] / 60:.1f} & "
            f"{r['sam6d_templates'] / 60:.1f} & {r['sam6d_inference'] / 60:.1f} & "
            f"{r['consolidation'] / 60:.1f} & {r['annotation_rendering'] / 60:.1f} & "
            f"{r['total'] / 60 if r['total'] is not None else float('nan'):.1f} \\\\")
    mean_cells = [f"${stats[k][0] / 60:.1f} \\pm {stats[k][1] / 60:.1f}$" for k, _, _ in STAGES]
    scene_tex += ["\\hline",
                  "Mean $\\pm$ std & & & & & & " + " & ".join(mean_cells)
                  + f" & ${stats['total'][0] / 60:.1f} \\pm {stats['total'][1] / 60:.1f}$ \\\\",
                  "\\hline", "\\end{tabular}", "\\end{table*}"]
    (out / "runtime_per_scene.tex").write_text("\n".join(scene_tex) + "\n")

    # --- scaling
    def pairs(xkey, ykey):
        pts = [(r[xkey], r[ykey]) for r in used if r[xkey] is not None and r[ykey] is not None]
        return [p[0] for p in pts], [p[1] for p in pts]
    scaling = {}
    for name, xk, yk in (("total_vs_registered", "frames_registered", "total"),
                         ("total_vs_sam6d_subset", "frames_sam6d_subset", "total"),
                         ("sam6d_vs_registered", "frames_registered", "sam6d_inference"),
                         ("sam6d_vs_sam6d_subset", "frames_sam6d_subset", "sam6d_inference"),
                         ("sam6d_vs_frame_object_tasks", "frame_object_tasks", "sam6d_inference"),
                         ("reconstruction_vs_registered", "frames_registered", "reconstruction")):
        scaling[name] = fit(*pairs(xk, yk))
    json.dump(scaling, open(out / "runtime_scaling.json", "w"), indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, (xk, yk, xl, yl, name) in zip(axes[:2], (
                ("frames_sam6d_subset", "total", "SAM-6D keyframes (n)",
                 "Total time (min)", "total_vs_sam6d_subset"),
                ("frames_sam6d_subset", "sam6d_inference", "SAM-6D keyframes (n)",
                 "SAM-6D stage (min)", "sam6d_vs_sam6d_subset"))):
            x, y = pairs(xk, yk)
            ax.scatter(x, np.array(y) / 60, s=22)
            f_ = scaling[name]
            if f_:
                xs = np.linspace(min(x), max(x), 50)
                ax.plot(xs, (f_["slope_s_per_frame"] * xs + f_["intercept_s"]) / 60, "--", lw=1)
                ax.set_title(f"r = {f_['pearson_r']:.2f}, n = {f_['n']}", fontsize=9)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.grid(alpha=0.3)
        bars = axes[2]
        bottom = np.zeros(len(used))
        for key, label_, _ in STAGES:
            values = np.array([r[key] / 60 for r in used])
            bars.bar(range(len(used)), values, bottom=bottom, label=label_, width=0.75)
            bottom += values
        bars.set_xticks(range(len(used)))
        bars.set_xticklabels([r["scene"] for r in used], rotation=45, fontsize=7)
        bars.set_ylabel("Time per scene (min)")
        bars.set_ylim(0, bottom.max() * 1.5)
        bars.legend(fontsize=6, loc="upper center", ncol=2)
        bars.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / "runtime_scaling.png", dpi=200)
    except ImportError:
        print("matplotlib not available: skipped runtime_scaling.png")

    # --- draft paragraph (numbers only; conclusions follow the measured correlations)
    dominant = max(((k, stats[k][0]) for k, _, _ in STAGES), key=lambda t: t[1])
    label = {k: l for k, l, _ in STAGES}[dominant[0]]
    r_sub = scaling["sam6d_vs_sam6d_subset"]
    registered_values = {r["frames_registered"] for r in used if r["frames_registered"]}
    sizes = [r["frames_sam6d_subset"] for r in used if r["frames_sam6d_subset"]]
    shares = [r["sam6d_subset_pct"] for r in used if r["sam6d_subset_pct"]]
    overlaps = {name: sorted({r[key] for r in used if r[key] is not None})
                for name, key in (("rec", "reconstruction_target_overlap"),
                                  ("sam", "sam6d_target_overlap"))}

    def overlap_text(values):
        return "/".join(f"{100 * v:.0f}\\%" for v in values) if values else "n/a"
    parts = []
    if len(registered_values) == 1 and sizes:
        parts.append(
            f"All sequences contain {registered_values.pop()} registered frames, while "
            f"{min(sizes)}--{max(sizes)} frames ({min(shares):.0f}--{max(shares):.0f}\\%) were "
            "selected for SAM-6D.")
    if r_sub:
        parts.append(
            f"The SAM-6D stage time grows linearly with the number of selected frames "
            f"(about {r_sub['slope_s_per_frame']:.1f}\\,s per frame plus "
            f"{r_sub['intercept_s']:.0f}\\,s fixed cost; Pearson $r={r_sub['pearson_r']:.2f}$, "
            f"$n={r_sub['n']}$).")
    scale_sentence = " ".join(parts) or "Too few scenes to fit the scaling relation."
    overlap_sentence = (
        f"Keyframes were selected with a target overlap of {overlap_text(overlaps['rec'])} for "
        f"the reconstruction and {overlap_text(overlaps['sam'])} for SAM-6D. ")
    hw = hardware or {}
    gpu_fields = [f.strip() for f in str(hw.get("gpu", "")).split(",")]
    gpu_text = gpu_fields[0] if gpu_fields and gpu_fields[0] else ""
    if len(gpu_fields) > 1 and gpu_fields[1].endswith("MiB"):
        gpu_text += f" ({round(int(gpu_fields[1].split()[0]) / 1024)}\\,GB)"
    hw_text = ", ".join(t for t in (gpu_text, hw.get("cpu", "")) if t)
    if hw.get("ram_gib"):
        hw_text += f", {hw['ram_gib']:.0f}\\,GiB RAM"
    para = (
        f"Table~\\ref{{tab:runtime}} reports the wall-clock time of each stage over {len(used)} "
        f"scenes, measured on {hw_text or '[hardware not recorded]'}. A scene takes "
        f"{stats['total'][0] / 60:.1f}\\,$\\pm$\\,{stats['total'][1] / 60:.1f}\\,min in total "
        f"(mean $\\pm$ std). The dominant stage is {label} "
        f"({100 * dominant[1] / total_mean:.0f}\\% of the total). " + overlap_sentence
        + scale_sentence)
    (out / "runtime_section.tex").write_text(para + "\n")
    if len({r["objects"] for r in used}) > 1 and all(r["total"] is not None for r in used):
        from runtime_groups import write_grouped  # separate blocks per object count
        write_grouped(out, used, hw_text)

    print(f"Scenes with all six stages: {len(complete)}/{len(rows)}"
          + ("" if a.allow_partial else " (statistics use these)"))
    for scene, miss in missing.items():
        print(f"  incomplete {scene}: missing {', '.join(miss) if miss else 'everything'}")
    rerun = [(r["scene"], r["sam6d_inference_reruns"]) for r in used if r["sam6d_inference_reruns"]]
    if rerun:
        print("Scenes with extra per-object SAM-6D re-runs recorded next to the full run "
              "(not included in the totals): "
              + ", ".join(f"{n} (+{t / 60:.1f} min)" for n, t in rerun))
    print("\n".join(lines))
    print(f"Results in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
