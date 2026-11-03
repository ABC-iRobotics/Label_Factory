"""Grouped runtime outputs: scenes with a different number of objects form separate blocks.

Used by summarize_runtime.py when the scenes do not all have the same number of objects.
Overwrites runtime_summary.md/.tex, runtime_per_scene.tex, runtime_scaling.png/.json and
runtime_section.tex with versions that report each group and all scenes together.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STAGE_KEYS = [  # key, label, short LaTeX label
    ("frame_selection", "Frame selection", "Frame selection"),
    ("reconstruction", "SfM/MVS reconstruction", "SfM/MVS reconstruction"),
    ("sam6d_templates", "SAM-6D template generation", "SAM-6D templates"),
    ("sam6d_inference", "SAM-6D segmentation + pose estimation", "SAM-6D seg.\\ + pose"),
    ("consolidation", "Multi-frame consolidation", "Consolidation"),
    ("annotation_rendering", "Annotation rendering", "Annotation rendering"),
]


def _ms(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"n": int(len(x)), "pearson_r": r, "r_squared": r * r,
            "slope_s_per_frame": float(slope), "intercept_s": float(intercept)}


def _pct(value):
    return "--" if value is None else f"{100 * value:.0f}"


def write_grouped(out: Path, used: list[dict], hw_text: str) -> None:
    groups = {k: [r for r in used if r["objects"] == k]
              for k in sorted({r["objects"] for r in used})}
    names = {k: f"{k} obj." for k in groups}
    columns = [(names[k], rows) for k, rows in groups.items()] + [("All", used)]

    # ---- summary table (mean +- std in minutes, one column per group)
    header = ["Stage"] + [f"{name} (n={len(rows)})" for name, rows in columns]
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    tex = ["\\begin{table}[t]", "\\centering",
           "\\caption{Per-stage wall-clock time of the pipeline (mean $\\pm$ sample std, minutes) "
           "for scenes with three and four annotated objects, and for all scenes together.}",
           "\\label{tab:runtime}", "\\begin{tabular}{l" + "c" * len(columns) + "}", "\\hline",
           "Stage & " + " & ".join(f"{name} ($n={len(rows)}$)" for name, rows in columns)
           + " \\\\", "\\hline"]
    for key, label, tex_label in STAGE_KEYS:
        cells = [_ms([r[key] / 60 for r in rows]) for _, rows in columns]
        md.append(f"| {label} | " + " | ".join(f"{m:.1f} ± {s:.1f}" for m, s in cells) + " |")
        tex.append(f"{tex_label} & " + " & ".join(f"${m:.1f} \\pm {s:.1f}$" for m, s in cells)
                   + " \\\\")
    cells = [_ms([r["total"] / 60 for r in rows]) for _, rows in columns]
    md.append("| **Total per scene** | " + " | ".join(f"{m:.1f} ± {s:.1f}" for m, s in cells) + " |")
    tex += ["\\hline", "Total per scene & "
            + " & ".join(f"${m:.1f} \\pm {s:.1f}$" for m, s in cells) + " \\\\", "\\hline",
            "\\end{tabular}", "\\end{table}"]
    (out / "runtime_summary.md").write_text("\n".join(md) + "\n")
    (out / "runtime_summary.tex").write_text("\n".join(tex) + "\n")

    # ---- per-scene table with one block per group
    tex = [
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

    def mean_row(label, rows):
        cells = [_ms([r[k] / 60 for r in rows]) for k, _, _ in STAGE_KEYS]
        total = _ms([r["total"] / 60 for r in rows])
        return (label + " & & & & & & "
                + " & ".join(f"${m:.1f} \\pm {s:.1f}$" for m, s in cells)
                + f" & ${total[0]:.1f} \\pm {total[1]:.1f}$ \\\\")
    for position, (k, rows) in enumerate(groups.items()):
        if position:
            tex.append("\\hline")
        for r in rows:
            tex.append(
                f"{r['scene'].replace('_', chr(92) + '_')} & {r['objects']} & "
                f"{_pct(r['reconstruction_target_overlap'])} & {r['frames_reconstruction_subset']} "
                f"({r['reconstruction_subset_pct']:.0f}\\%) & {_pct(r['sam6d_target_overlap'])} & "
                f"{r['frames_sam6d_subset']} ({r['sam6d_subset_pct']:.0f}\\%) & "
                f"{r['frame_selection'] / 60:.1f} & {r['reconstruction'] / 60:.1f} & "
                f"{r['sam6d_templates'] / 60:.1f} & {r['sam6d_inference'] / 60:.1f} & "
                f"{r['consolidation'] / 60:.1f} & {r['annotation_rendering'] / 60:.1f} & "
                f"{r['total'] / 60:.1f} \\\\")
        tex.append("\\hline")
        tex.append(mean_row(f"Mean $\\pm$ std ({k} obj.)", rows))
    tex += ["\\hline", mean_row("Mean $\\pm$ std (all)", used), "\\hline", "\\end{tabular}",
            "\\end{table*}"]
    (out / "runtime_per_scene.tex").write_text("\n".join(tex) + "\n")

    # ---- scaling per group
    scaling = {}
    for k, rows in groups.items():
        scaling[f"sam6d_vs_keyframes_{k}_objects"] = _fit(
            [r["frames_sam6d_subset"] for r in rows], [r["sam6d_inference"] for r in rows])
        scaling[f"total_vs_keyframes_{k}_objects"] = _fit(
            [r["frames_sam6d_subset"] for r in rows], [r["total"] for r in rows])
    scaling["sam6d_vs_frame_object_tasks_all"] = _fit(
        [r["frame_object_tasks"] for r in used], [r["sam6d_inference"] for r in used])
    json.dump(scaling, open(out / "runtime_scaling.json", "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for panel, (ykey, ylabel) in enumerate((("total", "Total time (min)"),
                                                ("sam6d_inference", "SAM-6D stage (min)"))):
            ax = axes[panel]
            titles = []
            for i, (k, rows) in enumerate(groups.items()):
                x = [r["frames_sam6d_subset"] for r in rows]
                y = np.array([r[ykey] for r in rows]) / 60
                ax.scatter(x, y, s=24, color=colors[i % 4], label=f"{k} objects")
                f_ = _fit(x, [r[ykey] for r in rows])
                if f_:
                    xs = np.linspace(min(x), max(x), 50)
                    ax.plot(xs, (f_["slope_s_per_frame"] * xs + f_["intercept_s"]) / 60, "--",
                            lw=1, color=colors[i % 4])
                    titles.append(f"{k} obj.: r = {f_['pearson_r']:.2f}")
            ax.set_title(", ".join(titles), fontsize=8)
            ax.set_xlabel("SAM-6D keyframes (n)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        bars = axes[2]
        bottom = np.zeros(len(used))
        for key, label, _ in STAGE_KEYS:
            values = np.array([r[key] / 60 for r in used])
            bars.bar(range(len(used)), values, bottom=bottom, label=label, width=0.75)
            bottom += values
        bars.set_xticks(range(len(used)))
        bars.set_xticklabels([r["scene"] for r in used], rotation=60, fontsize=6)
        bars.set_ylabel("Time per scene (min)")
        bars.set_ylim(0, bottom.max() * 1.5)
        bars.legend(fontsize=6, loc="upper center", ncol=2)
        bars.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / "runtime_scaling.png", dpi=200)
    except ImportError:
        print("matplotlib not available: skipped runtime_scaling.png")

    # ---- paragraph
    total = _ms([r["total"] / 60 for r in used])
    group_totals = ", ".join(
        f"{k} objects: {_ms([r['total'] / 60 for r in rows])[0]:.1f}\\,$\\pm$\\,"
        f"{_ms([r['total'] / 60 for r in rows])[1]:.1f}\\,min" for k, rows in groups.items())
    means = {key: _ms([r[key] for r in used])[0] for key, _, _ in STAGE_KEYS}
    dominant = max(means, key=means.get)
    label = {k: lab for k, lab, _ in STAGE_KEYS}[dominant]
    sizes = [r["frames_sam6d_subset"] for r in used]
    shares = [r["sam6d_subset_pct"] for r in used]
    overlaps = {name: sorted({r[key] for r in used if r[key] is not None})
                for name, key in (("rec", "reconstruction_target_overlap"),
                                  ("sam", "sam6d_target_overlap"))}

    def overlap_text(values):
        return "/".join(f"{100 * v:.0f}\\%" for v in values) if values else "n/a"
    registered = {r["frames_registered"] for r in used if r["frames_registered"]}
    slopes = []
    for k, rows in groups.items():
        f_ = scaling[f"sam6d_vs_keyframes_{k}_objects"]
        if f_:
            slopes.append(f"{f_['slope_s_per_frame']:.1f}\\,s per frame for {k} objects "
                          f"($r={f_['pearson_r']:.2f}$, $n={f_['n']}$)")
    para = (
        f"Table~\\ref{{tab:runtime}} reports the wall-clock time of each stage over {len(used)} "
        f"scenes, measured on {hw_text or '[hardware not recorded]'}. A scene takes "
        f"{total[0]:.1f}\\,$\\pm$\\,{total[1]:.1f}\\,min in total (mean $\\pm$ std; "
        f"{group_totals}). The dominant stage is {label} "
        f"({100 * means[dominant] / sum(means.values()):.0f}\\% of the total). "
        f"Keyframes were selected with a target overlap of {overlap_text(overlaps['rec'])} for "
        f"the reconstruction and {overlap_text(overlaps['sam'])} for SAM-6D. ")
    if len(registered) == 1:
        para += (f"All sequences contain {next(iter(registered))} registered frames, while "
                 f"{min(sizes)}--{max(sizes)} frames ({min(shares):.0f}--{max(shares):.0f}\\%) "
                 "were selected for SAM-6D. ")
    if slopes:
        para += ("The SAM-6D stage time grows linearly with the number of selected frames: "
                 + "; ".join(slopes) + ".")
    (out / "runtime_section.tex").write_text(para + "\n")
