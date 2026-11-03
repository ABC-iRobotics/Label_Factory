#!/usr/bin/env python3
"""Integrated multi-candidate ISM, de-duplication, PEM, and aggregation workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

import stage_timing
from dataset_config import dataset_path


def run(command: list[str | Path], cwd: Path) -> None:
    rendered = list(map(str, command))
    print("$ " + " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=cwd, check=True)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--python", required=True,
                        help="Python interpreter containing SAM-6D dependencies.")
    parser.add_argument("--object-name")
    parser.add_argument("--segmentor-model", choices=("sam", "fastsam"), default="sam")
    parser.add_argument("--stability-score-thresh", type=float, default=0.85)
    parser.add_argument("--det-score-thresh", type=float, default=0.2)
    parser.add_argument("--mask-iou-threshold", type=float, default=0.8)
    parser.add_argument("--containment-threshold", type=float, default=0.9)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    args = parser.parse_args()
    config = args.config.expanduser().resolve()

    ism_command: list[str | Path] = [
        sys.executable, project / "scripts/SAM6D_batch_inference.py",
        "--config", config, "--python", args.python,
        "--segmentor-model", args.segmentor_model,
        "--stability-score-thresh", args.stability_score_thresh,
        "--det-score-thresh", args.det_score_thresh, "--ism-only",
    ]
    audit_command: list[str | Path] = [
        sys.executable, project / "scripts/audit_ism_candidates.py",
        "--config", config, "--score-threshold", args.det_score_thresh,
        "--mask-iou-threshold", args.mask_iou_threshold,
        "--containment-threshold", args.containment_threshold,
    ]
    pem_command: list[str | Path] = [
        sys.executable, project / "scripts/run_experimental_multi_instance_pem.py",
        "--config", config, "--python", args.python,
    ]
    aggregate_command: list[str | Path] = [
        sys.executable, project / "scripts/aggregate_experimental_multi_instance_poses.py",
        "--config", config, "--require-poses",
    ]
    for command in (ism_command, audit_command):
        if args.start_frame is not None:
            command.extend(("--start-frame", args.start_frame))
        if args.end_frame is not None:
            command.extend(("--end-frame", args.end_frame))
    if args.object_name:
        ism_command.extend(("--object-name", args.object_name))
        for command in (audit_command, pem_command, aggregate_command):
            command.extend(("--object", args.object_name))
    if args.overwrite:
        ism_command.append("--overwrite")
        pem_command.append("--overwrite")
    if args.skip_visualizations:
        ism_command.append("--skip-visualizations")
        audit_command.append("--no-visualizations")

    dataset_config = yaml.safe_load(config.read_text())
    dataset = dataset_path(dataset_config["path"], dataset_config["dataset_id"])
    durations = {}

    def timed(name: str, command: list[str | Path]) -> None:
        started = time.monotonic()
        run(command, project)
        durations[name] = time.monotonic() - started

    templates_before = stage_timing.total(dataset, "sam6d_templates")
    print("=== Multi-instance stage 1/4: object-specific ISM ===")
    timed("ism", ism_command)
    print("=== Multi-instance stage 2/4: candidate de-duplication ===")
    timed("audit", audit_command)
    print("=== Multi-instance stage 3/4: persistent PEM ===")
    timed("pem", pem_command)
    print("=== Multi-instance stage 4/4: pose aggregation ===")
    timed("aggregate", aggregate_command)
    # Template rendering happens inside the ISM step and is reported as its own stage.
    templates = max(stage_timing.total(dataset, "sam6d_templates") - templates_before, 0.0)
    durations["ism"] = max(durations["ism"] - templates, 0.0)
    stage_timing.record(
        dataset, "sam6d_inference", sum(durations.values()),
        item=args.object_name or stage_timing.ALL_ITEMS,
        parts={key: round(value, 3) for key, value in durations.items()},
    )
    print("Multi-instance inference completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
