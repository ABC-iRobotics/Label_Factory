#!/usr/bin/env python3
"""Generate scene annotations and their RGB overlays as one workflow stage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
                        help="Python interpreter containing BlenderProc and SAM-6D.")
    parser.add_argument("--model-scale", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    annotation_command: list[str | Path] = [
        sys.executable, project / "scripts/create_blender_scene_annotations.py",
        "--config", args.config.expanduser().resolve(),
        "--python", args.python, "--model-scale", args.model_scale,
    ]
    if args.overwrite:
        annotation_command.append("--overwrite")
    print("=== Scene annotation stage 1/2: render masks and metadata ===")
    run(annotation_command, project)
    print("=== Scene annotation stage 2/2: generate RGB overlays ===")
    run([
        sys.executable, project / "scripts/visualize_scene_annotations.py",
        "--config", args.config.expanduser().resolve(), "--overwrite",
    ], project)
    print("Scene annotations and overlays generated successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
