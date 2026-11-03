#!/usr/bin/env python3
"""Encode annotation-overlay images as an MP4 video."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import imageio_ffmpeg
import yaml

from dataset_config import dataset_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def dataset_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text()) or {}
    if "path" not in config or "dataset_id" not in config:
        raise ValueError(f"{config_path} must define path and dataset_id")
    return dataset_path(config["path"], config["dataset_id"])


def frame_key(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[-1]) if numbers else 2**63 - 1, path.name)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project / "scripts/configs/default.yaml")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--codec", default="libx264",
                        help="FFmpeg video codec (default: browser-compatible libx264).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    dataset = dataset_from_config(args.config.expanduser().resolve())
    input_dir = (args.input_dir.expanduser().resolve() if args.input_dir else
                 dataset / "sam6d_scene/visualizations")
    output = (args.output.expanduser().resolve() if args.output else
              dataset / "sam6d_scene/annotation_overlays.mp4")
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    frames = sorted(
        (path for path in input_dir.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=frame_key,
    )
    if not frames:
        raise RuntimeError(f"No overlay images found in {input_dir}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output}; pass --overwrite to replace it")

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    writer = imageio_ffmpeg.write_frames(
        str(temporary), (width, height), fps=float(args.fps),
        codec=args.codec, pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        macro_block_size=2, ffmpeg_log_level="warning",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for index, path in enumerate(frames, 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {path}")
            if image.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame {path} has size {image.shape[1]}x{image.shape[0]}; "
                    f"expected {width}x{height}"
                )
            writer.send(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if index == 1 or index % 25 == 0 or index == len(frames):
                print(f"Encoded [{index}/{len(frames)}] {path.name}", flush=True)
    finally:
        writer.close()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Video encoder produced no data: {temporary}")
    temporary.replace(output)
    print(f"Saved {len(frames)} frames at {args.fps:g} FPS: {output}")
    print(f"Duration: {len(frames) / args.fps:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
