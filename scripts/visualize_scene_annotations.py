#!/usr/bin/env python3
"""Overlay rendered object masks on their corresponding RGB images."""

from __future__ import annotations

import argparse
import colorsys
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from dataset_config import dataset_path


def dataset_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    return dataset_path(config["path"], config["dataset_id"])


def object_color(index: int, count: int) -> tuple[int, int, int]:
    """Generate stable, saturated colors distributed around the hue wheel."""
    hue = (index * 0.618033988749895) % 1.0
    saturation = 0.88
    value = 1.0
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(round(channel * 255)) for channel in rgb)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel inner boundary without OpenCV/SciPy."""
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def label_position(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), max(0, int(ys.min()) - 18)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project / "scripts/configs/default.yaml",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Defaults to data/####/sam6d_scene/annotations.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to the annotations directory's visualizations folder.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.48,
        help="Mask opacity in [0, 1] (default: 0.48).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    config_path = args.config.expanduser().resolve()
    dataset = dataset_from_config(config_path)
    annotations_path = (
        args.annotations.expanduser().resolve()
        if args.annotations
        else dataset / "sam6d_scene/annotations.json"
    )
    annotations = json.loads(annotations_path.read_text())
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else annotations_path.parent / "visualizations"
    )
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    objects = sorted(
        annotations["objects"], key=lambda item: int(item["object_id"])
    )
    colors = {
        int(item["object_id"]): object_color(index, len(objects))
        for index, item in enumerate(objects)
    }
    names = {
        int(item["object_id"]): item.get("instance_name", item["object_name"])
        for item in objects
    }
    font = ImageFont.load_default()

    legend_height = max(28, 22 * len(objects) + 8)
    generated = 0
    for frame in annotations["frames"]:
        image_path = Path(frame["image_path"])
        mask_path = Path(frame["instance_mask_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8).copy()
        mask = np.asarray(Image.open(mask_path))
        if mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"Image/mask shape mismatch for frame {frame['frame_id']}: "
                f"{rgb.shape[:2]} versus {mask.shape}"
            )

        labels = []
        for object_id, name in names.items():
            selected = mask == object_id
            if not np.any(selected):
                continue
            color = np.asarray(colors[object_id], dtype=np.float32)
            rgb[selected] = np.clip(
                (1.0 - args.alpha) * rgb[selected].astype(np.float32)
                + args.alpha * color,
                0,
                255,
            ).astype(np.uint8)
            rgb[mask_boundary(selected)] = color.astype(np.uint8)
            position = label_position(selected)
            if position is not None:
                labels.append((position, name, colors[object_id]))

        result = Image.fromarray(rgb)
        canvas = Image.new(
            "RGB",
            (result.width, result.height + legend_height),
            (24, 24, 24),
        )
        canvas.paste(result, (0, 0))
        draw = ImageDraw.Draw(canvas)

        for (x, y), name, color in labels:
            text_box = draw.textbbox((x, y), name, font=font)
            draw.rectangle(
                (
                    text_box[0] - 3,
                    text_box[1] - 2,
                    text_box[2] + 3,
                    text_box[3] + 2,
                ),
                fill=(0, 0, 0),
            )
            draw.text((x, y), name, fill=color, font=font)

        legend_y = result.height + 7
        for item in objects:
            object_id = int(item["object_id"])
            color = colors[object_id]
            draw.rectangle((10, legend_y, 24, legend_y + 14), fill=color)
            draw.text(
                (31, legend_y + 1),
                f"{object_id}: {item.get('instance_name', item['object_name'])}",
                fill=(245, 245, 245),
                font=font,
            )
            legend_y += 22

        output_path = output_dir / f"{int(frame['frame_id']):06d}.png"
        canvas.save(output_path)
        generated += 1

    color_map = {
        str(object_id): {
            "instance_name": names[object_id],
            "rgb": list(colors[object_id]),
        }
        for object_id in names
    }
    (output_dir / "colors.json").write_text(
        json.dumps(color_map, indent=2) + "\n"
    )
    print(f"Saved {generated} mask overlays to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
