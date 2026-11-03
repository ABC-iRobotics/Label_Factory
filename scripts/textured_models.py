"""Optional textured CAD models for SAM-6D template rendering.

Dataset layout (the STL is always required, the textured model is optional):

    objects/stl/<name>.stl            geometry used by every pipeline stage
    objects/textured/<name>.obj       optional textured model, same coordinates/units
    objects/textured/<name>.mtl       material file referenced by the OBJ
    objects/textured/<name>.png       texture image(s) referenced by the MTL

When a complete textured model exists it is used to render the SAM-6D templates;
otherwise the STL is used. Nothing else changes: poses, masks and RANSAC keep using
the STL, so a textured and an untextured run are directly comparable.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

TEXTURE_DIR = "textured"


def _referenced(path: Path, pattern: str) -> list[str]:
    """Values of lines like 'mtllib x.mtl' / 'map_Kd x.png' (last token = file name)."""
    found = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(pattern, line.strip())
        if match:
            tokens = match.group(1).split()
            if tokens:
                found.append(tokens[-1].replace("\\", "/"))
    return found


def find_textured_model(dataset: Path, object_name: str):
    """Return (obj_path, files, problem).

    obj_path is None when there is no usable textured model; problem then explains
    why a model that does exist was rejected (None when there simply is none).
    files lists every file the textured model consists of (for hashing).
    """
    obj_path = dataset / "objects" / TEXTURE_DIR / f"{object_name}.obj"
    if not obj_path.is_file():
        return None, [], None
    files = [obj_path]
    materials = _referenced(obj_path, r"^mtllib\s+(.+)$")
    if not materials:
        return None, [], f"{obj_path.name} has no 'mtllib' line"
    textures = []
    for name in materials:
        mtl = obj_path.parent / name
        if not mtl.is_file():
            return None, [], f"material file {mtl} is missing"
        files.append(mtl)
        textures += [obj_path.parent / t for t in _referenced(mtl, r"^map_\w+\s+(.+)$")]
    if not textures:
        return None, [], f"{obj_path.name} references no texture image"
    for texture in textures:
        if not texture.is_file():
            return None, [], f"texture image {texture} is missing"
    return obj_path, files + sorted(set(textures)), None


def combined_sha256(geometry: Path, textured_files: list[Path]) -> str:
    """Identity of geometry + textured model; equals the plain file hash without texture."""
    digest = hashlib.sha256()
    for path in [geometry, *textured_files]:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if textured_files:
            digest.update(path.name.encode())
    return digest.hexdigest()


def export_textured_obj(vertices, faces, uv, normals, texture_image: Path,
                        out_dir: Path, name: str) -> Path:
    """Write <name>.obj/.mtl/.png (one UV per vertex, coordinates unchanged)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(texture_image, out_dir / f"{name}.png")
    (out_dir / f"{name}.mtl").write_text(
        f"newmtl {name}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd {name}.png\n")
    lines = [f"mtllib {name}.mtl", f"o {name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in normals]
    lines.append(f"usemtl {name}")
    lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in (faces + 1)]
    obj_path = out_dir / f"{name}.obj"
    obj_path.write_text("\n".join(lines) + "\n")
    return obj_path
