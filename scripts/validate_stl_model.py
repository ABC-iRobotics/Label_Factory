#!/usr/bin/env python3
"""Validate an STL mesh and emit a machine-readable geometry report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def inspect_stl(path: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    if path.suffix.lower() != ".stl":
        errors.append("The file extension must be .stl")
    if not path.is_file() or path.stat().st_size == 0:
        errors.append("The uploaded file is empty")
    mesh = o3d.geometry.TriangleMesh()
    if not errors:
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        if mesh.is_empty():
            errors.append("Open3D could not read triangle geometry from this STL")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    dimensions = np.zeros(3, dtype=np.float64)
    degenerate_count = 0
    if not errors:
        if len(vertices) < 3 or len(triangles) < 1:
            errors.append("The mesh does not contain enough vertices and triangles")
        elif not np.isfinite(vertices).all():
            errors.append("The mesh contains NaN or infinite coordinates")
        elif triangles.min() < 0 or triangles.max() >= len(vertices):
            errors.append("The mesh contains invalid triangle indices")
        else:
            dimensions = vertices.max(axis=0) - vertices.min(axis=0)
            if np.any(dimensions <= 1e-12):
                errors.append("The mesh has zero extent along one or more axes")
            edges_a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
            edges_b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
            double_area = np.linalg.norm(np.cross(edges_a, edges_b), axis=1)
            degenerate_count = int(np.count_nonzero(double_area <= 1e-15))
            if degenerate_count == len(triangles):
                errors.append("Every triangle in the mesh is degenerate")
            elif degenerate_count:
                warnings.append(f"{degenerate_count} degenerate triangles were found")

    watertight = bool(mesh.is_watertight()) if not errors else False
    edge_manifold = bool(mesh.is_edge_manifold(allow_boundary_edges=True)) if not errors else False
    if not errors and not watertight:
        warnings.append("Mesh is not watertight; verify that this is intentional")
    if not errors and not edge_manifold:
        warnings.append("Mesh has non-manifold edges")

    return {
        "valid": not errors,
        "filename": path.name,
        "file_size_bytes": path.stat().st_size if path.is_file() else 0,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "degenerate_triangle_count": degenerate_count,
        "dimensions_source_units": dimensions.tolist(),
        "watertight": watertight,
        "edge_manifold": edge_manifold,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", type=Path)
    args = parser.parse_args()
    report = inspect_stl(args.stl.expanduser().resolve())
    print("STL_REPORT_JSON=" + json.dumps(report, separators=(",", ":")))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
