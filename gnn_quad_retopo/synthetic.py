"""Synthetic meshes for quad-extractor tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .coords import yup_to_blender_zup
from .cross_field import compute_cross_field


def make_primitive(kind: str) -> trimesh.Trimesh:
    if kind == "plane":
        return trimesh.creation.box(extents=[2, 2, 0.02])
    if kind == "cylinder":
        return trimesh.creation.cylinder(radius=1.0, height=2.0, sections=48)
    if kind == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    if kind == "torus":
        return trimesh.creation.torus(1.0, 0.35, 48, 24)
    raise ValueError(kind)


def export_primitive(kind: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = make_primitive(kind)
    field = compute_cross_field(mesh)
    obj = out_dir / f"{kind}.obj"
    m = mesh.copy()
    m.vertices = yup_to_blender_zup(mesh.vertices)
    m.export(obj)
    np.save(out_dir / f"{kind}_cross_field.npy", field)
    return obj, out_dir / f"{kind}_cross_field.npy"
