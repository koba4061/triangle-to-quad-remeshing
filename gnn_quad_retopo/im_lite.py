"""IM-lite Python extractor: orientation + position field -> marching quads."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .im_bridge_core import build_adjacency, smooth_orientation_field, smooth_position_field
from .quad_extractor import estimate_grid_h, extract_quads_from_mesh


def im_lite_extract(
    mesh: trimesh.Trimesh,
    field: np.ndarray,
    out_dir: Path,
    *,
    target_quads: int | None = None,
    target_edge_length: float | None = None,
    smooth_orient_iters: int = 0,
    smooth_pos_iters: int = 2,
    bridge_seams: bool = False,
) -> Path:
    """Position-field-aware Python extraction (fallback when native IM unavailable)."""
    if field.shape[-1] == 6:
        field = field.reshape(len(field), 2, 3)
    v1, v2 = field[:, 0].astype(np.float64).copy(), field[:, 1].astype(np.float64).copy()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    nrm = np.asarray(mesh.vertex_normals, dtype=np.float64)
    adj = build_adjacency(faces, len(verts))

    if smooth_orient_iters > 0:
        v1, v2 = smooth_orientation_field(verts, nrm, v1, adj, iterations=smooth_orient_iters)

    if target_edge_length is None:
        target_edge_length = estimate_grid_h(mesh, v1, v2, target_quads or 300_000)

    # Position field: initialize at vertex positions, smooth on cross-field lattice
    p_field = verts.copy()
    if smooth_pos_iters > 0:
        smooth_position_field(verts, nrm, v1, p_field, adj, float(target_edge_length), iterations=smooth_pos_iters)

    # Snap vertices slightly toward smoothed positions (tangential only)
    refined = np.stack([v1, v2], axis=1)
    mesh_ref = mesh.copy()
    delta = p_field - verts
    delta -= nrm * np.sum(delta * nrm, axis=1, keepdims=True)
    mesh_ref.vertices = verts + 0.15 * delta

    return extract_quads_from_mesh(
        mesh_ref, refined, Path(out_dir),
        target_quads=target_quads,
        grid_h=target_edge_length,
        bridge_seams=bridge_seams,
    )
