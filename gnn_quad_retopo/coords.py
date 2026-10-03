"""glTF Y-up → Blender Z-up（BlenderのglTFインポートと同じ）"""
from __future__ import annotations

import numpy as np


def yup_to_blender_zup(points: np.ndarray) -> np.ndarray:
    """(x,y,z) Y-up → Blender Z-up: (x, -z, y)"""
    p = np.asarray(points, dtype=np.float64)
    out = np.empty_like(p)
    out[:, 0] = p[:, 0]
    out[:, 1] = -p[:, 2]
    out[:, 2] = p[:, 1]
    return out


def blender_zup_to_yup(points: np.ndarray) -> np.ndarray:
    """Blender Z-up → Y-up（yup_to_blender_zup の逆）"""
    p = np.asarray(points, dtype=np.float64)
    out = np.empty_like(p)
    out[:, 0] = p[:, 0]
    out[:, 1] = p[:, 2]
    out[:, 2] = -p[:, 1]
    return out


def align_vertices_to_reference_yup(
    vertices: np.ndarray,
    ref_positions: np.ndarray,
    *,
    sample: int = 4096,
) -> np.ndarray:
    """Undo an extra Z-up→Y-up flip when verts already match GLB Y-up."""
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=np.float64)
    ref_positions = np.asarray(ref_positions, dtype=np.float64)
    if v.size == 0 or ref_positions.size == 0:
        return v
    n = min(int(sample), len(v))
    pick = np.linspace(0, len(v) - 1, n, dtype=np.int64)
    vs = v[pick]
    tree = cKDTree(ref_positions)
    d0 = float(tree.query(vs)[0].mean())
    vs1 = yup_to_blender_zup(vs)
    d1 = float(tree.query(vs1)[0].mean())
    if d1 + 1e-4 < d0:
        return yup_to_blender_zup(v)
    return v
