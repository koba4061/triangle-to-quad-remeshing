"""pyinstantmeshes wrapper (pip wheel). Orientation teacher + baseline remesh."""
from __future__ import annotations

import numpy as np
import trimesh

_PYIM = None
_PYIM_ERR: str | None = None


def _load():
    global _PYIM, _PYIM_ERR
    if _PYIM is not None or _PYIM_ERR is not None:
        return _PYIM
    try:
        import pyinstantmeshes as pim  # type: ignore

        _PYIM = pim
        return pim
    except ImportError as e:
        _PYIM_ERR = str(e)
        return None


def is_pyim_available() -> bool:
    return _load() is not None


def pyim_status() -> str:
    if _load() is not None:
        return "pyinstantmeshes"
    return f"unavailable ({_PYIM_ERR})"


def remesh_mesh(
    mesh: trimesh.Trimesh,
    *,
    target_vertex_count: int = 2000,
    crease_angle: float = 40.0,
    align_to_boundaries: bool = True,
    pure_quad: bool = True,
) -> trimesh.Trimesh:
    pim = _load()
    if pim is None:
        raise ImportError("pyinstantmeshes not installed: pip install pyinstantmeshes")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    nv, nf = pim.remesh(
        v, f,
        target_vertex_count=target_vertex_count,
        crease_angle=crease_angle,
        align_to_boundaries=align_to_boundaries,
        smooth_iterations=2,
        pure_quad=pure_quad,
        deterministic=True,
    )
    if nf.shape[1] == 4:
        return trimesh.Trimesh(vertices=nv, faces=nf, process=True)
    return trimesh.Trimesh(vertices=nv, faces=nf, process=True)


def teacher_field_via_pyim(
    mesh: trimesh.Trimesh,
    *,
    target_vertex_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """IM remesh -> curvature cross field -> NN transfer to original vertices."""
    from .cross_field import compute_cross_field

    n = len(mesh.vertices)
    tgt = target_vertex_count or max(512, min(n, 4096))
    remeshed = remesh_mesh(mesh, target_vertex_count=tgt)
    field = compute_cross_field(remeshed, use_curvature=True).reshape(len(remeshed.vertices), 2, 3)
    rv = np.asarray(remeshed.vertices, dtype=np.float64)
    qv = np.asarray(mesh.vertices, dtype=np.float64)
    tree = trimesh.proximity.ProximityQuery(remeshed)
    _, idx = tree.vertex(qv)
    v1 = field[idx, 0].astype(np.float64)
    v2 = field[idx, 1].astype(np.float64)
    nrm = np.asarray(mesh.vertex_normals, dtype=np.float64)
    v1 -= nrm * np.sum(v1 * nrm, axis=1, keepdims=True)
    v2 = np.cross(nrm, v1)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12
    return v1, v2
