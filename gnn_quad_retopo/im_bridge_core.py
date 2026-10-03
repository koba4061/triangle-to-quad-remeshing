"""IM-style orientation/position field helpers (Python fallback when C++ bridge not built)."""
from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n


def lattice_op(p: np.ndarray, o: np.ndarray, n: np.ndarray, target: np.ndarray, scale: float, op=np.floor) -> np.ndarray:
    t = np.cross(n, o)
    d = target - p
    return p + scale * (o * op(np.dot(o, d) / scale) + t * op(np.dot(t, d) / scale))


def compat_orientation_extrinsic_4(q0: np.ndarray, n0: np.ndarray, q1: np.ndarray, n1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4-RoSy compatible orientations (extrinsic)."""
    cands0 = [q0, np.array([-q0[1], q0[0], q0[2]])]
    cands1 = [q1, np.array([-q1[1], q1[0], q1[2]])]
    best, best_score = q0, -1.0
    for a in cands0:
        for b in cands1:
            s = abs(float(np.dot(a, b)))
            if s > best_score:
                best_score = s
                best = a
    return best, q1


def smooth_orientation_field(
    vertices: np.ndarray,
    normals: np.ndarray,
    v1: np.ndarray,
    adjacency: list[list[int]],
    *,
    iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """IM-style hierarchical orientation smoothing (Python fallback)."""
    o = _normalize(v1.copy())
    n = _normalize(normals)
    v2 = _normalize(np.cross(n, o))
    for _ in range(iterations):
        o_new = o.copy()
        for i in range(len(vertices)):
            if not adjacency[i]:
                continue
            acc = np.zeros(3, dtype=np.float64)
            for j in adjacency[i]:
                c0, _ = compat_orientation_extrinsic_4(o[i], n[i], o[j], n[j])
                acc += c0
            if np.linalg.norm(acc) > 1e-8:
                o_new[i] = acc
            o_new[i] -= n[i] * np.dot(o_new[i], n[i])
            o_new[i] /= np.linalg.norm(o_new[i]) + 1e-12
        o = o_new
        v2 = _normalize(np.cross(n, o))
    return o, v2


def smooth_position_field(
    vertices: np.ndarray,
    normals: np.ndarray,
    o_field: np.ndarray,
    p_field: np.ndarray,
    adjacency: list[list[int]],
    scale: float,
    *,
    iterations: int = 5,
) -> np.ndarray:
    """IM-style position field smoothing (resources/im.py inspired)."""
    p = p_field.copy()
    n = _normalize(normals)
    o = _normalize(o_field)
    rng = np.random.default_rng(0)
    for _ in range(iterations):
        order = rng.permutation(len(vertices))
        for i in order:
            o_i, p_i, n_i, v_i, weight = o[i], p[i], n[i], vertices[i], 0.0
            for j in adjacency[i]:
                o_j, p_j, n_j, v_j = o[j], p[j], n[j], vertices[j]
                t0 = np.cross(n_i, o_i)
                t1 = np.cross(n_j, o_j)
                middle = 0.5 * (v_i + v_j)
                p0 = lattice_op(p_i, o_i, n_i, middle, scale)
                p1 = lattice_op(p_j, o_j, n_j, middle, scale)
                best = min(
                    [(p0, p1), (p0 + scale * t0, p1), (p0, p1 + scale * t1), (p0 + scale * t0, p1 + scale * t1)],
                    key=lambda ab: np.linalg.norm(ab[0] - ab[1]),
                )
                p_i = (weight * best[0] + best[1]) / (weight + 1.0)
                p_i -= n_i * np.dot(p_i - v_i, n_i)
                weight += 1.0
            p[i] = lattice_op(p_i, o_i, n_i, v_i, scale, op=np.round)
    return p


def build_adjacency(faces: np.ndarray, n_verts: int) -> list[list[int]]:
    adj: list[set[int]] = [set() for _ in range(n_verts)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])
    return [sorted(s) for s in adj]
