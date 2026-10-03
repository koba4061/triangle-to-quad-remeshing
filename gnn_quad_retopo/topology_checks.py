"""Topology validation for quad meshes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def quad_area(vertices: np.ndarray, face: np.ndarray) -> float:
    p = vertices[face]
    if len(face) == 3:
        return float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) * 0.5)
    return float(np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[1])))


def count_non_manifold_edges(faces: np.ndarray) -> int:
    edge_count: dict[tuple[int, int], int] = {}
    for quad in faces:
        for a, b in zip(quad, np.roll(quad, -1)):
            e = (min(int(a), int(b)), max(int(a), int(b)))
            edge_count[e] = edge_count.get(e, 0) + 1
    return sum(1 for c in edge_count.values() if c > 2)


def count_boundary_edges(faces: np.ndarray) -> int:
    edge_count: dict[tuple[int, int], int] = {}
    for quad in faces:
        for a, b in zip(quad, np.roll(quad, -1)):
            e = (min(int(a), int(b)), max(int(a), int(b)))
            edge_count[e] = edge_count.get(e, 0) + 1
    return sum(1 for c in edge_count.values() if c == 1)


def compute_vertex_valences(vertices: np.ndarray, faces: np.ndarray) -> dict[str, int]:
    if len(faces) == 0:
        return {}
    valences = np.bincount(faces.reshape(-1), minlength=len(vertices))
    ref_mask = np.zeros(len(vertices), dtype=bool)
    ref_mask[faces.reshape(-1)] = True
    ref_valences = valences[ref_mask]
    hist: dict[str, int] = {}
    for v in ref_valences:
        hist[str(v)] = hist.get(str(v), 0) + 1
    # Sort keys for clean output
    return {k: hist[k] for k in sorted(hist.keys(), key=int)}


def compute_max_edge_ratio(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 1.0
    p = vertices[faces]
    if p.ndim == 2:
        p = p[:, np.newaxis, :]
    p_next = np.roll(p, -1, axis=1)
    edge_lens = np.linalg.norm(p_next - p, axis=2)
    ratios = edge_lens.max(axis=1) / (edge_lens.min(axis=1) + 1e-12)
    return float(np.clip(ratios.max(), 1.0, 1e6))


QUALITY_PASS = 0.35


def _pole_ratio_from_hist(hist: dict[str, int]) -> float:
    if not hist:
        return 0.0
    total = sum(hist.values())
    if total <= 0:
        return 0.0
    poles = sum(hist.get(str(v), 0) for v in (3, 5, 6, 7, 8, 9))
    return float(poles) / float(total)


def score_quad_mesh(stats: dict, *, pass_threshold: float = QUALITY_PASS) -> tuple[float, dict]:
    """Score quad mesh quality from run_topology_checks output (lower is better)."""
    num_quads = int(stats.get("num_quads", 0))
    if num_quads <= 0:
        terms = {
            "pole_ratio": 1.0,
            "nonmanifold_ratio": 1.0,
            "hole_ratio": 1.0,
            "degenerate_ratio": 1.0,
            "stretch": 1.0,
            "quad_purity": 0.0,
        }
        return 1.0, {**terms, "passed": False, "pass_threshold": pass_threshold}

    hist = stats.get("edge_valence_hist", {})
    pole_ratio = _pole_ratio_from_hist(hist)
    non_manifold = int(stats.get("non_manifold_edges", 0))
    boundary = int(stats.get("boundary_edges", 0))
    tiny = int(stats.get("tiny_quad_count", 0))
    zero_area = int(stats.get("zero_area_quads", 0))
    max_edge_ratio = float(stats.get("max_edge_ratio", 1.0))

    nonmanifold_ratio = min(1.0, non_manifold / max(num_quads * 4, 1))
    hole_ratio = min(1.0, boundary / max(num_quads * 4, 1))
    degenerate_ratio = min(1.0, (tiny + zero_area) / num_quads)
    stretch = min(1.0, max(0.0, (max_edge_ratio - 1.0) / 4.0))
    quad_purity = 1.0 - pole_ratio

    score = (
        0.40 * pole_ratio
        + 0.25 * nonmanifold_ratio
        + 0.10 * hole_ratio
        + 0.15 * degenerate_ratio
        + 0.10 * stretch
    )
    passed = score <= pass_threshold
    terms = {
        "pole_ratio": pole_ratio,
        "nonmanifold_ratio": nonmanifold_ratio,
        "hole_ratio": hole_ratio,
        "degenerate_ratio": degenerate_ratio,
        "stretch": stretch,
        "quad_purity": quad_purity,
        "passed": passed,
        "pass_threshold": pass_threshold,
    }
    return float(score), terms


def write_quality_report(
    out_dir: Path | str,
    attempts: list[dict],
    *,
    selected: int,
    pass_threshold: float = QUALITY_PASS,
) -> Path:
    """Write quad_quality.json with a shared schema for all extractors."""
    path = Path(out_dir) / "quad_quality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pass_threshold": pass_threshold,
        "selected_attempt": selected,
        "attempts": attempts,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def run_topology_checks(vertices: np.ndarray, faces: np.ndarray, h: float) -> dict:
    if len(faces) == 0:
        return {"num_vertices": len(vertices), "num_quads": 0}
    referenced = np.zeros(len(vertices), dtype=bool)
    referenced[faces.reshape(-1)] = True
    areas = np.array([quad_area(vertices, q) for q in faces])
    tiny = areas < 0.02 * h * h
    return {
        "num_vertices": len(vertices),
        "num_quads": len(faces),
        "unreferenced_vertices": int((~referenced).sum()),
        "non_manifold_edges": count_non_manifold_edges(faces),
        "boundary_edges": count_boundary_edges(faces),
        "tiny_quad_count": int(tiny.sum()),
        "zero_area_quads": int((areas < 1e-12).sum()),
        "min_area": float(areas.min()),
        "max_area": float(areas.max()),
        "max_edge_ratio": compute_max_edge_ratio(vertices, faces),
        "edge_valence_hist": compute_vertex_valences(vertices, faces),
    }
