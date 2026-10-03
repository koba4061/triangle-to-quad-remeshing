"""Cross field quality metrics (GNN evaluation without quad extraction)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from .cross_field import compute_cross_field


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def _symm_align_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-vertex 4-way symmetric alignment -> degrees."""
    a = _normalize_rows(a)
    b = _normalize_rows(b)
    d0 = np.abs((a * b).sum(axis=1))
    b90 = np.stack([-b[:, 1], b[:, 0], b[:, 2]], axis=1)
    d1 = np.abs((a * b90).sum(axis=1))
    cos_a = np.maximum(d0, d1)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def _cross_4way_align(v1_i: np.ndarray, v2_i: np.ndarray, v1_j: np.ndarray, v2_j: np.ndarray) -> float:
    """IM-style 4-way edge discontinuity (0=perfect, 1=orthogonal)."""
    a = [
        abs(float(np.dot(v1_i, v1_j))),
        abs(float(np.dot(v1_i, v2_j))),
        abs(float(np.dot(v2_i, v1_j))),
        abs(float(np.dot(v2_i, v2_j))),
    ]
    return 1.0 - max(a)


def evaluate_cross_field(
    mesh: trimesh.Trimesh,
    field: np.ndarray,
    *,
    features_path: Path | None = None,
    teacher_field: np.ndarray | None = None,
) -> dict:
    """Compute field quality metrics. field shape (V,2,3) or (V,6)."""
    if field.shape[-1] == 6:
        field = field.reshape(len(field), 2, 3)
    v1, v2 = field[:, 0], field[:, 1]
    v1 = _normalize_rows(v1.astype(np.float64))
    v2 = _normalize_rows(v2.astype(np.float64))
    n_v = len(mesh.vertices)

    guide = compute_cross_field(mesh, use_curvature=True).reshape(n_v, 2, 3)
    g1, g2 = guide[:, 0], guide[:, 1]

    metrics: dict = {
        "vertices": n_v,
        "ortho_error": float(np.abs((v1 * v2).sum(axis=1)).mean()),
        "guide_distill_deg_mean": float(_symm_align_deg(v1, g1).mean()),
        "guide_distill_deg_p95": float(np.percentile(_symm_align_deg(v1, g1), 95)),
    }

    feat_path = features_path
    if feat_path is None:
        feat_path = Path(".")
    npz = feat_path / "features.npz" if feat_path.is_dir() else feat_path.parent / "features.npz"
    if npz.exists():
        feat = np.load(npz)
        pd1, pd2 = feat["pd1"], feat["pd2"]
        pd_norm = np.linalg.norm(pd1, axis=1)
        valid = pd_norm > 0.1
        if valid.any():
            ang1 = _symm_align_deg(v1[valid], pd1[valid])
            ang2 = _symm_align_deg(v1[valid], pd2[valid])
            curv_deg = np.minimum(ang1, ang2)
            metrics["align_curvature_deg_mean"] = float(curv_deg.mean())
            metrics["align_curvature_deg_p95"] = float(np.percentile(curv_deg, 95))

        if "feature_tangents" in feat:
            ft = feat["feature_tangents"]
            fm = feat.get("feature_mask", np.zeros(n_v))
            mask = (fm > 0.5) & (np.linalg.norm(ft, axis=1) > 0.1)
            if mask.any():
                ft_n = _normalize_rows(ft[mask])
                ang = _symm_align_deg(v1[mask], ft_n)
                metrics["align_feature_deg_mean"] = float(ang.mean())
                metrics["align_feature_deg_p95"] = float(np.percentile(ang, 95))
                metrics["feature_vertices"] = int(mask.sum())

        if "boundary_mask" in feat:
            bm = feat["boundary_mask"] > 0.5
            if bm.any() and "boundary_tangents" in feat:
                bt = _normalize_rows(feat["boundary_tangents"][bm])
                ang = _symm_align_deg(v1[bm], bt)
                metrics["align_boundary_deg_mean"] = float(ang.mean())

    # 4-way smoothness on edges
    e = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(e) > 0:
        disc = []
        for i, j in e:
            disc.append(_cross_4way_align(v1[i], v2[i], v1[j], v2[j]))
        metrics["smoothness_4way_mean"] = float(np.mean(disc))
        metrics["smoothness_4way_p95"] = float(np.percentile(disc, 95))

    if teacher_field is not None:
        if teacher_field.shape[-1] == 6:
            teacher_field = teacher_field.reshape(n_v, 2, 3)
        t1 = _normalize_rows(teacher_field[:, 0])
        ang = _symm_align_deg(v1, t1)
        metrics["teacher_align_deg_mean"] = float(ang.mean())
        metrics["teacher_align_deg_p95"] = float(np.percentile(ang, 95))

    # Optional libigl combed cross-field reference
    try:
        import igl  # type: ignore
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64)
        if npz.exists():
            feat = np.load(npz)
            pd1, pd2 = feat["pd1"], feat["pd2"]
            if len(pd1) == n_v and float(np.linalg.norm(pd1)) > 0:
                pd1o, pd2o = igl.comb_cross_field(v, f, pd1, pd2)
                metrics["libigl_comb_deg_mean"] = float(
                    np.minimum(_symm_align_deg(v1, pd1o), _symm_align_deg(v1, pd2o)).mean()
                )
    except ImportError:
        pass
    except Exception:
        pass

    return metrics


def save_field_metrics(metrics: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def check_thresholds(metrics: dict, kind: str = "plane") -> tuple[bool, list[str]]:
    """Return (passed, failures) for synthetic patch types."""
    fails: list[str] = []
    if kind == "plane":
        ac = metrics.get("align_curvature_deg_mean")
        if ac is not None and ac >= 5.0:
            fails.append(f"align_curvature_deg_mean={ac:.2f} >= 5")
        sm = metrics.get("smoothness_4way_mean")
        if sm is not None and sm >= 0.05:
            fails.append(f"smoothness_4way_mean={sm:.4f} >= 0.05")
    elif kind == "cylinder":
        ab = metrics.get("align_boundary_deg_mean", metrics.get("align_feature_deg_mean"))
        if ab is not None and ab >= 15.0:
            fails.append("boundary/feature align >= 15 deg")
    elif kind == "box":
        af = metrics.get("align_feature_deg_mean")
        if af is not None and af >= 10.0:
            fails.append(f"align_feature_deg_mean={af:.2f} >= 10")
    return len(fails) == 0, fails
