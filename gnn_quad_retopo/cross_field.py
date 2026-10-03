"""Cross field on mesh + Blender line visualization export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .config import BLENDER_ZUP_EXPORT, FIELD_ROTATE_AXIS, FIELD_ROTATE_DEG, UP_AXIS
from .coords import yup_to_blender_zup
from .debug_log import DebugLog


def _up_vector(axis: str) -> np.ndarray:
    axis = axis.upper()
    if axis == "X":
        return np.array([1.0, 0.0, 0.0])
    if axis == "Z":
        return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 1.0, 0.0])  # Y-up (glTF / VRoid)


def stable_frame(normals: np.ndarray, up_axis: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """v2 = 体の上方向（up）を接平面に投影, v1 = n × v2（90°回転した横方向）"""
    up_axis = up_axis or UP_AXIS
    up = _up_vector(up_axis)
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    v2 = up - np.einsum("ij,ij->i", np.broadcast_to(up, n.shape), n)[:, None] * n
    ln = np.linalg.norm(v2, axis=1)
    bad = ln < 1e-6
    if np.any(bad):
        ref = _up_vector("X" if up_axis == "Y" else "Y")
        v2[bad] = ref - (ref * n[bad]).sum(1, keepdims=True) * n[bad]
        ln = np.linalg.norm(v2, axis=1)
    v2 /= ln[:, None] + 1e-12
    v1 = np.cross(n, v2)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12
    return v1, v2


# Numbaによる並列JITコンパイル高速化バックエンドの定義
try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

if USE_NUMBA:
    @njit(fastmath=True)
    def _curv_row(i, vertices, normals, neighbor_flat, neighbor_offsets, pd1, pd2, k1, k2):
            n = normals[i]
            if abs(n[0]) < 0.9:
                ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            
            u = np.cross(n, ref)
            u_norm = np.linalg.norm(u) + 1e-12
            u = u / u_norm
            v = np.cross(n, u)
            v_norm = np.linalg.norm(v) + 1e-12
            v = v / v_norm

            start = neighbor_offsets[i]
            end = neighbor_offsets[i + 1]
            nb_len = end - start

            if nb_len < 3:
                pd1[i] = u
                pd2[i] = v
                return

            diffs = np.zeros((nb_len, 3), dtype=np.float64)
            for k in range(nb_len):
                nb_idx = neighbor_flat[start + k]
                diffs[k, 0] = vertices[nb_idx, 0] - vertices[i, 0]
                diffs[k, 1] = vertices[nb_idx, 1] - vertices[i, 1]
                diffs[k, 2] = vertices[nb_idx, 2] - vertices[i, 2]

            xs = np.zeros(nb_len, dtype=np.float64)
            ys = np.zeros(nb_len, dtype=np.float64)
            zs = np.zeros(nb_len, dtype=np.float64)
            for k in range(nb_len):
                xs[k] = diffs[k, 0] * u[0] + diffs[k, 1] * u[1] + diffs[k, 2] * u[2]
                ys[k] = diffs[k, 0] * v[0] + diffs[k, 1] * v[1] + diffs[k, 2] * v[2]
                zs[k] = diffs[k, 0] * n[0] + diffs[k, 1] * n[1] + diffs[k, 2] * n[2]

            A = np.zeros((nb_len, 3), dtype=np.float64)
            for k in range(nb_len):
                A[k, 0] = 0.5 * xs[k] * xs[k]
                A[k, 1] = xs[k] * ys[k]
                A[k, 2] = 0.5 * ys[k] * ys[k]

            AtA = np.zeros((3, 3), dtype=np.float64)
            AtB = np.zeros(3, dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    val = 0.0
                    for k in range(nb_len):
                        val += A[k, r] * A[k, c]
                    AtA[r, c] = val
                val_b = 0.0
                for k in range(nb_len):
                    val_b += A[k, r] * zs[k]
                AtB[r] = val_b

            AtA[0, 0] += 1e-10
            AtA[1, 1] += 1e-10
            AtA[2, 2] += 1e-10

            det = (
                AtA[0, 0] * (AtA[1, 1] * AtA[2, 2] - AtA[1, 2] * AtA[2, 1])
                - AtA[0, 1] * (AtA[1, 0] * AtA[2, 2] - AtA[1, 2] * AtA[2, 0])
                + AtA[0, 2] * (AtA[1, 0] * AtA[2, 1] - AtA[1, 1] * AtA[2, 0])
            )
            if abs(det) < 1e-12:
                pd1[i] = u
                pd2[i] = v
                return
            a = (
                AtB[0] * (AtA[1, 1] * AtA[2, 2] - AtA[1, 2] * AtA[2, 1])
                - AtA[0, 1] * (AtB[1] * AtA[2, 2] - AtA[1, 2] * AtB[2])
                + AtA[0, 2] * (AtB[1] * AtA[2, 1] - AtA[1, 1] * AtB[2])
            ) / det
            b = (
                AtA[0, 0] * (AtB[1] * AtA[2, 2] - AtA[1, 2] * AtB[2])
                - AtB[0] * (AtA[1, 0] * AtA[2, 2] - AtA[1, 2] * AtA[2, 0])
                + AtA[0, 2] * (AtA[1, 0] * AtB[2] - AtB[1] * AtA[2, 0])
            ) / det
            c = (
                AtA[0, 0] * (AtA[1, 1] * AtB[2] - AtB[1] * AtA[2, 1])
                - AtA[0, 1] * (AtA[1, 0] * AtB[2] - AtB[1] * AtA[2, 0])
                + AtB[0] * (AtA[1, 0] * AtA[2, 1] - AtA[1, 1] * AtA[2, 0])
            ) / det

            H = np.zeros((2, 2), dtype=np.float64)
            H[0, 0] = a
            H[0, 1] = b
            H[1, 0] = b
            H[1, 1] = c
            
            # 手動で 2x2 対称行列の固有値・固有ベクトルを解析的に求解し、爆速化
            trace = H[0, 0] + H[1, 1]
            gap = H[0, 0] - H[1, 1]
            disc = np.sqrt(gap * gap + 4.0 * H[0, 1] * H[0, 1])
            l1 = 0.5 * (trace + disc)
            l2 = 0.5 * (trace - disc)

            # 固有ベクトル算出
            if abs(H[0, 1]) > 1e-12:
                v1_2d = np.array([l1 - H[1, 1], H[0, 1]], dtype=np.float64)
                v2_2d = np.array([l2 - H[1, 1], H[0, 1]], dtype=np.float64)
            else:
                v1_2d = np.array([1.0, 0.0], dtype=np.float64)
                v2_2d = np.array([0.0, 1.0], dtype=np.float64)

            pd1_vec = v1_2d[0] * u + v1_2d[1] * v
            pd2_vec = v2_2d[0] * u + v2_2d[1] * v

            pd1_norm = np.linalg.norm(pd1_vec) + 1e-12
            pd2_norm = np.linalg.norm(pd2_vec) + 1e-12

            pd1[i] = pd1_vec / pd1_norm
            pd2[i] = pd2_vec / pd2_norm
            k1[i] = l1
            k2[i] = l2

    @njit(parallel=True, fastmath=True)
    def _estimate_principal_curvatures_numba(
        vertices, normals, neighbor_flat, neighbor_offsets
    ):
        n_verts = len(vertices)
        pd1 = np.zeros((n_verts, 3), dtype=np.float64)
        pd2 = np.zeros((n_verts, 3), dtype=np.float64)
        k1 = np.zeros(n_verts, dtype=np.float64)
        k2 = np.zeros(n_verts, dtype=np.float64)
        for i in prange(n_verts):
            _curv_row(i, vertices, normals, neighbor_flat, neighbor_offsets, pd1, pd2, k1, k2)
        return pd1, pd2, k1, k2


def estimate_principal_curvatures(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1-ring近傍の局所二次曲面フィッティングを用いて、各頂点の主曲率と主方向を極めて安定かつ超高速に算出する。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    n_verts = len(vertices)

    if USE_NUMBA:
        # Trimesh の vertex_neighbors リストのリストを、Numba が高速解釈できるフラットな配列へ展開
        neighbor_offsets = np.zeros(n_verts + 1, dtype=np.int64)
        
        # プロパティ評価を1回のみにすることで、63万回のゲッター呼び出しオーバーヘッドを完全抹殺
        neighbors = mesh.vertex_neighbors
        for i in range(n_verts):
            neighbor_offsets[i + 1] = neighbor_offsets[i] + len(neighbors[i])
        
        # 1つのフラット配列に一括結合
        neighbor_flat = np.concatenate(neighbors).astype(np.int64)
        
        # Numba JIT コンパイル並列実行バックエンドを呼び出す
        return _estimate_principal_curvatures_numba(
            vertices, normals, neighbor_flat, neighbor_offsets
        )

    # Numba が未インストールの環境におけるフォールバック（従来の挙動を完璧に保証）
    pd1 = np.zeros((n_verts, 3), dtype=np.float64)
    pd2 = np.zeros((n_verts, 3), dtype=np.float64)
    k1 = np.zeros(n_verts, dtype=np.float64)
    k2 = np.zeros(n_verts, dtype=np.float64)

    for i in range(n_verts):
        n = normals[i]
        if abs(n[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, ref)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(n, u)
        v /= np.linalg.norm(v) + 1e-12

        nb_indices = mesh.vertex_neighbors[i]
        if len(nb_indices) < 3:
            pd1[i] = u
            pd2[i] = v
            continue

        diffs = vertices[nb_indices] - vertices[i]
        xs = np.dot(diffs, u)
        ys = np.dot(diffs, v)
        zs = np.dot(diffs, n)

        A = np.stack([0.5 * xs**2, xs * ys, 0.5 * ys**2], axis=1)
        B = zs

        AtA = np.dot(A.T, A)
        AtB = np.dot(A.T, B)

        AtA[0, 0] += 1e-10
        AtA[1, 1] += 1e-10
        AtA[2, 2] += 1e-10

        try:
            w = np.linalg.solve(AtA, AtB)
            a, b, c = w
        except np.linalg.LinAlgError:
            pd1[i] = u
            pd2[i] = v
            continue

        H = np.array([[a, b], [b, c]], dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(H)
        
        d1_2d = eigvecs[:, 1]
        d2_2d = eigvecs[:, 0]

        pd1[i] = d1_2d[0] * u + d1_2d[1] * v
        pd2[i] = d2_2d[0] * u + d2_2d[1] * v
        k1[i] = eigvals[1]
        k2[i] = eigvals[0]

        pd1[i] /= np.linalg.norm(pd1[i]) + 1e-12
        pd2[i] /= np.linalg.norm(pd2[i]) + 1e-12

    return pd1, pd2, k1, k2


def curvature_frame(
    mesh: trimesh.Trimesh,
    up_axis: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """主曲率方向(pd1, pd2)とグローバルガイド(stable_frame)を適応的・連続的にブレンドする。

    NumPy 2D行列演算に完全ベクトル化し、63万頂点のループオーバーヘッドを抹殺。
    """
    s1, s2 = stable_frame(mesh.vertex_normals, up_axis=up_axis)
    pd1, pd2, k1, k2 = estimate_principal_curvatures(mesh)
    
    diff = np.abs(k1 - k2)
    # 曲率差の90パーセンタイルを頑健な局所正規化基準として動的に採用
    scale = np.percentile(diff, 90) + 1e-8
    gamma = 1.0 - np.exp(-3.0 * (diff / scale))
    gamma = np.clip(gamma, 0.0, 1.0)[:, None]

    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    # 4ウェイ回転不変性に基づく、stable_frameと主曲率方向の位相整合を一括ベクトル計算
    cos1 = np.einsum("ij,ij->i", pd1, s1)
    cos2 = np.einsum("ij,ij->i", pd1, s2)

    # アライメント判定マスク
    align_mask = np.abs(cos1) >= np.abs(cos2)
    
    # cosの値の符号に基づいて、主曲率の第一方向(c1)か第二方向(c2)を正逆に選択
    c1_choice = np.where(cos1[:, None] > 0, pd1, -pd1)
    c2_choice = np.where(cos2[:, None] > 0, pd2, -pd2)
    c1_align = np.where(align_mask[:, None], c1_choice, c2_choice)

    # 整合された第2方向の計算
    c2_align = np.cross(normals, c1_align)
    c2_align /= np.linalg.norm(c2_align, axis=1, keepdims=True) + 1e-12

    # ブレンディング
    v1_out = (1.0 - gamma) * s1 + gamma * c1_align
    v2_out = (1.0 - gamma) * s2 + gamma * c2_align

    # L0相当の正規化・直交化投影処理
    v1_out = v1_out - np.einsum("ij,ij->i", v1_out, normals)[:, None] * normals
    v1_out /= np.linalg.norm(v1_out, axis=1, keepdims=True) + 1e-12
    v2_out = np.cross(normals, v1_out)
    v2_out /= np.linalg.norm(v2_out, axis=1, keepdims=True) + 1e-12

    return v1_out, v2_out


def _rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    axis = axis.upper()
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotate_vectors(vectors: np.ndarray, axis: str, degrees: float) -> np.ndarray:
    if abs(degrees) < 1e-6:
        return vectors
    return vectors @ _rotation_matrix(axis, degrees).T


def _reproject_to_tangent(v1: np.ndarray, v2: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    v1p = v1 - np.einsum("ij,ij->i", v1, n)[:, None] * n
    v1p /= np.linalg.norm(v1p, axis=1, keepdims=True) + 1e-12
    v2p = np.cross(n, v1p)
    v2p /= np.linalg.norm(v2p, axis=1, keepdims=True) + 1e-12
    return v1p, v2p


def apply_axis_rotation(
    v1: np.ndarray,
    v2: np.ndarray,
    normals: np.ndarray,
    axis: str,
    degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(degrees) < 1e-6:
        return v1, v2
    v1r = _rotate_vectors(v1, axis, degrees)
    v2r = _rotate_vectors(v2, axis, degrees)
    return _reproject_to_tangent(v1r, v2r, normals)


def compute_cross_field(
    mesh: trimesh.Trimesh,
    up_axis: str | None = None,
    rotate_axis: str | None = None,
    rotate_deg: float | None = None,
    use_curvature: bool = False,
) -> np.ndarray:
    if use_curvature:
        v1, v2 = curvature_frame(mesh, up_axis=up_axis)
    else:
        v1, v2 = stable_frame(mesh.vertex_normals, up_axis=up_axis)
    ax = rotate_axis or FIELD_ROTATE_AXIS
    deg = FIELD_ROTATE_DEG if rotate_deg is None else rotate_deg
    v1, v2 = apply_axis_rotation(v1, v2, mesh.vertex_normals, ax, deg)
    return np.stack([v1, v2], axis=1)


def export_field_lines(
    mesh: trimesh.Trimesh,
    field: np.ndarray,
    out_path: Path,
    *,
    stride: int = 80,
    scale: float | None = None,
    draw_v2: bool = True,
) -> Path:
    """Blender用: メッシュ頂点から方向矢印（線分OBJ）."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v1 = np.asarray(field[:, 0], dtype=np.float64)
    v2 = np.asarray(field[:, 1], dtype=np.float64)
    if BLENDER_ZUP_EXPORT:
        v = yup_to_blender_zup(v)
        v1 = yup_to_blender_zup(v1)
        v2 = yup_to_blender_zup(v2)
        v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12
        v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12
    bbox = float(np.linalg.norm(v.max(0) - v.min(0)))
    if scale is None:
        scale = bbox * 0.05
    idx = np.arange(0, len(mesh.vertices), max(stride, 1))
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# cross field lines (v1=red-ish group, v2=blue-ish in Blender)\n")
        vid = 1
        for i in idx:
            p = v[i]
            q1 = p + v1[i] * scale
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
            f.write(f"v {q1[0]} {q1[1]} {q1[2]}\n")
            f.write(f"l {vid} {vid+1}\n")
            vid += 2
            if draw_v2:
                q2 = p + v2[i] * scale
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")
                f.write(f"v {q2[0]} {q2[1]} {q2[2]}\n")
                f.write(f"l {vid} {vid+1}\n")
                vid += 2
    return out_path


def run_field_stage(
    mesh: trimesh.Trimesh,
    out_dir: Path,
    stride: int = 80,
    scale: float | None = None,
    up_axis: str | None = None,
    rotate_axis: str | None = None,
    rotate_deg: float | None = None,
    use_curvature: bool = False,
) -> tuple[np.ndarray, Path]:
    log = DebugLog("cross-field", out_dir)
    ax = rotate_axis or FIELD_ROTATE_AXIS
    deg = FIELD_ROTATE_DEG if rotate_deg is None else rotate_deg
    field = compute_cross_field(mesh, up_axis=up_axis, rotate_axis=ax, rotate_deg=deg, use_curvature=use_curvature)
    from .run_layout import layout_for_stage_dir

    lay = layout_for_stage_dir(out_dir)
    lay.ensure_dirs()
    field_path = lay.cross_field_npy()
    np.save(field_path, field)
    bbox = float(np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0)))
    arrow_scale = scale if scale is not None else bbox * 0.05
    viz = export_field_lines(
        mesh, field, lay.cross_field_lines(), stride=stride, scale=arrow_scale
    )
    n_lines = (len(mesh.vertices) // max(stride, 1)) * (2 if True else 1)
    orth = np.abs((field[:, 0] * field[:, 1]).sum(1))
    log.set("blender_zup_export", BLENDER_ZUP_EXPORT)
    log.set("use_curvature", use_curvature)
    log.set("rotate_axis", ax)
    log.set("rotate_deg", deg)
    log.set("bbox_diag", bbox)
    log.set("arrow_scale", arrow_scale)
    log.set("line_segments_approx", n_lines * 2)
    log.set("max_orth_error", float(orth.max()))
    log.set("viz_stride", stride)
    log.set("viz_path", str(viz))
    log.save()
    return field, viz
