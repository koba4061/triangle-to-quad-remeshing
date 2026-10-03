"""Conservative mesh cleaning — no silent decimation."""
from __future__ import annotations

from pathlib import Path


import tempfile

import trimesh
import numpy as np

from .config import BLENDER_ZUP_EXPORT, DEGENERATE_HEIGHT, MERGE_PERCENT
from .mesh_io import (
    COORD_YUP,
    align_mesh_to_reference_yup,
    export_blender_colored_mesh,
    export_blender_mesh,
    export_yup_colored_mesh,
    export_yup_mesh,
    load_mesh_yup,
    write_coord_space_json,
)
from .debug_log import DebugLog
from .mesh_im_prep import (
    load_canonical_manifest,
    mesh_topology_stats,
    promote_canonical_copy,
    repair_mesh_for_im,
    should_skip_im_repair,
    write_canonical_manifest,
)
from .run_layout import RunLayout, layout_for_stage_dir, load_run_meta


def _ascii_cache(name: str) -> Path:
    p = Path(tempfile.gettempdir()) / "gnn_quad_retopo" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def clean_mesh(
    input_path: Path,
    out_dir: Path | RunLayout,
    *,
    light: bool = False,
) -> tuple[trimesh.Trimesh, Path]:
    layout = out_dir if isinstance(out_dir, RunLayout) else layout_for_stage_dir(out_dir)
    layout.ensure_dirs()
    log = DebugLog("mesh-cleaner", layout.root)
    log.set("input", str(input_path))

    manifest = load_canonical_manifest(layout.canonical_manifest())
    alias = bool(manifest and manifest.get("smoothed_alias_canonical"))
    try:
        canonical = layout.resolve_canonical_mesh(fallback=input_path)
    except FileNotFoundError:
        canonical = input_path
    try:
        active = layout.resolve_gnn_active_mesh(fallback=input_path)
    except FileNotFoundError:
        active = input_path
    log.set("canonical", str(canonical))
    log.set("gnn_active", str(active))
    if manifest and manifest.get("gnn_proxy_reason"):
        log.set("gnn_proxy_reason", manifest["gnn_proxy_reason"])

    load_path = active if active.is_file() else input_path
    log.step(
        f"loading {'gnn_proxy' if load_path != canonical else 'canonical'} mesh "
        f"({load_path.name})"
    )
    mesh = load_mesh_yup(load_path)
    run_meta_early = load_run_meta(layout.root)
    ref_glb = run_meta_early.get("input") or str(input_path)
    mesh = align_mesh_to_reference_yup(mesh, ref_glb)

    log.set("verts_in", len(mesh.vertices))
    log.set("faces_in", len(mesh.faces))

    pre = mesh_topology_stats(mesh)
    log.set("topology_validate", pre)
    skip, skip_reason = should_skip_im_repair(pre)
    out_path = layout.cleaned_mesh()
    repaired = False

    if skip:
        log.info(f"canonical validate-only ({skip_reason})")
        post = pre
        if alias and active.is_file():
            label = "gnn_proxy" if active.resolve() != canonical.resolve() else "canonical"
            log.step(f"cleaned_mesh = {label} ({active.name})")
            promote_canonical_copy(out_path, active)
        else:
            log.step(f"export {out_path.name} (pass-through)")
            export_yup_mesh(out_path, mesh.vertices, mesh.faces)
    else:
        if light:
            log.step("repair required: light IM-prep only")
        else:
            log.step("repair required: full IM-prep + merge")
        mesh, post = repair_mesh_for_im(mesh, light=light)
        repaired = True
        log.set("im_prep_after", post)
        if post.get("nonmanifold_edges", 0) > 0:
            log.warn(f"non-manifold edges remain: {post['nonmanifold_edges']}")
        log.step(f"export {out_path.name}")
        export_yup_mesh(out_path, mesh.vertices, mesh.faces)
        mesh = load_mesh_yup(out_path)
        mesh.remove_unreferenced_vertices()

    gnn_proxy_p = layout.gnn_proxy_mesh() if layout.gnn_proxy_mesh().is_file() else None
    proxy_topo = None
    if manifest and manifest.get("gnn_proxy_topology"):
        proxy_topo = manifest["gnn_proxy_topology"]
    elif gnn_proxy_p is not None and load_path.resolve() == gnn_proxy_p.resolve():
        proxy_topo = post
    canon_topo = manifest.get("topology", post) if manifest else post
    write_canonical_manifest(
        layout.root,
        layout.canonical_manifest(),
        canonical if canonical.is_file() else out_path,
        canon_topo,
        role="cleaned",
        repaired=repaired,
        smoothed_alias=alias and not repaired,
        active_path=out_path,
        gnn_proxy_path=gnn_proxy_p,
        gnn_proxy_topology=proxy_topo,
        gnn_proxy_reason=manifest.get("gnn_proxy_reason") if manifest else None,
    )

    vcols: np.ndarray | None = None
    has_vcol = False
    try:
        from .features_cache import (
            features_cache_key,
            save_features_cache,
            try_load_features_cache,
        )
        from .vertex_color import resolve_feature_vertex_colors

        ref_paths: list[tuple[str, Path]] = []
        run_meta = load_run_meta(layout.root)
        original = run_meta.get("input")
        if original:
            ref_paths.append(("original", Path(original)))
        if canonical.is_file():
            ref_paths.append(("canonical", canonical))
        feat_out = layout.features_npz()
        mesh = _drop_degenerate_faces(mesh, log)
        vcols, has_vcol, color_source = resolve_feature_vertex_colors(
            mesh,
            ref_paths,
            target_path=load_path,
            target_space=COORD_YUP,
        )
        log.set("vertex_color_source", color_source, quiet=True)
        cache_key = features_cache_key(
            Path(original) if original else None,
            load_path,
            n_verts=len(mesh.vertices),
            n_faces=len(mesh.faces),
            light_clean=light,
        )
        use_cache = try_load_features_cache(cache_key, feat_out)
        if use_cache:
            log.set("features_cache", "hit", quiet=True)
        else:
            save_features(
                mesh,
                layout.field_dir,
                log=log,
                features_path=feat_out,
                vertex_colors_override=vcols if has_vcol else None,
                has_vertex_colors_override=has_vcol if has_vcol else None,
            )
            save_features_cache(
                cache_key, feat_out, n_verts=len(mesh.vertices), n_faces=len(mesh.faces),
            )
        if has_vcol and vcols is not None and len(vcols) == len(mesh.vertices):
            export_yup_colored_mesh(out_path, mesh.vertices, mesh.faces, vcols)
            log.info(f"embedded vertex colors in {out_path.name} (pipeline Y-up)")
    except Exception as e:
        log.warn(f"save_features failed: {e}")

    if BLENDER_ZUP_EXPORT:
        bpy_path = layout.cleaned_mesh_blender()
        if has_vcol and vcols is not None and len(vcols) == len(mesh.vertices):
            export_blender_colored_mesh(bpy_path, mesh.vertices, mesh.faces, vcols)
        else:
            export_blender_mesh(bpy_path, mesh.vertices, mesh.faces)
    write_coord_space_json(layout.prep_dir, COORD_YUP)
    write_coord_space_json(layout.field_dir, COORD_YUP)
    log.set("verts_out", len(mesh.vertices))
    log.set("faces_out", len(mesh.faces))
    log.done("mesh-cleaner")
    log.save()
    return mesh, out_path


def compute_feature_mask(mesh: trimesh.Trimesh, min_angle_deg: float = 40.0) -> np.ndarray:
    face_adjacency = mesh.face_adjacency
    face_adjacency_edges = mesh.face_adjacency_edges
    
    n0 = mesh.face_normals[face_adjacency[:, 0]]
    n1 = mesh.face_normals[face_adjacency[:, 1]]
    
    cos_d = np.einsum("ij,ij->i", n0, n1)
    cos_threshold = np.cos(np.radians(min_angle_deg))
    
    sharp_adj_mask = cos_d < cos_threshold
    sharp_edges = face_adjacency_edges[sharp_adj_mask]
    
    feature_mask = np.zeros(len(mesh.vertices), dtype=np.float32)
    if len(sharp_edges) > 0:
        feature_mask[sharp_edges.reshape(-1)] = 1.0
    return feature_mask


def compute_feature_tangents(mesh: trimesh.Trimesh, min_angle_deg: float = 40.0) -> np.ndarray:
    """鋭角エッジ（特徴線）の接線方向。頂点ごとに単位ベクトル。"""
    face_adjacency = mesh.face_adjacency
    face_adjacency_edges = mesh.face_adjacency_edges
    n0 = mesh.face_normals[face_adjacency[:, 0]]
    n1 = mesh.face_normals[face_adjacency[:, 1]]
    cos_d = np.einsum("ij,ij->i", n0, n1)
    cos_threshold = np.cos(np.radians(min_angle_deg))
    sharp_edges = face_adjacency_edges[cos_d < cos_threshold]

    tangents = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
    if len(sharp_edges) == 0:
        return tangents

    pos = np.asarray(mesh.vertices, dtype=np.float32)
    diff = pos[sharp_edges[:, 1]] - pos[sharp_edges[:, 0]]
    norms = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
    t = diff / norms
    np.add.at(tangents, sharp_edges[:, 0], t)
    np.add.at(tangents, sharp_edges[:, 1], t)
    t_norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    tangents /= t_norms
    return tangents


def compute_boundary_mask_and_tangents(
    mesh: trimesh.Trimesh,
    log: DebugLog | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    
    boundary_mask = np.zeros(len(mesh.vertices), dtype=np.float32)
    boundary_tangents = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
    
    if len(boundary_edges) > 0:
        boundary_mask[boundary_edges.reshape(-1)] = 1.0
        
        pos = np.asarray(mesh.vertices, dtype=np.float32)
        diff = pos[boundary_edges[:, 1]] - pos[boundary_edges[:, 0]]
        norms = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
        t = diff / norms
        
        np.add.at(boundary_tangents, boundary_edges[:, 0], t)
        np.add.at(boundary_tangents, boundary_edges[:, 1], t)
        
        t_norms = np.linalg.norm(boundary_tangents, axis=1, keepdims=True) + 1e-8
        boundary_tangents /= t_norms
        
    # 境界からのDijkstra(BFS)ステップ数の算出 - SciPy C++ バックエンドによる超高速化
    n_verts = len(mesh.vertices)
    boundary_steps = np.full(n_verts, 999.0, dtype=np.float32)
    boundary_indices = np.where(boundary_mask > 0.5)[0]
    
    if len(boundary_indices) > 0:
        if log:
            log.step(
                f"boundary BFS depth<=30 (sources={len(boundary_indices):,}, verts={n_verts:,})"
            )
        from scipy.sparse import csr_matrix

        # 重みなしなので Dijkstra の (境界数 x 頂点数) 行列は作らない。30段の幅優先だけ。
        e_uniq = np.asarray(mesh.edges_unique)
        row = np.concatenate([e_uniq[:, 0], e_uniq[:, 1]])
        col = np.concatenate([e_uniq[:, 1], e_uniq[:, 0]])
        adj = csr_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)),
            shape=(n_verts, n_verts),
        )
        depth = np.full(n_verts, np.int16(999), dtype=np.int16)
        depth[boundary_indices] = 0
        front = np.zeros(n_verts, dtype=np.float32)
        front[boundary_indices] = 1.0
        for step in range(1, 31):
            nxt = adj @ front
            newly = (nxt > 0) & (depth == 999)
            if not np.any(newly):
                break
            depth[newly] = step
            front = newly.astype(np.float32)
        boundary_steps = depth.astype(np.float32)
        del adj, front
        if log:
            log.info(f"boundary BFS done (sources={len(boundary_indices):,})")

    return boundary_mask, boundary_tangents, boundary_steps


def compute_symmetry_pairs(mesh: trimesh.Trimesh, tolerance: float = 0.01) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    reflected = vertices.copy()
    reflected[:, 0] = -reflected[:, 0]

    from scipy.spatial import cKDTree
    tree = cKDTree(vertices)
    dist, idx = tree.query(reflected, distance_upper_bound=tolerance)

    invalid = ~np.isfinite(dist)
    idx[invalid] = np.arange(len(vertices))[invalid]

    return idx.astype(np.int64)


def compute_symmetry_confidence(
    mesh: trimesh.Trimesh,
    symmetry_pairs: np.ndarray,
    *,
    k1: np.ndarray | None = None,
    k2: np.ndarray | None = None,
    tolerance: float = 0.01,
    sigma_dist: float = 0.01,
    sigma_curv: float = 0.5,
) -> np.ndarray:
    """X=0 鏡面の対称ペア信頼度 c_sym。位置・法線に加えて主曲率の整合性を組み込んで強化。"""
    pos = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    j = symmetry_pairs
    n_v = len(pos)

    # 1. 位置の鏡像対称性チェック
    pos_partner = pos[j]
    pos_refl = pos_partner.copy()
    pos_refl[:, 0] *= -1.0
    dist = np.linalg.norm(pos - pos_refl, axis=1)
    c_dist = np.exp(-((dist / max(sigma_dist, 1e-6)) ** 2))

    # 2. 法線の鏡像対称性チェック
    n_partner = normals[j]
    n_refl = n_partner.copy()
    n_refl[:, 0] *= -1.0
    c_norm = np.maximum(0.0, np.einsum("ij,ij->i", normals, n_refl))

    # 3. 主曲率の鏡像整合性チェック（非対称な突起やアクセサリーの除去）
    if k1 is None or k2 is None:
        from .cross_field import estimate_principal_curvatures
        _, _, k1_t, k2_t = estimate_principal_curvatures(mesh)
        k1 = k1_t.astype(np.float32)
        k2 = k2_t.astype(np.float32)
    else:
        k1 = k1.astype(np.float32)
        k2 = k2.astype(np.float32)

    k1_partner = k1[j]
    k2_partner = k2[j]
    curv_diff = (k1 - k1_partner) ** 2 + (k2 - k2_partner) ** 2
    c_curv = np.exp(-(curv_diff / max(sigma_curv ** 2, 1e-6)))

    c_sym = (c_dist * c_norm * c_curv).astype(np.float32)

    # 未対応・非対称領域は重みゼロ
    self_pair = j == np.arange(n_v)
    c_sym[self_pair & (dist > tolerance)] = 0.0
    c_sym[dist > tolerance * 3.0] = 0.0
    
    # 中央線 (X=0) 付近の頂点の特別な処理
    near_center = np.abs(pos[:, 0]) < tolerance
    c_sym[self_pair & near_center] = 1.0
    
    return c_sym


def compute_vertex_normal_variance(mesh: trimesh.Trimesh) -> np.ndarray:
    """頂点周辺の面法線ばらつきを一括行列演算（NumPy ベクトル化）で超高速算出。

    従来の 63万回 Python ループ ➔ 0回へ。
    """
    n_v = len(mesh.vertices)
    vn = np.asarray(mesh.vertex_normals, dtype=np.float32)
    fn = np.asarray(mesh.face_normals, dtype=np.float32)
    try:
        vf = np.asarray(mesh.vertex_faces, dtype=np.int64)
    except Exception:
        # trimesh can fail to build vertex_faces on meshes with degenerate faces.
        return np.zeros(n_v, dtype=np.float32)

    # 有効な面（>= 0）のマスク
    valid_mask = vf >= 0
    counts = valid_mask.sum(axis=1)

    # fnの末尾にダミーのゼロベクトルを追加し、負のインデックス(-1)が自動的にダミーを参照するようにする
    fn_padded = np.vstack([fn, np.zeros((1, 3), dtype=np.float32)])
    
    # 全頂点の隣接面法線を一括取得 (n_v, cols, 3)
    adj_fns = fn_padded[vf]

    # 頂点法線と隣接面法線の内積を一括バッチ計算
    dots = np.abs(np.einsum("vi,vci->vc", vn, adj_fns))

    # 有効な面のドット積平均を求める
    dots_sum = (dots * valid_mask).sum(axis=1)
    dots_mean = np.zeros(n_v, dtype=np.float32)
    
    # 隣接面が 2 つ以上ある頂点のみ計算対象とする
    ok = counts >= 2
    dots_mean[ok] = dots_sum[ok] / counts[ok]

    # ばらつきを [0, 1] の範囲で算出してクリップ
    out = np.clip(1.0 - dots_mean, 0.0, 1.0).astype(np.float32)
    out[counts < 2] = 0.0  # 近傍面数が不足している頂点は 0.0 とする

    return out


def extract_vertex_colors(mesh: trimesh.Trimesh) -> tuple[np.ndarray, bool]:
    from .vertex_color import extract_vertex_colors as _extract

    return _extract(mesh)


def _drop_degenerate_faces(mesh: trimesh.Trimesh, log: DebugLog | None = None) -> trimesh.Trimesh:
    """Drop degenerate faces before feature extraction."""
    try:
        mask = mesh.nondegenerate_faces(height=DEGENERATE_HEIGHT)
        if mask is not None and len(mask) == len(mesh.faces) and (not bool(mask.all())):
            mesh.update_faces(mask)
            mesh.remove_unreferenced_vertices()
            mesh.process(validate=True)
            if log:
                log.warn("features: dropped degenerate faces before extraction")
    except Exception:
        pass
    return mesh


def save_features(
    mesh: trimesh.Trimesh,
    out_dir: Path,
    log: DebugLog | None = None,
    *,
    features_path: Path | None = None,
    vertex_colors_override: np.ndarray | None = None,
    has_vertex_colors_override: bool | None = None,
    ref_paths: list[tuple[str, Path]] | None = None,
    target_path: Path | str | None = None,
    target_space: str | None = None,
) -> None:
    from .cross_field import estimate_principal_curvatures

    mesh = _drop_degenerate_faces(mesh, log)
    if log:
        log.step("features: principal curvature")
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    pd1, pd2, k1, k2 = estimate_principal_curvatures(mesh)

    if log:
        log.step("features: sharp edges + boundary")
    feature_mask = compute_feature_mask(mesh)
    feature_tangents = compute_feature_tangents(mesh)
    boundary_mask, boundary_tangents, boundary_steps = compute_boundary_mask_and_tangents(mesh, log=log)

    n_v = len(mesh.vertices)
    if log:
        log.step("features: symmetry pairs")
    try:
        symmetry_pairs = compute_symmetry_pairs(mesh)
        c_sym = compute_symmetry_confidence(mesh, symmetry_pairs, k1=k1, k2=k2)
    except Exception as e:
        if log:
            log.warn(f"symmetry features skipped ({e}); using identity pairs")
        symmetry_pairs = np.arange(n_v, dtype=np.int64)
        c_sym = np.zeros(n_v, dtype=np.float32)
    try:
        normal_variance = compute_vertex_normal_variance(mesh)
    except Exception as e:
        normal_variance = np.zeros(n_v, dtype=np.float32)
        if log:
            log.warn(f"normal_variance skipped ({e})")
    if (
        has_vertex_colors_override
        and vertex_colors_override is not None
        and len(vertex_colors_override) == n_v
    ):
        vertex_colors = np.asarray(vertex_colors_override, dtype=np.float32)
        has_vertex_colors = True
        color_source = "override"
    elif ref_paths:
        from .vertex_color import resolve_feature_vertex_colors

        vertex_colors, has_vertex_colors, color_source = resolve_feature_vertex_colors(
            mesh,
            ref_paths,
            target_path=target_path,
            target_space=target_space,
        )
        if log and not has_vertex_colors:
            log.warn(
                "vertex colors unavailable for GNN guide; "
                "check input GLB COLOR_0 or coord_space"
            )
    else:
        vertex_colors, has_vertex_colors = extract_vertex_colors(mesh)
        color_source = "active" if has_vertex_colors else "none"
    feat_out = features_path or (out_dir / "features.npz")
    np.savez_compressed(
        feat_out,
        normals=normals,
        pd1=pd1.astype(np.float32),
        pd2=pd2.astype(np.float32),
        k1=k1.astype(np.float32),
        k2=k2.astype(np.float32),
        feature_mask=feature_mask,
        feature_tangents=feature_tangents,
        boundary_mask=boundary_mask,
        boundary_tangents=boundary_tangents,
        boundary_steps=boundary_steps,
        symmetry_pairs=symmetry_pairs,
        c_sym=c_sym,
        normal_variance=normal_variance,
        vertex_colors=vertex_colors,
        has_vertex_colors=np.array(has_vertex_colors, dtype=np.bool_),
    )
    if log:
        n_feat = int((feature_mask > 0.5).sum())
        n_bound = int((boundary_mask > 0.5).sum())
        n_sym = int((c_sym > 0.1).sum())
        log.set("feature_verts", n_feat, quiet=True)
        log.set("boundary_verts", n_bound, quiet=True)
        log.set("sym_confident_verts", n_sym, quiet=True)
        log.set("has_vertex_colors", has_vertex_colors, quiet=True)
        if has_vertex_colors_override is not None:
            log.set("vertex_colors_override", bool(has_vertex_colors_override), quiet=True)
        log.set("vertex_color_source", color_source, quiet=True)
        log.info(
            f"features.npz saved | feature={n_feat:,} boundary={n_bound:,} sym={n_sym:,} "
            f"vertex_colors={has_vertex_colors}"
        )
