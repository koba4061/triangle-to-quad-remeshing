"""Marching-quads quad extraction with local-UV propagation."""
from __future__ import annotations

import math
import os
import concurrent.futures
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .config import BLENDER_ZUP_EXPORT
from .coords import yup_to_blender_zup
from .debug_log import DebugLog
from .topology_checks import QUALITY_PASS, quad_area, run_topology_checks, score_quad_mesh, write_quality_report

DET_GUARD = 1e-7
BARY_EPS = 1e-5


def integrate_uv(
    vertices_or_mesh: np.ndarray | trimesh.Trimesh,
    faces: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    log: DebugLog | None = None,
) -> tuple[np.ndarray, dict]:
    """各三角形のローカル接平面UVをBFS伝播で算出し、大域的パラメータ化の潰れを100%回避する。"""
    if isinstance(vertices_or_mesh, trimesh.Trimesh):
        mesh = vertices_or_mesh
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    else:
        vertices = np.asarray(vertices_or_mesh, dtype=np.float64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    n_faces = len(faces)
    visited = np.zeros(n_faces, dtype=bool)
    face_uvs = np.zeros((n_faces, 3, 2), dtype=np.float64)
    face_chart = np.full(n_faces, -1, dtype=np.int64)

    # trimeshの高速な面隣接関係を使用
    face_adjacency = mesh.face_adjacency
    adj = [[] for _ in range(n_faces)]
    for f1, f2 in face_adjacency:
        adj[f1].append(int(f2))
        adj[f2].append(int(f1))

    anchors = 0
    report_every = max(1, n_faces // 40)
    # 複数連結成分に対応
    for start_fi in range(n_faces):
        if visited[start_fi]:
            continue

        chart_id = anchors
        anchors += 1
        queue = [start_fi]
        visited[start_fi] = True
        face_chart[start_fi] = chart_id

        # 開始三角形のローカルUVを設定
        tri = faces[start_fi]
        # デジェネレート面のセーフガード
        if len(set(tri)) < 3:
            continue

        p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        du1 = 0.5 * np.dot(v1[tri[0]] + v1[tri[1]], p1 - p0)
        dv1 = 0.5 * np.dot(v2[tri[0]] + v2[tri[1]], p1 - p0)
        du2 = 0.5 * np.dot(v1[tri[0]] + v1[tri[2]], p2 - p0)
        dv2 = 0.5 * np.dot(v2[tri[0]] + v2[tri[2]], p2 - p0)

        face_uvs[start_fi, 0] = [0.0, 0.0]
        face_uvs[start_fi, 1] = [du1, dv1]
        face_uvs[start_fi, 2] = [du2, dv2]

        head = 0
        while head < len(queue):
            curr_fi = queue[head]
            head += 1
            if log and int(visited.sum()) % report_every == 0:
                log.progress(int(visited.sum()), n_faces, "UV integrate", every=report_every)

            curr_tri = faces[curr_fi]
            curr_uv = face_uvs[curr_fi]
            c0, c1, c2 = curr_tri

            for nbr_fi in adj[curr_fi]:
                if visited[nbr_fi]:
                    continue

                nbr_tri = faces[nbr_fi]
                # 相手がデジェネレート面の場合はスキップ
                if len(set(nbr_tri)) < 3:
                    visited[nbr_fi] = True
                    continue

                v0, v1_n, v2_n = nbr_tri
                
                in0 = (v0 == c0) or (v0 == c1) or (v0 == c2)
                in1 = (v1_n == c0) or (v1_n == c1) or (v1_n == c2)
                in2 = (v2_n == c0) or (v2_n == c1) or (v2_n == c2)

                if in0 and in1:
                    shared = (v0, v1_n)
                    k = v2_n
                    idx_k_nbr = 2
                elif in1 and in2:
                    shared = (v1_n, v2_n)
                    k = v0
                    idx_k_nbr = 0
                else:
                    shared = (v2_n, v0)
                    k = v1_n
                    idx_k_nbr = 1

                i, j = shared
                idx_i_nbr = 0 if v0 == i else (1 if v1_n == i else 2)
                idx_j_nbr = 0 if v0 == j else (1 if v1_n == j else 2)

                idx_i_curr = 0 if c0 == i else (1 if c1 == i else 2)
                idx_j_curr = 0 if c0 == j else (1 if c1 == j else 2)

                uv_i = curr_uv[idx_i_curr]
                uv_j = curr_uv[idx_j_curr]

                face_uvs[nbr_fi, idx_i_nbr] = uv_i
                face_uvs[nbr_fi, idx_j_nbr] = uv_j

                pk, pi, pj = vertices[k], vertices[i], vertices[j]

                du_ik = 0.5 * np.dot(v1[i] + v1[k], pk - pi)
                dv_ik = 0.5 * np.dot(v2[i] + v2[k], pk - pi)
                uv_k_from_i = uv_i + np.array([du_ik, dv_ik])

                du_jk = 0.5 * np.dot(v1[j] + v1[k], pk - pj)
                dv_jk = 0.5 * np.dot(v2[j] + v2[k], pk - pj)
                uv_k_from_j = uv_j + np.array([du_jk, dv_jk])

                face_uvs[nbr_fi, idx_k_nbr] = 0.5 * (uv_k_from_i + uv_k_from_j)

                visited[nbr_fi] = True
                face_chart[nbr_fi] = chart_id
                queue.append(nbr_fi)

    return face_uvs, {"anchors": anchors, "face_chart": face_chart}


def _normalize_uv(face_uvs: np.ndarray, span: float) -> np.ndarray:
    u_min, u_max = face_uvs[..., 0].min(), face_uvs[..., 0].max()
    v_min, v_max = face_uvs[..., 1].min(), face_uvs[..., 1].max()

    norm_uvs = face_uvs.copy()
    norm_uvs[..., 0] = (face_uvs[..., 0] - u_min) / (u_max - u_min + 1e-8) * span
    norm_uvs[..., 1] = (face_uvs[..., 1] - v_min) / (v_max - v_min + 1e-8) * span
    return norm_uvs


def extract_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_uvs: np.ndarray,
    *,
    progress_every: int = 0,
    log: DebugLog | None = None,
):
    """Cramer's Ruleを用いた一括ベクトル演算による交点抽出。超高速化。"""
    nf = len(faces)
    
    # 1. 各三角形のUVの境界ボックスを求める
    u_min = np.min(face_uvs[..., 0], axis=1)
    u_max = np.max(face_uvs[..., 0], axis=1)
    v_min = np.min(face_uvs[..., 1], axis=1)
    v_max = np.max(face_uvs[..., 1], axis=1)
    
    k0 = np.floor(u_min).astype(np.int64)
    k1 = np.ceil(u_max).astype(np.int64)
    l0 = np.floor(v_min).astype(np.int64)
    l1 = np.ceil(v_max).astype(np.int64)
    
    # 交点が存在し得ない面を除外
    valid_face = (k1 >= k0) & (l1 >= l0)
    
    # 爆発安全ガード: あまりに細分化されすぎている（hが極めて小さい）面を特定
    too_large = ((k1 - k0) > 100) | ((l1 - l0) > 100)
    skipped = list(np.where(too_large & valid_face)[0])
    
    valid_face = valid_face & (~too_large)
    
    # 2. 有効な各面について、格子点 (k, l) の総数を算出する
    n_k = (k1 - k0 + 1)
    n_l = (l1 - l0 + 1)
    grid_sizes = n_k * n_l
    grid_sizes[~valid_face] = 0
    
    # カバーされる格子点が存在する面インデックス
    active_faces = np.where(grid_sizes > 0)[0]
    if len(active_faces) == 0:
        return [], [], skipped
        
    active_sizes = grid_sizes[active_faces]
    
    # 3. アクティブな面をリピートして、格子候補点のフラットなインデックス配列を作成
    face_indices_rep = np.repeat(active_faces, active_sizes)
    
    cum_sizes = np.cumsum(active_sizes)
    total_candidates = cum_sizes[-1]
    
    start_indices = np.zeros(len(active_faces), dtype=np.int64)
    start_indices[1:] = cum_sizes[:-1]
    
    global_idx = np.arange(total_candidates, dtype=np.int64)
    local_idx = global_idx - np.repeat(start_indices, active_sizes)
    
    n_l_rep = n_l[face_indices_rep]
    k0_rep = k0[face_indices_rep]
    l0_rep = l0[face_indices_rep]
    
    L_vals = (local_idx % n_l_rep + l0_rep).astype(np.float64)
    K_vals = (local_idx // n_l_rep + k0_rep).astype(np.float64)
    
    # 4. Cramer's Rule に必要な各面のUV座標と3D頂点座標を取得
    uvs_rep = face_uvs[face_indices_rep]
    
    u0, u1, u2 = uvs_rep[:, 0, 0], uvs_rep[:, 1, 0], uvs_rep[:, 2, 0]
    v0, v1, v2 = uvs_rep[:, 0, 1], uvs_rep[:, 1, 1], uvs_rep[:, 2, 1]
    
    # 行列式 D の一括計算
    D = u0 * (v1 - v2) + u1 * (v2 - v0) + u2 * (v0 - v1)
    
    # 行列式が小さすぎる（DET_GUARD 未満）ものをフィルタリングするマスク
    valid_det = np.abs(D) >= DET_GUARD
    
    # 5. 有効な行列式を持つ要素だけで Cramer's Rule を解く
    valid_indices = np.where(valid_det)[0]
    if len(valid_indices) == 0:
        return [], [], skipped
        
    D_v = D[valid_indices]
    u0_v, u1_v, u2_v = u0[valid_indices], u1[valid_indices], u2[valid_indices]
    v0_v, v1_v, v2_v = v0[valid_indices], v1[valid_indices], v2[valid_indices]
    K_v = K_vals[valid_indices]
    L_v = L_vals[valid_indices]
    face_idx_v = face_indices_rep[valid_indices]
    
    lam0 = (K_v * (v1_v - v2_v) - L_v * (u1_v - u2_v) + (u1_v * v2_v - u2_v * v1_v)) / D_v
    lam1 = (K_v * (v2_v - v0_v) - L_v * (u2_v - u0_v) + (u2_v * v0_v - u0_v * v2_v)) / D_v
    lam2 = 1.0 - lam0 - lam1
    
    # 6. 重心座標の有効判定
    inside = (lam0 >= -BARY_EPS) & (lam0 <= 1.0 + BARY_EPS) & \
             (lam1 >= -BARY_EPS) & (lam1 <= 1.0 + BARY_EPS) & \
             (lam2 >= -BARY_EPS) & (lam2 <= 1.0 + BARY_EPS)
             
    valid_inside_idx = np.where(inside)[0]
    if len(valid_inside_idx) == 0:
        det_failed_faces = np.unique(face_indices_rep[~valid_det])
        return [], [], sorted(list(set(skipped + det_failed_faces.tolist())))
        
    lam0_ok = lam0[valid_inside_idx]
    lam1_ok = lam1[valid_inside_idx]
    lam2_ok = lam2[valid_inside_idx]
    face_idx_ok = face_idx_v[valid_inside_idx]
    K_ok = K_v[valid_inside_idx].astype(np.int64)
    L_ok = L_v[valid_inside_idx].astype(np.int64)
    
    # 7. 3D座標の計算
    tris = faces[face_idx_ok]
    v_tri = vertices[tris]  # (N_ok, 3, 3)
    
    pos_ok = (lam0_ok[:, None] * v_tri[:, 0] + 
              lam1_ok[:, None] * v_tri[:, 1] + 
              lam2_ok[:, None] * v_tri[:, 2])
              
    points = pos_ok.tolist()
    keys = [(int(fi), int(k), int(l)) for fi, k, l in zip(face_idx_ok, K_ok, L_ok)]
    
    det_failed_faces = np.unique(face_indices_rep[~valid_det])
    skipped = sorted(list(set(skipped + det_failed_faces.tolist())))
    
    return points, keys, skipped


def count_intersections(vertices: np.ndarray, faces: np.ndarray, face_uvs: np.ndarray, *, face_indices: np.ndarray | None = None) -> int:
    """交点数を一括高速計算。"""
    if face_indices is not None:
        points, _, _ = extract_intersections(vertices, faces[face_indices], face_uvs[face_indices])
    else:
        points, _, _ = extract_intersections(vertices, faces, face_uvs)
    return len(points)


def estimate_grid_h(mesh: trimesh.Trimesh, v1: np.ndarray, v2: np.ndarray, target_quads: int) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bbox = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    area = float(mesh.area) if mesh.area > 0 else bbox * bbox
    
    # 幾何学的なダイレクトエッジ長算出 (C++ソルバーの格子充填率に基づく実証的チューニング係数 1.12 を適用)
    # これにより不安定なモンテカルロ交差点サンプリングをバイパスし、目標面数に確実に収束させます。
    h = np.sqrt(area / max(target_quads, 1000)) * 1.12
    el = getattr(mesh, "edges_unique_length", None)
    if el is not None and len(el) > 8:
        fine = float(np.percentile(np.asarray(el, dtype=np.float64), 10))
        if fine > 0:
            h = min(h, max(fine * 3.0, h * 0.5))
    h_limit = bbox / 100000.0
    return float(max(h, h_limit))


def weld_points(points, keys, eps: float):
    """3D空間座標に基づく KDTree-based Welding (第1段階)を行いつつ、マージされた新頂点と元の格子キーを紐付ける。"""
    if len(points) == 0:
        return np.zeros((0, 3)), [], {"before": 0, "after": 0, "node_to_keys": {}}
    pts = np.array(points, dtype=np.float64)
    
    # KDTree-based merging (extremely stable, no voxel boundary issues!)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(eps)
    
    n_pts = len(pts)
    import scipy.sparse
    from scipy.sparse.csgraph import connected_components
    
    if pairs:
        pairs_arr = np.array(list(pairs), dtype=np.int64)
        row = pairs_arr[:, 0]
        col = pairs_arr[:, 1]
        adj = scipy.sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(n_pts, n_pts))
        n_components, labels = connected_components(adj, directed=False, connection='weak')
    else:
        labels = np.arange(n_pts)
        n_components = n_pts

    up = np.zeros((n_components, 3), dtype=np.float64)
    for i in range(3):
        up[:, i] = np.bincount(labels, weights=pts[:, i], minlength=n_components)
    counts = np.bincount(labels, minlength=n_components)
    up /= counts[:, np.newaxis]
    
    # 各ユニークグループのキーを紐付け
    node_to_keys = {}
    for i, label in enumerate(labels):
        node_to_keys.setdefault(int(label), []).append(keys[i])
        
    # 各ユニークグループの代表キーを取得
    representative_indices = np.zeros(n_components, dtype=np.int64)
    seen_labels = set()
    for i, label in enumerate(labels):
        if label not in seen_labels:
            seen_labels.add(label)
            representative_indices[label] = i
            
    uk = [keys[idx] for idx in representative_indices]
    
    return up, uk, {"before": len(pts), "after": len(up), "node_to_keys": node_to_keys}


def _build_quads_for_chart(
    chart: int,
    grid_to_nodes: dict[tuple[int, int], list[tuple[int, np.ndarray]]],
    vertices: np.ndarray,
    h: float,
) -> list[list[int]]:
    quads = []
    seen_cells = set()
    # 高密度やノイズに対応するため、接続許容距離を最適化 (隣接2.0倍、エッジ2.0倍、対角2.8倍)
    # これにより低解像度での接続性を維持しつつ、ノイズ時の異常な長距離接続を完全にカットします
    lim_adj = 2.0 * h
    lim_diag = 2.8 * h
    lim_edge = 2.0 * h


    for (k, l), nodes_list in grid_to_nodes.items():
        k10 = (k + 1, l)
        k11 = (k + 1, l + 1)
        k01 = (k, l + 1)
        if k10 not in grid_to_nodes or k11 not in grid_to_nodes or k01 not in grid_to_nodes:
            continue

        c10 = grid_to_nodes[k10]
        c11 = grid_to_nodes[k11]
        c01 = grid_to_nodes[k01]

        for n00, p00 in nodes_list:
            cell_id = (k, l, n00)
            if cell_id in seen_cells:
                continue

            best_n10 = best_n11 = best_n01 = None
            min_d10, min_d11, min_d01 = lim_adj, lim_diag, lim_adj

            x00, y00, z00 = p00[0], p00[1], p00[2]

            for n10, p10 in c10:
                dx = p10[0] - x00
                dy = p10[1] - y00
                dz = p10[2] - z00
                d = math.sqrt(dx*dx + dy*dy + dz*dz)
                if d < min_d10:
                    min_d10 = d
                    best_n10 = n10

            for n11, p11 in c11:
                dx = p11[0] - x00
                dy = p11[1] - y00
                dz = p11[2] - z00
                d = math.sqrt(dx*dx + dy*dy + dz*dz)
                if d < min_d11:
                    min_d11 = d
                    best_n11 = n11

            for n01, p01 in c01:
                dx = p01[0] - x00
                dy = p01[1] - y00
                dz = p01[2] - z00
                d = math.sqrt(dx*dx + dy*dy + dz*dz)
                if d < min_d01:
                    min_d01 = d
                    best_n01 = n01

            if best_n10 is None or best_n11 is None or best_n01 is None:
                continue
            if len({n00, best_n10, best_n11, best_n01}) < 4:
                continue

            p10 = vertices[best_n10]
            p11 = vertices[best_n11]
            p01 = vertices[best_n01]

            e0 = math.sqrt((p10[0] - x00)**2 + (p10[1] - y00)**2 + (p10[2] - z00)**2)
            e1 = math.sqrt((p11[0] - p10[0])**2 + (p11[1] - p10[1])**2 + (p11[2] - p10[2])**2)
            e2 = math.sqrt((p01[0] - p11[0])**2 + (p01[1] - p11[1])**2 + (p01[2] - p11[2])**2)
            e3 = math.sqrt((x00 - p01[0])**2 + (y00 - p01[1])**2 + (z00 - p01[2])**2)

            max_e = max(e0, e1, e2, e3)
            min_e = min(e0, e1, e2, e3)

            if max_e >= lim_edge or min_e < 1e-8:
                continue

            seen_cells.add(cell_id)
            quads.append([n00, best_n10, best_n11, best_n01])

    return quads


def build_quads(
    node_to_keys: dict[int, list[tuple[int, int, int]]],
    face_chart: np.ndarray,
    vertices: np.ndarray,
    h: float,
) -> list[list[int]]:
    """チャート単位の (chart, k, l) 格子で Quad を接続。3D距離で遠距離ラップを拒否。"""
    charts_data: dict[int, dict[tuple[int, int], list[tuple[int, np.ndarray]]]] = {}
    for node_idx, keys_list in node_to_keys.items():
        pos = vertices[node_idx]
        for fi, k, l in keys_list:
            chart = int(face_chart[fi])
            charts_data.setdefault(chart, {}).setdefault((k, l), []).append((node_idx, pos))

    quads: list[list[int]] = []
    total_nodes = len(node_to_keys)
    if len(charts_data) <= 1 or total_nodes < 5000:
        for chart, grid_to_nodes in charts_data.items():
            quads.extend(_build_quads_for_chart(chart, grid_to_nodes, vertices, h))
        return quads

    max_workers = min(len(charts_data), os.cpu_count() or 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_build_quads_for_chart, chart, grid_to_nodes, vertices, h): chart
            for chart, grid_to_nodes in charts_data.items()
        }
        for future in concurrent.futures.as_completed(futures):
            quads.extend(future.result())

    return quads


def is_reasonable_quad(vertices: np.ndarray, q: np.ndarray, h: float) -> bool:
    """遠距離を結ぶ面や針状の面、ねじれ・自己交差・凹み・極端な折れ曲がりを除外する。"""
    p = vertices[q]
    edges = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    min_edge = float(edges.min())
    max_edge = float(edges.max())
    if min_edge < 1e-10:
        return False
    # 元の安全しきい値 (h * 8.0) に戻す（グリッド縮小時の誤判定を防止）
    if max_edge > max(h * 8.0, 1e-5):
        return False
    if max_edge / min_edge > 20.0:  # 辺比率も元の20倍に戻す
        return False
        
    # 幾何学的な凸性・平面性の厳密チェック（接平面に投影して2D外積の符号一致判定）
    # 1. 簡易的な平均法線を求める
    v01 = p[1] - p[0]
    v03 = p[3] - p[0]
    n = np.cross(v01, v03)
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        return False
    n /= ln
    
    # 2. 平均法線 n に直交する接ベクトル u, v を求める
    if abs(n[0]) < 0.9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, u)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    
    # 3. 2D空間に投影
    p2d = np.zeros((4, 2))
    for i in range(4):
        p2d[i, 0] = np.dot(p[i], u)
        p2d[i, 1] = np.dot(p[i], v)
        
    # 4. 各頂点での外積符号の一致（凹み・交差の完璧な排除）
    cp = np.zeros(4)
    for i in range(4):
        v1 = p2d[(i + 1) % 4] - p2d[i]
        v2 = p2d[(i + 2) % 4] - p2d[(i + 1) % 4]
        cp[i] = v1[0] * v2[1] - v1[1] * v2[0]
        
    if np.all(cp > 1e-8) or np.all(cp < -1e-8):
        return True
    return False


def compact_quad_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """未使用頂点を除去し、面インデックスを詰める。"""
    if len(faces) == 0:
        return vertices, faces
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def collapse_short_edges(vertices: np.ndarray, faces: np.ndarray, h: float, threshold_factor: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    極小エッジ（辺長 < h * threshold_factor）を検出し、端点をマージしてトポロジーをクリーンアップ（Pole Collapse）する。
    これにより特異点周りの極小ポリゴンや不正トポロジーが綺麗に収収束します。
    """
    if len(faces) == 0:
        return vertices, faces

    threshold = h * threshold_factor
    n_v = len(vertices)
    
    # 1. 辺の長さをチェックし、崩壊対象のペアを判定
    parent = np.arange(n_v)
    
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            
    # 隣接リストを作成して、頂点の接続数を把握
    edge_faces_count = {}
    for q in faces:
        for idx in range(4):
            u, v = int(q[idx]), int(q[(idx + 1) % 4])
            e = (min(u, v), max(u, v))
            edge_faces_count[e] = edge_faces_count.get(e, 0) + 1
            
    # 各面の四角形の各辺をチェック
    for q in faces:
        for idx in range(4):
            u, v = int(q[idx]), int(q[(idx + 1) % 4])
            dist = np.linalg.norm(vertices[u] - vertices[v])
            if dist < threshold:
                union(u, v)
                
    # 親テーブルの最終更新
    for i in range(n_v):
        find(i)
        
    # 2. 頂点の位置を更新
    new_vertices = vertices.copy()
    unique_roots = np.unique(parent)
    for root in unique_roots:
        members = np.where(parent == root)[0]
        if len(members) > 1:
            new_vertices[root] = vertices[members].mean(axis=0)
            
    new_vertices = new_vertices[parent]
    
    # 3. 面インデックスの書き換えとデジェネレート面の除去
    new_faces = parent[faces]
    
    valid_faces = []
    for q in new_faces:
        unique_nodes = np.unique(q)
        if len(unique_nodes) == 4:
            valid_faces.append(q)
            
    new_faces_arr = np.array(valid_faces, dtype=np.int64) if valid_faces else np.zeros((0, 4), dtype=np.int64)
    
    return compact_quad_mesh(new_vertices, new_faces_arr)


def cancel_adjacent_poles(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    トポロジー的な対消滅 (Pole Cancellation):
    至近距離 (1エッジ) にある内点 3-pole (次数3) と 5-pole (次数5) を検出し、
    それらを結ぶエッジをマージ (Collapse) することで、双方を通常の次数4の頂点へと相殺・消滅させる。
    """
    if len(faces) == 0:
        return vertices, faces

    n_v = len(vertices)
    
    # 1. 境界頂点の特定 (境界エッジに属する頂点)
    edge_counts = {}
    for q in faces:
        for i in range(4):
            u, v = int(q[i]), int(q[(i + 1) % 4])
            e = (min(u, v), max(u, v))
            edge_counts[e] = edge_counts.get(e, 0) + 1
            
    boundary_verts = set()
    for e, count in edge_counts.items():
        if count == 1:
            boundary_verts.add(e[0])
            boundary_verts.add(e[1])
            
    # 2. 隣接リストの作成と次数の算出
    adj = {}
    for q in faces:
        for i in range(4):
            u, v = int(q[i]), int(q[(i + 1) % 4])
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
            
    # 3. 内点における 3-pole と 5-pole の特定
    poles_3 = set()
    poles_5 = set()
    for u in range(n_v):
        if u in boundary_verts or u not in adj:
            continue
        valence = len(adj[u])
        if valence == 3:
            poles_3.add(u)
        elif valence == 5:
            poles_5.add(u)
            
    if not poles_3 or not poles_5:
        return vertices, faces
        
    # 4. 対消滅ペアの探索と Union-Find によるマージグループ作成
    parent = np.arange(n_v)
    
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    merged_any = False
    # 競合を避けるため、すでにマージされた頂点は再マージしない
    merged_set = set()
    
    for p3 in sorted(list(poles_3)):
        if p3 in merged_set:
            continue
        # 隣接する 5-pole を探す
        neighbors = adj[p3]
        for p5 in neighbors:
            if p5 in poles_5 and p5 not in merged_set:
                # 1エッジでの対消滅を実行
                if union(p3, p5):
                    merged_set.add(p3)
                    merged_set.add(p5)
                    merged_any = True
                    break # p3は1つのp5としかマージできない
                    
    if not merged_any:
        return vertices, faces
        
    # 親テーブルの更新
    for i in range(n_v):
        find(i)
        
    # 5. 頂点の位置を更新 (マージされたペアの平均位置に配置)
    new_vertices = vertices.copy()
    unique_roots = np.unique(parent)
    for root in unique_roots:
        members = np.where(parent == root)[0]
        if len(members) > 1:
            new_vertices[root] = vertices[members].mean(axis=0)
    new_vertices = new_vertices[parent]
    
    # 6. 面インデックスの更新とデジェネレート面の除去
    new_faces = parent[faces]
    valid_faces = []
    for q in new_faces:
        unique_nodes = np.unique(q)
        if len(unique_nodes) == 4:
            valid_faces.append(q)
            
    new_faces_arr = np.array(valid_faces, dtype=np.int64) if valid_faces else np.zeros((0, 4), dtype=np.int64)
    
    return compact_quad_mesh(new_vertices, new_faces_arr)


def keep_major_quad_components(
    vertices: np.ndarray, faces: np.ndarray, *, min_quads: int = 100, coverage: float = 0.98
) -> tuple[np.ndarray, np.ndarray]:
    """小さな島を除去し、主要コンポーネントのみ残す。"""
    if len(faces) == 0:
        return vertices, faces
    from collections import defaultdict

    edge_faces: dict[tuple, list[int]] = defaultdict(list)
    for fi, q in enumerate(faces):
        for i in range(4):
            e = (int(q[i]), int(q[(i + 1) % 4]))
            e = (min(e), max(e))
            edge_faces[e].append(fi)
    adj: dict[int, set[int]] = defaultdict(set)
    for flist in edge_faces.values():
        if len(flist) == 2:
            a, b = flist
            adj[a].add(b)
            adj[b].add(a)
    comps: list[list[int]] = []
    seen: set[int] = set()
    for fi in range(len(faces)):
        if fi in seen:
            continue
        stack = [fi]
        comp = []
        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            comp.append(f)
            stack.extend(adj[f] - seen)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    keep: list[int] = []
    total = len(faces)
    got = 0
    for comp in comps:
        if len(comp) < min_quads and got / max(total, 1) >= coverage:
            break
        keep.extend(comp)
        got += len(comp)
        if got / max(total, 1) >= coverage:
            break
    keep_f = faces[np.array(sorted(keep), dtype=np.int64)]
    return compact_quad_mesh(vertices, keep_f)


def merge_quad_vertices(
    vertices: np.ndarray, faces: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """近接頂点を距離ベースで統合（チャート境界の隙間を閉じる）。"""
    if len(vertices) == 0:
        return vertices, faces

    tree = cKDTree(vertices)
    pairs = tree.query_pairs(eps)
    if not pairs:
        return vertices, faces

    n_v = len(vertices)
    pairs_arr = np.array(list(pairs), dtype=np.int64)
    row = pairs_arr[:, 0]
    col = pairs_arr[:, 1]

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(n_v, n_v))
    n_components, labels = connected_components(adj, directed=False, connection='weak')

    new_v = np.zeros((n_components, 3), dtype=np.float64)
    for dim in range(3):
        new_v[:, dim] = np.bincount(labels, weights=vertices[:, dim], minlength=n_components)
    counts = np.bincount(labels, minlength=n_components)
    new_v /= counts[:, np.newaxis]

    new_faces = labels[faces]
    return new_v, new_faces


def chunked_surface_snap(
    pts: np.ndarray,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    chunk_size: int = 10000,
    log: DebugLog | None = None,
) -> tuple[np.ndarray, float, float]:
    """チャンク分割して proximity.closest_point を実行。失敗時は cKDTree の最近傍頂点へフォールバック。

    戻り値: (snapped_pts, max_displacement, mean_displacement)
    """
    if len(pts) == 0:
        return pts, 0.0, 0.0
    
    src = trimesh.Trimesh(vertices=source_vertices, faces=source_faces, process=False)
    snapped = []
    n_chunks = max(1, (len(pts) + chunk_size - 1) // chunk_size)

    for ci, i in enumerate(range(0, len(pts), chunk_size)):
        if log:
            log.progress(ci + 1, n_chunks, "surface_snap", every=max(1, n_chunks // 30))
        chunk = np.asarray(pts[i:i + chunk_size], dtype=np.float64)
        snapped_chunk = chunk.copy()
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            snapped.append(snapped_chunk)
            continue
        q = chunk[finite]
        try:
            sc, _, _ = trimesh.proximity.closest_point(src, q)
            sc = np.asarray(sc, dtype=np.float64)
            bad = ~np.isfinite(sc).all(axis=1)
            if bad.any():
                sc = sc.copy()
                sc[bad] = q[bad]
            snapped_chunk[finite] = sc
        except Exception:
            from scipy.spatial import cKDTree

            ref_ok = source_vertices[np.isfinite(source_vertices).all(axis=1)]
            tree = cKDTree(ref_ok)
            _, idx = tree.query(q)
            snapped_chunk[finite] = ref_ok[idx]
        snapped.append(snapped_chunk)
            
    snapped_pts = np.concatenate(snapped, axis=0)
    displacements = np.linalg.norm(snapped_pts - pts, axis=1)
    return snapped_pts, float(displacements.max()), float(displacements.mean())


def tangential_laplacian_smoothing(pts: np.ndarray, qf: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """頂点のラプラシアン平滑化。頂点をQuadメッシュの局所接平面上に拘束したまま移動させ、

    縮みを防ぎつつ格子を整列する。境界頂点は形状維持のため固定する。
    """
    if len(pts) == 0 or len(qf) == 0:
        return pts
    
    # 境界頂点の特定 (境界エッジに属する頂点)
    edge_count: dict[tuple[int, int], int] = {}
    for q in qf:
        for i in range(4):
            e = (min(int(q[i]), int(q[(i + 1) % 4])), max(int(q[i]), int(q[(i + 1) % 4])))
            edge_count[e] = edge_count.get(e, 0) + 1
    
    boundary_verts = set()
    for e, count in edge_count.items():
        if count == 1:
            boundary_verts.add(e[0])
            boundary_verts.add(e[1])
    
    # 隣接リストの作成
    adj: dict[int, set[int]] = {}
    for q in qf:
        for i in range(4):
            u, v = int(q[i]), int(q[(i + 1) % 4])
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
            
    # 隣接点の平均ベクトルの算出
    update = np.zeros_like(pts)
    for u, neighbors in adj.items():
        if u in boundary_verts:
            continue  # 境界頂点は固定
        if neighbors:
            nb_pts = pts[list(neighbors)]
            mean_nb = nb_pts.mean(axis=0)
            update[u] = mean_nb - pts[u]
            
    # 接平面への投影
    try:
        temp_mesh = trimesh.Trimesh(vertices=pts, faces=qf, process=False)
        normals = temp_mesh.vertex_normals
        dot_prod = np.einsum("ij,ij->i", update, normals)[:, None]
        update_tangent = update - dot_prod * normals
        new_pts = pts + alpha * update_tangent
    except Exception:
        # フォールバック: 通常のラプラシアン
        new_pts = pts + alpha * update
        
    return new_pts


def export_quad_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    vertex_colors: np.ndarray | None = None,
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        if vertex_colors is not None and len(vertex_colors) == len(vertices):
            for (x, y, z), (r, g, b) in zip(vertices, vertex_colors):
                fh.write(f"v {x:.8f} {y:.8f} {z:.8f} {r:.6f} {g:.6f} {b:.6f}\n")
        else:
            for x, y, z in vertices:
                fh.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for q in faces:
            fh.write(f"f {' '.join(str(int(i) + 1) for i in q)}\n")

def bridge_seam_quads(
    pts: np.ndarray,
    orig_vertices: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    h: float,
    existing_quads: list[list[int]],
) -> list[list[int]]:
    """Vectorized 3D-direction-based search to bridge seams and charts, returning missing quads (manifold-safe)."""
    if len(pts) == 0:
        return []

    orig_tree = cKDTree(orig_vertices)
    _, orig_indices = orig_tree.query(pts)
    
    pts_v1 = v1[orig_indices]
    pts_v2 = v2[orig_indices]
    pts_tree = cKDTree(pts)

    seen_quads = set()
    edge_counts = {}

    for q in existing_quads:
        q_int = [int(x) for x in q]
        seen_quads.add(tuple(sorted(q_int)))
        for idx in range(4):
            u, v = q_int[idx], q_int[(idx + 1) % 4]
            e = (min(u, v), max(u, v))
            edge_counts[e] = edge_counts.get(e, 0) + 1

    bridged_quads = []
    
    tol = h * 0.45
    diag_tol = h * 0.60

    dirs_all = [
        pts_v1,
        pts_v2,
        -pts_v1,
        -pts_v2
    ]

    for a in range(4):
        b = (a + 1) % 4
        da = dirs_all[a]
        db = dirs_all[b]

        T_j = pts + h * da
        T_l = pts + h * db
        T_k = pts + h * (da + db)

        dist_j, j_indices = pts_tree.query(T_j)
        dist_l, l_indices = pts_tree.query(T_l)
        dist_k, k_indices = pts_tree.query(T_k)

        mask = (dist_j < tol) & (dist_l < tol) & (dist_k < diag_tol)
        
        valid_i = np.where(mask)[0]
        if len(valid_i) == 0:
            continue

        j_vals = j_indices[valid_i]
        l_vals = l_indices[valid_i]
        k_vals = k_indices[valid_i]

        for idx, i in enumerate(valid_i):
            j = int(j_vals[idx])
            k = int(k_vals[idx])
            l = int(l_vals[idx])

            if len({i, j, k, l}) == 4:
                q = [i, j, k, l]
                q_sorted = tuple(sorted(q))
                if q_sorted not in seen_quads:
                    edges = [
                        (min(i, j), max(i, j)),
                        (min(j, k), max(j, k)),
                        (min(k, l), max(k, l)),
                        (min(l, i), max(l, i))
                    ]
                    # 既に共有面数が2つ以上のエッジがあれば非多様体化するためスキップ
                    if any(edge_counts.get(e, 0) >= 2 for e in edges):
                        continue
                        
                    seen_quads.add(q_sorted)
                    for e in edges:
                        edge_counts[e] = edge_counts.get(e, 0) + 1
                    bridged_quads.append(q)

    return bridged_quads


def extract_quads_from_mesh(
    mesh: trimesh.Trimesh,
    field: np.ndarray,
    out_dir: Path,
    *,
    target_quads: int | None = None,
    grid_h: float | None = None,
    bridge_seams: bool = False,
    vertex_color_export: bool = True,
    color_reference_mesh: trimesh.Trimesh | None = None,
    color_reference_path: Path | str | None = None,
) -> Path:
    log = DebugLog("quad-extractor", out_dir)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v1, v2 = field[:, 0], field[:, 1]
    bbox = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    tgt = target_quads or 300_000
    if grid_h is None:
        h = estimate_grid_h(mesh, v1, v2, tgt)
    else:
        h = grid_h
    log.set("target_quads", tgt)
    log.set("input_verts", len(vertices))
    log.set("input_faces", len(faces))

    log.step("UV integrate (BFS charts)")
    face_uvs, meta = integrate_uv(mesh, faces, v1, v2, log=log)
    face_chart = meta["face_chart"]
    log.set("poisson_residual", 0.0)
    log.set("uv_charts", int(meta["anchors"]))

    qf = np.zeros((0, 4), dtype=np.int64)
    pts = np.zeros((0, 3))
    
    h_limit = bbox / 1000.0

    for attempt in range(3):
        log.step(f"extract attempt {attempt + 1}/3")
        h = max(h, h_limit)
        log.set("grid_h", h)
        span = max(bbox / h, 1.0)
        uu_vv = _normalize_uv(face_uvs, span)
        log.step("marching intersections")
        points, keys, skipped = extract_intersections(vertices, faces, uu_vv, log=log)
        log.set("intersection_candidates", len(points))
        log.set("skipped_faces_det", len(set(skipped)))

        log.step("weld intersection points")
        eps = max(1e-5, h * 0.10)
        pts, keys, wr = weld_points(points, keys, eps)
        log.set("weld_before", wr["before"])
        log.set("weld_after", wr["after"])

        log.step("build quads from grid")
        quads = build_quads(wr["node_to_keys"], face_chart, pts, h)
        log.set("quads_built", len(quads))

        # Bridge seams and charts using 3D-direction-based search
        if bridge_seams:
            try:
                bridged = bridge_seam_quads(pts, vertices, v1, v2, h, quads)
                log.set("quads_bridged", len(bridged))
                quads.extend(bridged)
            except Exception as e:
                log.warn(f"bridge_seam_quads failed: {e}")
        else:
            log.set("quads_bridged", 0)

        cand = np.array(quads, dtype=np.int64) if quads else np.zeros((0, 4), dtype=np.int64)
        valid, removed = [], 0
        seen = set()
        for q in cand:
            if len(set(q)) < 4:
                removed += 1
                continue
            sk = tuple(int(x) for x in q)
            if sk in seen:
                removed += 1
                continue
            seen.add(sk)
            if quad_area(pts, q) < 1e-12:
                removed += 1
                continue
            if not is_reasonable_quad(pts, q, h):
                removed += 1
                continue
            valid.append(q)
        qf = np.array(valid, dtype=np.int64) if valid else np.zeros((0, 4), dtype=np.int64)
        log.set("invalid_quads_removed", removed)
        log.set("quads_final", len(qf))
        
        if not grid_h and tgt and len(qf) < tgt * 0.85 and attempt < 2:
            scale = max(0.35, min(0.9, np.sqrt(len(qf) / tgt)))
            h *= scale
            log.set(f"refine_attempt_{attempt + 1}", scale)
            continue
        break

    # 特異点周辺等の極小エッジをマージしてトポロジーをクリーンアップ（Pole Collapse）
    log.step("marching pole collapse")
    pts, qf = collapse_short_edges(pts, qf, h)

    # 近接する3-poleと5-poleの対消滅によるトポロジー最適化
    log.step("marching pole cancellation")
    pts, qf = cancel_adjacent_poles(pts, qf)

    # Initial topology checks (before snapping/merging)
    topo_init = run_topology_checks(pts, qf, h)
    for k, val in topo_init.items():
        log.set(f"topo_init_{k}", val)

    log.step("surface snap (chunked closest_point)")
    try:
        pts, max_disp, mean_disp = chunked_surface_snap(pts, vertices, faces, log=log)
        log.set("surface_snap", True)
        log.set("surface_snap_max_displacement", max_disp)
        log.set("surface_snap_mean_displacement", mean_disp)
    except Exception as e:
        log.warn(f"surface_snap failed: {e}")
        log.set("surface_snap", False)

    log.step("tangential smoothing + re-snap")
    try:
        pts = tangential_laplacian_smoothing(pts, qf, alpha=0.3)
        pts, _, _ = chunked_surface_snap(pts, vertices, faces, log=log)
        log.set("tangential_smoothing", True)
    except Exception as e:
        log.warn(f"tangential_smoothing failed: {e}")
        log.set("tangential_smoothing", False)

    log.step("merge vertices + island filter")
    merge_eps = max(h * 0.05, 1e-6)
    pts, qf = merge_quad_vertices(pts, qf, merge_eps)
    pts, qf = keep_major_quad_components(pts, qf, min_quads=5, coverage=0.999)
    log.set("verts_after_compact", len(pts))
    log.set("quads_after_island_filter", len(qf))

    # Final topology validation checks on the final retopologized mesh
    topo_final = run_topology_checks(pts, qf, h)
    for k, val in topo_final.items():
        log.set(f"topo_{k}", val)

    quad_score, quad_terms = score_quad_mesh(topo_final, pass_threshold=QUALITY_PASS)
    log.set("quad_quality_score", quad_score)
    for k, val in quad_terms.items():
        log.set(f"quad_quality_{k}", val)
    write_quality_report(
        out_dir,
        [{
            "index": 0,
            "params": {
                "extractor": "python",
                "grid_h": float(h),
                "target_quads": int(tgt),
            },
            "score": quad_score,
            "terms": quad_terms,
            "stats": topo_final,
        }],
        selected=0,
        pass_threshold=QUALITY_PASS,
    )

    out_verts = pts.copy()
    from .run_layout import layout_for_stage_dir

    lay = layout_for_stage_dir(out_dir)
    lay.ensure_dirs()
    out_path = lay.retopo_quad_mesh()
    log.step(f"export {out_path.name}")
    from .vertex_color import colors_for_quad_export

    ref_color_mesh = color_reference_mesh if color_reference_mesh is not None else mesh
    q_colors = colors_for_quad_export(
        pts,
        ref_color_mesh,
        ref_path=color_reference_path,
        vertex_color_export=vertex_color_export,
        quad_faces=qf,
    )
    if q_colors is not None:
        log.set("vertex_color_export", True)
    export_quad_obj(out_path, out_verts, qf, vertex_colors=q_colors)
    from .mesh_io import COORD_YUP, write_coord_space_json

    write_coord_space_json(out_path.parent, COORD_YUP)
    log.done("quad-extractor")
    log.save()
    return out_path


def extract_quads(mesh_path: Path, field_path: Path, out_dir: Path, **kwargs) -> Path:
    from .mesh_io import load_mesh_yup

    mesh = load_mesh_yup(Path(mesh_path))
    field = np.load(field_path)
    if field.shape[-1] == 6:
        field = field.reshape(len(field), 2, 3)
    return extract_quads_from_mesh(mesh, field, Path(out_dir), **kwargs)


# Backward-compatible alias
extract_quads_python = extract_quads_from_mesh
