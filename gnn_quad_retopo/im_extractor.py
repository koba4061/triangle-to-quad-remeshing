"""Instant Meshes bridge: native > pyinstantmeshes > im-lite Python."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import trimesh

from .cross_field import compute_cross_field
from .im_bridge_core import build_adjacency, smooth_orientation_field
from .im_lite import im_lite_extract
from .im_pyinstantmeshes import is_pyim_available, pyim_status, teacher_field_via_pyim
from .quad_extractor import estimate_grid_h, extract_quads_from_mesh
from .run_layout import layout_for_stage_dir
from .topology_checks import QUALITY_PASS, run_topology_checks, score_quad_mesh, write_quality_report

def _sanitize_quad_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN/Inf vertices and faces that reference them (prevents origin fan artifacts)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    ok = np.isfinite(vertices).all(axis=1)
    if ok.all():
        return vertices, faces
    new_idx = np.full(len(vertices), -1, dtype=np.int64)
    new_idx[np.flatnonzero(ok)] = np.arange(int(ok.sum()))
    vertices = vertices[ok]
    kept: list[list[int]] = []
    for f in faces:
        if f.shape[0] == 4 and f[2] == f[3]:
            idx = f[:3]
        elif f.shape[0] == 4:
            idx = f
        else:
            idx = f
        idx = [int(i) for i in idx]
        if len(set(idx)) < len(idx):
            continue
        if not np.all(ok[idx]):
            continue
        kept.append([int(new_idx[i]) for i in idx])
    if not kept:
        raise ValueError("no valid faces after sanitizing NaN/Inf vertices")
    return vertices, np.asarray(kept, dtype=np.int64)


def _repair_nonfinite_vertices(
    vertices: np.ndarray,
    ref_mesh: trimesh.Trimesh,
    faces: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Replace NaN/Inf quad verts with localized neighborhood averages or surface projections.

    Prevents central collapse spiderweb artifacts by utilizing connection adjacency.
    """
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    bad = ~np.isfinite(vertices).all(axis=1)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return vertices, 0

    # 1. Try localized Laplacian smoothing based on adjacent finite neighbors
    if faces is not None and len(faces) > 0:
        from collections import defaultdict
        adj = defaultdict(set)
        for f in faces:
            nv = len(f)
            for i in range(nv):
                u, v = int(f[i]), int(f[(i + 1) % nv])
                adj[u].add(v)
                adj[v].add(u)

        # Multi-pass localized interpolation to heal contiguous NaN clusters
        for _ in range(3):
            still_bad = np.flatnonzero(~np.isfinite(vertices).all(axis=1))
            if len(still_bad) == 0:
                break
            for idx in still_bad:
                neighbors = adj.get(int(idx), set())
                valid_n = [int(n) for n in neighbors if np.isfinite(vertices[int(n)]).all()]
                if valid_n:
                    vertices[int(idx)] = vertices[valid_n].mean(axis=0)

    # 2. Global fallback for isolated islands/complete failures (project to closest valid surface point)
    bad_still = ~np.isfinite(vertices).all(axis=1)
    n_still = int(bad_still.sum())
    if n_still > 0:
        ref_v = np.asarray(ref_mesh.vertices, dtype=np.float64)
        ref_ok = ref_v[np.isfinite(ref_v).all(axis=1)]
        if len(ref_ok) == 0:
            raise ValueError("reference mesh has no finite vertices")
        from scipy.spatial import cKDTree

        tree = cKDTree(ref_ok)
        seed = ref_ok.mean(axis=0, keepdims=True)
        _, idx = tree.query(seed)
        fill = ref_ok[int(idx[0])]
        vertices[bad_still] = fill

    return vertices, n_bad


def _quad_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if f.shape[1] == 3:
        p = vertices[f]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    else:
        n = np.cross(vertices[f[:, 2]] - vertices[f[:, 0]], vertices[f[:, 3]] - vertices[f[:, 1]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ok = ln[:, 0] > 1e-12
    out = np.zeros_like(n)
    out[ok] = n[ok] / ln[ok]
    return out


def _orient_quad_faces_to_reference(
    vertices: np.ndarray,
    faces: np.ndarray,
    ref_mesh: trimesh.Trimesh,
    *,
    chunk_size: int = 50000,
) -> tuple[np.ndarray, int]:
    """Flip quad winding where face normal disagrees with closest ref triangle."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).copy()
    cent = v[f].mean(axis=1)
    n = _quad_face_normals(v, f)
    flipped = 0
    ref = ref_mesh
    for i in range(0, len(f), chunk_size):
        c = cent[i : i + chunk_size]
        nn = n[i : i + chunk_size]
        _, _, tri_id = trimesh.proximity.closest_point(ref, c)
        rn = np.asarray(ref.face_normals[tri_id], dtype=np.float64)
        bad = np.einsum("ij,ij->i", nn, rn) < 0.0
        if bad.any():
            idx = np.flatnonzero(bad) + i
            f[idx] = f[idx][:, ::-1]
            flipped += int(len(idx))
    return f, flipped


def _weld_boundary_vertices(v: np.ndarray, f: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """継ぎ目（境界辺の頂点）どうしを eps 以内なら 1 つにまとめる。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    k = f.shape[1]
    e = np.sort(np.concatenate([f[:, [i, (i + 1) % k]] for i in range(k)]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    bv = np.unique(u[cnt == 1])
    if len(bv) < 2 or eps <= 0:
        return f, 0
    pairs = cKDTree(v[bv]).query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return f, 0
    n = len(bv)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    rep = np.zeros(lab.max() + 1, dtype=np.int64)
    rep[lab[::-1]] = bv[::-1]
    remap = np.arange(len(v))
    remap[bv] = rep[lab]
    return remap[f], int((remap[bv] != bv).sum())


def _orient_consistently(v: np.ndarray, f: np.ndarray, ref_mesh: trimesh.Trimesh | None) -> tuple[np.ndarray, int]:
    """隣どうしの向きを辺でそろえ、部品の重心から外を表にする。

    元メッシュの法線を真似ると、元が裏の部品まで裏のまま残る。
    平らで内外が決まらない部品だけ、元メッシュの法線を見る。
    """
    from collections import deque

    nf, k = f.shape
    a = f.reshape(-1)
    b = np.roll(f, -1, axis=1).reshape(-1)
    fid = np.repeat(np.arange(nf), k)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * (int(len(v)) + 1) + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    size = np.diff(np.r_[start, len(ks)])
    two = start[size == 2]
    e0, e1 = order[two], order[two + 1]
    f0, f1 = fid[e0], fid[e1]
    same = a[e0] == a[e1]
    nbr = [[] for _ in range(nf)]
    for x, y, s in zip(f0.tolist(), f1.tolist(), same.tolist()):
        nbr[x].append((y, s))
        nbr[y].append((x, s))
    flip = np.zeros(nf, dtype=bool)
    comp = np.full(nf, -1, dtype=np.int64)
    nc = 0
    for s0 in range(nf):
        if comp[s0] >= 0:
            continue
        comp[s0] = nc
        dq = deque([s0])
        while dq:
            x = dq.popleft()
            for y, s in nbr[x]:
                if comp[y] < 0:
                    comp[y] = nc
                    flip[y] = flip[x] ^ s
                    dq.append(y)
        nc += 1
    out = f.copy()
    out[flip] = out[flip][:, ::-1]
    n = _quad_face_normals(v, out)
    cent = v[out].mean(axis=1)
    area = np.linalg.norm(np.cross(v[out[:, 2]] - v[out[:, 0]], v[out[:, -1]] - v[out[:, 1]]), axis=1)
    comp_cent = np.zeros((nc, 3))
    wsum = np.zeros(nc)
    np.add.at(comp_cent, comp, cent * area[:, None])
    np.add.at(wsum, comp, area)
    comp_cent /= np.maximum(wsum, 1e-12)[:, None]
    vote_out = np.zeros(nc)
    np.add.at(vote_out, comp, np.einsum("ij,ij->i", cent - comp_cent[comp], n) * area)
    vote_ref = np.zeros(nc)
    if ref_mesh is not None and len(ref_mesh.faces):
        from scipy.spatial import cKDTree

        rn = np.asarray(ref_mesh.face_normals, dtype=np.float64)
        _, ti = cKDTree(np.asarray(ref_mesh.triangles_center)).query(cent, workers=-1)
        np.add.at(vote_ref, comp, area * np.sign(np.einsum("ij,ij->i", n, rn[ti])))
    use_out = np.abs(vote_out) > 0.15 * np.maximum(wsum, 1e-12)
    vote = np.where(use_out, vote_out, vote_ref)
    bad_comp = vote < 0
    fl = bad_comp[comp]
    out[fl] = out[fl][:, ::-1]
    return out, int((flip ^ fl).sum())


def clean_quad_topology(v: np.ndarray, f: np.ndarray, ref_mesh: trimesh.Trimesh | None) -> tuple[np.ndarray, dict]:
    """重複面を消し、継ぎ目を溶接し、表裏をそろえる。"""
    f = np.asarray(f, dtype=np.int64)
    stats = {}
    if f.ndim != 2 or len(f) == 0:
        return f, stats
    edge = float(np.median(np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)))
    f, stats["welded"] = _weld_boundary_vertices(v, f, edge * 0.35)
    distinct = np.array([len(set(r)) == f.shape[1] for r in f.tolist()])
    stats["degenerate"] = int((~distinct).sum())
    f = f[distinct]
    _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    stats["duplicate"] = int(len(f) - len(keep))
    f = f[np.sort(keep)]
    f, stats["flipped"] = _orient_consistently(v, f, ref_mesh)
    f, stats["unfolded"] = _unfold_folded_quads(v, f)
    return f, stats


def _unfold_folded_quads(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, int]:
    """折れ曲がった四角は、黒い三角にならない対角へ頂点順を変える。"""
    if f.ndim != 2 or f.shape[1] != 4 or len(f) == 0:
        return f, 0

    def _nrm(a, b, c):
        return np.cross(v[b] - v[a], v[c] - v[a])

    fold = np.einsum("ij,ij->i", _nrm(f[:, 0], f[:, 1], f[:, 2]), _nrm(f[:, 0], f[:, 2], f[:, 3])) < 0
    if not np.any(fold):
        return f, 0
    other = np.einsum("ij,ij->i", _nrm(f[:, 1], f[:, 2], f[:, 3]), _nrm(f[:, 1], f[:, 3], f[:, 0])) >= 0
    take = fold & other
    out = f.copy()
    out[take] = f[take][:, [1, 2, 3, 0]]
    return out, int(take.sum())


def _boundary_edge_loops(faces: np.ndarray) -> list[list[int]]:
    """Return vertex loops along boundary edges (count==1)."""
    from collections import Counter, defaultdict

    edges: Counter[tuple[int, int]] = Counter()
    for q in np.asarray(faces, dtype=np.int64):
        ids = list(q) + [int(q[0])]
        for a, b in zip(ids, ids[1:]):
            if a > b:
                a, b = b, a
            edges[(a, b)] += 1
    boundary = [e for e, c in edges.items() if c == 1]
    if not boundary:
        return []

    nbr: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        nbr[a].append(b)
        nbr[b].append(a)

    used_e: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for e0 in boundary:
        if e0 in used_e:
            continue
        a0, b0 = e0
        loop = [a0, b0]
        used_e.add(e0)
        prev, cur = a0, b0
        while True:
            nxt_cands = [x for x in nbr[cur] if x != prev]
            if not nxt_cands:
                break
            nxt = nxt_cands[0]
            e = (cur, nxt) if cur < nxt else (nxt, cur)
            if e in used_e:
                if nxt == loop[0]:
                    break
                break
            used_e.add(e)
            if nxt == loop[0]:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
            if len(loop) > len(boundary) + 4:
                break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _stitch_small_quad_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    ref_mesh: trimesh.Trimesh,
    *,
    max_loop_edges: int = 4,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Add missing quads for tiny IM hole loops (typically 4 edges)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    added = 0
    extra: list[np.ndarray] = []
    for loop in _boundary_edge_loops(f):
        n = len(loop)
        if n != max_loop_edges or n < 3:
            continue
        quad = np.asarray(loop[:4], dtype=np.int64)
        if len(set(int(x) for x in quad)) < 4:
            continue
        extra.append(quad)
        added += 1
    if not extra:
        return v, f, 0
    f_out = np.vstack([f, np.asarray(extra, dtype=np.int64)])
    return v, f_out, added


def _export_im_quad_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    vertex_colors: np.ndarray | None = None,
) -> None:
    """Write IM quad mesh as OBJ (preserve quads; pipeline Y-up, same as cleaned_mesh)."""
    path = Path(path)
    vertices, faces = _sanitize_quad_mesh(vertices, faces)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] not in (3, 4):
        raise ValueError(f"expected (N,3|4) faces, got {faces.shape}")
    lines = []
    if vertex_colors is not None and len(vertex_colors) == len(vertices):
        for v, (r, g, b) in zip(vertices, vertex_colors):
            lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {r:.6f} {g:.6f} {b:.6f}")
    else:
        for v in vertices:
            lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}")
    tris = []
    for f in faces:
        if len(f) >= 4:
            tris.append((int(f[0]), int(f[1]), int(f[2])))
            tris.append((int(f[0]), int(f[2]), int(f[3])))
        else:
            tris.append(tuple(int(x) for x in f[:3]))
    acc = np.zeros_like(vertices, dtype=np.float64)
    tri_a = np.asarray(tris, dtype=np.int64)
    fn = np.cross(vertices[tri_a[:, 1]] - vertices[tri_a[:, 0]], vertices[tri_a[:, 2]] - vertices[tri_a[:, 0]])
    for k in range(3):
        np.add.at(acc, tri_a[:, k], fn)
    ln = np.linalg.norm(acc, axis=1, keepdims=True)
    acc = acc / np.maximum(ln, 1e-12)
    for n in acc:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    for f in faces:
        if f.shape[0] == 4 and f[2] == f[3]:
            idx = f[:3]
        elif f.shape[0] == 4:
            idx = f
        else:
            idx = f
        if len(set(int(x) for x in idx)) < len(idx):
            continue
        parts = " ".join("%d//%d" % (int(i) + 1, int(i) + 1) for i in idx)
        lines.append(f"f {parts}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

_NATIVE = None
_NATIVE_ERR: str | None = None
# im_lite uses Python UV integration; unsafe on large kitbash meshes (hang/OOM).
_IM_LITE_MAX_VERTS = 120_000


def _try_native():
    global _NATIVE, _NATIVE_ERR
    if _NATIVE is not None:
        return _NATIVE
    try:
        import importlib
        import sys

        pkg_dir = str(Path(__file__).resolve().parent)
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        nb = importlib.import_module("im_gnn_bridge")
        if getattr(nb, "has_instant_meshes", False):
            _NATIVE = nb
            return nb
        _NATIVE_ERR = "im_gnn_bridge stub (run scripts/build_native.sh with VS Build Tools)"
    except ImportError as e:
        _NATIVE_ERR = str(e)
    return None


def is_im_available() -> bool:
    return _try_native() is not None or is_pyim_available()


def im_status() -> str:
    nb = _try_native()
    if nb is not None:
        return "native"
    if is_pyim_available():
        return "pyinstantmeshes+im_lite"
    return f"im_lite_only ({_NATIVE_ERR or 'no native/pyim'})"


def export_orientation_field(
    mesh: trimesh.Trimesh,
    *,
    crease_angle: float = 40.0,
    align_to_boundaries: bool = True,
    smooth_iterations: int = 10,
    use_pyim: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """IM-style orientation field for teacher generation."""
    nb = _try_native()
    if nb is not None:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int32)
        return nb.export_orientation_field(
            v, f, crease_angle=crease_angle,
            align_to_boundaries=align_to_boundaries,
            smooth_iterations=smooth_iterations,
        )

    if use_pyim and is_pyim_available():
        try:
            return teacher_field_via_pyim(mesh)
        except Exception:
            pass

    field = compute_cross_field(mesh, use_curvature=True).reshape(len(mesh.vertices), 2, 3)
    v1, v2 = field[:, 0].astype(np.float64), field[:, 1].astype(np.float64)
    adj = build_adjacency(np.asarray(mesh.faces), len(mesh.vertices))
    n = np.asarray(mesh.vertex_normals, dtype=np.float64)
    return smooth_orientation_field(
        np.asarray(mesh.vertices), n, v1, adj, iterations=smooth_iterations
    )


def _resolve_target_edge_length(
    mesh: trimesh.Trimesh,
    field: np.ndarray,
    *,
    target_quads: int | None,
    target_edge_length: float | None,
    grid_h: float | None,
    quad_density: float,
) -> float:
    v1, v2 = field[:, 0], field[:, 1]
    edge = target_edge_length
    if edge is None and grid_h is not None:
        edge = grid_h
    if edge is None:
        if target_quads is not None:
            edge = estimate_grid_h(mesh, v1, v2, target_quads)
        else:
            edge = -1.0
    if edge > 0:
        density = float(quad_density)
        if density <= 0:
            ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
            diag = float(np.linalg.norm(ext))
            n_faces = int(len(mesh.faces))
            # Auto: pick density in [1.0, 1.2] from mesh scale + how fine base_edge wants to go.
            detail_div = 520 if n_faces < 500_000 else (580 if n_faces < 1_000_000 else 640)
            desired = diag / detail_div if diag > 0 else float(edge)
            raw = float(edge) / max(desired, 1e-9)
            boost_ratio = float(np.clip((raw - 1.0) / 3.0, 0.0, 1.0))
            boost_size = float(
                np.clip((math.log10(max(n_faces, 1)) - 5.2) / 1.2, 0.0, 1.0)
            )
            score = 0.55 * boost_ratio + 0.45 * boost_size
            density = float(np.clip(1.0 + 0.2 * score, 1.0, 1.2))
            print(
                f"[im] auto quad_density={density:.3f} "
                f"(base_edge={float(edge):.6f}, desired_edge={desired:.6f}, "
                f"raw={raw:.2f}, score={score:.2f}, faces={n_faces:,})",
                flush=True,
            )
        edge = float(edge) / density
    return float(edge)


def _relax_quad_vertices(v: np.ndarray, f: np.ndarray, iters: int) -> np.ndarray:
    """抜いたあとの頂点を隣へ少し寄せる。Instant Meshes の平滑化段はここでは使わない。"""
    iters = int(iters)
    if iters <= 0 or len(f) == 0:
        return v
    out = np.asarray(v, dtype=np.float64).copy()
    ff = np.asarray(f, dtype=np.int64)
    pairs = []
    n = ff.shape[1]
    for i in range(n):
        a = ff[:, i]
        b = ff[:, (i + 1) % n]
        ok = a != b
        if np.any(ok):
            pairs.append(np.stack([a[ok], b[ok]], axis=1))
    if not pairs:
        return out
    edges = np.unique(np.sort(np.vstack(pairs), axis=1), axis=0)
    for _ in range(iters):
        acc = np.zeros_like(out)
        cnt = np.zeros(len(out), dtype=np.float64)
        np.add.at(acc, edges[:, 0], out[edges[:, 1]])
        np.add.at(acc, edges[:, 1], out[edges[:, 0]])
        np.add.at(cnt, edges[:, 0], 1.0)
        np.add.at(cnt, edges[:, 1], 1.0)
        good = cnt > 0
        mean = np.zeros_like(out)
        mean[good] = acc[good] / cnt[good, None]
        out[good] = 0.5 * out[good] + 0.5 * mean[good]
    return out


def _native_extract_with_retries(
    nb,
    v: np.ndarray,
    f: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    base_edge: float,
    *,
    crease_angle: float,
    align_to_boundaries: bool,
    smooth_orient_iters: int,
    smooth_pos_iters: int,
    pure_quad: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """同じ辺長で一度だけ抜く。落ちる抽出後の平滑化は呼ばない。"""
    relax = max(0, int(smooth_pos_iters))
    attempts: list[float] = [
        float(base_edge),
        float(base_edge) * 1.15 if base_edge > 0 else -1.0,
        float(base_edge) * 1.35 if base_edge > 0 else -1.0,
        -1.0,
    ]
    last_err: RuntimeError | None = None
    for i, edge in enumerate(attempts, start=1):
        edge_s = "auto" if edge <= 0 else f"{edge:.6f}"
        print(
            f"[im] native attempt {i}/{len(attempts)} "
            f"target_edge={edge_s} smooth_pos_iters=0 relax={relax}",
            flush=True,
        )
        try:
            qv, qf = nb.extract_with_cross_field(
                v,
                f,
                v1.astype(np.float64),
                v2.astype(np.float64),
                float(edge),
                crease_angle=crease_angle,
                align_to_boundaries=align_to_boundaries,
                smooth_orient_iters=smooth_orient_iters,
                smooth_pos_iters=0,
                pure_quad=pure_quad,
            )
            if relax:
                qv = _relax_quad_vertices(qv, qf, relax)
            return qv, qf
        except RuntimeError as e:
            last_err = e
            print(f"[im] native attempt {i} failed: {e}", flush=True)
    assert last_err is not None
    raise RuntimeError(
        f"native Instant Meshes failed after {len(attempts)} attempts: {last_err}"
    ) from last_err


def shrinkwrap_vertices_to_mesh(
    verts: np.ndarray,
    ref_mesh: trimesh.Trimesh,
    *,
    faces: np.ndarray | None = None,
    passes: int = 3,
    log=None,
) -> tuple[np.ndarray, float, float]:
    """Project quad vertices onto reference mesh; multi-pass for complex kitbash assets."""
    from .quad_extractor import chunked_surface_snap

    out = np.asarray(verts, dtype=np.float64).copy()
    bad_in = ~np.isfinite(out).all(axis=1)
    if bad_in.any():
        out, n_rep = _repair_nonfinite_vertices(out, ref_mesh, faces=faces)
        print(f"[im] shrinkwrap: repaired {n_rep} non-finite input vertices", flush=True)
    max_d = mean_d = 0.0
    n_pass = max(1, int(passes))
    ref_v = np.asarray(ref_mesh.vertices, dtype=np.float64)
    ref_f = np.asarray(ref_mesh.faces, dtype=np.int64)
    for pi in range(n_pass):
        prev = out.copy()
        snapped, max_d, mean_d = chunked_surface_snap(out, ref_v, ref_f, log=log)
        bad = ~np.isfinite(snapped).all(axis=1)
        if bad.any():
            snapped = snapped.copy()
            snapped[bad] = prev[bad]
            print(
                f"[im] shrinkwrap pass {pi + 1}/{n_pass}: "
                f"kept original for {int(bad.sum())} invalid points",
                flush=True,
            )
        out = snapped
        print(
            f"[im] shrinkwrap pass {pi + 1}/{n_pass}: "
            f"max_disp={max_d:.6f} mean_disp={mean_d:.6f}",
            flush=True,
        )
        if max_d < 1e-7:
            break
    return out, max_d, mean_d


def _quality_grid_h(base_edge: float, vertices: np.ndarray, faces: np.ndarray) -> float:
    if base_edge > 0:
        return float(base_edge)
    if len(faces) == 0:
        return 1.0
    ext = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    return max(ext / 1000.0, 1e-6)


def _prepare_quad_for_quality(
    new_v: np.ndarray,
    new_f: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    stitch_small_holes: bool,
    max_hole_loop_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    new_v, _ = _repair_nonfinite_vertices(new_v, mesh, faces=new_f)
    new_v, new_f = _sanitize_quad_mesh(new_v, new_f)
    if stitch_small_holes:
        new_v, new_f, _ = _stitch_small_quad_holes(
            new_v, new_f, mesh, max_loop_edges=max_hole_loop_edges,
        )
    return new_v, new_f


def _evaluate_quad_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
    base_edge: float,
    *,
    pass_threshold: float = QUALITY_PASS,
) -> tuple[float, dict, dict]:
    h = _quality_grid_h(base_edge, vertices, faces)
    stats = run_topology_checks(vertices, faces, h)
    score, terms = score_quad_mesh(stats, pass_threshold=pass_threshold)
    return score, terms, stats


def _quality_param_candidates(
    *,
    quad_density: float,
    crease_angle: float,
    smooth_orient_iters: int,
    smooth_pos_iters: int,
    quality_retries: int,
) -> list[dict]:
    base = {
        "quad_density": float(quad_density),
        "crease_angle": float(crease_angle),
        "smooth_orient_iters": int(smooth_orient_iters),
        "smooth_pos_iters": int(smooth_pos_iters),
    }
    if quality_retries <= 0:
        return [base]
    variants = [
        {"quad_density": float(quad_density) * 0.92},
        {"crease_angle": float(crease_angle) - 10.0},
        {"smooth_orient_iters": int(smooth_orient_iters) + 2},
        {
            "quad_density": float(quad_density) * 0.88,
            "smooth_pos_iters": int(smooth_pos_iters) + 1,
        },
        {"crease_angle": float(crease_angle) + 10.0, "quad_density": float(quad_density) * 0.95},
    ]
    out = [base]
    for variant in variants[:quality_retries]:
        merged = dict(base)
        merged.update(variant)
        out.append(merged)
    return out


def _run_native_extract_raw(
    mesh: trimesh.Trimesh,
    field: np.ndarray,
    base_edge: float,
    *,
    crease_angle: float,
    align_to_boundaries: bool,
    smooth_orient_iters: int,
    smooth_pos_iters: int,
    pure_quad: bool,
) -> tuple[np.ndarray, np.ndarray]:
    nb = _try_native()
    if nb is None:
        raise RuntimeError("native Instant Meshes unavailable")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    v1, v2 = field[:, 0], field[:, 1]
    return _native_extract_with_retries(
        nb, v, f, v1, v2, base_edge,
        crease_angle=crease_angle,
        align_to_boundaries=align_to_boundaries,
        smooth_orient_iters=smooth_orient_iters,
        smooth_pos_iters=smooth_pos_iters,
        pure_quad=pure_quad,
    )


def _run_im_lite_extract_raw(
    mesh: trimesh.Trimesh,
    field: np.ndarray,
    out_dir: Path,
    *,
    target_quads: int | None,
    base_edge: float,
    smooth_orient_iters: int,
    smooth_pos_iters: int,
    bridge_seams: bool,
) -> tuple[np.ndarray, np.ndarray]:
    out_path = im_lite_extract(
        mesh, field, Path(out_dir),
        target_quads=target_quads,
        target_edge_length=base_edge if base_edge > 0 else None,
        smooth_orient_iters=smooth_orient_iters,
        smooth_pos_iters=max(smooth_pos_iters, 2),
        bridge_seams=bridge_seams,
    )
    qm = trimesh.load(out_path, process=False)
    return np.asarray(qm.vertices), np.asarray(qm.faces)



COVER_EDGE_FACTOR = 1.5
COVER_RETRY_SCALES = (0.5,)
COVER_RETRY_MAX_FACES = 60_000


def _uncovered_source_faces(quad_v, src_v, src_f, edge: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    if len(quad_v) == 0:
        return np.arange(len(src_f))
    tree = cKDTree(np.asarray(quad_v, dtype=np.float64))
    cents = src_v[src_f].mean(axis=1)
    dist = np.empty(len(cents), dtype=np.float64)
    for s in range(0, len(cents), 100_000):
        dist[s:s + 100_000], _ = tree.query(cents[s:s + 100_000], workers=1)
    return np.flatnonzero(dist > edge * COVER_EDGE_FACTOR)


def _face_patches(faces: np.ndarray) -> list[np.ndarray]:
    import scipy.sparse
    from scipy.sparse.csgraph import connected_components

    n = len(faces)
    if n == 0:
        return []
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    fid = np.tile(np.arange(n), 3)
    _, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    same = inv[order][1:] == inv[order][:-1]
    a, b = fid[order][:-1][same], fid[order][1:][same]
    adj = scipy.sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    k, lab = connected_components(adj, directed=False)
    return [np.flatnonzero(lab == i) for i in range(k)]


def tri_to_quads(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """三角 1 枚を、重心と辺の中点で四角 3 枚にする。穴も欠けも出ない全四角の保険。"""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.reshape(3, -1).T + len(v)
    mids = v[uniq].mean(axis=1)
    cen = np.arange(len(f)) + len(v) + len(uniq)
    m01, m12, m20 = inv[:, 0], inv[:, 1], inv[:, 2]
    q = np.vstack([
        np.stack([f[:, 0], m01, cen, m20], axis=1),
        np.stack([f[:, 1], m12, cen, m01], axis=1),
        np.stack([f[:, 2], m20, cen, m12], axis=1),
    ])
    return np.vstack([v, mids, v[f].mean(axis=1)]), q


def _patch_edge(pv, pf, edge):
    """部品の一番薄い向きの厚みの 1/3 を辺長にする。元の三角より細かくはしない。"""
    c = pv - pv.mean(0)
    try:
        axes = np.linalg.svd(c, full_matrices=False)[2]
        thin = float(np.ptp(c @ axes[-1]))
        wide = float(np.ptp(c @ axes[1]))
    except np.linalg.LinAlgError:
        thin = wide = edge
    src_edge = float(np.median(np.linalg.norm(pv[pf[:, 0]] - pv[pf[:, 1]], axis=1)))
    size = max(thin, wide * 0.25)
    return float(np.clip(size / 3.0, src_edge * 1.2, edge * COVER_RETRY_SCALES[0]))


def _patch_quads(src_v, src_f, field, patch, edge, native_kwargs):
    used, local = np.unique(src_f[patch], return_inverse=True)
    pv, pf = src_v[used], local.reshape(-1, 3)
    if _try_native() is None or not (64 <= len(patch) <= COVER_RETRY_MAX_FACES):
        qv, qf = tri_to_quads(pv, pf)
        return qv, qf, "fallback"
    e = _patch_edge(pv, pf, edge)
    try:
        sub = trimesh.Trimesh(vertices=pv, faces=pf, process=False)
        qv, qf = _run_native_extract_raw(sub, field[used], e, **native_kwargs)
        qv, qf = np.asarray(qv, dtype=np.float64), np.asarray(qf, dtype=np.int64)
    except Exception:
        qv = np.zeros((0, 3))
        qf = np.zeros((0, 4), dtype=np.int64)
    if qf.ndim != 2 or qf.shape[1] != 4 or len(qf) == 0:
        qv, qf = tri_to_quads(pv, pf)
        return qv, qf, "fallback"
    left = _uncovered_source_faces(qv, pv, pf, e)
    if len(left) == 0:
        return qv, qf, "retry"
    lv, lf = tri_to_quads(pv, pf[left])
    return np.vstack([qv, lv]), np.vstack([qf, lf + len(qv)]), "retry+fallback"


def fill_uncovered_with_quads(quad_v, quad_f, mesh, field, edge, native_kwargs):
    """四角が届かなかった元の部品を、細かい辺で四角化し直す。だめなら全四角分割で覆う。"""
    quad_v = np.asarray(quad_v, dtype=np.float64)
    quad_f = np.asarray(quad_f, dtype=np.int64)
    if quad_f.ndim != 2 or quad_f.shape[0] == 0:
        raise RuntimeError(
            "Instant Meshes extracted 0 faces. Clean this mesh before quad remesh."
        )
    src_v = np.asarray(mesh.vertices, dtype=np.float64)
    src_f = np.asarray(mesh.faces, dtype=np.int64)
    miss = _uncovered_source_faces(quad_v, src_v, src_f, edge)
    if len(miss) == 0:
        print("[im] coverage ok (uncovered=0)", flush=True)
        return quad_v, quad_f
    counts = {"retry": 0, "retry+fallback": 0, "fallback": 0}
    vs, fs, off = [quad_v], [quad_f], len(quad_v)
    for p in _face_patches(src_f[miss]):
        qv, qf, how = _patch_quads(src_v, src_f, field, miss[p], edge, native_kwargs)
        counts[how] += 1
        vs.append(qv)
        fs.append(qf + off)
        off += len(qv)
    print(
        f"[im] coverage uncovered={len(miss)} faces -> "
        f"patches {counts}",
        flush=True,
    )
    return np.vstack(vs), np.vstack(fs)


def extract_with_cross_field(
    mesh: trimesh.Trimesh,
    field: np.ndarray,
    out_dir: Path,
    *,
    target_quads: int | None = None,
    target_edge_length: float | None = None,
    grid_h: float | None = None,
    quad_density: float = 1.0,
    crease_angle: float = 75.0,
    align_to_boundaries: bool = True,
    smooth_orient_iters: int = 4,
    smooth_pos_iters: int = 2,
    pure_quad: bool = True,
    bridge_seams: bool = False,
    shrinkwrap: bool = True,
    shrinkwrap_passes: int = 3,
    shrinkwrap_reference_mesh: trimesh.Trimesh | None = None,
    stitch_small_holes: bool = True,
    max_hole_loop_edges: int = 4,
    fix_face_normals: bool = True,
    allow_im_lite_fallback: bool = False,
    vertex_color_export: bool = True,
    color_reference_mesh: trimesh.Trimesh | None = None,
    color_reference_path: Path | str | None = None,
    quality_retries: int = 0,
    quality_pass: float | None = None,
) -> Path:
    """GNN cross field -> Quad mesh (native IM or im-lite)."""
    if field.shape[-1] == 6:
        field = field.reshape(len(field), 2, 3)

    pass_threshold = float(quality_pass if quality_pass is not None else QUALITY_PASS)

    def _finalize(new_v: np.ndarray, new_f: np.ndarray) -> Path:
        from .vertex_color import colors_for_quad_export

        sw_ref = shrinkwrap_reference_mesh if shrinkwrap_reference_mesh is not None else mesh
        color_ref = (
            color_reference_mesh
            if color_reference_mesh is not None
            else shrinkwrap_reference_mesh
            if shrinkwrap_reference_mesh is not None
            else mesh
        )
        if shrinkwrap_reference_mesh is not None:
            print(
                f"[im] shrinkwrap reference=canonical ({len(sw_ref.vertices):,} verts) "
                f"| field mesh=proxy ({len(mesh.vertices):,} verts)",
                flush=True,
            )
        new_v, n_repaired = _repair_nonfinite_vertices(new_v, mesh, faces=new_f)
        if n_repaired:
            print(f"[im] repaired {n_repaired} non-finite quad vertices before shrinkwrap", flush=True)
        new_v, new_f = _sanitize_quad_mesh(new_v, new_f)
        if stitch_small_holes:
            new_v, new_f, n_stitch = _stitch_small_quad_holes(
                new_v, new_f, mesh, max_loop_edges=max_hole_loop_edges,
            )
            if n_stitch:
                print(f"[im] stitched {n_stitch} small quad holes (loop={max_hole_loop_edges} edges)", flush=True)
        if fix_face_normals:
            new_f, topo = clean_quad_topology(new_v, new_f, sw_ref)
            print(f"[im] quad cleanup {topo}", flush=True)
        lay = layout_for_stage_dir(out_dir)
        lay.ensure_dirs()
        pre_path = lay.retopo_quad_mesh_pre_shrinkwrap()
        _export_im_quad_obj(pre_path, new_v, new_f, vertex_colors=None)
        print(f"[im] saved pre-shrinkwrap (geometry only) -> {pre_path}", flush=True)
        if shrinkwrap:
            new_v, max_d, mean_d = shrinkwrap_vertices_to_mesh(
                new_v, sw_ref, faces=new_f, passes=shrinkwrap_passes,
            )
            print(
                f"[im] shrinkwrap done ({shrinkwrap_passes} pass max): "
                f"max_disp={max_d:.6f} mean_disp={mean_d:.6f}",
                flush=True,
            )
        out_path = lay.retopo_quad_mesh()
        q_colors = colors_for_quad_export(
            new_v,
            color_ref,
            ref_path=color_reference_path,
            vertex_color_export=vertex_color_export,
            quad_faces=new_f,
        )
        _export_im_quad_obj(out_path, new_v, new_f, vertex_colors=q_colors)
        from .mesh_io import COORD_YUP, write_coord_space_json

        write_coord_space_json(out_path.parent, COORD_YUP)
        return out_path

    def _extract_with_params(params: dict) -> tuple[np.ndarray, np.ndarray, float, dict, dict, float]:
        fitted_quads = target_quads
        if target_quads is not None:
            from .mesh_budget import suggest_target_quads
            fitted = int(suggest_target_quads(int(len(mesh.faces))))
            if int(target_quads) > fitted:
                print(
                    f"[im] target_quads {int(target_quads):,} -> {fitted:,} "
                    f"to match {len(mesh.faces):,} faces",
                    flush=True,
                )
                fitted_quads = fitted
        edge = _resolve_target_edge_length(
            mesh,
            field,
            target_quads=fitted_quads,
            target_edge_length=target_edge_length,
            grid_h=grid_h,
            quad_density=params["quad_density"],
        )
        nb = _try_native()
        if nb is not None:
            raw_v, raw_f = _run_native_extract_raw(
                mesh, field, edge,
                crease_angle=params["crease_angle"],
                align_to_boundaries=align_to_boundaries,
                smooth_orient_iters=params["smooth_orient_iters"],
                smooth_pos_iters=params["smooth_pos_iters"],
                pure_quad=pure_quad,
            )
        else:
            n_verts = len(mesh.vertices)
            if n_verts > _IM_LITE_MAX_VERTS:
                raise RuntimeError(
                    f"native IM unavailable and im_lite blocked for large mesh ({n_verts:,} verts). "
                    "Build native IM (scripts/build_native.sh) or decimate further."
                )
            raw_v, raw_f = _run_im_lite_extract_raw(
                mesh, field, out_dir,
                target_quads=target_quads,
                base_edge=edge,
                smooth_orient_iters=params["smooth_orient_iters"],
                smooth_pos_iters=params["smooth_pos_iters"],
                bridge_seams=bridge_seams,
            )
        if np.asarray(raw_f).ndim != 2 or np.asarray(raw_f).shape[0] == 0:
            raise RuntimeError(
                "Instant Meshes extracted 0 faces. Clean this mesh before quad remesh."
            )
        prep_v, prep_f = _prepare_quad_for_quality(
            raw_v, raw_f, mesh,
            stitch_small_holes=stitch_small_holes,
            max_hole_loop_edges=max_hole_loop_edges,
        )
        score, terms, stats = _evaluate_quad_quality(
            prep_v, prep_f, edge, pass_threshold=pass_threshold,
        )
        return prep_v, prep_f, score, terms, stats, edge

    candidates = _quality_param_candidates(
        quad_density=quad_density,
        crease_angle=crease_angle,
        smooth_orient_iters=smooth_orient_iters,
        smooth_pos_iters=smooth_pos_iters,
        quality_retries=quality_retries,
    )

    attempts: list[dict] = []
    best_v: np.ndarray | None = None
    best_f: np.ndarray | None = None
    best_score = float("inf")
    best_idx = 0

    for idx, params in enumerate(candidates):
        try:
            prep_v, prep_f, score, terms, stats, edge = _extract_with_params(params)
        except RuntimeError as e:
            attempts.append({
                "index": idx,
                "params": params,
                "error": str(e),
            })
            print(f"[im] quality attempt {idx + 1}/{len(candidates)} failed: {e}", flush=True)
            continue

        attempt = {
            "index": idx,
            "params": params,
            "score": score,
            "terms": terms,
            "stats": stats,
            "target_edge": edge,
        }
        attempts.append(attempt)
        print(
            f"[im] quality attempt {idx + 1}/{len(candidates)} "
            f"score={score:.4f} pole_ratio={terms['pole_ratio']:.4f} "
            f"passed={terms['passed']}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_v = prep_v
            best_f = prep_f
            best_idx = idx
        if terms["passed"]:
            break

    if best_v is None or best_f is None:
        n_verts = len(mesh.vertices)
        if allow_im_lite_fallback and n_verts <= _IM_LITE_MAX_VERTS and _try_native() is not None:
            import warnings

            warnings.warn(
                "native Instant Meshes failed all quality attempts; falling back to im_lite",
                stacklevel=2,
            )
            base = candidates[0]
            edge = _resolve_target_edge_length(
                mesh, field,
                target_quads=target_quads,
                target_edge_length=target_edge_length,
                grid_h=grid_h,
                quad_density=base["quad_density"],
            )
            raw_v, raw_f = _run_im_lite_extract_raw(
                mesh, field, out_dir,
                target_quads=target_quads,
                base_edge=edge,
                smooth_orient_iters=base["smooth_orient_iters"],
                smooth_pos_iters=base["smooth_pos_iters"],
                bridge_seams=bridge_seams,
            )
            best_v, best_f = _prepare_quad_for_quality(
                raw_v, raw_f, mesh,
                stitch_small_holes=stitch_small_holes,
                max_hole_loop_edges=max_hole_loop_edges,
            )
            best_score, terms, stats = _evaluate_quad_quality(
                best_v, best_f, edge, pass_threshold=pass_threshold,
            )
            best_idx = len(attempts)
            attempts.append({
                "index": best_idx,
                "params": base,
                "score": best_score,
                "terms": terms,
                "stats": stats,
                "target_edge": edge,
                "fallback": "im_lite",
            })
        elif attempts and all("error" in a for a in attempts):
            raise RuntimeError(attempts[-1]["error"])
        else:
            raise RuntimeError("all quality-driven extraction attempts failed")

    report_path = write_quality_report(
        out_dir, attempts, selected=best_idx, pass_threshold=pass_threshold,
    )
    print(
        f"[im] quality selected attempt {best_idx + 1}/{max(len(candidates), len(attempts))} "
        f"score={best_score:.4f} -> {report_path}",
        flush=True,
    )

    if os.environ.get("BRIEF153_FILL_UNCOVERED", "1") != "0":
        sel = attempts[best_idx]
        prm = sel["params"]
        best_v, best_f = fill_uncovered_with_quads(
            best_v, best_f, mesh, field, float(sel["target_edge"]),
            dict(
                crease_angle=prm["crease_angle"],
                align_to_boundaries=align_to_boundaries,
                smooth_orient_iters=prm["smooth_orient_iters"],
                smooth_pos_iters=prm["smooth_pos_iters"],
                pure_quad=True,
            ),
        )

    return _finalize(best_v, best_f)


def extract_quads_unified(
    mesh: trimesh.Trimesh,
    field: np.ndarray,
    out_dir: Path,
    *,
    extractor: str = "python",
    **kwargs,
) -> Path:
    if extractor == "im":
        return extract_with_cross_field(mesh, field, out_dir, **kwargs)
    if field.shape[-1] == 6:
        field = field.reshape(len(field), 2, 3)
    py_keys = {"target_quads", "grid_h", "bridge_seams", "vertex_color_export", "color_reference_mesh", "color_reference_path"}
    py_kwargs = {k: v for k, v in kwargs.items() if k in py_keys}
    if "color_reference_mesh" not in py_kwargs:
        py_kwargs["color_reference_mesh"] = kwargs.get("shrinkwrap_reference_mesh")
    return extract_quads_from_mesh(mesh, field, Path(out_dir), **py_kwargs)
