"""Post-SDF / pre-IM mesh repair — topology only (no groove cutting)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .config import DEGENERATE_HEIGHT, MERGE_PERCENT

CANONICAL_MANIFEST_VERSION = 1
CANONICAL_MANIFEST_NAME = "canonical.json"


def _ascii_temp(suffix: str) -> Path:
    p = Path(tempfile.gettempdir()) / "gnn_quad_retopo" / f"gnn_im_prep_{uuid.uuid4().hex[:8]}{suffix}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def should_skip_im_repair(stats: dict) -> tuple[bool, str]:
    """Skip PyMeshLab repair when topology is OK or noise is export-scale only."""
    if stats.get("topology_fast"):
        return False, "topology unchecked (fast)"
    nm_raw = stats.get("nonmanifold_edges")
    boundary_raw = stats.get("boundary_edges")
    if nm_raw is None or boundary_raw is None:
        return False, "topology unknown"
    nm = int(nm_raw)
    if nm == 0 and boundary_raw == 0 and stats.get("watertight") is True:
        return True, "non-manifold=0 watertight"
    if nm == 0:
        return True, "non-manifold=0"
    faces = int(stats.get("faces", 0))
    boundary = int(stats.get("boundary_edges") or 0)
    if faces < 100_000:
        return False, "small mesh"
    cap = min(128, max(64, int(faces * 5e-5)))
    if boundary == 0 and nm <= cap:
        return True, f"export noise (nm={nm} <= {cap})"
    return False, f"nm={nm} > {cap} or boundary={boundary}"


def write_canonical_manifest(
    run_root: Path,
    manifest_path: Path,
    canonical_path: Path,
    topology: dict,
    *,
    role: str = "sdf",
    repaired: bool = False,
    smoothed_alias: bool = False,
    active_path: Path | None = None,
    gnn_proxy_path: Path | None = None,
    gnn_proxy_topology: dict | None = None,
    gnn_proxy_reason: str | None = None,
) -> None:
    """Record canonical surface + topology for downstream stages."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel_canon = canonical_path.relative_to(run_root).as_posix()
    except ValueError:
        rel_canon = str(canonical_path)

    def _rel(p: Path) -> str:
        try:
            return p.relative_to(run_root).as_posix()
        except ValueError:
            return str(p)

    payload: dict[str, Any] = {
        "version": CANONICAL_MANIFEST_VERSION,
        "canonical": rel_canon,
        "role": role,
        "repaired": bool(repaired),
        "smoothed_alias_canonical": bool(smoothed_alias),
        "topology": dict(topology),
        "openings": [],
    }
    if active_path is not None:
        payload["active"] = _rel(active_path)
    if gnn_proxy_path is not None:
        payload["gnn_proxy"] = _rel(gnn_proxy_path)
    if gnn_proxy_topology is not None:
        payload["gnn_proxy_topology"] = dict(gnn_proxy_topology)
    if gnn_proxy_reason:
        payload["gnn_proxy_reason"] = gnn_proxy_reason
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_canonical_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def promote_canonical_copy(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


_FAST_TOPO_FACE_LIMIT = 2_000_000


def mesh_topology_stats(
    mesh: trimesh.Trimesh,
    *,
    fast: bool = False,
    force_check: bool = False,
) -> dict:
    n_f = int(len(mesh.faces))
    base = {"verts": int(len(mesh.vertices)), "faces": n_f}
    if not force_check and (fast or n_f >= _FAST_TOPO_FACE_LIMIT):
        base.update(
            watertight=None,
            euler=None,
            boundary_edges=None,
            nonmanifold_edges=None,
            topology_fast=True,
        )
        return base
    ec = np.bincount(mesh.edges_unique_inverse)
    return {
        **base,
        "watertight": bool(mesh.is_watertight),
        "euler": int(mesh.euler_number),
        "boundary_edges": int((ec == 1).sum()),
        "nonmanifold_edges": int((ec != 2).sum()),
    }


def mesh_topology_ok(stats: dict) -> bool:
    """True when topology was measured and is clean enough for GNN/IM."""
    if stats.get("topology_fast"):
        return False
    nm = stats.get("nonmanifold_edges")
    boundary = stats.get("boundary_edges")
    watertight = stats.get("watertight")
    if nm is None or boundary is None or watertight is None:
        return False
    return int(nm) == 0 and int(boundary) == 0 and bool(watertight)


def validate_mesh_topology(mesh: trimesh.Trimesh) -> dict:
    """Full edge/watertight stats even for large meshes."""
    return mesh_topology_stats(mesh, force_check=True)


def _split_nonmanifold(vertices, faces):
    """3枚以上が共有する辺で頂点を複製し、面は残したまま多様体にする。"""
    from collections import defaultdict

    faces = np.asarray(faces, dtype=np.int64).copy()
    vertices = np.asarray(vertices, dtype=np.float64)
    edge_users = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_users[(min(int(u), int(v)), max(int(u), int(v)))].append(fi)
    incident = defaultdict(list)
    for fi, f in enumerate(faces):
        for v in f:
            incident[int(v)].append(fi)

    def neighbors(v, fi):
        f = faces[fi]
        out = []
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            if v not in (a, b):
                continue
            users = edge_users[(min(a, b), max(a, b))]
            if len(users) != 2:
                continue
            other = users[0] if users[1] == fi else users[1]
            out.append(other)
        return out

    assign = {}
    extra = []
    next_id = len(vertices)
    for v, flist in incident.items():
        flist = list(dict.fromkeys(flist))
        seen = set()
        fans = []
        for fi in flist:
            if fi in seen:
                continue
            stack = [fi]
            seen.add(fi)
            fan = []
            while stack:
                cur = stack.pop()
                fan.append(cur)
                for nb in neighbors(v, cur):
                    if nb not in seen and v in faces[nb]:
                        seen.add(nb)
                        stack.append(nb)
            fans.append(fan)
        for fan in fans[1:]:
            vid = next_id
            next_id += 1
            extra.append(vertices[v])
            for fi in fan:
                assign[(fi, v)] = vid
    if not extra:
        return vertices, faces
    for (fi, v), vid in assign.items():
        tri = faces[fi]
        for k in range(3):
            if int(tri[k]) == v:
                tri[k] = vid
    return np.vstack([vertices, np.asarray(extra, dtype=np.float64)]), faces


def _fill_boundary_loops(vertices, faces, max_loop: int = 48, rounds: int = 6):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).copy()
    filled = 0
    for _ in range(rounds):
        edges = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uniq, _, cnt = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
        boundary = uniq[cnt == 1]
        if len(boundary) == 0:
            break
        adj = {}
        for a, b in boundary:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))
        seen = set()
        extra_v = []
        extra_f = []
        for start in adj:
            if start in seen:
                continue
            loop = [start]
            seen.add(start)
            prev, cur, ok = -1, start, True
            while True:
                nxts = [n for n in adj[cur] if n != prev]
                if not nxts:
                    ok = False
                    break
                nxt = nxts[0]
                if nxt == start:
                    break
                if nxt in seen or len(loop) >= max_loop:
                    ok = False
                    break
                seen.add(nxt)
                loop.append(nxt)
                prev, cur = cur, nxt
            if not ok or len(loop) < 3:
                continue
            ci = len(v) + len(extra_v)
            extra_v.append(v[loop].mean(axis=0))
            extra_f.extend((ci, loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop)))
            filled += 1
        if not extra_f:
            break
        v = np.vstack([v, np.asarray(extra_v, dtype=np.float64)])
        f = np.vstack([f, np.asarray(extra_f, dtype=np.int64)])
    return v, f, filled


IM_PREP_CACHE_VERSION = 1


def _im_prep_cache_path(mesh: trimesh.Trimesh, merge_percent, light: bool) -> Path | None:
    if os.environ.get("BRIEF153_IM_PREP_CACHE", "1") == "0":
        return None
    import hashlib
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    h.update(repr((IM_PREP_CACHE_VERSION, merge_percent, bool(light))).encode())
    root = Path(os.environ.get("BRIEF153_CACHE", tempfile.gettempdir())) / "brief153_im_prep"
    return root / (h.hexdigest() + ".npz")


def repair_mesh_for_im(
    mesh: trimesh.Trimesh,
    *,
    merge_percent: float | None = None,
    light: bool = False,
) -> tuple[trimesh.Trimesh, dict]:
    """Repair with an on-disk cache keyed by the input geometry."""
    path = _im_prep_cache_path(mesh, merge_percent, light)
    if path is not None and path.is_file():
        try:
            d = np.load(path, allow_pickle=False)
            out = trimesh.Trimesh(vertices=d["v"], faces=d["f"], process=False)
            stats = json.loads(str(d["stats"]))
            stats["cache"] = "hit"
            return out, stats
        except Exception:
            pass
    out, stats = _repair_mesh_for_im(mesh, merge_percent=merge_percent, light=light)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, v=np.asarray(out.vertices), f=np.asarray(out.faces),
                     stats=np.array(json.dumps(stats, default=str)))
        except Exception:
            pass
    return out, stats


def _repair_mesh_for_im(
    mesh: trimesh.Trimesh,
    *,
    merge_percent: float | None = None,
    light: bool = False,
) -> tuple[trimesh.Trimesh, dict]:
    """Repair non-manifold / degenerate geometry for GNN + Instant Meshes input."""
    before = mesh_topology_stats(mesh)
    work = mesh

    try:
        import pymeshlab as ml

        tmp = _ascii_temp(".obj")
        mesh.export(tmp)
        ms = ml.MeshSet()
        ms.load_new_mesh(str(tmp))

        if merge_percent is None:
            pct = 0.0 if light else (MERGE_PERCENT * 100)
        else:
            pct = float(merge_percent)
        if hasattr(ms, "meshing_merge_close_vertices") and pct > 0:
            ms.meshing_merge_close_vertices(threshold=ml.PercentageValue(float(pct)))
        if hasattr(ms, "meshing_remove_duplicate_faces"):
            ms.meshing_remove_duplicate_faces()
        if hasattr(ms, "meshing_repair_non_manifold_edges"):
            ms.meshing_repair_non_manifold_edges()
        if hasattr(ms, "meshing_repair_non_manifold_vertices"):
            ms.meshing_repair_non_manifold_vertices()
        if hasattr(ms, "meshing_remove_unreferenced_vertices"):
            ms.meshing_remove_unreferenced_vertices()
        if hasattr(ms, "meshing_re_orient_faces_coherently"):
            ms.meshing_re_orient_faces_coherently()

        ms.save_current_mesh(str(tmp))
        work = trimesh.load(tmp, force="mesh")
        if isinstance(work, trimesh.Scene):
            work = work.dump(concatenate=True)
        if tmp.exists():
            tmp.unlink()
    except Exception as e:
        before["repair_error"] = str(e)
        return mesh, before

    try:
        mask = work.nondegenerate_faces(height=DEGENERATE_HEIGHT)
        work.update_faces(mask)
    except Exception:
        pass
    work.remove_unreferenced_vertices()
    work.process(validate=True)

    after = mesh_topology_stats(work)
    after["before"] = before
    b0 = int(before.get("boundary_edges") or 0)
    n0 = int(before.get("nonmanifold_edges") or 0)
    b1 = int(after.get("boundary_edges") or 0)
    n1 = int(after.get("nonmanifold_edges") or 0)
    if b0 + n0 > 0 and (b1 + n1) > (b0 + n0):
        before["repair_rejected"] = f"opened edges {b0 + n0} -> {b1 + n1}"
        work = mesh
    n_before_split = len(work.vertices)
    sv, sf = _split_nonmanifold(np.asarray(work.vertices), np.asarray(work.faces))
    if len(sv) != n_before_split:
        work = trimesh.Trimesh(vertices=sv, faces=sf, process=False)
        work.remove_unreferenced_vertices()
    fv, ff, n_holes = _fill_boundary_loops(np.asarray(work.vertices), np.asarray(work.faces))
    if n_holes:
        work = trimesh.Trimesh(vertices=fv, faces=ff, process=False)
        work.remove_unreferenced_vertices()
    after = mesh_topology_stats(work)
    after["before"] = before
    after["split_added_verts"] = int(len(sv) - n_before_split)
    after["holes_filled"] = int(n_holes)
    return work, after


def pymeshlab_decimate_for_im(ms, target_faces: int) -> None:
    """Quadric decimation + immediate topology repair on same MeshSet."""
    kwargs = dict(
        targetfacenum=int(target_faces),
        qualitythr=0.6,
        preservenormal=True,
        preserveboundary=True,
        optimalplacement=True,
        planarquadric=True,
        autoclean=True,
        planarweight=0.02,
    )
    try:
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", **kwargs)
    except TypeError:
        kwargs.pop("planarweight", None)
        try:
            ms.apply_filter("meshing_decimation_quadric_edge_collapse", **kwargs)
        except TypeError:
            ms.apply_filter(
                "meshing_decimation_quadric_edge_collapse",
                targetfacenum=int(target_faces),
            )

    for fn in (
        "meshing_remove_duplicate_faces",
        "meshing_repair_non_manifold_edges",
        "meshing_repair_non_manifold_vertices",
        "meshing_remove_unreferenced_vertices",
        "meshing_re_orient_faces_coherently",
    ):
        if hasattr(ms, fn):
            getattr(ms, fn)()
