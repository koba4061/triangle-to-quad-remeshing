"""Vertex color extraction, validation, and KDTree transfer for quad export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

try:
    from pykdtree.kdtree import KDTree as _FastKDTree
    _USE_PYKDTREE = True
except ImportError:
    from scipy.spatial import cKDTree as _FastKDTree
    _USE_PYKDTREE = False

from .glb_vertex_color import load_glb_vertex_colors
from .mesh_io import COORD_YUP, positions_to_yup, resolve_coord_space

# KDTree on refs above this cap can exhaust RAM (e.g. 14M-vert GLB → swap hang).
_KDTREE_REF_VERTEX_CAP = 1_000_000
# Surface samples for remeshed targets (SDF); avoids wrong colors from vertex NN.
_SURFACE_SAMPLE_COUNT = 500_000
# Closest-triangle transfer chunk size (GLB ~2M faces).
_CLOSEST_SURFACE_CHUNK = 50_000
# Light vertex-color smoothing after transfer (0 = off).
_COLOR_SMOOTH_ITERS = 2
_COLOR_SMOOTH_ALPHA = 0.28
# Use surface transfer when target vertex count differs from ref by this fraction.
_SURFACE_TRANSFER_VERT_RATIO = 0.05
# Weld duplicate OBJ verts before KDTree when target exceeds this count.
_KDTREE_WELD_TARGET_MIN = 500_000
# Minimum per-channel spread to treat colors as informative (not flat white).
_MIN_COLOR_SPREAD = 1e-3


def _colors_valid(rgb: np.ndarray) -> bool:
    if rgb.size == 0:
        return False
    spread = float(np.max(rgb) - np.min(rgb))
    return spread >= _MIN_COLOR_SPREAD


def extract_vertex_colors(mesh: trimesh.Trimesh) -> tuple[np.ndarray, bool]:
    """Return (N,3) RGB in [0,1] and whether colors are valid boundary guides."""
    n_v = len(mesh.vertices)
    fallback = np.ones((n_v, 3), dtype=np.float32)
    vc = None
    visual = getattr(mesh, "visual", None)
    if visual is not None and hasattr(visual, "vertex_colors"):
        raw = visual.vertex_colors
        if raw is not None and len(raw) > 0:
            vc = np.asarray(raw)

    if vc is None or len(vc) != n_v:
        return fallback, False
    if vc.ndim != 2 or vc.shape[1] < 3:
        return fallback, False

    rgb = vc[:, :3].astype(np.float64)
    mx = float(rgb.max()) if rgb.size else 0.0
    if np.issubdtype(rgb.dtype, np.integer) or mx > 1.0 + 1e-6:
        rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    if not _colors_valid(rgb):
        return fallback.astype(np.float32), False
    return rgb.astype(np.float32), True


def read_reference_colors(path: Path | str) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load reference positions + RGB in Y-up. GLB uses direct COLOR_0 reader (F32_VEC3 etc.)."""
    path = Path(path)
    if path.suffix.lower() == ".glb":
        pos, rgb, has = load_glb_vertex_colors(path)
        if has and len(pos) > _KDTREE_REF_VERTEX_CAP:
            step = max(1, len(pos) // _KDTREE_REF_VERTEX_CAP)
            pos, rgb = pos[::step], rgb[::step]
        if has:
            return pos, rgb, True

    mesh = _load_mesh_path(path)
    if mesh is None:
        return np.empty((0, 3)), np.empty((0, 3)), False
    rgb, has = extract_vertex_colors(mesh)
    ref_space = resolve_coord_space(path)
    pos = positions_to_yup(mesh.vertices, ref_space)
    if not has:
        return pos, rgb, False
    return pos, rgb, True


def _should_use_surface_transfer(n_target: int, n_ref: int) -> bool:
    if n_ref <= 0:
        return False
    return abs(n_target - n_ref) / float(n_ref) > _SURFACE_TRANSFER_VERT_RATIO


def _reference_mesh_with_colors(
    ref_path: Path | str | None,
    ref_pos: np.ndarray,
    ref_rgb: np.ndarray,
    ref_mesh: trimesh.Trimesh | None,
) -> tuple[trimesh.Trimesh | None, np.ndarray | None]:
    """Build a reference trimesh aligned with ref_pos/ref_rgb for surface sampling."""
    ref_pos = np.asarray(ref_pos, dtype=np.float64)
    ref_rgb = np.asarray(ref_rgb, dtype=np.float64)
    mesh = ref_mesh
    if mesh is None and ref_path is not None:
        mesh = _load_mesh_path(Path(ref_path))
    if mesh is None or len(mesh.faces) == 0:
        return None, None
    mesh = mesh.copy()
    if len(mesh.vertices) == len(ref_pos):
        mesh.vertices = ref_pos
    else:
        space = resolve_coord_space(Path(ref_path)) if ref_path else COORD_YUP
        mesh.vertices = positions_to_yup(mesh.vertices, space)
    if len(ref_rgb) != len(mesh.vertices):
        return None, None
    return mesh, ref_rgb


def _load_mesh_path(path: Path) -> trimesh.Trimesh | None:
    try:
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return mesh
    except Exception:
        return None


def colors_for_mesh_features(
    target_mesh: trimesh.Trimesh,
    ref_mesh: trimesh.Trimesh,
    *,
    ref_path: Path | str | None = None,
    target_path: Path | str | None = None,
    target_space: str | None = None,
    force_surface_transfer: bool = False,
) -> tuple[np.ndarray, bool]:
    """Nearest-neighbor RGB transfer from ref mesh to target vertices for GNN features."""
    n_v = len(target_mesh.vertices)
    fallback = np.ones((n_v, 3), dtype=np.float32)

    ref_pos: np.ndarray | None = None
    ref_rgb: np.ndarray | None = None
    has = False
    if ref_path is not None:
        ref_pos, ref_rgb, has = read_reference_colors(ref_path)
    if not has:
        ref_rgb, has = extract_vertex_colors(ref_mesh)
        if has:
            ref_space = (
                resolve_coord_space(Path(ref_path))
                if ref_path is not None
                else COORD_YUP
            )
            ref_pos = positions_to_yup(ref_mesh.vertices, ref_space)

    if not has or ref_pos is None or ref_rgb is None:
        return fallback, False
    t_space = target_space
    if t_space is None and target_path is not None:
        t_space = resolve_coord_space(Path(target_path))
    if t_space is None:
        t_space = COORD_YUP
    try:
        use_surface = force_surface_transfer or _should_use_surface_transfer(
            len(target_mesh.vertices), len(ref_pos),
        )
        if use_surface:
            ref_m: trimesh.Trimesh | None = None
            ref_c: np.ndarray | None = None
            if ref_path is not None:
                p = Path(ref_path)
                ref_m = _load_mesh_path(p)
                if ref_m is not None and len(ref_m.faces) > 0:
                    if p.suffix.lower() == ".glb":
                        from .glb_vertex_color import load_glb_vertex_colors

                        pos, rgb, has_glb = load_glb_vertex_colors(p)
                        if has_glb and len(pos) == len(ref_m.vertices):
                            ref_m = ref_m.copy()
                            ref_m.vertices = np.asarray(pos, dtype=np.float64)
                            ref_c = np.asarray(rgb, dtype=np.float64)
                    else:
                        ref_m, ref_c = _reference_mesh_with_colors(
                            ref_path, ref_pos, ref_rgb, ref_mesh,
                        )
            if ref_m is not None and ref_c is not None:
                try:
                    colors = transfer_vertex_colors_closest_surface(
                        target_mesh.vertices,
                        ref_m.vertices,
                        ref_m.faces,
                        ref_c,
                        target_space=t_space,
                        ref_space=COORD_YUP,
                    )
                except Exception:
                    colors = transfer_vertex_colors_surface_sampled(
                        target_mesh.vertices,
                        ref_m.vertices,
                        ref_m.faces,
                        ref_c,
                        target_space=t_space,
                        ref_space=COORD_YUP,
                    )
                colors = smooth_vertex_colors(target_mesh, colors)
            else:
                colors = transfer_vertex_colors_nearest(
                    target_mesh.vertices,
                    ref_pos,
                    ref_rgb,
                    target_space=t_space,
                    ref_space=COORD_YUP,
                )
        else:
            colors = transfer_vertex_colors_nearest(
                target_mesh.vertices,
                ref_pos,
                ref_rgb,
                target_space=t_space,
                ref_space=COORD_YUP,
            )
    except Exception:
        return fallback, False
    if len(colors) != n_v or not _colors_valid(colors):
        return fallback, False
    return colors.astype(np.float32), True


def resolve_feature_vertex_colors(
    target_mesh: trimesh.Trimesh,
    ref_paths: list[tuple[str, Path]],
    *,
    target_path: Path | str | None = None,
    target_space: str | None = None,
) -> tuple[np.ndarray, bool, str]:
    """Resolve per-vertex RGB for GNN features: active mesh, then ref path fallbacks."""
    n_v = len(target_mesh.vertices)
    fallback = np.ones((n_v, 3), dtype=np.float32)
    t_space = target_space
    if t_space is None and target_path is not None:
        t_space = resolve_coord_space(Path(target_path))
    if t_space is None:
        t_space = COORD_YUP

    colors, has = extract_vertex_colors(target_mesh)
    if has:
        return colors, True, "active"

    seen: set[str] = set()
    for label, path in ref_paths:
        p = Path(path)
        key = str(p.resolve()) if p.is_file() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if not p.is_file():
            continue
        ref_mesh = _load_mesh_path(p)
        if ref_mesh is not None:
            colors, has = colors_for_mesh_features(
                target_mesh, ref_mesh, ref_path=p, target_space=t_space,
            )
            if has and len(colors) == n_v and _colors_valid(colors):
                return colors.astype(np.float32), True, f"{label}_transfer"
        ref_pos, ref_rgb, has = read_reference_colors(p)
        if has:
            try:
                colors = transfer_vertex_colors_nearest(
                    target_mesh.vertices,
                    ref_pos,
                    ref_rgb,
                    target_space=t_space,
                    ref_space=COORD_YUP,
                )
            except Exception:
                continue
            if len(colors) == n_v and _colors_valid(colors):
                return colors.astype(np.float32), True, f"{label}_transfer"
        if ref_mesh is None:
            continue
        colors, has = colors_for_mesh_features(
            target_mesh, ref_mesh, ref_path=p, target_space=t_space
        )
        if has and len(colors) == n_v:
            return colors.astype(np.float32), True, f"{label}_transfer"

    return fallback, False, "none"


def resolve_vertex_color_guide(
    mesh: trimesh.Trimesh,
    feat: dict[str, np.ndarray] | None,
    *,
    vertex_color_guide: bool = True,
) -> tuple[np.ndarray, bool]:
    """Combine CLI flag with stored features and live mesh colors."""
    if not vertex_color_guide:
        n_v = len(mesh.vertices)
        return np.ones((n_v, 3), dtype=np.float32), False

    if feat is not None:
        stored = feat.get("vertex_colors")
        has_flag = bool(feat.get("has_vertex_colors", False))
        if stored is not None and len(stored) == len(mesh.vertices) and has_flag:
            return np.asarray(stored, dtype=np.float32), True

    return extract_vertex_colors(mesh)


def transfer_vertex_colors_nearest(
    target_positions: np.ndarray,
    ref_positions: np.ndarray,
    ref_colors: np.ndarray,
    *,
    target_space: str = COORD_YUP,
    ref_space: str = COORD_YUP,
    ref_vertex_cap: int = _KDTREE_REF_VERTEX_CAP,
    weld_target: bool = True,
) -> np.ndarray:
    """Nearest-neighbor RGB transfer from reference mesh vertices (Y-up aligned)."""
    ref_positions = positions_to_yup(ref_positions, ref_space)
    ref_colors = np.asarray(ref_colors, dtype=np.float64)
    target_positions = positions_to_yup(target_positions, target_space)
    n_ref = len(ref_positions)
    if n_ref > ref_vertex_cap > 0:
        rng = np.random.default_rng(0)
        pick = rng.choice(n_ref, ref_vertex_cap, replace=False)
        print(
            f"[vertex-color] KDTree ref random subsample {n_ref:,} -> {len(pick):,}",
            flush=True,
        )
        ref_positions = ref_positions[pick]
        ref_colors = ref_colors[pick]
    n_ref = len(ref_positions)
    n_tgt = len(target_positions)
    inverse: np.ndarray | None = None
    if weld_target and n_tgt >= _KDTREE_WELD_TARGET_MIN:
        try:
            unique_pos, inverse = trimesh.grouping.merge_vertices(
                target_positions, digits_vertex=5
            )
            if len(unique_pos) < int(n_tgt * 0.92):
                print(
                    f"[vertex-color] weld target {n_tgt:,} -> {len(unique_pos):,} for KDTree",
                    flush=True,
                )
                target_positions = unique_pos
                n_tgt = len(target_positions)
            else:
                inverse = None
        except Exception:
            inverse = None
    if n_ref >= 500_000 or n_tgt >= 500_000:
        backend = "pykdtree" if _USE_PYKDTREE else "scipy"
        print(
            f"[vertex-color] KDTree ({backend}) build/query ref={n_ref:,} target={n_tgt:,}",
            flush=True,
        )
    tree = _FastKDTree(ref_positions)
    chunk = 500_000

    def _query_chunk(pts: np.ndarray) -> np.ndarray:
        _, idx = tree.query(pts)
        return np.asarray(idx, dtype=np.int64).reshape(-1)

    if n_tgt <= chunk:
        idx = _query_chunk(target_positions)
        colors = np.clip(ref_colors[idx], 0.0, 1.0).astype(np.float32)
    else:
        colors = np.empty((n_tgt, 3), dtype=np.float32)
        for start in range(0, n_tgt, chunk):
            end = min(start + chunk, n_tgt)
            idx = _query_chunk(target_positions[start:end])
            colors[start:end] = np.clip(ref_colors[idx], 0.0, 1.0)
            done = end
            if done == n_tgt or done % chunk == 0:
                pct = 100.0 * done / n_tgt
                print(
                    f"[vertex-color] KDTree query {done:,}/{n_tgt:,} ({pct:.0f}%)",
                    flush=True,
                )
    if inverse is not None:
        return colors[inverse].astype(np.float32)
    return colors.astype(np.float32)


def smooth_vertex_colors(
    mesh: trimesh.Trimesh,
    colors: np.ndarray,
    *,
    iterations: int = _COLOR_SMOOTH_ITERS,
    alpha: float = _COLOR_SMOOTH_ALPHA,
) -> np.ndarray:
    """Light Laplacian smooth on per-vertex RGB to reduce transfer noise."""
    if iterations <= 0 or alpha <= 0.0:
        return np.asarray(colors, dtype=np.float32)
    out = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    n = len(out)
    if n == 0 or len(mesh.vertices) != n:
        return out.astype(np.float32)
    try:
        adj = mesh.vertex_neighbors
    except Exception:
        return out.astype(np.float32)
    for _ in range(int(iterations)):
        nxt = out.copy()
        for i in range(n):
            nb = adj[i]
            if not nb:
                continue
            nxt[i] = (1.0 - alpha) * out[i] + alpha * out[nb].mean(axis=0)
        out = np.clip(nxt, 0.0, 1.0)
    return out.astype(np.float32)


def transfer_vertex_colors_closest_surface(
    target_positions: np.ndarray,
    ref_vertices: np.ndarray,
    ref_faces: np.ndarray,
    ref_colors: np.ndarray,
    *,
    target_space: str = COORD_YUP,
    ref_space: str = COORD_YUP,
    chunk: int = _CLOSEST_SURFACE_CHUNK,
) -> np.ndarray:
    """Barycentric color transfer from closest point on reference triangles."""

    ref_vertices = positions_to_yup(np.asarray(ref_vertices, dtype=np.float64), ref_space)
    ref_faces = np.asarray(ref_faces, dtype=np.int64)
    ref_colors = np.clip(np.asarray(ref_colors, dtype=np.float64), 0.0, 1.0)
    if len(ref_colors) != len(ref_vertices):
        raise ValueError("ref_colors length must match ref_vertices")
    target_positions = positions_to_yup(
        np.asarray(target_positions, dtype=np.float64), target_space,
    )
    from scipy.spatial import cKDTree
    from .fast_closest import closest_on_faces

    n_tgt = len(target_positions)
    print(
        f"[vertex-color] closest-surface transfer faces={len(ref_faces):,} "
        f"target={n_tgt:,}",
        flush=True,
    )
    tri = ref_vertices[ref_faces]
    tree = cKDTree(tri.mean(axis=1))
    tri_id, bc = closest_on_faces(target_positions, tri, tree, k=8)
    tri_rgb = ref_colors[ref_faces[tri_id]]
    colors = np.clip((tri_rgb * bc[:, :, None]).sum(axis=1), 0.0, 1.0)
    bad = ~np.isfinite(colors).all(axis=1)
    if np.any(bad):
        colors[bad] = tri_rgb[bad, 0]
    print(f"[vertex-color] closest-surface {n_tgt:,}/{n_tgt:,} (100%)", flush=True)
    return colors.astype(np.float32)


def transfer_vertex_colors_surface_sampled(
    target_positions: np.ndarray,
    ref_vertices: np.ndarray,
    ref_faces: np.ndarray,
    ref_colors: np.ndarray,
    *,
    target_space: str = COORD_YUP,
    ref_space: str = COORD_YUP,
    n_samples: int = _SURFACE_SAMPLE_COUNT,
) -> np.ndarray:
    """Transfer colors via surface samples + NN (for remeshed targets with new topology)."""
    from trimesh.triangles import points_to_barycentric

    ref_vertices = positions_to_yup(np.asarray(ref_vertices, dtype=np.float64), ref_space)
    ref_faces = np.asarray(ref_faces, dtype=np.int64)
    ref_colors = np.clip(np.asarray(ref_colors, dtype=np.float64), 0.0, 1.0)
    if len(ref_colors) != len(ref_vertices):
        raise ValueError("ref_colors length must match ref_vertices")
    target_positions = positions_to_yup(np.asarray(target_positions, dtype=np.float64), target_space)
    mesh = trimesh.Trimesh(vertices=ref_vertices, faces=ref_faces, process=False)
    n_samples = int(min(max(50_000, n_samples), max(50_000, len(ref_faces) * 2)))
    print(
        f"[vertex-color] surface transfer faces={len(ref_faces):,} "
        f"samples={n_samples:,} target={len(target_positions):,}",
        flush=True,
    )
    samples, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    tri = ref_vertices[ref_faces[face_idx]]
    bc = points_to_barycentric(tri, samples)
    tri_rgb = ref_colors[ref_faces[face_idx]]
    sample_rgb = (tri_rgb * bc[:, :, None]).sum(axis=1).astype(np.float32)
    tree = _FastKDTree(samples)
    n_tgt = len(target_positions)
    chunk = 500_000
    if n_tgt <= chunk:
        _, idx = tree.query(target_positions)
        return np.clip(sample_rgb[idx], 0.0, 1.0).astype(np.float32)
    colors = np.empty((n_tgt, 3), dtype=np.float32)
    for start in range(0, n_tgt, chunk):
        end = min(start + chunk, n_tgt)
        _, idx = tree.query(target_positions[start:end])
        colors[start:end] = np.clip(sample_rgb[idx], 0.0, 1.0)
    return colors.astype(np.float32)


def colors_for_quad_export(
    quad_vertices: np.ndarray,
    ref_mesh: trimesh.Trimesh | None = None,
    *,
    ref_path: Path | str | None = None,
    vertex_color_export: bool = True,
    target_space: str = COORD_YUP,
    quad_faces: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return per-quad-vertex RGB for OBJ export, or None if unavailable."""
    if not vertex_color_export:
        return None
    try:
        ref_m, ref_c = _quad_color_reference_mesh(ref_path, ref_mesh)
        if ref_m is None or ref_c is None:
            return None
        colors = transfer_vertex_colors_closest_surface(
            quad_vertices,
            ref_m.vertices,
            ref_m.faces,
            ref_c,
            target_space=target_space,
            ref_space=COORD_YUP,
        )
        if quad_faces is not None and len(quad_faces) > 0:
            f = np.asarray(quad_faces, dtype=np.int64)
            v_yup = positions_to_yup(np.asarray(quad_vertices, dtype=np.float64), target_space)
            if f.ndim == 2 and f.shape[1] == 4:
                tris = np.vstack([f[:, :3], f[:, [0, 2, 3]]])
            else:
                tris = f
            tm = trimesh.Trimesh(vertices=v_yup, faces=tris, process=False)
            colors = smooth_vertex_colors(tm, colors)
        return colors
    except Exception:
        return None


def _quad_color_reference_mesh(
    ref_path: Path | str | None,
    ref_mesh: trimesh.Trimesh | None,
) -> tuple[trimesh.Trimesh | None, np.ndarray | None]:
    """Full-color reference mesh for final quad export (no GLB vertex subsampling)."""
    if ref_path is not None:
        p = Path(ref_path)
        if p.suffix.lower() == ".glb":
            ref_m = _load_mesh_path(p)
            if ref_m is None or len(ref_m.faces) == 0:
                return None, None
            pos, rgb, has = load_glb_vertex_colors(p)
            if not has or len(pos) != len(ref_m.vertices):
                return None, None
            ref_m = ref_m.copy()
            ref_m.vertices = np.asarray(pos, dtype=np.float64)
            return ref_m, np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
        ref_pos, ref_rgb, has = read_reference_colors(p)
        if has:
            return _reference_mesh_with_colors(p, ref_pos, ref_rgb, ref_mesh)
    if ref_mesh is not None:
        ref_rgb, has = extract_vertex_colors(ref_mesh)
        if not has:
            return None, None
        ref_m = ref_mesh.copy()
        ref_m.vertices = positions_to_yup(ref_m.vertices, COORD_YUP)
        return ref_m, np.clip(np.asarray(ref_rgb, dtype=np.float64), 0.0, 1.0)
    return None, None


def _sample_image(img, uv):
    pic = np.asarray(img)
    if pic.ndim == 2:
        pic = pic[..., None]
    if pic.ndim != 3:
        return None
    h, w = pic.shape[:2]
    u = np.clip(np.asarray(uv[:, 0], dtype=np.float64), 0.0, 0.999)
    v = np.clip(1.0 - np.asarray(uv[:, 1], dtype=np.float64), 0.0, 0.999)
    x = (u * (w - 1)).astype(np.int32)
    y = (v * (h - 1)).astype(np.int32)
    rgb = pic[y, x, :3].astype(np.float64)
    if rgb.size and float(rgb.max()) > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _texture_vertex_rgb(mesh: trimesh.Trimesh) -> np.ndarray | None:
    vis = getattr(mesh, "visual", None)
    uv = getattr(vis, "uv", None) if vis is not None else None
    mat = getattr(vis, "material", None) if vis is not None else None
    img = None
    if mat is not None:
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if uv is None or img is None:
        return None
    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) == len(mesh.vertices):
        return _sample_image(img, uv)
    if len(uv) != len(mesh.faces) * 3:
        return None
    samples = _sample_image(img, uv)
    if samples is None:
        return None
    rgb = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    acc = np.zeros(len(mesh.vertices), dtype=np.float64)
    flat = np.asarray(mesh.faces, dtype=np.int64).reshape(-1)
    np.add.at(rgb, flat, samples[:, :3])
    np.add.at(acc, flat, 1.0)
    ok = acc > 0
    rgb[ok] /= acc[ok, None]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _source_rgb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    pos, rgb, has = load_glb_vertex_colors(path)
    if has and len(pos) == len(loaded.vertices):
        return np.asarray(pos, dtype=np.float64), np.asarray(loaded.faces, dtype=np.int64), np.clip(rgb, 0.0, 1.0)
    colors, ok = extract_vertex_colors(loaded)
    if not ok:
        sampled = _texture_vertex_rgb(loaded)
        if sampled is None:
            raise RuntimeError("元の GLB に頂点カラーもベースカラーも無い")
        colors = sampled
    return (
        np.asarray(loaded.vertices, dtype=np.float64),
        np.asarray(loaded.faces, dtype=np.int64),
        np.asarray(colors, dtype=np.float32),
    )


def _read_poly_obj(path: Path):
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = [int(token.split("/")[0]) for token in line.split()[1:]]
                if len(idx) >= 3 and len(set(idx)) == len(idx):
                    faces.append(idx)
    if not verts or not faces:
        raise RuntimeError("四角 OBJ に面が無い: %s" % path)
    return np.asarray(verts, dtype=np.float64), faces


def write_quad_vertex_colors(quad_path, src_path, out_path) -> None:
    """IM の四角を保ったまま、元 GLB の色を頂点カラーだけにする。"""
    quad_path = Path(quad_path)
    src_path = Path(src_path)
    out_path = Path(out_path)
    verts, faces = _read_poly_obj(quad_path)
    ref_v, ref_f, ref_c = _source_rgb(src_path)
    print("vertex color quads=%d src_faces=%d" % (len(faces), len(ref_f)), flush=True)
    colors = transfer_vertex_colors_closest_surface(
        verts, ref_v, ref_f, ref_c, target_space=COORD_YUP, ref_space=COORD_YUP,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        for (x, y, z), (r, g, b) in zip(verts, colors):
            handle.write("v %.8f %.8f %.8f %.6f %.6f %.6f\n" % (x, y, z, r, g, b))
        for face in faces:
            handle.write("f " + " ".join(str(i) for i in face) + "\n")
    print("vertex color obj", out_path, flush=True)
