"""メッシュ読み込み（座標系を処理用 Y-up に統一）"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import trimesh

from .coords import blender_zup_to_yup

COORD_YUP = "yup"
COORD_BLENDER_ZUP = "blender_zup"

# Display-only *_blender.obj is also pipeline Y-up; coord_space.json applies.
_BLENDER_EXPORT_OBJ_SUFFIXES: tuple[str, ...] = ()


def _is_blender_export_obj(path: Path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(sfx) for sfx in _BLENDER_EXPORT_OBJ_SUFFIXES)


def resolve_coord_space(path: Path) -> str:
    """Return COORD_YUP or COORD_BLENDER_ZUP for a mesh file path."""
    path = Path(path)
    # Blender Z-up OBJ (prep/coord_space.json applies to cleaned_mesh etc. only).
    if _is_blender_export_obj(path):
        return COORD_BLENDER_ZUP
    for directory in (path.parent, path.parent.parent, path.parent.parent.parent):
        if directory == directory.parent:
            break
        meta = directory / "coord_space.json"
        if meta.is_file():
            try:
                space = json.loads(meta.read_text(encoding="utf-8")).get("space", COORD_YUP)
                if space in (COORD_YUP, COORD_BLENDER_ZUP):
                    return space
            except Exception:
                pass
    return COORD_YUP


def vertices_for_blender_obj_export(vertices: np.ndarray) -> np.ndarray:
    """Map pipeline Y-up vertices to Blender Z-up for OBJ files viewed in Blender."""
    from .config import BLENDER_ZUP_EXPORT
    from .coords import yup_to_blender_zup

    v = np.asarray(vertices, dtype=np.float64)
    return yup_to_blender_zup(v) if BLENDER_ZUP_EXPORT else v


def preview_export_enabled() -> bool:
    """Optional PLY/GLB viewport previews (off by default; not used in final deliverable)."""
    return os.environ.get("RETOPO_PREVIEW_EXPORT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def obj_has_vertex_colors(path: Path) -> bool:
    """True if OBJ has xyzrgb vertex lines (v x y z r g b)."""
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if parts and parts[0] == "v" and len(parts) >= 7:
                    return True
                if parts and parts[0] == "v" and len(parts) == 4:
                    return False
    except Exception:
        pass
    return False


def rewrite_obj_as_yup(path: Path, mesh: trimesh.Trimesh | None = None) -> trimesh.Trimesh:
    """Rewrite Blender Z-up OBJ on disk to pipeline Y-up (prep/proxy/cleaned)."""
    path = Path(path)
    if mesh is None:
        m = trimesh.load(path, force="mesh", process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        mesh = m.copy()
        mesh.vertices = blender_zup_to_yup(np.asarray(mesh.vertices, dtype=np.float64))
    else:
        mesh = mesh.copy()
    with path.open("w", encoding="utf-8") as fh:
        for x, y, z in mesh.vertices:
            fh.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for f in mesh.faces:
            fh.write(f"f {' '.join(str(int(i) + 1) for i in f)}\n")
    return mesh


def export_yup_colored_mesh(
    path_obj: Path,
    vertices_yup: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
) -> None:
    """Write pipeline Y-up OBJ with per-vertex rgb (v x y z r g b)."""
    path_obj = Path(path_obj)
    vertices_yup = np.asarray(vertices_yup, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    colors = np.clip(np.asarray(vertex_colors, dtype=np.float64), 0.0, 1.0)
    with path_obj.open("w", encoding="utf-8") as fh:
        for (x, y, z), (r, g, b) in zip(vertices_yup, colors):
            fh.write(f"v {x:.8f} {y:.8f} {z:.8f} {r:.6f} {g:.6f} {b:.6f}\n")
        for f in faces:
            fh.write(f"f {' '.join(str(int(i) + 1) for i in f)}\n")


def export_yup_mesh(
    path_obj: Path,
    vertices_yup: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Write pipeline Y-up OBJ (geometry only)."""
    path_obj = Path(path_obj)
    vertices_yup = np.asarray(vertices_yup, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    with path_obj.open("w", encoding="utf-8") as fh:
        for x, y, z in vertices_yup:
            fh.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for f in faces:
            fh.write(f"f {' '.join(str(int(i) + 1) for i in f)}\n")


def export_blender_mesh(
    path_obj: Path,
    vertices_yup: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Write Blender-view OBJ (pipeline Y-up geometry)."""
    export_yup_mesh(path_obj, vertices_yup, faces)


def export_blender_colored_mesh(
    path_obj: Path,
    vertices_yup: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
) -> None:
    """Write Blender-view OBJ (pipeline Y-up xyzrgb; import at rotation 0)."""
    path_obj = Path(path_obj)
    vertices_yup = np.asarray(vertices_yup, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    colors = np.clip(np.asarray(vertex_colors, dtype=np.float64), 0.0, 1.0)
    # GLB-aligned Y-up coords: upright in Blender at object rotation 0.
    # (Z-up transform here caused +90° X on import → model appears fallen.)
    v_out = vertices_yup
    with path_obj.open("w", encoding="utf-8") as fh:
        fh.write("# Blender\n")
        fh.write("o geometry_0\n")
        for (x, y, z), (r, g, b) in zip(v_out, colors):
            fh.write(f"v {x:.8f} {y:.8f} {z:.8f} {r:.6f} {g:.6f} {b:.6f}\n")
        for f in faces:
            fh.write(f"f {' '.join(str(int(i) + 1) for i in f)}\n")
    if not preview_export_enabled():
        return
    rgba = (colors * 255.0).astype(np.uint8)
    rgba = np.column_stack([rgba, np.full(len(rgba), 255, dtype=np.uint8)])
    ply_path = path_obj.with_suffix(".ply")
    tm = trimesh.Trimesh(vertices=v_out, faces=faces, process=False)
    tm.visual.vertex_colors = rgba
    tm.export(ply_path)
    print(f"[mesh-io] colored PLY preview -> {ply_path}", flush=True)
    glb_path = path_obj.with_name(f"{path_obj.stem}_blender.glb")
    tm_yup = trimesh.Trimesh(vertices=vertices_yup, faces=faces, process=False)
    tm_yup.visual.vertex_colors = rgba
    tm_yup.export(glb_path)
    print(f"[mesh-io] Blender-view GLB (Y-up glTF) -> {glb_path}", flush=True)


def _fan_triangulate_face_indices(faces: list[list[int]]) -> np.ndarray:
    """OBJ f-lines may be quads/ngons; trimesh needs triangles."""
    tris: list[list[int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        if len(face) == 3:
            tris.append(face)
        else:
            v0 = face[0]
            for i in range(1, len(face) - 1):
                tris.append([v0, face[i], face[i + 1]])
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def export_ply_preview_from_colored_obj(path_obj: Path) -> bool:
    """Build .ply next to an xyzrgb OBJ (for Blender viewport preview)."""
    if not preview_export_enabled():
        return False
    path_obj = Path(path_obj)
    ply_path = path_obj.with_suffix(".ply")
    if ply_path.is_file():
        return True
    verts: list[list[float]] = []
    colors: list[list[float]] = []
    faces: list[list[int]] = []
    try:
        with path_obj.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v" and len(parts) >= 7:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
                elif parts[0] == "f":
                    faces.append([int(p.split("/")[0]) - 1 for p in parts[1:]])
    except Exception:
        return False
    if not verts or not colors:
        return False
    v = np.asarray(verts, dtype=np.float64)
    f = _fan_triangulate_face_indices(faces)
    if f.size == 0:
        return False
    c = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    rgba = (c * 255.0).astype(np.uint8)
    rgba = np.column_stack([rgba, np.full(len(rgba), 255, dtype=np.uint8)])
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    tm.visual.vertex_colors = rgba
    tm.export(ply_path)
    print(f"[mesh-io] colored PLY preview -> {ply_path}", flush=True)
    return True


def positions_to_yup(positions: np.ndarray, space: str) -> np.ndarray:
    """Map vertex positions into pipeline Y-up space."""
    positions = np.asarray(positions, dtype=np.float64)
    if space == COORD_BLENDER_ZUP:
        return blender_zup_to_yup(positions)
    return positions


def write_coord_space_json(directory: Path, space: str = COORD_YUP) -> None:
    """Record coord space for meshes under directory (e.g. prep/)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "coord_space.json").write_text(
        json.dumps({"space": space}, separators=(",", ":")),
        encoding="utf-8",
    )


def align_mesh_to_reference_yup(
    mesh: trimesh.Trimesh,
    ref_path: Path | str | None,
    *,
    sample: int = 4096,
) -> trimesh.Trimesh:
    """Align prep mesh Y-up to GLB reference (fixes double coord conversion)."""
    if ref_path is None:
        return mesh
    ref_path = Path(ref_path)
    if not ref_path.is_file() or ref_path.suffix.lower() != ".glb":
        return mesh
    try:
        from .glb_vertex_color import load_glb_vertex_colors
        from .coords import align_vertices_to_reference_yup

        pos, _, has = load_glb_vertex_colors(ref_path)
        if not has:
            return mesh
        out = mesh.copy()
        out.vertices = align_vertices_to_reference_yup(
            mesh.vertices, pos, sample=sample,
        )
        return out
    except Exception:
        return mesh


def load_mesh_yup(path: Path) -> trimesh.Trimesh:
    # OBJ written by this pipeline keeps split non-manifold vertices; welding on load undoes the split.
    process = Path(path).suffix.lower() != ".obj"
    try:
        mesh = trimesh.load(str(path), force="mesh", process=process)
    except Exception as e:
        # Fall back to opening via file object with relative path to bypass Windows path corruption
        rel_path = str(path).replace('\\', '/')
        file_type = path.suffix[1:].lower()
        # If relative path starts with absolute drive letter (like c:), make it relative if possible
        if len(rel_path) > 1 and rel_path[1] == ':':
            # Try to resolve relative to current dir
            try:
                import os
                cwd = os.getcwd().replace('\\', '/')
                if rel_path.lower().startswith(cwd.lower()):
                    rel_path = rel_path[len(cwd):].lstrip('/')
            except Exception:
                pass
        with open(rel_path, "rb") as f:
            mesh = trimesh.load(f, file_type=file_type, force="mesh", process=process)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    space = resolve_coord_space(path)
    if space == COORD_BLENDER_ZUP:
        mesh = mesh.copy()
        mesh.vertices = positions_to_yup(mesh.vertices, space)
    return mesh
