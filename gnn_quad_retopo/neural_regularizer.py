"""Neural mesh regularization preprocessing stage using pure-PyTorch Hash Grid Neural SDF and adaptive decimation."""
from __future__ import annotations

import math
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import trimesh

from .debug_log import DebugLog
from .auto_tune import build_sdf_plan, detect_runtime, format_sdf_plan_line, save_auto_tune_report
from .run_layout import layout_for_stage_dir
from .blender_sdf_remesh import (
    blender_openvdb_sdf_remesh,
    blender_persistent_enabled,
    format_blender_plan_line,
    plan_blender_openvdb_sdf,
    run_blender_sdf_pipeline,
    sdf_output_ceiling_faces,
)
from .sdf_cache import save_sdf_cache, sdf_cache_key, try_load_sdf_cache
import os

from .mesh_budget import (
    CANONICAL_INPUT_FACE_RATIO_MAX,
    LARGE_MESH_INPUT_FACES,
    plan_pre_sdf_input_faces,
    resolve_canonical_sdf_face_target,
    resolve_dynamic_sdf_target,
    resolve_gnn_proxy_face_target,
)


def _sdf_decimate_skip_band(sdf_target: int) -> tuple[int, int]:
    """Accept SDF output within ~45%..112% of planned target without post-decimate."""
    t = max(100_000, int(sdf_target))
    return max(300_000, int(t * 0.45)), int(t * 1.12)
from .neural_sdf import train_neural_sdf, extract_mesh_from_sdf
from .mesh_im_prep import (
    mesh_topology_stats,
    promote_canonical_copy,
    repair_mesh_for_im,
    should_skip_im_repair,
    validate_mesh_topology,
    write_canonical_manifest,
)

_ASCII_TMP = Path(tempfile.gettempdir()) / "gnn_quad_retopo"
_SKIP_DECIMATE_BELOW = 500_000
_SKIP_IF_WITHIN_RATIO = 0.92


def _plan_dict_for_blender(plan: dict) -> dict:
    return {
        "voxel_size": float(plan["voxel_size"]),
        "band_width": int(plan["band_width"]),
        "offset_distance": float(plan["offset_distance"]),
        "threshold": float(plan["threshold"]),
        "adaptivity": float(plan["adaptivity"]),
    }


def _load_sdf_result_obj(path: Path) -> trimesh.Trimesh:
    from .mesh_io import load_mesh_yup

    m = load_mesh_yup(path)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


def _record_sdf_output(
    path: Path,
    target_faces: int,
    log: DebugLog,
    *,
    label: str,
) -> int:
    """Log SDF result face count (voxel-sized; no post decimate or hole-fill)."""
    from .mesh_budget import count_mesh_faces
    from .blender_sdf_remesh import sdf_face_count_in_band

    n = count_mesh_faces(path)
    mesh = _load_sdf_result_obj(path)
    stats = mesh_topology_stats(mesh, force_check=len(mesh.faces) < 1_500_000)
    log.set(f"{label}_capped_faces", n)
    log.set(f"{label}_topology", stats)
    if sdf_face_count_in_band(n, target_faces):
        log.info(f"{label}: {n:,} tri (target {target_faces:,}, band OK)")
    else:
        log.info(f"{label}: {n:,} tri (target {target_faces:,})")
    return n


def _run_sdf_stage1_persistent(
    *,
    sdf_input_path: Path,
    raw_sdf_path: Path,
    proxy_path: Path,
    mesh_for_plan: trimesh.Trimesh,
    canon_plan: dict,
    canonical_sdf_target: int,
    input_faces: int,
    log: DebugLog,
    proxy_adaptivity: float,
    blender_voxel_size: float,
    blender_band_width: int,
    gnn_proxy_as_canonical: bool,
    proxy_target: int,
    needs_proxy: bool,
) -> tuple[bool, bool]:
    """Run canonical (+ optional proxy) SDF in one Blender session. Returns (canon_built, proxy_built)."""
    import shutil

    steps: list[dict] = []
    canon_built = False
    proxy_built = False

    if gnn_proxy_as_canonical:
        p_plan = plan_blender_openvdb_sdf(
            mesh_for_plan,
            voxel_size=blender_voxel_size if blender_voxel_size > 0 else None,
            adaptivity=proxy_adaptivity,
            band_width=blender_band_width if blender_band_width > 0 else None,
            target_sdf_faces=proxy_target,
        )
        log.step(format_blender_plan_line(p_plan) + " (proxy-as-canonical)")
        p_key = sdf_cache_key(sdf_input_path, p_plan, role="proxy_canonical", input_face_count=input_faces)
        cached = try_load_sdf_cache(p_key)
        if cached is not None:
            shutil.copy2(cached, raw_sdf_path)
            shutil.copy2(cached, proxy_path)
            return True, True
        ceiling = sdf_output_ceiling_faces(proxy_target, input_faces)
        steps.append(
            {
                "op": "sdf",
                "input": str(sdf_input_path),
                "output": str(raw_sdf_path),
                "plan": _plan_dict_for_blender(p_plan),
                "cap_faces": ceiling,
            }
        )
        run_blender_sdf_pipeline(steps)
        shutil.copy2(raw_sdf_path, proxy_path)
        _record_sdf_output(raw_sdf_path, ceiling, log, label="proxy_canonical")
        shutil.copy2(raw_sdf_path, proxy_path)
        out = _load_sdf_result_obj(raw_sdf_path)
        save_sdf_cache(p_key, raw_sdf_path, faces=len(out.faces), plan=p_plan)
        return True, True

    c_key = sdf_cache_key(
        sdf_input_path, canon_plan, role="canonical", input_face_count=input_faces,
    )
    cached_canon = try_load_sdf_cache(c_key)
    if cached_canon is not None:
        shutil.copy2(cached_canon, raw_sdf_path)
        canon_built = True
    else:
        ceiling = sdf_output_ceiling_faces(canonical_sdf_target, input_faces)
        steps.append(
            {
                "op": "sdf",
                "input": str(sdf_input_path),
                "output": str(raw_sdf_path),
                "plan": _plan_dict_for_blender(canon_plan),
                "cap_faces": ceiling,
            }
        )

    if needs_proxy:
        cached_proxy = None
        if raw_sdf_path.is_file():
            canon_loaded = _load_sdf_result_obj(raw_sdf_path)
            p_plan = plan_blender_openvdb_sdf(
                canon_loaded,
                voxel_size=blender_voxel_size if blender_voxel_size > 0 else None,
                adaptivity=proxy_adaptivity,
                band_width=blender_band_width if blender_band_width > 0 else None,
                target_sdf_faces=proxy_target,
                input_face_count=len(canon_loaded.faces),
                voxel_reference_faces=proxy_target,
            )
            p_key = sdf_cache_key(raw_sdf_path, p_plan, role="proxy")
            cached_proxy = try_load_sdf_cache(p_key, max_faces=proxy_target)
            if cached_proxy is not None:
                shutil.copy2(cached_proxy, proxy_path)
                proxy_built = True

    if steps:
        run_blender_sdf_pipeline(steps)
        if not canon_built:
            out = _load_sdf_result_obj(raw_sdf_path)
            save_sdf_cache(c_key, raw_sdf_path, faces=len(out.faces), plan=canon_plan)
            canon_built = True

    if canon_built:
        canon_ceiling = sdf_output_ceiling_faces(canonical_sdf_target, input_faces)
        _record_sdf_output(raw_sdf_path, canonical_sdf_target, log, label="canonical_sdf")

    if needs_proxy and not proxy_built:
        if not raw_sdf_path.is_file():
            raise RuntimeError(f"canonical SDF missing for proxy step: {raw_sdf_path}")
        canon_loaded = _load_sdf_result_obj(raw_sdf_path)
        p_plan = plan_blender_openvdb_sdf(
            canon_loaded,
            voxel_size=blender_voxel_size if blender_voxel_size > 0 else None,
            adaptivity=proxy_adaptivity,
            band_width=blender_band_width if blender_band_width > 0 else None,
            target_sdf_faces=proxy_target,
            input_face_count=len(canon_loaded.faces),
            voxel_reference_faces=proxy_target,
        )
        p_ceiling = sdf_output_ceiling_faces(proxy_target, len(canon_loaded.faces))
        log.step(format_blender_plan_line(p_plan) + " (proxy persistent)")
        run_blender_sdf_pipeline(
            [
                {
                    "op": "sdf",
                    "input": str(raw_sdf_path),
                    "output": str(proxy_path),
                    "plan": _plan_dict_for_blender(p_plan),
                    "cap_faces": p_ceiling,
                }
            ]
        )
        p_key = sdf_cache_key(raw_sdf_path, p_plan, role="proxy")
        capped = _record_sdf_output(
            proxy_path, proxy_target, log, label="gnn_proxy",
        )
        save_sdf_cache(p_key, proxy_path, faces=capped, plan=p_plan)
        proxy_built = True
    elif needs_proxy and proxy_built:
        capped = _record_sdf_output(proxy_path, proxy_target, log, label="gnn_proxy")
        log.set("gnn_proxy_faces", capped)

    return canon_built, proxy_built if needs_proxy else False


def _build_gnn_proxy_sdf(
    canonical_path: Path,
    proxy_path: Path,
    target_faces: int,
    log: DebugLog,
    *,
    blender_voxel_size: float = 0.0,
    blender_adaptivity: float = -1.0,
    blender_band_width: int = 0,
) -> trimesh.Trimesh:
    """Low-detail SDF proxy from canonical (for GNN/IM); shrinkwrap uses canonical later."""
    canon = trimesh.load(canonical_path, force="mesh")
    if isinstance(canon, trimesh.Scene):
        canon = canon.dump(concatenate=True)
    tgt = int(target_faces)
    for attempt, scale in enumerate((1.0, 0.85, 0.72)):
        t_try = max(150_000, int(tgt * scale))
        plan = plan_blender_openvdb_sdf(
            canon,
            voxel_size=blender_voxel_size if blender_voxel_size > 0 else None,
            adaptivity=blender_adaptivity if blender_adaptivity >= 0 else None,
            band_width=blender_band_width if blender_band_width > 0 else None,
            target_sdf_faces=t_try,
            voxel_reference_faces=t_try,
        )
        log.step(format_blender_plan_line(plan) + (f" (proxy attempt {attempt + 1})" if attempt else ""))
        mesh = blender_openvdb_sdf_remesh(
            canonical_path,
            proxy_path,
            plan=plan,
            transfer_color=False,
            cache_role="proxy",
        )
        st = mesh_topology_stats(mesh)
        log.set(f"gnn_proxy_topology_try{attempt + 1}", st)
        if st["nonmanifold_edges"] == 0 and st["boundary_edges"] == 0:
            return mesh
        log.warn(
            f"proxy SDF topology nm={st['nonmanifold_edges']} boundary={st['boundary_edges']}; "
            f"retry coarser target"
        )
    return mesh


def _assign_sdf_gnn_meshes(
    *,
    lay,
    input_path: Path,
    output_path: Path,
    log: DebugLog,
    iters: int,
    input_faces: int,
    blender_voxel_size: float,
    blender_adaptivity: float,
    blender_band_width: int,
    proxy_adaptivity: float,
    gnn_proxy_as_canonical: bool = False,
    proxy_already_built: bool = False,
) -> trimesh.Trimesh | None:
    """GNN/IM use canonical or proxy SDF only (no collapse decimate on SDF body)."""
    canon_path = lay.sdf_watertight_mesh()
    if not canon_path.is_file():
        raise RuntimeError(f"SDF canonical missing: {canon_path}")

    log.step("GNN route: SDF canonical/proxy (skip decimate + IM prep on SDF)")
    log.set("decimate_skipped", "sdf_direct_gnn_route")

    canon = trimesh.load(canon_path, force="mesh", process=False)
    if isinstance(canon, trimesh.Scene):
        canon = canon.dump(concatenate=True)
    canon_topo = mesh_topology_stats(canon, fast=True)
    canon_faces = int(canon_topo["faces"])
    log.set("canonical_sdf_file_faces", canon_faces)
    log.set("canonical_sdf_topology", canon_topo)

    proxy_path = lay.gnn_proxy_mesh()
    proxy_topo: dict | None = None
    active_path = canon_path

    rt = detect_runtime()
    if gnn_proxy_as_canonical:
        proxy_target, proxy_reason, needs_proxy = resolve_gnn_proxy_face_target(
            input_faces, rt, input_faces=input_faces,
        )
        needs_proxy = False
        proxy_reason = "gnn_proxy_as_canonical"
        if proxy_path.is_file():
            active_path = proxy_path
            proxy_topo = mesh_topology_stats(_load_sdf_result_obj(proxy_path), fast=True)
            log.info(
                f"proxy-as-canonical | GNN+shrinkwrap={canon_faces:,} faces ({proxy_reason})"
            )
        else:
            active_path = canon_path
            log.warn("proxy-as-canonical requested but proxy mesh missing; using canonical")
    else:
        proxy_target, proxy_reason, needs_proxy = resolve_gnn_proxy_face_target(
            canon_faces, rt, input_faces=input_faces,
        )

    log.set("gnn_proxy_input_faces_ref", input_faces)
    p_min, p_max, _, _ = resolve_dynamic_sdf_target(
        input_faces, role="proxy", runtime=rt, canonical_faces=canon_faces,
    )
    log.set("gnn_proxy_min_faces", p_min)
    log.set("gnn_proxy_max_faces", p_max)
    log.set("gnn_proxy_target_faces", proxy_target)
    log.set("gnn_proxy_reason", proxy_reason)

    if needs_proxy and not proxy_already_built:
        log.step(
            f"GNN proxy SDF: canonical {canon_faces:,} -> target {proxy_target:,} "
            f"({proxy_reason})"
        )
        log.info(f"proxy SDF adaptivity={proxy_adaptivity}")
        proxy_mesh = _build_gnn_proxy_sdf(
            canon_path,
            proxy_path,
            proxy_target,
            log,
            blender_voxel_size=blender_voxel_size,
            blender_adaptivity=proxy_adaptivity,
            blender_band_width=blender_band_width,
        )
        capped = _record_sdf_output(
            proxy_path, proxy_target, log, label="gnn_proxy",
        )
        proxy_mesh = _load_sdf_result_obj(proxy_path)
        proxy_topo = validate_mesh_topology(proxy_mesh)
        proxy_topo["faces"] = capped
        log.set("gnn_proxy_faces", capped)
        log.set("gnn_proxy_nonmanifold", proxy_topo["nonmanifold_edges"])
        log.set("gnn_proxy_boundary", proxy_topo["boundary_edges"])
        active_path = proxy_path
        log.info(
            f"GNN mesh=proxy | shrinkwrap reference=canonical | "
            f"proxy_faces={proxy_topo['faces']:,} canonical_faces={canon_faces:,}"
        )
    elif needs_proxy and proxy_already_built and proxy_path.is_file():
        capped = _record_sdf_output(
            proxy_path, proxy_target, log, label="gnn_proxy",
        )
        proxy_topo = validate_mesh_topology(_load_sdf_result_obj(proxy_path))
        proxy_topo["faces"] = capped
        active_path = proxy_path
        log.set("gnn_proxy_faces", capped)
        log.info(
            f"GNN mesh=proxy (persistent) | shrinkwrap reference=canonical | "
            f"proxy_faces={capped:,} (cap target {proxy_target:,})"
        )
    else:
        log.info(
            f"GNN mesh=canonical SDF ({proxy_reason}) | faces={canon_faces:,}"
        )

    active_topo = proxy_topo if needs_proxy and proxy_topo else canon_topo
    log.set("canonical_faces", canon_faces)
    log.set("canonical_nonmanifold", canon_topo["nonmanifold_edges"])
    log.set("canonical_boundary", canon_topo["boundary_edges"])
    log.set("active_gnn_faces", active_topo["faces"])
    log.set("active_gnn_nonmanifold", active_topo["nonmanifold_edges"])

    write_canonical_manifest(
        lay.root,
        lay.canonical_manifest(),
        canon_path,
        canon_topo,
        role="sdf",
        repaired=False,
        smoothed_alias=True,
        active_path=active_path,
        gnn_proxy_path=proxy_path if needs_proxy and proxy_path.is_file() else None,
        gnn_proxy_topology=proxy_topo,
        gnn_proxy_reason=proxy_reason if needs_proxy else None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label = "gnn_proxy" if needs_proxy else "canonical_sdf"
    log.step(f"smoothed_mesh = {label} (byte copy, no collapse decimate)")
    promote_canonical_copy(output_path, active_path)

    if iters <= 0:
        log.done("neural-regularizer-pass-through")
        log.save()
        print(f"[regularize] pass-through mesh saved to {output_path}", flush=True)
        return None

    active = trimesh.load(active_path, force="mesh", process=False)
    if isinstance(active, trimesh.Scene):
        active = active.dump(concatenate=True)
    return active


def _ascii_temp_path(suffix: str) -> Path:
    """Unicode path on Desktop breaks PyMeshLab; use ASCII-only temp files."""
    _ASCII_TMP.mkdir(parents=True, exist_ok=True)
    return _ASCII_TMP / f"gnn_quad_{uuid.uuid4().hex[:8]}{suffix}"


def _decimate_mesh_to_faces(mesh: trimesh.Trimesh, target_faces: int, log: DebugLog) -> trimesh.Trimesh:
    """Quadric decimation (PyMeshLab preferred). Huge meshes are reduced in stages."""
    target_faces = int(max(50_000, target_faces))
    current = mesh
    while len(current.faces) > int(target_faces * 1.05):
        n_cur = len(current.faces)
        if n_cur > 8_000_000:
            step_target = max(target_faces, int(n_cur * 0.35))
        elif n_cur > 4_000_000:
            step_target = max(target_faces, int(n_cur * 0.5))
        else:
            step_target = target_faces
        log.info(f"decimation step {n_cur:,} -> {step_target:,}")
        current = _decimate_mesh_once(current, step_target, log)
        if len(current.faces) >= n_cur:
            log.warn("decimation made no progress; stopping staged reduction")
            break
    return current


def _export_mesh_for_decimate(mesh: trimesh.Trimesh, path: Path) -> None:
    """Plain geometry OBJ for Blender decimate (no vertex colors)."""
    plain = mesh.copy()
    plain.visual = trimesh.visual.ColorVisuals(mesh=plain)
    plain.export(path)


def _decimate_mesh_once(mesh: trimesh.Trimesh, target_faces: int, log: DebugLog) -> trimesh.Trimesh:
    # Quadric decimation with PyMeshLab. Blender is not used.
    target_faces = int(max(50_000, target_faces))
    if len(mesh.faces) <= target_faces:
        return mesh
    tmp_in = _ascii_temp_path(".obj")
    tmp_out = _ascii_temp_path(".obj")
    try:
        _export_mesh_for_decimate(mesh, tmp_in)
        log.info(f"PyMeshLab decimate {len(mesh.faces):,} -> {target_faces:,}")
        try:
            import pymeshlab as ml
            from .mesh_im_prep import pymeshlab_decimate_for_im

            ms = ml.MeshSet()
            ms.load_new_mesh(str(tmp_in))
            pymeshlab_decimate_for_im(ms, target_faces)
            ms.save_current_mesh(str(tmp_out))
            out = trimesh.load(tmp_out, force="mesh", process=False)
            if isinstance(out, trimesh.Scene):
                out = out.dump(concatenate=True)
            log.info(f"PyMeshLab decimation successful (faces: {len(out.faces):,})")
            return out
        except Exception as pml_err:
            log.warn(f"PyMeshLab decimation failed: {pml_err}. Trying Trimesh fallback.")
        out = mesh.simplify_quadric_decimation(face_count=target_faces)
        log.info(f"Trimesh decimation successful (faces: {len(out.faces):,})")
        return out
    except Exception as tri_err:
        log.warn(f"decimation failed: {tri_err}. Using original mesh.")
        return mesh
    finally:
        for p in (tmp_in, tmp_out):
            if p.exists():
                p.unlink()


def _obj_has_vertex_colors(path: Path) -> bool:
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 200:
                    break
                parts = line.split()
                if parts and parts[0] == "v" and len(parts) >= 7:
                    return True
    except Exception:
        pass
    return False


# Re-export for large-mesh fast path (color skip, etc.)
_HUGE_INPUT_FACES = LARGE_MESH_INPUT_FACES


def _embed_vertex_colors_from_reference(
    mesh_path: Path,
    reference_mesh: trimesh.Trimesh,
    log: DebugLog,
    *,
    label: str,
    reference_path: Path | None = None,
    target_mesh: trimesh.Trimesh | None = None,
) -> bool:
    """Write OBJ vertex RGB from reference mesh via nearest-neighbor transfer."""
    if mesh_path.suffix.lower() != ".obj" or not mesh_path.is_file():
        return False
    if _obj_has_vertex_colors(mesh_path):
        log.info(f"{label}: vertex colors already present in OBJ")
        log.set(f"{label}_vertex_colors", True)
        return True
    ref_name = Path(reference_path).name if reference_path else "mesh"
    log.step(f"{label}: vertex color transfer (ref={ref_name})")
    try:
        mesh = target_mesh
        if mesh is None:
            log.info(f"{label}: loading target OBJ {mesh_path.name}")
            from .mesh_io import load_mesh_yup

            mesh = load_mesh_yup(mesh_path)
        else:
            log.info(f"{label}: using in-memory mesh ({len(mesh.vertices):,} verts)")
        from .blender_sdf_remesh import _write_obj_with_vertex_colors
        from .mesh_io import COORD_YUP
        from .vertex_color import colors_for_mesh_features

        colors, has_colors = colors_for_mesh_features(
            mesh,
            reference_mesh,
            ref_path=reference_path,
            target_path=mesh_path,
            target_space=COORD_YUP,
        )
        if not has_colors or len(colors) != len(mesh.vertices):
            log.warn(f"{label}: reference vertex colors unavailable")
            return False
        _write_obj_with_vertex_colors(
            mesh_path,
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            np.asarray(colors, dtype=np.float64),
        )
        log.info(f"{label}: vertex colors embedded ({len(mesh.vertices):,} verts)")
        log.set(f"{label}_vertex_colors", True)
        from .mesh_io import write_coord_space_json

        write_coord_space_json(mesh_path.parent)
        return True
    except Exception as e:
        log.warn(f"{label}: vertex color embed failed ({e})")
        return False


def _keep_ratio_for_input_faces(n_faces: int) -> float:
    """Larger inputs keep a smaller fraction; tiny meshes are barely touched."""
    if n_faces < _SKIP_DECIMATE_BELOW:
        return 1.0
    logf = math.log10(float(n_faces))
    # ~5% reduction per log10 decade above 80k faces (tuned for kitbash assets).
    ratio = 1.42 - 0.125 * logf
    return float(max(0.52, min(0.95, ratio)))


def _vram_cap_for_input_faces(
    n_faces: int,
    *,
    decimate_only: bool = False,
    target_quads: int | None = None,
) -> int:
    """Upper bound so GNN/PyTorch3D stays feasible on consumer GPUs.

    Automatically scales limits based on physical GPU VRAM and class tier.
    """
    try:
        rt = detect_runtime()
        tier = rt.gpu_tier
        vram_eff = rt.vram_effective_gb or 8.0
    except Exception:
        tier = "mid"
        vram_eff = 8.0

    # Adapt cap scaling based on active GPU Tier (base is tuned for GTX 1070 8GB)
    if tier == "cpu":
        vram_scale = 0.5
    elif tier == "low":
        vram_scale = 0.75
    elif tier == "high":
        vram_scale = 2.0       # RTX 3090 / 4090 class (16-24GB) -> 2x cap
    elif tier in ("xl", "xxl"):
        vram_scale = 4.0       # A100 / H100 class (40-80GB) -> 4x cap
    else:
        # mid tier (GTX 1070/RTX 3060, T4, L4 etc.) scales with effective VRAM
        vram_scale = max(1.0, min(1.6, vram_eff / 8.0))

    if n_faces >= 4_000_000:
        base = int(1_800_000 * vram_scale)
    elif n_faces >= 2_000_000:
        base = int(1_500_000 * vram_scale)
    elif n_faces >= 1_000_000:
        base = int(1_200_000 * vram_scale)
    else:
        base = n_faces

    if decimate_only and target_quads is not None and n_faces >= 500_000:
        lean = int(max(500_000, target_quads * 2.2 * vram_scale))
        return min(base, lean)
    return base


# 70万面台。79万まではそのままで、それを超えた入力だけ 79万面へ落とす。
FIELD_FACE_CAP = 790_000


@dataclass(frozen=True)
class DecimatePlan:
    target_faces: int
    keep_ratio: float
    note: str


def compute_mesh_complexity(mesh: trimesh.Trimesh) -> tuple[float, str]:
    """Analyze variance of face normals to estimate shape complexity/detail level."""
    if len(mesh.face_normals) == 0:
        return 0.5, "default complexity"
    std_axes = np.std(mesh.face_normals, axis=0)
    complexity = float(np.mean(std_axes))
    
    if complexity < 0.15:
        desc = "very flat / simple geometric shape"
    elif complexity < 0.35:
        desc = "moderate complexity"
    else:
        desc = "highly complex / organic detail"
        
    return complexity, desc


def plan_pre_decimation(
    n_faces: int,
    *,
    target_quads: int | None = None,
    max_faces: int | None = None,
    decimate_only: bool = False,
    complexity: float = 0.5,
) -> DecimatePlan:
    """Adaptive pre-decimation budget scaling dynamically with shape complexity."""
    if n_faces < _SKIP_DECIMATE_BELOW:
        return DecimatePlan(n_faces, 1.0, "input already small")

    if max_faces is not None:
        cap = int(max_faces)
        target = min(n_faces, cap)
        keep = target / n_faces
        note = f"manual cap {cap:,}"
    else:
        keep = _keep_ratio_for_input_faces(n_faces)
        target = int(n_faces * keep)
        
        # Scale face budget based on shape complexity to preserve fine features
        complexity_scale = max(0.6, min(1.5, complexity * 2.5))
        notes: list[str] = [f"size-based keep {keep:.0%}"]

        if target_quads is not None:
            if decimate_only:
                multiplier = 2.2 * complexity_scale
                quad_cap = int(min(n_faces, max(300_000, target_quads * multiplier)))
                if target > quad_cap:
                    target = quad_cap
                    notes.append(f"decimate-only cap {quad_cap:,} ({multiplier:.1f}× target-quads)")
                else:
                    notes.append(f"target-quads {target_quads:,} (decimate-only)")
            else:
                multiplier = 3.5 * complexity_scale
                quad_budget = int(min(n_faces, target_quads * multiplier))
                if quad_budget > target:
                    target = quad_budget
                    notes.append(f"floor {quad_budget:,} ({multiplier:.1f}× target-quads)")
                else:
                    notes.append(f"target-quads {target_quads:,}")

        vram_cap = _vram_cap_for_input_faces(
            n_faces, decimate_only=decimate_only, target_quads=target_quads
        )
        if target > vram_cap:
            target = vram_cap
            notes.append(f"VRAM cap {vram_cap:,}")
        note = "; ".join(notes)

    if target >= int(n_faces * _SKIP_IF_WITHIN_RATIO):
        return DecimatePlan(n_faces, 1.0, f"{note}; skip (<8% reduction)")
    target = max(50_000, target)
    return DecimatePlan(target, target / n_faces, note)


# height/longest < 0.02. All 15 local GLBs: brief150 <= 0.10%, raw generated >= 0.20%.
_SKINNY_ASPECT = 0.02
_SKINNY_STOP = 0.0015


def _reject_if_too_skinny(mesh: trimesh.Trimesh) -> None:
    """Stop before GNN when skinny triangles are past the measured gap."""
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        raise RuntimeError("input mesh has no faces")
    tri = np.asarray(mesh.vertices, dtype=np.float64)[faces]
    longest = np.maximum(
        np.maximum(
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        ),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    )
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    skinny = float((area2 < _SKINNY_ASPECT * np.square(longest)).mean())
    print(f"[prep] skinny triangles {skinny:.2%} (limit {_SKINNY_STOP:.2%})", flush=True)
    if skinny > _SKINNY_STOP:
        raise RuntimeError(
            f"too many skinny triangles ({skinny:.2%} > {_SKINNY_STOP:.2%}). "
            "Clean this mesh before quad remesh."
        )


def neural_regularize_mesh(
    input_path: Path | str,
    output_path: Path | str,
    *,
    iters: int = 0,
    lr: float = 0.01,
    w_chamfer: float = 1.0,
    w_laplacian: float = 0.05,
    w_edge: float = 0.1,
    decimate: bool = True,
    target_quads: int | None = None,
    max_decimate_faces: int | None = None,
    sdf_remesh: bool = False,
    sdf_backend: str = "blender",
    sdf_res: int = 0,
    sdf_method: str = "dual_contouring",
    sdf_query_faces: int = 0,
    no_sdf_simplify: bool = False,
    blender_voxel_size: float = 0.0,
    blender_adaptivity: float = -1.0,
    blender_canonical_adaptivity: float = -1.0,
    blender_band_width: int = 0,
    **kwargs,
) -> None:
    """SDF prep remesh (Blender OpenVDB default) + adaptive decimation."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    log = DebugLog("neural-regularizer", output_path.parent)
    log.step(f"loading input mesh {input_path}")
    
    # 1. Load input mesh
    mesh = trimesh.load(input_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    input_faces = len(mesh.faces)
    _reject_if_too_skinny(mesh)
    rt = detect_runtime()
    canonical_sdf_target = resolve_canonical_sdf_face_target(
        max_decimate_faces, input_faces=input_faces, runtime=rt,
    )
    c_min, c_max, _, c_reason = resolve_dynamic_sdf_target(
        input_faces, role="canonical", runtime=rt,
    )
    if blender_canonical_adaptivity >= 0:
        canon_adaptivity = float(blender_canonical_adaptivity)
    else:
        try:
            canon_adaptivity = float(os.environ.get("RETOPO_CANONICAL_ADAPTIVITY", "0.04"))
        except ValueError:
            canon_adaptivity = 0.06
    proxy_adaptivity = float(blender_adaptivity) if blender_adaptivity >= 0 else 0.08
    log.set("input_faces", input_faces)
    log.set("canonical_sdf_min", c_min)
    log.set("canonical_sdf_max", c_max)
    log.set("canonical_sdf_target", canonical_sdf_target)
    log.set("canonical_sdf_reason", c_reason)
    log.set("canonical_sdf_cap", int(input_faces * CANONICAL_INPUT_FACE_RATIO_MAX))
    log.set("canonical_adaptivity", canon_adaptivity)
    log.set("proxy_adaptivity", proxy_adaptivity)

    # Check for legacy vox-remesh flag mapping
    if kwargs.get("vox_remesh", False) or kwargs.get("vox_size", None) is not None:
        sdf_remesh = True
        sdf_res = 0

    # 2. SDF remesh (Blender OpenVDB default, or legacy Neural SDF)
    proxy_already_built = False
    gnn_proxy_as_canonical = bool(kwargs.get("gnn_proxy_as_canonical", False))
    if sdf_remesh:
        lay = layout_for_stage_dir(output_path.parent)
        lay.ensure_dirs()
        raw_sdf_path = lay.sdf_watertight_mesh()
        backend = (sdf_backend or "blender").strip().lower()
        sdf_input_path = input_path
        pre_sdf_tmp: Path | None = None
        if backend in ("blender", "openvdb", "blender_openvdb"):
            pre_budget, pre_note = plan_pre_sdf_input_faces(
                input_faces, canonical_sdf_target, runtime=rt,
            )
            if pre_budget < input_faces:
                log.step(
                    f"pre-SDF decimation ({pre_note}) to keep Blender SDF feasible"
                )
                pre_sdf_tmp = _ascii_temp_path(".obj")
                if input_faces >= 4_000_000:
                    from .blender_sdf_remesh import blender_decimate_mesh_staged

                    log.info("using Blender staged decimate for huge input (PyMeshLab-safe path)")
                    blender_decimate_mesh_staged(
                        input_path,
                        pre_sdf_tmp,
                        pre_budget,
                        initial_faces=input_faces,
                        progress=log.info,
                    )
                    mesh = trimesh.load(pre_sdf_tmp, force="mesh")
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                else:
                    mesh = _decimate_mesh_to_faces(mesh, pre_budget, log)
                    mesh.export(pre_sdf_tmp)
                sdf_input_path = pre_sdf_tmp
                log.set("pre_sdf_input_faces", len(mesh.faces))
                log.set("pre_sdf_note", pre_note)
                if input_faces >= _HUGE_INPUT_FACES:
                    log.info(
                        "pre_sdf: skip vertex colors (SDF intermediate; "
                        "colors applied once after canonical SDF)"
                    )
                else:
                    _embed_vertex_colors_from_reference(
                        pre_sdf_tmp,
                        mesh,
                        log,
                        label="pre_sdf",
                        reference_path=input_path,
                        target_mesh=mesh,
                    )
        if backend in ("blender", "openvdb", "blender_openvdb"):
            use_persistent = bool(kwargs.get("blender_persistent", blender_persistent_enabled()))
            proxy_path = lay.gnn_proxy_mesh()
            rt_proxy = detect_runtime()
            _pt, _pr, _needs = resolve_gnn_proxy_face_target(
                input_faces if gnn_proxy_as_canonical else canonical_sdf_target,
                rt_proxy,
                input_faces=input_faces,
            )

            if gnn_proxy_as_canonical:
                log.info(
                    f"proxy-as-canonical: single SDF target={_pt:,} "
                    f"(skip separate canonical {canonical_sdf_target:,})"
                )
                log.set("gnn_proxy_as_canonical", True)
            elif use_persistent:
                log.info("Blender persistent session: canonical + proxy SDF in one process")
                b_plan = plan_blender_openvdb_sdf(
                    mesh,
                    voxel_size=blender_voxel_size if blender_voxel_size > 0 else None,
                    adaptivity=canon_adaptivity,
                    band_width=blender_band_width if blender_band_width > 0 else None,
                    target_sdf_faces=canonical_sdf_target,
                )
                log.step(format_blender_plan_line(b_plan))
                log.set("blender_sdf_plan", b_plan)
                save_auto_tune_report(output_path.parent, {"blender_sdf_plan": b_plan})
                log.info("Blender SDF: skip canonical vertex colors (shrinkwrap geometry only)")
                _, proxy_already_built = _run_sdf_stage1_persistent(
                    sdf_input_path=sdf_input_path,
                    raw_sdf_path=raw_sdf_path,
                    proxy_path=proxy_path,
                    mesh_for_plan=mesh,
                    canon_plan=b_plan,
                    canonical_sdf_target=canonical_sdf_target,
                    input_faces=input_faces,
                    log=log,
                    proxy_adaptivity=proxy_adaptivity,
                    blender_voxel_size=blender_voxel_size,
                    blender_band_width=blender_band_width,
                    gnn_proxy_as_canonical=False,
                    proxy_target=_pt,
                    needs_proxy=_needs,
                )
                log.set("canonical_sdf_vertex_colors", False)
                mesh = _load_sdf_result_obj(raw_sdf_path)
            else:
                log.info(
                    f"canonical SDF target={canonical_sdf_target:,} "
                    f"(input {input_faces:,} tris, cap {CANONICAL_INPUT_FACE_RATIO_MAX:.1f}×, "
                    f"adaptivity={canon_adaptivity})"
                )
                b_plan = plan_blender_openvdb_sdf(
                    mesh,
                    voxel_size=blender_voxel_size if blender_voxel_size > 0 else None,
                    adaptivity=canon_adaptivity,
                    band_width=blender_band_width if blender_band_width > 0 else None,
                    target_sdf_faces=canonical_sdf_target,
                )
                log.step(format_blender_plan_line(b_plan))
                log.set("blender_sdf_plan", b_plan)
                save_auto_tune_report(output_path.parent, {"blender_sdf_plan": b_plan})
                log.step("applying Blender OpenVDB SDF remesh (Mesh to SDF Grid → Grid to Mesh)")
                log.info("Blender SDF: skip canonical vertex colors (shrinkwrap geometry only)")
                mesh = blender_openvdb_sdf_remesh(
                    sdf_input_path,
                    raw_sdf_path,
                    plan=b_plan,
                    transfer_color=False,
                    color_reference_path=input_path,
                    input_face_count=input_faces,
                    canonical_target_faces=canonical_sdf_target,
                    cache_role="canonical",
                )
                log.set("canonical_sdf_vertex_colors", False)

            if gnn_proxy_as_canonical:
                b_plan = plan_blender_openvdb_sdf(
                    mesh,
                    voxel_size=blender_voxel_size if blender_voxel_size > 0 else None,
                    adaptivity=proxy_adaptivity,
                    band_width=blender_band_width if blender_band_width > 0 else None,
                    target_sdf_faces=_pt,
                )
                log.step(format_blender_plan_line(b_plan) + " (proxy-as-canonical)")
                if use_persistent:
                    _run_sdf_stage1_persistent(
                        sdf_input_path=sdf_input_path,
                        raw_sdf_path=raw_sdf_path,
                        proxy_path=proxy_path,
                        mesh_for_plan=mesh,
                        canon_plan=b_plan,
                        canonical_sdf_target=_pt,
                        input_faces=input_faces,
                        log=log,
                        proxy_adaptivity=proxy_adaptivity,
                        blender_voxel_size=blender_voxel_size,
                        blender_band_width=blender_band_width,
                        gnn_proxy_as_canonical=True,
                        proxy_target=_pt,
                        needs_proxy=False,
                    )
                else:
                    mesh = blender_openvdb_sdf_remesh(
                        sdf_input_path,
                        raw_sdf_path,
                        plan=b_plan,
                        transfer_color=False,
                        input_face_count=input_faces,
                        canonical_target_faces=_pt,
                        cache_role="proxy_canonical",
                    )
                    shutil.copy2(raw_sdf_path, proxy_path)
                proxy_already_built = True
                mesh = _load_sdf_result_obj(raw_sdf_path)

            if pre_sdf_tmp is not None and pre_sdf_tmp.exists():
                pre_sdf_tmp.unlink()
            from .mesh_io import write_coord_space_json

            write_coord_space_json(raw_sdf_path.parent)
            log.step(f"saving SDF intermediate mesh to {raw_sdf_path}")
        elif backend in ("neural", "instant_ngp"):
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_complexity, complexity_desc = compute_mesh_complexity(mesh)
            sdf_plan = build_sdf_plan(
                len(mesh.faces),
                len(mesh.vertices),
                complexity=input_complexity,
                device=device,
                sdf_res=None if sdf_res <= 0 else sdf_res,
                query_face_limit=None if sdf_query_faces <= 0 else sdf_query_faces,
                no_sdf_simplify=no_sdf_simplify,
            )
            resolved_res = int(sdf_plan["sdf_res"])
            log.step(format_sdf_plan_line(sdf_plan))
            log.set("sdf_plan", sdf_plan)
            save_auto_tune_report(output_path.parent, {"sdf_plan": sdf_plan})
            log.step(
                f"applying Neural SDF remesh (res: {resolved_res}^3, method: {sdf_method})"
            )
            model, normalizer = train_neural_sdf(
                mesh, device=device, plan=sdf_plan, complexity=input_complexity
            )
            mesh = extract_mesh_from_sdf(
                model, normalizer, resolution=resolved_res, device=device, method=sdf_method
            )
            log.step(f"saving SDF intermediate mesh to {raw_sdf_path}")
            from .mesh_io import export_yup_mesh

            export_yup_mesh(raw_sdf_path, mesh.vertices, mesh.faces)
            from .mesh_io import write_coord_space_json

            write_coord_space_json(raw_sdf_path.parent)
        else:
            raise ValueError(f"unknown sdf_backend: {sdf_backend!r} (use blender or neural)")

    if sdf_remesh:
        lay = layout_for_stage_dir(output_path.parent)
        lay.ensure_dirs()
        routed = _assign_sdf_gnn_meshes(
            lay=lay,
            input_path=input_path,
            output_path=output_path,
            log=log,
            iters=iters,
            input_faces=input_faces,
            blender_voxel_size=blender_voxel_size,
            blender_adaptivity=proxy_adaptivity,
            blender_band_width=blender_band_width,
            proxy_adaptivity=proxy_adaptivity,
            gnn_proxy_as_canonical=gnn_proxy_as_canonical,
            proxy_already_built=proxy_already_built,
        )
        if routed is None:
            return
        mesh = routed

    if not sdf_remesh:
        complexity, complexity_desc = compute_mesh_complexity(mesh)
        log.set("mesh_complexity", round(complexity, 4))
        log.set("mesh_complexity_desc", complexity_desc)
        log.step(f"analyzed mesh complexity: {complexity:.4f} ({complexity_desc})")

        n_v = len(mesh.vertices)
        n_f = len(mesh.faces)
        log.set("verts_in", n_v)
        log.set("faces_in", n_f)

        run_decimate = bool(decimate)
        target_faces = n_f
        plan = None
        if run_decimate:
            plan = plan_pre_decimation(
                n_f,
                target_quads=target_quads,
                max_faces=max_decimate_faces,
                decimate_only=(iters <= 0),
                complexity=complexity,
            )
            target_faces = plan.target_faces
        if n_f > FIELD_FACE_CAP and target_faces > FIELD_FACE_CAP:
            target_faces = FIELD_FACE_CAP
            plan = DecimatePlan(
                FIELD_FACE_CAP,
                FIELD_FACE_CAP / float(n_f),
                "field face cap 790,000",
            )
            run_decimate = True
        if plan is not None:
            log.set("decimate_target_faces", target_faces)
            log.set("decimate_note", plan.note)
            log.set("decimate_keep_ratio", round(plan.keep_ratio, 4))

        shape_full = None
        if run_decimate and n_f > target_faces and n_f > FIELD_FACE_CAP:
            lay_cap = layout_for_stage_dir(output_path.parent)
            lay_cap.ensure_dirs()
            shape_full = lay_cap.prep_dir / f"{lay_cap.stem}_shape_full.obj"
            log.step(
                f"shrinkwrap keeps full mesh {n_f:,} faces -> {shape_full.name}"
            )
            trimesh.Trimesh(
                vertices=mesh.vertices, faces=mesh.faces, process=False
            ).export(shape_full)
            full_topo = mesh_topology_stats(mesh, fast=True)
        else:
            full_topo = None

        if run_decimate and n_f > target_faces:
            log.step(
                f"smart pre-decimating mesh from {n_f:,} to {target_faces:,} faces "
                f"(keep {plan.keep_ratio:.0%}; {plan.note})"
            )
            mesh = _decimate_mesh_once(mesh, target_faces, log)
            n_v = len(mesh.vertices)
            n_f = len(mesh.faces)
            log.set("verts_after_decimate", n_v)
            log.set("faces_after_decimate", n_f)
            if n_f > int(target_faces * 1.05):
                raise RuntimeError(
                    f"field face cap failed: still {n_f:,} faces, target {target_faces:,}"
                )

        pre = mesh_topology_stats(mesh)
        log.set("im_prep_before", pre)
        skip, skip_reason = should_skip_im_repair(pre)
        if skip:
            log.info(f"IM prep: skip repair ({skip_reason})")
            im_stats = pre
        else:
            log.step(
                f"IM prep repair (non-manifold={pre['nonmanifold_edges']}, "
                f"watertight={pre['watertight']})"
            )
            mesh, im_stats = repair_mesh_for_im(mesh, light=True)
            log.set("im_prep_after", im_stats)
            if im_stats.get("nonmanifold_edges", 0) > 0:
                log.warn(
                    f"IM prep: {im_stats['nonmanifold_edges']} non-manifold edges remain after repair"
                )
            else:
                log.info("IM prep: non-manifold edges cleared")
        log.set("verts_after_im_prep", len(mesh.vertices))
        log.set("faces_after_im_prep", len(mesh.faces))

        lay = layout_for_stage_dir(output_path.parent)
        lay.ensure_dirs()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False).export(
            output_path
        )
        field_topo = mesh_topology_stats(mesh, fast=True)
        canon_path = output_path
        proxy_path = None
        proxy_reason = None
        if shape_full is not None and shape_full.is_file():
            canon_path = shape_full
            proxy_path = lay.gnn_proxy_mesh()
            if proxy_path.resolve() != output_path.resolve():
                shutil.copy2(output_path, proxy_path)
            proxy_reason = "field face cap 790000"
        write_canonical_manifest(
            lay.root,
            lay.canonical_manifest(),
            canon_path,
            full_topo if full_topo is not None else field_topo,
            role="smoothed",
            repaired=not skip,
            smoothed_alias=False,
            active_path=output_path,
            gnn_proxy_path=proxy_path,
            gnn_proxy_topology=field_topo if proxy_path is not None else None,
            gnn_proxy_reason=proxy_reason,
        )

        if iters <= 0:
            log.step("skipping PyTorch3D smoothing (iters=0)")
            log.step("export smoothed mesh")
            log.done("neural-regularizer-pass-through")
            log.save()
            print(f"[regularize] pass-through mesh saved to {output_path}", flush=True)
            return

    # 6. Optional PyTorch3D Vertex Optimization Loop (only if iters > 0)
    log.step("importing PyTorch & PyTorch3D (dynamic check)")
    use_pytorch3d = True
    try:
        import torch
        from pytorch3d.structures import Meshes
        from pytorch3d.loss import mesh_laplacian_smoothing, mesh_edge_loss
        from pytorch3d.loss.chamfer import chamfer_distance
    except ImportError as e:
        use_pytorch3d = False
        log.warn(f"PyTorch or PyTorch3D not found: {e}. Activating high-quality CPU fallback pipeline.")

    if not use_pytorch3d:
        log.step("running PyMeshLab Laplacian Smoothing fallback")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pymeshlab as ml
            ms = ml.MeshSet()
            tmp_in = _ascii_temp_path(".obj")
            mesh.export(tmp_in)
            ms.load_new_mesh(str(tmp_in))
            
            # Apply PyMeshLab Laplacian Smoothing
            if hasattr(ms, "apply_coord_laplacian_smoothing"):
                try:
                    ms.apply_coord_laplacian_smoothing(stepsnum=5, keepboundary=True)
                except Exception:
                    ms.apply_coord_laplacian_smoothing(steps=5, keepboundary=True)
            elif hasattr(ms, "meshing_smoothing_laplacian"):
                try:
                    ms.meshing_smoothing_laplacian(steps=5, keepboundary=True)
                except Exception:
                    ms.meshing_smoothing_laplacian(stepnum=5, keepboundary=True)
            else:
                try:
                    ms.apply_filter("apply_coord_laplacian_smoothing", stepsnum=5, keepboundary=True)
                except Exception:
                    ms.apply_filter("apply_coord_laplacian_smoothing", steps=5, keepboundary=True)
            
            ms.save_current_mesh(str(output_path))
            if tmp_in.exists():
                tmp_in.unlink()
                
            log.done("neural-regularizer-fallback-pymeshlab")
            log.save()
            print(f"[regularize] (Fallback) PyMeshLab Laplacian Smoothing successful! Saved to {output_path}", flush=True)
            return
        except Exception as pml_err:
            log.warn(f"PyMeshLab fallback failed: {pml_err}. Trying Trimesh Laplacian filter.")
            try:
                out_mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=10)
                out_mesh.export(output_path)
                log.done("neural-regularizer-fallback-trimesh")
                log.save()
                print(f"[regularize] (Fallback) Trimesh Laplacian Smoothing successful! Saved to {output_path}", flush=True)
                return
            except Exception as tri_err:
                err_msg = f"All fallbacks failed. PyMeshLab error: {pml_err}, Trimesh error: {tri_err}"
                log.warn(err_msg)
                raise RuntimeError(err_msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.set("device", str(device))

    verts_init = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
    faces = torch.from_numpy(mesh.faces.astype(np.int64)).to(device)
    
    verts_opt = torch.nn.Parameter(verts_init.clone())
    optimizer = torch.optim.Adam([verts_opt], lr=lr)
    verts_target = verts_init.clone().unsqueeze(0)
    
    log.step(f"starting neural regularization loop ({iters} iterations)")
    log_interval = max(1, iters // 5)
    
    for step in range(iters):
        optimizer.zero_grad()
        current_mesh = Meshes(verts=[verts_opt], faces=[faces])
        loss = torch.tensor(0.0, device=device)
        terms = {}
        
        if w_chamfer > 0:
            verts_curr = verts_opt.unsqueeze(0)
            ch_loss, _ = chamfer_distance(verts_curr, verts_target)
            loss = loss + w_chamfer * ch_loss
            terms["chamfer"] = float(ch_loss.item())
            
        if w_laplacian > 0:
            lap_loss = mesh_laplacian_smoothing(current_mesh, method="uniform")
            loss = loss + w_laplacian * lap_loss
            terms["laplacian"] = float(lap_loss.item())
            
        if w_edge > 0:
            edge_loss = mesh_edge_loss(current_mesh)
            loss = loss + w_edge * edge_loss
            terms["edge"] = float(edge_loss.item())
            
        loss.backward()
        optimizer.step()
        
        if (step + 1) % log_interval == 0 or step == 0 or (step + 1) == iters:
            terms_s = " ".join([f"{k}={v:.5f}" for k, v in terms.items()])
            log.info(f"step {step + 1:03d}/{iters} | total_loss={loss.item():.5f} | {terms_s}")
            
    log.step(f"exporting optimized mesh to {output_path}")
    verts_final = verts_opt.detach().cpu().numpy()
    faces_final = mesh.faces
    
    out_mesh = trimesh.Trimesh(vertices=verts_final, faces=faces_final, process=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(output_path)
    
    log.done("neural-regularizer")
    log.save()
    print(f"[regularize] successfully smoothed mesh saved to {output_path}", flush=True)
