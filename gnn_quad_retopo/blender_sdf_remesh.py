"""Blender 5.x OpenVDB SDF remesh (Mesh to SDF Grid → Grid to Mesh) for prep Stage 1."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path

import numpy as np
import trimesh

from .sdf_cache import save_sdf_cache, sdf_cache_key, try_load_sdf_cache
from .coords import blender_zup_to_yup
from .mesh_io import (
    COORD_YUP,
    export_blender_colored_mesh,
    export_ply_preview_from_colored_obj,
    export_yup_colored_mesh,
    export_yup_mesh,
    obj_has_vertex_colors,
    write_coord_space_json,
)


def _write_obj_with_vertex_colors(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
) -> None:
    export_blender_colored_mesh(path, vertices, faces, vertex_colors)


_BLENDER_SCRIPT = textwrap.dedent(
    r'''
    import bpy
    import sys

    def b_log(msg):
        print(f"[Blender-Log] {msg}", flush=True)

    def import_mesh(path):
        p = path.lower()
        if p.endswith((".glb", ".gltf")):
            bpy.ops.import_scene.gltf(filepath=path)
        else:
            bpy.ops.wm.obj_import(filepath=path)

    def export_mesh(path):
        p = path.lower()
        if p.endswith((".glb", ".gltf")):
            bpy.ops.export_scene.gltf(
                filepath=path, use_selection=True, export_materials="EXPORT"
            )
        else:
            bpy.ops.wm.obj_export(
                filepath=path,
                export_colors=True,
                export_normals=True,
            )

    try:
        args = sys.argv[sys.argv.index("--") + 1:]
        in_path, out_path = args[0], args[1]
        v_size, b_width = float(args[2]), int(args[3])
        dist, thresh = float(args[4]), float(args[5])
        adapt_val = float(args[6])
        transfer_color = args[7].lower() in ("1", "true", "yes")

        b_log("=== Blender OpenVDB SDF remesh start ===")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        b_log("import mesh")
        import_mesh(in_path)
        meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        if not meshes:
            raise RuntimeError("no mesh object after import")
        target = max(meshes, key=lambda o: len(o.data.polygons))
        bpy.context.view_layer.objects.active = target
        target.select_set(True)

        source = None
        if transfer_color:
            bpy.ops.object.duplicate()
            source = bpy.context.active_object
            source.name = "Color_Source"
            bpy.context.view_layer.objects.active = target
            target.select_set(True)
            source.select_set(False)

        b_log(f"build Geometry Nodes (voxel={v_size}, adaptivity={adapt_val})")
        tree = bpy.data.node_groups.new(name="Auto_SDF_Remesh", type="GeometryNodeTree")
        tree.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        tree.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )
        node_in = tree.nodes.new("NodeGroupInput")
        node_out = tree.nodes.new("NodeGroupOutput")
        node_mesh_to_sdf = tree.nodes.new("GeometryNodeMeshToSDFGrid")
        node_sdf_offset = tree.nodes.new("GeometryNodeSDFGridOffset")
        node_grid_to_mesh = tree.nodes.new("GeometryNodeGridToMesh")

        node_mesh_to_sdf.inputs["Voxel Size"].default_value = v_size
        node_mesh_to_sdf.inputs["Band Width"].default_value = b_width
        node_sdf_offset.inputs["Distance"].default_value = dist
        node_grid_to_mesh.inputs["Threshold"].default_value = thresh
        node_grid_to_mesh.inputs["Adaptivity"].default_value = adapt_val

        tree.links.new(node_in.outputs[0], node_mesh_to_sdf.inputs["Mesh"])
        tree.links.new(
            node_mesh_to_sdf.outputs["SDF Grid"], node_sdf_offset.inputs["Grid"]
        )
        tree.links.new(
            node_sdf_offset.outputs["Grid"], node_grid_to_mesh.inputs["Grid"]
        )
        tree.links.new(node_grid_to_mesh.outputs["Mesh"], node_out.inputs[0])

        b_log("apply OpenVDB modifier")
        mod = target.modifiers.new("SDF_Remesh", "NODES")
        mod.node_group = tree
        bpy.ops.object.modifier_apply(modifier=mod.name)

        if transfer_color and source is not None:
            b_log("vertex color transfer")
            if not target.data.color_attributes:
                target.data.color_attributes.new(
                    name="Color", type="BYTE_COLOR", domain="CORNER"
                )
            dt_mod = target.modifiers.new("ColorTransfer", "DATA_TRANSFER")
            dt_mod.object = source
            dt_mod.use_loop_data = True
            dt_mod.data_types_loops = {"COLOR_CORNER"}
            dt_mod.loop_mapping = "POLYINTERP_NEAREST"
            bpy.ops.object.modifier_apply(modifier=dt_mod.name)
            if target.data.color_attributes:
                try:
                    target.data.color_attributes.active_color = target.data.color_attributes[0]
                    bpy.ops.geometry.color_attribute_convert(domain="POINT")
                    b_log("corner colors converted to POINT for OBJ export")
                except Exception as ex:
                    b_log(f"color POINT convert skipped: {ex}")
            bpy.data.objects.remove(source, do_unlink=True)

        bpy.ops.object.shade_smooth()
        b_log(f"export {out_path}")
        export_mesh(out_path)
        b_log("=== Blender OpenVDB SDF remesh done ===")
    except Exception as e:
        print(f"BLENDER_ERROR: {e}", flush=True)
        sys.exit(1)
    '''
)


def blender_persistent_enabled() -> bool:
    v = os.environ.get("RETOPO_BLENDER_PERSISTENT", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


_BLENDER_PIPELINE_SCRIPT = textwrap.dedent(
    r'''
    import bpy
    import json
    import sys

    def b_log(msg):
        print(f"[Blender-Log] {msg}", flush=True)

    def import_mesh(path):
        p = path.lower()
        if p.endswith((".glb", ".gltf")):
            bpy.ops.import_scene.gltf(filepath=path)
        else:
            bpy.ops.wm.obj_import(filepath=path)

    def export_mesh(path):
        p = path.lower()
        if p.endswith((".glb", ".gltf")):
            bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_materials="EXPORT")
        else:
            bpy.ops.wm.obj_export(filepath=path, export_normals=True)

    def clear_scene():
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

    def pick_mesh():
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not meshes:
            raise RuntimeError("no mesh object")
        obj = max(meshes, key=lambda o: len(o.data.polygons))
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        return obj

    def apply_sdf(obj, plan):
        v_size = float(plan["voxel_size"])
        b_width = int(plan["band_width"])
        dist = float(plan["offset_distance"])
        thresh = float(plan["threshold"])
        adapt_val = float(plan["adaptivity"])
        tree = bpy.data.node_groups.new(name="Auto_SDF_Remesh", type="GeometryNodeTree")
        tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        node_in = tree.nodes.new("NodeGroupInput")
        node_out = tree.nodes.new("NodeGroupOutput")
        node_mesh_to_sdf = tree.nodes.new("GeometryNodeMeshToSDFGrid")
        node_sdf_offset = tree.nodes.new("GeometryNodeSDFGridOffset")
        node_grid_to_mesh = tree.nodes.new("GeometryNodeGridToMesh")
        node_mesh_to_sdf.inputs["Voxel Size"].default_value = v_size
        node_mesh_to_sdf.inputs["Band Width"].default_value = b_width
        node_sdf_offset.inputs["Distance"].default_value = dist
        node_grid_to_mesh.inputs["Threshold"].default_value = thresh
        node_grid_to_mesh.inputs["Adaptivity"].default_value = adapt_val
        tree.links.new(node_in.outputs[0], node_mesh_to_sdf.inputs["Mesh"])
        tree.links.new(node_mesh_to_sdf.outputs["SDF Grid"], node_sdf_offset.inputs["Grid"])
        tree.links.new(node_sdf_offset.outputs["Grid"], node_grid_to_mesh.inputs["Grid"])
        tree.links.new(node_grid_to_mesh.outputs["Mesh"], node_out.inputs[0])
        mod = obj.modifiers.new("SDF_Remesh", "NODES")
        mod.node_group = tree
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.ops.object.shade_smooth()

    def decimate_to(obj, target_faces):
        def count_tris(obj):
            return sum(len(p.vertices) - 2 for p in obj.data.polygons)

        target_faces = int(max(50000, target_faces))
        n = count_tris(obj)
        while n > int(target_faces * 1.02):
            stage = target_faces
            if n > target_faces * 2.5:
                stage = max(target_faces, int(n * 0.35))
            elif n > target_faces * 1.8:
                stage = max(target_faces, int(n * 0.5))
            ratio = min(1.0, max(0.01, float(stage) / max(1, n)))
            b_log(f"decimate ratio={ratio:.4f} tris_in={n} target_stage={stage}")
            mod = obj.modifiers.new("Cap_Decimate", "DECIMATE")
            mod.decimate_type = "COLLAPSE"
            mod.ratio = ratio
            bpy.ops.object.modifier_apply(modifier=mod.name)
            n = count_tris(obj)
        b_log(f"decimate done tris_out={n} polys_out={len(obj.data.polygons)}")

    try:
        cfg_path = sys.argv[sys.argv.index("--") + 1]
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        b_log("=== Blender persistent SDF pipeline start ===")
        for i, step in enumerate(cfg.get("steps", [])):
            b_log(f"step {i + 1}/{len(cfg['steps'])} op={step.get('op')}")
            clear_scene()
            in_path = step["input"]
            out_path = step["output"]
            b_log(f"import {in_path}")
            import_mesh(in_path)
            obj = pick_mesh()
            if step.get("op") == "sdf":
                plan = step["plan"]
                b_log(f"SDF voxel={plan['voxel_size']} adaptivity={plan['adaptivity']}")
                apply_sdf(obj, plan)
                cap = step.get("cap_faces")
                if cap:
                    tri_n = sum(len(p.vertices) - 2 for p in obj.data.polygons)
                    b_log(f"cap skipped (voxel-only) tris={tri_n} cap={cap}")
            elif step.get("op") == "decimate":
                decimate_to(obj, int(step["target_faces"]))
            b_log(f"export {out_path}")
            export_mesh(out_path)
        b_log("=== Blender persistent SDF pipeline done ===")
    except Exception as e:
        print(f"BLENDER_ERROR: {e}", flush=True)
        sys.exit(1)
    '''
)


def run_blender_sdf_pipeline(
    steps: list[dict],
    *,
    blender_exe: Path | str | None = None,
    timeout_sec: int | None = None,
) -> None:
    """Run multiple SDF/decimate steps in one Blender process."""
    if not steps:
        return
    exe = Path(blender_exe) if blender_exe else find_blender_executable()
    with tempfile.TemporaryDirectory(prefix="gnn_blender_pipe_") as tmp:
        cfg_path = Path(tmp) / "pipeline.json"
        script_path = Path(tmp) / "blender_pipeline.py"
        cfg_path.write_text(json.dumps({"steps": steps}, indent=0), encoding="utf-8")
        script_path.write_text(_BLENDER_PIPELINE_SCRIPT, encoding="utf-8")
        cmd = [str(exe), "-b", "-P", str(script_path), "--", str(cfg_path)]
        print(f"[BlenderPipe] {len(steps)} step(s) in one session", flush=True)
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec,
            encoding="utf-8", errors="replace", env=env,
        )
        if proc.stdout:
            for line in proc.stdout.splitlines():
                print(f"  {line}", flush=True)
        if proc.stderr:
            for line in proc.stderr.splitlines():
                if line not in (proc.stdout or ""):
                    print(f"  {line}", flush=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Blender pipeline failed (code {proc.returncode}): {err}")


def find_blender_executable() -> Path:
    """Resolve Blender binary (GNN_BLENDER, PATH, OS defaults, Colab bundle)."""
    env = os.environ.get("GNN_BLENDER", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"GNN_BLENDER not found: {p}")

    found = shutil.which("blender")
    if found:
        return Path(found).resolve()

    if sys.platform == "win32":
        roots = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        ]
        for root in roots:
            foundation = root / "Blender Foundation"
            if not foundation.is_dir():
                continue
            for exe in sorted(foundation.glob("Blender */blender.exe"), reverse=True):
                return exe.resolve()

    for candidate in (
        Path("blender-5.1.1-linux-x64/blender"),
        Path("/usr/bin/blender"),
    ):
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Blender not found. Set GNN_BLENDER to blender.exe (Blender 5.1+)."
    )


_BLENDER_DECIMATE_SCRIPT = textwrap.dedent(
    r'''
    import bpy
    import sys

    def b_log(msg):
        print(f"[Blender-Log] {msg}", flush=True)

    def import_mesh(path):
        p = path.lower()
        if p.endswith((".glb", ".gltf")):
            bpy.ops.import_scene.gltf(filepath=path)
        else:
            bpy.ops.wm.obj_import(filepath=path)

    def export_mesh(path):
        p = path.lower()
        if p.endswith((".glb", ".gltf")):
            bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_materials="EXPORT")
        else:
            bpy.ops.wm.obj_export(filepath=path)

    try:
        args = sys.argv[sys.argv.index("--") + 1:]
        in_path, out_path, target_faces = args[0], args[1], int(args[2])
        b_log(f"decimate start target_faces={target_faces}")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        import_mesh(in_path)
        meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        if not meshes:
            raise RuntimeError("no mesh after import")
        obj = max(meshes, key=lambda o: len(o.data.polygons))
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        def count_tris(obj):
            return sum(len(p.vertices) - 2 for p in obj.data.polygons)

        n_in = count_tris(obj)
        ratio = min(1.0, max(0.01, float(target_faces) / max(1, n_in)))
        b_log(f"decimate ratio={ratio:.4f} tris_in={n_in} polys_in={len(obj.data.polygons)}")
        mod = obj.modifiers.new("PreSDF_Decimate", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        bpy.ops.object.modifier_apply(modifier=mod.name)
        b_log(f"decimate done tris_out={count_tris(obj)} polys_out={len(obj.data.polygons)}")
        export_mesh(out_path)
        b_log("decimate export done")
    except Exception as e:
        print(f"BLENDER_ERROR: {e}", flush=True)
        sys.exit(1)
    '''
)


def blender_decimate_mesh(
    input_path: Path | str,
    output_path: Path | str,
    target_faces: int,
    *,
    blender_exe: Path | str | None = None,
    timeout_sec: int | None = None,
) -> Path:
    """Headless Blender collapse decimate (for huge inputs PyMeshLab cannot handle)."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exe = Path(blender_exe) if blender_exe else find_blender_executable()
    with tempfile.TemporaryDirectory(prefix="gnn_blender_dec_") as tmp:
        script_path = Path(tmp) / "blender_decimate.py"
        script_path.write_text(_BLENDER_DECIMATE_SCRIPT, encoding="utf-8")
        cmd = [
            str(exe), "-b", "-P", str(script_path), "--",
            str(input_path), str(output_path), str(int(target_faces)),
        ]
        print(f"[BlenderDec] target_faces={int(target_faces):,} in={input_path.name}", flush=True)
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec,
            encoding="utf-8", errors="replace", env=env,
        )
        if proc.stdout:
            for line in proc.stdout.splitlines():
                print(f"  {line}", flush=True)
        if proc.stderr:
            for line in proc.stderr.splitlines():
                if line not in (proc.stdout or ""):
                    print(f"  {line}", flush=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Blender decimate failed (code {proc.returncode}): {err}")
    if not output_path.is_file():
        raise RuntimeError(f"Blender decimate did not write: {output_path}")
    return output_path


def _staged_decimate_step_target(n_cur: int, target_faces: int) -> int:
    """Match neural_regularizer._decimate_mesh_to_faces staging ratios."""
    if n_cur > 8_000_000:
        return max(target_faces, int(n_cur * 0.35))
    if n_cur > 4_000_000:
        return max(target_faces, int(n_cur * 0.5))
    return target_faces


def _blender_stage_temp(suffix: str = ".obj") -> Path:
    root = Path(tempfile.gettempdir()) / "gnn_quad_retopo"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"gnn_blender_stg_{uuid.uuid4().hex[:8]}{suffix}"


def blender_decimate_mesh_staged(
    input_path: Path | str,
    output_path: Path | str,
    target_faces: int,
    *,
    initial_faces: int | None = None,
    progress=None,
    blender_exe: Path | str | None = None,
    timeout_sec: int | None = None,
) -> Path:
    """Staged Blender collapse decimate (same step policy as PyMeshLab path)."""
    target_faces = int(max(50_000, target_faces))
    n_cur = int(initial_faces) if initial_faces else None
    if n_cur is None:
        from .mesh_budget import count_mesh_faces

        n_cur = count_mesh_faces(input_path)

    current_in = Path(input_path).resolve()
    final_out = Path(output_path).resolve()
    final_out.parent.mkdir(parents=True, exist_ok=True)
    owned_temps: list[Path] = []

    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)
        else:
            print(f"[BlenderDec] {msg}", flush=True)

    try:
        while n_cur > int(target_faces * 1.05):
            step_target = _staged_decimate_step_target(n_cur, target_faces)
            _log(f"decimation step {n_cur:,} -> {step_target:,}")
            step_out = final_out if step_target <= int(target_faces * 1.05) else _blender_stage_temp()
            if step_out != final_out:
                owned_temps.append(step_out)
            blender_decimate_mesh(
                current_in,
                step_out,
                step_target,
                blender_exe=blender_exe,
                timeout_sec=timeout_sec,
            )
            if current_in in owned_temps and current_in.exists():
                current_in.unlink()
            current_in = step_out
            n_cur = step_target

        if current_in.resolve() != final_out.resolve():
            _log(f"decimation step {n_cur:,} -> {target_faces:,}")
            blender_decimate_mesh(
                current_in,
                final_out,
                target_faces,
                blender_exe=blender_exe,
                timeout_sec=timeout_sec,
            )
        return final_out
    finally:
        for tmp in owned_temps:
            if tmp.resolve() != final_out.resolve() and tmp.exists():
                tmp.unlink(missing_ok=True)


from .mesh_budget import (
    CANONICAL_INPUT_FACE_RATIO_MAX,
    auto_sdf_face_target,
    resolve_sdf_face_target,
)

# Fallback only when input face count is unknown.
DEFAULT_SDF_FACE_TARGET = 2_000_000

# Accept SDF output within this band without collapse decimate (voxel-only sizing).
SDF_FACE_BAND_LOW = 0.82
SDF_FACE_BAND_HIGH = 1.22
# Empirical OpenVDB overshoot vs voxels_across (lower -> finer grid).
SDF_VOXEL_EMPIRICAL_DIV = 1.55


def _sdf_voxel_detail_boost() -> float:
    v = os.environ.get("RETOPO_SDF_DETAIL_BOOST", "1.10").strip()
    try:
        return float(max(0.90, min(1.20, float(v))))
    except ValueError:
        return 1.10


def sdf_face_count_in_band(n_faces: int, target_faces: int) -> bool:
    tgt = max(50_000, int(target_faces))
    n = int(n_faces)
    return int(tgt * SDF_FACE_BAND_LOW) <= n <= int(tgt * SDF_FACE_BAND_HIGH)


def _voxels_across_for_target(
    ext: float,
    target_faces: int,
    adaptivity: float,
    input_faces: int,
) -> int:
    """Calibrate OpenVDB grid density so SDF output lands near target in one pass."""
    import math

    tgt = max(100_000, int(target_faces))
    n_in = max(tgt, int(input_faces))
    ref_faces = 2_000_000
    ref_voxels = 520
    adapt = max(0.01, float(adaptivity))
    # faces ~ voxels^2; lower target -> coarser grid (fewer voxels_across)
    voxels = ref_voxels * math.sqrt(tgt / ref_faces)
    # Lower adaptivity -> OpenVDB subdivides more -> coarser voxels compensate
    voxels *= math.sqrt(adapt / 0.12)
    if n_in > int(tgt * 1.1):
        voxels /= math.sqrt(n_in / tgt)
    voxels /= math.sqrt(SDF_VOXEL_EMPIRICAL_DIV)
    voxels *= _sdf_voxel_detail_boost()
    return int(max(256, min(960, round(voxels))))


def sdf_output_floor_faces(target_faces: int, input_faces: int) -> int:
    """Minimum acceptable SDF triangle count; below this we retry with finer voxels."""
    tgt = max(100_000, int(target_faces))
    inp = max(100_000, int(input_faces))
    return int(max(400_000, min(inp, tgt) * 0.75, tgt * 0.55))


def sdf_output_ceiling_faces(target_faces: int, input_faces: int) -> int:
    """Maximum canonical SDF triangles (input cap and ~12% above plan target)."""
    tgt = max(100_000, int(target_faces))
    inp = max(100_000, int(input_faces))
    cap = int(inp * CANONICAL_INPUT_FACE_RATIO_MAX)
    return int(min(cap, max(tgt, int(tgt * 1.12))))


def refine_blender_sdf_plan(plan: dict, attempt: int) -> dict:
    """Finer voxel + lower adaptivity for SDF retry when output was too coarse."""
    out = dict(plan)
    factor = 0.82 ** max(0, int(attempt))
    out["voxel_size"] = float(out["voxel_size"]) * factor
    out["adaptivity"] = max(0.01, float(out["adaptivity"]) * (0.65 ** max(0, int(attempt))))
    ext = float(out.get("mesh_extent_m", 1.0))
    if ext > 0:
        out["voxels_across"] = int(round(ext / out["voxel_size"]))
    return out


def plan_blender_openvdb_sdf(
    mesh: trimesh.Trimesh,
    *,
    voxel_size: float | None = None,
    band_width: int | None = None,
    offset_distance: float | None = None,
    threshold: float | None = None,
    adaptivity: float | None = None,
    target_sdf_faces: int | None = None,
    input_face_count: int | None = None,
    voxel_reference_faces: int | None = None,
) -> dict:
    """Auto voxel/adaptivity from mesh bounds (prep: IM budget, keep adaptivity detail)."""
    import math

    ext = float(np.max(mesh.extents))
    if ext <= 0:
        ext = 1.0
    n_faces = int(input_face_count if input_face_count is not None else len(mesh.faces))
    n_voxel = int(voxel_reference_faces if voxel_reference_faces is not None else n_faces)
    tgt = resolve_sdf_face_target(target_sdf_faces, input_faces=n_faces)

    adapt_val = 0.12 if adaptivity is None or adaptivity < 0 else float(adaptivity)

    if voxel_size is None or voxel_size <= 0:
        voxels = _voxels_across_for_target(ext, tgt, adapt_val, n_voxel)
        voxel_size = ext / voxels

    if band_width is None:
        band_width = 56 if n_faces >= 1_000_000 else 44 if n_faces >= 300_000 else 32

    if offset_distance is None:
        offset_distance = 0.0

    if threshold is None or threshold <= 0:
        threshold = voxel_size * 0.5

    if adaptivity is None or adaptivity < 0:
        adaptivity = 0.12 if n_faces >= 500_000 else 0.2

    return {
        "backend": "blender_openvdb",
        "voxel_size": float(voxel_size),
        "band_width": int(band_width),
        "offset_distance": float(offset_distance),
        "threshold": float(threshold),
        "adaptivity": float(adaptivity),
        "mesh_extent_m": ext,
        "n_faces_in": n_faces,
        "target_sdf_faces": tgt,
        "sdf_detail_ratio": round(tgt / max(n_faces, 1), 4),
        "voxels_across": int(round(ext / voxel_size)) if voxel_size > 0 else 0,
    }


def format_blender_plan_line(plan: dict) -> str:
    return (
        f"blender-sdf: voxel={plan['voxel_size']:.6g}m band={plan['band_width']} "
        f"offset={plan['offset_distance']:.6g} threshold={plan['threshold']:.6g} "
        f"adaptivity={plan['adaptivity']:.2f} extent={plan['mesh_extent_m']:.4g}m "
        f"faces_in={plan['n_faces_in']:,} sdf_target={plan.get('target_sdf_faces', 0):,} "
        f"(ratio={plan.get('sdf_detail_ratio', 0):.3f})"
    )


def _run_blender_sdf_subprocess(
    input_path: Path,
    output_path: Path,
    plan: dict,
    *,
    blender_exe: Path,
    transfer_color: bool,
    timeout_sec: int | None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="gnn_blender_sdf_") as tmp:
        script_path = Path(tmp) / "blender_sdf_remesh.py"
        script_path.write_text(_BLENDER_SCRIPT, encoding="utf-8")
        tc = "1" if transfer_color and input_path.suffix.lower() in (".glb", ".gltf") else "0"
        cmd = [
            str(blender_exe),
            "-b",
            "-P",
            str(script_path),
            "--",
            str(input_path),
            str(output_path),
            str(plan["voxel_size"]),
            str(plan["band_width"]),
            str(plan["offset_distance"]),
            str(plan["threshold"]),
            str(plan["adaptivity"]),
            tc,
        ]
        print(f"[BlenderSDF] {format_blender_plan_line(plan)}", flush=True)
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        if proc.stdout:
            for line in proc.stdout.splitlines():
                print(f"  {line}", flush=True)
        if proc.stderr:
            for line in proc.stderr.splitlines():
                if line not in (proc.stdout or ""):
                    print(f"  {line}", flush=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Blender SDF remesh failed (code {proc.returncode}): {err}")
    if not output_path.is_file():
        raise RuntimeError(f"Blender did not write output: {output_path}")


def blender_openvdb_sdf_remesh(
    input_path: Path | str,
    output_path: Path | str,
    *,
    plan: dict | None = None,
    blender_exe: Path | str | None = None,
    transfer_color: bool = True,
    color_reference_path: Path | str | None = None,
    timeout_sec: int | None = None,
    input_face_count: int | None = None,
    canonical_target_faces: int | None = None,
    max_retries: int = 3,
    cache_role: str | None = None,
) -> trimesh.Trimesh:
    """Run headless Blender OpenVDB SDF remesh; return loaded result mesh."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(input_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if plan is None:
        plan = plan_blender_openvdb_sdf(mesh)

    cache_key: str | None = None
    if cache_role:
        cache_key = sdf_cache_key(
            input_path, plan, role=cache_role, input_face_count=input_face_count,
        )
        cached = try_load_sdf_cache(cache_key)
        if cached is not None:
            import shutil
            shutil.copy2(cached, output_path)
            out = trimesh.load(output_path, force="mesh")
            if isinstance(out, trimesh.Scene):
                out = out.dump(concatenate=True)
            out = out.copy()
            out.vertices = blender_zup_to_yup(np.asarray(out.vertices, dtype=np.float64))
            return out

    n_in = int(input_face_count if input_face_count is not None else len(mesh.faces))
    tgt = int(
        canonical_target_faces
        if canonical_target_faces is not None
        else plan.get("target_sdf_faces", n_in)
    )
    floor_faces = sdf_output_floor_faces(tgt, n_in)
    ceiling_faces = sdf_output_ceiling_faces(tgt, n_in)

    exe = Path(blender_exe) if blender_exe else find_blender_executable()
    print(f"[BlenderSDF] blender={exe}", flush=True)
    print(
        f"[BlenderSDF] target={tgt:,} input={n_in:,} "
        f"floor={floor_faces:,} ceiling={ceiling_faces:,}",
        flush=True,
    )

    best: trimesh.Trimesh | None = None
    best_faces = 0
    work_plan = dict(plan)
    for attempt in range(max(1, int(max_retries))):
        if attempt > 0:
            work_plan = refine_blender_sdf_plan(plan, attempt)
            print(
                f"[BlenderSDF] retry {attempt + 1}/{max_retries} "
                f"(prev {best_faces:,} < floor {floor_faces:,})",
                flush=True,
            )
        _run_blender_sdf_subprocess(
            input_path,
            output_path,
            work_plan,
            blender_exe=exe,
            transfer_color=transfer_color and attempt == 0,
            timeout_sec=timeout_sec,
        )
        print(f"[BlenderSDF] loading result {output_path.name}", flush=True)
        cand = trimesh.load(output_path, force="mesh")
        if isinstance(cand, trimesh.Scene):
            cand = cand.dump(concatenate=True)
        n_out = len(cand.faces)
        print(f"[BlenderSDF] attempt {attempt + 1} output_faces={n_out:,}", flush=True)
        if n_out > best_faces:
            best = cand
            best_faces = n_out
        if n_out >= floor_faces:
            break

    if best is None:
        raise RuntimeError("Blender SDF remesh produced no mesh")
    if best_faces < floor_faces:
        print(
            f"[BlenderSDF] WARN: best output {best_faces:,} still below floor {floor_faces:,}",
            flush=True,
        )

    if best_faces > ceiling_faces:
        print(
            f"[BlenderSDF] cap skipped (voxel-only): {best_faces:,} > ceiling {ceiling_faces:,}",
            flush=True,
        )

    out = best.copy()
    if isinstance(out, trimesh.Scene):
        out = out.dump(concatenate=True)
    # Blender glTF/OBJ export is Z-up; normalize to pipeline Y-up for downstream KDTree.
    out = out.copy()
    out.vertices = blender_zup_to_yup(np.asarray(out.vertices, dtype=np.float64))
    out_verts = np.asarray(out.vertices, dtype=np.float64)
    out_faces = np.asarray(out.faces, dtype=np.int64)
    if output_path.suffix.lower() == ".obj":
        write_coord_space_json(output_path.parent, COORD_YUP)
        # File is Z-up; per-file resolve uses _is_blender_export_obj suffix.
        color_written = obj_has_vertex_colors(output_path)
        # Prefer Blender-exported colors; fall back to KDTree from reference GLB.
        if transfer_color and not color_written:
            try:
                from .vertex_color import colors_for_mesh_features

                cref = Path(color_reference_path) if color_reference_path else input_path
                print(
                    f"[BlenderSDF] vertex color embed ref={cref.name} "
                    f"target={len(out.vertices):,} verts",
                    flush=True,
                )
                out_colors, has_out = colors_for_mesh_features(
                    out,
                    mesh,
                    ref_path=cref,
                    target_space=COORD_YUP,
                )
                if has_out and len(out_colors) == len(out.vertices):
                    _write_obj_with_vertex_colors(
                        output_path,
                        out_verts,
                        out_faces,
                        np.asarray(out_colors, dtype=np.float64),
                    )
                    color_written = True
            except Exception:
                pass
        if color_written:
            try:
                export_ply_preview_from_colored_obj(output_path)
            except Exception as exc:
                print(f"[BlenderSDF] PLY preview skipped: {exc}", flush=True)
        if not color_written:
            export_yup_mesh(output_path, out_verts, out_faces)
    if cache_key:
        save_sdf_cache(cache_key, output_path, faces=best_faces, plan=plan)
    return out
