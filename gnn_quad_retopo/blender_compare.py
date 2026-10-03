"""Blender comparison helpers (also runnable via MCP execute_blender_code)."""
from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_TEST_MESH, PROJECT_ROOT


def comparison_paths(run_dir: Path | None = None) -> dict[str, Path]:
    from .run_layout import find_cleaned_mesh, find_retopo_quad, layout_from_run_dir

    run = Path(run_dir) if run_dir else PROJECT_ROOT / "runs" / "current"
    lay = layout_from_run_dir(run)
    return {
        "input": DEFAULT_TEST_MESH,
        "cleaned": find_cleaned_mesh(run),
        "cross_field_viz": lay.cross_field_lines(),
        "quad_out": find_retopo_quad(run),
    }


BLENDER_SETUP_SCRIPT = r'''
import bpy, os

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_mesh(path, name, x_offset):
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return None
    if path.lower().endswith('.glb') or path.lower().endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        bpy.ops.wm.obj_import(filepath=path)
    obj = bpy.context.selected_objects[-1]
    obj.name = name
    obj.location.x = x_offset
    me = obj.data
    print(f"{name}: V={len(me.vertices)} F={len(me.polygons)}")
    return obj

clear_scene()
paths = {paths_repr}
x = 0
for label, p in paths.items():
    import_mesh(p, label, x)
    x += 3.0

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'SOLID'
        override = bpy.context.copy()
        override['area'] = area
        override['region'] = area.regions[-1]
        with bpy.context.temp_override(**override):
            bpy.ops.view3d.view_all(center=True)
print("COMPARE_SCENE_READY")
'''


def build_blender_compare_code(run_dir: Path) -> str:
    paths = comparison_paths(run_dir)
    paths_str = ",\n".join(f'"{k}": r"{p}"' for k, p in paths.items() if p.exists() or k == "input")
    return BLENDER_SETUP_SCRIPT.replace("{paths_repr}", "{" + paths_str + "}")
