"""IM teacher field generation for patch pretraining."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh

from .im_extractor import export_orientation_field, im_status
from .mesh_cleaner import save_features


def _teacher_one(obj_path: Path, patch_dir: Path) -> Path | None:
    try:
        mesh = trimesh.load(obj_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        save_features(mesh, patch_dir)
        v1, v2 = export_orientation_field(mesh, smooth_iterations=10)
        out = patch_dir / (obj_path.stem + "_teacher.npz")
        np.savez_compressed(
            out,
            teacher_v1=v1.astype(np.float32),
            teacher_v2=v2.astype(np.float32),
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
        )
        return out
    except Exception:
        return None


def build_teachers_for_dir(patch_dir: Path, *, workers: int = 1) -> list[Path]:
    patch_dir = Path(patch_dir)
    objs = sorted(patch_dir.rglob("*.obj"))
    objs = [p for p in objs if "_teacher" not in p.stem and "retopo" not in p.stem.lower()]
    results: list[Path] = []
    if workers <= 1:
        for obj in objs:
            r = _teacher_one(obj, obj.parent)
            if r:
                results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_teacher_one, obj, obj.parent): obj for obj in objs}
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    results.append(r)
    return results


def load_patch_teacher(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    return d["teacher_v1"], d["teacher_v2"]
