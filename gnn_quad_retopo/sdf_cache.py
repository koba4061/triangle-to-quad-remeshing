"""Disk cache for Blender SDF remesh outputs (runs/_cache/sdf/)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .config import RUNS_DIR

_CACHE_ROOT = RUNS_DIR / "_cache" / "sdf"
_META_SUFFIX = ".json"


def sdf_cache_enabled() -> bool:
    v = os.environ.get("RETOPO_SDF_CACHE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def sdf_cache_dir() -> Path:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _CACHE_ROOT


def _file_sha1(path: Path) -> str:
    if not path.is_file():
        return f"none_{path.name}"
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:20]


def sdf_cache_key(
    input_path: Path,
    plan: dict,
    *,
    role: str = "canonical",
    input_face_count: int | None = None,
) -> str:
    """Stable cache key from input hash + SDF plan parameters."""
    parts = [
        _file_sha1(input_path),
        role,
        f"a{plan.get('adaptivity', 0):.4f}",
        f"t{int(plan.get('target_sdf_faces', 0))}",
        f"v{plan.get('voxel_size', 0):.8g}",
        f"b{int(plan.get('band_width', 0))}",
    ]
    if input_face_count is not None:
        parts.append(f"f{int(input_face_count)}")
    return "_".join(parts)


def sdf_cache_paths(key: str) -> tuple[Path, Path]:
    d = sdf_cache_dir()
    return d / f"{key}.obj", d / f"{key}{_META_SUFFIX}"


def try_load_sdf_cache(key: str, *, max_faces: int | None = None) -> Path | None:
    """Return cached OBJ path if present and valid."""
    if not sdf_cache_enabled():
        return None
    obj, meta = sdf_cache_paths(key)
    if not obj.is_file() or obj.stat().st_size < 1024:
        return None
    if meta.is_file():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            faces = int(data.get("faces", 0))
            if faces <= 0:
                return None
            if max_faces is not None and faces > int(max_faces * 1.05):
                print(
                    f"[BlenderSDF] cache reject oversized {faces:,} > {int(max_faces):,} ({key[:24]}...)",
                    flush=True,
                )
                return None
        except Exception:
            pass
    print(f"[BlenderSDF] cache hit {key[:24]}... -> {obj.name}", flush=True)
    return obj


def save_sdf_cache(key: str, obj_path: Path, *, faces: int, plan: dict) -> None:
    if not sdf_cache_enabled() or not obj_path.is_file():
        return
    dst, meta = sdf_cache_paths(key)
    if dst.resolve() == obj_path.resolve():
        return
    try:
        shutil.copy2(obj_path, dst)
        meta.write_text(
            json.dumps({"faces": int(faces), "plan": plan}, indent=0),
            encoding="utf-8",
        )
        print(f"[BlenderSDF] cache save {key[:24]}... ({faces:,} faces)", flush=True)
    except Exception as exc:
        print(f"[BlenderSDF] cache save skipped: {exc}", flush=True)
