"""Disk cache for features.npz (runs/_cache/features/)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .config import RUNS_DIR

_CACHE_ROOT = RUNS_DIR / "_cache" / "features"
_META_SUFFIX = ".json"


def features_cache_enabled() -> bool:
    v = os.environ.get("RETOPO_FEATURES_CACHE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def features_cache_dir() -> Path:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _CACHE_ROOT


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:20]


def features_cache_key(
    original_input: Path | None,
    mesh_path: Path,
    *,
    n_verts: int,
    n_faces: int,
    light_clean: bool = False,
) -> str:
    parts = [
        _file_sha1(original_input) if original_input and original_input.is_file() else "noinput",
        _file_sha1(mesh_path) if mesh_path.is_file() else mesh_path.stem,
        f"v{n_verts}",
        f"f{n_faces}",
        "light" if light_clean else "full",
        "vcol2",
    ]
    return "_".join(parts)


def features_cache_paths(key: str) -> tuple[Path, Path]:
    d = features_cache_dir()
    return d / f"{key}.npz", d / f"{key}{_META_SUFFIX}"


def try_load_features_cache(key: str, dest: Path) -> bool:
    if not features_cache_enabled():
        return False
    src, meta = features_cache_paths(key)
    if not src.is_file() or src.stat().st_size < 256:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        if meta.is_file():
            json.loads(meta.read_text(encoding="utf-8"))
        print(f"[features] cache hit -> {dest.name}", flush=True)
        return True
    except Exception as exc:
        print(f"[features] cache load failed: {exc}", flush=True)
        return False


def save_features_cache(key: str, npz_path: Path, *, n_verts: int, n_faces: int) -> None:
    if not features_cache_enabled() or not npz_path.is_file():
        return
    dst, meta = features_cache_paths(key)
    if dst.resolve() == npz_path.resolve():
        return
    try:
        shutil.copy2(npz_path, dst)
        meta.write_text(
            json.dumps({"n_verts": n_verts, "n_faces": n_faces}, indent=0),
            encoding="utf-8",
        )
        print(f"[features] cache save ({n_verts:,} verts)", flush=True)
    except Exception as exc:
        print(f"[features] cache save skipped: {exc}", flush=True)
