"""Run directory layout: runs/<input_stem>/q<target>|auto/ with staged artifacts."""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, RUNS_DIR

RUN_SHARED_DIR = RUNS_DIR / "_shared"
LEGACY_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "gnn_base.pt"
SHARED_BASE_CHECKPOINT = RUN_SHARED_DIR / "gnn_base.pt"

LAYOUT_VERSION = 2
PIPELINE_LOG_NAME = "pipeline.log"
RUN_META_NAME = "run.json"

LEGACY_CLEANED = "cleaned_mesh.obj"
LEGACY_SDF = "sdf_watertight_mesh.obj"
LEGACY_SMOOTHED = "smoothed_mesh.obj"
LEGACY_CROSS = "cross_field.npy"
LEGACY_RETOPO = "retopologized_quad_mesh.obj"
LEGACY_FEATURES = "features.npz"
LEGACY_CHECKPOINT_NAME = "gnn_checkpoint.pt"
CANONICAL_MANIFEST_NAME = "canonical.json"


def input_stem(path: Path | str) -> str:
    return Path(path).stem


def sanitize_run_name(stem: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", stem.strip())
    return s or "mesh"


def target_quads_tag(target_quads: int) -> str:
    return "qauto" if target_quads <= 0 else f"q{int(target_quads)}"


def run_dir_for_input(
    input_path: Path | str,
    *,
    target_quads: int = 0,
    runs_root: Path | None = None,
    stage: str = "run",
    suffix: str | None = None,
) -> Path:
    root = Path(runs_root) if runs_root else RUNS_DIR
    stem = sanitize_run_name(input_stem(input_path))
    return unique_run_dir(stem, runs_root=root, suffix=suffix)


def unique_run_dir(
    stem: str,
    *,
    runs_root: Path | None = None,
    suffix: str | None = None,
) -> Path:
    """Pick a non-existing runs/<stem>[_suffix|_vN]/ directory."""
    root = Path(runs_root) if runs_root else RUNS_DIR
    stem = sanitize_run_name(stem)
    names: list[str] = []
    if suffix:
        names.append(f"{stem}_{suffix}")
    names.append(stem)
    for i in range(2, 100):
        names.append(f"{stem}_v{i}")
    for name in names:
        cand = root / name
        if not cand.exists():
            return cand
    return root / f"{stem}_v99"


def _run_dir_name_candidates(stem: str, *, suffix: str | None = None) -> list[str]:
    stem = sanitize_run_name(stem)
    names: list[str] = []
    if suffix:
        names.append(f"{stem}_{suffix}")
    names.append(stem)
    for i in range(2, 100):
        names.append(f"{stem}_v{i}")
    return names


def _run_dirs_for_stem(
    stem: str,
    *,
    runs_root: Path | None = None,
    suffix: str | None = None,
) -> list[Path]:
    root = Path(runs_root) if runs_root else RUNS_DIR
    out: list[Path] = []
    for name in _run_dir_name_candidates(stem, suffix=suffix):
        cand = root / name
        if cand.is_dir():
            out.append(cand)
    return out


def _run_ready_for_extract(run_dir: Path) -> bool:
    try:
        find_cleaned_mesh(run_dir)
        find_cross_field(run_dir)
        return True
    except FileNotFoundError:
        return False


def _extract_run_rank(run_dir: Path) -> tuple[int, float]:
    """Higher = better for extract: ready runs first, then newest cross_field."""
    ready = 1 if _run_ready_for_extract(run_dir) else 0
    mtime = run_dir.stat().st_mtime
    try:
        mtime = max(mtime, find_cross_field(run_dir).stat().st_mtime)
    except FileNotFoundError:
        pass
    return ready, mtime


def resolve_existing_run_dir(
    stem: str,
    *,
    runs_root: Path | None = None,
    suffix: str | None = None,
) -> Path:
    """Pick the best existing run dir for extract (latest run with artifacts)."""
    cands = _run_dirs_for_stem(stem, runs_root=runs_root, suffix=suffix)
    if cands:
        return max(cands, key=_extract_run_rank)
    return unique_run_dir(stem, runs_root=runs_root, suffix=suffix)


@dataclass(frozen=True)
class RunLayout:
    """Per-input artifact paths under runs/<stem>/<q-tag>/."""

    root: Path
    stem: str
    input_file: str = ""

    @classmethod
    def create(
        cls,
        input_path: Path | str,
        *,
        out_dir: Path | str | None = None,
        target_quads: int = 0,
        stage: str = "run",
    ) -> RunLayout:
        inp = Path(input_path)
        stem = sanitize_run_name(input_stem(inp))
        root = Path(out_dir) if out_dir else run_dir_for_input(
            inp, target_quads=target_quads, stage=stage,
        )
        return cls(root=root, stem=stem, input_file=inp.name)

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def prep_dir(self) -> Path:
        return self.root / "prep"

    @property
    def field_dir(self) -> Path:
        return self.root / "field"

    @property
    def extract_dir(self) -> Path:
        return self.root / "extract"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    def _p(self, subdir: Path, suffix: str) -> Path:
        return subdir / f"{self.stem}_{suffix}"

    def sdf_watertight_mesh(self) -> Path:
        return self._p(self.prep_dir, "sdf_watertight_mesh.obj")

    def smoothed_mesh(self) -> Path:
        return self._p(self.prep_dir, "smoothed_mesh.obj")

    def cleaned_mesh(self) -> Path:
        return self._p(self.prep_dir, "cleaned_mesh.obj")

    def gnn_proxy_mesh(self) -> Path:
        return self._p(self.prep_dir, "gnn_proxy_mesh.obj")

    def cleaned_mesh_blender(self) -> Path:
        return self._p(self.prep_dir, "cleaned_mesh_blender.obj")

    def canonical_manifest(self) -> Path:
        return self.prep_dir / CANONICAL_MANIFEST_NAME

    def resolve_canonical_mesh(self, fallback: Path | str | None = None) -> Path:
        """Shape master: manifest → SDF obj → fallback."""
        data = None
        if self.canonical_manifest().is_file():
            try:
                data = json.loads(self.canonical_manifest().read_text(encoding="utf-8"))
                rel = data.get("canonical")
                if rel:
                    p = (self.root / rel).resolve()
                    if p.is_file():
                        return p
            except Exception:
                pass
        sdf = self.sdf_watertight_mesh()
        if sdf.is_file():
            return sdf
        if fallback is not None:
            return Path(fallback)
        raise FileNotFoundError(f"canonical mesh not found under {self.root}")

    def resolve_gnn_active_mesh(self, fallback: Path | str | None = None) -> Path:
        """GNN/IM field mesh: gnn_proxy → active → canonical → fallback."""
        if self.canonical_manifest().is_file():
            try:
                data = json.loads(self.canonical_manifest().read_text(encoding="utf-8"))
                for key in ("gnn_proxy", "active"):
                    rel = data.get(key)
                    if rel:
                        p = (self.root / rel).resolve()
                        if p.is_file():
                            return p
            except Exception:
                pass
        proxy = self.gnn_proxy_mesh()
        if proxy.is_file():
            return proxy
        try:
            return self.resolve_canonical_mesh(fallback=fallback)
        except FileNotFoundError:
            if fallback is not None:
                return Path(fallback)
            raise

    def features_npz(self) -> Path:
        return self._p(self.field_dir, "features.npz")

    def cross_field_npy(self) -> Path:
        return self._p(self.field_dir, "cross_field.npy")

    def cross_field_lines(self) -> Path:
        return self._p(self.field_dir, "cross_field_lines.obj")

    def gnn_checkpoint(self) -> Path:
        return self._p(self.field_dir, "gnn_checkpoint.pt")

    def retopo_quad_mesh(self) -> Path:
        return self._p(self.extract_dir, "retopologized_quad_mesh.obj")

    def retopo_quad_mesh_pre_shrinkwrap(self) -> Path:
        return self._p(self.extract_dir, "retopologized_quad_mesh_pre_shrinkwrap.obj")

    def retopo_quad_quality(self) -> Path:
        return self._p(self.extract_dir, "retopologized_quad_mesh_quality.obj")

    def export_retopo_quad(self) -> Path:
        return self._p(self.export_dir, "retopologized_quad_mesh.obj")

    def pipeline_log(self) -> Path:
        return self.logs_dir / PIPELINE_LOG_NAME

    def run_meta(self) -> Path:
        return self.root / RUN_META_NAME

    @property
    def extract_attempts_dir(self) -> Path:
        return self.extract_dir / "attempts"

    def ensure_dirs(self) -> None:
        for d in (
            self.root,
            self.logs_dir,
            self.prep_dir,
            self.field_dir,
            self.extract_dir,
            self.export_dir,
            self.extract_attempts_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def artifact_paths(self) -> dict[str, str]:
        return {
            "canonical_manifest": rel(self.canonical_manifest()),
            "sdf_watertight": rel(self.sdf_watertight_mesh()),
            "gnn_proxy": rel(self.gnn_proxy_mesh()),
            "smoothed": rel(self.smoothed_mesh()),
            "cleaned": rel(self.cleaned_mesh()),
            "features": rel(self.features_npz()),
            "cross_field": rel(self.cross_field_npy()),
            "checkpoint": rel(self.gnn_checkpoint()),
            "retopo_quad": rel(self.retopo_quad_mesh()),
            "retopo_quad_pre_shrinkwrap": rel(self.retopo_quad_mesh_pre_shrinkwrap()),
            "export_retopo": rel(self.export_retopo_quad()),
        }


def rel(path: Path) -> str:
    return path.as_posix()


def load_run_meta(out_dir: Path | str) -> dict[str, Any]:
    p = Path(out_dir) / RUN_META_NAME
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def layout_for_stage_dir(out_dir: Path | str) -> RunLayout:
    """Resolve run root from run root or a stage subfolder (prep/field/...)."""
    p = Path(out_dir).resolve()
    if (p / RUN_META_NAME).is_file():
        return layout_from_run_dir(p)
    if p.name in ("prep", "field", "extract", "logs", "export", "attempts"):
        return layout_from_run_dir(p.parent)
    return layout_from_run_dir(p)


def layout_from_run_dir(out_dir: Path | str) -> RunLayout:
    root = Path(out_dir).resolve()
    meta = load_run_meta(root)
    if meta.get("input_stem"):
        stem = sanitize_run_name(str(meta["input_stem"]))
    elif root.parent.name not in ("runs", "_shared", ""):
        stem = sanitize_run_name(root.parent.name)
    else:
        stem = sanitize_run_name(root.name)
    return RunLayout(
        root=root,
        stem=stem,
        input_file=str(meta.get("input_file") or ""),
    )


def _resolve_under(root: Path, p: Path | str | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = root / path
    return path if path.is_file() else None


def _meta_path(meta: dict, key: str) -> Path | None:
    paths = meta.get("paths") or {}
    rel_p = paths.get(key) or meta.get(key)
    if not rel_p:
        return None
    return Path(str(rel_p))


def find_canonical_mesh(out_dir: Path | str) -> Path:
    lay = layout_from_run_dir(out_dir)
    try:
        return lay.resolve_canonical_mesh()
    except FileNotFoundError:
        return find_cleaned_mesh(out_dir)


def find_cleaned_mesh(out_dir: Path | str) -> Path:
    root = Path(out_dir).resolve()
    meta = load_run_meta(root)
    lay = layout_from_run_dir(root)
    for cand in (
        _resolve_under(root, _meta_path(meta, "cleaned")),
        lay.cleaned_mesh(),
        root / LEGACY_CLEANED,
        root / "prep" / LEGACY_CLEANED,
    ):
        if cand is not None:
            return cand
    raise FileNotFoundError(f"cleaned mesh not found under {root}")


def find_cross_field(out_dir: Path | str) -> Path:
    root = Path(out_dir).resolve()
    meta = load_run_meta(root)
    lay = layout_from_run_dir(root)
    for cand in (
        _resolve_under(root, _meta_path(meta, "cross_field")),
        lay.cross_field_npy(),
        root / LEGACY_CROSS,
        root / "field" / LEGACY_CROSS,
    ):
        if cand is not None:
            return cand
    raise FileNotFoundError(f"cross_field.npy not found under {root}")


def find_retopo_quad(out_dir: Path | str) -> Path:
    root = Path(out_dir).resolve()
    meta = load_run_meta(root)
    lay = layout_from_run_dir(root)
    for cand in (
        _resolve_under(root, _meta_path(meta, "export_retopo")),
        lay.export_retopo_quad(),
        lay.retopo_quad_mesh(),
        root / LEGACY_RETOPO,
    ):
        if cand is not None:
            return cand
    raise FileNotFoundError(f"retopo quad mesh not found under {root}")


def _safe_copy2(src: Path, dst: Path, *, retries: int = 5, delay_s: float = 0.25) -> None:
    """Windows: skip self-copy; retry when AV/Explorer briefly locks the file."""
    src_r = Path(src).resolve()
    dst_r = Path(dst).resolve()
    if src_r == dst_r:
        return
    dst_r.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            shutil.copy2(src_r, dst_r)
            return
        except PermissionError as e:
            last = e
            if attempt + 1 < retries:
                time.sleep(delay_s)
    if last is not None:
        raise last


def publish_final_retopo(src: Path, layout: RunLayout) -> Path:
    """Copy best extract to canonical + export paths."""
    from .mesh_io import COORD_YUP, write_coord_space_json

    layout.ensure_dirs()
    src_p = Path(src).resolve()
    for dst in (layout.retopo_quad_mesh(), layout.export_retopo_quad()):
        _safe_copy2(src_p, dst)
    _safe_copy2(src_p, layout.root / LEGACY_RETOPO)
    write_coord_space_json(layout.extract_dir, COORD_YUP)
    write_coord_space_json(layout.export_dir, COORD_YUP)
    write_coord_space_json(layout.root, COORD_YUP)
    return layout.export_retopo_quad()


def resolve_out_dir(
    out: Path | str | None,
    *,
    input_path: Path | str | None = None,
    mesh_path: Path | str | None = None,
    target_quads: int = 0,
) -> Path:
    if out is not None and str(out).strip():
        p = Path(out)
        if p.resolve() != (RUNS_DIR / "final").resolve():
            return p
    if mesh_path is not None:
        mp = Path(mesh_path)
        if mp.name.endswith("_cleaned_mesh.obj") or mp.name == LEGACY_CLEANED:
            return mp.parent.parent if mp.parent.name == "prep" else mp.parent
        if load_run_meta(mp.parent).get("layout_version"):
            return mp.parent
    if input_path is not None:
        return run_dir_for_input(input_path, target_quads=target_quads)
    return RUNS_DIR / "final"


def default_pretrained_checkpoint() -> Path:
    if SHARED_BASE_CHECKPOINT.is_file():
        return SHARED_BASE_CHECKPOINT
    return LEGACY_CHECKPOINT


def pipeline_log_path(out_dir: Path | str) -> Path:
    root = Path(out_dir)
    lay = layout_from_run_dir(root)
    return lay.pipeline_log()


def write_run_meta(out_dir: Path, meta: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / RUN_META_NAME
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def write_layout_meta(layout: RunLayout, extra: dict | None = None) -> Path:
    meta = {
        "layout_version": LAYOUT_VERSION,
        "input_stem": layout.stem,
        "input_file": layout.input_file,
        "run_root": rel(layout.root),
        "paths": layout.artifact_paths(),
    }
    if extra:
        meta.update(extra)
    return write_run_meta(layout.root, meta)
