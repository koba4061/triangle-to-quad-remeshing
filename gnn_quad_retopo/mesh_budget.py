"""Face / quad budgets from input mesh size."""
from __future__ import annotations

import os
from pathlib import Path

import trimesh

from .auto_tune import RuntimeEnv, _full_batch_vertex_limit, detect_runtime


def count_mesh_faces(path: Path | str) -> int:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return int(len(mesh.faces))


def suggest_target_quads(n_input_faces: int) -> int:
    """Heuristic IM quad count from input triangle count.

    Large mechanical meshes (~2M tris) land near 700k quads.
    """
    if n_input_faces <= 0:
        return 300_000
    if n_input_faces < 80_000:
        return max(5_000, n_input_faces // 4)
    if n_input_faces < 400_000:
        return int(n_input_faces / 2.8)
    q = int(n_input_faces / 2.6)
    lo = 250_000 if n_input_faces >= 500_000 else 50_000
    hi = 2_000_000 if n_input_faces >= 4_000_000 else 1_200_000
    return int(max(lo, min(hi, q)))


def suggest_target_quads_from_cleaned(n_clean_faces: int) -> int:
    """When only cleaned_mesh.obj exists (~decimated), map back to input-scale budget."""
    return suggest_target_quads(int(max(n_clean_faces, 1) * 1.29))


def auto_sdf_face_target(
    input_faces: int,
    *,
    detail_ratio: float = 1.0,
    min_faces: int = 300_000,
    max_faces: int = 2_800_000,
) -> int:
    """SDF output target ≈ input_faces (detail preserve), clamped for GNN/IM budget.

    N_sdf ≈ N_in × detail_ratio × k,  k∈[0.92,1.0] by size.
    """
    n = int(input_faces)
    if n <= 0:
        return int(min_faces)
    if n >= 2_500_000:
        k = 0.92
    elif n >= 1_000_000:
        k = 0.98
    else:
        k = 1.0
    tgt = int(round(n * float(detail_ratio) * k))
    return int(max(min_faces, min(max_faces, tgt)))


def resolve_sdf_face_target(
    explicit: int | None,
    *,
    input_faces: int,
    detail_ratio: float = 1.0,
) -> int:
    """0/None → auto from input; else use explicit cap."""
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    return auto_sdf_face_target(input_faces, detail_ratio=detail_ratio)


CANONICAL_INPUT_FACE_RATIO_MAX = 1.5
# Shrinkwrap canonical: aim above input tri count for surface fidelity.
CANONICAL_DETAIL_TARGET_RATIO = 1.38
# Above this: pre-SDF decimate, fast color path, etc.
LARGE_MESH_INPUT_FACES = 3_500_000

# SDF face budgets: (min, max) per GPU tier — scaled from input, clamped to these bounds.
CANONICAL_SDF_TIER_LIMITS: dict[str, tuple[int, int]] = {
    "cpu": (250_000, 800_000),
    "low": (250_000, 2_000_000),
    "mid": (300_000, 3_200_000),
    "high": (300_000, 4_000_000),
    "xl": (300_000, 5_000_000),
    "xxl": (300_000, 8_000_000),
}
PROXY_SDF_TIER_LIMITS: dict[str, tuple[int, int]] = {
    "cpu": (100_000, 350_000),
    "low": (150_000, 750_000),
    "mid": (150_000, 1_100_000),
    "high": (200_000, 1_300_000),
    "xl": (200_000, 1_500_000),
    "xxl": (250_000, 2_000_000),
}
# Below this input tri count, use input size as-is (no upscaling).
CANONICAL_SDF_INPUT_FLOOR = 300_000
PROXY_SDF_INPUT_FLOOR = 150_000
# Proxy target ≈ this fraction of original input (GNN learning mesh).
PROXY_INPUT_RATIO = 0.58


def clamp_face_budget(value: int, min_faces: int, max_faces: int) -> int:
    lo = int(min(min_faces, max_faces))
    hi = int(max(min_faces, max_faces))
    return int(max(lo, min(hi, int(value))))


def _tier_sdf_limits(
    role: str,
    runtime: RuntimeEnv | None,
    *,
    input_faces: int,
) -> tuple[int, int]:
    rt = runtime or detect_runtime()
    table = CANONICAL_SDF_TIER_LIMITS if role == "canonical" else PROXY_SDF_TIER_LIMITS
    min_f, max_f = table.get(rt.gpu_tier, table["mid"])
    if rt.vram_effective_gb is not None and rt.gpu_tier in ("mid", "high", "xl", "xxl"):
        boost = int(max(0.0, rt.vram_effective_gb - 6.0) * 200_000)
        max_f = min(int(input_faces * CANONICAL_INPUT_FACE_RATIO_MAX), max_f + boost)
    return int(min_f), int(max_f)


def resolve_dynamic_sdf_target(
    input_faces: int,
    *,
    role: str,
    runtime: RuntimeEnv | None = None,
    canonical_faces: int | None = None,
) -> tuple[int, int, int, str]:
    """Return (min_faces, max_faces, target_faces, reason) from input + environment."""
    n_in = max(1, int(input_faces))
    min_f, max_f = _tier_sdf_limits(role, runtime, input_faces=n_in)
    rt = runtime or detect_runtime()

    manual = _env_canonical_sdf_budget() if role == "canonical" else _env_proxy_budget()
    if manual is not None:
        tgt = clamp_face_budget(manual, min_f, max_f)
        return min_f, max_f, min(tgt, int(n_in * CANONICAL_INPUT_FACE_RATIO_MAX)), "manual_env"

    if role == "canonical":
        if n_in <= min_f:
            return min_f, max_f, n_in, "input_below_min_use_input"
        raw = int(round(n_in * CANONICAL_DETAIL_TARGET_RATIO))
        raw = min(raw, int(n_in * CANONICAL_INPUT_FACE_RATIO_MAX))
        # Shrinkwrap: never target below input when input is already substantial.
        if n_in >= CANONICAL_SDF_INPUT_FLOOR:
            raw = max(raw, n_in)
        tgt = clamp_face_budget(raw, min_f, max_f)
        return min_f, max_f, tgt, f"auto_{rt.gpu_tier}_in{n_in:,}"

    canon = int(canonical_faces) if canonical_faces and canonical_faces > 0 else n_in
    if n_in <= PROXY_SDF_INPUT_FLOOR or canon <= min_f:
        tgt = min(canon, n_in)
        return min_f, max_f, tgt, "input_below_min_use_input"
    raw = int(round(n_in * PROXY_INPUT_RATIO))
    raw = min(raw, int(canon * 0.92))
    tgt = clamp_face_budget(raw, min_f, max_f)
    return min_f, max_f, tgt, f"auto_{rt.gpu_tier}_ref{n_in:,}"


def _env_canonical_sdf_budget() -> int | None:
    for key in ("RETOPO_CANONICAL_SDF_FACE_BUDGET", "RETOPO_SDF_FACE_BUDGET"):
        v = os.environ.get(key, "").strip()
        if not v or v == "0":
            continue
        try:
            return max(100_000, int(v))
        except ValueError:
            continue
    return None


def auto_canonical_sdf_face_target(
    input_faces: int,
    *,
    max_input_ratio: float = CANONICAL_INPUT_FACE_RATIO_MAX,
    target_ratio: float = CANONICAL_DETAIL_TARGET_RATIO,
    runtime: RuntimeEnv | None = None,
) -> int:
    """High-detail canonical SDF for shrinkwrap (input + VRAM tier min/max)."""
    _, _, tgt, _ = resolve_dynamic_sdf_target(
        input_faces, role="canonical", runtime=runtime,
    )
    return tgt


def plan_pre_sdf_input_faces(
    input_faces: int,
    canonical_sdf_target: int,
    runtime: RuntimeEnv | None = None,
) -> tuple[int, str]:
    """Decimate before Blender SDF when input is huge (quadric, detail-preserving)."""
    n = int(input_faces)
    if n <= LARGE_MESH_INPUT_FACES:
        return n, "skip"

    rt = runtime or detect_runtime()
    tier_pre_cap = {
        "cpu": 600_000,
        "low": 1_200_000,
        "mid": 2_500_000,
        "high": 3_500_000,
        "xl": 5_000_000,
        "xxl": 8_000_000,
    }
    pre_cap = tier_pre_cap.get(rt.gpu_tier, 2_500_000)
    manual = _env_canonical_sdf_budget()
    if manual is not None:
        pre_cap = min(pre_cap, manual)

    canon = int(max(300_000, canonical_sdf_target))
    budget = min(n, max(canon, pre_cap))
    if budget >= int(n * 0.95):
        return n, "skip"
    return budget, f"pre-sdf quadric {n:,}->{budget:,} (canonical={canon:,})"


def resolve_canonical_sdf_face_target(
    explicit: int | None,
    *,
    input_faces: int,
    runtime: RuntimeEnv | None = None,
) -> int:
    """Canonical SDF face target; explicit cap clamped to 1.5× input."""
    n = int(input_faces)
    cap = int(n * CANONICAL_INPUT_FACE_RATIO_MAX) if n > 0 else 2_800_000
    if explicit is not None and int(explicit) > 0:
        return min(cap, int(explicit))
    return auto_canonical_sdf_face_target(n, runtime=runtime)


def resolve_target_quads(target: int, *, input_faces: int | None = None) -> int:
    if target > 0:
        return int(target)
    if input_faces is None or input_faces <= 0:
        return 300_000
    return suggest_target_quads(input_faces)


def _env_proxy_budget() -> int | None:
    v = os.environ.get("RETOPO_GNN_PROXY_FACE_BUDGET", "").strip()
    if not v or v == "0":
        return None
    try:
        return max(100_000, int(v))
    except ValueError:
        return None


def auto_gnn_proxy_face_target(
    canonical_faces: int,
    runtime: RuntimeEnv | None = None,
    *,
    input_faces: int | None = None,
) -> tuple[int, str]:
    """GNN proxy SDF face budget from original input + VRAM tier min/max."""
    n_canon = int(canonical_faces)
    ref = int(input_faces) if input_faces and input_faces > 0 else n_canon
    if n_canon <= 0:
        return PROXY_SDF_INPUT_FLOOR, "default"

    min_f, max_f, tgt, reason = resolve_dynamic_sdf_target(
        ref,
        role="proxy",
        runtime=runtime,
        canonical_faces=n_canon,
    )
    if n_canon <= int(tgt * 1.05):
        return n_canon, "skip_canonical_fits_vram"
    return tgt, reason


def resolve_gnn_proxy_face_target(
    canonical_faces: int,
    runtime: RuntimeEnv | None = None,
    *,
    input_faces: int | None = None,
) -> tuple[int, str, bool]:
    """Returns (target_faces, reason, needs_proxy)."""
    tgt, reason = auto_gnn_proxy_face_target(
        canonical_faces, runtime, input_faces=input_faces,
    )
    needs = canonical_faces > int(tgt * 1.05)
    return tgt, reason, needs


def needs_gnn_proxy(
    canonical_faces: int,
    runtime: RuntimeEnv | None = None,
    *,
    input_faces: int | None = None,
) -> bool:
    _, _, needs = resolve_gnn_proxy_face_target(
        canonical_faces, runtime, input_faces=input_faces,
    )
    return needs
