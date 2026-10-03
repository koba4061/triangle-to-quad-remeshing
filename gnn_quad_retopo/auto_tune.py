"""GPU / Colab / mesh-size aware training defaults."""
from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

AUTO_BATCH = -1
SDF_QUERY_FACE_HARD_MAX = 800_000

# 実効 VRAM = min(total, free) * safety — OOM 回避用（Colab 切断対策）
_DEFAULT_SAFETY = 0.88


@dataclass
class RuntimeEnv:
    platform: str
    is_colab: bool
    device: str
    cuda_available: bool
    gpu_name: str | None
    vram_total_gb: float | None
    vram_free_gb: float | None
    vram_effective_gb: float | None
    gpu_tier: str
    torch_version: str | None
    ram_gb: float | None
    notes: list[str]


def _is_colab() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def gpu_mem_info() -> tuple[float, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return free / 1e9, total / 1e9
    except Exception:
        return None


def _env_float(name: str) -> float | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def effective_vram_gb(*, free_gb: float | None, total_gb: float | None, safety: float) -> float | None:
    """計画用の保守的 VRAM（GB）。"""
    override = _env_float("GNN_AUTO_VRAM_GB")
    if override is not None:
        return max(1.0, override)
    if total_gb is None and free_gb is None:
        return None
    if total_gb is None:
        return max(1.0, free_gb * safety) if free_gb else None
    if free_gb is None:
        return max(1.0, total_gb * safety)
    return max(1.0, min(total_gb, free_gb) * safety)


def classify_gpu_tier(vram_effective_gb: float | None, *, cuda: bool) -> str:
    if not cuda or vram_effective_gb is None:
        return "cpu"
    v = vram_effective_gb
    if v < 6:
        return "low"       # T4 15GB 実効 ~12, Colab 無料枠のばらつき
    if v < 14:
        return "mid"       # 8–12 GB クラス
    if v < 28:
        return "high"      # 16–24 GB
    if v < 48:
        return "xl"        # A100 40GB 等
    return "xxl"           # A100 80GB, H100


def detect_runtime(requested_device: str | None = None) -> RuntimeEnv:
    notes: list[str] = []
    is_colab = _is_colab()
    if is_colab:
        notes.append("colab")

    torch_version = None
    cuda_available = False
    gpu_name = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0) or None
    except Exception:
        pass

    mem = gpu_mem_info()
    free_gb = mem[0] if mem else None
    total_gb = mem[1] if mem else None
    safety = _env_float("GNN_AUTO_SAFETY") or _DEFAULT_SAFETY
    if is_colab:
        safety = min(safety, 0.85)
        notes.append("colab_safety=0.85")

    vram_eff = effective_vram_gb(free_gb=free_gb, total_gb=total_gb, safety=safety)
    tier = classify_gpu_tier(vram_eff, cuda=cuda_available)
    device = requested_device or ("cuda" if cuda_available else "cpu")

    ram_gb = None
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        pass

    if gpu_name:
        gn = gpu_name.upper()
        if "A100" in gn:
            notes.append("gpu_a100")
        elif "T4" in gn:
            notes.append("gpu_t4")
        elif "L4" in gn:
            notes.append("gpu_l4")
        elif "V100" in gn:
            notes.append("gpu_v100")

    return RuntimeEnv(
        platform=platform.system(),
        is_colab=is_colab,
        device=device,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        vram_total_gb=round(total_gb, 2) if total_gb else None,
        vram_free_gb=round(free_gb, 2) if free_gb else None,
        vram_effective_gb=round(vram_eff, 2) if vram_eff else None,
        gpu_tier=tier,
        torch_version=torch_version,
        ram_gb=ram_gb,
        notes=notes,
    )


def detect_device(requested: str | None = None) -> str:
    return detect_runtime(requested).device


def _round_up_pow2(n: int) -> int:
    if n <= 0:
        return 4096
    return 1 << (n - 1).bit_length()


def _batch_cap_for_tier(tier: str) -> int:
    env_cap = _env_int("GNN_AUTO_BATCH_CAP")
    if env_cap is not None:
        return max(4096, env_cap)
    caps = {
        "cpu": 4096,
        "low": 8192,
        "mid": 32768,
        "high": 65536,
        "xl": 131072,
        "xxl": 262144,
    }
    return caps.get(tier, 32768)


def _target_steps_for_tier(tier: str, *, is_colab: bool) -> int:
    base = {
        "cpu": 48,
        "low": 32,
        "mid": 18,
        "high": 14,
        "xl": 12,
        "xxl": 10,
    }.get(tier, 20)
    if is_colab and tier in ("low", "mid"):
        base += 4
    return base


def _full_batch_vertex_limit(vram_eff_gb: float) -> int:
    """full-batch 許容頂点数（経験則: 8GB 実効 ≈ 120k）。"""
    return int(120_000 * (vram_eff_gb / 8.0))


def suggest_batch_size(
    n_vertices: int,
    *,
    in_dim: int = 18,
    gpu_total_gb: float | None = None,
    runtime: RuntimeEnv | None = None,
) -> int:
    """0 = full-batch; else mini-batch seed count (power of 2)."""
    if n_vertices <= 0:
        return 4096

    rt = runtime or detect_runtime()
    mem = gpu_total_gb
    if mem is None and rt.vram_effective_gb is not None:
        mem = rt.vram_effective_gb
    if mem is None:
        return min(4096, n_vertices)

    tier = rt.gpu_tier if rt.vram_effective_gb is not None else classify_gpu_tier(mem, cuda=True)
    full_lim = _full_batch_vertex_limit(mem)
    if n_vertices <= full_lim and tier != "cpu" and mem >= 5.0:
        return 0

    target_steps = _target_steps_for_tier(tier, is_colab=rt.is_colab)
    raw = max(4096, (n_vertices + target_steps - 1) // target_steps)
    batch = _round_up_pow2(raw)
    batch = min(batch, n_vertices, _batch_cap_for_tier(tier))
    return max(4096, batch)





def build_train_plan(
    n_vertices: int,
    *,
    in_dim: int = 18,
    device: str | None = None,
) -> dict:
    rt = detect_runtime(device)
    batch = suggest_batch_size(n_vertices, in_dim=in_dim, runtime=rt)
    use_mini = batch > 0 and n_vertices > batch
    steps = max(1, (n_vertices + batch - 1) // batch) if use_mini else 1
    return {
        "runtime": asdict(rt),
        "n_vertices": n_vertices,
        "in_dim": in_dim,
        "batch_size": batch if use_mini else 0,
        "batch_mode": "mini" if use_mini else "full",
        "steps_per_epoch": steps,
    }


def resolve_batch_size(
    batch_size: int,
    n_vertices: int,
    *,
    in_dim: int = 18,
    device: str = "cuda",
) -> int:
    if batch_size != AUTO_BATCH:
        return batch_size
    rt = detect_runtime(device if device else None)
    mem = rt.vram_effective_gb if device.startswith("cuda") else None
    return suggest_batch_size(n_vertices, in_dim=in_dim, gpu_total_gb=mem, runtime=rt)


def save_auto_tune_report(out_dir: Path, plan: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "auto_tune.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def format_auto_train_line(
    *,
    device: str,
    batch_size: int,
    n_vertices: int,
    steps_per_epoch: int,
    gpu_total_gb: float | None,
    runtime: RuntimeEnv | None = None,
) -> str:
    rt = runtime or detect_runtime(device)
    mode = "full" if batch_size == 0 else "mini"
    gpu_s = rt.gpu_name or ("CPU" if not rt.cuda_available else "?")
    mem_s = f"{gpu_total_gb:.1f}GB eff" if gpu_total_gb else "CPU"
    colab_s = "colab " if rt.is_colab else ""
    return (
        f"auto-tune: {colab_s}tier={rt.gpu_tier} device={device} gpu={gpu_s} vram={mem_s} "
        f"verts={n_vertices:,} batch={batch_size if batch_size else 'full'} mode={mode} "
        f"steps/epoch={steps_per_epoch}"
    )


_SDF_RES_CANDIDATES = (512, 448, 384, 320, 256, 224, 192, 160, 144, 128, 112, 96, 80, 64)


def _hardware_max_sdf_resolution(vram_eff_gb: float | None) -> int:
    """VRAM-based ceiling for SDF grid (eval + dual contouring)."""
    if vram_eff_gb is None:
        return 128
    v = vram_eff_gb
    if v < 5:
        return 96
    if v < 7:
        return 192
    if v < 9:
        return 224
    if v < 11:
        return 256
    if v < 14:
        return 256
    if v < 22:
        return 320
    if v < 36:
        return 384
    return 512


def _clamp_int(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(value))))


def _snap_sdf_resolution(target: int, tier_max: int) -> int:
    target = max(64, min(tier_max, target))
    for res in _SDF_RES_CANDIDATES:
        if res <= target:
            return res
    return 64


def _sdf_max_resolution(tier: str, vram_eff_gb: float | None) -> int:
    """Hard ceiling for SDF grid resolution (VRAM-first, tier as fallback).

    ``GNN_SDF_RES_MAX`` (e.g. 256 / 320) raises the ceiling above the VRAM default
    for a one-off high-quality run (may OOM on low VRAM).
    """
    if vram_eff_gb is not None:
        hw = _hardware_max_sdf_resolution(vram_eff_gb)
    else:
        hw = {
            "cpu": 96,
            "low": 128,
            "mid": 160,
            "high": 224,
            "xl": 256,
            "xxl": 320,
        }.get(tier, 128)
    env = _env_int("GNN_SDF_RES_MAX")
    if env is not None and env > 0:
        return max(hw, _snap_sdf_resolution(env, 512))
    return hw


def _sdf_query_face_cap(n_faces: int, tier: str, ram_gb: float | None) -> int:
    cap = {
        "cpu": 150_000,
        "low": 350_000,
        "mid": 700_000,
        "high": 1_200_000,
        "xl": 1_800_000,
        "xxl": 2_500_000,
    }.get(tier, 700_000)
    if ram_gb is not None:
        if ram_gb >= 64:
            cap = max(cap, min(n_faces, 2_500_000))
        elif ram_gb >= 32:
            cap = max(cap, min(n_faces, 1_500_000))
        elif ram_gb >= 24:
            cap = max(cap, min(n_faces, 1_000_000))
        elif ram_gb < 12:
            cap = int(cap * 0.55)
    return min(n_faces, cap)


def _sdf_query_face_limit(
    n_faces: int,
    sdf_res: int,
    complexity: float,
    tier: str,
    *,
    ram_gb: float | None = None,
    override: int | None = None,
    no_simplify: bool = False,
) -> int:
    if no_simplify or override is not None and override <= 0:
        return n_faces
    if override is not None and override > 0:
        return min(n_faces, override)

    complexity_scale = max(0.75, min(1.7, complexity * 2.2))
    res_scale = (sdf_res / 128.0) ** 1.2

    keep_ratio = {
        "cpu": 0.12,
        "low": 0.22,
        "mid": 0.38,
        "high": 0.52,
        "xl": 0.62,
        "xxl": 0.72,
    }.get(tier, 0.38)
    if complexity >= 0.35:
        keep_ratio *= max(0.92, min(1.4, complexity * 2.0))
    if n_faces >= 800_000 and ram_gb is not None and ram_gb >= 24:
        keep_ratio = max(keep_ratio, 0.50 + min(0.22, complexity * 0.25))
    if n_faces >= 1_500_000 and ram_gb is not None and ram_gb >= 32:
        keep_ratio = max(keep_ratio, 0.58 + min(0.15, complexity * 0.12))

    face_cap = _sdf_query_face_cap(n_faces, tier, ram_gb)
    res_based = sdf_res**2 * 12.0 * complexity_scale * res_scale
    face_based = n_faces * keep_ratio
    target = int(max(80_000, res_based, face_based))
    return min(n_faces, max(80_000, target), face_cap)


def _sdf_pool_sizes(
    n_faces: int,
    complexity: float,
    tier: str,
    *,
    ram_gb: float | None,
) -> tuple[int, int]:
    complexity_scale = max(0.65, min(1.75, complexity * 2.5))
    face_scale = (max(n_faces, 1) / 100_000.0) ** 0.6
    tier_lo_hi = {
        "cpu": (20_000, 80_000),
        "low": (40_000, 150_000),
        "mid": (60_000, 280_000),
        "high": (90_000, 450_000),
        "xl": (120_000, 650_000),
        "xxl": (150_000, 900_000),
    }
    lo, hi = tier_lo_hi.get(tier, (60_000, 280_000))
    if ram_gb is not None and ram_gb < 12:
        hi = min(hi, 120_000)
    elif ram_gb is not None and ram_gb < 24:
        hi = min(hi, 280_000)
    elif ram_gb is not None and ram_gb >= 32:
        hi = int(hi * 1.35)
    elif ram_gb is not None and ram_gb >= 48:
        hi = int(hi * 1.6)

    num_surf = _clamp_int(35_000 * face_scale * complexity_scale, lo, hi)
    uni_ratio = 0.6 if complexity < 0.35 else 0.85
    num_uni = _clamp_int(num_surf * uni_ratio, lo // 2, hi // 2)
    return num_surf, num_uni


def _sdf_batch_size(pool_size: int, tier: str) -> int:
    caps = {
        "cpu": 4096,
        "low": 8192,
        "mid": 16_384,
        "high": 32_768,
        "xl": 65_536,
        "xxl": 131_072,
    }
    cap = caps.get(tier, 16_384)
    batch = min(pool_size, cap)
    batch = _round_up_pow2(max(4096, batch // 2))
    return min(batch, pool_size)


def _sdf_train_steps(
    pool_size: int,
    batch_size: int,
    complexity: float,
    tier: str,
) -> int:
    base_passes = {
        "cpu": 8,
        "low": 10,
        "mid": 14,
        "high": 18,
        "xl": 22,
        "xxl": 24,
    }.get(tier, 14)
    passes = int(base_passes * max(0.85, min(1.5, complexity * 2.0)))
    min_steps = {
        "cpu": 400,
        "low": 600,
        "mid": 800,
        "high": 1000,
        "xl": 1200,
        "xxl": 1500,
    }.get(tier, 800)
    max_steps = {
        "cpu": 1500,
        "low": 2500,
        "mid": 4000,
        "high": 6000,
        "xl": 8000,
        "xxl": 10_000,
    }.get(tier, 4000)
    steps = max(min_steps, (pool_size * passes) // max(batch_size, 1))
    return min(steps, max_steps)


def _estimate_query_vertices(n_vertices: int, n_faces: int, query_faces: int) -> int:
    if n_faces <= 0:
        return n_vertices
    ratio = query_faces / n_faces
    return max(1000, int(n_vertices * ratio * 0.98))


def _sdf_query_face_hard_max() -> int:
    override = _env_int("GNN_SDF_QUERY_FACE_MAX")
    if override is not None and override > 0:
        return override
    return SDF_QUERY_FACE_HARD_MAX


def _mesh_footprint_gb(n_faces: int, n_vertices: int) -> float:
    return max(0.10, (n_faces * 450 + n_vertices * 64) / 1e9)


def _cap_query_faces_for_sd_ram(
    n_faces: int,
    n_vertices: int,
    query_faces: int,
    *,
    ram_gb: float | None,
    sd_workers: int,
    sd_gb_per_worker: float,
) -> tuple[int, int | None]:
    """Cap SDF query faces so ProcessPool mesh copies fit RAM (workers unchanged)."""
    if ram_gb is None or query_faces <= 80_000:
        return query_faces, None

    floor = 80_000

    mesh_copy_gb = (sd_workers + 1) * sd_gb_per_worker
    mesh_budget = ram_gb * (0.065 if ram_gb < 16 else 0.070 if ram_gb < 24 else 0.075)
    if mesh_copy_gb <= mesh_budget:
        return query_faces, None

    ratio = mesh_budget / mesh_copy_gb
    capped = max(floor, min(query_faces, int(query_faces * ratio * 0.98)))
    if capped >= query_faces:
        return query_faces, None
    return capped, query_faces


def compute_sdf_sd_settings(
    n_query_faces: int,
    n_query_vertices: int,
    n_points: int,
    *,
    ram_gb: float | None = None,
    cpu_count: int | None = None,
    n_input_faces: int | None = None,
) -> dict:
    """ProcessPool workers / chunk size from RAM and query mesh size."""
    import os

    cpu = max(1, cpu_count or os.cpu_count() or 4)
    ram = ram_gb if ram_gb is not None else 8.0
    n_points = max(1, n_points)
    n_query_faces = max(1, n_query_faces)
    n_query_vertices = max(3, n_query_vertices)

    # trimesh BVH + mesh copy per worker (conservative)
    gb_per_worker = max(
        0.10,
        (n_query_faces * 450 + n_query_vertices * 64) / 1e9,
    )
    if ram < 16:
        usable_frac = 0.65
        train_reserve = max(4.0, ram * 0.30)
    elif ram < 24:
        usable_frac = 0.78
        train_reserve = max(5.0, ram * 0.25)
    else:
        usable_frac = 0.88
        train_reserve = max(6.0, min(12.0, ram * 0.22))
    os_reserve = max(2.0, ram * 0.08)
    budget = max(0.5, ram * usable_frac - train_reserve - os_reserve)
    by_ram = max(1, int(budget / gb_per_worker))

    for threshold, cap in (
        (2_000_000, 2),
        (1_500_000, 3),
        (1_000_000, 4),
        (500_000, 6),
        (250_000, 8),
    ):
        if n_query_faces >= threshold:
            by_ram = min(by_ram, cap)
            break

    workers = max(1, min(cpu, by_ram))
    if n_input_faces is not None:
        if n_input_faces >= 1_000_000:
            workers = min(workers, 4)
        elif n_input_faces >= 500_000:
            workers = min(workers, 6)
    target_chunks = max(workers * 3, 4)
    chunk = _clamp_int(n_points // target_chunks, 10_000, 50_000)

    executor = "process"
    if gb_per_worker * 2 > budget:
        executor = "thread"
        workers = max(1, min(cpu, max(workers, 2)))

    return {
        "sd_workers": workers,
        "sd_chunk": chunk,
        "sd_executor": executor,
        "sd_gb_per_worker": round(gb_per_worker, 3),
        "sd_ram_budget_gb": round(budget, 2),
    }


def _sdf_resolution(
    n_faces: int,
    complexity: float,
    tier: str,
    *,
    vram_eff_gb: float | None = None,
    override: int | None = None,
) -> int:
    max_res = _sdf_max_resolution(tier, vram_eff_gb)
    if override is not None and override > 0:
        return _snap_sdf_resolution(override, max_res)

    base = {
        "cpu": 64,
        "low": 96,
        "mid": 128,
        "high": 160,
        "xl": 192,
        "xxl": 256,
    }.get(tier, 128)
    if complexity >= 0.35:
        base = int(base * 1.25)
    if complexity >= 0.5:
        base = int(base * 1.15)
    if n_faces >= 300_000:
        base = int(base * 1.1)
    if n_faces >= 1_000_000:
        base = int(base * 1.08)
    return _snap_sdf_resolution(base, max_res)


def build_sdf_plan(
    n_faces: int,
    n_vertices: int,
    *,
    complexity: float = 0.5,
    device: str | None = None,
    sdf_res: int | None = None,
    query_face_limit: int | None = None,
    no_sdf_simplify: bool = False,
    runtime: RuntimeEnv | None = None,
) -> dict:
    """Mesh size + GPU/RAM aware Neural SDF training defaults."""
    rt = runtime or detect_runtime(device)
    tier = rt.gpu_tier
    max_res = _sdf_max_resolution(tier, rt.vram_effective_gb)
    resolved_res = _sdf_resolution(
        n_faces, complexity, tier, vram_eff_gb=rt.vram_effective_gb, override=sdf_res
    )
    num_surf, num_uni = _sdf_pool_sizes(n_faces, complexity, tier, ram_gb=rt.ram_gb)
    pool_size = num_surf + num_uni
    batch_size = _sdf_batch_size(pool_size, tier)
    steps = _sdf_train_steps(pool_size, batch_size, complexity, tier)
    resolved_query = _sdf_query_face_limit(
        n_faces,
        resolved_res,
        complexity,
        tier,
        ram_gb=rt.ram_gb,
        override=query_face_limit,
        no_simplify=no_sdf_simplify,
    )
    log_interval = max(100, steps // 8)
    query_verts = _estimate_query_vertices(n_vertices, n_faces, resolved_query)
    sd = compute_sdf_sd_settings(
        resolved_query,
        query_verts,
        pool_size,
        ram_gb=rt.ram_gb,
        n_input_faces=n_faces,
    )
    query_before_ram = resolved_query
    resolved_query, query_ram_from = _cap_query_faces_for_sd_ram(
        n_faces,
        n_vertices,
        resolved_query,
        ram_gb=rt.ram_gb,
        sd_workers=int(sd["sd_workers"]),
        sd_gb_per_worker=float(sd["sd_gb_per_worker"]),
    )
    if resolved_query != query_before_ram:
        query_verts = _estimate_query_vertices(n_vertices, n_faces, resolved_query)
        sd = compute_sdf_sd_settings(
            resolved_query,
            query_verts,
            pool_size,
            ram_gb=rt.ram_gb,
            n_input_faces=n_faces,
        )
    hard_max = _sdf_query_face_hard_max()
    if not no_sdf_simplify and resolved_query > hard_max:
        resolved_query = hard_max
        query_verts = _estimate_query_vertices(n_vertices, n_faces, resolved_query)
        sd = compute_sdf_sd_settings(
            resolved_query,
            query_verts,
            pool_size,
            ram_gb=rt.ram_gb,
            n_input_faces=n_faces,
        )
    return {
        "runtime": asdict(rt),
        "n_faces": n_faces,
        "n_vertices": n_vertices,
        "complexity": round(complexity, 4),
        "sdf_res": resolved_res,
        "num_surf": num_surf,
        "num_uni": num_uni,
        "pool_size": pool_size,
        "steps": steps,
        "batch_size": batch_size,
        "query_face_limit": resolved_query,
        "query_face_limit_before_ram": query_ram_from or resolved_query,
        "query_keep_ratio": round(resolved_query / max(n_faces, 1), 4),
        "no_sdf_simplify": no_sdf_simplify,
        "lr": 0.005,
        "log_interval": log_interval,
        "sdf_res_max": max_res,
        "sdf_res_mode": "manual" if sdf_res and sdf_res > 0 else "auto",
        **sd,
    }


def format_sdf_plan_line(plan: dict) -> str:
    rt = plan.get("runtime", {})
    tier = rt.get("gpu_tier", "?")
    gpu = rt.get("gpu_name") or "CPU"
    vram = rt.get("vram_effective_gb")
    vram_s = f"{vram:.1f}GB" if vram else "CPU"
    colab = "colab " if rt.get("is_colab") else ""
    line = (
        f"sdf-auto: {colab}tier={tier} gpu={gpu} vram={vram_s} "
        f"faces={plan['n_faces']:,} complexity={plan['complexity']:.3f} "
        f"res={plan['sdf_res']}^3 pool={plan['pool_size']:,} "
        f"steps={plan['steps']} batch={plan['batch_size']:,} "
        f"query_faces={plan['query_face_limit']:,} "
        f"({plan.get('query_keep_ratio', 0) * 100:.1f}% of input) "
        f"sd_workers={plan.get('sd_workers', '?')} "
        f"sd_chunk={plan.get('sd_chunk', '?'):,} "
        f"sd_exec={plan.get('sd_executor', '?')}"
    )
    if plan.get("query_face_limit_before_ram", plan["query_face_limit"]) != plan[
        "query_face_limit"
    ]:
        before = plan["query_face_limit_before_ram"]
        line += f" query_ram_cap={before:,}->{plan['query_face_limit']:,}"
    return line

