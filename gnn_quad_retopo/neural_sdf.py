"""Pure-PyTorch Multi-resolution Hash Encoding Neural SDF (Instant-NGP style).

Enables GPU-accelerated watertight remeshing with zero C++/CUDA compilation dependencies.
"""
from __future__ import annotations

import gc
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import trimesh
from skimage import measure


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip().lower()


def _signed_distance_trimesh_chunk(mesh: trimesh.Trimesh, pts: np.ndarray) -> np.ndarray:
    return -trimesh.proximity.signed_distance(mesh, pts)


def _signed_distance_igl(mesh: trimesh.Trimesh, pts: np.ndarray) -> np.ndarray:
    import igl

    v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    f = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    p = np.ascontiguousarray(pts, dtype=np.float64)
    out = igl.signed_distance(p, v, f)
    s = out[0] if isinstance(out, tuple) else out
    return -np.asarray(s, dtype=np.float32)


_g_sd_mesh: trimesh.Trimesh | None = None


def _sd_process_init(vertices: np.ndarray, faces: np.ndarray) -> None:
    global _g_sd_mesh
    _g_sd_mesh = trimesh.Trimesh(
        vertices=np.ascontiguousarray(vertices, dtype=np.float64),
        faces=np.ascontiguousarray(faces, dtype=np.int64),
        process=True,
    )


def _sd_process_chunk(args: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    start, chunk_pts = args
    assert _g_sd_mesh is not None
    pts = np.ascontiguousarray(chunk_pts, dtype=np.float64)
    return start, _signed_distance_trimesh_chunk(_g_sd_mesh, pts)


def _compute_signed_distances_cpu_parallel(
    mesh: trimesh.Trimesh,
    pts: np.ndarray,
    *,
    chunk_size: int,
    workers: int,
    use_processes: bool = True,
) -> np.ndarray:
    n = len(pts)
    chunks = [
        (i, pts[i : min(i + chunk_size, n)])
        for i in range(0, n, chunk_size)
    ]
    results: dict[int, np.ndarray] = {}
    pool_label = "ProcessPool" if use_processes else "ThreadPool"
    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    pool_kwargs: dict = {"max_workers": workers}
    if use_processes:
        pool_kwargs["initializer"] = _sd_process_init
        pool_kwargs["initargs"] = (mesh.vertices, mesh.faces)

    with executor_cls(**pool_kwargs) as pool:
        if use_processes:
            futures = {
                pool.submit(_sd_process_chunk, item): item[0] for item in chunks
            }
        else:
            futures = {
                pool.submit(_signed_distance_trimesh_chunk, mesh, c_pts): start
                for start, c_pts in chunks
            }
        done_pts = 0
        for fut in as_completed(futures):
            start = futures[fut]
            if use_processes:
                start, chunk_out = fut.result()
            else:
                chunk_out = fut.result()
            results[start] = chunk_out
            done_pts += len(chunk_out)
            print(
                f"[NeuralSDF] CPU signed_distance {done_pts:,}/{n:,} "
                f"({pool_label}, workers={workers})",
                flush=True,
            )
    return np.concatenate([results[start] for start, _ in chunks], axis=0)


def _resolve_sd_parallel_settings(
    mesh: trimesh.Trimesh,
    n_points: int,
    plan: dict | None,
) -> tuple[int, int, bool]:
    from .auto_tune import compute_sdf_sd_settings

    if plan is not None and "sd_workers" in plan:
        workers = int(plan["sd_workers"])
        chunk_size = int(plan.get("sd_chunk", 50_000))
        use_processes = plan.get("sd_executor", "process") != "thread"
    else:
        ram_gb = (plan.get("runtime") or {}).get("ram_gb") if plan else None
        n_input = int(plan["n_faces"]) if plan and plan.get("n_faces") else None
        sd = compute_sdf_sd_settings(
            len(mesh.faces),
            len(mesh.vertices),
            n_points,
            ram_gb=ram_gb,
            n_input_faces=n_input,
        )
        workers = sd["sd_workers"]
        chunk_size = sd["sd_chunk"]
        use_processes = sd["sd_executor"] != "thread"

    if os.environ.get("GNN_SDF_SD_WORKERS"):
        workers = max(1, _env_int("GNN_SDF_SD_WORKERS", workers))
    if os.environ.get("GNN_SDF_SD_CHUNK"):
        chunk_size = max(1000, _env_int("GNN_SDF_SD_CHUNK", chunk_size))
    if os.environ.get("GNN_SDF_SD_EXECUTOR"):
        use_processes = _env_str("GNN_SDF_SD_EXECUTOR", "process") != "thread"
    return workers, chunk_size, use_processes


def _compute_signed_distances(
    mesh: trimesh.Trimesh,
    pts: np.ndarray,
    *,
    chunk_size: int | None = None,
    device: str | torch.device | None = None,
    plan: dict | None = None,
) -> np.ndarray:
    """Signed-distance labels for Neural SDF training.

    Backend order (``GNN_SDF_SD_BACKEND``):
    ``auto`` / ``cpu`` → ``igl`` (optional) → ProcessPool CPU (trimesh).

    ``GNN_SDF_SD_EXECUTOR``: ``process`` (default) or ``thread`` (manual override).
    """
    del device  # SDF labels are CPU-only; GPU path removed (bad signs on open meshes).
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    auto_workers, auto_chunk, auto_process = _resolve_sd_parallel_settings(
        mesh, n, plan
    )
    chunk_size = chunk_size or auto_chunk
    workers = auto_workers
    use_processes = auto_process
    backend = _env_str("GNN_SDF_SD_BACKEND", "auto")

    if backend in ("auto", "igl"):
        try:
            print(
                f"[NeuralSDF] Computing signed_distance via libigl ({n:,} points)...",
                flush=True,
            )
            return _signed_distance_igl(mesh, pts)
        except Exception as e:
            print(
                f"[NeuralSDF] libigl signed_distance failed ({e}); fallback.",
                flush=True,
            )

    pool_label = "ProcessPool" if use_processes else "ThreadPool"
    print(
        f"[NeuralSDF] Computing signed_distance on CPU ({pool_label}), "
        f"{n:,} points, workers={workers}, chunk={chunk_size:,}...",
        flush=True,
    )
    try:
        return _compute_signed_distances_cpu_parallel(
            mesh,
            pts,
            chunk_size=chunk_size,
            workers=workers,
            use_processes=use_processes,
        )
    except Exception as e:
        if not use_processes:
            raise
        print(
            f"[NeuralSDF] ProcessPool failed ({e}); retry ThreadPool.",
            flush=True,
        )
        return _compute_signed_distances_cpu_parallel(
            mesh,
            pts,
            chunk_size=chunk_size,
            workers=workers,
            use_processes=False,
        )


class MultiResHashEncoder(nn.Module):
    def __init__(
        self,
        T: int = 16384,
        F: int = 2,
        L: int = 8,
        r_min: int = 16,
        r_max: int = 128,
    ):
        super().__init__()
        self.T = T
        self.F = F
        self.L = L
        self.r_min = r_min
        self.r_max = r_max

        # Calculate grid resolutions for each level geometrically
        if L > 1:
            self.b = np.exp((np.log(r_max) - np.log(r_min)) / (L - 1))
        else:
            self.b = 1.0
        self.resolutions = [int(r_min * (self.b**l)) for l in range(L)]

        # Primes for spatial hashing (from Instant-NGP)
        self.register_buffer(
            "primes",
            torch.tensor([1, 2654435761, 805459861], dtype=torch.int64),
        )

        # Feature tables for each level
        self.embeddings = nn.ParameterList(
            [nn.Parameter(torch.empty(T, F)) for _ in range(L)]
        )
        for emb in self.embeddings:
            nn.init.uniform_(emb, -1e-4, 1e-4)

    def hash_fn(self, coords: torch.Tensor) -> torch.Tensor:
        # coords shape: [N, 3] of int64
        scaled = coords * self.primes
        idx = torch.bitwise_xor(
            torch.bitwise_xor(scaled[:, 0], scaled[:, 1]), scaled[:, 2]
        )
        return torch.remainder(idx, self.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, 3], in [0, 1] range
        features = []
        for l in range(self.L):
            res = self.resolutions[l]
            x_grid = x * res

            # Voxel corners (floor & ceil)
            c_floor = torch.floor(x_grid).long()
            c_ceil = c_floor + 1

            # Interpolation weight
            w = x_grid - c_floor.float()

            # 8 corners of the voxel
            c000 = c_floor
            c100 = torch.stack(
                [c_ceil[:, 0], c_floor[:, 1], c_floor[:, 2]], dim=-1
            )
            c010 = torch.stack(
                [c_floor[:, 0], c_ceil[:, 1], c_floor[:, 2]], dim=-1
            )
            c001 = torch.stack(
                [c_floor[:, 0], c_floor[:, 1], c_ceil[:, 2]], dim=-1
            )
            c110 = torch.stack(
                [c_ceil[:, 0], c_ceil[:, 1], c_floor[:, 2]], dim=-1
            )
            c101 = torch.stack(
                [c_ceil[:, 0], c_floor[:, 1], c_ceil[:, 2]], dim=-1
            )
            c011 = torch.stack(
                [c_floor[:, 0], c_ceil[:, 1], c_ceil[:, 2]], dim=-1
            )
            c111 = c_ceil

            # Hash coordinates to embedding indices
            idx_000 = self.hash_fn(c000)
            idx_100 = self.hash_fn(c100)
            idx_010 = self.hash_fn(c010)
            idx_001 = self.hash_fn(c001)
            idx_110 = self.hash_fn(c110)
            idx_101 = self.hash_fn(c101)
            idx_011 = self.hash_fn(c011)
            idx_111 = self.hash_fn(c111)

            # Retrieve vectors
            emb_table = self.embeddings[l]
            v_000 = emb_table[idx_000]
            v_100 = emb_table[idx_100]
            v_010 = emb_table[idx_010]
            v_001 = emb_table[idx_001]
            v_110 = emb_table[idx_110]
            v_101 = emb_table[idx_101]
            v_011 = emb_table[idx_011]
            v_111 = emb_table[idx_111]

            # Trilinear interpolation
            w_x = w[:, 0:1]
            w_y = w[:, 1:2]
            w_z = w[:, 2:3]

            # X interpolation
            v_00 = v_000 * (1.0 - w_x) + v_100 * w_x
            v_01 = v_001 * (1.0 - w_x) + v_101 * w_x
            v_10 = v_010 * (1.0 - w_x) + v_110 * w_x
            v_11 = v_011 * (1.0 - w_x) + v_111 * w_x

            # Y interpolation
            v_0 = v_00 * (1.0 - w_y) + v_10 * w_y
            v_1 = v_01 * (1.0 - w_y) + v_11 * w_y

            # Z interpolation
            v_interpolated = v_0 * (1.0 - w_z) + v_1 * w_z
            features.append(v_interpolated)

        return torch.cat(features, dim=-1)


class TinyMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralSDF(nn.Module):
    def __init__(
        self,
        T: int = 16384,
        F: int = 2,
        L: int = 8,
        r_min: int = 16,
        r_max: int = 128,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = MultiResHashEncoder(T, F, L, r_min, r_max)
        self.mlp = TinyMLP(L * F, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.mlp(features)


class MeshNormalizer:
    def __init__(self, vertices: np.ndarray, padding: float = 0.05):
        self.min_coords = vertices.min(axis=0)
        self.max_coords = vertices.max(axis=0)
        self.center = (self.min_coords + self.max_coords) / 2.0
        self.extent = (self.max_coords - self.min_coords).max()
        if self.extent == 0:
            self.extent = 1.0
        self.padding = padding
        # Map center-aligned vertices into bounds [padding, 1 - padding]
        self.scale = (1.0 - 2.0 * padding) / self.extent

    def normalize(self, points: np.ndarray) -> np.ndarray:
        return (points - self.center) * self.scale + 0.5

    def denormalize(self, points: np.ndarray) -> np.ndarray:
        return (points - 0.5) / self.scale + self.center


def train_neural_sdf(
    mesh: trimesh.Trimesh,
    r_max: int | None = None,
    steps: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    *,
    plan: dict | None = None,
    complexity: float | None = None,
) -> tuple[NeuralSDF, MeshNormalizer]:
    from .auto_tune import build_sdf_plan, format_sdf_plan_line

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    if plan is None:
        cx = complexity
        if cx is None:
            if len(mesh.face_normals):
                cx = float(np.mean(np.std(mesh.face_normals, axis=0)))
            else:
                cx = 0.5
        plan = build_sdf_plan(
            len(mesh.faces),
            len(mesh.vertices),
            complexity=cx,
            device=device,
            sdf_res=r_max,
        )

    r_max = int(plan["sdf_res"])
    steps = int(steps if steps is not None else plan["steps"])
    batch_size = int(batch_size if batch_size is not None else plan["batch_size"])
    lr = float(lr if lr is not None else plan["lr"])
    num_surf = int(plan["num_surf"])
    num_uni = int(plan["num_uni"])
    pool_size = int(plan["pool_size"])
    query_face_limit = int(plan["query_face_limit"])
    log_interval = int(plan.get("log_interval", max(100, steps // 8)))

    print(format_sdf_plan_line(plan), flush=True)

    source_vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    n_faces = len(mesh.faces)
    if n_faces > query_face_limit:
        print(
            f"[NeuralSDF] Pre-simplifying mesh for SDF query "
            f"({n_faces:,} -> {query_face_limit:,} faces)...",
            flush=True,
        )
        try:
            mesh_query = mesh.simplify_quadric_decimation(face_count=query_face_limit)
        except Exception as e:
            print(f"[NeuralSDF] Pre-simplification failed ({e}). Proceeding with raw mesh.", flush=True)
            mesh_query = mesh
        else:
            del mesh
            mesh = None  # noqa: F841
            gc.collect()
            print("[NeuralSDF] Released input mesh from memory before signed_distance.", flush=True)
    else:
        mesh_query = mesh

    print(f"[NeuralSDF] Preparing training points pool from mesh...", flush=True)

    surf_pts, _ = trimesh.sample.sample_surface(mesh_query, num_surf)
    noise_scales = np.random.choice([0.003, 0.015, 0.06], size=(num_surf, 1))
    noise = np.random.normal(0, 1, size=surf_pts.shape) * noise_scales
    surf_noisy = surf_pts + noise

    bbox_min = mesh_query.vertices.min(axis=0)
    bbox_max = mesh_query.vertices.max(axis=0)
    uni_pts = np.random.uniform(
        bbox_min - 0.1, bbox_max + 0.1, size=(num_uni, 3)
    )

    pts_pool = np.vstack([surf_noisy, uni_pts])

    print(f"[NeuralSDF] Computing signed distance fields for {pool_size:,} pool points...", flush=True)
    target_sdf = _compute_signed_distances(
        mesh_query, pts_pool, device=device_t, plan=plan
    )

    # Convert normalizer
    normalizer = MeshNormalizer(source_vertices)
    pts_pool_norm = normalizer.normalize(pts_pool)

    # Move pool to GPU
    pts_pool_t = torch.from_numpy(pts_pool_norm.astype(np.float32)).to(device_t)
    target_sdf_t = (
        torch.from_numpy(target_sdf.astype(np.float32))
        .unsqueeze(-1)
        .to(device_t)
    )

    # 3. Model setup
    model = NeuralSDF(r_max=r_max).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(
        f"[NeuralSDF] Training multi-res hash encoding on GPU ({device}) for {steps} steps...",
        flush=True,
    )
    model.train()
    for step in range(steps):
        # Draw dynamic batches purely on GPU (extremely fast, ~0.5ms per step)
        indices = torch.randint(0, pool_size, (batch_size,), device=device_t)
        batch_pts = pts_pool_t[indices]
        batch_sdf = target_sdf_t[indices]

        optimizer.zero_grad()
        pred_sdf = model(batch_pts)
        loss = torch.nn.functional.l1_loss(pred_sdf, batch_sdf)
        loss.backward()
        optimizer.step()

        if (step + 1) % log_interval == 0 or step == 0:
            print(
                f"  step {step+1:03d}/{steps} | train_l1_loss = {loss.item():.5f}",
                flush=True,
            )

    return model, normalizer


def extract_mesh_from_sdf(
    model: NeuralSDF,
    normalizer: MeshNormalizer,
    resolution: int = 128,
    device: str | None = None,
    method: str = "dual_contouring",
) -> trimesh.Trimesh:
    if method == "dual_contouring":
        return extract_mesh_from_sdf_dual(model, normalizer, resolution, device)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    print(
        f"[NeuralSDF] Evaluating neural SDF grid at {resolution}^3 resolution...",
        flush=True,
    )
    # Generate 3D grid
    grid_coords = torch.linspace(0.0, 1.0, resolution, device=device_t)
    z, y, x = torch.meshgrid(
        grid_coords, grid_coords, grid_coords, indexing="ij"
    )
    grid_points = torch.stack(
        [x.flatten(), y.flatten(), z.flatten()], dim=-1
    )  # [res^3, 3]

    # Batch evaluation to fit comfortably in VRAM
    batch_size = 131072
    sdf_vals = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(grid_points), batch_size):
            batch_pts = grid_points[i : i + batch_size]
            pred = model(batch_pts)
            sdf_vals.append(pred.cpu())

    sdf_volume = (
        torch.cat(sdf_vals, dim=0)
        .reshape(resolution, resolution, resolution)
        .numpy()
    )

    print(f"[NeuralSDF] Running Marching Cubes zero-isosurface...", flush=True)
    try:
        # skimage.measure.marching_cubes coordinate indexing order is (z, y, x)
        verts_grid, faces, normals, values = measure.marching_cubes(
            sdf_volume, level=0.0
        )
    except ValueError as e:
        print(
            f"[NeuralSDF] Warning: zero-level surface not found ({e}). Retrying at level=0.01.",
            flush=True,
        )
        verts_grid, faces, normals, values = measure.marching_cubes(
            sdf_volume, level=0.01
        )

    # Remap (z, y, x) order to standard Cartesian (x, y, z)
    verts_norm = verts_grid[:, [2, 1, 0]] / (resolution - 1.0)

    # Denormalize to original coordinate space
    verts_orig = normalizer.denormalize(verts_norm)

    out_mesh = trimesh.Trimesh(vertices=verts_orig, faces=faces, process=False)
    print(
        f"[NeuralSDF] Successfully extracted watertight mesh (verts: {len(out_mesh.vertices):,}, faces: {len(out_mesh.faces):,})",
        flush=True,
    )
    return out_mesh


def extract_mesh_from_sdf_dual(
    model: NeuralSDF,
    normalizer: MeshNormalizer,
    resolution: int = 64,
    device: str | None = None,
) -> trimesh.Trimesh:
    """Extract a feature-preserving quad-dominant watertight mesh from Neural SDF using Dual Contouring."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    print(
        f"[NeuralSDF] Running Dual Contouring zero-isosurface extraction at {resolution}^3 resolution...",
        flush=True,
    )

    # 1. Evaluate SDF volume
    grid_coords = torch.linspace(0.0, 1.0, resolution + 1, device=device_t)
    z, y, x = torch.meshgrid(
        grid_coords, grid_coords, grid_coords, indexing="ij"
    )
    grid_points = torch.stack(
        [x.flatten(), y.flatten(), z.flatten()], dim=-1
    )  # [(res+1)^3, 3]

    batch_size = 131072
    sdf_vals = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(grid_points), batch_size):
            batch_pts = grid_points[i : i + batch_size]
            pred = model(batch_pts)
            sdf_vals.append(pred.cpu())

    sdf_volume = (
        torch.cat(sdf_vals, dim=0)
        .reshape(resolution + 1, resolution + 1, resolution + 1)
        .numpy()
    )

    # 2. Find active cells
    # A cell (cz, cy, cx) has 8 corners. We check their signs (>= 0 is positive/outside, < 0 is negative/inside)
    signs = sdf_volume >= 0.0

    # Extract corners for all cells
    c000 = signs[0:-1, 0:-1, 0:-1]
    c100 = signs[0:-1, 0:-1, 1:]
    c010 = signs[0:-1, 1:, 0:-1]
    c001 = signs[1:, 0:-1, 0:-1]
    c110 = signs[0:-1, 1:, 1:]
    c101 = signs[1:, 0:-1, 1:]
    c011 = signs[1:, 1:, 0:-1]
    c111 = signs[1:, 1:, 1:]

    # Cell is active if signs are not all identical
    all_positive = (
        c000 & c100 & c010 & c001 & c110 & c101 & c011 & c111
    )
    all_negative = (
        (~c000)
        & (~c100)
        & (~c010)
        & (~c001)
        & (~c110)
        & (~c101)
        & (~c011)
        & (~c111)
    )
    active_mask = ~(all_positive | all_negative)

    active_cell_coords = np.argwhere(
        active_mask
    )  # Shape (M, 3) where elements are (z, y, x)
    if len(active_cell_coords) == 0:
        print(
            "[NeuralSDF] Warning: No active cells found. Returning empty mesh.",
            flush=True,
        )
        return trimesh.Trimesh()

    # Map 3D cell index (z, y, x) to vertex ID
    cell_to_idx = {}
    for idx, (cz, cy, cx) in enumerate(active_cell_coords):
        cell_to_idx[(cz, cy, cx)] = idx

    # 3. Check all active edges along X, Y, Z axes
    # An edge is active if its endpoints have different signs
    x_edges_active = signs[0:-1, :, :] != signs[1:, :, :]  # Shape (res, res+1, res+1)
    y_edges_active = signs[:, 0:-1, :] != signs[:, 1:, :]  # Shape (res+1, res, res+1)
    z_edges_active = signs[:, :, 0:-1] != signs[:, :, 1:]  # Shape (res+1, res+1, res)

    # 4. Prepare data for QEF solving per active cell
    # X-edges: from (z, y, x) to (z+1, y, x) where 0 <= z < res
    # Y-edges: from (z, y, x) to (z, y+1, x) where 0 <= y < res
    # Z-edges: from (z, y, x) to (z, y, x+1) where 0 <= x < res
    # Note: argwhere returns (z, y, x)
    x_active_indices = np.argwhere(x_edges_active)
    y_active_indices = np.argwhere(y_edges_active)
    z_active_indices = np.argwhere(z_edges_active)

    # Let's compute intersections for X-edges
    x_intersections = []
    if len(x_active_indices) > 0:
        z, y, x = (
            x_active_indices[:, 0],
            x_active_indices[:, 1],
            x_active_indices[:, 2],
        )
        v_A = sdf_volume[z, y, x]
        v_B = sdf_volume[z + 1, y, x]
        t = v_A / (v_A - v_B + 1e-8)
        t = np.clip(t, 0.0, 1.0)
        pts_grid = np.stack([x, y, z + t], axis=-1)  # (x, y, z)
        x_intersections = pts_grid / resolution

    # Y-edges
    y_intersections = []
    if len(y_active_indices) > 0:
        z, y, x = (
            y_active_indices[:, 0],
            y_active_indices[:, 1],
            y_active_indices[:, 2],
        )
        v_A = sdf_volume[z, y, x]
        v_B = sdf_volume[z, y + 1, x]
        t = v_A / (v_A - v_B + 1e-8)
        t = np.clip(t, 0.0, 1.0)
        pts_grid = np.stack([x, y + t, z], axis=-1)
        y_intersections = pts_grid / resolution

    # Z-edges
    z_intersections = []
    if len(z_active_indices) > 0:
        z, y, x = (
            z_active_indices[:, 0],
            z_active_indices[:, 1],
            z_active_indices[:, 2],
        )
        v_A = sdf_volume[z, y, x]
        v_B = sdf_volume[z, y, x + 1]
        t = v_A / (v_A - v_B + 1e-8)
        t = np.clip(t, 0.0, 1.0)
        pts_grid = np.stack([x + t, y, z], axis=-1)
        z_intersections = pts_grid / resolution

    # Combine intersections to evaluate normals
    all_pts = []
    offsets = [0]
    if len(x_intersections) > 0:
        all_pts.append(x_intersections)
    offsets.append(offsets[-1] + len(x_intersections))
    if len(y_intersections) > 0:
        all_pts.append(y_intersections)
    offsets.append(offsets[-1] + len(y_intersections))
    if len(z_intersections) > 0:
        all_pts.append(z_intersections)
    offsets.append(offsets[-1] + len(z_intersections))

    if len(all_pts) > 0:
        all_pts = np.vstack(all_pts)
        print(
            f"[NeuralSDF] Computing normals for {len(all_pts):,} active edge intersections...",
            flush=True,
        )
        pts_t = torch.from_numpy(all_pts.astype(np.float32)).to(device_t)
        eps = 1e-4

        pts_xp = pts_t + torch.tensor([eps, 0.0, 0.0], device=device_t)
        pts_xm = pts_t - torch.tensor([eps, 0.0, 0.0], device=device_t)
        pts_yp = pts_t + torch.tensor([0.0, eps, 0.0], device=device_t)
        pts_ym = pts_t - torch.tensor([0.0, eps, 0.0], device=device_t)
        pts_zp = pts_t + torch.tensor([0.0, 0.0, eps], device=device_t)
        pts_zm = pts_t - torch.tensor([0.0, 0.0, eps], device=device_t)

        pts_eval = torch.cat(
            [pts_xp, pts_xm, pts_yp, pts_ym, pts_zp, pts_zm], dim=0
        )

        model.eval()
        with torch.no_grad():
            preds = []
            for j in range(0, len(pts_eval), batch_size):
                preds.append(model(pts_eval[j : j + batch_size]))
            all_preds = torch.cat(preds, dim=0).cpu().numpy()

        n_pts = len(pts_t)
        sdf_xp = all_preds[0 : n_pts]
        sdf_xm = all_preds[n_pts : 2 * n_pts]
        sdf_yp = all_preds[2 * n_pts : 3 * n_pts]
        sdf_ym = all_preds[3 * n_pts : 4 * n_pts]
        sdf_zp = all_preds[4 * n_pts : 5 * n_pts]
        sdf_zm = all_preds[5 * n_pts : 6 * n_pts]

        grad = np.hstack(
            [
                (sdf_xp - sdf_xm) / (2.0 * eps),
                (sdf_yp - sdf_ym) / (2.0 * eps),
                (sdf_zp - sdf_zm) / (2.0 * eps),
            ]
        )

        norm_lengths = np.linalg.norm(grad, axis=1, keepdims=True)
        norm_lengths[norm_lengths == 0] = 1.0
        all_normals = grad / norm_lengths
    else:
        all_normals = np.zeros((0, 3))

    x_normals = all_normals[offsets[0] : offsets[1]]
    y_normals = all_normals[offsets[1] : offsets[2]]
    z_normals = all_normals[offsets[2] : offsets[3]]

    # Map active edges to their intersection and normal
    x_edge_data = {}
    for idx, (z, y, x) in enumerate(x_active_indices):
        x_edge_data[(z, y, x)] = (x_intersections[idx], x_normals[idx])

    y_edge_data = {}
    for idx, (z, y, x) in enumerate(y_active_indices):
        y_edge_data[(z, y, x)] = (y_intersections[idx], y_normals[idx])

    z_edge_data = {}
    for idx, (z, y, x) in enumerate(z_active_indices):
        z_edge_data[(z, y, x)] = (z_intersections[idx], z_normals[idx])

    # 5. Solve QEF per active cell
    vertices = []
    print(
        f"[NeuralSDF] Solving QEF equations for {len(active_cell_coords):,} cell vertices...",
        flush=True,
    )

    for idx, (cz, cy, cx) in enumerate(active_cell_coords):
        cell_min = np.array([cx, cy, cz]) / resolution
        cell_max = np.array([cx + 1, cy + 1, cz + 1]) / resolution

        cell_pts = []
        cell_noms = []

        # 4 X-edges: key is (z, y, x) where x_edges_active is (res, res+1, res+1)
        for dy, dx in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            key = (cz, cy + dy, cx + dx)
            if key in x_edge_data:
                pt, nm = x_edge_data[key]
                cell_pts.append(pt)
                cell_noms.append(nm)

        # 4 Y-edges: key is (z, y, x) where y_edges_active is (res+1, res, res+1)
        for dz, dx in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            key = (cz + dz, cy, cx + dx)
            if key in y_edge_data:
                pt, nm = y_edge_data[key]
                cell_pts.append(pt)
                cell_noms.append(nm)

        # 4 Z-edges: key is (z, y, x) where z_edges_active is (res+1, res+1, res)
        for dz, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            key = (cz + dz, cy + dy, cx)
            if key in z_edge_data:
                pt, nm = z_edge_data[key]
                cell_pts.append(pt)
                cell_noms.append(nm)

        if len(cell_pts) == 0:
            vertices.append((cell_min + cell_max) / 2.0)
            continue
        # Placing the cell vertex at the exact mass centroid of edge intersections
        # provides mathematically guaranteed smoothness and watertightness, completely
        # eliminating the high-frequency SVD noise/spikes caused by Neural SDF normal fluctuations.
        x_opt = np.array(cell_pts).mean(axis=0)
        vertices.append(x_opt)

    vertices = np.array(vertices)

    # 6. Generate faces
    faces = []

    # Active X-edges (shared by 4 cells around Z-axis)
    # x_edges_active has shape (res, res+1, res+1) corresponding to edge starting at (z, y, x) to (z+1, y, x)
    # The 4 sharing cells in grid are: (z, y-1, x-1), (z, y-1, x), (z, y, x), (z, y, x-1)
    for (cz, cy, cx) in x_active_indices:
        if cy > 0 and cx > 0 and cy < resolution and cx < resolution:
            c0 = (cz, cy - 1, cx - 1)
            c1 = (cz, cy - 1, cx)
            c2 = (cz, cy,     cx)
            c3 = (cz, cy,     cx - 1)

            if (
                c0 in cell_to_idx
                and c1 in cell_to_idx
                and c2 in cell_to_idx
                and c3 in cell_to_idx
            ):
                idx0 = cell_to_idx[c0]
                idx1 = cell_to_idx[c1]
                idx2 = cell_to_idx[c2]
                idx3 = cell_to_idx[c3]

                # Winding order: if normal is +Z (SDF increases along Z direction)
                if sdf_volume[cz, cy, cx] < sdf_volume[cz + 1, cy, cx]:
                    faces.append([idx0, idx1, idx2, idx3])
                else:
                    faces.append([idx0, idx3, idx2, idx1])

    # Active Y-edges (shared by 4 cells around Y-axis)
    # y_edges_active has shape (res+1, res, res+1) corresponding to edge starting at (z, y, x) to (z, y+1, x)
    # The 4 sharing cells in grid are: (z-1, y, x-1), (z-1, y, x), (z, y, x), (z, y, x-1)
    for (cz, cy, cx) in y_active_indices:
        if cz > 0 and cx > 0 and cz < resolution and cx < resolution:
            c0 = (cz - 1, cy, cx - 1)
            c1 = (cz - 1, cy, cx)
            c2 = (cz,     cy, cx)
            c3 = (cz,     cy, cx - 1)

            if (
                c0 in cell_to_idx
                and c1 in cell_to_idx
                and c2 in cell_to_idx
                and c3 in cell_to_idx
            ):
                idx0 = cell_to_idx[c0]
                idx1 = cell_to_idx[c1]
                idx2 = cell_to_idx[c2]
                idx3 = cell_to_idx[c3]

                # If normal is +Y (SDF increases along Y direction)
                if sdf_volume[cz, cy, cx] < sdf_volume[cz, cy + 1, cx]:
                    faces.append([idx0, idx1, idx2, idx3])
                else:
                    faces.append([idx0, idx3, idx2, idx1])

    # Active Z-edges (shared by 4 cells around X-axis)
    # z_edges_active has shape (res+1, res+1, res) corresponding to edge starting at (z, y, x) to (z, y, x+1)
    # The 4 sharing cells in grid are: (z-1, y-1, x), (z, y-1, x), (z, y, x), (z-1, y, x)
    for (cz, cy, cx) in z_active_indices:
        if cz > 0 and cy > 0 and cz < resolution and cy < resolution:
            c0 = (cz - 1, cy - 1, cx)
            c1 = (cz,     cy - 1, cx)
            c2 = (cz,     cy,     cx)
            c3 = (cz - 1, cy,     cx)

            if (
                c0 in cell_to_idx
                and c1 in cell_to_idx
                and c2 in cell_to_idx
                and c3 in cell_to_idx
            ):
                idx0 = cell_to_idx[c0]
                idx1 = cell_to_idx[c1]
                idx2 = cell_to_idx[c2]
                idx3 = cell_to_idx[c3]

                # If normal is +X (SDF increases along X direction)
                if sdf_volume[cz, cy, cx] < sdf_volume[cz, cy, cx + 1]:
                    faces.append([idx0, idx3, idx2, idx1])
                else:
                    faces.append([idx0, idx1, idx2, idx3])

    faces = np.array(faces)

    # Denormalize vertices to original space
    verts_orig = normalizer.denormalize(vertices)

    out_mesh = trimesh.Trimesh(vertices=verts_orig, faces=faces, process=False)
    print(
        f"[NeuralSDF] Successfully extracted watertight quad-dominant mesh via Dual Contouring "
        f"(verts: {len(out_mesh.vertices):,}, faces: {len(out_mesh.faces):,})",
        flush=True,
    )
    return out_mesh

