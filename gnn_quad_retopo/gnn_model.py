"""Pure PyTorch GraphSAGE cross-field training (no torch_geometric)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from .cross_field import compute_cross_field
from .debug_log import DebugLog

BASE_IN_DIM = 6
EXTENDED_IN_DIM = 18  # +normal_variance(1)
EXTENDED_IN_DIM_WITH_COLOR = 21  # +vertex RGB when color guide active
EDGE_ATTR_DIM = 5  # dir(3)+dist(1)+dihedral/pi(1)
DEFAULT_W_COLOR_ALIGN = 0.50


def load_features_npz(
    mesh: trimesh.Trimesh,
    feature_dir: Path | None,
    log: DebugLog | None = None,
) -> tuple[dict[str, np.ndarray], bool]:
    """Load features.npz; regenerate on failure. Returns (arrays, has_real_features)."""
    n_v = len(mesh.vertices)
    zeros3 = np.zeros((n_v, 3), dtype=np.float32)
    fallback = {
        "pd1": zeros3.copy(),
        "pd2": zeros3.copy(),
        "k1": np.zeros(n_v, dtype=np.float32),
        "k2": np.zeros(n_v, dtype=np.float32),
        "feature_mask": np.zeros(n_v, dtype=np.float32),
        "feature_tangents": zeros3.copy(),
        "boundary_mask": np.zeros(n_v, dtype=np.float32),
        "boundary_tangents": zeros3.copy(),
        "boundary_steps": np.full(n_v, 999.0, dtype=np.float32),
        "symmetry_pairs": np.arange(n_v, dtype=np.int64),
        "c_sym": np.zeros(n_v, dtype=np.float32),
        "normal_variance": np.zeros(n_v, dtype=np.float32),
        "vertex_colors": np.ones((n_v, 3), dtype=np.float32),
        "has_vertex_colors": False,
    }
    if feature_dir is None:
        return fallback, False
    feat_dir = Path(feature_dir)
    stem_candidates = sorted(feat_dir.glob("*_features.npz"))
    feat_path = stem_candidates[0] if stem_candidates else feat_dir / "features.npz"
    if not feat_path.exists():
        if log:
            log.warn("features.npz missing — using guide-only fallback weights")
        return fallback, False

    def _read():
        features = np.load(feat_path)
        out = {
            "pd1": features["pd1"],
            "pd2": features["pd2"],
            "k1": features["k1"],
            "k2": features["k2"],
            "feature_mask": features["feature_mask"],
            "feature_tangents": features.get("feature_tangents", zeros3),
            "boundary_mask": features["boundary_mask"],
            "boundary_tangents": features["boundary_tangents"],
            "boundary_steps": features.get("boundary_steps", np.full(n_v, 999.0, dtype=np.float32)),
            "symmetry_pairs": features["symmetry_pairs"],
            "c_sym": features["c_sym"] if "c_sym" in features else np.zeros(n_v, dtype=np.float32),
            "normal_variance": features.get("normal_variance", np.zeros(n_v, dtype=np.float32)),
            "vertex_colors": features.get(
                "vertex_colors", np.ones((n_v, 3), dtype=np.float32)
            ),
            "has_vertex_colors": bool(features["has_vertex_colors"])
            if "has_vertex_colors" in features
            else False,
        }
        if len(out["pd1"]) != n_v:
            raise ValueError(f"features.npz vertex count {len(out['pd1'])} != mesh {n_v}")
        if "c_sym" not in features:
            from .mesh_cleaner import compute_symmetry_confidence
            out["c_sym"] = compute_symmetry_confidence(mesh, out["symmetry_pairs"])
        if "normal_variance" not in features:
            from .mesh_cleaner import compute_vertex_normal_variance
            out["normal_variance"] = compute_vertex_normal_variance(mesh)
        return out

    try:
        out = _read()
        if log:
            log.info("features.npz loaded (L0-L3 losses enabled)")
        return out, True
    except Exception as e:
        if log:
            log.warn(f"Failed to load features.npz: {e}")
            log.step("regenerate features.npz on current mesh")
        from .mesh_cleaner import save_features
        save_features(mesh, Path(feature_dir), log=log)
        try:
            out = _read()
            if log:
                log.info("features.npz regenerated")
            return out, True
        except Exception:
            return fallback, False


def build_node_features(
    vertices: np.ndarray,
    normals: np.ndarray,
    feat: dict[str, np.ndarray],
    *,
    extended: bool = True,
    flat_threshold: float = 0.05,
    use_vertex_color_guide: bool = False,
) -> np.ndarray:
    """Build per-vertex node features for GNN."""
    v = np.asarray(vertices, dtype=np.float32)
    n = np.asarray(normals, dtype=np.float32)
    if not extended:
        return np.hstack([v, n]).astype(np.float32)
    pd1 = feat["pd1"].astype(np.float32).copy()
    pd2 = feat["pd2"].astype(np.float32).copy()
    k1, k2 = feat["k1"].astype(np.float32), feat["k2"].astype(np.float32)
    flat = (np.abs(k1 - k2) < flat_threshold).astype(np.float32)
    pd1 *= (1.0 - flat)[:, None]
    pd2 *= (1.0 - flat)[:, None]
    log_k = np.log1p(np.stack([np.abs(k1), np.abs(k2)], axis=1))
    masks = np.stack([feat["feature_mask"], feat["boundary_mask"]], axis=1).astype(np.float32)
    c_sym = feat["c_sym"].astype(np.float32).reshape(-1, 1)
    nvar = feat["normal_variance"].astype(np.float32).reshape(-1, 1)
    parts = [v, n, pd1, pd2, log_k, masks, c_sym, nvar]
    if use_vertex_color_guide and bool(feat.get("has_vertex_colors", False)):
        vcol = np.asarray(
            feat.get("vertex_colors", np.ones((len(v), 3), dtype=np.float32)),
            dtype=np.float32,
        )
        if len(vcol) == len(v):
            parts.append(vcol)
    return np.hstack(parts).astype(np.float32)



def mesh_to_edge_index(mesh: trimesh.Trimesh) -> torch.Tensor:
    e = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(e) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    ei = torch.from_numpy(e.T.copy())
    return torch.cat([ei, ei.flip(0)], dim=1)


def compute_edge_dihedral_map(mesh: trimesh.Trimesh) -> dict[tuple[int, int], float]:
    """無向エッジ -> 隣接面法線のなす角 [0, pi]（境界は未登録）。"""
    out: dict[tuple[int, int], float] = {}
    if not hasattr(mesh, "face_adjacency") or len(mesh.face_adjacency) == 0:
        return out
    fa = mesh.face_adjacency
    fae = mesh.face_adjacency_edges
    n0 = mesh.face_normals[fa[:, 0]]
    n1 = mesh.face_normals[fa[:, 1]]
    cos_d = np.clip(np.einsum("ij,ij->i", n0, n1), -1.0, 1.0)
    angles = np.arccos(cos_d).astype(np.float32)
    for k, (i, j) in enumerate(fae):
        a, b = int(i), int(j)
        key = (a, b) if a < b else (b, a)
        out[key] = float(angles[k])
    return out


def build_edge_attr(
    vertices: np.ndarray | torch.Tensor,
    edge_index: torch.Tensor,
    *,
    dihedral_map: dict[tuple[int, int], float] | None = None,
) -> torch.Tensor:
    """エッジ特徴: 相対方向(3) + 距離(1) + 二面角/pi(1)。"""
    if edge_index.numel() == 0:
        return torch.zeros((0, EDGE_ATTR_DIM), dtype=torch.float32)
    pos = torch.as_tensor(vertices, dtype=torch.float32)
    src, dst = edge_index[0], edge_index[1]
    diff = pos[dst] - pos[src]
    dist = torch.linalg.norm(diff, dim=1, keepdim=True).clamp_min(1e-6)
    dir_vec = diff / dist
    if dihedral_map:
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        keys = np.stack([src_np, dst_np], axis=1)
        keys.sort(axis=1)
        inv_pi = 1.0 / np.pi
        dia_np = np.array([dihedral_map.get((int(u), int(v)), 0.0) * inv_pi for u, v in keys], dtype=np.float32)
        dia = torch.from_numpy(dia_np).unsqueeze(1).to(pos.device)
    else:
        dia = torch.zeros(len(src), 1, dtype=torch.float32)
    return torch.cat([dir_vec, dist, dia], dim=1)


def build_adjacency_list(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adj
    src, dst = edge_index.cpu().numpy()
    for s, d in zip(src, dst):
        adj[int(d)].append(int(s))
    return adj


class NeighborBatchSampler:
    """PyG NeighborLoader 相当のミニバッチサンプラ（Pure PyTorch）。"""

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        *,
        edge_attr: torch.Tensor | None = None,
        batch_size: int = 4096,
        num_neighbors: tuple[int, ...] = (15, 15, 15),
    ):
        self.num_nodes = num_nodes
        self.batch_size = min(batch_size, num_nodes)
        self.num_neighbors = num_neighbors
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.device = edge_index.device

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Mesh vertex degree (~6) is below the fan-out, so each hop keeps all neighbours.
        dev = self.device
        seeds = torch.randperm(self.num_nodes, device=dev)[: self.batch_size]
        src, dst = self.edge_index[0], self.edge_index[1]
        mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=dev)
        mask[seeds] = True
        for _ in self.num_neighbors:
            nxt = mask.clone()
            nxt[dst[mask[src]]] = True
            mask = nxt
        nodes = torch.nonzero(mask, as_tuple=False).flatten()
        remap = torch.full((self.num_nodes,), -1, dtype=torch.long, device=dev)
        remap[nodes] = torch.arange(nodes.numel(), dtype=torch.long, device=dev)
        emask = mask[src] & mask[dst]
        sub_ei = torch.stack([remap[src[emask]], remap[dst[emask]]], dim=0)
        sub_ea = self.edge_attr[emask] if self.edge_attr is not None else None
        return nodes, sub_ei, remap[seeds], sub_ea


def prepare_graph_data(
    mesh: trimesh.Trimesh,
    *,
    up_axis: str | None = None,
    rotate_axis: str | None = None,
    rotate_deg: float | None = None,
    use_curvature: bool = False,
    feature_dir: Path | None = None,
    extended_features: bool = True,
    vertex_color_guide: bool = True,
    log: DebugLog | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
    from .vertex_color import resolve_vertex_color_guide

    v = np.asarray(mesh.vertices, dtype=np.float32)
    n = np.asarray(mesh.vertex_normals, dtype=np.float32)
    feat, _ = load_features_npz(mesh, feature_dir, log=log)
    colors, has_colors = resolve_vertex_color_guide(mesh, feat, vertex_color_guide=vertex_color_guide)
    feat["vertex_colors"] = colors
    feat["has_vertex_colors"] = has_colors
    use_color_feat = bool(vertex_color_guide and has_colors)
    x_np = build_node_features(
        v, n, feat, extended=extended_features, use_vertex_color_guide=use_color_feat
    )
    x = torch.from_numpy(x_np)
    guide = compute_cross_field(
        mesh, up_axis=up_axis, rotate_axis=rotate_axis, rotate_deg=rotate_deg, use_curvature=use_curvature
    )
    g = torch.from_numpy(guide.reshape(len(v), 6).astype(np.float32))
    edge_index = mesh_to_edge_index(mesh)
    dihedral_map = compute_edge_dihedral_map(mesh)
    edge_attr = build_edge_attr(v, edge_index, dihedral_map=dihedral_map)
    normals = torch.from_numpy(n)
    return x, edge_index, edge_attr, g, normals, feat


class GraphSAGEConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, edge_dim: int = EDGE_ATTR_DIM):
        super().__init__()
        self.edge_dim = edge_dim
        self.lin_self = nn.Linear(in_ch, out_ch)
        self.lin_nei = nn.Linear(in_ch, out_ch)
        self.lin_edge = nn.Linear(edge_dim, in_ch, bias=False) if edge_dim > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.lin_self(x)
        src, dst = edge_index
        msg = x[src]
        if edge_attr is not None and self.lin_edge is not None:
            msg = msg + self.lin_edge(edge_attr)
        agg = torch.zeros_like(x)
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, msg)
        deg.index_add_(0, dst, torch.ones(src.size(0), device=x.device, dtype=x.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(1)
        neigh = agg / deg
        return self.lin_self(x) + self.lin_nei(neigh)


class QuadFieldGNN(nn.Module):
    def __init__(self, in_dim: int = EXTENDED_IN_DIM, hidden: int = 64, edge_dim: int = EDGE_ATTR_DIM):
        super().__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.c1 = GraphSAGEConv(in_dim, hidden, edge_dim=edge_dim)
        self.c2 = GraphSAGEConv(hidden, hidden, edge_dim=edge_dim)
        self.c3 = GraphSAGEConv(hidden, hidden, edge_dim=edge_dim)
        self.out = nn.Linear(hidden, 6)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = F.relu(self.c1(x, edge_index, edge_attr))
        h = F.relu(self.c2(h, edge_index, edge_attr))
        h = F.relu(self.c3(h, edge_index, edge_attr))
        return self.out(h)


def project_and_orthogonalize(
    raw: torch.Tensor,
    normals: torch.Tensor,
    pd1: torch.Tensor,
    pd2: torch.Tensor,
    eps: float = 1e-5,
    delta: float = 1e-8,
) -> torch.Tensor:
    """L0 Hard Projection and Gram-Schmidt Orthogonalization Layer."""
    u1 = raw[:, :3]
    u2 = raw[:, 3:]
    
    # 2.1 Tangent Projection
    v1_proj = u1 - (u1 * normals).sum(dim=1, keepdim=True) * normals
    v2_proj = u2 - (u2 * normals).sum(dim=1, keepdim=True) * normals
    
    # 2.2 Gram-Schmidt with Small Norm Guard (Lerp Fallback)
    r1 = torch.linalg.norm(v1_proj, dim=1, keepdim=True)
    alpha1 = torch.clamp(r1 / eps, 0.0, 1.0)
    v1_fade = alpha1 * v1_proj + (1.0 - alpha1) * pd1
    v1 = v1_fade / torch.sqrt((v1_fade ** 2).sum(dim=1, keepdim=True) + delta)
    
    # Orthogonalize v2 relative to v1
    r2 = torch.linalg.norm(v2_proj, dim=1, keepdim=True)
    alpha2 = torch.clamp(r2 / eps, 0.0, 1.0)
    v2_fade = alpha2 * v2_proj + (1.0 - alpha2) * pd2
    
    v2_ortho = v2_fade - (v2_fade * v1).sum(dim=1, keepdim=True) * v1
    r2_ortho = torch.linalg.norm(v2_ortho, dim=1, keepdim=True)
    
    # Second guard: if v2_ortho itself is extremely small, use cross product (n x v1)
    v2_cross = torch.cross(normals, v1, dim=1)
    alpha2_ortho = torch.clamp(r2_ortho / eps, 0.0, 1.0)
    v2_final = alpha2_ortho * v2_ortho + (1.0 - alpha2_ortho) * v2_cross
    
    v2 = v2_final / torch.sqrt((v2_final ** 2).sum(dim=1, keepdim=True) + delta)
    
    return torch.cat([v1, v2], dim=1)


def project_cross_to_tangent(raw: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    # 互換性維持のためのシンプルな L0 射影（fallback にゼロを使用）
    pd1 = torch.zeros_like(normals)
    pd2 = torch.zeros_like(normals)
    return project_and_orthogonalize(raw, normals, pd1, pd2)


def _symm_align(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d0 = torch.abs((a * b).sum(1))
    d1 = torch.abs((a * torch.stack([-b[:, 1], b[:, 0], b[:, 2]], dim=1)).sum(1))
    return torch.maximum(d0, d1)


def _symmetry_loss(
    v1: torch.Tensor,
    v2: torch.Tensor,
    symmetry_pairs: torch.Tensor,
    c_sym: torch.Tensor,
) -> torch.Tensor:
    """対称ペア間 4-way クロスフィールド整合（c_sym 重み付き平均）。"""
    sym = symmetry_pairs.long()
    vj1 = v1[sym]
    vj2 = v2[sym]
    ri1 = v1.clone()
    ri1[:, 0] *= -1.0
    ri2 = v2.clone()
    ri2[:, 0] *= -1.0
    a0 = torch.abs((vj1 * ri1).sum(dim=1))
    a1 = torch.abs((vj1 * ri2).sum(dim=1))
    a2 = torch.abs((vj2 * ri1).sum(dim=1))
    a3 = torch.abs((vj2 * ri2).sum(dim=1))
    align = torch.maximum(torch.maximum(a0, a1), torch.maximum(a2, a3))
    per_v = 1.0 - align
    w = c_sym.clamp_min(0.0)
    return (w * per_v).sum() / w.sum().clamp_min(1e-8)


def _loss_stage_scales(epoch: int, epochs: int, *, phased: bool) -> dict[str, float]:
    """段階 Loss: guide+smooth → feat+boundary → align+sym。"""
    if not phased or epochs <= 1:
        return {"guide": 1.0, "smooth": 1.0, "align": 1.0, "feat": 1.0, "boundary": 1.0, "sym": 1.0}
    p = epoch / max(epochs - 1, 1)
    if p < 0.2:
        return {"guide": 1.0, "smooth": 1.0, "align": 0.0, "feat": 0.0, "boundary": 0.0, "sym": 0.0}
    if p < 0.5:
        return {"guide": 0.8, "smooth": 1.0, "align": 0.3, "feat": 1.0, "boundary": 1.0, "sym": 0.0}
    return {"guide": 0.5, "smooth": 1.0, "align": 1.0, "feat": 1.0, "boundary": 1.0, "sym": 1.0}


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    w = mask.clamp_min(0.0)
    return (x * w).sum() / w.sum().clamp_min(1e-8)


def _compute_color_tangents(
    vertex_colors: torch.Tensor,
    normals: torch.Tensor,
    edge_index: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-vertex boundary tangent (cross(n, gradC)) and strength / variance."""
    n_v = vertex_colors.shape[0]
    device = vertex_colors.device
    grad = torch.zeros(n_v, 3, device=device)
    strength = torch.zeros(n_v, device=device)
    color_var = torch.zeros(n_v, device=device)
    if edge_index.numel() == 0:
        z = torch.zeros(n_v, 3, device=device)
        return z, strength, color_var

    src, dst = edge_index
    dc = vertex_colors[dst] - vertex_colors[src]
    diff_mag = torch.linalg.norm(dc, dim=1)
    grad.index_add_(0, src, dc)
    grad.index_add_(0, dst, -dc)
    strength.index_add_(0, src, diff_mag)
    strength.index_add_(0, dst, diff_mag)
    color_var.index_add_(0, src, diff_mag * diff_mag)
    color_var.index_add_(0, dst, diff_mag * diff_mag)

    deg = torch.zeros(n_v, device=device)
    deg.index_add_(0, src, torch.ones_like(diff_mag))
    deg.index_add_(0, dst, torch.ones_like(diff_mag))
    inv_deg = 1.0 / deg.clamp_min(1.0)
    grad = grad * inv_deg.unsqueeze(1)
    strength = strength * inv_deg
    color_var = color_var * inv_deg - strength * strength

    grad_t = grad - (grad * normals).sum(dim=1, keepdim=True) * normals
    gnorm = torch.linalg.norm(grad_t, dim=1, keepdim=True).clamp_min(1e-6)
    grad_unit = grad_t / gnorm
    color_tangent = torch.cross(normals, grad_unit, dim=1)
    ct_norm = torch.linalg.norm(color_tangent, dim=1, keepdim=True).clamp_min(1e-6)
    color_tangent = color_tangent / ct_norm
    return color_tangent, strength, color_var.clamp_min(0.0)


def l0_l3_integrated_loss(
    pred: torch.Tensor,
    guide: torch.Tensor,
    edge_index: torch.Tensor,
    normals: torch.Tensor,
    pd1: torch.Tensor,
    pd2: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    feature_mask: torch.Tensor,
    feature_tangents: torch.Tensor,
    boundary_mask: torch.Tensor,
    boundary_tangents: torch.Tensor,
    boundary_steps: torch.Tensor,
    symmetry_pairs: torch.Tensor,
    c_sym: torch.Tensor,
    epoch: int,
    epochs: int,
    *,
    w_guide: float = 0.4,
    w_sym: float = 0.5,
    loss_stage: str = "phased",
    vertex_mask: torch.Tensor | None = None,
    teacher_v1: torch.Tensor | None = None,
    teacher_v2: torch.Tensor | None = None,
    w_distill: float = 0.0,
    vertex_colors: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    w_color_align: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L0-L3 統合幾何損失。total と項別スカラーを返す。"""
    epoch_mid = epochs / 2.0
    tau = max(epochs / 10.0, 1.0)
    tc = 0.1 + 0.9 * torch.sigmoid(torch.tensor((epoch - epoch_mid) / tau, device=pred.device))

    v1, v2 = pred[:, :3], pred[:, 3:]
    g1, g2 = guide[:, :3], guide[:, 3:]

    curv_diff = torch.abs(k1 - k2)
    pct95 = float(np.percentile(curv_diff.detach().cpu().numpy(), 95)) if len(curv_diff) > 0 else 1.0
    local_pct95 = pct95 + 1e-8
    gamma = torch.sigmoid(5.0 * (curv_diff / local_pct95) - 2.5)

    w_feat = feature_mask
    w_bound = torch.exp(-0.5 * boundary_steps)
    w_feat_resolved = w_feat * (1.0 - w_bound)
    gamma_resolved = gamma * (1.0 - w_feat_resolved) * (1.0 - w_bound)

    lambda_align = tc * 1.0 * gamma_resolved
    lambda_smooth = (1.0 - tc) * 0.2 + tc * 0.1 * (1.0 - gamma_resolved)
    lambda_feat = 1.5 * w_feat_resolved
    lambda_boundary = 2.0 * w_bound

    stage = _loss_stage_scales(epoch, epochs, phased=(loss_stage == "phased"))
    lambda_align = lambda_align * stage["align"]
    lambda_smooth = lambda_smooth * stage["smooth"]
    lambda_feat = lambda_feat * stage["feat"]
    lambda_boundary = lambda_boundary * stage["boundary"]
    w_guide_eff = w_guide * stage["guide"]
    w_sym_eff = w_sym * stage["sym"]

    dot_v1_pd1 = torch.abs((v1 * pd1).sum(dim=1))
    dot_v1_pd2 = torch.abs((v1 * pd2).sum(dim=1))
    loss_align_v1 = 1.0 - torch.maximum(dot_v1_pd1, dot_v1_pd2)
    dot_v2_pd1 = torch.abs((v2 * pd1).sum(dim=1))
    dot_v2_pd2 = torch.abs((v2 * pd2).sum(dim=1))
    loss_align_v2 = 1.0 - torch.maximum(dot_v2_pd1, dot_v2_pd2)
    loss_align = 0.5 * (loss_align_v1 + loss_align_v2)

    loss_guide_per = (1.0 - _symm_align(v1, g1)) + (1.0 - _symm_align(v2, g2))
    loss_guide = loss_guide_per.mean()

    pd_norm = torch.linalg.norm(pd1, dim=1)
    pd_valid = (pd_norm > 0.1).float()
    loss_align_combined = pd_valid * loss_align + (1.0 - pd_valid) * (loss_guide_per * 0.5)

    loss_smooth = torch.tensor(0.0, device=pred.device)
    if edge_index.numel() > 0:
        src, dst = edge_index
        smooth_v1 = 1.0 - torch.maximum(
            torch.abs((v1[src] * v1[dst]).sum(dim=1)),
            torch.abs((v1[src] * v2[dst]).sum(dim=1)),
        )
        smooth_v2 = 1.0 - torch.maximum(
            torch.abs((v2[src] * v1[dst]).sum(dim=1)),
            torch.abs((v2[src] * v2[dst]).sum(dim=1)),
        )
        loss_smooth = 0.5 * (smooth_v1 + smooth_v2)

    ft_norm = torch.linalg.norm(feature_tangents, dim=1)
    ft_valid = (ft_norm > 0.1).float()
    dot_v1_ft = torch.abs((v1 * feature_tangents).sum(dim=1))
    dot_v2_ft = torch.abs((v2 * feature_tangents).sum(dim=1))
    loss_feat_edge = 1.0 - torch.maximum(dot_v1_ft, dot_v2_ft)
    loss_feat = ft_valid * loss_feat_edge + (1.0 - ft_valid) * loss_align_v1
    dot_v1_t = torch.abs((v1 * boundary_tangents).sum(dim=1))
    dot_v2_t = torch.abs((v2 * boundary_tangents).sum(dim=1))
    loss_boundary = 1.0 - torch.maximum(dot_v1_t, dot_v2_t)
    loss_sym = _symmetry_loss(v1, v2, symmetry_pairs, c_sym)
    loss_ortho_per = torch.abs((v1 * v2).sum(dim=1))

    term_align = _masked_mean(lambda_align * loss_align_combined, vertex_mask)
    term_smooth = (
        lambda_smooth.mean() * loss_smooth.mean()
        if edge_index.numel() > 0
        else torch.tensor(0.0, device=pred.device)
    )
    term_feat = _masked_mean(lambda_feat * loss_feat, vertex_mask)
    term_boundary = _masked_mean(lambda_boundary * loss_boundary, vertex_mask)
    term_guide = w_guide_eff * _masked_mean(loss_guide_per, vertex_mask)
    term_sym = w_sym_eff * loss_sym
    term_ortho = 0.1 * _masked_mean(loss_ortho_per, vertex_mask)

    term_distill = torch.tensor(0.0, device=pred.device)
    if w_distill > 0 and teacher_v1 is not None and teacher_v2 is not None:
        d1 = 1.0 - torch.maximum(
            torch.abs((v1 * teacher_v1).sum(dim=1)),
            torch.abs((v1 * teacher_v2).sum(dim=1)),
        )
        d2 = 1.0 - torch.maximum(
            torch.abs((v2 * teacher_v1).sum(dim=1)),
            torch.abs((v2 * teacher_v2).sum(dim=1)),
        )
        term_distill = w_distill * _masked_mean(d1 + d2, vertex_mask)

    term_color = torch.tensor(0.0, device=pred.device)
    color_active = 0.0
    if (
        w_color_align > 0.0
        and vertex_colors is not None
        and positions is not None
        and edge_index.numel() > 0
    ):
        color_tangent, color_strength, color_var = _compute_color_tangents(
            vertex_colors, normals, edge_index, positions
        )
        med = color_strength.median()
        if float(med.detach()) > 1e-5:
            robust = med + 1.4826 * (color_strength - med).abs().median()
            thresh = max(float(robust.detach()) * 2.0, 1e-4)
            w_color_v = torch.sigmoid(8.0 * (color_strength / (thresh + 1e-6) - 1.0))
            w_color_v = w_color_v * (1.0 - feature_mask.clamp(0.0, 1.0))
            var_thr = float(torch.quantile(color_var.detach(), 0.90).clamp_min(1e-8))
            if var_thr > 1e-8:
                w_color_v = w_color_v * torch.sigmoid(2.0 * (var_thr - color_var) / (var_thr + 1e-6))
            ct_valid = (torch.linalg.norm(color_tangent, dim=1) > 0.1).float()
            dot1 = torch.abs((v1 * color_tangent).sum(dim=1))
            dot2 = torch.abs((v2 * color_tangent).sum(dim=1))
            loss_color_per = 1.0 - torch.maximum(dot1, dot2)
            denom = (w_color_v * ct_valid).sum().clamp_min(1e-8)
            term_color = w_color_align * stage["align"] * (w_color_v * ct_valid * loss_color_per).sum() / denom
            color_active = float((w_color_v * ct_valid).mean().detach())

    total = (
        term_align
        + term_smooth
        + term_feat
        + term_boundary
        + term_guide
        + term_sym
        + term_ortho
        + term_distill
        + term_color
    )

    comps = {
        "align": float(term_align.detach()),
        "smooth": float(term_smooth.detach()),
        "feat": float(term_feat.detach()),
        "boundary": float(term_boundary.detach()),
        "guide": float(term_guide.detach()),
        "sym": float(term_sym.detach()),
        "ortho": float(term_ortho.detach()),
        "distill": float(term_distill.detach()),
        "color": float(term_color.detach()),
        "color_active": color_active,
        "total": float(total.detach()),
        "tc": float(tc.detach()),
        "sym_raw": float(loss_sym.detach()),
        "c_sym_mean": float(c_sym.mean().detach()),
        "stage": stage,
    }
    return total, comps


def unsupervised_loss(
    pred: torch.Tensor,
    guide: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    w_guide: float = 1.0,
    w_ortho: float = 0.5,
    w_smooth: float = 0.2,
) -> torch.Tensor:
    # 互換性維持のためのラップ関数
    v1, v2 = pred[:, :3], pred[:, 3:]
    g1, g2 = guide[:, :3], guide[:, 3:]
    guide_loss = (1.0 - _symm_align(v1, g1)).mean() + (1.0 - _symm_align(v2, g2)).mean()
    ortho = torch.abs((v1 * v2).sum(1)).mean()
    if edge_index.numel() == 0:
        return w_guide * guide_loss + w_ortho * ortho
    src, dst = edge_index
    smooth = (
        (1.0 - _symm_align(v1[src], v1[dst])).mean()
        + (1.0 - _symm_align(v2[src], v2[dst])).mean()
    ) * 0.5
    return w_guide * guide_loss + w_ortho * ortho + w_smooth * smooth


def _remap_symmetry_pairs(symmetry_pairs: torch.Tensor, nodes) -> torch.Tensor:
    dev = symmetry_pairs.device
    nodes = torch.as_tensor(nodes, dtype=torch.long, device=dev)
    remap = torch.full((symmetry_pairs.numel(),), -1, dtype=torch.long, device=dev)
    local = torch.arange(nodes.numel(), dtype=torch.long, device=dev)
    remap[nodes] = local
    out = remap[symmetry_pairs[nodes].long()]
    return torch.where(out >= 0, out, local)



def save_model(path: Path, model: QuadFieldGNN, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = dict(meta or {})
    m.setdefault("edge_attr_dim", model.edge_dim)
    payload = {
        "state_dict": model.state_dict(),
        "in_dim": model.in_dim,
        "edge_attr_dim": model.edge_dim,
        "meta": m,
    }
    torch.save(payload, path)


def _load_model_from_payload(payload: dict, device: torch.device) -> QuadFieldGNN:
    in_dim = int(payload.get("in_dim", EXTENDED_IN_DIM))
    edge_dim = int(payload.get("edge_attr_dim", payload.get("meta", {}).get("edge_attr_dim", EDGE_ATTR_DIM)))
    model = QuadFieldGNN(in_dim=in_dim, edge_dim=edge_dim).to(device)
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError:
        model = QuadFieldGNN(in_dim=in_dim, edge_dim=EDGE_ATTR_DIM).to(device)
    return model


def load_model(path: Path, device: str | None = None) -> QuadFieldGNN:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=dev, weights_only=False)
    model = _load_model_from_payload(payload, dev)
    model.eval()
    return model


def load_checkpoint(path: Path, device: str | None = None) -> tuple[QuadFieldGNN, dict]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=dev, weights_only=False)
    model = _load_model_from_payload(payload, dev)
    return model, payload.get("meta", {})


def release_cuda_cache() -> None:
    """学習後: PyTorch CUDA キャッシュを返却（同一プロセス内の IM 等に余裕を作る）。"""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _configure_gnn_training_backend(device: torch.device) -> None:
    """Quality-neutral PyTorch backend tuning (AMP/compile/cudnn)."""
    import os

    if device.type != "cuda":
        return
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _maybe_compile_gnn(model: nn.Module) -> nn.Module:
    import os

    flag = os.environ.get("GNN_TORCH_COMPILE", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        return model


def _materialize_minibatch(
    *,
    dev: torch.device,
    nodes,
    sub_ei: torch.Tensor,
    seed_local,
    sym_sub: torch.Tensor,
    sub_ea: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    idx = torch.as_tensor(nodes, dtype=torch.long, device=dev)
    seed_mask = torch.zeros(idx.numel(), device=dev)
    seed_mask[torch.as_tensor(seed_local, dtype=torch.long, device=dev)] = 1.0
    out: dict[str, torch.Tensor | None] = {
        "idx": idx,
        "sub_ei": sub_ei.to(dev, non_blocking=True),
        "seed_mask": seed_mask,
        "sym_sub": sym_sub.to(dev, non_blocking=True),
        "sub_ea": sub_ea.to(dev, non_blocking=True) if sub_ea is not None else None,
    }
    return out


def _ensure_torch_utils() -> None:
    """Colab の一部の torch は torch._utils を公開しない。学習開始で参照される。"""
    import sys
    import types

    if "torch._utils" in sys.modules:
        return
    try:
        import torch._utils  # noqa: F401
        return
    except Exception:
        pass
    mod = types.ModuleType("torch._utils")

    def _get_device_index(device, optional=False, allow_cpu=False):
        if device is None:
            if optional or not torch.cuda.is_available():
                return -1
            return torch.cuda.current_device()
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda":
            return -1
        if device.index is None:
            return torch.cuda.current_device() if torch.cuda.is_available() else -1
        return int(device.index)

    def _element_size(dtype):
        return torch.empty((), dtype=dtype).element_size()

    mod._get_device_index = _get_device_index
    mod._element_size = _element_size
    sys.modules["torch._utils"] = mod


def train_cross_field(
    mesh: trimesh.Trimesh,
    *,
    epochs: int = 100,
    lr: float = 0.005,
    device: str | None = None,
    up_axis: str | None = None,
    rotate_axis: str | None = None,
    rotate_deg: float | None = None,
    patience: int = 80,
    min_delta: float = 1e-4,
    use_curvature: bool = False,
    feature_dir: Path | None = None,
    batch_size: int = 0,
    loss_stage: str = "phased",
    checkpoint: Path | None = None,
    extended_features: bool = True,
    w_distill: float = 0.0,
    teacher_v1: np.ndarray | None = None,
    teacher_v2: np.ndarray | None = None,
    save_checkpoint: Path | None = None,
    vertex_color_guide: bool = True,
    w_color_align: float = DEFAULT_W_COLOR_ALIGN,
) -> tuple[QuadFieldGNN, np.ndarray]:
    log_dir = Path(feature_dir) if feature_dir else Path(".")
    log = DebugLog("gnn-field", log_dir)
    _ensure_torch_utils()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    log.step("build graph + stable guide field")
    x, edge_index, edge_attr, guide, normals, feat_np = prepare_graph_data(
        mesh, up_axis=up_axis, rotate_axis=rotate_axis, rotate_deg=rotate_deg,
        use_curvature=use_curvature, feature_dir=feature_dir, extended_features=extended_features,
        vertex_color_guide=vertex_color_guide, log=log,
    )
    n_v = len(mesh.vertices)
    n_e = edge_index.shape[1] // 2
    in_dim = x.shape[1]
    auto_tuned = batch_size < 0
    auto_plan: dict | None = None
    if auto_tuned:
        from .auto_tune import build_train_plan, resolve_batch_size, save_auto_tune_report

        batch_size = resolve_batch_size(batch_size, n_v, in_dim=in_dim, device=str(dev))
        auto_plan = build_train_plan(n_v, in_dim=in_dim, device=str(dev))
        save_auto_tune_report(log_dir, auto_plan)
    log.set("vertices", n_v)
    log.set("edges", n_e)
    log.set("in_dim", in_dim)
    log.set("edge_attr_dim", EDGE_ATTR_DIM)
    color_active = bool(vertex_color_guide and feat_np.get("has_vertex_colors", False))
    log.set("vertex_color_guide", vertex_color_guide)
    log.set("vertex_color_active", color_active)
    w_color_eff = w_color_align if color_active else 0.0
    log.set("w_color_align", w_color_eff)
    log.set("device", str(dev))
    if dev.type == "cuda":
        try:
            free, total = torch.cuda.mem_get_info()
            log.set("gpu_mem_free_gb", round(free / 1e9, 2))
            log.set("gpu_mem_total_gb", round(total / 1e9, 2))
        except Exception:
            pass

    x = x.to(dev)
    edge_index = edge_index.to(dev)
    edge_attr = edge_attr.to(dev)
    guide = guide.to(dev)
    normals = normals.to(dev)

    pd1_np = feat_np["pd1"]
    pd2_np = feat_np["pd2"]
    k1_np = feat_np["k1"]
    k2_np = feat_np["k2"]
    feature_mask_np = feat_np["feature_mask"]
    feature_tangents_np = feat_np["feature_tangents"]
    boundary_mask_np = feat_np["boundary_mask"]
    boundary_tangents_np = feat_np["boundary_tangents"]
    boundary_steps_np = feat_np["boundary_steps"]
    symmetry_pairs_np = feat_np["symmetry_pairs"]
    c_sym_np = feat_np["c_sym"]

    pd1 = torch.from_numpy(pd1_np.copy()).to(dev)
    pd2 = torch.from_numpy(pd2_np.copy()).to(dev)
    k1 = torch.from_numpy(k1_np.copy()).to(dev)
    k2 = torch.from_numpy(k2_np.copy()).to(dev)
    feature_mask = torch.from_numpy(feature_mask_np.copy()).to(dev)
    feature_tangents = torch.from_numpy(feature_tangents_np.copy()).to(dev)
    boundary_mask = torch.from_numpy(boundary_mask_np.copy()).to(dev)
    boundary_tangents = torch.from_numpy(boundary_tangents_np.copy()).to(dev)
    boundary_steps = torch.from_numpy(boundary_steps_np.copy()).to(dev)
    symmetry_pairs = torch.from_numpy(symmetry_pairs_np.copy()).to(dev)
    c_sym = torch.from_numpy(c_sym_np.copy()).to(dev)
    log.set("c_sym_mean", round(float(c_sym_np.mean()), 4))
    log.set("c_sym_active", int((c_sym_np > 0.1).sum()))

    positions = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(dev)
    vertex_colors_t = None
    if color_active:
        vertex_colors_t = torch.from_numpy(feat_np["vertex_colors"].astype(np.float32)).to(dev)

    tv1_t, tv2_t = None, None
    if w_distill > 0 and teacher_v1 is not None and teacher_v2 is not None:
        tv1_t = torch.from_numpy(np.asarray(teacher_v1, dtype=np.float32)).to(dev)
        tv2_t = torch.from_numpy(np.asarray(teacher_v2, dtype=np.float32)).to(dev)
        log.set("w_distill", w_distill)

    ck_meta: dict = {}
    if checkpoint and Path(checkpoint).exists():
        log.step(f"load checkpoint {checkpoint}")
        model, ck_meta = load_checkpoint(checkpoint, device=str(dev))
        if model.in_dim != in_dim or model.edge_dim != EDGE_ATTR_DIM:
            log.warn(
                f"checkpoint in_dim={model.in_dim} edge_dim={model.edge_dim} "
                f"!= data in_dim={in_dim} edge_dim={EDGE_ATTR_DIM}; re-init model"
            )
            model = QuadFieldGNN(in_dim=in_dim, edge_dim=EDGE_ATTR_DIM).to(dev)
            ck_meta = {}
        else:
            log.info(f"checkpoint loaded (meta={ck_meta})")
    else:
        model = QuadFieldGNN(in_dim=in_dim, edge_dim=EDGE_ATTR_DIM).to(dev)
    model = _maybe_compile_gnn(model)
    _configure_gnn_training_backend(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    use_minibatch = batch_size > 0 and n_v > batch_size
    sampler = (
        NeighborBatchSampler(
            edge_index, n_v, edge_attr=edge_attr, batch_size=batch_size or 4096
        )
        if use_minibatch
        else None
    )
    steps_per_epoch = max(1, (n_v + (batch_size or n_v) - 1) // (batch_size or n_v)) if use_minibatch else 1
    log.set("batch_size", batch_size if use_minibatch else "full")
    log.set("steps_per_epoch", steps_per_epoch)
    log.set("loss_stage", loss_stage)
    if auto_tuned:
        from .auto_tune import detect_runtime, format_auto_train_line

        rt = detect_runtime(str(dev))
        log.info(
            format_auto_train_line(
                device=str(dev),
                batch_size=batch_size if use_minibatch else 0,
                n_vertices=n_v,
                steps_per_epoch=steps_per_epoch,
                gpu_total_gb=rt.vram_effective_gb,
                runtime=rt,
            )
        )
        if auto_plan:
            log.set("auto_tune", str(log_dir / "auto_tune.json"), quiet=True)

    # Mixed Precision 用の GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    def _sample_epoch_batches() -> list[dict[str, torch.Tensor | None]]:
        if not use_minibatch or sampler is None:
            return []
        batches: list[dict[str, torch.Tensor | None]] = []
        for _ in range(steps_per_epoch):
            nodes, sub_ei, seed_local, sub_ea = sampler.sample()
            sym_sub = _remap_symmetry_pairs(symmetry_pairs, nodes)
            batches.append(
                _materialize_minibatch(
                    dev=dev,
                    nodes=nodes,
                    sub_ei=sub_ei,
                    seed_local=seed_local,
                    sym_sub=sym_sub,
                    sub_ea=sub_ea,
                )
            )
        return batches

    # 1 epoch 分を事前サンプル（GPUへ転送）。毎 epoch 再サンプルで品質維持 + ベクトル化で高速。
    log.step(f"train GNN ({epochs} epochs, patience={patience}, batch={'mini' if use_minibatch else 'full'})")
    best_loss = float(ck_meta.get("best_loss", float("inf")))
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()} if ck_meta else None
    if best_state is not None and best_loss < float("inf"):
        log.info(f"resume best_loss={best_loss:.6f} (checkpoint weights kept until improved)")
    patience_counter = 0
    log_interval = 1
    reset_for_final_stage = loss_stage != "phased"

    model.train()
    for ep in range(epochs):
        epoch_batches = _sample_epoch_batches()
        ep_loss = 0.0
        ep_comps: dict[str, float] = {}
        for step in range(steps_per_epoch):
            opt.zero_grad()
            with torch.autocast(device_type=dev.type, enabled=(dev.type == "cuda")):
                if use_minibatch and sampler is not None:
                    batch = epoch_batches[step]
                    idx = batch["idx"]
                    seed_mask = batch["seed_mask"]
                    sub_ei = batch["sub_ei"]
                    sub_ea_d = batch["sub_ea"]
                    sym_sub = batch["sym_sub"]
                    x_b = x[idx]
                    guide_b = guide[idx]
                    normals_b = normals[idx]
                    pd1_b, pd2_b = pd1[idx], pd2[idx]
                    k1_b, k2_b = k1[idx], k2[idx]
                    fm_b = feature_mask[idx]
                    ft_b = feature_tangents[idx]
                    bm_b = boundary_mask[idx]
                    bt_b = boundary_tangents[idx]
                    bs_b = boundary_steps[idx]
                    cs_b = c_sym[idx]
                    raw = model(x_b, sub_ei, sub_ea_d)
                    pred = project_and_orthogonalize(raw, normals_b, pd1_b, pd2_b)
                    loss, comps = l0_l3_integrated_loss(
                        pred, guide_b, sub_ei, normals_b, pd1_b, pd2_b, k1_b, k2_b,
                        fm_b, ft_b, bm_b, bt_b, bs_b, sym_sub, cs_b, ep, epochs,
                        loss_stage=loss_stage, vertex_mask=seed_mask,
                        teacher_v1=tv1_t[idx] if tv1_t is not None else None,
                        teacher_v2=tv2_t[idx] if tv2_t is not None else None,
                        w_distill=w_distill,
                        vertex_colors=vertex_colors_t[idx] if vertex_colors_t is not None else None,
                        positions=positions[idx],
                        w_color_align=w_color_eff,
                    )
                else:
                    raw = model(x, edge_index, edge_attr)
                    pred = project_and_orthogonalize(raw, normals, pd1, pd2)
                    loss, comps = l0_l3_integrated_loss(
                        pred, guide, edge_index, normals, pd1, pd2, k1, k2,
                        feature_mask, feature_tangents, boundary_mask, boundary_tangents, boundary_steps,
                        symmetry_pairs, c_sym, ep, epochs, loss_stage=loss_stage,
                        teacher_v1=tv1_t, teacher_v2=tv2_t, w_distill=w_distill,
                        vertex_colors=vertex_colors_t,
                        positions=positions,
                        w_color_align=w_color_eff,
                    )

            loss_val = loss.item()
            
            # AMP対応のbackward & step
            if dev.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
                
            ep_loss += loss_val
            ep_comps = comps

        scheduler.step()
        loss_val = ep_loss / steps_per_epoch

        stage_p = ep / max(epochs - 1, 1)
        final_stage = loss_stage != "phased" or stage_p >= 0.5
        if final_stage and not reset_for_final_stage:
            best_loss = float("inf")
            patience_counter = 0
            reset_for_final_stage = True
            log.info("phased: reset best checkpoint for final align stage")
        delta = min_delta if best_loss == float("inf") else max(min_delta, 0.002 * best_loss)
        if loss_val < best_loss - delta:
            best_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if final_stage:
                patience_counter = 0
        elif final_stage:
            patience_counter += 1

        epoch_mid = epochs / 2.0
        tau = max(epochs / 10.0, 1.0)
        tc = float(0.1 + 0.9 * torch.sigmoid(torch.tensor((ep - epoch_mid) / tau)).item())

        if (ep + 1) % log_interval == 0 or ep == 0 or (ep + 1) == epochs or patience_counter >= patience:
            current_lr = opt.param_groups[0]["lr"]
            st = ep_comps.get("stage", {})
            log.info(
                f"epoch {ep + 1:03d}/{epochs} | total={loss_val:.4f} | best={best_loss:.4f} "
                f"| lr={current_lr:.6f} | Tc={tc:.3f} | patience={patience_counter}/{patience}"
            )
            log.info(
                f"  terms: guide={ep_comps['guide']:.4f} align={ep_comps['align']:.4f} "
                f"smooth={ep_comps['smooth']:.4f} feat={ep_comps['feat']:.4f} "
                f"bound={ep_comps['boundary']:.4f} sym={ep_comps['sym']:.4f} "
                f"color={ep_comps.get('color', 0):.4f} "
                f"distill={ep_comps.get('distill', 0):.4f} "
                f"(raw={ep_comps['sym_raw']:.4f} c_sym={ep_comps['c_sym_mean']:.4f})"
            )
            if st:
                log.info(
                    f"  stage: guide={st.get('guide', 0):.1f} align={st.get('align', 0):.1f} "
                    f"feat={st.get('feat', 0):.1f} bound={st.get('boundary', 0):.1f} sym={st.get('sym', 0):.1f}"
                )

        if final_stage and patience_counter >= patience:
            log.info(f"early stop at epoch {ep + 1} (no improvement >= {min_delta} for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    ck_path = save_checkpoint or (log_dir / "gnn_checkpoint.pt")
    save_model(
        ck_path,
        model,
        meta={"best_loss": best_loss, "in_dim": in_dim, "edge_attr_dim": EDGE_ATTR_DIM, "epochs": epochs},
    )
    log.set("checkpoint", str(ck_path))

    log.step("export cross field")
    model.eval()
    with torch.no_grad():
        pred = project_and_orthogonalize(model(x, edge_index, edge_attr), normals, pd1, pd2)
    field = pred.cpu().numpy().reshape(len(mesh.vertices), 2, 3)
    log.set("best_loss", round(best_loss, 6))
    log.done("gnn-field")
    log.save()

    del pred, model, opt, scheduler, sampler, best_state
    del x, edge_index, edge_attr, guide, normals, pd1, pd2, k1, k2
    del feature_mask, feature_tangents, boundary_mask, boundary_tangents, boundary_steps
    del symmetry_pairs, c_sym, positions
    if vertex_colors_t is not None:
        del vertex_colors_t
    if tv1_t is not None:
        del tv1_t, tv2_t
    release_cuda_cache()
    log.info("GPU/CPU training buffers released")

    return None, field


def predict_cross_field(
    mesh: trimesh.Trimesh,
    model: QuadFieldGNN,
    device: str | None = None,
    feature_dir: Path | None = None,
    vertex_color_guide: bool = True,
) -> np.ndarray:
    dev = next(model.parameters()).device
    use_color = model.in_dim >= EXTENDED_IN_DIM_WITH_COLOR
    x, edge_index, edge_attr, _, normals, feat_np = prepare_graph_data(
        mesh,
        feature_dir=feature_dir,
        extended_features=(model.in_dim > BASE_IN_DIM),
        vertex_color_guide=vertex_color_guide and use_color,
    )
    pd1 = torch.from_numpy(feat_np["pd1"].astype(np.float32)).to(dev)
    pd2 = torch.from_numpy(feat_np["pd2"].astype(np.float32)).to(dev)
    x = x.to(dev)
    edge_index = edge_index.to(dev)
    edge_attr = edge_attr.to(dev)
    normals = normals.to(dev)
    with torch.no_grad():
        pred = project_and_orthogonalize(model(x, edge_index, edge_attr), normals, pd1, pd2)
    return pred.cpu().numpy().reshape(len(mesh.vertices), 2, 3)


def save_field(path: Path, field: np.ndarray) -> None:
    np.save(path, field)
