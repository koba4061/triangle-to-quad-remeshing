"""Patch-based GNN pretraining with IM teacher distillation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .debug_log import DebugLog
from .gnn_model import (
    DEFAULT_W_COLOR_ALIGN,
    EDGE_ATTR_DIM,
    EXTENDED_IN_DIM,
    NeighborBatchSampler,
    QuadFieldGNN,
    build_edge_attr,
    build_node_features,
    compute_edge_dihedral_map,
    l0_l3_integrated_loss,
    load_features_npz,
    mesh_to_edge_index,
    project_and_orthogonalize,
    save_model,
)
from .vertex_color import resolve_vertex_color_guide
from .im_teacher import load_patch_teacher


def _load_patch(obj_path: Path) -> tuple[trimesh.Trimesh, np.ndarray | None, np.ndarray | None]:
    mesh = trimesh.load(obj_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    teacher_path = obj_path.parent / (obj_path.stem + "_teacher.npz")
    if teacher_path.exists():
        tv1, tv2 = load_patch_teacher(teacher_path)
        return mesh, tv1, tv2
    return mesh, None, None


def list_patch_objs(patch_dir: Path) -> list[Path]:
    patch_dir = Path(patch_dir)
    objs = sorted(patch_dir.rglob("*.obj"))
    return [p for p in objs if "_teacher" not in p.stem and "retopo" not in p.stem.lower()]


def pretrain_on_patches(
    patch_dir: Path,
    *,
    epochs: int = 100,
    lr: float = 0.005,
    device: str | None = None,
    batch_size: int = 2048,
    out_path: Path | None = None,
    w_distill: float = 1.0,
    loss_stage: str = "phased",
    vertex_color_guide: bool = False,
    w_color_align: float = DEFAULT_W_COLOR_ALIGN,
) -> Path:
    import torch

    patch_dir = Path(patch_dir)
    log = DebugLog("pretrain", patch_dir)
    objs = list_patch_objs(patch_dir)
    if not objs:
        raise FileNotFoundError(f"no patch OBJ in {patch_dir}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = QuadFieldGNN(in_dim=EXTENDED_IN_DIM, edge_dim=EDGE_ATTR_DIM).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log.step(f"pretrain on {len(objs)} patches, {epochs} epochs, w_distill={w_distill}")

    # Cache patch graph datasets in memory to eliminate redundant disk I/O and features extraction across epochs
    log.step("caching patch datasets...")
    cached_patches = []
    for obj_path in objs:
        mesh, tv1, tv2 = _load_patch(obj_path)
        feat, _ = load_features_npz(mesh, obj_path.parent, log=None)
        v = np.asarray(mesh.vertices, dtype=np.float32)
        n = np.asarray(mesh.vertex_normals, dtype=np.float32)
        
        x_cpu = torch.from_numpy(build_node_features(v, n, feat))
        edge_index = mesh_to_edge_index(mesh)
        dihedral_map = compute_edge_dihedral_map(mesh)
        edge_attr = build_edge_attr(v, edge_index, dihedral_map=dihedral_map)
        
        n_v = len(v)
        guide_cpu = torch.zeros(n_v, 6)
        normals_cpu = torch.from_numpy(n)
        pd1_cpu = torch.from_numpy(feat["pd1"].astype(np.float32))
        pd2_cpu = torch.from_numpy(feat["pd2"].astype(np.float32))
        k1_cpu = torch.from_numpy(feat["k1"].astype(np.float32))
        k2_cpu = torch.from_numpy(feat["k2"].astype(np.float32))
        fm_cpu = torch.from_numpy(feat["feature_mask"].astype(np.float32))
        ft_cpu = torch.from_numpy(feat["feature_tangents"].astype(np.float32))
        bm_cpu = torch.from_numpy(feat["boundary_mask"].astype(np.float32))
        bt_cpu = torch.from_numpy(feat["boundary_tangents"].astype(np.float32))
        bs_cpu = torch.from_numpy(feat["boundary_steps"].astype(np.float32))
        sym_cpu = torch.from_numpy(feat["symmetry_pairs"].astype(np.int64))
        cs_cpu = torch.from_numpy(feat["c_sym"].astype(np.float32))
        vcols, has_vcol = resolve_vertex_color_guide(
            mesh, feat, vertex_color_guide=vertex_color_guide
        )
        vcols_cpu = torch.from_numpy(vcols.astype(np.float32))
        pos_cpu = torch.from_numpy(v.astype(np.float32))
        w_color_patch = w_color_align if (vertex_color_guide and has_vcol) else 0.0

        tv1_cpu = torch.from_numpy(tv1.astype(np.float32)) if tv1 is not None else None
        tv2_cpu = torch.from_numpy(tv2.astype(np.float32)) if tv2 is not None else None

        cached_patches.append({
            "x": x_cpu,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "guide": guide_cpu,
            "normals": normals_cpu,
            "pd1": pd1_cpu,
            "pd2": pd2_cpu,
            "k1": k1_cpu,
            "k2": k2_cpu,
            "fm": fm_cpu,
            "ft": ft_cpu,
            "bm": bm_cpu,
            "bt": bt_cpu,
            "bs": bs_cpu,
            "sym": sym_cpu,
            "cs": cs_cpu,
            "tv1": tv1_cpu,
            "tv2": tv2_cpu,
            "n_v": n_v,
            "vertex_colors": vcols_cpu,
            "positions": pos_cpu,
            "w_color_align": w_color_patch,
        })
    log.info(f"successfully cached {len(cached_patches)} patches in memory")

    for ep in range(epochs):
        ep_loss = 0.0
        # Shuffle cached patch graphs
        indices = np.arange(len(cached_patches))
        np.random.shuffle(indices)
        for idx_p in indices:
            pdata = cached_patches[idx_p]
            x = pdata["x"].to(dev)
            edge_index = pdata["edge_index"]
            edge_attr = pdata["edge_attr"]
            guide = pdata["guide"].to(dev)
            normals = pdata["normals"].to(dev)
            pd1 = pdata["pd1"].to(dev)
            pd2 = pdata["pd2"].to(dev)
            k1 = pdata["k1"].to(dev)
            k2 = pdata["k2"].to(dev)
            fm = pdata["fm"].to(dev)
            ft = pdata["ft"].to(dev)
            bm = pdata["bm"].to(dev)
            bt = pdata["bt"].to(dev)
            bs = pdata["bs"].to(dev)
            sym = pdata["sym"].to(dev)
            cs = pdata["cs"].to(dev)
            
            tv1_t = pdata["tv1"].to(dev) if pdata["tv1"] is not None else None
            tv2_t = pdata["tv2"].to(dev) if pdata["tv2"] is not None else None
            wd = w_distill if tv1_t is not None else 0.0
            w_color = float(pdata.get("w_color_align", 0.0))
            vcols_t = pdata["vertex_colors"].to(dev)
            pos_t = pdata["positions"].to(dev)

            n_v = pdata["n_v"]
            use_mb = batch_size > 0 and n_v > batch_size
            if use_mb:
                sampler = NeighborBatchSampler(
                    edge_index, n_v, edge_attr=edge_attr, batch_size=batch_size
                )
                nodes, sub_ei, seed_local, sub_ea = sampler.sample()
                idx = torch.tensor(nodes, dtype=torch.long, device=dev)
                seed_mask = torch.zeros(len(nodes), device=dev)
                seed_mask[torch.from_numpy(seed_local).to(dev)] = 1.0
                from .gnn_model import _remap_symmetry_pairs
                sym_sub = _remap_symmetry_pairs(sym, nodes)
                sub_ea_d = sub_ea.to(dev) if sub_ea is not None else None
                raw = model(x[idx], sub_ei.to(dev), sub_ea_d)
                pred = project_and_orthogonalize(raw, normals[idx], pd1[idx], pd2[idx])
                loss, _ = l0_l3_integrated_loss(
                    pred, guide[idx], sub_ei.to(dev), normals[idx], pd1[idx], pd2[idx],
                    k1[idx], k2[idx], fm[idx], ft[idx], bm[idx], bt[idx], bs[idx],
                    sym_sub, cs[idx], ep, epochs, loss_stage=loss_stage, vertex_mask=seed_mask,
                    teacher_v1=tv1_t[idx] if tv1_t is not None else None,
                    teacher_v2=tv2_t[idx] if tv2_t is not None else None,
                    w_distill=wd,
                    vertex_colors=vcols_t[idx] if w_color > 0 else None,
                    positions=pos_t[idx],
                    w_color_align=w_color,
                )
            else:
                edge_index_d = edge_index.to(dev)
                edge_attr_d = edge_attr.to(dev)
                raw = model(x, edge_index_d, edge_attr_d)
                pred = project_and_orthogonalize(raw, normals, pd1, pd2)
                loss, _ = l0_l3_integrated_loss(
                    pred, guide, edge_index_d, normals, pd1, pd2, k1, k2,
                    fm, ft, bm, bt, bs, sym, cs, ep, epochs, loss_stage=loss_stage,
                    teacher_v1=tv1_t, teacher_v2=tv2_t, w_distill=wd,
                    vertex_colors=vcols_t if w_color > 0 else None,
                    positions=pos_t,
                    w_color_align=w_color,
                )

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())

        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            log.info(f"pretrain epoch {ep + 1}/{epochs} loss={ep_loss / len(objs):.4f}")

    out = Path(out_path) if out_path else patch_dir / "checkpoints" / "gnn_base.pt"
    save_model(
        out,
        model,
        meta={"epochs": epochs, "patches": len(objs), "w_distill": w_distill, "edge_attr_dim": EDGE_ATTR_DIM},
    )
    log.set("checkpoint", str(out))
    log.done("pretrain")
    log.save()
    return out
