"""Patch extraction for GNN pretraining."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .gnn_model import build_adjacency_list, mesh_to_edge_index


def _classify_patch(k1: np.ndarray, k2: np.ndarray, feature_mask: np.ndarray, boundary_mask: np.ndarray) -> str:
    curv = float(np.abs(k1 - k2).mean())
    if boundary_mask.mean() > 0.2:
        return "hole"
    if feature_mask.mean() > 0.15:
        return "corner"
    if curv > 0.3:
        return "high_curv"
    if curv < 0.05:
        return "flat"
    return "torso"


def extract_patch(
    mesh: trimesh.Trimesh,
    seed: int,
    *,
    target_verts: int = 2048,
    hops: int = 3,
) -> tuple[trimesh.Trimesh, np.ndarray, dict]:
    """BFS expand from seed, return submesh + original vertex indices."""
    n = len(mesh.vertices)
    if seed >= n:
        seed = seed % n
    edge_index = mesh_to_edge_index(mesh)
    adj = build_adjacency_list(edge_index, n)
    active = {seed}
    frontier = {seed}
    for _ in range(hops):
        nxt: set[int] = set()
        for u in frontier:
            for v in adj[u]:
                active.add(v)
                nxt.add(v)
        frontier = nxt
        if len(active) >= target_verts:
            break
    if len(active) > target_verts:
        # trim farthest from seed
        pos = np.asarray(mesh.vertices)
        sp = pos[seed]
        dists = {i: float(np.linalg.norm(pos[i] - sp)) for i in active}
        active = set(sorted(dists, key=dists.get)[:target_verts])

    old_indices = sorted(active)
    remap = {old: i for i, old in enumerate(old_indices)}
    faces = []
    for f in mesh.faces:
        if all(int(v) in remap for v in f):
            faces.append([remap[int(v)] for v in f])
    if not faces:
        raise ValueError(f"empty patch at seed {seed}")
    sub = trimesh.Trimesh(
        vertices=mesh.vertices[old_indices],
        faces=np.array(faces, dtype=np.int64),
        process=True,
    )
    meta = {"seed": seed, "orig_indices": np.array(old_indices, dtype=np.int64)}
    return sub, np.array(old_indices), meta


def build_patches_from_mesh(
    mesh: trimesh.Trimesh,
    out_dir: Path,
    *,
    patch_verts: int = 2048,
    patches_per_mesh: int = 8,
    prefix: str = "patch",
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(mesh.vertices)
    rng = np.random.default_rng(42)
    seeds = rng.choice(n, size=min(patches_per_mesh, n), replace=False)
    paths = []
    for i, seed in enumerate(seeds):
        try:
            sub, _, meta = extract_patch(mesh, int(seed), target_verts=patch_verts)
            p = out_dir / f"{prefix}_{i:05d}.obj"
            sub.export(p)
            np.savez_compressed(
                out_dir / f"{prefix}_{i:05d}_meta.npz",
                seed=int(seed),
                source_verts=n,
            )
            paths.append(p)
        except ValueError:
            continue
    return paths


def build_patches_from_dir(
    input_dir: Path,
    out_dir: Path,
    *,
    patch_verts: int = 2048,
    patches_per_mesh: int = 8,
) -> list[Path]:
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    all_paths: list[Path] = []
    exts = {".obj", ".ply", ".glb", ".stl"}
    files = [f for f in input_dir.rglob("*") if f.suffix.lower() in exts]
    for fi, fp in enumerate(files):
        try:
            mesh = trimesh.load(fp, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            subdir = out_dir / fp.stem
            paths = build_patches_from_mesh(
                mesh, subdir, patch_verts=patch_verts, patches_per_mesh=patches_per_mesh, prefix=f"m{fi}"
            )
            all_paths.extend(paths)
        except Exception:
            continue
    return all_paths
