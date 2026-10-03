"""三角形上の最近点（面の重心 KD 木で候補を絞り、GPU があれば torch でまとめて計算）。"""
import numpy as np


def _closest_np(p, a, b, c):
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.maximum(np.einsum("ij,ij->i", n, n), 1e-30)
    ap = p - a
    v = np.einsum("ij,ij->i", np.cross(ap, ac), n) / nn
    w = np.einsum("ij,ij->i", np.cross(ab, ap), n) / nn
    bary = np.clip(np.stack([1 - v - w, v, w], axis=1), 0, None)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
    q = bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:] * c
    return q, bary


def _torch_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return None


def _closest_torch(points, tri, cand, dev):
    import torch
    p = torch.as_tensor(points, dtype=torch.float32, device=dev)
    t = torch.as_tensor(tri[cand], dtype=torch.float32, device=dev)  # (n,k,3,3)
    a, b, c = t[:, :, 0], t[:, :, 1], t[:, :, 2]
    pp = p[:, None, :]
    ab, ac = b - a, c - a
    n = torch.cross(ab, ac, dim=-1)
    nn = (n * n).sum(-1).clamp_min(1e-30)
    ap = pp - a
    v = (torch.cross(ap, ac, dim=-1) * n).sum(-1) / nn
    w = (torch.cross(ab, ap, dim=-1) * n).sum(-1) / nn
    bary = torch.stack([1 - v - w, v, w], -1).clamp_min(0)
    bary = bary / bary.sum(-1, keepdim=True).clamp_min(1e-12)
    q = bary[..., :1] * a + bary[..., 1:2] * b + bary[..., 2:] * c
    d = ((q - pp) ** 2).sum(-1)
    j = d.argmin(1)
    ar = torch.arange(len(p), device=dev)
    return j.cpu().numpy(), bary[ar, j].double().cpu().numpy()


def closest_on_faces(points, tri, tree, k=8, chunk=400_000):
    """points (n,3) と三角形 tri (m,3,3)。最近の面番号と重心座標を返す。"""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    best_f = np.zeros(n, dtype=np.int64)
    best_w = np.zeros((n, 3))
    k = int(min(k, len(tri)))
    dev = _torch_device()
    for s in range(0, n, chunk):
        pts = points[s:s + chunk]
        _, cand = tree.query(pts, k=k, workers=-1)
        cand = np.asarray(cand).reshape(len(pts), k)
        if dev is not None:
            j, w = _closest_torch(pts, tri, cand, dev)
            best_f[s:s + len(pts)] = cand[np.arange(len(pts)), j]
            best_w[s:s + len(pts)] = w
            continue
        bd = np.full(len(pts), np.inf)
        for jj in range(k):
            fi = cand[:, jj]
            q, w = _closest_np(pts, tri[fi, 0], tri[fi, 1], tri[fi, 2])
            d = np.einsum("ij,ij->i", q - pts, q - pts)
            m = d < bd
            bd[m] = d[m]
            best_f[s:s + len(pts)][m] = fi[m]
            best_w[s:s + len(pts)][m] = w[m]
    return best_f, best_w
