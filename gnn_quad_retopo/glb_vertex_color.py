"""Direct glTF/GLB COLOR_0 reader (F32_VEC3, U8/U16_VEC3/4, per-vertex or per-corner)."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_COMPONENT = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
_TYPE_DIM = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_INTEGER_COMPONENTS = {5120, 5121, 5122, 5123}


def _parse_glb(path: Path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    if len(data) < 12 or data[:4] != b"glTF":
        raise ValueError(f"not a GLB: {path}")
    off = 12
    jc: dict | None = None
    bin_data = b""
    while off + 8 <= len(data):
        chunk_len = struct.unpack_from("<I", data, off)[0]
        chunk_type = data[off + 4 : off + 8]
        chunk = data[off + 8 : off + 8 + chunk_len]
        if chunk_type == b"JSON":
            jc = json.loads(chunk.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            bin_data = chunk
        off += 8 + chunk_len
    if jc is None:
        raise ValueError(f"GLB JSON chunk missing: {path}")
    return jc, bin_data


def _read_accessor(
    jc: dict,
    bin_data: bytes,
    accessor_index: int,
) -> np.ndarray:
    acc = jc["accessors"][accessor_index]
    bv = jc["bufferViews"][acc["bufferView"]]
    comp_code = acc["componentType"]
    comp_char, comp_size = _COMPONENT[comp_code]
    dim = _TYPE_DIM[acc["type"]]
    count = int(acc["count"])
    start = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    byte_stride = bv.get("byteStride")
    if byte_stride is None or byte_stride == comp_size * dim:
        arr = np.frombuffer(
            bin_data,
            dtype=np.dtype(comp_char),
            count=count * dim,
            offset=start,
        ).reshape(count, dim)
    else:
        rows = []
        for i in range(count):
            row_start = start + i * int(byte_stride)
            row = np.frombuffer(
                bin_data,
                dtype=np.dtype(comp_char),
                count=dim,
                offset=row_start,
            )
            rows.append(row)
        arr = np.stack(rows, axis=0)
    return arr, comp_code


def _normalize_rgb(raw: np.ndarray, comp_code: int) -> np.ndarray:
    rgb = np.asarray(raw, dtype=np.float64)
    if rgb.ndim == 1:
        rgb = rgb.reshape(-1, 1)
    if rgb.shape[1] == 1:
        rgb = np.repeat(rgb, 3, axis=1)
    rgb = rgb[:, :3]
    if comp_code in _INTEGER_COMPONENTS:
        denom = 65535.0 if comp_code == 5123 else 255.0
        rgb = rgb / denom
    return np.clip(rgb, 0.0, 1.0)


def _colors_valid(rgb: np.ndarray, *, min_spread: float = 1e-3) -> bool:
    if rgb.size == 0:
        return False
    return float(np.max(rgb) - np.min(rgb)) >= min_spread


def _corner_colors_to_vertex(
    colors_corner: np.ndarray,
    indices: np.ndarray,
    n_vertices: int,
) -> np.ndarray:
    rgb_v = np.zeros((n_vertices, 3), dtype=np.float64)
    cnt = np.zeros(n_vertices, dtype=np.int32)
    for vi, rgb in zip(indices.reshape(-1), colors_corner):
        rgb_v[int(vi)] += rgb
        cnt[int(vi)] += 1
    mask = cnt > 0
    rgb_v[mask] /= cnt[mask, None]
    return rgb_v.astype(np.float32)


def load_glb_vertex_colors(path: Path | str) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (positions Nx3, rgb Nx3 in [0,1], has_valid_colors) from first mesh primitive."""
    path = Path(path)
    if path.suffix.lower() not in (".glb", ".gltf"):
        return np.empty((0, 3)), np.empty((0, 3)), False
    if path.suffix.lower() == ".gltf":
        return np.empty((0, 3)), np.empty((0, 3)), False

    jc, bin_data = _parse_glb(path)
    meshes = jc.get("meshes") or []
    if not meshes:
        return np.empty((0, 3)), np.empty((0, 3)), False

    prim = meshes[0].get("primitives", [{}])[0]
    attrs = prim.get("attributes") or {}
    if "COLOR_0" not in attrs or "POSITION" not in attrs:
        return np.empty((0, 3)), np.empty((0, 3)), False

    pos_raw, _ = _read_accessor(jc, bin_data, attrs["POSITION"])
    positions = np.asarray(pos_raw, dtype=np.float64)[:, :3]
    col_raw, comp_code = _read_accessor(jc, bin_data, attrs["COLOR_0"])
    rgb_corner = _normalize_rgb(col_raw, comp_code)
    n_pos = len(positions)

    if len(rgb_corner) == n_pos:
        rgb = rgb_corner.astype(np.float32)
        return positions, rgb, _colors_valid(rgb)

    if "indices" in prim and len(rgb_corner) > 0:
        idx_raw, _ = _read_accessor(jc, bin_data, prim["indices"])
        indices = np.asarray(idx_raw, dtype=np.int64).reshape(-1)
        if len(rgb_corner) == len(indices):
            rgb = _corner_colors_to_vertex(rgb_corner, indices, n_pos)
            return positions, rgb, _colors_valid(rgb)

    return np.empty((0, 3)), np.empty((0, 3)), False


def convert_glb_vertex_colors_to_u8_vec4(
    src: Path | str,
    dst: Path | str,
) -> bool:
    """Rewrite GLB COLOR_0 as normalized U8 VEC4 (RGB + A=255). Returns True on success."""
    src, dst = Path(src), Path(dst)
    jc, bin_data = _parse_glb(src)
    meshes = jc.get("meshes") or []
    if not meshes:
        return False
    prim = meshes[0].get("primitives", [{}])[0]
    attrs = prim.get("attributes") or {}
    if "COLOR_0" not in attrs:
        return False

    positions, rgb, has = load_glb_vertex_colors(src)
    if not has:
        return False

    rgba = np.ones((len(rgb), 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)

    bin_list = bytearray(bin_data)
    byte_offset = len(bin_list)
    bin_list.extend(rgba.tobytes())

    buffer_views = jc.get("bufferViews") or []
    new_bv_idx = len(buffer_views)
    buffer_views.append(
        {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": rgba.nbytes,
            "target": 34962,
        }
    )
    jc["bufferViews"] = buffer_views

    accessors = jc.get("accessors") or []
    new_acc_idx = len(accessors)
    accessors.append(
        {
            "bufferView": new_bv_idx,
            "componentType": 5121,
            "count": len(rgba),
            "type": "VEC4",
            "normalized": True,
        }
    )
    jc["accessors"] = accessors
    attrs["COLOR_0"] = new_acc_idx
    prim["attributes"] = attrs

    json_bytes = json.dumps(jc, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_pad
    bin_pad = (4 - (len(bin_list) % 4)) % 4
    bin_list.extend(b"\x00" * bin_pad)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_list)
    header = struct.pack("<4sII", b"glTF", 2, total)
    out = header + struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    out += struct.pack("<I4s", len(bin_list), b"BIN\x00") + bytes(bin_list)
    dst.write_bytes(out)
    return True
