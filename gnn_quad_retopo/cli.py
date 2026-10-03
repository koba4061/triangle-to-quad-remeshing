"""CLI."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh

from .auto_tune import AUTO_BATCH, detect_runtime, save_auto_tune_report
from .config import DEFAULT_TEST_MESH, RUNS_DIR
from .debug_log import PipelineTracker
from .pipeline_log import install_pipeline_log
from .run_layout import (
    RUN_SHARED_DIR,
    SHARED_BASE_CHECKPOINT,
    RunLayout,
    default_pretrained_checkpoint,
    find_cleaned_mesh,
    find_cross_field,
    resolve_out_dir,
    write_layout_meta,
    write_run_meta,
    layout_from_run_dir,
    load_run_meta,
    publish_final_retopo,
)
from .cross_field import export_field_lines, run_field_stage
from .field_eval import evaluate_cross_field, save_field_metrics
from .gnn_model import save_field, train_cross_field
from .im_extractor import extract_quads_unified, im_status
from .im_teacher import build_teachers_for_dir
from .mesh_budget import count_mesh_faces, resolve_target_quads
from .mesh_cleaner import clean_mesh
from .mesh_io import load_mesh_yup
from .patch_dataset import build_patches_from_dir, build_patches_from_mesh
from .pretrain import pretrain_on_patches
from .quad_extractor import extract_quads, extract_quads_from_mesh
from .synthetic import export_primitive, make_primitive


def _add_extractor_args(p):
    p.add_argument("--extractor", default="python", choices=["python", "im"], help="quad extractor backend")
    return p


def _add_im_quality_args(p):
    p.add_argument(
        "--shrinkwrap",
        action="store_true",
        default=True,
        help="snap IM quads onto cleaned mesh surface (default: on)",
    )
    p.add_argument("--no-shrinkwrap", action="store_false", dest="shrinkwrap")
    p.add_argument(
        "--shrinkwrap-passes",
        type=int,
        default=3,
        help="surface snap iterations onto cleaned_mesh (complex models: 3-5)",
    )
    p.add_argument("--quad-density", type=float, default=1.0, help="IM density scale (0=auto, >1=finer)")
    p.add_argument(
        "--im-lite-fallback",
        action="store_true",
        help="allow im_lite Python fallback when native IM fails (small meshes only)",
    )
    p.add_argument("--im-crease-angle", type=float, default=75.0, help="IM crease angle (lower=more creases; 75=kitbash flats)")
    p.add_argument(
        "--im-smooth-pos-iters",
        type=int,
        default=2,
        help="IM position-field smoothing (1-2: fast+quality; 0 skips IM grid align; 5 slow)",
    )
    p.add_argument(
        "--im-smooth-orient-iters",
        type=int,
        default=4,
        help="IM orientation smoothing before extract (3-4 helps flat kitbash panels)",
    )
    p.add_argument(
        "--quality-retries",
        type=int,
        default=0,
        help="IM quality-driven re-extract attempts after baseline (0=off)",
    )
    p.add_argument(
        "--quality-pass",
        type=float,
        default=None,
        help="quality score pass threshold override (lower is better)",
    )
    return p


def _im_extract_kwargs(args) -> dict:
    vcg = getattr(args, "vertex_color_guide", True)
    return dict(
        shrinkwrap=getattr(args, "shrinkwrap", True),
        shrinkwrap_passes=getattr(args, "shrinkwrap_passes", 3),
        quad_density=getattr(args, "quad_density", 1.0),
        crease_angle=getattr(args, "im_crease_angle", 75.0),
        smooth_orient_iters=getattr(args, "im_smooth_orient_iters", 4),
        smooth_pos_iters=getattr(args, "im_smooth_pos_iters", 2),
        bridge_seams=getattr(args, "bridge_seams", False),
        allow_im_lite_fallback=getattr(args, "im_lite_fallback", False),
        vertex_color_export=vcg,
        quality_retries=getattr(args, "quality_retries", 0),
        quality_pass=getattr(args, "quality_pass", None),
    )


def _add_vertex_color_args(p):
    p.add_argument(
        "--vertex-color-guide",
        action="store_true",
        default=True,
        help="vertex color boundary guide when valid colors exist (default: on)",
    )
    p.add_argument(
        "--no-vertex-color-guide",
        action="store_false",
        dest="vertex_color_guide",
        help="disable vertex color guide and RGB OBJ export",
    )
    return p


def _add_auto_arg(p):
    p.add_argument(
        "--auto",
        action="store_true",
        help="GPU/メッシュ規模から device・batch-sizeを自動設定",
    )
    return p


def _add_sdf_args(p):
    p.add_argument(
        "--sdf-remesh",
        action="store_true",
        default=False,
        help="enable prep SDF remesh (default backend: Blender OpenVDB)",
    )
    p.add_argument(
        "--sdf-backend",
        default="blender",
        choices=["blender", "neural"],
        help="SDF remesh backend: blender=OpenVDB GN (default), neural=Instant-NGP",
    )
    p.add_argument(
        "--blender-voxel-size",
        type=float,
        default=0.0,
        help="Blender Mesh-to-SDF voxel size in meters (0=auto from mesh bounds)",
    )
    p.add_argument(
        "--blender-adaptivity",
        type=float,
        default=-1.0,
        help="Proxy SDF adaptivity (0-1, -1=0.08; lower = sharper)",
    )
    p.add_argument(
        "--blender-canonical-adaptivity",
        type=float,
        default=-1.0,
        help="Canonical SDF adaptivity for shrinkwrap (0-1, -1=0.06; lower = sharper)",
    )
    p.add_argument(
        "--blender-band-width",
        type=int,
        default=0,
        help="Blender OpenVDB band width (0=auto)",
    )
    p.add_argument(
        "--sdf-res",
        type=int,
        default=0,
        help="Marching Cubes grid resolution (0=auto from mesh/GPU, default: 0)",
    )
    p.add_argument(
        "--sdf-method",
        default="dual_contouring",
        choices=["marching_cubes", "dual_contouring"],
        help="SDF-to-mesh reconstruction method (default: dual_contouring)",
    )
    p.add_argument(
        "--sdf-query-faces",
        type=int,
        default=0,
        help="max faces for SDF signed-distance query (0=auto, keep detail)",
    )
    p.add_argument(
        "--no-sdf-simplify",
        action="store_true",
        help="disable SDF pre-simplification (use full input for distance query)",
    )
    p.add_argument(
        "--gnn-proxy-as-canonical",
        action="store_true",
        default=False,
        help="skip canonical SDF; single proxy SDF for GNN and shrinkwrap",
    )
    p.add_argument(
        "--no-blender-persistent",
        action="store_true",
        help="disable single-session Blender SDF pipeline (legacy multi-launch)",
    )
    return p


def _ensure_out(
    args,
    *,
    input_path: Path | str | None = None,
    mesh_path: Path | str | None = None,
    target_quads: int | None = None,
) -> Path:
    tq = target_quads
    if tq is None:
        tq = int(getattr(args, "target_quads", 0) or 0)
    out = resolve_out_dir(
        getattr(args, "out", None),
        input_path=input_path,
        mesh_path=mesh_path,
        target_quads=tq,
    )
    args.out = str(out)
    return out


def _start_run_logging(out: Path, meta: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log_p = install_pipeline_log(out)
    print(f"[run] output={out} log={log_p}", flush=True)
    if meta:
        write_run_meta(out, meta)


def _default_checkpoint(args) -> None:
    if not getattr(args, "checkpoint", None):
        ck = default_pretrained_checkpoint()
        if ck.is_file():
            args.checkpoint = str(ck)


def _apply_auto_run(args) -> None:
    if not getattr(args, "auto", False):
        return
    rt = detect_runtime(args.device)
    if args.device is None:
        args.device = rt.device
    args.batch_size = AUTO_BATCH
    out = Path(args.out)
    plan: dict = {"runtime": asdict(rt), "phase": "run_start"}
    save_auto_tune_report(out, plan)


def _apply_auto_train(args) -> None:
    if not getattr(args, "auto", False):
        return
    rt = detect_runtime(args.device)
    if args.device is None:
        args.device = rt.device
    args.batch_size = AUTO_BATCH


def cmd_clean(args):
    inp = Path(args.input)
    out = _ensure_out(args, input_path=inp)
    _start_run_logging(out, {"input": str(inp.resolve()), "cmd": "clean"})
    clean_mesh(inp, out, light=getattr(args, "light_clean", False))


def cmd_regularize(args):
    from .neural_regularizer import neural_regularize_mesh
    neural_regularize_mesh(
        Path(args.input), Path(args.output),
        iters=args.iters, lr=args.lr,
        w_chamfer=args.w_chamfer, w_laplacian=args.w_laplacian, w_edge=args.w_edge,
        sdf_remesh=args.sdf_remesh,
        sdf_backend=args.sdf_backend,
        sdf_res=args.sdf_res,
        sdf_method=args.sdf_method,
        sdf_query_faces=args.sdf_query_faces,
        no_sdf_simplify=args.no_sdf_simplify,
        blender_voxel_size=args.blender_voxel_size,
        blender_adaptivity=args.blender_adaptivity,
        blender_canonical_adaptivity=args.blender_canonical_adaptivity,
        blender_band_width=args.blender_band_width,
        gnn_proxy_as_canonical=getattr(args, "gnn_proxy_as_canonical", False),
        blender_persistent=not getattr(args, "no_blender_persistent", False),
    )


def cmd_field(args):
    out = _ensure_out(args)
    mesh = load_mesh_yup(find_cleaned_mesh(out))
    run_field_stage(mesh, out, stride=args.stride, scale=args.arrow_scale,
                    up_axis=args.up_axis, rotate_axis=args.rotate_axis, rotate_deg=args.rotate_deg,
                    use_curvature=args.curvature)


def cmd_train(args):
    _apply_auto_train(args)
    mesh_path = Path(args.mesh) if args.mesh else None
    out = _ensure_out(args, mesh_path=mesh_path, input_path=None)
    if mesh_path is None:
        mesh_path = find_cleaned_mesh(out)
    _start_run_logging(out, {"mesh": str(mesh_path.resolve()), "cmd": "train"})
    _default_checkpoint(args)
    from .run_layout import layout_from_run_dir

    layout = layout_from_run_dir(out)
    layout.ensure_dirs()
    mesh = load_mesh_yup(mesh_path)
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    _, field = train_cross_field(
        mesh, epochs=args.epochs, lr=args.lr, device=args.device,
        up_axis=args.up_axis, rotate_axis=args.rotate_axis, rotate_deg=args.rotate_deg,
        patience=args.patience, min_delta=args.min_delta, use_curvature=args.curvature,
        feature_dir=layout.field_dir, batch_size=args.batch_size, loss_stage=args.loss_stage,
        checkpoint=ckpt, w_distill=args.w_distill,
        save_checkpoint=layout.gnn_checkpoint(),
        vertex_color_guide=getattr(args, "vertex_color_guide", True),
    )
    save_field(layout.cross_field_npy(), field)
    if args.stride:
        bbox = float(np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0)))
        export_field_lines(
            mesh, field, layout.cross_field_lines(), stride=args.stride, scale=bbox * 0.05,
        )
    print(f"saved {layout.cross_field_npy()}")


def cmd_extract(args):
    from .mesh_io import load_mesh_yup
    mesh_p = Path(args.mesh)
    out = _ensure_out(args, mesh_path=mesh_p)
    layout = layout_from_run_dir(out)
    layout.ensure_dirs()
    existing_meta = load_run_meta(layout.root)
    _start_run_logging(out, None)
    if not existing_meta:
        write_layout_meta(layout, {"mesh": str(mesh_p.resolve()), "cmd": "extract"})
    try:
        mesh = load_mesh_yup(mesh_p)
        n_in = len(mesh.faces)
        meta_tq = 0
        if int(args.target_quads or 0) <= 0:
            try:
                meta_tq = int(existing_meta.get("target_quads", 0) or 0)
            except Exception:
                meta_tq = 0
        tq = meta_tq if meta_tq > 0 else resolve_target_quads(args.target_quads, input_faces=n_in)
        if tq != args.target_quads:
            src = "run.json" if meta_tq > 0 else f"mesh {n_in:,} tris"
            print(f"[extract] auto target_quads={tq:,} from {src}", flush=True)
            args.target_quads = tq
        field = np.load(args.field)
        if field.shape[-1] == 6:
            field = field.reshape(len(field), 2, 3)
        from .run_layout import find_canonical_mesh

        try:
            canon_mesh = load_mesh_yup(find_canonical_mesh(layout.root))
        except FileNotFoundError:
            canon_mesh = mesh
        # Prefer original input mesh for vertex color export (canonical OBJ may drop colors).
        color_ref_mesh = None
        try:
            inp = existing_meta.get("input") if existing_meta else None
            if inp:
                color_ref_mesh = trimesh.load(Path(inp), force="mesh", process=False)
                if isinstance(color_ref_mesh, trimesh.Scene):
                    color_ref_mesh = color_ref_mesh.dump(concatenate=True)
        except Exception:
            color_ref_mesh = None
        im_kw = _im_extract_kwargs(args)
        im_kw["shrinkwrap_reference_mesh"] = canon_mesh
        im_kw["color_reference_mesh"] = color_ref_mesh or canon_mesh
        if inp:
            im_kw["color_reference_path"] = Path(inp)
        path = extract_quads_unified(
            mesh, field, layout.extract_dir,
            extractor=args.extractor,
            target_quads=args.target_quads, grid_h=args.grid_h,
            **im_kw,
        )
        final = publish_final_retopo(path, layout)
        print(f"[extract] done -> {final}", flush=True)
    except Exception as e:
        print(f"\n[extract] FAILED: {e}\n", flush=True)
        raise


def cmd_synthetic(args):
    out = Path(args.out)
    obj, fld = export_primitive(args.type, out)
    mesh = load_mesh_yup(obj)
    field = np.load(fld)
    if field.shape[-1] == 6:
        field = field.reshape(len(field), 2, 3)
    extract_quads_unified(
        mesh, field, out,
        extractor=args.extractor,
        target_quads=args.target_quads,
        bridge_seams=args.bridge_seams,
    )


def cmd_evaluate_field(args):
    mesh = load_mesh_yup(Path(args.mesh))
    field = np.load(args.field)
    teacher = np.load(args.teacher) if args.teacher else None
    metrics = evaluate_cross_field(
        mesh, field,
        features_path=Path(args.mesh).parent,
        teacher_field=teacher,
    )
    out = Path(args.out) if args.out else Path(args.mesh).parent / "field_metrics.json"
    save_field_metrics(metrics, out)
    print(f"field metrics -> {out}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def cmd_build_patches(args):
    inp = Path(args.input_dir)
    out = Path(args.out)
    if inp.is_file():
        mesh = trimesh.load(inp, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        paths = build_patches_from_mesh(mesh, out, patch_verts=args.patch_verts, patches_per_mesh=args.patches_per_mesh)
    else:
        paths = build_patches_from_dir(inp, out, patch_verts=args.patch_verts, patches_per_mesh=args.patches_per_mesh)
    print(f"built {len(paths)} patches -> {out}")


def cmd_build_teachers(args):
    paths = build_teachers_for_dir(Path(args.patch_dir), workers=args.workers)
    print(f"built {len(paths)} teachers ({im_status()})")


def cmd_pretrain(args):
    RUN_SHARED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else SHARED_BASE_CHECKPOINT
    ckpt = pretrain_on_patches(
        Path(args.patch_dir), epochs=args.epochs, lr=args.lr, device=args.device,
        batch_size=args.batch_size, out_path=out_path,
        w_distill=args.w_distill, loss_stage=args.loss_stage,
    )
    print(f"pretrain checkpoint -> {ckpt}")



def cmd_run(args):

    inp = Path(args.input)
    print(f"[run] loading {inp}", flush=True)
    n_in = count_mesh_faces(inp)
    tq = resolve_target_quads(args.target_quads, input_faces=n_in)
    if tq != args.target_quads:
        print(
            f"[run] auto target_quads={tq:,} from input {n_in:,} tris "
            f"(~{tq / max(n_in, 1):.0%} of input; use --target-quads N to override)",
            flush=True,
        )
    args.target_quads = tq
    layout = RunLayout.create(inp, out_dir=getattr(args, "out", None), target_quads=tq)
    layout.ensure_dirs()
    args.out = str(layout.root)
    out = layout.root
    _apply_auto_run(args)
    _default_checkpoint(args)
    write_layout_meta(
        layout,
        {
            "input": str(inp.resolve()),
            "cmd": "run",
            "checkpoint": args.checkpoint,
            "input_faces": n_in,
            "target_quads": tq,
        },
    )
    _start_run_logging(out, None)
    pipe = PipelineTracker(out)
    stages = ["neural-regularizer", "mesh-cleaner", "features.npz"]
    if args.epochs > 0:
        stages.append(f"gnn-field ({args.epochs} ep)")
    else:
        stages.append("stable-field")
    stages.append(f"quad-extractor ({args.extractor})")
    pipe.configure(stages)

    # Stage 1: Neural Regularization & Smart Pre-Decimation
    pipe.enter("neural-regularizer")
    smoothed_mesh_path = layout.smoothed_mesh()
    from .neural_regularizer import neural_regularize_mesh
    reg_iters = 30 if getattr(args, "reg_smooth", False) else args.reg_iters
    neural_regularize_mesh(
        inp, smoothed_mesh_path,
        iters=reg_iters, lr=args.reg_lr,
        w_chamfer=args.w_chamfer, w_laplacian=args.w_laplacian, w_edge=args.w_edge,
        decimate=args.reg_decimate,
        target_quads=args.target_quads,
        max_decimate_faces=args.reg_max_faces,
        sdf_remesh=args.sdf_remesh,
        sdf_backend=args.sdf_backend,
        sdf_res=args.sdf_res,
        sdf_method=args.sdf_method,
        sdf_query_faces=args.sdf_query_faces,
        no_sdf_simplify=args.no_sdf_simplify,
        blender_voxel_size=args.blender_voxel_size,
        blender_adaptivity=args.blender_adaptivity,
        blender_canonical_adaptivity=args.blender_canonical_adaptivity,
        blender_band_width=args.blender_band_width,
        gnn_proxy_as_canonical=getattr(args, "gnn_proxy_as_canonical", False),
        blender_persistent=not getattr(args, "no_blender_persistent", False),
    )

    # Stage 2: Mesh Cleaning on the smoothed result
    pipe.enter("mesh-cleaner")
    mesh, _ = clean_mesh(smoothed_mesh_path, layout, light=getattr(args, "light_clean", False))

    from .mesh_io import load_mesh_yup
    from .run_layout import find_canonical_mesh

    try:
        canon_mesh = load_mesh_yup(find_canonical_mesh(layout.root))
        if len(canon_mesh.vertices) != len(mesh.vertices):
            print(
                f"[run] GNN/IM field mesh={len(mesh.vertices):,} verts | "
                f"shrinkwrap canonical={len(canon_mesh.vertices):,} verts",
                flush=True,
            )
    except FileNotFoundError:
        canon_mesh = mesh

    if args.epochs > 0:
        pipe.enter(f"gnn-field ({args.epochs} epochs)")
        ckpt = Path(args.checkpoint) if args.checkpoint else None
        _, field = train_cross_field(
            mesh, epochs=args.epochs, lr=args.lr, device=args.device,
            patience=args.patience, min_delta=args.min_delta, use_curvature=args.curvature,
            feature_dir=layout.field_dir, batch_size=args.batch_size, loss_stage=args.loss_stage,
            checkpoint=ckpt, w_distill=args.w_distill,
            save_checkpoint=layout.gnn_checkpoint(),
            vertex_color_guide=getattr(args, "vertex_color_guide", True),
        )
        save_field(layout.cross_field_npy(), field)
        bbox = float(np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0)))
        export_field_lines(
            mesh, field, layout.cross_field_lines(), stride=args.field_stride, scale=bbox * 0.05,
        )
    else:
        pipe.enter("stable-field")
        run_field_stage(mesh, layout.field_dir, stride=args.field_stride, use_curvature=args.curvature)

    try:
        pipe.enter(f"quad-extractor ({args.extractor})")
        field = np.load(layout.cross_field_npy())
        if field.shape[-1] == 6:
            field = field.reshape(len(field), 2, 3)
        im_kw = _im_extract_kwargs(args)
        im_kw["shrinkwrap_reference_mesh"] = canon_mesh
        # Prefer original input mesh for vertex color export (canonical OBJ may drop colors).
        color_ref_mesh = None
        try:
            color_ref_mesh = trimesh.load(inp, force="mesh", process=False)
            if isinstance(color_ref_mesh, trimesh.Scene):
                color_ref_mesh = color_ref_mesh.dump(concatenate=True)
        except Exception:
            color_ref_mesh = None
        im_kw["color_reference_mesh"] = color_ref_mesh or canon_mesh
        if inp:
            im_kw["color_reference_path"] = Path(inp)
        path = extract_quads_unified(
            mesh, field, layout.extract_dir,
            extractor=args.extractor,
            target_quads=args.target_quads,
            **im_kw,
        )
        final = publish_final_retopo(path, layout)
        print(f"[run] done -> {final}", flush=True)
        pipe.finish()
    except Exception as e:
        pipe.fail(str(e))
        raise


def cmd_status(args):
    from .run_layout import layout_from_run_dir

    inp = getattr(args, "input", None)
    out = _ensure_out(args, input_path=Path(inp) if inp else None)
    lay = layout_from_run_dir(out)
    checks = [
        ("run.json", lay.run_meta()),
        ("pipeline.log", lay.pipeline_log()),
        ("auto_tune.json", out / "auto_tune.json"),
        ("canonical.json", lay.canonical_manifest()),
        ("sdf", lay.sdf_watertight_mesh()),
        ("cleaned", lay.cleaned_mesh()),
        ("features", lay.features_npz()),
        ("cross_field", lay.cross_field_npy()),
        ("export", lay.export_retopo_quad()),
    ]
    for label, p in checks:
        print(f"{'OK' if p.exists() else '--'} {label}: {p.relative_to(out)}")
    print(f"IM bridge: {im_status()}")


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="gnn_quad_retopo")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clean")
    c.add_argument("input", nargs="?", default=str(DEFAULT_TEST_MESH))
    c.add_argument("--out", default=None, help="省略時 runs/<入力ファイル名>/")
    c.add_argument(
        "--light-clean",
        action="store_true",
        help="light IM-prep repair only (skip heavy vertex merge; still fixes non-manifold)",
    )
    c.set_defaults(func=cmd_clean)

    reg = sub.add_parser("regularize", help="Neural mesh regularization using PyTorch3D (Chamfer + Laplacian + Edge loss)")
    reg.add_argument("input", help="input raw bumpy mesh .obj")
    reg.add_argument("output", help="output smoothed mesh .obj")
    reg.add_argument("--iters", type=int, default=100, help="number of optimization iterations")
    reg.add_argument("--lr", type=float, default=0.01, help="learning rate")
    reg.add_argument("--w-chamfer", type=float, default=1.0, help="Chamfer loss weight")
    reg.add_argument("--w-laplacian", type=float, default=0.5, help="Laplacian smoothing weight")
    reg.add_argument("--w-edge", type=float, default=0.1, help="Edge length uniformity weight")
    _add_sdf_args(reg)
    reg.set_defaults(func=cmd_regularize)

    f = sub.add_parser("field")
    f.add_argument("--out", default=None, help="省略時 runs/final（clean 済みディレクトリを指定推奨）")
    f.add_argument("--stride", type=int, default=120)
    f.add_argument("--arrow-scale", type=float, default=None)
    f.add_argument("--up-axis", default="Y")
    f.add_argument("--rotate-axis", default=None)
    f.add_argument("--rotate-deg", type=float, default=0)
    f.add_argument("--curvature", action="store_true", help="align cross-field with principal curvatures")
    f.set_defaults(func=cmd_field)

    t = sub.add_parser("train")
    t.add_argument("mesh", nargs="?", default=None)
    t.add_argument("--out", default=None, help="省略時 runs/<meshの親または入力名>/")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--lr", type=float, default=0.005)
    t.add_argument("--device", default=None)
    t.add_argument("--patience", type=int, default=80)
    t.add_argument("--min-delta", type=float, default=1e-4)
    t.add_argument("--stride", type=int, default=120)
    t.add_argument("--up-axis", default="Y")
    t.add_argument("--rotate-axis", default=None)
    t.add_argument("--rotate-deg", type=float, default=0)
    t.add_argument("--curvature", action="store_true")
    t.add_argument("--batch-size", type=int, default=0)
    _add_auto_arg(t)
    t.add_argument("--loss-stage", default="phased", choices=["phased", "full"])
    t.add_argument("--checkpoint", default=None, help="省略時 runs/_shared/gnn_base.pt または checkpoints/gnn_base.pt")
    t.add_argument("--w-distill", type=float, default=0.0)
    _add_vertex_color_args(t)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("extract")
    e.add_argument("mesh")
    e.add_argument("field")
    e.add_argument("--out", default=".")
    e.add_argument("--target-quads", type=int, default=300000)
    e.add_argument("--grid-h", type=float, default=None)
    e.add_argument("--bridge-seams", action="store_true", default=False)
    _add_extractor_args(e)
    _add_im_quality_args(e)
    _add_vertex_color_args(e)
    e.set_defaults(func=cmd_extract)

    s = sub.add_parser("synthetic")
    s.add_argument("--type", default="cylinder", choices=["plane", "cylinder", "sphere", "torus"])
    s.add_argument("--out", default=str(RUNS_DIR / "test_syn"))
    s.add_argument("--target-quads", type=int, default=5000)
    s.add_argument("--bridge-seams", action="store_true", default=False)
    _add_extractor_args(s)
    s.set_defaults(func=cmd_synthetic)

    ev = sub.add_parser("evaluate-field")
    ev.add_argument("--mesh", required=True)
    ev.add_argument("--field", required=True)
    ev.add_argument("--teacher", default=None, help="optional teacher field .npy")
    ev.add_argument("--out", default=None)
    ev.set_defaults(func=cmd_evaluate_field)

    bp = sub.add_parser("build-patches")
    bp.add_argument("--input-dir", required=True)
    bp.add_argument("--out", required=True)
    bp.add_argument("--patch-verts", type=int, default=2048)
    bp.add_argument("--patches-per-mesh", type=int, default=8)
    bp.set_defaults(func=cmd_build_patches)

    bt = sub.add_parser("build-teachers")
    bt.add_argument("--patch-dir", required=True)
    bt.add_argument("--workers", type=int, default=1)
    bt.set_defaults(func=cmd_build_teachers)

    pt = sub.add_parser("pretrain")
    pt.add_argument("--patch-dir", required=True)
    pt.add_argument("--out", default=None, help="省略時 runs/_shared/gnn_base.pt")
    pt.add_argument("--epochs", type=int, default=100)
    pt.add_argument("--lr", type=float, default=0.005)
    pt.add_argument("--device", default=None)
    pt.add_argument("--batch-size", type=int, default=2048)
    pt.add_argument("--w-distill", type=float, default=1.0)
    pt.add_argument("--loss-stage", default="phased", choices=["phased", "full"])
    pt.set_defaults(func=cmd_pretrain)

    r = sub.add_parser("run")
    r.add_argument("input", nargs="?", default=str(DEFAULT_TEST_MESH))
    r.add_argument("--out", default=None, help="省略時 runs/<入力ファイル名>/")
    r.add_argument(
        "--target-quads",
        type=int,
        default=0,
        help="final IM quad budget (0=auto from input triangle count, ~input/2.6)",
    )
    r.add_argument("--field-stride", type=int, default=120)
    r.add_argument("--epochs", type=int, default=120)
    r.add_argument("--lr", type=float, default=0.005)
    r.add_argument("--device", default=None)
    r.add_argument("--patience", type=int, default=20)
    r.add_argument("--min-delta", type=float, default=1e-4)
    r.add_argument("--curvature", action="store_true")
    r.add_argument("--batch-size", type=int, default=0)
    _add_auto_arg(r)
    r.add_argument("--loss-stage", default="phased", choices=["phased", "full"])
    r.add_argument("--checkpoint", default=None, help="省略時共有 pretrain → 学習後は同フォルダに gnn_checkpoint.pt")
    r.add_argument("--w-distill", type=float, default=0.0)
    r.add_argument("--bridge-seams", action="store_true", default=False)
    _add_vertex_color_args(r)

    # Preprocessing / Neural Regularizer & Smart Decimation options
    r.add_argument("--reg-decimate", action="store_true", default=True, help="auto pre-decimation from input face count + --target-quads")
    r.add_argument("--no-reg-decimate", action="store_false", dest="reg_decimate", help="disable pre-decimation (keep full input detail)")
    r.add_argument(
        "--reg-max-faces",
        type=int,
        default=None,
        help="SDF/decimate face cap (default: auto ≈ input triangle count for detail)",
    )
    r.add_argument("--reg-iters", type=int, default=0, help="PyTorch3D smooth iterations (0=decimate only, keeps flat faces)")
    r.add_argument("--reg-smooth", action="store_true", help="enable Laplacian smoothing (30 iters, w_laplacian=0.05)")
    r.add_argument("--reg-lr", type=float, default=0.005, help="regularization learning rate")
    r.add_argument("--w-chamfer", type=float, default=1.0, help="Chamfer loss weight")
    r.add_argument("--w-laplacian", type=float, default=0.05, help="Laplacian weight (only if --reg-smooth or reg-iters>0)")
    r.add_argument("--w-edge", type=float, default=0.1, help="Edge length uniformity weight")
    r.add_argument(
        "--light-clean",
        action="store_true",
        help="light IM-prep repair in mesh-cleaner (minimal merge; still fixes non-manifold)",
    )
    _add_sdf_args(r)

    _add_extractor_args(r)
    _add_im_quality_args(r)
    r.set_defaults(func=cmd_run)

    st = sub.add_parser("status")
    st.add_argument("input", nargs="?", default=None, help="省略時 --out のみ使用")
    st.add_argument("--out", default=None)
    st.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
