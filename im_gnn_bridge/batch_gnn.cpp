/*
 * batch_gnn.cpp -- GNN cross-field injection for Instant Meshes batch pipeline
 */
#include "batch_gnn.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <chrono>
#include <cstdio>
#include <limits>
#include <thread>
#include <tuple>

namespace py = pybind11;
using BatchGnnParams = BatchGnn::Params;

#ifdef HAS_INSTANT_MESHES

#include "meshio.h"
#include "dedge.h"
#include "subdivide.h"
#include "meshstats.h"
#include "hierarchy.h"
#include "field.h"
#include "normal.h"
#include "extract.h"
#include "bvh.h"

int nprocs = -1;

static void log_im(const char *msg) {
    std::fprintf(stderr, "[im] %s\n", msg);
    std::fflush(stderr);
}

static void wait_with_progress(Optimizer &optimizer, const char *phase) {
    auto t0 = std::chrono::steady_clock::now();
    auto next = t0;
    std::fprintf(stderr, "[im] %s\n", phase);
    std::fflush(stderr);
    while (optimizer.active()) {
        auto now = std::chrono::steady_clock::now();
        if (now >= next) {
            double sec = std::chrono::duration<double>(now - t0).count();
            std::fprintf(stderr, "[im] %s %.0fs level=%d %.0f%%\n",
                         phase, sec, optimizer.level(),
                         double(optimizer.progress()) * 100.0);
            std::fflush(stderr);
            next = now + std::chrono::seconds(5);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    optimizer.wait();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[im] %s done %.0fs\n", phase, sec);
    std::fflush(stderr);
}

static MatrixXf numpy_vertices_to_eigen(py::array_t<double> vertices) {
    auto v = vertices.unchecked<2>();
    MatrixXf V(3, v.shape(0));
    for (py::ssize_t i = 0; i < v.shape(0); ++i)
        for (int j = 0; j < 3; ++j)
            V(j, i) = (Float)v(i, j);
    return V;
}

static MatrixXu numpy_faces_to_eigen(py::array_t<int> faces) {
    auto f = faces.unchecked<2>();
    MatrixXu F(3, f.shape(0));
    for (py::ssize_t i = 0; i < f.shape(0); ++i)
        for (int j = 0; j < 3; ++j)
            F(j, i) = (uint32_t)f(i, j);
    return F;
}

static py::array_t<double> eigen_matrix_to_numpy_vertices(const MatrixXf &M) {
    py::array_t<double> out({(py::ssize_t)M.cols(), (py::ssize_t)3});
    auto buf = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < M.cols(); ++i)
        for (int j = 0; j < 3; ++j)
            buf(i, j) = (double)M(j, i);
    return out;
}

static py::array_t<int> eigen_faces_to_numpy(const MatrixXu &F) {
    const int rows = (int)F.rows();
    const py::ssize_t cols = (py::ssize_t)F.cols();
    const int out_cols = rows >= 4 ? 4 : rows;
    py::array_t<int> out({cols, (py::ssize_t)out_cols});
    auto buf = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < cols; ++i) {
        for (int j = 0; j < out_cols; ++j)
            buf(i, j) = (int)F(j, i);
    }
    return out;
}

static void setup_hierarchy(
    MatrixXu &F, MatrixXf &V, MatrixXf &N, VectorXf &A,
    MultiResolutionHierarchy &mRes, BVH *&bvh, MeshStats &stats,
    const BatchGnnParams &params, Float &scale_out) {
    bool pointcloud = F.size() == 0;
    if (pointcloud)
        throw std::runtime_error("point clouds not supported in GNN bridge");

    stats = compute_mesh_stats(F, V, params.deterministic);
    VectorXu V2E, E2E;
    VectorXb boundary, nonManifold;
    Float scale = params.scale;

    if (scale < 0 && params.vertex_count < 0 && params.face_count < 0) {
        scale = std::sqrt(stats.mSurfaceArea / (Float)std::max(1, (int)(V.cols() / 16)));
    } else if (scale > 0) {
        /* keep */
    } else if (params.face_count > 0) {
        Float face_area = stats.mSurfaceArea / params.face_count;
        scale = std::sqrt(face_area);
    } else if (params.vertex_count > 0) {
        Float face_area = stats.mSurfaceArea / params.vertex_count;
        scale = std::sqrt(face_area);
    } else {
        scale = std::sqrt(stats.mSurfaceArea / (Float)std::max(1, (int)(V.cols() / 16)));
    }
    scale_out = scale;

    // Split edges longer than half the target edge so the grid can land.
    // scale comes from the mesh this solve receives, so a uniform mesh
    // is split about once.
    Float edge_limit = scale > 0 ? scale / 2.f : (Float)stats.mAverageEdgeLength * 2.f;
    if (stats.mMaximumEdgeLength > edge_limit) {
        std::fprintf(stderr,
                     "[im] subdivide edges longer than %.6g (avg %.6g, scale %.6g, max %.6g)\n",
                     edge_limit, stats.mAverageEdgeLength, scale, stats.mMaximumEdgeLength);
        std::fflush(stderr);
        build_dedge(F, V, V2E, E2E, boundary, nonManifold);
        subdivide(F, V, V2E, E2E, boundary, nonManifold, edge_limit, params.deterministic);
    } else {
        std::fprintf(stderr,
                     "[im] keep input resolution (verts=%u, max %.6g, limit %.6g)\n",
                     (unsigned)V.cols(), stats.mMaximumEdgeLength, edge_limit);
        std::fflush(stderr);
    }

    build_dedge(F, V, V2E, E2E, boundary, nonManifold);
    AdjacencyMatrix adj = generate_adjacency_matrix_uniform(F, V2E, E2E, nonManifold);

    if (params.crease_angle >= 0) {
        std::set<uint32_t> crease_in;
        generate_crease_normals(F, V, V2E, E2E, boundary, nonManifold, params.crease_angle, N, crease_in);
    } else {
        generate_smooth_normals(F, V, V2E, E2E, nonManifold, N);
    }

    compute_dual_vertex_areas(F, V, V2E, E2E, nonManifold, A);

    mRes.setE2E(std::move(E2E));
    mRes.setAdj(std::move(adj));
    mRes.setF(std::move(F));
    mRes.setV(std::move(V));
    mRes.setA(std::move(A));
    mRes.setN(std::move(N));
    mRes.setScale(scale);
    mRes.build(params.deterministic);
    mRes.resetSolution();

    if (params.align_to_boundaries) {
        mRes.clearConstraints();
        for (uint32_t i = 0; i < 3 * mRes.F().cols(); ++i) {
            if (mRes.E2E()[i] == INVALID) {
                uint32_t i0 = mRes.F()(i % 3, i / 3);
                uint32_t i1 = mRes.F()((i + 1) % 3, i / 3);
                Vector3f p0 = mRes.V().col(i0), p1 = mRes.V().col(i1);
                Vector3f edge = p1 - p0;
                if (edge.squaredNorm() > 0) {
                    edge.normalize();
                    mRes.CO().col(i0) = p0;
                    mRes.CO().col(i1) = p1;
                    mRes.CQ().col(i0) = mRes.CQ().col(i1) = edge;
                    mRes.CQw()[i0] = mRes.CQw()[i1] = mRes.COw()[i0] = mRes.COw()[i1] = 1.0f;
                }
            }
        }
        mRes.propagateConstraints(params.rosy, params.posy);
    }

    bvh = new BVH(&mRes.F(), &mRes.V(), &mRes.N(), stats.mAABB);
    bvh->build();
}

static void resample_vertex_fields(
    const MatrixXf &V_src, const MatrixXf &f1_src, const MatrixXf &f2_src,
    const MatrixXf &V_dst, MatrixXf &f1_dst, MatrixXf &f2_dst) {
  f1_dst.resize(3, V_dst.cols());
  f2_dst.resize(3, V_dst.cols());
  tbb::parallel_for(tbb::blocked_range<uint32_t>(0, (uint32_t)V_dst.cols()),
    [&](const tbb::blocked_range<uint32_t> &range) {
      for (uint32_t i = range.begin(); i != range.end(); ++i) {
        Vector3f p = V_dst.col(i);
        uint32_t best = 0;
        Float best_d = std::numeric_limits<Float>::infinity();
        for (uint32_t j = 0; j < (uint32_t)V_src.cols(); ++j) {
          Float d = (V_src.col(j) - p).squaredNorm();
          if (d < best_d) {
            best_d = d;
            best = j;
          }
        }
        f1_dst.col(i) = f1_src.col(best);
        f2_dst.col(i) = f2_src.col(best);
      }
    });
}

void inject_orientation_field(
    MultiResolutionHierarchy &mRes,
    const MatrixXf &field_v1,
    const MatrixXf &field_v2) {
  MatrixXf &Q = mRes.Q(0);
  const MatrixXf &N = mRes.N(0);
  uint32_t n = mRes.size(0);
  if ((uint32_t)field_v1.cols() != n || (uint32_t)field_v2.cols() != n)
    throw std::runtime_error("field vertex count mismatch after hierarchy build");

  tbb::parallel_for(tbb::blocked_range<uint32_t>(0, n),
    [&](const tbb::blocked_range<uint32_t> &range) {
      for (uint32_t i = range.begin(); i != range.end(); ++i) {
        Vector3f v1(field_v1(0, i), field_v1(1, i), field_v1(2, i));
        Vector3f v2(field_v2(0, i), field_v2(1, i), field_v2(2, i));
        Vector3f n = N.col(i);
        v1 -= n * n.dot(v1);
        v2 -= n * n.dot(v2);
        if (v1.squaredNorm() > RCPOVERFLOW) {
          v1.normalize();
        } else if (v2.squaredNorm() > RCPOVERFLOW) {
          v1 = v2.cross(n).normalized();
        } else {
          coordinate_system(n, v1, v2);
        }
        Q.col(i) = v1;
      }
    });
}

std::tuple<MatrixXf, MatrixXf> export_orientation_field_impl(
    py::array_t<double> vertices,
    py::array_t<int> faces,
    const BatchGnnParams &params) {
  MatrixXu F = numpy_faces_to_eigen(faces);
  MatrixXf V = numpy_vertices_to_eigen(vertices);
  MatrixXf V_orig = V;
  MatrixXf N;
  VectorXf A;
  MultiResolutionHierarchy mRes;
  BVH *bvh = nullptr;
  MeshStats stats;
  Float scale = 0;
  log_im("hierarchy");
  setup_hierarchy(F, V, N, A, mRes, bvh, stats, params, scale);
  log_im("hierarchy done");

  Optimizer optimizer(mRes, false);
  optimizer.setRoSy(params.rosy);
  optimizer.setPoSy(params.posy);
  optimizer.setExtrinsic(params.extrinsic);
  if (params.smooth_orient_iters > 0)
    mRes.setIterationsQ(params.smooth_orient_iters);

  optimizer.optimizeOrientations(-1);
  optimizer.notify();
  wait_with_progress(optimizer, "orientations");
  optimizer.shutdown();

  const MatrixXf &Q = mRes.Q(0);
  const MatrixXf &Nf = mRes.N(0);
  MatrixXf V2(3, Q.cols());
  for (uint32_t i = 0; i < Q.cols(); ++i) {
    Vector3f n(Nf(0, i), Nf(1, i), Nf(2, i));
    Vector3f q(Q(0, i), Q(1, i), Q(2, i));
    V2.col(i) = n.cross(q).normalized();
  }

  MatrixXf Q_out = Q, V2_out = V2;
  if ((uint32_t)V_orig.cols() != (uint32_t)Q.cols()) {
    resample_vertex_fields(mRes.V(), Q, V2, V_orig, Q_out, V2_out);
  }

  if (bvh) delete bvh;
  return std::make_tuple(Q_out, V2_out);
}

std::tuple<MatrixXf, MatrixXu> extract_with_cross_field_impl(
    py::array_t<double> vertices,
    py::array_t<int> faces,
    py::array_t<double> field_v1,
    py::array_t<double> field_v2,
    const BatchGnnParams &params) {
  MatrixXu F = numpy_faces_to_eigen(faces);
  MatrixXf V = numpy_vertices_to_eigen(vertices);
  MatrixXf V_orig = V;
  MatrixXf fv1 = numpy_vertices_to_eigen(field_v1);
  MatrixXf fv2 = numpy_vertices_to_eigen(field_v2);
  MatrixXf N;
  VectorXf A;
  MultiResolutionHierarchy mRes;
  BVH *bvh = nullptr;
  MeshStats stats;
  Float scale = 0;
  log_im("hierarchy");
  setup_hierarchy(F, V, N, A, mRes, bvh, stats, params, scale);
  log_im("hierarchy done");

  MatrixXf fv1_use = fv1, fv2_use = fv2;
  if ((uint32_t)fv1.cols() != mRes.size(0)) {
    resample_vertex_fields(V_orig, fv1, fv2, mRes.V(), fv1_use, fv2_use);
  }
  inject_orientation_field(mRes, fv1_use, fv2_use);
  mRes.propagateSolution(params.rosy);

  Optimizer optimizer(mRes, false);
  optimizer.setRoSy(params.rosy);
  optimizer.setPoSy(params.posy);
  optimizer.setExtrinsic(params.extrinsic);

  if (params.smooth_orient_iters > 0) {
    mRes.setIterationsQ(params.smooth_orient_iters);
    optimizer.optimizeOrientations(-1);
    optimizer.notify();
    wait_with_progress(optimizer, "orientations");
  }

  if (params.smooth_pos_iters > 0)
    mRes.setIterationsO(params.smooth_pos_iters);

  optimizer.optimizePositions(-1);
  optimizer.notify();
  wait_with_progress(optimizer, "positions");
  optimizer.shutdown();

  std::set<uint32_t> crease_in, crease_out;
  MatrixXf O_extr, N_extr, Nf_extr;
  std::vector<std::vector<TaggedLink>> adj_extr;
  extract_graph(mRes, params.extrinsic, params.rosy, params.posy, adj_extr,
                O_extr, N_extr, crease_in, crease_out, params.deterministic);

  MatrixXu F_extr;
  log_im("extract faces");
  extract_faces(adj_extr, O_extr, N_extr, Nf_extr, F_extr, params.posy,
                mRes.scale(), crease_out, true, params.pure_quad, bvh,
                params.smooth_pos_iters);
  log_im("extract faces done");

  if (bvh) delete bvh;
  return std::make_tuple(O_extr, F_extr);
}

#endif  // HAS_INSTANT_MESHES

#ifdef HAS_INSTANT_MESHES
py::tuple py_export_orientation_field(
    py::array_t<double> vertices, py::array_t<int> faces, const BatchGnnParams &params) {
    auto tup = export_orientation_field_impl(vertices, faces, params);
    return py::make_tuple(eigen_matrix_to_numpy_vertices(std::get<0>(tup)),
                          eigen_matrix_to_numpy_vertices(std::get<1>(tup)));
}

py::tuple py_extract_with_cross_field(
    py::array_t<double> vertices, py::array_t<int> faces,
    py::array_t<double> field_v1, py::array_t<double> field_v2,
    const BatchGnnParams &params) {
    auto tup = extract_with_cross_field_impl(vertices, faces, field_v1, field_v2, params);
    return py::make_tuple(eigen_matrix_to_numpy_vertices(std::get<0>(tup)),
                          eigen_faces_to_numpy(std::get<1>(tup)));
}
#endif
