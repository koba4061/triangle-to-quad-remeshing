/*
 * im_gnn_bridge: Pybind11 bridge for GNN cross-field -> Instant Meshes extraction.
 */
#include "batch_gnn.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using BatchGnnParams = BatchGnn::Params;

#ifdef HAS_INSTANT_MESHES
py::tuple py_export_orientation_field(py::array_t<double>, py::array_t<int>, const BatchGnnParams &);
py::tuple py_extract_with_cross_field(py::array_t<double>, py::array_t<int>,
    py::array_t<double>, py::array_t<double>, const BatchGnnParams &);
#endif

static void check_field_shapes(
    py::array_t<double> V, py::array_t<int> F,
    py::array_t<double> field_v1, py::array_t<double> field_v2) {
    if (V.ndim() != 2 || V.shape(1) != 3)
        throw std::invalid_argument("vertices must be (N, 3)");
    if (F.ndim() != 2 || F.shape(1) != 3)
        throw std::invalid_argument("faces must be (M, 3)");
    py::ssize_t n = V.shape(0);
    if (field_v1.shape(0) != n || field_v2.shape(0) != n)
        throw std::invalid_argument("field must match vertex count");
}

py::tuple export_orientation_field(
    py::array_t<double> vertices,
    py::array_t<int> faces,
    double crease_angle = 40.0,
    bool align_to_boundaries = true,
    int smooth_iterations = 10) {
#ifndef HAS_INSTANT_MESHES
    throw py::import_error(
        "im_gnn_bridge: Instant Meshes not linked. Run scripts/build_native.bat");
#else
    BatchGnnParams p;
    p.crease_angle = (float)crease_angle;
    p.align_to_boundaries = align_to_boundaries;
    p.smooth_orient_iters = smooth_iterations;
    return py_export_orientation_field(vertices, faces, p);
#endif
}

py::tuple extract_with_cross_field(
    py::array_t<double> vertices,
    py::array_t<int> faces,
    py::array_t<double> field_v1,
    py::array_t<double> field_v2,
    double target_edge_length,
    double crease_angle = 40.0,
    bool align_to_boundaries = true,
    int smooth_orient_iters = 0,
    int smooth_pos_iters = 2,
    bool pure_quad = true) {
    check_field_shapes(vertices, faces, field_v1, field_v2);
#ifndef HAS_INSTANT_MESHES
    throw py::import_error(
        "im_gnn_bridge: Instant Meshes not linked. Run scripts/build_native.bat");
#else
    BatchGnnParams p;
    p.scale = (float)target_edge_length;
    p.crease_angle = (float)crease_angle;
    p.align_to_boundaries = align_to_boundaries;
    p.smooth_orient_iters = smooth_orient_iters;
    p.smooth_pos_iters = smooth_pos_iters;
    p.pure_quad = pure_quad;
    return py_extract_with_cross_field(vertices, faces, field_v1, field_v2, p);
#endif
}

PYBIND11_MODULE(im_gnn_bridge, m) {
    m.doc() = "GNN cross-field -> Instant Meshes extraction bridge";
    m.def("export_orientation_field", &export_orientation_field,
          py::arg("vertices"), py::arg("faces"),
          py::arg("crease_angle") = 40.0,
          py::arg("align_to_boundaries") = true,
          py::arg("smooth_iterations") = 10);
    m.def("extract_with_cross_field", &extract_with_cross_field,
          py::arg("vertices"), py::arg("faces"),
          py::arg("field_v1"), py::arg("field_v2"),
          py::arg("target_edge_length"),
          py::arg("crease_angle") = 40.0,
          py::arg("align_to_boundaries") = true,
          py::arg("smooth_orient_iters") = 0,
          py::arg("smooth_pos_iters") = 2,
          py::arg("pure_quad") = true);
#ifdef HAS_INSTANT_MESHES
    m.attr("has_instant_meshes") = true;
#else
    m.attr("has_instant_meshes") = false;
#endif
}
