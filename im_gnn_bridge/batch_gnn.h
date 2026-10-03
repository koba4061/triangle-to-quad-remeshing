#pragma once

namespace BatchGnn {

struct Params {
    int rosy = 4;
    int posy = 4;
    float scale = -1.f;
    int face_count = -1;
    int vertex_count = -1;
    float crease_angle = 40.f;
    bool extrinsic = false;
    bool align_to_boundaries = true;
    int smooth_orient_iters = 0;
    int smooth_pos_iters = 2;
    bool pure_quad = true;
    bool deterministic = true;
};

}  // namespace BatchGnn
