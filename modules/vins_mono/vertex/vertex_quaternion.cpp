//
// Created by Cain on 2024/4/12.
//

#include "vertex_quaternion.h"

graph_optimization::VertexQuaternion::VertexQuaternion() : Vertex(4, 3) {
    _parameters[0] = 1.;
    _parameters[1] = 0.;
    _parameters[2] = 0.;
    _parameters[3] = 0.;
}

void graph_optimization::VertexQuaternion::plus(const graph_optimization::VecX &delta) {
    VecX &params = parameters();
    Qd q(params[0], params[1], params[2], params[3]);
    q = q * Sophus::SO3d::exp(Vec3(delta[0], delta[1], delta[2])).unit_quaternion();  // q = q * dq
    q.normalized();
    params[0] = q.w();
    params[1] = q.x();
    params[2] = q.y();
    params[3] = q.z();
}