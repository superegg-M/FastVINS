//
// Created by Cain on 2024/1/2.
//

#include "vertex_pose.h"

void graph_optimization::VertexPose::plus(const graph_optimization::VecX &delta) {
    VecX &params = parameters();
    params.head<3>() += delta.head<3>();
    Qd q(params[6], params[3], params[4], params[5]);
    q = q * Sophus::SO3d::exp(Vec3(delta[3], delta[4], delta[5])).unit_quaternion();  // q = q * dq
    q.normalized();
    params[3] = q.x();
    params[4] = q.y();
    params[5] = q.z();
    params[6] = q.w();
}
