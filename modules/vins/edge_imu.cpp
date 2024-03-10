//
// Created by Cain on 2024/3/10.
//

#include "edge_imu.h"

namespace graph_optimization {
    EdgeImu::EdgeImu(const vins::IMUIntegration &imu_integration)
    : Edge(15, 4, std::vector<std::string>{"VertexPose", "VertexMotion", "VertexPose", "VertexMotion"}),
      _imu_integration(imu_integration) {

    }

    void EdgeImu::compute_residual() {
        auto pose_i = _vertices[0]->get_parameters();
        auto motion_i = _vertices[1]->get_parameters();
        auto pose_j = _vertices[2]->get_parameters();
        auto motion_j = _vertices[3]->get_parameters();

        Vec3 p_i = pose_i.head<3>();
        Qd q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Vec3 v_i = motion_i.head<3>();
        Vec3 ba_i = motion_i.segment(3, 3);
        Vec3 bg_i = motion_i.tail<3>();

        Vec3 p_j = pose_j.head<3>();
        Qd q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);
        Vec3 v_j = motion_j.head<3>();
        Vec3 ba_j = motion_j.segment(3, 3);
        Vec3 bg_j = motion_j.tail<3>();

        auto &&dt = _imu_integration.get_sum_dt();

        Vec3 r_p = q_i * (p_j - (p_i + v_i * dt + 0.5 * _gravity * dt * dt)) - alpha;
        Vec3 r_q = 2. * (q_ji * (q_i * q_j)).img();
        Vec3 r_v = q_i * (v_j - (v_i + _gravity * dt)) - beta;
        Vec3 r_ba = ba_j - ba_i;
        Vec3 r_bg = bg_j - bg_i;
    }

    void EdgeImu::compute_jacobians() {

    }
}