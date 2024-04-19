//
// Created by Cain on 2024/4/11.
//

#include "edge_pnp_sim3.h"
#include "vertex_pose.h"

void graph_optimization::EdgePnPSim3::compute_residual() {
    // imu的位姿
    const auto &pose = _vertices[0]->get_parameters();
    Vec3 t_i = pose.head<3>();
    Qd q_i {pose[6], pose[3], pose[4], pose[5]};

    // 重投影误差
    Vec3 p_imu_i = q_i.inverse() * (_p_world - t_i);
    Vec3 p_camera_c = _q_ic.inverse() * (p_imu_i - _t_ic);

    // 尺度
    if (!_is_scale_initialized) {
        _vertices[1]->set_parameters(Vec1(p_camera_c.z() > 0.1 ? p_camera_c.z() : 0.1));
        _is_scale_initialized = true;
    }
    const auto &scale = _vertices[1]->get_parameters();
    double inv_depth = 1. / scale[0];

    // 误差
    _residual = p_camera_c * inv_depth - _p_pixel;
}

void graph_optimization::EdgePnPSim3::compute_jacobians() {
    // imu的位姿
    const auto &pose = _vertices[0]->get_parameters();
    Vec3 t_i = pose.head<3>();
    Qd q_i {pose[6], pose[3], pose[4], pose[5]};
    Mat33 R_i = q_i.toRotationMatrix();

    // 尺度
    const auto &scale = _vertices[1]->get_parameters();
    double inv_depth = 1. / scale[0];

    // 重投影
    Vec3 p_imu_i = q_i.inverse() * (_p_world - t_i);
    Vec3 p_camera_c = _q_ic.inverse() * (p_imu_i - _t_ic);

    // 误差对imu位姿的偏导
    Eigen::Matrix<double, 3, 6> jacobian_pose;
    Mat33 R_ic = _q_ic.toRotationMatrix();
    jacobian_pose.leftCols<3>() = -inv_depth * R_ic.transpose() * R_i.transpose();
    jacobian_pose.rightCols<3>() = inv_depth * R_ic.transpose() * Sophus::SO3d::hat(p_imu_i);

    Eigen::Matrix<double, 3, 1> jacobian_scale = -inv_depth * p_camera_c;

    _jacobians[0] = jacobian_pose;
    _jacobians[1] = jacobian_scale;
}