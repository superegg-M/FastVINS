//
// Created by Cain on 2024/4/9.
//

#include "edge_pnp.h"
#include "../vertex/vertex_pose.h"

void graph_optimization::EdgePnP::compute_residual() {
    // imu的位姿
    const auto &params_i = _vertices[0]->get_parameters();
    Vec3 t_i = params_i.head<3>();
    Qd q_i {params_i[6], params_i[3], params_i[4], params_i[5]};

    // 重投影误差
    Vec3 p_imu_i = q_i.inverse() * (_p_world - t_i);
    Vec3 p_camera_i = _q_ic.inverse() * (p_imu_i - _t_ic);

    // 逆深度
    double inv_depth_i = 1. / p_camera_i.z();

    // 误差
    _residual = (p_camera_i * inv_depth_i).head<2>() - _p_pixel.head<2>();
}

void graph_optimization::EdgePnP::compute_jacobians() {
    // imu的位姿
    const auto &params_i = _vertices[0]->get_parameters();
    Vec3 t_i = params_i.head<3>();
    Qd q_i {params_i[6], params_i[3], params_i[4], params_i[5]};
    Mat33 R_i = q_i.toRotationMatrix();

    // 重投影
    Vec3 p_imu_i = q_i.inverse() * (_p_world - t_i);
    Vec3 p_camera_i = _q_ic.inverse() * (p_imu_i - _t_ic);

    // 逆深度
    double inv_depth_i = 1. / p_camera_i.z();

    // 误差对投影点的偏导
    Mat23 dr_dpci;
    dr_dpci << inv_depth_i, 0., -p_camera_i[0] * inv_depth_i * inv_depth_i,
            0., inv_depth_i, -p_camera_i[1] * inv_depth_i * inv_depth_i;

    // 投影点对imu位姿的偏导
    Eigen::Matrix<double, 3, 6> dpci_dpose_i;
    Mat33 R_ic = _q_ic.toRotationMatrix();
    dpci_dpose_i.leftCols<3>() = -R_ic.transpose() * R_i.transpose();
    dpci_dpose_i.rightCols<3>() = R_ic.transpose() * Sophus::SO3d::hat(p_imu_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    jacobian_pose_i = dr_dpci * dpci_dpose_i;

    _jacobians[0] = jacobian_pose_i;
}