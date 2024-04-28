//
// Created by Cain on 2024/3/10.
//

#include "edge_imu.h"
#include "../vertex/vertex_pose.h"
#include "../vertex/vertex_motion.h"

namespace graph_optimization {
    EdgeImu::EdgeImu(const vins::IMUIntegration &imu_integration)
    : Edge(15, 4, std::vector<std::string>{"VertexPose", "VertexMotion", "VertexPose", "VertexMotion"}),
      _imu_integration(imu_integration) {

    }

    void EdgeImu::compute_residual() {
        auto &&pose_i = _vertices[0]->get_parameters();
        auto &&motion_i = _vertices[1]->get_parameters();
        auto &&pose_j = _vertices[2]->get_parameters();
        auto &&motion_j = _vertices[3]->get_parameters();

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

        Eigen::Matrix<double, 3, 3> R_i_w = q_i.inverse().toRotationMatrix();
        auto &&dt = _imu_integration.get_sum_dt();

        Eigen::Vector3d delta_ba_i = ba_i - _imu_integration.get_ba();
        Eigen::Vector3d delta_bg_i = bg_i - _imu_integration.get_bg();
        _imu_integration.correct(delta_ba_i, delta_bg_i);

        _residual.head<3>() = R_i_w * (p_j - (p_i + v_i * dt + (0.5 * dt * dt) * vins::IMUIntegration::get_gravity())) - _imu_integration.get_delta_p();
        _residual.segment<3>(3) = 2. * (_imu_integration.get_delta_q().inverse() * (q_i.inverse() * q_j)).vec();
        _residual.segment<3>(6) = R_i_w * (v_j - (v_i + dt * vins::IMUIntegration::get_gravity())) - _imu_integration.get_delta_v();
        _residual.segment<3>(9) = ba_j - ba_i;
        _residual.segment<3>(12) = bg_j - bg_i;

        set_information(_imu_integration.get_covariance().inverse());
    }

    void EdgeImu::compute_jacobians() {
        auto &&pose_i = _vertices[0]->get_parameters();
        auto &&motion_i = _vertices[1]->get_parameters();
        auto &&pose_j = _vertices[2]->get_parameters();
        auto &&motion_j = _vertices[3]->get_parameters();

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

        Eigen::Matrix<double, 3, 3> R_i_w = q_i.inverse().toRotationMatrix();
        auto &&dt = _imu_integration.get_sum_dt();
        Eigen::Vector3d delta_ba_i = ba_i - _imu_integration.get_ba();
        Eigen::Vector3d delta_bg_i = bg_i - _imu_integration.get_bg();
        _imu_integration.correct(delta_ba_i, delta_bg_i);
        auto &&dr_dbg = _imu_integration.get_dr_dbg();
        auto &&dp_dba = _imu_integration.get_dp_dba();
        auto &&dp_dbg = _imu_integration.get_dp_dbg();
        auto &&dv_dba = _imu_integration.get_dv_dba();
        auto &&dv_dbg = _imu_integration.get_dv_dbg();

        // jacobian[0]: 15x6, (e_p, e_r, e_v, e_ba, e_bg) x (p_i, r_i)
        _jacobians[0] = Eigen::Matrix<double, 15, 6>::Zero();
        // e_p, p_i
        _jacobians[0].block<3, 3>(0, 0) = -R_i_w;
        // e_p, r_i
        _jacobians[0].block<3, 3>(0, 3) = Sophus::SO3d::hat(R_i_w * (p_j - (p_i + dt * v_i + (0.5 * dt * dt) * vins::IMUIntegration::get_gravity())));
        // e_r, r_i
        _jacobians[0].block<3, 3>(3, 3) = -get_quat_left(q_j.inverse() * q_i).bottomRows<3>() * get_quat_right(_imu_integration.get_delta_q()).rightCols<3>();
        // e_v, r_i
        _jacobians[0].block<3, 3>(6, 3) = Sophus::SO3d::hat((R_i_w * (v_j - (v_i + dt * vins::IMUIntegration::get_gravity()))));

        // jacobian[1]: 15x9, (e_p, e_r, e_v, e_ba, e_bg) x (v_i, ba_i, bg_i)
        _jacobians[1] = Eigen::Matrix<double, 15, 9>::Zero();
        // e_p, v_i
        _jacobians[1].block<3, 3>(0, 0) = -R_i_w * dt;
        // e_p, ba_i
        _jacobians[1].block<3, 3>(0, 3) = -dp_dba;
        // e_p, bg_i
        _jacobians[1].block<3, 3>(0, 6) = -dp_dbg;
        // e_r, bg_i
        _jacobians[1].block<3, 3>(3, 6) = -get_quat_left(q_j.inverse() * q_i * _imu_integration.get_delta_q()).bottomRightCorner<3, 3>() * dr_dbg;
        // e_v, v_i
        _jacobians[1].block<3, 3>(6, 0) = -R_i_w;
        // e_v, ba_i
        _jacobians[1].block<3, 3>(6, 3) = -dv_dba;
        // e_v, bg_i
        _jacobians[1].block<3, 3>(6, 6) = -dv_dbg;
        // e_ba, ba_i
        _jacobians[1].block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();
        // e_bg, bg_i
        _jacobians[1].block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();

        // jacobian[2]: 15x6, (e_p, e_r, e_v, e_ba, e_bg) x (p_j, r_j)
        _jacobians[2] = Eigen::Matrix<double, 15, 6>::Zero();
        // e_p, p_j
        _jacobians[2].block<3, 3>(0, 0) = R_i_w;
        // e_r, r_j
        _jacobians[2].block<3, 3>(3, 3) = get_quat_left(_imu_integration.get_delta_q().inverse() * q_i.inverse() * q_j).bottomRightCorner<3, 3>();

        // jacobian[3]: 15x9, (e_p, e_r, e_v, e_ba, e_bg) x (v_j, ba_j, bg_j)
        _jacobians[3] = Eigen::Matrix<double, 15, 9>::Zero();
        // e_v, v_j
        _jacobians[3].block<3, 3>(6, 0) = R_i_w;
        // e_ba, ba_j
        _jacobians[3].block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();
        // e_bg, bg_j
        _jacobians[3].block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();
    }

    Eigen::Matrix<double, 4, 4> EdgeImu::get_quat_left(const Qd &q) {
        Eigen::Matrix<double, 4, 4> m;
        m(0, 0) = q.w();
        m.block<3, 1>(1, 0) = q.vec();
        m.block<1, 3>(0, 1) = -q.vec().transpose();
        m.block<3, 3>(1, 1) << q.w(), -q.z(), q.y(),
                                              q.z(), q.w(), -q.x(),
                                              -q.y(), q.x(), q.w();
        return m;
    }

    Eigen::Matrix<double, 4, 4> EdgeImu::get_quat_right(const Qd &q) {
        Eigen::Matrix<double, 4, 4> m;
        m(0, 0) = q.w();
        m.block<3, 1>(1, 0) = q.vec();
        m.block<1, 3>(0, 1) = -q.vec().transpose();
        m.block<3, 3>(1, 1) << q.w(), q.z(), -q.y(),
                                              -q.z(), q.w(), q.x(),
                                              q.y(), -q.x(), q.w();
        return m;
    }
}