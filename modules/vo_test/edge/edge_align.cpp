//
// Created by Cain on 2024/4/24.
//

#include "edge_align.h"
#include "../vertex/vertex_scale.h"
#include "../vertex/vertex_spherical.h"
#include "../vertex/vertex_velocity.h"
#include "../vertex/vertex_bias.h"
#include <iostream>

namespace graph_optimization {
    EdgeAlign::EdgeAlign(const vins::IMUIntegration &imu_integration, const Vec3 &t_ij, const Qd &q_ij, const Qd &q_0i)
    : Edge(9, 6, std::vector<std::string>{"VertexScale", "VertexSpherical", "VertexVelocity", "VertexVelocity", "VertexBias", "VertexBias"}),
      _imu_integration(imu_integration), _t_ij(t_ij), _q_ij(q_ij), _q_0i(q_0i) {

    }

    void EdgeAlign::compute_residual() {
        double scale = _vertices[0]->get_parameters()[0];
        Qd q_wb0(_vertices[1]->get_parameters()[0], _vertices[1]->get_parameters()[1], _vertices[1]->get_parameters()[2], _vertices[1]->get_parameters()[3]);
        Vec3 v_i = _vertices[2]->get_parameters();
        Vec3 v_j = _vertices[3]->get_parameters();
        Vec3 ba = _vertices[4]->get_parameters();
        Vec3 bg = _vertices[5]->get_parameters();

        double dt = _imu_integration.get_sum_dt();
        Vec3 g_b0 = q_wb0.inverse() * _imu_integration.get_gravity();

        // (e_p, e_r, e_v)
        _residual.head<3>() = _q_0i.inverse() * (scale * _t_ij - (scale * v_i + 0.5 * dt * g_b0) * dt) - _imu_integration.get_delta_p();
        _residual.segment<3>(3) = 2. * (_imu_integration.get_delta_q().inverse() * _q_ij).vec();
        _residual.tail<3>() = _q_0i.inverse() * (scale * v_j - (scale * v_i + g_b0 * dt)) - _imu_integration.get_delta_v();

        _residual /= dt;

//        std::cout << "residual: " << _residual << std::endl;

//        set_information(_imu_integration.get_covariance().topLeftCorner<9, 9>().inverse());
    }

    void EdgeAlign::compute_jacobians() {
        double scale = _vertices[0]->get_parameters()[0];
        Qd q_wb0(_vertices[1]->get_parameters()[0], _vertices[1]->get_parameters()[1], _vertices[1]->get_parameters()[2], _vertices[1]->get_parameters()[3]);
        Vec3 v_i = _vertices[2]->get_parameters();
        Vec3 v_j = _vertices[3]->get_parameters();
        Vec3 ba = _vertices[4]->get_parameters();
        Vec3 bg = _vertices[5]->get_parameters();

        double dt = _imu_integration.get_sum_dt();
        Vec3 g_b0 = q_wb0.inverse() * _imu_integration.get_gravity();
        Vec3 dg_b0 = g_b0 * dt;
        Mat33 R_0i_T = _q_0i.toRotationMatrix().transpose();

        // 修正预积分
        Eigen::Vector3d delta_ba_i = ba - _imu_integration.get_ba();
        Eigen::Vector3d delta_bg_i = bg - _imu_integration.get_bg();
        _imu_integration.correct(delta_ba_i, delta_bg_i);

        // 预积分的jacobian
        auto &&dr_dbg = _imu_integration.get_dr_dbg();
        auto &&dp_dba = _imu_integration.get_dp_dba();
        auto &&dp_dbg = _imu_integration.get_dp_dbg();
        auto &&dv_dba = _imu_integration.get_dv_dba();
        auto &&dv_dbg = _imu_integration.get_dv_dbg();

        // jacobian[0]: 9x1, (e_p, e_r, e_v) x (scale)
        _jacobians[0] = Eigen::Matrix<double, 9, 1>::Zero();
        _jacobians[0].block<3, 1>(0, 0) = R_0i_T * _t_ij * scale;
        _jacobians[0].block<3, 1>(6, 0) = R_0i_T * (v_j - v_i) * scale;

        // jacobian[1]: 9x2, (e_p, e_r, e_v) x (q_wb0)
        _jacobians[1] = Eigen::Matrix<double, 9, 2>::Zero();
        _jacobians[1].block<3, 2>(6, 0) = (R_0i_T * Sophus::SO3d::hat(-dg_b0)).leftCols<2>();
        _jacobians[1].block<3, 2>(0, 0) =  0.5 * dt * _jacobians[1].block<3, 2>(6, 0);

        // jacobian[2]: 9x3, (e_p, e_r, e_v) x (v_i)
        _jacobians[2] = Eigen::Matrix<double, 9, 3>::Zero();
        _jacobians[2].block<3, 3>(6, 0) = -R_0i_T * scale;
        _jacobians[2].block<3, 3>(0, 0) = dt * _jacobians[2].block<3, 3>(6, 0);

        // jacobian[3]: 9x3, (e_p, e_r, e_v) x (v_j)
        _jacobians[3] = Eigen::Matrix<double, 9, 3>::Zero();
        _jacobians[3].block<3, 3>(6, 0) = R_0i_T * scale;

        // jacobian[4]: 9x3, (e_p, e_r, e_v) x (ba)
        _jacobians[4] = Eigen::Matrix<double, 9, 3>::Zero();
        _jacobians[4].block<3, 3>(0, 0) = -dp_dba;
        _jacobians[4].block<3, 3>(6, 0) = -dv_dba;

        // jacobian[5]: 9x3, (e_p, e_r, e_v) x (bg)
        _jacobians[5] = Eigen::Matrix<double, 9, 3>::Zero();
        _jacobians[5].block<3, 3>(0, 0) = -dp_dbg;
        _jacobians[5].block<3, 3>(6, 0) = -dv_dbg;
        _jacobians[5].block<3, 3>(3, 0) = -get_quat_left(_q_ij.inverse() * _imu_integration.get_delta_q()).bottomRightCorner<3, 3>() * dr_dbg;

        _jacobians[0] /= dt;
        _jacobians[1] /= dt;
        _jacobians[2] /= dt;
        _jacobians[3] /= dt;
        _jacobians[4] /= dt;
        _jacobians[5] /= dt;
    }

    Eigen::Matrix<double, 4, 4> EdgeAlign::get_quat_left(const Qd &q) {
        Eigen::Matrix<double, 4, 4> m;
        m(0, 0) = q.w();
        m.block<3, 1>(1, 0) = q.vec();
        m.block<1, 3>(0, 1) = -q.vec().transpose();
        m.block<3, 3>(1, 1) << q.w(), -q.z(), q.y(),
                q.z(), q.w(), -q.x(),
                -q.y(), q.x(), q.w();
        return m;
    }

    Eigen::Matrix<double, 4, 4> EdgeAlign::get_quat_right(const Qd &q) {
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
