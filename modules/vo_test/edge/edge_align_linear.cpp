//
// Created by Cain on 2024/4/25.
//

#include "edge_align_linear.h"
#include "../vertex/vertex_vec1.h"
#include "../vertex/vertex_velocity.h"
#include "../vertex/vertex_bias.h"
#include <iostream>

namespace graph_optimization {
    EdgeAlignLinear::EdgeAlignLinear(const vins::IMUIntegration &imu_integration, const Vec3 &t_ij, const Qd &q_ij, const Qd &q_0i)
            : Edge(6, 4, std::vector<std::string>{"VertexVec1", "VertexBias", "VertexVelocity", "VertexVelocity"}),
              _imu_integration(imu_integration), _t_ij(t_ij), _q_ij(q_ij), _q_0i(q_0i) {

    }

    void EdgeAlignLinear::compute_residual() {
        double scale = _vertices[0]->get_parameters()[0] / 100.;
        Vec3 g_b0 = _vertices[1]->get_parameters();
        Vec3 v_i = _vertices[2]->get_parameters();
        Vec3 v_j = _vertices[3]->get_parameters();

        double dt = _imu_integration.get_sum_dt();

        // (e_p, e_r, e_v)
        _residual.head<3>() = _q_0i.inverse() * (scale * _t_ij - (v_i + 0.5 * dt * g_b0) * dt) - _imu_integration.get_delta_p();
        _residual.tail<3>() = _q_0i.inverse() * (v_j - (v_i + g_b0 * dt)) - _imu_integration.get_delta_v();

        _residual /= dt;

//        std::cout << "scale: " << scale << std::endl;
//        std::cout << "g_b0: " << g_b0.transpose() << std::endl;
//        std::cout << "v_i: " << v_i.transpose() << std::endl;
//        std::cout << "v_j: " << v_j.transpose() << std::endl;
//        std::cout << "dt: " << dt << std::endl;
//        std::cout << "delta_v: " << _imu_integration.get_delta_v().transpose() << std::endl;
//        std::cout << "delta_p: " << _imu_integration.get_delta_p().transpose() << std::endl;

//        std::cout << "_residual.head<3>() = " << _residual.head<3>().transpose() << std::endl;
//        std::cout << "_imu_integration.get_delta_p() = " << _imu_integration.get_delta_p().transpose() << std::endl;
//        std::cout << "_residual.tail<3>() = " << _residual.tail<3>().transpose() << std::endl;
//        std::cout << "_imu_integration.get_delta_v() = " << _imu_integration.get_delta_v().transpose() << std::endl;

//        set_information(_imu_integration.get_covariance().topLeftCorner<6, 6>().inverse());
//        Mat66 information;
//        information.setIdentity();
//        information *= 1000.;
//        set_information(information);
    }

    void EdgeAlignLinear::compute_jacobians() {
        double dt = _imu_integration.get_sum_dt();
        Mat33 R_0i_T = _q_0i.toRotationMatrix().transpose();

        // jacobian[0]: 6x1, (e_p, e_v) x (scale)
        _jacobians[0] = Eigen::Matrix<double, 6, 1>::Zero();
        _jacobians[0].block<3, 1>(0, 0) = R_0i_T * _t_ij / 100.;

        // jacobian[1]: 6x2, (e_p, e_v) x (g_b0)
        _jacobians[1] = Eigen::Matrix<double, 6, 3>::Zero();
        _jacobians[1].block<3, 3>(3, 0) = -R_0i_T * dt;
        _jacobians[1].block<3, 3>(0, 0) =  0.5 * dt * _jacobians[1].block<3, 3>(3, 0);

        // jacobian[2]: 6x3, (e_p, e_v) x (v_i)
        _jacobians[2] = Eigen::Matrix<double, 6, 3>::Zero();
        _jacobians[2].block<3, 3>(3, 0) = -R_0i_T;
        _jacobians[2].block<3, 3>(0, 0) = dt * _jacobians[2].block<3, 3>(3, 0);

        // jacobian[3]: 6x3, (e_p, e_v) x (v_j)
        _jacobians[3] = Eigen::Matrix<double, 6, 3>::Zero();
        _jacobians[3].block<3, 3>(3, 0) = R_0i_T;

        _jacobians[0] /= dt;
        _jacobians[1] /= dt;
        _jacobians[2] /= dt;
        _jacobians[3] /= dt;
    }
}