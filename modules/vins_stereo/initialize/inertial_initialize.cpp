//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"
#include "../imu_integration.h"
#include "../vertex/vertex_vec1.h"
#include "../vertex/vertex_bias.h"
#include "../vertex/vertex_velocity.h"
#include "../edge/edge_align_linear.h"
#include "../edge/edge_align.h"

#include "tic_toc/tic_toc.h"
#include "backend/eigen_types.h"

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    bool Estimator::align_visual_to_imu() {
        if (_windows.size() < 2) {
            return false;
        }

        /* 非线性问题, 以线性问题的结果作为初值, 进一步优化出v, alpha, R0, ba, bg */
        Problem nonlinear_problem;  // 非线性求解

        // 顶点
        shared_ptr<VertexScale> vertex_scale = make_shared<VertexScale>();  // 尺度因子
        shared_ptr<VertexSpherical> vertex_q_wb0 = make_shared<VertexSpherical>();  // 重力方向
        shared_ptr<VertexBias> vertex_ba = make_shared<VertexBias>();   // 加速度偏移
        shared_ptr<VertexBias> vertex_bg = make_shared<VertexBias>();   // 角速度偏移
        shared_ptr<VertexVelocity> vertex_v[_windows.size() + 1];   // 速度
        for (auto &v : vertex_v) {
            v = make_shared<VertexVelocity>();
            v->parameters().setZero();
        }

        // 把顶点加入到problem中
        nonlinear_problem.add_vertex(vertex_scale);
        nonlinear_problem.add_vertex(vertex_q_wb0);
        nonlinear_problem.add_vertex(vertex_ba);
        nonlinear_problem.add_vertex(vertex_bg);
        nonlinear_problem.add_vertex(vertex_v[0]);

        // 固定尺度因子
        vertex_scale->set_fixed();

        // 线性计算 Ax=b
        unsigned long n_state = 3 + 3 * (_windows.size() + 1);
        MatXX A(MatXX::Zero(n_state, n_state));
        MatXX b(MatXX::Zero(n_state, 1));

        // 遍历windows
        for (unsigned long i = 1; i < _windows.size(); ++i) {
            // 提取位姿
            auto &&pose_i = _windows[i - 1]->vertex_pose->get_parameters();
            auto &&pose_j = _windows[i]->vertex_pose->get_parameters();

            Vec3 p_i {pose_i(0), pose_i(1), pose_i(2)};
            Qd q_i {pose_i(6), pose_i(3), pose_i(4), pose_i(5)};

            Vec3 p_j {pose_j(0), pose_j(1), pose_j(2)};
            Qd q_j {pose_j(6), pose_j(3), pose_j(4), pose_j(5)};

            Vec3 tij = p_j - p_i;
            Qd qij = (q_i.inverse() * q_j).normalized();

            /* 非线性问题的边 */
            shared_ptr<EdgeAlign> nonlinear_edge = make_shared<EdgeAlign>(*_windows[i]->imu_integration,
                                                                          tij,
                                                                          qij,
                                                                          q_i);
            nonlinear_edge->add_vertex(vertex_scale);
            nonlinear_edge->add_vertex(vertex_q_wb0);
            nonlinear_edge->add_vertex(vertex_v[i - 1]);
            nonlinear_edge->add_vertex(vertex_v[i]);
            nonlinear_edge->add_vertex(vertex_ba);
            nonlinear_edge->add_vertex(vertex_bg);

            // 加入到problem
            nonlinear_problem.add_edge(nonlinear_edge);
            nonlinear_problem.add_vertex(vertex_v[i]);

            // Ax = b
            {
                double dt = _windows[i]->imu_integration->get_sum_dt();
                Mat33 R_0i_T = q_i.toRotationMatrix().transpose();

                MatXX A_tmp(MatXX::Zero(6, 9));
                MatXX b_tmp(MatXX::Zero(6, 1));

                A_tmp.block<3, 3>(3, 0) = -R_0i_T * dt;
                A_tmp.block<3, 3>(0, 0) =  (0.5 * dt) * A_tmp.block<3, 3>(3, 0);
                A_tmp.block<3, 3>(3, 3) = -R_0i_T;
                A_tmp.block<3, 3>(0, 3) = dt * A_tmp.block<3, 3>(3, 3);
                A_tmp.block<3, 3>(3, 6) = R_0i_T;

                b_tmp.topRows<3>() = _windows[i]->imu_integration->get_delta_p() - R_0i_T * tij;
                b_tmp.bottomRows<3>() = _windows[i]->imu_integration->get_delta_v();

                auto &&cov = _windows[i]->imu_integration->get_covariance();
                Mat66 cov_tmp;
                cov_tmp.block<3, 3>(0, 0) = cov.block<3, 3>(0, 0);
                cov_tmp.block<3, 3>(0, 3) = cov.block<3, 3>(0, 6);
                cov_tmp.block<3, 3>(3, 3) = cov.block<3, 3>(6, 6);
                cov_tmp.block<3, 3>(3, 0) = cov.block<3, 3>(6, 0);
                auto &&cov_tmp_ldlt = cov_tmp.ldlt();
                MatXX ATA = A_tmp.transpose() * cov_tmp_ldlt.solve(A_tmp);
                MatXX ATb = A_tmp.transpose() * cov_tmp_ldlt.solve(b_tmp);

                unsigned long index = 3 + 3 * (i - 1);
                A.block<3, 3>(0, 0) += ATA.block<3, 3>(0, 0);
                A.block<3, 6>(0, index) += ATA.block<3, 6>(0, 3);
                A.block<6, 3>(index, 0) += ATA.block<6, 3>(3, 0);
                A.block<6, 6>(index, index) += ATA.block<6, 6>(3, 3);
                b.block<3, 1>(0, 0) += ATb.block<3, 1>(0, 0);
                b.block<6, 1>(index, 0) += ATb.block<6, 1>(3, 0);
            }
        }

        // 当前imu
        {
            // 提取位姿
            auto &&pose_i = _windows.newest()->vertex_pose->get_parameters();
            auto &&pose_j = _imu_node->vertex_pose->get_parameters();

            Vec3 p_i {pose_i(0), pose_i(1), pose_i(2)};
            Qd q_i {pose_i(6), pose_i(3), pose_i(4), pose_i(5)};

            Vec3 p_j {pose_j(0), pose_j(1), pose_j(2)};
            Qd q_j {pose_j(6), pose_j(3), pose_j(4), pose_j(5)};

            Vec3 tij = p_j - p_i;
            Qd qij = (q_i.inverse() * q_j).normalized();

            /* 非线性问题的边 */
            shared_ptr<EdgeAlign> nonlinear_edge = make_shared<EdgeAlign>(*_imu_node->imu_integration,
                                                                          tij,
                                                                          qij,
                                                                          q_i);
            nonlinear_edge->add_vertex(vertex_scale);
            nonlinear_edge->add_vertex(vertex_q_wb0);
            nonlinear_edge->add_vertex(vertex_v[_windows.size() - 1]);
            nonlinear_edge->add_vertex(vertex_v[_windows.size()]);
            nonlinear_edge->add_vertex(vertex_ba);
            nonlinear_edge->add_vertex(vertex_bg);

            // 加入到problem
            nonlinear_problem.add_edge(nonlinear_edge);
            nonlinear_problem.add_vertex(vertex_v[_windows.size()]);

            // Ax = b
            {
                double dt = _imu_node->imu_integration->get_sum_dt();
                Mat33 R_0i_T = q_i.toRotationMatrix().transpose();

                MatXX A_tmp(MatXX::Zero(6, 9));
                MatXX b_tmp(MatXX::Zero(6, 1));

                A_tmp.block<3, 3>(3, 0) = -R_0i_T * dt;
                A_tmp.block<3, 3>(0, 0) =  (0.5 * dt) * A_tmp.block<3, 3>(3, 0);
                A_tmp.block<3, 3>(3, 3) = -R_0i_T;
                A_tmp.block<3, 3>(0, 3) = dt * A_tmp.block<3, 3>(3, 3);
                A_tmp.block<3, 3>(3, 6) = R_0i_T;

                b_tmp.topRows<3>() = _imu_node->imu_integration->get_delta_p() - R_0i_T * tij;
                b_tmp.bottomRows<3>() = _imu_node->imu_integration->get_delta_v();

                auto &&cov = _imu_node->imu_integration->get_covariance();
                Mat66 cov_tmp;
                cov_tmp.block<3, 3>(0, 0) = cov.block<3, 3>(0, 0);
                cov_tmp.block<3, 3>(0, 3) = cov.block<3, 3>(0, 6);
                cov_tmp.block<3, 3>(3, 3) = cov.block<3, 3>(6, 6);
                cov_tmp.block<3, 3>(3, 0) = cov.block<3, 3>(6, 0);
                auto &&cov_tmp_ldlt = cov_tmp.ldlt();
                MatXX ATA = A_tmp.transpose() * cov_tmp_ldlt.solve(A_tmp);
                MatXX ATb = A_tmp.transpose() * cov_tmp_ldlt.solve(b_tmp);

                unsigned long index = 3 + 3 * (_windows.size() - 1);
                A.block<3, 3>(0, 0) += ATA.block<3, 3>(0, 0);
                A.block<3, 6>(0, index) += ATA.block<3, 6>(0, 3);
                A.block<6, 3>(index, 0) += ATA.block<6, 3>(3, 0);
                A.block<6, 6>(index, index) += ATA.block<6, 6>(3, 3);
                b.block<3, 1>(0, 0) += ATb.block<3, 1>(0, 0);
                b.block<6, 1>(index, 0) += ATb.block<6, 1>(3, 0);
            }
        }

        // 求解线性问题
        VecX x = A.ldlt().solve(b);

        Vec3 v_nav_est = x.segment<3>(0);
        Vec3 v_nav_true = IMUIntegration::get_gravity();
        double v_nav_est2 = v_nav_est.squaredNorm();
        double v_nav_true2 = v_nav_true.squaredNorm();

        if (v_nav_est2 > 1.1 * 1.1 * v_nav_true2 || v_nav_est2 < 0.9 * 0.9 * v_nav_true2) { // 重力加速度的模值与先验差异不得大于10%
            return false;
        }

        // 把线性问题的求解结果当成非线性问题的初值
        double norm2 = sqrt(v_nav_est.squaredNorm() * v_nav_true.squaredNorm());
        double cos_psi = v_nav_est.dot(v_nav_true);
        Vec3 sin_psi = v_nav_est.cross(v_nav_true);
        Qd q0_est(norm2 + cos_psi, sin_psi(0), sin_psi(1), sin_psi(2));
        q0_est.normalize();

        Vec4 q_wb0_est {q0_est.w(), q0_est.x(), q0_est.y(), q0_est.z()};
        vertex_q_wb0->set_parameters(q_wb0_est);

        vertex_scale->set_parameters(Vec1(1.));

        for (unsigned long i = 0; i < _windows.size() + 1; ++i) {
            vertex_v[i]->set_parameters(x.segment<3>(3 + 3 * i));
            std::cout << "v[" << i << "] = " << x.segment<3>(3 + 3 * i).transpose() << std::endl;
        }

        // 求解非线性问题
        nonlinear_problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
        vertex_ba->set_fixed();
        vertex_bg->set_fixed();
        nonlinear_problem.solve(30);

        // 把求解后的结果赋值到顶点中
        Qd q_wb0 {vertex_q_wb0->get_parameters()[0], vertex_q_wb0->get_parameters()[1], vertex_q_wb0->get_parameters()[2], vertex_q_wb0->get_parameters()[3]};

        // windows
        for (unsigned long i = 0; i < _windows.size(); ++i) {
            auto &pose = _windows[i]->vertex_pose->parameters();
            Vec3 p {pose(0), pose(1), pose(2)};
            p = q_wb0 * p;
            pose(0) = p.x();
            pose(1) = p.y();
            pose(2) = p.z();

            Qd q {pose(6), pose(3), pose(4), pose(5)};
            q = q_wb0 * q;
            pose(3) = q.x();
            pose(4) = q.y();
            pose(5) = q.z();
            pose(6) = q.w();

            auto &motion = _windows[i]->vertex_motion->parameters();
            Vec3 v = vertex_v[i]->get_parameters();
            v = q_wb0 * v;
            motion << v[0], v[1], v[2],
                      vertex_ba->get_parameters()[0], vertex_ba->get_parameters()[1],vertex_ba->get_parameters()[2],
                      vertex_bg->get_parameters()[0], vertex_bg->get_parameters()[1],vertex_bg->get_parameters()[2];
        }

        // 当前imu
        {
            auto &pose = _imu_node->vertex_pose->parameters();
            Vec3 p {pose(0), pose(1), pose(2)};
            p = q_wb0 * p;
            pose(0) = p.x();
            pose(1) = p.y();
            pose(2) = p.z();

            Qd q {pose(6), pose(3), pose(4), pose(5)};
            q = (q_wb0 * q).normalized();
            pose(3) = q.x();
            pose(4) = q.y();
            pose(5) = q.z();
            pose(6) = q.w();

            auto &motion = _imu_node->vertex_motion->parameters();
            Vec3 v = vertex_v[_windows.size()]->get_parameters();
            v = q_wb0 * v;
            motion << v[0], v[1], v[2],
                    vertex_ba->get_parameters()[0], vertex_ba->get_parameters()[1],vertex_ba->get_parameters()[2],
                    vertex_bg->get_parameters()[0], vertex_bg->get_parameters()[1],vertex_bg->get_parameters()[2];
        }

        return true;
    }
}