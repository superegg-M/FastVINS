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

        // 线性与非线性问题的公共顶点
        shared_ptr<VertexVelocity> vertex_v[_windows.size()];   // 速度
        for (auto &v : vertex_v) {
            v = make_shared<VertexVelocity>();
            v->parameters().setZero();
        }

        /* 线性问题, 假设ba = 0, bg = 0, 粗略的估算v, alpha, R0 */
        Problem linear_problem;     // 线性求解
        shared_ptr<TrivialLoss> trivial_loss = make_shared<TrivialLoss>();  // 不使用鲁棒核函数

        shared_ptr<VertexVec1> vertex_linear_scale = make_shared<VertexVec1>(); // 尺度因子
        vertex_linear_scale->parameters().setZero();    // 初始化为0
//        vertex_linear_scale->parameters()[0] = 1.;

        shared_ptr<VertexBias> vertex_g_b0 = make_shared<VertexBias>(); // 重力加速度
        vertex_g_b0->parameters().setZero();    // 初始化为0
//        vertex_g_b0->parameters()[0] = 0.;
//        vertex_g_b0->parameters()[1] = 0.;
//        vertex_g_b0->parameters()[2] = -9.8;

        // 把顶点加入到problem中
        linear_problem.add_vertex(vertex_linear_scale);
        linear_problem.add_vertex(vertex_g_b0);
        linear_problem.add_vertex(vertex_v[0]);

        /* 非线性问题, 以线性问题的结果作为初值, 进一步优化出v, alpha, R0, ba, bg */
        Problem nonlinear_problem;  // 非线性求解

        // 顶点
        shared_ptr<VertexScale> vertex_scale = make_shared<VertexScale>();  // 尺度因子
        shared_ptr<VertexSpherical> vertex_q_wb0 = make_shared<VertexSpherical>();  // 重力方向
        shared_ptr<VertexBias> vertex_ba = make_shared<VertexBias>();   // 加速度偏移
        shared_ptr<VertexBias> vertex_bg = make_shared<VertexBias>();   // 角速度偏移

        // 把顶点加入到problem中
        nonlinear_problem.add_vertex(vertex_scale);
        nonlinear_problem.add_vertex(vertex_q_wb0);
        nonlinear_problem.add_vertex(vertex_ba);
        nonlinear_problem.add_vertex(vertex_bg);
        nonlinear_problem.add_vertex(vertex_v[0]);

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

            /* 线性问题的边 */
            shared_ptr<EdgeAlignLinear> linear_edge = make_shared<EdgeAlignLinear>(*_windows[i]->imu_integration,
                                                                                   tij,
                                                                                   qij,
                                                                                   q_i);
            linear_edge->set_loss_function(trivial_loss);
            linear_edge->add_vertex(vertex_linear_scale);
            linear_edge->add_vertex(vertex_g_b0);
            linear_edge->add_vertex(vertex_v[i - 1]);
            linear_edge->add_vertex(vertex_v[i]);

            // 加入到problem
            linear_problem.add_edge(linear_edge);
            linear_problem.add_vertex(vertex_v[i]);

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
        }

        // 求解线性问题
        linear_problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
        linear_problem.solve(15);
        linear_problem.solve(15);

        // 通过scale和g_b0的模值判断解是否可用
        Vec3 v_nav_est = vertex_g_b0->get_parameters();
        Vec3 v_nav_true = IMUIntegration::get_gravity();
        double v_nav_est2 = v_nav_est.squaredNorm();
        double v_nav_true2 = v_nav_true.squaredNorm();
        double scale_est = vertex_linear_scale->get_parameters()[0];

        std::cout << "linear scale = " << scale_est << std::endl;

        if (scale_est < 0.) {    // 尺度必须大于0
            return false;
        }
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

        vertex_scale->set_parameters(Vec1(scale_est));

        for (auto &v : vertex_v) {
            v->set_parameters(v->get_parameters() / scale_est);
        }

        // 求解非线性问题
        nonlinear_problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
        vertex_ba->set_fixed();
        vertex_bg->set_fixed();
        nonlinear_problem.solve(15);

        // 把求解后的结果赋值到顶点中
        scale_est = vertex_scale->get_parameters()[0];
        Qd q_wb0 {vertex_q_wb0->get_parameters()[0], vertex_q_wb0->get_parameters()[1], vertex_q_wb0->get_parameters()[2], vertex_q_wb0->get_parameters()[3]};
        for (unsigned long i = 0; i < _windows.size(); ++i) {
            auto &pose = _windows[i]->vertex_pose->parameters();
            Vec3 p {pose(0), pose(1), pose(2)};
            p = q_wb0 * p * scale_est;
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
            v = q_wb0 * v * scale_est;
            motion << v[0], v[1], v[2],
                      vertex_ba->get_parameters()[0], vertex_ba->get_parameters()[1],vertex_ba->get_parameters()[2],
                      vertex_bg->get_parameters()[0], vertex_bg->get_parameters()[1],vertex_bg->get_parameters()[2];
        }

        std::cout << "nonlinear scale = " << scale_est << std::endl;

        return true;
    }
}