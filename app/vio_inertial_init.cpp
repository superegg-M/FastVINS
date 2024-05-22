//
// Created by Cain on 2024/5/22.
//

#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Dense>

#include "../modules/vo_test/estimator.h"
#include "../modules/vo_test/edge/edge_align.h"
#include "../modules/vo_test/edge/edge_align_linear.h"
#include "../modules/vo_test/vertex/vertex_vec1.h"
#include "../modules/vo_test/vertex/vertex_scale.h"
#include "../modules/vo_test/vertex/vertex_spherical.h"
#include "../modules/vo_test/vertex/vertex_velocity.h"
#include "../modules/vo_test/vertex/vertex_bias.h"
#include "../lib/backend/loss_function.h"

using namespace graph_optimization;
using namespace std;


static unordered_map<unsigned long, Vec3> landmark_map;

constexpr static unsigned int num_steps = 20;
static Qd q_per_imu[num_steps];
static Vec3 t_per_imu[num_steps];
static unordered_map<unsigned long, vector<pair<unsigned long, Vec3>>> landmarks_per_imu[num_steps];
static unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> f_per_imu[num_steps];

constexpr static unsigned int nun_cameras = 1;
static Qd q_ic[nun_cameras];
static Vec3 t_ic[nun_cameras];

class ImuAlignTest {
public:
    explicit ImuAlignTest(double dt, double w=0.5, double r=10.) : _dt(dt), _w(w), _r(r) {
        _ba.setZero();
        _bg.setZero();
    }

    void generate_data(unsigned int num_data) {
        _theta_buff.resize(num_data);
        _p_buff.resize(num_data);
        _v_buff.resize(num_data);
        _a_buff.resize(num_data);
        _w_buff.resize(num_data);
        for (unsigned int i = 0; i < num_data; ++i) {
            _theta_buff[i] = double(i) * _dt * _w;

            _p_buff[i].x() = _r * cos(_theta_buff[i]);
            _p_buff[i].y() = _r * sin(_theta_buff[i]);
            _p_buff[i].z() = 0.;

            _v_buff[i].x() = -_r * _w * sin(_theta_buff[i]);
            _v_buff[i].y() = _r * _w * cos(_theta_buff[i]);
            _v_buff[i].z() = 0.;

            _a_buff[i].x() = -_r * _w * _w + _ba.x();
            _a_buff[i].y() = 0. + _ba.y();
            _a_buff[i].z() = 9.8 + _ba.z();

            _w_buff[i].x() = 0. + _bg.x();
            _w_buff[i].y() = 0. + _bg.y();
            _w_buff[i].z() = _w + _bg.z();
        }
    }

public:
    vector<double> _theta_buff;
    vector<Vec3> _p_buff;
    vector<Vec3> _v_buff;
    vector<Vec3> _a_buff;
    vector<Vec3> _w_buff;

public:
    Vec3 _ba;
    Vec3 _bg;

public:
    double _dt;
    double _w;
    double _r;

};

int main() {
    static ImuAlignTest imu_align(0.005);
    imu_align.generate_data(201);

    Vec3 ba_init {0., 0., 0.};
    Vec3 bg_init {0., 0., 0.};
    double scale = 10.;
    Qd q0 {cos(0.5 * 0.2), sin(0.5 * 0.2), 0., 0.};

    vector<Vec3> t;
    vector<Qd> q;
    vector<Vec3> v;

    Problem problem;
    shared_ptr<VertexScale> vertex_scale = make_shared<VertexScale>();
    shared_ptr<VertexSpherical> vertex_q_wb0 = make_shared<VertexSpherical>();
    shared_ptr<VertexBias> vertex_ba = make_shared<VertexBias>();
    shared_ptr<VertexBias> vertex_bg = make_shared<VertexBias>();
    vector<shared_ptr<VertexVelocity>> vertices_v;
    shared_ptr<vins::IMUIntegration> imu_integration = nullptr;

    problem.add_vertex(vertex_scale);
    problem.add_vertex(vertex_q_wb0);
    problem.add_vertex(vertex_ba);
    problem.add_vertex(vertex_bg);

    Problem problem_linear;
    shared_ptr<VertexVec1> vertex_linear_scale = make_shared<VertexVec1>();
    shared_ptr<VertexBias> vertex_g_b0 = make_shared<VertexBias>();
    vertex_linear_scale->parameters().setZero();
//    vertex_linear_scale->set_parameters(Vec1(1.));
    vertex_g_b0->parameters().setZero();
//    vertex_g_b0->set_parameters(imu_integration->get_gravity());
    problem_linear.add_vertex(vertex_linear_scale);
    problem_linear.add_vertex(vertex_g_b0);
    shared_ptr<TrivialLoss> trivial_loss = make_shared<TrivialLoss>();

    Eigen::Matrix<double, 37, 37> A;
    Eigen::Matrix<double, 37, 1> b;
    for (unsigned int i = 0; i < 201; ++i) {
        if (imu_integration) {
//            std::cout << "1: " << imu_integration->get_delta_p().transpose() << std::endl;
//            std::cout << "2: " << imu_integration->get_delta_v().transpose() << std::endl;
            imu_integration->push_back(imu_align._dt, imu_align._a_buff[i], imu_align._w_buff[i]);
        }

        if (i % 20 == 0) {
            t.emplace_back(imu_align._p_buff[i]);
            q.emplace_back(cos(0.5 * imu_align._theta_buff[i]), 0., 0., sin(0.5 * imu_align._theta_buff[i]));
            v.emplace_back(imu_align._v_buff[i]);

            Vec3 v_init = imu_align._v_buff[i];
            vertices_v.emplace_back(make_shared<VertexVelocity>());
            vertices_v.back()->parameters().setZero();
//            vertices_v.back()->set_parameters(v_init);

            // 非线性
            problem.add_vertex(vertices_v.back());

            // 线性
            problem_linear.add_vertex(vertices_v.back());

            if (imu_integration) {
                Vec3 ti = q0.inverse() * t[t.size() - 2];
                Vec3 tj = q0.inverse() * t[t.size() - 1];
                Qd qi = (q0.inverse() * q[q.size() - 2]).normalized();
                Qd qj = (q0.inverse() * q[q.size() - 1]).normalized();

                Vec3 tij = (tj - ti) / scale;
                Qd qij = (qi.inverse() * qj).normalized();

                // 非线性
//                Vec3 v_gauss = tij / imu_integration->get_sum_dt();
//                vertices_v[vertices_v.size() - 2]->set_parameters(v_gauss);
//                vertices_v[vertices_v.size() - 1]->set_parameters(v_gauss);
                shared_ptr<EdgeAlign> edge = make_shared<EdgeAlign>(*imu_integration,
                                                                    tij,
                                                                    qij,
                                                                    qi);
                edge->add_vertex(vertex_scale);
                edge->add_vertex(vertex_q_wb0);
                edge->add_vertex(vertices_v[vertices_v.size() - 2]);
                edge->add_vertex(vertices_v[vertices_v.size() - 1]);
                edge->add_vertex(vertex_ba);
                edge->add_vertex(vertex_bg);
                problem.add_edge(edge);

                // 线性
                shared_ptr<EdgeAlignLinear> edge_linear = make_shared<EdgeAlignLinear>(*imu_integration,
                                                                                       tij,
                                                                                       qij,
                                                                                       qi);
                edge_linear->set_loss_function(trivial_loss);
                edge_linear->add_vertex(vertex_linear_scale);
                edge_linear->add_vertex(vertex_g_b0);
                edge_linear->add_vertex(vertices_v[vertices_v.size() - 2]);
                edge_linear->add_vertex(vertices_v[vertices_v.size() - 1]);
                problem_linear.add_edge(edge_linear);

//                {
//                    Vec9 residual;
//                    Vec3 v_i = v[v.size() - 2];
//                    Vec3 v_j = v[v.size() - 1];
//                    double dt = imu_integration->get_sum_dt();
//                    Vec3 g_b0 = q0.inverse() * imu_integration->get_gravity();
//                    residual.head<3>() = qi.inverse() * (scale * tij - (v_i + 0.5 * dt * g_b0) * dt) - imu_integration->get_delta_p();
//                    residual.segment<3>(3) = 2. * (imu_integration->get_delta_q().inverse() * qij).vec();
//                    residual.tail<3>() = qi.inverse() * (v_j - (v_i + g_b0 * dt)) - imu_integration->get_delta_v();
//                    std::cout << "residual: " << residual << std::endl;
//                }

                imu_integration = make_shared<vins::IMUIntegration>(imu_align._a_buff[i - 1],
                                                                    imu_align._w_buff[i - 1],
                                                                    ba_init,
                                                                    bg_init);
            } else {
                imu_integration = make_shared<vins::IMUIntegration>(imu_align._a_buff[i],
                                                                    imu_align._w_buff[i],
                                                                    ba_init,
                                                                    bg_init);
            }
        }
    }

    // 线性
    problem_linear.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
    problem_linear.solve(15);

    Vec3 g_b0 = vertex_g_b0->get_parameters();
//    Vec3 v_nav_est = (q0.inverse() * q[0]).normalized() * g_b0;
    Vec3 v_nav_est = g_b0;

    double norm2 = sqrt(v_nav_est.squaredNorm() * imu_integration->get_gravity().squaredNorm());
    double cos_psi = v_nav_est.dot(imu_integration->get_gravity());
    Vec3 sin_psi = v_nav_est.cross(imu_integration->get_gravity());
    Qd q0_est(norm2 + cos_psi, sin_psi(0), sin_psi(1), sin_psi(2));
    q0_est.normalize();

    std::cout << "linear scale: " << vertex_linear_scale->get_parameters().transpose() << std::endl;
    std::cout << "linear g_b0: " << vertex_g_b0->get_parameters().transpose() << std::endl;
    std::cout << "linear q_wb0: " << q0_est.w() << ", " << q0_est.x() << ", " << q0_est.y() << ", " << q0_est.z() << std::endl;

    // 线性转非线性
    vertex_scale->set_parameters(vertex_linear_scale->get_parameters());
    Vec4 q_wb0_est {q0_est.w(), q0_est.x(), q0_est.y(), q0_est.z()};
    vertex_q_wb0->set_parameters(q_wb0_est);
    for (auto &vertex : vertices_v) {
        vertex->set_parameters(vertex->get_parameters() / vertex_linear_scale->get_parameters()[0]);
    }

    // 非线性
    problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
    vertex_ba->set_fixed();
    vertex_bg->set_fixed();
    problem.solve(30);
    std::cout << "nonlinear scale: " << vertex_scale->get_parameters().transpose() << std::endl;
    std::cout << "nonlinear q_wb0: " << vertex_q_wb0->get_parameters().transpose() << std::endl;
    std::cout << "nonlinear ba: " << vertex_ba->get_parameters().transpose() << std::endl;
    std::cout << "nonlinear bg: " << vertex_bg->get_parameters().transpose() << std::endl;

    // Ground truth
    std::cout << "ground truth scale: " << scale << std::endl;
    std::cout << "ground truth q_wb0: " << q0.w() << ", " << q0.x() << ", " << q0.y() << ", " << q0.z() << std::endl;

    return 0;
}