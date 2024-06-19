//
// Created by Cain on 2024/6/11.
//

#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Dense>

#include "../lib/backend/loss_function.h"
#include "../modules/vins_stereo/estimator.h"

using namespace graph_optimization;
using namespace std;

class Simulator {
public:
    explicit Simulator(unsigned int num_data, double dt, double v=1., double r=0.1) : _num_data(num_data), _dt(dt), _v(v), _r(r) {
        _theta_buff.resize(num_data);
        _p_buff.resize(num_data);
        _v_buff.resize(num_data);
        _a_buff.resize(num_data);
        _w_buff.resize(num_data);

        _ba.setZero();
        _bg.setZero();
    }

    void generate_data() {
        for (unsigned int i = 0; i < _num_data; ++i) {
            double t = double(i) * _dt;
            _theta_buff[i] = _r * sin(t);
            Qd q_wi {cos(0.5 * _theta_buff[i]), 0., 0., sin(0.5 * _theta_buff[i])};

            _p_buff[i].x() = _v * t;
            _p_buff[i].y() = 1. - cos(t);
            _p_buff[i].z() = 0.;

            _v_buff[i].x() = _v;
            _v_buff[i].y() = sin(t);
            _v_buff[i].z() = 0.;

            _a_buff[i].x() = 0.;
            _a_buff[i].y() = cos(t);
            _a_buff[i].z() = 9.8;
            _a_buff[i] = q_wi.inverse() * _a_buff[i];

            _w_buff[i].x() = 0. + _bg.x();
            _w_buff[i].y() = 0. + _bg.y();
            _w_buff[i].z() = _r * cos(t) + _bg.z();
        }

        double t = double(_num_data) * _dt;
        double s = _v * t;
        std::uniform_real_distribution<double> x_rand(0., s);
        std::uniform_real_distribution<double> y_rand(10., 20.);
        std::uniform_real_distribution<double> z_rand(-5., 5.);
        for (auto &landmark : _landmarks) {
            landmark.x() = x_rand(_generator);
            landmark.y() = y_rand(_generator);
            landmark.z() = z_rand(_generator);
        }
    }

    unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> get_landmarks_per_pose(unsigned int id) {
//        array<unsigned long, 150> local_index_map {};
//        vector<unsigned long> global_index_map(NUM_LANDMARKS);
//        for (unsigned long k = 0; k < NUM_LANDMARKS; ++k) {
//            global_index_map[k] = k;
//        }
//        for (unsigned int k = 0; k < 150; ++k) {
//            std::uniform_int_distribution<unsigned int> dist(0, global_index_map.size() - 1);
//            unsigned int rand_i = dist(_generator);
//            auto index = global_index_map[rand_i];
//            local_index_map[k] = index;
//
//            global_index_map[rand_i] = global_index_map.back();
//            global_index_map.pop_back();
//        }

        Qd q_wi {cos(0.5 * _theta_buff[id]), 0., 0., sin(0.5 * _theta_buff[id])};
        Vec3 t_wi = _p_buff[id];

        Vec3 p_i, p_c;
        Vec7 f;
        unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> landmarks_map;
        for (unsigned i = 0; i < NUM_LANDMARKS; ++i) {
            p_i = q_wi.inverse() * (_landmarks[i] - t_wi);

            // 左目
            p_c = q_ic.inverse() * (p_i - t_ic);
            f << p_c.x() / p_c.z(), p_c.y() / p_c.z(), 1., 0., 0., 0., 0.;
            landmarks_map[i].emplace_back(0, f);

            // 右目
            p_c = q_ic.inverse() * (p_i - t_ic - b);
            f << p_c.x() / p_c.z(), p_c.y() / p_c.z(), 1., 0., 0., 0., 0.;
            landmarks_map[i].emplace_back(1, f);
        }
//        for (auto &i : local_index_map) {
//            p_i = q_wi.inverse() * (_landmarks[i] - t_wi);
//            p_c = q_ic.inverse() * (p_i - t_ic);
//            f << p_c.x() / p_c.z(), p_c.y() / p_c.z(), 1., 0., 0., 0., 0.;
//            landmarks_map[i].emplace_back(0, f);
//        }

        return landmarks_map;
    }

public:
    constexpr static unsigned int NUM_LANDMARKS {100};

public:
    unsigned int _num_data;
    vector<double> _theta_buff;
    vector<Vec3> _p_buff;
    vector<Vec3> _v_buff;
    vector<Vec3> _a_buff;
    vector<Vec3> _w_buff;

public:
    Vec3 _ba {0., 0., 0.};
    Vec3 _bg {0.0, 0.0, 0.0};

public:
    double _dt;
    double _v;
    double _r;
    double _t;

public:
    Vec3 _landmarks[NUM_LANDMARKS];
    std::default_random_engine _generator;

public:
    Qd q_ic {cos(-0.5 * double(EIGEN_PI) * 0.5), sin(-0.5 * double(EIGEN_PI) * 0.5), 0., 0.};
    Vec3 t_ic {0., 0., 0.};
    Vec3 b {0.1, 0., 0.};
};

int main() {
    vins::Estimator estimator;

    double dt = 0.005;
    unsigned int num_data = 6000 + 1;
    static Simulator simulator(num_data, dt);
    simulator.generate_data();

    for (unsigned int n = 0; n < num_data; ++n) {
        estimator.process_imu(dt, simulator._a_buff[n], simulator._w_buff[n]);
        if (n % 20 == 0) {
            Qd q_wi {cos(0.5 * simulator._theta_buff[n]), 0., 0., sin(0.5 * simulator._theta_buff[n])};
            std::cout << "q_gt: " << q_wi.w() << ", " << q_wi.x() << ", " << q_wi.y() << ", " << q_wi.z()<< std::endl;
            std::cout << "p_gt: " << (simulator._p_buff[n] - simulator._p_buff[0]).transpose() << std::endl;
            std::cout << "v_gt: " << (simulator._v_buff[n]).transpose() << std::endl;
            auto &&f_per_imu = simulator.get_landmarks_per_pose(n);
            estimator.process_image(f_per_imu, dt * double(n));
        }
    }


    return 0;
}