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

class Simulator {
public:
    explicit Simulator(double dt, double w=0.5, double r=10.) : _dt(dt), _w(w), _r(r) {
        _ba.setZero();
        _bg.setZero();

        // landmarks生成
        double deg2rad = EIGEN_PI / 180.;
        for (int i = 0; i < 360; ++i) {
            double angle = double(i % 360) * deg2rad;
            double cos_ang = cos(angle);
            double sin_ang = sin(angle);
            // 轴向
            for (int j = 0; j < 5; ++j) {
                double l = r + double(j);
                for (int k = 0; k < 5; ++k) {
                    /*
                     * 把 p = (0, l, k), 旋转R
                     * 其中,
                     * R = [cos(theta) -sin(theta) 0
                     *      sin(theta) cos(theta) 0
                     *      0 0 1]
                     * */
                    landmarks[i][j][k] = {-l * sin_ang, l * cos_ang, double(k) - 2.};
                }
            }
        }
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

    unsigned long get_landmark_id(unsigned int i, unsigned int j, unsigned int k) {
        return i + j * 1000 + k * 10000;
    }

    unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> get_landmarks_per_pose(double theta, const Vec3 &t_wi) {
        static double rad2deg = 180. / EIGEN_PI;
        Qd q_wi {cos(0.5 * theta), 0., 0., sin(0.5 * theta)};

        Vec3 p_i, p_c;
        Vec7 f;
        unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> landmarks_map;

        int ang = (int(theta * rad2deg) + 90 + 360) % 360;
        for (int i = -10; i <= 10; ++i) {
            int index = (ang + i + 360) % 360;
            for (int j = 0; j < 5; ++j) {
                for (int k = 0; k < 5; ++k) {
                    p_i = q_wi.inverse() * (landmarks[index][j][k] - t_wi);
                    p_c = q_ic.inverse() * (p_i - t_ic);
                    f << p_c.x() / p_c.z(), p_c.y() / p_c.z(), 1., 0., 0., 0., 0.;
                    landmarks_map[get_landmark_id(index, j, k)].emplace_back(0, f);
                }
            }
        }

        return landmarks_map;
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

public:
    Vec3 landmarks[360][5][5];

public:
    Qd q_ic {cos(-0.5 * EIGEN_PI * 0.5), 0., sin(-0.5 * EIGEN_PI * 0.5), 0.};
    Vec3 t_ic {0., 0., 0.};
};

int main() {
    vins::Estimator estimator;

    double dt = 0.005;
    unsigned int num_data = 6000 + 1;
    static Simulator simulator(dt);
    simulator.generate_data(num_data);

    for (unsigned int n = 0; n < num_data; ++n) {
        estimator.process_imu(dt, simulator._a_buff[n], simulator._w_buff[n]);
        if (n % 20 == 0) {
            std::cout << "p_gt: " << (simulator._p_buff[n] - simulator._p_buff[0]).transpose() << std::endl;
            auto &&f_per_imu = simulator.get_landmarks_per_pose(simulator._theta_buff[n], simulator._p_buff[n]);
            estimator.process_image(f_per_imu, dt * double(n));
        }
    }


    return 0;
}