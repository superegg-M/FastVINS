//
// Created by Cain on 2024/5/22.
//

#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Dense>

#include "../modules/vo_test/estimator.h"


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

static void sim_data() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-3., 3.);

    q_ic[0] = {cos(0.5 * double(EIGEN_PI) / 2.), -sin(0.5 * double(EIGEN_PI) / 2.), 0., 0.};
    t_ic[0] = {0., 0., 0.};

    double r = 10.;
    int ang_int = 3600;
    for (unsigned int i = 0; i < num_steps; ++i) {
        // 产生位姿
        double theta = double(ang_int) / 10. / 180. * double(EIGEN_PI);
        q_per_imu[i] = {cos(0.5 * theta), 0., 0., sin(0.5 * theta)};
        t_per_imu[i] = {r * sin(theta), r - r * cos(theta), 0.};

        for (int j = -200; j <= 200; ++j) {
            // 产生特征点
            auto ang_int_tmp = ang_int + j;
            unsigned long landmark_index = ang_int_tmp;
            Vec3 p_w;
            auto landmark_it = landmark_map.find(landmark_index);
            if (landmark_it == landmark_map.end()) {
                double ang_tmp = double(ang_int_tmp) / 10. / 180. * double(EIGEN_PI);
                double z = (landmark_index % 2 == 0) ? 1. : -1.;
//                double z = 0.;
                p_w = {-r * sin(ang_tmp) + dist(gen), r + r * cos(ang_tmp) + dist(gen),  dist(gen)};
                landmark_map.insert(make_pair(landmark_index, p_w));
            } else {
                p_w = landmark_it->second;
            }

            // 记录每帧的特征点
            landmarks_per_imu[i][landmark_index].emplace_back(0, p_w);

            Vec3 p_i = q_per_imu[i].inverse() * (p_w - t_per_imu[i]);
            Vec3 p_c = q_ic[0].inverse() * (p_i - t_ic[0]);
            p_c /= p_c.z();
            Vec7 f;
            f << p_c.x(), p_c.y(), 1., 0., 0., 0., 0.;
            f_per_imu[i][landmark_index].emplace_back(0, f);

//            std::cout << "feature: " << p_c << std::endl;
        }
        ang_int += 30;

//        std::cout << "angle = " << theta / double(EIGEN_PI) * 180. << std::endl;
    }
}

int main() {
    sim_data();

    unsigned int ref_index = 5;

    Qd q_wc[num_steps];
    Vec3 t_wc[num_steps];
    for (unsigned int i = 0; i < num_steps; ++i) {
        q_wc[i] = q_per_imu[i] * q_ic[0];
        t_wc[i] = t_per_imu[i] + q_per_imu[i] * t_ic[0];
    }
    Vec3 t_5_to_10_on_5 = q_per_imu[ref_index].inverse() * (t_per_imu[10] - t_per_imu[ref_index]);
    Qd q_5_10 = q_per_imu[ref_index].inverse() * q_per_imu[10];
    std::cout << t_5_to_10_on_5 << std::endl;
    std::cout << q_5_10.w() << ", " << q_5_10.x() << ", " << q_5_10.y() << ", " << q_5_10.z() << std::endl;

    vins::Estimator estimator;

    for (unsigned int i = 0; i < estimator._windows.capacity() + 1; ++i) {
        estimator.process_image(f_per_imu[i], 0.);
    }

    for (unsigned int i = 0; i < estimator._windows.size(); ++i) {
        Vec7 pose = estimator._windows[i]->vertex_pose->get_parameters();
        Vec3 t {pose(0), pose(1), pose(2)};
        Qd q {pose(6), pose(3), pose(4), pose(5)};
//        t = t_per_imu[ref_index] + q_per_imu[ref_index] * t * (t_per_imu[ref_index] - t_per_imu[estimator._windows.capacity()]).norm();
//        q = q_per_imu[ref_index] * q;
        t = t_per_imu[0] + q_per_imu[0] * t * (t_per_imu[ref_index] - t_per_imu[estimator._windows.capacity()]).norm();
        q = q_per_imu[0] * q;

        std::cout << "ground truth: " << std::endl;
        std::cout << "t = " << t_per_imu[i].transpose() << std::endl;
        std::cout << "q = " << q_per_imu[i].w() << ", "  << q_per_imu[i].x() << ", "  << q_per_imu[i].y() << ", "  << q_per_imu[i].z() << std::endl;
        std::cout << "estimate: " << std::endl;
        std::cout << "t = " << t.transpose() << std::endl;
        std::cout << "q = " << q.w() << ", "  << q.x() << ", "  << q.y() << ", "  << q.z() << std::endl;
    }

//    for (auto &feature_in_cameras : estimator._windows[5]->features_in_cameras) {
//        auto &&feature_it = estimator._feature_map.find(feature_in_cameras.first);
//        if (feature_it == estimator._feature_map.end()) {
//            continue;
//        }
//        if (feature_it->second->vertex_point3d) {
//            Vec3 p_est = feature_it->second->vertex_point3d->get_parameters();
//            p_est = t_per_imu[5] + q_per_imu[5] * p_est;
//            Vec3 p_gt = landmark_map[feature_it->first];
//
//            std::cout << "gt : " << p_gt.transpose() << std::endl;
//            std::cout << "est: " << p_est.transpose() << std::endl;
//        }
//    }

    return 0;
}