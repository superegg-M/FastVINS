//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"
#include "../vertex/vertex_pose.h"

#include "tic_toc/tic_toc.h"
#include "backend/eigen_types.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    bool Estimator::compute_essential_matrix(Mat33 &R, Vec3 &t, ImuNode *imu_i, ImuNode *imu_j,
                                             bool is_init_landmark, unsigned int max_iters) {
        constexpr static double th_e2 = 3.841;
        constexpr static double th_score = 5.991;
        constexpr static unsigned long th_count = 20;

        unsigned long max_num_points = max(imu_i->features_in_cameras.size(), imu_j->features_in_cameras.size());
        vector<pair<const Vec3*, const Vec3*>> match_pairs;
        match_pairs.reserve(max_num_points);
        vector<unsigned long> feature_ids;
        feature_ids.reserve(max_num_points);

        // 找出匹配对
        for (auto &feature_in_cameras : imu_i->features_in_cameras) {
            unsigned long feature_id = feature_in_cameras.first;
            auto &&it = imu_j->features_in_cameras.find(feature_id);
            if (it == imu_j->features_in_cameras.end()) {
                continue;
            }
            match_pairs.emplace_back(&feature_in_cameras.second[0].second, &it->second[0].second);
            feature_ids.emplace_back(feature_id);
        }

        // 匹配对必须大于一定数量
        unsigned long num_points = match_pairs.size();
        if (num_points < th_count) {
            return false;
        }

        // 计算平均视差
        double average_parallax = 0.;
        for (auto &match_pair : match_pairs) {
            double du = match_pair.first->x() - match_pair.second->x();
            double dv = match_pair.first->y() - match_pair.second->y();
            average_parallax += max(abs(du), abs(dv));
        }
        average_parallax /= double(num_points);
        std::cout << "average_parallax = " << average_parallax << std::endl;

        // 平均视差必须大于一定值
        if (average_parallax * 460. < 30.) {
            return false;
        }

        // 归一化变换参数
        Mat33 Ti, Tj;
        double meas_x_i = 0., meas_y_i = 0.;
        double dev_x_i = 0., dev_y_i = 0.;
        double meas_x_j = 0., meas_y_j = 0.;
        double dev_x_j = 0., dev_y_j = 0.;

        // 计算均值
        for (auto &match_pair : match_pairs) {
            meas_x_i += match_pair.first->x();
            meas_y_i += match_pair.first->y();
            meas_x_j += match_pair.second->x();
            meas_y_j += match_pair.second->y();
        }
        meas_x_i /= double(num_points);
        meas_y_i /= double(num_points);
        meas_x_j /= double(num_points);
        meas_y_j /= double(num_points);

        // 计算Dev
        for (auto &match_pair : match_pairs) {
            dev_x_i += abs(match_pair.first->x() - meas_x_i);
            dev_y_i += abs(match_pair.first->y() - meas_y_i);
            dev_x_j += abs(match_pair.second->x() - meas_x_j);
            dev_y_j += abs(match_pair.second->y() - meas_y_j);
        }
        dev_x_i /= double(num_points);
        dev_y_i /= double(num_points);
        dev_x_j /= double(num_points);
        dev_y_j /= double(num_points);

        Ti << 1. / dev_x_i, 0., -meas_x_i / dev_x_i,
                0., 1. / dev_y_i, -meas_y_i / dev_y_i,
                0., 0., 1.;
        Tj << 1. / dev_x_j, 0., -meas_x_j / dev_x_j,
                0., 1. / dev_y_j, -meas_y_j / dev_y_j,
                0., 0., 1.;

        // 归一化后的点
        vector<pair<Vec2, Vec2>> normal_match_pairs(num_points);
        for (unsigned long k = 0; k < num_points; ++k) {
            normal_match_pairs[k].first.x() = (match_pairs[k].first->x() - meas_x_i) / dev_x_i;
            normal_match_pairs[k].first.y() = (match_pairs[k].first->y() - meas_y_i) / dev_y_i;
            normal_match_pairs[k].second.x() = (match_pairs[k].second->x() - meas_x_j) / dev_x_j;
            normal_match_pairs[k].second.y() = (match_pairs[k].second->y() - meas_y_j) / dev_y_j;
        }

        // 构造随机index batch
        std::random_device rd;
        std::mt19937 gen(rd());
        vector<array<unsigned long, 8>> point_indices_set(max_iters);  // TODO: 设为静态变量

        array<unsigned long, 8> local_index_map {};
        vector<unsigned long> global_index_map(num_points);
        for (unsigned long k = 0; k < num_points; ++k) {
            global_index_map[k] = k;
        }

        for (unsigned int n = 0; n < max_iters; ++n) {
            for (unsigned int k = 0; k < 8; ++k) {
                std::uniform_int_distribution<unsigned int> dist(0, global_index_map.size() - 1);
                unsigned int rand_i = dist(gen);
                auto index = global_index_map[rand_i];
                point_indices_set[n][k] = index;
                local_index_map[k] = index;

                global_index_map[rand_i] = global_index_map.back();
                global_index_map.pop_back();
            }

            for (unsigned int k = 0; k < 8; ++k) {
                global_index_map.emplace_back(local_index_map[k]);
            }
        }

        // RANSAC: 计算本质矩阵
        Mat33 best_E;
        double best_score = 0.;
        unsigned int best_iter = 0;
        unsigned long num_outliers = 0;
        vector<bool> is_outliers(num_points, false);
        for (unsigned int n = 0; n < max_iters; ++n) {
            // 八点法算E
            Mat89 D;
            for (unsigned int k = 0; k < 8; ++k) {
                unsigned int index = point_indices_set[n][k];
                double u1 = normal_match_pairs[index].first.x();
                double v1 = normal_match_pairs[index].first.y();
                double u2 = normal_match_pairs[index].second.x();
                double v2 = normal_match_pairs[index].second.y();
                D(k, 0) = u1 * u2;
                D(k, 1) = u1 * v2;
                D(k, 2) = u1;
                D(k, 3) = v1 * u2;
                D(k, 4) = v1 * v2;
                D(k, 5) = v1;
                D(k, 6) = u2;
                D(k, 7) = v2;
                D(k, 8) = 1.;
            }
            Eigen::JacobiSVD<Mat89> D_svd(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Vec9 e = D_svd.matrixV().col(8);
            Mat33 E_raw;
            E_raw << e(0), e(1), e(2),
                    e(3), e(4), e(5),
                    e(6), e(7), e(8);
            Eigen::JacobiSVD<Mat33> E_svd(E_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Vec3 s = E_svd.singularValues();
            s(0) = 0.5 * (s(0) + s(1));
            s(1) = s(0);
            s(2) = 0.;
            Mat33 E = E_svd.matrixU() * s.asDiagonal() * E_svd.matrixV().transpose();
            double e00 = E(0, 0), e01 = E(0, 1), e02 = E(0, 2),
                    e10 = E(1, 0), e11 = E(1, 1), e12 = E(1, 2),
                    e20 = E(2, 0), e21 = E(2, 1), e22 = E(2, 2);

            // 计算分数
            double score = 0.;
            for (unsigned long k = 0; k < num_points; ++k) {
                bool is_outlier = false;

                double u1 = normal_match_pairs[k].first.x();
                double v1 = normal_match_pairs[k].first.y();
                double u2 = normal_match_pairs[k].second.x();
                double v2 = normal_match_pairs[k].second.y();

                double a = u1 * e00 + v1 * e10 + e20;
                double b = u1 * e01 + v1 * e11 + e21;
                double c = u1 * e02 + v1 * e12 + e22;
                double num = a * u2 + b * v2 + c;
                double e2 = num * num / (a * a + b * b);
                if (e2 > th_e2) {
                    is_outlier = true;
                } else {
                    score += th_score - e2;
                }

                a = u2 * e00 + v2 * e01 + e02;
                b = u2 * e10 + v2 * e11 + e12;
                c = u2 * e20 + v2 * e21 + e22;
                num = u1 * a + v1 * b + c;
                e2 = num * num / (a * a + b * b);
                if (e2 > th_e2) {
                    is_outlier = true;
                } else {
                    score += th_score - e2;
                }

                is_outliers[k] = is_outlier;
                if (is_outlier) {
                    ++num_outliers;
                }
            }

            if (score > best_score) {
                best_score = score;
                best_iter = n;
                best_E = E;
            }
        }

        // outlier的点过多
        if (10 * num_outliers > 5 * num_points) {
            std::cout << "10 * num_outliers > 5 * num_points" << std::endl;
            return false;
        }

        // 从E中还原出R, t
        best_E = Ti.transpose() * best_E * Tj;
        Eigen::JacobiSVD<Mat33> E_svd(best_E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat33 V = E_svd.matrixV();
        Mat33 U1 = E_svd.matrixU();

        if (V.determinant() < 0.) {
            V *= -1.;
        }
        if (U1.determinant() < 0.) {
            U1 *= -1.;
        }

        Vec3 t1 = U1.col(2);
        t1 = t1 / t1.norm();
        Vec3 t2 = -t1;

        U1.col(0).swap(U1.col(1));
        Mat33 U2 = U1;
        U1.col(1) *= -1.;
        U2.col(0) *= -1.;

        Mat33 R1 = U1 * V.transpose();
        Mat33 R2 = U2 * V.transpose();

        // 进行三角化，通过深度筛选出正确的R, t
        auto tri = [&](const Vec3 *point_i, const Vec3 *point_j, const Mat33 &R, const Vec3 &t, Vec3 &p) -> bool {
            Vec3 RTt = R.transpose() * t;
            Mat43 A;
            A.row(0) << 1., 0., -point_i->x();
            A.row(1) << 0., 1., -point_i->y();
            A.row(2) = R.col(0).transpose() - point_j->x() * R.col(2).transpose();
            A.row(3) = R.col(1).transpose() - point_j->y() * R.col(2).transpose();
            Vec4 b;
            b << 0., 0., RTt[0] - RTt[2] * point_j->x(), RTt[1] - RTt[2] * point_j->y();

//            Mat33 ATA = A.transpose() * A;
//            Vec3 ATb = A.transpose() * b;
//            auto &&ATA_ldlt = ATA.ldlt();
//            if (ATA_ldlt.info() == Eigen::Success) {
//                p = ATA_ldlt.solve(ATb);
//                return true;
//            } else {
//                return false;
//            }

            // 使用QR分解求解最小二乘问题，这样数值精度更加稳定
            p = A.fullPivHouseholderQr().solve(b);
            return true;
        };

        auto tri_all_points = [&](const Mat33 &R, const Vec3 &t, vector<pair<bool, Vec3>> &points) -> unsigned long {
            unsigned long succeed_count = 0;
            for (unsigned long k = 0; k < num_points; ++k) {
                if (is_outliers[k]) {
                    continue;
                }

                points[k].first = tri(match_pairs[k].first, match_pairs[k].second, R, t, points[k].second);
                if (!points[k].first) {
                    continue;
                }

                points[k].first = points[k].second[2] > 0.;
//                std::cout << "points[k].second[2] = " << points[k].second[2] << std::endl;
                if (!points[k].first) {
                    continue;
                }

                Vec3 pj = R.transpose() * (points[k].second - t);
                points[k].first = pj[2] > 0.;
//                std::cout << "pj[2] = " << pj[2] << std::endl;
                if (!points[k].first) {
                    continue;
                }

                ++succeed_count;
            }
            return succeed_count;
        };

        vector<pair<bool, Vec3>> points_w[4];
        unsigned long succeed_points[4];
        points_w[0].resize(num_points);
        points_w[1].resize(num_points);
        points_w[2].resize(num_points);
        points_w[3].resize(num_points);

        succeed_points[0] = tri_all_points(R1, t1, points_w[0]);
        succeed_points[1] = tri_all_points(R1, t2, points_w[1]);
        succeed_points[2] = tri_all_points(R2, t1, points_w[2]);
        succeed_points[3] = tri_all_points(R2, t2, points_w[3]);

        unsigned long max_succeed_points = max(succeed_points[0], max(succeed_points[1], max(succeed_points[2], succeed_points[3])));
        unsigned long min_succeed_points = 9 * (num_points - num_outliers) / 10; // 至少要超过90%的点成功被三角化

        if (max_succeed_points < min_succeed_points) {
            std::cout << "max_succeed_points = " << max_succeed_points << ", min_succeed_points = " << min_succeed_points << std::endl;
            std::cout << "succeed_points[0] = " << succeed_points[0] << std::endl;
            std::cout << "succeed_points[1] = " << succeed_points[1] << std::endl;
            std::cout << "succeed_points[2] = " << succeed_points[2] << std::endl;
            std::cout << "succeed_points[3] = " << succeed_points[3] << std::endl;
            return false;
        }

        unsigned long lim_succeed_points = 7 * max_succeed_points / 10;
        unsigned long num_similar = 0;  // 记录有多少组解使得70%的点都能三角化
        if (succeed_points[0] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[1] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[2] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[3] > lim_succeed_points) {
            ++num_similar;
        }
        if (num_similar > 1) {  // 不允许超过1组解使得70%的点都能三角化
            std::cout << "num_similar > 1" << std::endl;
            return false;
        }

        unsigned int which_case;
        if (succeed_points[0] == max_succeed_points) {
            which_case = 0;
            R = R1;
            t = t1;
        } else if (succeed_points[1] == max_succeed_points) {
            which_case = 1;
            R = R1;
            t = t2;
        } else if (succeed_points[2] == max_succeed_points) {
            which_case = 2;
            R = R2;
            t = t1;
        } else {
            which_case = 3;
            R = R2;
            t = t2;
        }

        // 转到imu坐标系
        Qd q12;
        Vec3 t12;
        q12 = _q_ic[0] * R * _q_ic[0].inverse();
        t12 = _q_ic[0] * t - q12 * _t_ic[0] + _t_ic[0];

        R = q12;
        t = t12;

        // 把三角化的结果赋值给landmark
        if (is_init_landmark) {
            for (unsigned long k = 0; k < num_points; ++k) {
                auto &&feature_it = _feature_map.find(feature_ids[k]);
                if (feature_it == _feature_map.end()) {
                    continue;
                }

                // 没有outliers以及深度为正的点才会进行赋值
                if (!is_outliers[k] && points_w[which_case][k].first) {
                    // 转到imu系
                    Vec3 p = _q_ic[0] * points_w[which_case][k].second + _t_ic[0];

                    feature_it->second->vertex_point3d = std::make_shared<VertexPoint3d>();
                    feature_it->second->vertex_point3d->set_parameters(p);

                    feature_it->second->is_triangulated = true;
                    feature_it->second->is_outlier = false;
                } else {
                    feature_it->second->is_triangulated = false;
                    feature_it->second->is_outlier = true;
                }
            }
        }

        return true;
    }
}