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

    void Estimator::epnp(ImuNode *imu_i) {
        TicToc pnp_t;
        // 读取3d, 2d点
        vector<Vec3> p_w;
        vector<Vec2> uv;
        p_w.reserve(imu_i->features_in_cameras.size());
        uv.reserve(imu_i->features_in_cameras.size());
        for (auto &feature_in_cameras : imu_i->features_in_cameras) {
            auto &&feature_it = _feature_map.find(feature_in_cameras.first);
            if (feature_it == _feature_map.end()) {
                std::cout << "Error: feature not in feature_map when running pnp" << std::endl;
                continue;
            }
            if (feature_it->second->vertex_point3d) {
                p_w.emplace_back(feature_it->second->vertex_point3d->get_parameters());
                uv.emplace_back(feature_in_cameras.second[0].second.x(), feature_in_cameras.second[0].second.y());
            }
        }

        // 世界坐标系下的控制点以及以控制点组成的基
        // 控制点到世界坐标的转换矩阵(基)
        Mat33 Transf;
        Mat33 Transf_inv;
        // 计算c_w0
        Vec3 c_w[4];
        c_w[0].setZero();
        unsigned long num_points = p_w.size();
        for (unsigned long k = 0; k < num_points; ++k) {
            c_w[0] += p_w[k];
        }
        c_w[0] /= double(num_points);

        // 计算c_w1, c_w2, c_w3
        double sqrt_n = sqrt(double(num_points));
        MatXX A;
        A.resize(num_points, 3);
        for (unsigned long k = 0; k < num_points; ++k) {
            A.row(k) = (p_w[k] - c_w[0]).transpose();
        }
        auto &&A_svd = A.jacobiSvd(Eigen::ComputeThinV);
        Vec3 s = A_svd.singularValues();
        Mat33 V = A_svd.matrixV();
        Transf.col(0) = s[0] / sqrt_n * V.col(0);
        Transf.col(1) = s[1] / sqrt_n * V.col(1);
        Transf.col(2) = s[2] / sqrt_n * V.col(2);
        Transf_inv.row(0) = sqrt_n / s[0] * V.col(0).transpose();
        Transf_inv.row(1) = sqrt_n / s[1] * V.col(1).transpose();
        Transf_inv.row(2) = sqrt_n / s[2] * V.col(2).transpose();
        c_w[1] = c_w[0] + Transf.col(0);
        c_w[2] = c_w[0] + Transf.col(1);
        c_w[3] = c_w[0] + Transf.col(2);

//        // 世界坐标到控制点的转换矩阵
//        auto Transf_lup = Transf.fullPivLu();

        // 特征点在控制点坐标系中的坐标
        vector<Vec4> alpha;
        alpha.resize(num_points);
        for (unsigned long k = 0; k < num_points; ++k) {
//            alpha[k].segment<3>(1) = Transf_lup.solve(p_w[k] - c_w[0]);4
            alpha[k].segment<3>(1) = Transf_inv * A.row(k).transpose();
            alpha[k][0] = 1. - alpha[k][1] - alpha[k][2] - alpha[k][3];
        }

        // M矩阵的计算
        MatXX M;
        M.resize(2 * num_points, 12);
        for (unsigned long k = 0; k < num_points; ++k) {
            M.row(2 * k) << alpha[k][0], 0., -uv[k][0] * alpha[k][0],
                    alpha[k][1], 0., -uv[k][0] * alpha[k][1],
                    alpha[k][2], 0., -uv[k][0] * alpha[k][2],
                    alpha[k][3], 0., -uv[k][0] * alpha[k][3];
            M.row(2 * k + 1) << 0., alpha[k][0], -uv[k][1] * alpha[k][0],
                    0., alpha[k][1], -uv[k][1] * alpha[k][1],
                    0., alpha[k][2], -uv[k][1] * alpha[k][2],
                    0., alpha[k][3], -uv[k][1] * alpha[k][3];

        }

        // SVD算零空间
        auto M_svd = M.jacobiSvd(Eigen::ComputeThinV);
        auto &&M_kernel = M_svd.matrixV();

        Vec3 dv[4][6];
        for (unsigned int i = 0; i < 4; ++i) {
            unsigned int j = 11 - i;
            dv[i][0] = M_kernel.col(j).segment<3>(0) - M_kernel.col(j).segment<3>(3);
            dv[i][1] = M_kernel.col(j).segment<3>(0) - M_kernel.col(j).segment<3>(6);
            dv[i][2] = M_kernel.col(j).segment<3>(0) - M_kernel.col(j).segment<3>(9);
            dv[i][3] = M_kernel.col(j).segment<3>(3) - M_kernel.col(j).segment<3>(6);
            dv[i][4] = M_kernel.col(j).segment<3>(3) - M_kernel.col(j).segment<3>(9);
            dv[i][5] = M_kernel.col(j).segment<3>(6) - M_kernel.col(j).segment<3>(9);
        }

        Vec3 dc[6];
        dc[0] = c_w[0] - c_w[1];
        dc[1] = c_w[0] - c_w[2];
        dc[2] = c_w[0] - c_w[3];
        dc[3] = c_w[1] - c_w[2];
        dc[4] = c_w[1] - c_w[3];
        dc[5] = c_w[2] - c_w[3];

        // beta求解器
        auto beta_solver = [](const Eigen::Matrix<double, 6, 10> &L, const Eigen::Matrix<double, 6, 1> &b, Vec4 &beta) {
            constexpr static unsigned int num_iters = 5;
            Eigen::Matrix<double, 6, 4> J;
            Eigen::Matrix<double, 6, 1> e;
            for (unsigned int n = 0; n < num_iters; ++n) {
                for (unsigned int i = 0; i < 6; ++i) {
                    J.row(i) << 2. * beta[0] * L(i, 0) + beta[1] * L(i, 1) + beta[2] * L(i, 2) + beta[3] * L(i, 3),
                            beta[0] * L(i, 1) + 2. * beta[1] * L(i, 4) + beta[2] * L(i, 5) + beta[3] * L(i, 6),
                            beta[0] * L(i, 2) + beta[1] * L(i, 5) + 2. * beta[2] * L(i, 7) + beta[3] * L(i, 8),
                            beta[0] * L(i, 3) + beta[1] * L(i, 6) + beta[3] * L(i, 8) + 2. * beta[3] * L(i, 9);
                    e(i) = b(i) - (beta[0] * beta[0] * L(i, 0) + beta[0] * beta[1] * L(i, 1) + beta[0] * beta[2] * L(i, 2) + beta[0] * beta[3] * L(i, 3) +
                                   beta[1] * beta[1] * L(i, 4) + beta[1] * beta[2] * L(i, 5) + beta[1] * beta[3] * L(i, 6) +
                                   beta[2] * beta[2] * L(i, 7) + beta[2] * beta[3] * L(i, 8) +
                                   beta[3] * beta[3] * L(i, 9));
                }
                beta += J.fullPivHouseholderQr().solve(e);
            }
        };

        // c_c求解器
        auto c_c_solver = [](const Eigen::Matrix<double, 12, 12> &V, const Vec4 &beta, Vec3 c_c[4]) {
            for (unsigned int i = 0; i < 4; ++i) {
                c_c[i] = beta[0] * V.col(11).segment<3>(3 * i) + beta[1] * V.col(10).segment<3>(3 * i) + beta[2] * V.col(9).segment<3>(3 * i) + beta[3] * V.col(8).segment<3>(3 * i);
            }

            // 控制点的深度应该为正
            if (c_c[0].z() < 0.) {
                for (unsigned int i = 0; i < 4; ++i) {
                    c_c[i] = -c_c[i];
                }
            }
        };

        // p_c求解器
        auto p_c_solver = [](const Vec3 c_c[4], const vector<Vec4> &alpha, vector<Vec3> &p_c) {
            for (unsigned long k = 0; k < alpha.size(); ++k) {
                p_c[k] = c_c[0] * alpha[k][0] + c_c[1] * alpha[k][1] + c_c[2] * alpha[k][2] + c_c[3] * alpha[k][3];
            }
        };

        // pose求解器
        auto pose_solver = [](const vector<Vec3> &p_w, const vector<Vec3> &p_c, Mat33 &R, Vec3 &t) {
            Vec3 mean_w, mean_c;
            mean_w.setZero();
            mean_c.setZero();
            for (unsigned long k = 0; k < p_w.size(); ++k) {
                mean_w += p_w[k];
                mean_c += p_c[k];
            }
            mean_w /= double(p_w.size());
            mean_c /= double(p_w.size());

            Mat33 H;
            H.setZero();
            for (unsigned long k = 0; k < p_w.size(); ++k) {
                H += (p_w[k] - mean_w) * (p_c[k] - mean_c).transpose();
            }
            auto &&H_svd = H.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

            R = H_svd.matrixU() * H_svd.matrixV().transpose();
            if (R.determinant() < 0.) {
                R = -R;
            }

            t = mean_w - R * mean_c;
        };

        // error求解器
        auto error_solver = [](const vector<Vec3> &p_w, const vector<Vec2> &uv, const Mat33 &R, const Vec3 &t) -> double {
            double sum = 0.;
            for (unsigned long k = 0; k < p_w.size(); ++k) {
                Vec3 p_c = R.transpose() * (p_w[k] - t);
                double depth_inv = 1. / p_c.z();
                double e_u = uv[k].x() - p_c.x() * depth_inv;
                double e_v = uv[k].y() - p_c.y() * depth_inv;
                sum += e_u * e_u + e_v * e_v;
            }
            return sum / double(p_w.size());
        };

        // camera到imu的转换
        auto from_camera_to_imu = [this](Mat33 &R, Vec3 &t) {
            R = R * _q_ic[0].inverse();
            t -= R * _t_ic[0];
        };

        // 4个beta的情况
        Vec4 beta[4];
        Mat33 R[4];
        Vec3 t[4];
        double chi2[4];
        Vec3 c_c[4];
        vector<Vec3> p_c;
        p_c.resize(num_points);

        Eigen::Matrix<double, 6, 10> L;
        Vec6 b;
        for (unsigned int i = 0; i < 6; ++i) {
            L.row(i) << dv[0][i].squaredNorm(), 2. * dv[0][i].dot(dv[1][i]), 2. * dv[0][i].dot(dv[2][i]), 2. * dv[0][i].dot(dv[3][i]),
                    dv[1][i].squaredNorm(), 2. * dv[1][i].dot(dv[2][i]), 2. * dv[1][i].dot(dv[3][i]),
                    dv[2][i].squaredNorm(), 2. * dv[2][i].dot(dv[3][i]),
                    dv[3][i].squaredNorm();

            b(i) = dc[i].squaredNorm();
        }

        // N = 1
        beta[0].setZero();
        double num = 0.;
        double den = 0.;
        for (unsigned int i = 0; i < 6; ++i) {
            num += dv[0][i].dot(dc[i]);
            den += dv[0][i].squaredNorm();
        }
        beta[0][0] = num / den;
        beta_solver(L, b, beta[0]);
        c_c_solver(M_kernel, beta[0], c_c);
        p_c_solver(c_c, alpha, p_c);
        pose_solver(p_w, p_c, R[0], t[0]);
        chi2[0] = error_solver(p_w, uv, R[0], t[0]);
        from_camera_to_imu(R[0], t[0]);

        Qd q(R[0]);
        Vec7 pose;
        pose << t[0].x(), t[0].y(), t[0].z(),
                q.x(), q.y(), q.z(), q.w();
        imu_i->vertex_pose->set_parameters(pose);

        std::cout << "epnp: takse " << pnp_t.toc() << " ms" << std::endl;

        std::cout << "epnp: chi2 = " << chi2[0] << std::endl;
        std::cout << "epnp: beta = " << beta[0].transpose() << std::endl;
        std::cout << "epnp: q = " << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << std::endl;
        std::cout << "epnp: t = " << t[0].transpose() << std::endl;

//        // N = 2
//        Eigen::Matrix<double, 6, 3> A2;
//        A2.col(0) = L.col(0);
//        A2.col(1) = L.col(1);
//        A2.col(2) = L.col(4);
//        Mat33 ATA2 = A2.transpose() * A2;
//        Vec3 ATb2 = A2.transpose() * b;
//        Vec3 ans2 = ATA2.fullPivLu().solve(ATb2);
//        if (ans2[0] > 0.) {
//            beta[1][0] = sqrt(ans2[0]);
//        }
//        if (ans2[2] > 0.) {
//            beta[1][1] = sqrt(ans2[2]);
//        }
//        if (ans2[1] < 0.) {
//            beta[1][0] = -beta[1][0];
//        }
//        beta_solver(L, b, beta[1]);
//        c_c_solver(M_kernel, beta[1], c_c);
//        p_c_solver(c_c, alpha, p_c);
//        pose_solver(p_w, p_c, R[1], t[1]);
//        chi2[1] = error_solver(p_w, uv, R[1], t[1]);
//
//        // N = 3
//        Eigen::Matrix<double, 6, 5> A3;
//        A3.col(0) = L.col(0);
//        A3.col(1) = L.col(1);
//        A3.col(2) = L.col(2);
//        A3.col(3) = L.col(4);
//        A3.col(4) = L.col(5);
//        Eigen::Matrix<double, 5, 5> ATA3 = A3.transpose() * A3;
//        Eigen::Matrix<double, 5, 1> ATb3 = A3.transpose() * b;
//        Eigen::Matrix<double, 5, 1> ans3 = ATA3.fullPivLu().solve(ATb3);
//        if (ans3[0] > 0.) {
//            beta[2][0] = sqrt(ans3[0]);
//            beta[2][2] = ans3[2] / beta[2][0];
//        }
//        if (ans3[3] > 0.) {
//            beta[2][1] = sqrt(ans3[3]);
//        }
//        if (ans3[1] < 0.) {
//            beta[2][0] = -beta[2][0];
//        }
//        beta_solver(L, b, beta[2]);
//        c_c_solver(M_kernel, beta[2], c_c);
//        p_c_solver(c_c, alpha, p_c);
//        pose_solver(p_w, p_c, R[2], t[2]);
//        chi2[2] = error_solver(p_w, uv, R[2], t[2]);
//
//        // N = 4
//        Eigen::Matrix<double, 6, 4> A4;
//        A4.col(0) = L.col(0);
//        A4.col(1) = L.col(1);
//        A4.col(2) = L.col(2);
//        A4.col(3) = L.col(3);
//        Eigen::Matrix<double, 4, 4> ATA4 = A4.transpose() * A4;
//        Eigen::Matrix<double, 4, 1> ATb4 = A4.transpose() * b;
//        Eigen::Matrix<double, 4, 1> ans4 = ATA4.fullPivLu().solve(ATb4);
//        if (ans4[0] > 0.) {
//            beta[3][0] = sqrt(ans4[0]);
//            beta[3][1] = ans4[1] / beta[3][0];
//            beta[3][2] = ans4[2] / beta[3][0];
//            beta[3][3] = ans4[3] / beta[3][0];
//        }
//        beta_solver(L, b, beta[3]);
//        c_c_solver(M_kernel, beta[3], c_c);
//        p_c_solver(c_c, alpha, p_c);
//        pose_solver(p_w, p_c, R[3], t[3]);
//        chi2[3] = error_solver(p_w, uv, R[3], t[3]);
//
//        unsigned int best_index = 0;
//        for (unsigned int i = 1; i < 4; ++i) {
//            if (chi2[i] < chi2[best_index]) {
//                best_index = i;
//            }
//        }
    }
}