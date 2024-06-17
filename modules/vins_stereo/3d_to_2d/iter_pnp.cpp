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

    bool Estimator::iter_pnp(ImuNode *imu_i, Qd *q_wi_init, Vec3 *t_wi_init) {
        constexpr static double th_e2 = 3.841;
        constexpr static unsigned int num_iters = 5;
        constexpr static unsigned int num_fix = 5;

        TicToc pnp_t;

        // 读取3d, 2d点
        vector<Vec3> p_w;
        vector<unsigned long> p_w_id;
        vector<Vec2> uv;
        p_w.reserve(imu_i->features_in_cameras.size());
        p_w_id.reserve(imu_i->features_in_cameras.size());
        uv.reserve(imu_i->features_in_cameras.size());

        // 遍历imu中的所有features, 计算其3d坐标以及记录其在imu的cameras中的2d坐标
        for (auto &feature_in_cameras_it : imu_i->features_in_cameras) {
            auto &&feature_it = _feature_map.find(feature_in_cameras_it.first);
            if (feature_it == _feature_map.end()) {
                std::cout << "Error: feature not in feature_map when running pnp" << std::endl;
                continue;
            }

            auto &&feature_node = feature_it->second;
            if (feature_node->is_outlier || !feature_node->is_triangulated) {
                continue;
            }

            double inv_depth = feature_node->vertex_landmark->get_parameters()[0];
            if (inv_depth < 0.) {
                continue;
            }

            // 若feature是在_imu_node中才被首次观测到, 则feature_node->imu_deque为空,
            // 因为feature此时还没被加入到feature_node->imu_deque中.
            auto &&imu_deque = feature_node->imu_deque;
            if (imu_deque.empty()) {
                continue;
            }

            auto &&host_imu = imu_deque.oldest();
            if (host_imu == imu_i) {
                continue;
            }

            auto &&host_pose = host_imu->vertex_pose->get_parameters();
            Vec3 t {host_pose[0], host_pose[1], host_pose[2]};
            Qd q {host_pose[6], host_pose[3], host_pose[4], host_pose[5]};

            // 计算feature的global 3d坐标以及记录其在imu_i的左目中的2d坐标
            Vec3 p_c = host_imu->features_in_cameras[feature_it->first][0].second / inv_depth;
            Vec3 p_i = _q_ic[0] * p_c + _t_ic[0];
            p_w.emplace_back(q * p_i + t);
            p_w_id.emplace_back(feature_it->first);
            uv.emplace_back(feature_in_cameras_it.second[0].second.x(), feature_in_cameras_it.second[0].second.y());
        }

        // 初始化
        Qd q_wi;
        Vec3 t_wi;
        if (q_wi_init) {
            q_wi = *q_wi_init;
        } else {
            q_wi.setIdentity();
        }
        if (t_wi_init) {
            t_wi = *t_wi_init;
        } else {
            t_wi.setZero();
        }

        // TODO: 应该使用RANSAC
        for (unsigned int n = 0; n < num_iters; ++n) {
            Mat66 H;
            Vec6 b;
            H.setZero();
            b.setZero();
            for (unsigned long k = 0; k < p_w.size(); ++k) {
                Vec3 p_imu_i = q_wi.inverse() * (p_w[k] - t_wi);
                Vec3 p_camera_i = _q_ic[0].inverse() * (p_imu_i - _t_ic[0]);
                double inv_depth_i = 1. / p_camera_i.z();

                // 重投影误差
                Vec2 e = (p_camera_i * inv_depth_i).head<2>() - uv[k];

                // 误差对投影点的偏导
                Mat23 dr_dpci;
                dr_dpci << inv_depth_i, 0., -p_camera_i[0] * inv_depth_i * inv_depth_i,
                        0., inv_depth_i, -p_camera_i[1] * inv_depth_i * inv_depth_i;

                // 投影点对imu位姿的偏导
                Eigen::Matrix<double, 3, 6> dpci_dpose_i;
                Mat33 R_ic = _q_ic[0].toRotationMatrix();
                dpci_dpose_i.leftCols<3>() = -R_ic.transpose() * q_wi.inverse();
                dpci_dpose_i.rightCols<3>() = R_ic.transpose() * Sophus::SO3d::hat(p_imu_i);

                // Jacobian
                Eigen::Matrix<double, 2, 6> jacobian_pose_i;
                jacobian_pose_i = dr_dpci * dpci_dpose_i;

                H += jacobian_pose_i.transpose() * jacobian_pose_i;
                b -= jacobian_pose_i.transpose() * e;
            }

            auto H_ldlt = H.ldlt();

            // 修复GN无解的情况
            if (H_ldlt.info() != Eigen::Success) {
                Vec6 lambda;
                for (unsigned int i = 0; i < 6; ++i) {
                    lambda[i] = min(max(H(i, i) * 1e-5, 1e-6), 1e6);
                }

                double v = 2.;
                for (unsigned int m = 0; m < num_fix; ++m) {
                    for (unsigned int i = 0; i < 6; ++i) {
                        H(i, i) += v * lambda[i];
                    }
                    H_ldlt = H.ldlt();
                    if (H_ldlt.info() == Eigen::Success){
                        break;
                    }
                    v *= 2.;
                }
            }

            // 只有成功才能进行更新
            if (H_ldlt.info() == Eigen::Success) {
                Vec6 delta = H_ldlt.solve(b);
                t_wi += delta.head(3);
                q_wi *= Sophus::SO3d::exp(delta.tail(3)).unit_quaternion();
                q_wi.normalize();
            } else {
                return false;
            }
        }

        // 判断landmark是否outlier
        unsigned long outlier_count = 0;
        for (unsigned long k = 0; k < p_w.size(); ++k) {
            auto id = p_w_id[k];

            Vec3 p_imu_i = q_wi.inverse() * (p_w[k] - t_wi);
            Vec3 p_camera_i = _q_ic[0].inverse() * (p_imu_i - _t_ic[0]);
            double inv_depth_i = 1. / p_camera_i.z();
            // 逆深度
            if (inv_depth_i < 0.) {
                _feature_map[id]->is_outlier = true;
            }

            // 重投影误差
            Vec2 e = (p_camera_i * inv_depth_i).head<2>() - uv[k];
            double e2 = e.squaredNorm();
            if (e2 > th_e2) {
                _feature_map[id]->is_outlier = true;
                ++outlier_count;
            }
        }

        if (outlier_count * 100 > p_w.size() * 30) {
            return false;
        }

        Vec7 pose;
        pose << t_wi.x(), t_wi.y(), t_wi.z(),
                q_wi.x(), q_wi.y(), q_wi.z(), q_wi.w();
        imu_i->vertex_pose->set_parameters(pose);

        std::cout << "iter_pnp takes " << pnp_t.toc() << " ms" << std::endl;
        std::cout << "iter_pnp: q = " << q_wi.w() << ", " << q_wi.x() << ", " << q_wi.y() << ", " << q_wi.z() << std::endl;
        std::cout << "iter_pnp: t = " << t_wi.transpose() << std::endl;
        return true;
    }
}