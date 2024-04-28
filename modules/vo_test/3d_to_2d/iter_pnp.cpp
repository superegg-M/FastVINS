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
        constexpr static unsigned int num_iters = 5;
        constexpr static unsigned int num_fix = 5;
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