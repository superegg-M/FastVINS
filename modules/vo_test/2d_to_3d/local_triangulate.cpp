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

    void Estimator::local_triangulate_with(ImuNode *imu_i, ImuNode *imu_j, bool enforce) {
        if (imu_i == imu_j) {
            return;
        }

        // imu的位姿
        auto &&i_pose = imu_i->vertex_pose;
        Vec3 p_i {i_pose->get_parameters()(0), i_pose->get_parameters()(1), i_pose->get_parameters()(2)};
        Qd q_i {i_pose->get_parameters()(6), i_pose->get_parameters()(3), i_pose->get_parameters()(4), i_pose->get_parameters()(5)};
        Mat33 r_i {q_i.toRotationMatrix()};

        // imu的位姿
        auto &&j_pose = imu_j->vertex_pose;
        Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
        Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
        Mat33 r_j {q_j.toRotationMatrix()};

        for (auto &feature_in_cameras : imu_i->features_in_cameras) {
            auto &&feature_it = _feature_map.find(feature_in_cameras.first);
            if (feature_it == _feature_map.end()) {
                std::cout << "Error: feature not in feature_map when running local_triangulate_with" << std::endl;
                continue;
            }
            auto &&feature_in_cameras_j = imu_j->features_in_cameras.find(feature_in_cameras.first);
            if (feature_in_cameras_j == imu_j->features_in_cameras.end()) {
                continue;
            }

            if (!feature_it->second->vertex_landmark) {
                shared_ptr<VertexInverseDepth> vertex_inverse_depth(new VertexInverseDepth);
                feature_it->second->vertex_landmark = vertex_inverse_depth;
            } else if (!enforce) {
                continue;
            }

            Eigen::MatrixXd svd_A(4, 4);
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Vector3d f;

            // imu_i的信息
            auto &&i_cameras = feature_in_cameras.second;    // imu中，与feature对应的相机信息
            auto &&i_camera_id = i_cameras[0].first;  // 左目的id
            auto &&i_pixel_coord = i_cameras[0].second;    // feature在imu的左目的像素坐标

            Eigen::Vector3d t_wci_w = p_i + r_i * _t_ic[i_camera_id];
            Eigen::Matrix3d r_wci = r_i * _q_ic[i_camera_id];

            P.leftCols<3>().setIdentity();
            P.rightCols<1>().setZero();

            f = i_pixel_coord / i_pixel_coord.z();
            svd_A.row(0) = f[0] * P.row(2) - P.row(0);
            svd_A.row(1) = f[1] * P.row(2) - P.row(1);

            // imu_j的信息
            auto &&j_cameras = feature_in_cameras_j->second;    // imu中，与feature对应的相机信息
            auto &&j_camera_id = j_cameras[0].first;  // 左目的id
            auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

            Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * _q_ic[j_camera_id];

            P.leftCols<3>() = r_wcj.transpose() * r_wci;
            P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w);

            f = j_pixel_coord / j_pixel_coord.z();
            svd_A.row(2) = f[0] * P.row(2) - P.row(0);
            svd_A.row(3) = f[1] * P.row(2) - P.row(1);

            // 最小二乘计算深度
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            Vec1 inverse_depth {svd_V[3] / svd_V[2]};
            feature_it->second->vertex_landmark->set_parameters(inverse_depth);

            // 检查深度
            if (inverse_depth[0] < 0.) {
                feature_it->second->vertex_landmark = nullptr;
                continue;
            }
            Vec3 p_cj = r_wcj.transpose() * (r_wci * i_pixel_coord / inverse_depth[0] + t_wci_w - t_wcj_w);
            if (p_cj.z() < 0.) {
                feature_it->second->vertex_landmark = nullptr;
            }
        }
    }

    void Estimator::local_triangulate_feature(FeatureNode* feature, bool enforce) {
        if (!feature) {
            return;
        }

        bool is_in_current = _imu_node->features_in_cameras.find(feature->id()) != _imu_node->features_in_cameras.end();

        // 若imu数小于2，则无法进行三角化
        auto &&imu_deque = feature->imu_deque;
        unsigned long num_imu = imu_deque.size() + is_in_current ? 1 : 0;
        if (num_imu < 2) {
            return;
        }

        if (!feature->vertex_landmark) {
            shared_ptr<VertexInverseDepth> vertex_inverse_depth(new VertexInverseDepth);
            feature->vertex_landmark = vertex_inverse_depth;
        } else if (!enforce) {
            return;
        }

        Eigen::MatrixXd svd_A(2 * num_imu, 4);
        Eigen::Matrix<double, 3, 4> P;
        Eigen::Vector3d f;

        // imu_i的信息
        auto &&imu_i = imu_deque.oldest();
        auto &&i_pose = imu_i->vertex_pose;   // imu的位姿
        auto &&i_feature_in_cameras = imu_i->features_in_cameras.find(feature->id());
        if (i_feature_in_cameras == imu_i->features_in_cameras.end()) {
            std::cout << "Error: feature not in features_in_cameras when running local_triangulate_feature" << std::endl;
            return;
        }
        auto &&i_cameras = i_feature_in_cameras->second;    // imu中，与feature对应的相机信息
        auto &&i_camera_id = i_cameras[0].first;  // 左目的id
        auto &&i_pixel_coord = i_cameras[0].second;    // feature在imu的左目的像素坐标

        Vec3 p_i {i_pose->get_parameters()(0), i_pose->get_parameters()(1), i_pose->get_parameters()(2)};
        Qd q_i {i_pose->get_parameters()(6), i_pose->get_parameters()(3), i_pose->get_parameters()(4), i_pose->get_parameters()(5)};
        Mat33 r_i {q_i.toRotationMatrix()};

        Eigen::Vector3d t_wci_w = p_i + r_i * _t_ic[i_camera_id];
        Eigen::Matrix3d r_wci = r_i * _q_ic[i_camera_id];

        P.leftCols<3>().setIdentity();
        P.rightCols<1>().setZero();

        f = i_pixel_coord / i_pixel_coord.z();
        svd_A.row(0) = f[0] * P.row(2) - P.row(0);
        svd_A.row(1) = f[1] * P.row(2) - P.row(1);

        for (unsigned long j = 1; j < imu_deque.size(); ++j) {
            // imu_j的信息
            auto &&imu_j = imu_deque[j];
            auto &&j_pose = imu_j->vertex_pose;   // imu的位姿
            auto &&j_feature_in_cameras = imu_j->features_in_cameras.find(feature->id());
            if (j_feature_in_cameras == imu_j->features_in_cameras.end()) {
                std::cout << "Error: feature not in features_in_cameras when running global_triangulate_feature" << std::endl;
                continue;
            }
            auto &&j_cameras = j_feature_in_cameras->second;    // imu中，与feature对应的相机信息
            auto &&j_camera_id = j_cameras[0].first;  // 左目的id
            auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

            Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
            Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
            Mat33 r_j {q_j.toRotationMatrix()};

            Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * _q_ic[j_camera_id];

            P.leftCols<3>() = r_wcj.transpose() * r_wci;
            P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w);

            f = j_pixel_coord / j_pixel_coord.z();
            svd_A.row(2 * j) = f[0] * P.row(2) - P.row(0);
            svd_A.row(2 * j + 1) = f[1] * P.row(2) - P.row(1);
        }

        if (is_in_current) {
            unsigned long j = imu_deque.size();

            // imu_j的信息
            auto &&j_pose = _imu_node->vertex_pose;   // imu的位姿
            auto &&j_feature_in_cameras = _imu_node->features_in_cameras.find(feature->id());
            if (j_feature_in_cameras == _imu_node->features_in_cameras.end()) {
                std::cout << "Error: feature not in features_in_cameras when running global_triangulate_feature" << std::endl;
            } else {
                auto &&j_cameras = j_feature_in_cameras->second;    // imu中，与feature对应的相机信息
                auto &&j_camera_id = j_cameras[0].first;  // 左目的id
                auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

                Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
                Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
                Mat33 r_j {q_j.toRotationMatrix()};

                Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[j_camera_id];
                Eigen::Matrix3d r_wcj = r_j * _q_ic[j_camera_id];

                P.leftCols<3>() = r_wcj.transpose() * r_wci;
                P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w);

                f = j_pixel_coord / j_pixel_coord.z();
                svd_A.row(2 * j) = f[0] * P.row(2) - P.row(0);
                svd_A.row(2 * j + 1) = f[1] * P.row(2) - P.row(1);
            }
        }

        // 最小二乘计算深度
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        Vec1 inverse_depth {svd_V[3] / svd_V[2]};
        feature->vertex_landmark->set_parameters(inverse_depth);
    }
}