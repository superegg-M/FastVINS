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

    void Estimator::global_triangulate_with(ImuNode *imu_i, ImuNode *imu_j, bool enforce) {
        TicToc tri_t;

        if (imu_i == imu_j) {
            return;
        }

        // imu i的位姿
        auto &&i_pose = imu_i->vertex_pose;
        Vec3 p_i {i_pose->get_parameters()(0), i_pose->get_parameters()(1), i_pose->get_parameters()(2)};
        Qd q_i {i_pose->get_parameters()(6), i_pose->get_parameters()(3), i_pose->get_parameters()(4), i_pose->get_parameters()(5)};
        Mat33 r_i {q_i.toRotationMatrix()};

        // imu j的位姿
        auto &&j_pose = imu_j->vertex_pose;
        Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
        Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
        Mat33 r_j {q_j.toRotationMatrix()};

        for (auto &feature_in_cameras : imu_i->features_in_cameras) {
            auto &&feature_it = _feature_map.find(feature_in_cameras.first);
            if (feature_it == _feature_map.end()) {
                std::cout << "Error: feature not in feature_map when running global_triangulate_with" << std::endl;
                continue;
            }
            auto &&feature_in_cameras_j = imu_j->features_in_cameras.find(feature_in_cameras.first);
            if (feature_in_cameras_j == imu_j->features_in_cameras.end()) {
                continue;
            }

            if (!feature_it->second->vertex_point3d) {
                shared_ptr<VertexPoint3d> vertex_point3d(new VertexPoint3d);
                feature_it->second->vertex_point3d = vertex_point3d;
            } else if (!enforce) {
                continue;
            }

            Mat43 A;
            Vec4 b;

//            Eigen::MatrixXd svd_A(4, 4);
//            Eigen::Matrix<double, 3, 4> P;
//            Eigen::Vector3d f;

            // imu_i的信息
            auto &&i_cameras = feature_in_cameras.second;    // imu中，与feature对应的相机信息
            auto &&i_camera_id = i_cameras[0].first;  // 左目的id
            auto &&i_pixel_coord = i_cameras[0].second;    // feature在imu的左目的像素坐标

            Eigen::Vector3d t_wci_w = p_i + r_i * _t_ic[i_camera_id];
            Eigen::Matrix3d r_wci = r_i * _q_ic[i_camera_id];

            Vec3 t1 = r_wci.transpose() * t_wci_w;
            A.row(0) = (i_pixel_coord.x() * r_wci.col(2) - r_wci.col(0)).transpose();
            A.row(1) = (i_pixel_coord.y() * r_wci.col(2) - r_wci.col(1)).transpose();
            b(0) = i_pixel_coord.x() * t1.z() - t1.x();
            b(1) = i_pixel_coord.y() * t1.z() - t1.y();

//            P.leftCols<3>() = r_wci.transpose();
//            P.rightCols<1>() = -r_wci.transpose() * t_wci_w;
//
//            f = i_pixel_coord / i_pixel_coord.z();
//            svd_A.row(0) = f[0] * P.row(2) - P.row(0);
//            svd_A.row(1) = f[1] * P.row(2) - P.row(1);

            // imu_j的信息
            auto &&j_cameras = feature_in_cameras_j->second;    // imu中，与feature对应的相机信息
            auto &&j_camera_id = j_cameras[0].first;  // 左目的id
            auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

            Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * _q_ic[j_camera_id];

            Vec3 t2 = r_wcj.transpose() * t_wcj_w;
            A.row(2) = (j_pixel_coord.x() * r_wcj.col(2) - r_wcj.col(0)).transpose();
            A.row(3) = (j_pixel_coord.y() * r_wcj.col(2) - r_wcj.col(1)).transpose();
            b(2) = j_pixel_coord.x() * t2.z() - t2.x();
            b(3) = j_pixel_coord.y() * t2.z() - t2.y();

//            P.leftCols<3>() = r_wcj.transpose();
//            P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;
//
//            f = j_pixel_coord / j_pixel_coord.z();
//            svd_A.row(2) = f[0] * P.row(2) - P.row(0);
//            svd_A.row(3) = f[1] * P.row(2) - P.row(1);

            // 最小二乘计算世界坐标
//            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
//            Vec3 point {svd_V[0] / svd_V[3], svd_V[1] / svd_V[3], svd_V[2] / svd_V[3]};
//            feature_it->second->vertex_point3d->set_parameters(point);

            Mat33 ATA = A.transpose() * A;
            Vec3 ATb = A.transpose() * b;
            auto &&ATA_ldlt = ATA.ldlt();
            Vec3 point = ATA_ldlt.solve(ATb);
            feature_it->second->vertex_point3d->set_parameters(point);

            // 检查深度
            Vec3 p_ci = r_wci.transpose() * (point - t_wci_w);
            if (p_ci.z() < 0.) {
                feature_it->second->vertex_point3d = nullptr;
                continue;
            }
            Vec3 p_cj = r_wcj.transpose() * (point - t_wcj_w);
            if (p_cj.z() < 0.) {
                feature_it->second->vertex_point3d = nullptr;
            }
        }

        std::cout << "global 2d_to_3d takes " << tri_t.toc() << " ms" << std::endl;
    }



    void Estimator::global_triangulate_feature(FeatureNode* feature, bool enforce) {
        TicToc tri_t;

        if (!feature) {
            return;
        }

        bool is_in_current = _imu_node->features_in_cameras.find(feature->id()) != _imu_node->features_in_cameras.end();

        // 若imu数小于2，则无法进行三角化
        auto &&imu_deque = feature->imu_deque;
        unsigned long num_imu = imu_deque.size() + (is_in_current ? 1 : 0);
        if (num_imu < 2) {
            return;
        }

        if (!feature->vertex_point3d) {
            shared_ptr<VertexPoint3d> vertex_point3d(new VertexPoint3d);
            feature->vertex_point3d = vertex_point3d;
        } else if (!enforce) {
            return;
        }

        Eigen::MatrixXd svd_A(2 * num_imu, 4);
        Eigen::Matrix<double, 3, 4> P;
        Eigen::Vector3d f;
        for (unsigned long j = 0; j < imu_deque.size(); ++j) {
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

            P.leftCols<3>() = r_wcj.transpose();
            P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;

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

                P.leftCols<3>() = r_wcj.transpose();
                P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;

                f = j_pixel_coord / j_pixel_coord.z();
                svd_A.row(2 * j) = f[0] * P.row(2) - P.row(0);
                svd_A.row(2 * j + 1) = f[1] * P.row(2) - P.row(1);
            }
        }

        // 最小二乘计算世界坐标
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        Vec3 point {svd_V[0] / svd_V[3], svd_V[1] / svd_V[3], svd_V[2] / svd_V[3]};
        feature->vertex_point3d->set_parameters(point);

        std::cout << "global 2d_to_3d takes " << tri_t.toc() << " ms" << std::endl;
    }
}