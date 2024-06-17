//
// Created by Cain on 2024/4/2.
//

#include "data_manager.h"

#include <memory>

namespace vins {
    ImuNode::ImuNode(IMUIntegration *imu_integration_pt, unsigned int num_cameras)
    : features_in_cameras(2 * num_cameras), imu_integration(imu_integration_pt) {
    }

    ImuNode::~ImuNode() {
        delete imu_integration;
    }


//    FrameNode::FrameNode(unsigned long id, ImuNode *imu) : _camera_id(id), _imu_pt(imu) {
//
//    }


    FeatureNode::FeatureNode(unsigned long id) : imu_deque(WINDOW_SIZE), _feature_id(id) {

    }

    void FeatureNode::from_global_to_local(const std::vector<Qd> &q_ic, const vector<Vec3> &t_ic) {
        if (vertex_point3d) {
            if (!vertex_landmark) {
                vertex_landmark = std::make_shared<VertexInverseDepth>();
            }

            // 获取基准imu的信息
            auto &&imu_host = imu_deque.oldest();
            auto &&pose = imu_host->vertex_pose->get_parameters();
            auto &&feature_in_cameras = imu_host->features_in_cameras.find(id());
            if (feature_in_cameras == imu_host->features_in_cameras.end()) {
                std::cout << "feature not in features_in_cameras when running from_global_to_local" << std::endl;
                vertex_point3d = nullptr;
                vertex_landmark = nullptr;
                return;
            }
            auto &&cameras = feature_in_cameras->second;    // imu中，与feature对应的相机信息
            auto &&camera_id = cameras[0].first;  // 左目的id
            auto &&point_pixel = cameras[0].second;    // feature在imu的左目的像素坐标

            Vec3 p {pose(0), pose(1), pose(2)};
            Qd q {pose(6), pose(3), pose(4), pose(5)};
            Mat33 r {q.toRotationMatrix()};

            Eigen::Vector3d t_wc_w = p + r * t_ic[camera_id];
            Eigen::Matrix3d r_wc = r * q_ic[camera_id].toRotationMatrix();

            // 计算深度
            point_pixel /= point_pixel.z();
            Vec3 point_world = vertex_point3d->get_parameters();
            double depth = point_pixel.dot(r_wc.transpose() * (point_world - t_wc_w)) / point_pixel.squaredNorm();
            if (depth < 0.1) {
                depth = INIT_DEPTH;
            }
            vertex_landmark->set_parameters(Vec1(1. / depth));

        } else {
            vertex_landmark = nullptr;
        }
    }

    void FeatureNode::from_local_to_global(const std::vector<Qd> &q_ic, const vector<Vec3> &t_ic) {
        if (vertex_landmark) {
            if (!vertex_point3d) {
                vertex_point3d = std::make_shared<VertexPoint3d>();
            }

            // 获取基准imu的信息
            auto &&imu_host = imu_deque.oldest();
            auto &&pose = imu_host->vertex_pose->get_parameters();
            auto &&feature_in_cameras = imu_host->features_in_cameras.find(id());
            if (feature_in_cameras == imu_host->features_in_cameras.end()) {
                std::cout << "feature not in features_in_cameras when running from_local_to_global" << std::endl;
                vertex_point3d = nullptr;
                vertex_landmark = nullptr;
                return;
            }
            auto &&cameras = feature_in_cameras->second;    // imu中，与feature对应的相机信息
            auto &&camera_id = cameras[0].first;  // 左目的id
            auto &&point_pixel = cameras[0].second;    // feature在imu的左目的像素坐标

            Vec3 p {pose(0), pose(1), pose(2)};
            Qd q {pose(6), pose(3), pose(4), pose(5)};
            Mat33 r {q.toRotationMatrix()};

            Eigen::Vector3d t_wc_w = p + r * t_ic[camera_id];
            Eigen::Matrix3d r_wc = r * q_ic[camera_id].toRotationMatrix();

            // 计算世界坐标
            point_pixel /= point_pixel.z();
            double depth = 1. / vertex_landmark->get_parameters()(0);
            Vec3 point_world = r_wc * point_pixel * depth + t_wc_w;
            vertex_point3d->set_parameters(point_world);

        } else {
            vertex_point3d = nullptr;
        }
    }


    ImuWindows::ImuWindows(unsigned long n) : Deque<ImuNode *>(n) {

    }

    bool ImuWindows::is_feature_in_newest(unsigned long feature_id) const {
        if (empty()) {
            return false;
        }
        return true;
//            if (_imu_deque.back()) {
//                _imu_deque.back().
//            }
    }
    bool ImuWindows::is_feature_suitable_to_reproject(unsigned long feature_id) const {
        if (size() > 2) {
            auto feature_in_cameras = operator[](size() - 3)->features_in_cameras.find(feature_id);
            if (feature_in_cameras == operator[](size() - 3)->features_in_cameras.end()) {
                return false;
            } else {
                return true;
            }
        }
        return false;
    }
}