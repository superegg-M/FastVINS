//
// Created by Cain on 2024/1/11.
//

#include "estimator.h"
#include "vertex/vertex_inverse_depth.h"
#include "vertex/vertex_pose.h"
#include "vertex/vertex_motion.h"
#include "edge/edge_reprojection.h"
#include "edge/edge_imu.h"

#include "tic_toc/tic_toc.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    Estimator::Estimator() : _vertex_ext(NUM_OF_CAM), _windows(WINDOW_SIZE) {
        _q_ic.resize(NUM_OF_CAM);
        _t_ic.resize(NUM_OF_CAM);
        _vertex_ext.resize(NUM_OF_CAM);
        _q_ic[0] = {cos(-0.5 * double(EIGEN_PI) * 0.5), 0., sin(-0.5 * double(EIGEN_PI) * 0.5), 0.};
        _t_ic[0] = {0., 0., 0.};
        Vec7 pose;
        pose << _t_ic[0].x(), _t_ic[0].y(), _t_ic[0].z(), _q_ic[0].x(), _q_ic[0].y(), _q_ic[0].z(), _q_ic[0].w();
        _vertex_ext[0] = std::make_shared<VertexPose>();
        _vertex_ext[0]->set_parameters(pose);

        solver_flag = INITIAL;
    }

    bool Estimator::initialize() {
        std::cout << "running initialize" << std::endl;
        TicToc t_sfm;

        if (structure_from_motion()) {
            std::cout << "done structure_from_motion" << std::endl;
            if (align_visual_to_imu()) {
                std::cout << "done align_visual_to_imu" << std::endl;
                return true;
            }
        }

        return false;
    }

    void Estimator::solve_odometry() {
        if (_windows.full() && solver_flag == NON_LINEAR) {
            TicToc t_tri;
//            backend_optimization();
        }
    }

    void Estimator::slide_window() {
        TicToc t_margin;

        std::cout << "_windows.size() = " << _windows.size() << std::endl;

        // 只有当windows满了才进行滑窗操作
        if (_windows.full()) {
            if (marginalization_flag == MARGIN_OLD) {
                std::cout << "MARGIN_OLD" << std::endl;
                ImuNode *imu_oldest {nullptr};
                _windows.pop_oldest(imu_oldest);    // 弹出windows中最老的imu

                // 遍历被删除的imu的所有特征点，在特征点的imu队列中，删除该imu
                for (auto &feature_in_cameras : imu_oldest->features_in_cameras) {
                    auto &&feature_id = feature_in_cameras.first;
                    auto &&feature_it = _feature_map.find(feature_id);
                    if (feature_it == _feature_map.end()) {
                        std::cout << "!!!!!!!! Can't find feature id in feature map when marg oldest !!!!!!!!!" << std::endl;
                        continue;
                    }
                    auto &&feature_node = feature_it->second;
                    auto &&vertex_landmark = feature_node->vertex_landmark;
                    auto &&imu_deque = feature_node->imu_deque;
                    // 应该是不需要进行判断的
//                    if (imu_deque.oldest() == imu_oldest) {
//                        imu_deque.pop_oldest();
//                    }
                    imu_deque.pop_oldest();

                    // TODO: 删除重投影edge与预积分edge

                    // 若特征点的keyframe小于2，则删除该特征点, 否则需要为特征点重新计算深度并且重新构建重投影edge,
                    if (imu_deque.size() < 2) {
                        // 在map中删除特征点
                        _feature_map.erase(feature_id);

                        // 释放特征点node的空间
                        delete feature_node;

                        if (vertex_landmark) {
                            // 在problem中删除特征点
                            _problem.remove_vertex(vertex_landmark);
                        }

                        // TODO: 在新的oldest imu中，把feature删除
                    } else {
                        if (vertex_landmark) {
                            // 曾经的host imu
                            auto &&oldest_cameras = feature_in_cameras.second;    // imu中，与feature对应的相机信息
                            auto &&oldest_imu_pose = imu_oldest->vertex_pose;   // imu的位姿
                            auto &&oldest_camera_id = oldest_cameras[0].first;  // camera的id
                            auto &&oldest_pixel_coord = oldest_cameras[0].second;    // feature在imu的左目的像素坐标

                            Vec3 p_i {oldest_imu_pose->get_parameters()(0), oldest_imu_pose->get_parameters()(1), oldest_imu_pose->get_parameters()(2)};
                            Qd q_i {oldest_imu_pose->get_parameters()(6), oldest_imu_pose->get_parameters()(3), oldest_imu_pose->get_parameters()(4), oldest_imu_pose->get_parameters()(5)};
                            Mat33 r_i {q_i.toRotationMatrix()};

                            Eigen::Vector3d t_wci_w = p_i + r_i * _t_ic[oldest_camera_id];
                            Eigen::Matrix3d r_wci = r_i * _q_ic[oldest_camera_id];

                            // 现在的host imu
                            auto &&host_imu = imu_deque.oldest();
                            auto &&host_cameras = host_imu->features_in_cameras[feature_id];
                            auto &&host_imu_pose = host_imu->vertex_pose;
                            auto &&host_camera_id = host_cameras[0].first;
                            auto &&host_pixel_coord = host_cameras[0].second;

                            Vec3 p_j {host_imu_pose->get_parameters()(0), host_imu_pose->get_parameters()(1), host_imu_pose->get_parameters()(2)};
                            Qd q_j {host_imu_pose->get_parameters()(6), host_imu_pose->get_parameters()(3), host_imu_pose->get_parameters()(4), host_imu_pose->get_parameters()(5)};
                            Mat33 r_j {q_j.toRotationMatrix()};

                            Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[host_camera_id];
                            Eigen::Matrix3d r_wcj = r_j * _q_ic[host_camera_id];

                            // 从i重投影到j
                            Vec3 p_cif_ci = oldest_pixel_coord / vertex_landmark->get_parameters()(0);
                            Vec3 p_wf_w = r_wci * p_cif_ci + t_wci_w;
                            Vec3 p_cjf_cj = r_wcj.transpose() * (p_wf_w - t_wcj_w);
                            double depth = p_cjf_cj.z();
                            if (depth < 0.1) {
                                depth = INIT_DEPTH;
                            }
                            vertex_landmark->set_parameters(Vec1(1. / depth));

                            // 计算重投影误差edge
                            // host imu的其他camera
                            for (unsigned long j = 1; j < host_cameras.size(); ++j) {
                                auto &&other_camera_id = host_cameras[j].first; // camera的id
                                auto &&other_pixel_coord = host_cameras[j].second;    // feature在imu的左目的像素坐标

                                shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                        host_pixel_coord,
                                        other_pixel_coord
                                ));
                                edge_reproj->add_vertex(vertex_landmark);
                                edge_reproj->add_vertex(host_imu_pose);
                                edge_reproj->add_vertex(host_imu_pose);
                                edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
                            }

                            // 其他imu
                            for (unsigned long i = 1; i < imu_deque.size(); ++i) {
                                auto &&other_imu = imu_deque[i];
                                auto &&other_cameras = other_imu->features_in_cameras[feature_id];
                                auto &&other_imu_pose = other_imu->vertex_pose;

                                // 遍历所有imu
                                for (unsigned j = 0; j < other_cameras.size(); ++j) {
                                    auto &&other_camera_id = other_cameras[j].first; // camera的id
                                    auto &&other_pixel_coord = other_cameras[j].second;    // feature在imu的左目的像素坐标

                                    shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                            host_pixel_coord,
                                            other_pixel_coord
                                    ));
                                    edge_reproj->add_vertex(vertex_landmark);
                                    edge_reproj->add_vertex(host_imu_pose);
                                    edge_reproj->add_vertex(other_imu_pose);
                                    edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
                                }
                            }
                        }
                    }
                }
                // 释放imu node的空间
                delete imu_oldest;
            } else {
                std::cout << "MARGIN_NEW" << std::endl;
                ImuNode *imu_newest {nullptr};
                _windows.pop_newest(imu_newest);    // 弹出windows中最新的imu

                // 遍历被删除的imu的所有特征点，在特征点的imu队列中，删除该imu
                for (auto &feature_in_cameras : imu_newest->features_in_cameras) {
                    auto &&feature_id = feature_in_cameras.first;
                    auto &&feature_it = _feature_map.find(feature_id);
                    if (feature_it == _feature_map.end()) {
                        std::cout << "!!!!!!!! Can't find feature id in feature map when marg newest !!!!!!!!!" << std::endl;
                        continue;
                    }
                    auto &&feature_node = feature_it->second;
                    auto &&vertex_landmark = feature_node->vertex_landmark;
                    auto &&imu_deque = feature_node->imu_deque;

                    // 移除feature的imu队列中的最新帧
                    imu_deque.pop_newest();

                    // TODO: 删除重投影edge与预积分edge

                    // 如果feature不在所有的imu中出现了，则需要删除feature
                    if (feature_node->imu_deque.empty()
                        && _imu_node->features_in_cameras.find(feature_id) == _imu_node->features_in_cameras.end()) {
                        // 在problem中删除特征点
                        _problem.remove_vertex(vertex_landmark);

                        // 在map中删除特征点
                        _feature_map.erase(feature_id);

                        // 释放特征点node的空间
                        delete feature_node;
                    }
                }
                // 释放imu node的空间
                delete imu_newest;
            }
        }

        // 把当前imu加入到windows中
        _windows.push_newest(_imu_node);

        // 遍历当前imu的所有特征点，把该imu加入到特征点的队列中
        for (auto &feature_in_cameras : _imu_node->features_in_cameras) {
            auto &&feature_id = feature_in_cameras.first;
            auto &&feature_it = _feature_map.find(feature_id);
            if (feature_it == _feature_map.end()) {
                std::cout << "!!!!!!! ERROR: feature of current imu not in feature map. !!!!!!!" << std::endl;
                continue;
            }
            auto &&feature_node = feature_it->second;
            feature_node->imu_deque.push_newest(_imu_node);
        }
    }
}