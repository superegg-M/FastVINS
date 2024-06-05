//
// Created by Cain on 2024/5/23.
//

#include "../estimator.h"

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

    void Estimator::local_bundle_adjustment(vector<shared_ptr<VertexPose>> *fixed_poses) {
        std::unordered_map<unsigned long, unsigned long> from_vertex_to_feature(_feature_map.size());

        // VO优化
        ProblemSLAM problem;

        // fix住imu
        if (fixed_poses) {
            for (auto &pose : *fixed_poses) {
                pose->set_fixed();
            }
        }

//        // 把外参加入到problem中
//        problem.add_vertex(_vertex_ext[0]);

        // 把windows中的imu加入到problem中
        for (unsigned long i = 0; i < _windows.size(); ++i) {
            problem.add_vertex(_windows[i]->vertex_pose);
        }

        // 把当前的imu加入到problem中
        problem.add_vertex(_imu_node->vertex_pose);

        // 遍历所有特征点
        for (auto &feature_it : _feature_map) {
            unsigned long feature_id = feature_it.first;
            auto feature_node = feature_it.second;

            if (feature_node->is_outlier) {
                continue;
            }

            // 只有进行了初始化的特征点才参与计算
            if (feature_node->vertex_landmark) {
                from_vertex_to_feature.emplace(feature_node->vertex_landmark->id(), feature_id);

                // 构建重投影edge
                auto &&imu_deque = feature_node->imu_deque;
                auto &&curr_feature_in_cameras = _imu_node->features_in_cameras.find(feature_id);
                bool is_feature_in_curr_imu = curr_feature_in_cameras != _imu_node->features_in_cameras.end();

                if (imu_deque.size() > 1 || (imu_deque.size() == 1 && is_feature_in_curr_imu)) {
                    // 把特征点加入到problem中
                    problem.add_vertex(feature_node->vertex_landmark);

                    // host imu
                    auto &&host_imu = imu_deque.oldest();
                    auto &&host_feature_in_cameras = host_imu->features_in_cameras.find(feature_id);
                    if (host_feature_in_cameras == host_imu->features_in_cameras.end()) {
                        continue;
                    }
                    auto &&host_cameras = host_feature_in_cameras->second;
                    auto &&host_camera_id = host_cameras[0].first;
                    auto &&host_point_pixel = host_cameras[0].second;
                    host_point_pixel /= host_point_pixel.z();

                    // 其他imu
                    for (unsigned long j = 1; j < imu_deque.size(); ++j) {
                        auto &j_imu = imu_deque[j];
                        auto &&j_feature_in_cameras = j_imu->features_in_cameras.find(feature_id);
                        if (j_feature_in_cameras == j_imu->features_in_cameras.end()) {
                            continue;
                        }
                        auto &&j_cameras = j_feature_in_cameras->second;
                        auto &&j_camera_id = j_cameras[0].first;
                        auto &&j_point_pixel = j_cameras[0].second;
                        j_point_pixel /= j_point_pixel.z();

//                        auto edge_reproj = std::make_shared<EdgeReprojection>(host_point_pixel, j_point_pixel);
                        auto edge_reproj = std::make_shared<EdgeReprojectionLocal>(host_point_pixel, j_point_pixel);
                        edge_reproj->add_vertex(feature_node->vertex_landmark);
                        edge_reproj->add_vertex(host_imu->vertex_pose);
                        edge_reproj->add_vertex(j_imu->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[0]);

                        // 把edge加入到problem中
                        problem.add_edge(edge_reproj);
                    }

                    // 当前imu
                    if (is_feature_in_curr_imu) {
                        auto &&j_cameras = curr_feature_in_cameras->second;
                        auto &&j_camera_id = j_cameras[0].first;
                        auto &&j_point_pixel = j_cameras[0].second;
                        j_point_pixel /= j_point_pixel.z();

//                        auto edge_reproj = std::make_shared<EdgeReprojection>(host_point_pixel, j_point_pixel);
                        auto edge_reproj = std::make_shared<EdgeReprojectionLocal>(host_point_pixel, j_point_pixel);
                        edge_reproj->add_vertex(feature_node->vertex_landmark);
                        edge_reproj->add_vertex(host_imu->vertex_pose);
                        edge_reproj->add_vertex(_imu_node->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[0]);

                        // 把edge加入到problem中
                        problem.add_edge(edge_reproj);
                    }
                }
            }
        }

        // 优化
        problem.solve(5);

        // 检查outlier
        for (const auto &edge : problem.edges()) {
            if (edge.second->get_chi2() > 3.) {
                _feature_map[from_vertex_to_feature[edge.second->vertices()[0]->id()]]->is_outlier = true;
            }
        }

        // 解锁imu
        if (fixed_poses) {
            for (auto &pose : *fixed_poses) {
                pose->set_fixed(false);
            }
        }
    }
}