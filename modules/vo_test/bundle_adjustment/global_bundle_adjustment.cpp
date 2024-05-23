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

    void Estimator::global_bundle_adjustment(vector<shared_ptr<VertexPose>> *fixed_poses) {
        // VO优化
        ProblemSLAM problem;

        // fix住imu
        if (fixed_poses) {
            for (auto &pose : *fixed_poses) {
                pose->set_fixed();
            }
        }

        // 把外参加入到problem中
        problem.add_vertex(_vertex_ext[0]);

        // 把windows中的imu加入到problem中
        for (unsigned long i = 0; i < _windows.size(); ++i) {
            problem.add_vertex(_windows[i]->vertex_pose);
        }

        // 把当前的imu加入到problem中
        problem.add_vertex(_imu_node->vertex_pose);

        for (auto &feature_it : _feature_map) {
            auto &&feature_id = feature_it.first;
            auto &&feature_node = feature_it.second;

            // 只有进行了初始化的特征点才参与计算
            if (feature_node->vertex_point3d) {
                auto &&imu_deque = feature_node->imu_deque;
                auto &&curr_feature_in_cameras = _imu_node->features_in_cameras.find(feature_id);
                bool is_feature_in_curr_imu = curr_feature_in_cameras != _imu_node->features_in_cameras.end();

                if (imu_deque.size() > 1 || (imu_deque.size() == 1 && is_feature_in_curr_imu)) {
                    // 把特征点加入到problem中
                    problem.add_vertex(feature_node->vertex_point3d);

                    // deque中的imu
                    for (unsigned long j = 0; j < imu_deque.size(); ++j) {
                        auto &&imu_node = imu_deque[j];
                        auto &&feature_in_cameras = imu_node->features_in_cameras.find(feature_id);
                        if (feature_in_cameras == imu_node->features_in_cameras.end()) {
                            continue;
                        }

                        // 构建重投影edge
                        auto edge_reproj = std::make_shared<EdgeReprojectionPoint3d>(feature_in_cameras->second[0].second);
                        edge_reproj->add_vertex(feature_node->vertex_point3d);
                        edge_reproj->add_vertex(imu_node->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[0]);

                        // 把edge加入到problem中
                        problem.add_edge(edge_reproj);
                    }

                    // 当前imu
                    if (is_feature_in_curr_imu) {
                        auto &&feature_in_cameras = _imu_node->features_in_cameras.find(feature_id);
                        if (feature_in_cameras == _imu_node->features_in_cameras.end()) {
                            continue;
                        }

                        // 构建重投影edge
                        auto edge_reproj = std::make_shared<EdgeReprojectionPoint3d>(feature_in_cameras->second[0].second);
                        edge_reproj->add_vertex(feature_node->vertex_point3d);
                        edge_reproj->add_vertex(_imu_node->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[0]);

                        // 把edge加入到problem中
                        problem.add_edge(edge_reproj);
                    }
                }
            }
        }

        // 优化
        problem.solve(10);

        // 解锁imu
        if (fixed_poses) {
            for (auto &pose : *fixed_poses) {
                pose->set_fixed(false);
            }
        }
    }
}