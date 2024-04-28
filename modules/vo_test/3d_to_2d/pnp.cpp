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

    void Estimator::pnp(ImuNode *imu_i, Qd *q_wi_init, Vec3 *t_wi_init) {
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

        // 对imu的pose进行初始化
        Vec7 pose;
        pose << t_wi.x(), t_wi.y(), t_wi.z(),
                q_wi.x(), q_wi.y(), q_wi.z(), q_wi.w();
        imu_i->vertex_pose->set_parameters(pose);

        // 相机外参
        Vec3 t_ic = _t_ic[0];
        Qd q_ic = _q_ic[0];

        Problem problem;
        problem.add_vertex(imu_i->vertex_pose); // 加入imu的位姿
        for (auto &feature_in_cameras : imu_i->features_in_cameras) {
            auto &&feature_it = _feature_map.find(feature_in_cameras.first);
            if (feature_it == _feature_map.end()) {
                std::cout << "Error: feature not in feature_map when running pnp" << std::endl;
                continue;
            }
            if (feature_it->second->vertex_point3d) {
                auto &&point_pixel = feature_in_cameras.second[0].second;
                Vec3 point_world = feature_it->second->vertex_point3d->get_parameters();

                // 重投影edge
                shared_ptr<EdgePnP> edge_pnp(new EdgePnP(point_pixel, point_world));
                edge_pnp->set_translation_imu_from_camera(q_ic, t_ic);
                edge_pnp->add_vertex(imu_i->vertex_pose);

                problem.add_edge(edge_pnp);
            }
        }
        problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
        problem.solve(5);
    }
}