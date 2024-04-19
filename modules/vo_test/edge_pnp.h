//
// Created by Cain on 2024/4/9.
//

#ifndef GRAPH_OPTIMIZATION_EDGE_PNP_H
#define GRAPH_OPTIMIZATION_EDGE_PNP_H

#include <utility>
#include "backend/edge.h"

namespace graph_optimization {
    class EdgePnP : public Edge {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        explicit EdgePnP(Vec3 point_pixel, Vec3 point_world)
                : Edge(2, 1, std::vector<std::string>{"VertexPose"}),
                  _p_pixel(std::move(point_pixel)), _p_world(std::move(point_world)) {}

        /// 返回边的类型信息
        std::string type_info() const override { return "EdgePnP"; }

        /// 计算残差
        void compute_residual() override;

        /// 计算雅可比
        void compute_jacobians() override;

        void set_translation_imu_from_camera(Eigen::Quaterniond &qic, Vec3 &tic) { _q_ic = qic; _t_ic = tic; }

    private:
        Qd _q_ic;
        Vec3 _t_ic;
        Vec3 _p_pixel;
        Vec3 _p_world;
    };
}

#endif //GRAPH_OPTIMIZATION_EDGE_PNP_H
