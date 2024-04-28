//
// Created by Cain on 2024/4/12.
//

#ifndef GRAPH_OPTIMIZATION_EDGE_EPIPOLAR_H
#define GRAPH_OPTIMIZATION_EDGE_EPIPOLAR_H

#include <utility>
#include "backend/edge.h"

namespace graph_optimization {
    class EdgeEpipolar : public Edge {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        explicit EdgeEpipolar(Vec3 point_pixel1, Vec3 point_pixel2)
                : Edge(1, 2,
                       std::vector<std::string>{"VertexQuaternion", "VertexSpherical"}),
                  _p1(std::move(point_pixel1)), _p2(std::move(point_pixel2)) {}

        /// 返回边的类型信息
        std::string type_info() const override { return "EdgeEpipolar"; }

        /// 计算残差
        void compute_residual() override;

        /// 计算雅可比
        void compute_jacobians() override;

        void set_translation_imu_from_camera(Eigen::Quaterniond &qic, Vec3 &tic) { _q_ic = qic; _t_ic = tic; }

    private:
        Qd _q_ic;
        Vec3 _t_ic;
        Vec3 _p1;
        Vec3 _p2;
    };
}

#endif //GRAPH_OPTIMIZATION_EDGE_EPIPOLAR_H
