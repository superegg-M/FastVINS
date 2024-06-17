//
// Created by Cain on 2024/4/24.
//

#ifndef GRAPH_OPTIMIZATION_EDGE_ALIGN_H
#define GRAPH_OPTIMIZATION_EDGE_ALIGN_H

#include "backend/edge.h"
#include "../imu_integration.h"

namespace graph_optimization {
    class EdgeAlign : public Edge {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        explicit EdgeAlign(const vins::IMUIntegration &imu_integration, const Vec3 &t_ij, const Qd &q_ij, const Qd &q_0i);

        std::string type_info() const override { return "EdgeAlign"; }   ///< 返回边的类型信息
        void compute_residual() override;   ///< 计算残差
        void compute_jacobians() override;  ///< 计算雅可比

    protected:
        vins::IMUIntegration _imu_integration;
        Vec3 _t_ij;
        Qd _q_ij;
        Qd _q_0i;

        static Eigen::Matrix<double, 4, 4> get_quat_left(const Qd &q);
        static Eigen::Matrix<double, 4, 4> get_quat_right(const Qd &q);
    };
}

#endif //GRAPH_OPTIMIZATION_EDGE_ALIGN_H
