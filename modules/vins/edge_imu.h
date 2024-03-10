//
// Created by Cain on 2024/3/10.
//

#ifndef GRAPH_OPTIMIZATION_EDGE_IMU_H
#define GRAPH_OPTIMIZATION_EDGE_IMU_H

#include <lib/backend/edge.h>
#include <utility>
#include "imu_integration.h"

namespace graph_optimization {
    class EdgeImu : public Edge {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        explicit EdgeImu(const vins::IMUIntegration &imu_integration);

        std::string type_info() const override { return "EdgeImu"; }   ///< 返回边的类型信息
        void compute_residual() override;   ///< 计算残差
        void compute_jacobians() override;  ///< 计算雅可比

    protected:
        vins::IMUIntegration _imu_integration;
    };
}

#endif //GRAPH_OPTIMIZATION_EDGE_IMU_H
