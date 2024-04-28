//
// Created by Cain on 2024/1/2.
//

#ifndef GRAPH_OPTIMIZATION_EDGE_REPROJECTION_H
#define GRAPH_OPTIMIZATION_EDGE_REPROJECTION_H

//#include <lib/backend/edge.h>

#include <utility>
#include "backend/edge.h"

namespace graph_optimization {
    class EdgeReprojection : public Edge {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        explicit EdgeReprojection(Vec3 pt_i, Vec3 pt_j)
                : Edge(2, 4, std::vector<std::string>{"VertexInverseDepth", "VertexPose", "VertexPose", "VertexPose"},
                       LossFunction::Type::CAUCHY),
                  _pt_i(std::move(pt_i)), _pt_j(std::move(pt_j)) {}

        /// 返回边的类型信息
        std::string type_info() const override { return "EdgeReprojection"; }

        /// 计算残差
        void compute_residual() override;

        /// 计算雅可比
        void compute_jacobians() override;

        void set_translation_imu_from_camera(Eigen::Quaterniond &qic, Vec3 &tic) { _qic = qic; _tic = tic; }

    private:
        Qd _qic;
        Vec3 _tic;
        Vec3 _pt_i, _pt_j;
    };

    class EdgeReprojectionLocal : public Edge {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        explicit EdgeReprojectionLocal(Vec3 pt_i, Vec3 pt_j);

        /// 返回边的类型信息
        std::string type_info() const override { return "EdgeReprojectionLocal"; }

        /// 计算残差
        void compute_residual() override;

        /// 计算雅可比
        void compute_jacobians() override;

        void set_translation_imu_from_camera(Eigen::Quaterniond &qic, Vec3 &tic) { _qic = qic; _tic = tic; }

    private:
        Qd _qic;
        Vec3 _tic;
        Vec3 _pt_i, _pt_j;
        Mat23 _Nj;
    };


    class EdgeReprojectionPoint3d : public Edge {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        explicit EdgeReprojectionPoint3d(Vec3 pt_i)
                : Edge(2, 3, std::vector<std::string>{"VertexPoint3d", "VertexPose", "VertexPose"}),
                  _pt_i(std::move(pt_i)) {}

        /// 返回边的类型信息
        std::string type_info() const override { return "EdgeReprojectionPoint3d"; }

        /// 计算残差
        void compute_residual() override;

        /// 计算雅可比
        void compute_jacobians() override;

        void set_translation_imu_from_camera(Eigen::Quaterniond &qic, Vec3 &tic) { _qic = qic; _tic = tic; }

    private:
        Qd _qic;
        Vec3 _tic;
        Vec3 _pt_i;
    };
}

#endif //GRAPH_OPTIMIZATION_EDGE_REPROJECTION_H
