//
// Created by Cain on 2023/2/20.
//

#ifndef GRAPH_OPTIMIZATION_EDGE_H
#define GRAPH_OPTIMIZATION_EDGE_H

#include <memory>
#include <string>
#include "eigen_types.h"
#include "loss_function.h"

namespace graph_optimization {
    class Vertex;

    class Edge {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        /**
         * 构造函数，会自动化配雅可比的空间
         * @param residual_dimension 残差维度
         * @param num_verticies 顶点数量
         * @param verticies_types 顶点类型名称，可以不给，不给的话check中不会检查
         */
        explicit Edge(unsigned long residual_dimension, unsigned long num_vertices,
                      const std::vector<std::string> &vertices_types = std::vector<std::string>(),
                      LossFunction::Type loss_function_type=LossFunction::Type::TRIVIAL);

        virtual ~Edge() = default;

        /// 返回id
        unsigned long id() const { return _id; }

        /**
         * 设置一个顶点
         * @param vertex 对应的vertex对象
         */
        bool add_vertex(const std::shared_ptr<Vertex>& vertex) {
            _vertices.emplace_back(vertex);
            return true;
        }

        /**
         * 设置一些顶点
         * @param vertices 顶点，按引用顺序排列
         * @return
         */
        bool set_vertices(const std::vector<std::shared_ptr<Vertex>> &vertices) {
            _vertices = vertices;
            return true;
        }

        bool set_vertex(const std::shared_ptr<Vertex> &vertex, unsigned long which_vertex) {
            if (which_vertex < _vertices.size()) {
                _vertices[which_vertex] = vertex;
                return true;
            }
            return false;
        }

        void set_loss_function(const std::shared_ptr<LossFunction> &loss_function) { _loss_function = loss_function; }

        /// 返回第i个顶点
        std::shared_ptr<Vertex> get_vertex(unsigned long i) { return _vertices[i]; }

        /// 返回所有顶点
        const std::vector<std::shared_ptr<Vertex>> &vertices() const { return _vertices; }

        /// 返回关联顶点个数
        size_t num_vertices() const { return _vertices.size(); }

        /// 返回边的类型信息，在子类中实现
        virtual std::string type_info() const = 0;

        /// 计算残差，由子类实现
        virtual void compute_residual() = 0;

        /// 计算雅可比，由子类实现
        virtual void compute_jacobians() = 0;

        void compute_chi2();

        /// 计算平方误差，会乘以信息矩阵
        double get_chi2() const { return _chi2; }

        double get_robust_chi2() const { return _rho[0]; }

        /// 返回残差
        const VecX &residual() const { return _residual; }

        /// 返回雅可比
        const std::vector<MatXX> &jacobians() const { return _jacobians; }

        /// 设置信息矩阵, information_ = sqrt_Omega = w
        void set_information(const MatXX &information) { _information = information; }

        /// 返回信息矩阵
        MatXX information() const { return _information; }

        void robust_information(double &drho, MatXX &info) const;

        /// 设置观测信息
        void set_observation(const VecX &observation) { _observation = observation; }

        /// 返回观测信息
        VecX observation() const { return _observation; }

        /// 检查边的信息是否全部设置
        bool check_valid();

//        unsigned long ordering_id() const { return _ordering_id; }
//
//        void set_ordering_id(unsigned long id) { _ordering_id = id; };

    protected:
        unsigned long _id;  ///< edge id
//        unsigned long _ordering_id;   ///< edge id in problem
        std::vector<std::string> _vertices_types;  ///< 各顶点类型信息，用于debug
        std::vector<std::shared_ptr<Vertex>> _vertices; ///< 该边对应的顶点
        VecX _residual;                 ///< 残差
        std::vector<MatXX> _jacobians;  ///< 雅可比，每个顶点对应一个雅可比, 每个雅可比得维度是 residual x vertex.local_dimension
        double _chi2 {0};               ///< e^2
        Vec3 _rho;                      ///< rho(e^2), rho'(e^2), rho''(e^2)
        MatXX _information;             ///< 信息矩阵
        VecX _observation;              ///< 观测信息
        std::shared_ptr<LossFunction> _loss_function;

    private:
        static unsigned long _global_edge_id;
    };
}

#endif //GRAPH_OPTIMIZATION_EDGE_H
