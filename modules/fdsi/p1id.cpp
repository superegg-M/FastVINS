//
// Created by Cain on 2023/12/29.
//

#include <iostream>
#include "p1id.h"

namespace system_identification {
    namespace frequency_domain {
        using namespace graph_optimization;


        void P1IDEdge::compute_residual() {
            const Vec3 &p1id = _vertices[0]->parameters();  // 估计的参数
            double Kp = p1id(0), Tp = p1id(1), Td = p1id(2);
            double Tpw = Tp * _w;
            double den = _w * (1. + Tpw * Tpw);
            double delay_ang = Td * _w;
            double cos_ang = cos(delay_ang);
            double sin_ang = sin(delay_ang);
            double re = -Kp * (Tpw * cos_ang + sin_ang) / den;
            double im = Kp * (-cos_ang + Tpw * sin_ang) / den;
            _residual(0) = re - _re;
            _residual(1) = im - _im;
        }

        void P1IDEdge::compute_jacobians() {
            const Vec3 &p1id = _vertices[0]->parameters();
            double Kp = p1id(0), Tp = p1id(1), Td = p1id(2);
            double Tpw = Tp * _w;
            double den = (1. + Tpw * Tpw);
            double den2 = den * den;
            double wden = _w * den;
            double delay_ang = Td * _w;
            double cos_ang = cos(delay_ang);
            double sin_ang = sin(delay_ang);

            Eigen::Matrix<double, 2, 3> jacobians;
            jacobians(0, 0) = -(Tpw * cos_ang + sin_ang) / wden;
            jacobians(0, 1) = Kp * ((-1. + Tpw * Tpw) * cos_ang + 2. * Tpw * sin_ang) / den2;
            jacobians(0, 2) = Kp * (-cos_ang + Tpw * sin_ang) / den;
            jacobians(1, 0) = (-cos_ang + Tpw * sin_ang) / wden;
            jacobians(1, 1) = Kp * (sin_ang + Tpw * (2. * cos_ang - Tpw * sin_ang)) / den2;
            jacobians(1, 2) = Kp * (Tpw * cos_ang + sin_ang) / den;
            _jacobians[0] = jacobians;
        }

        void P1IDSolver::operator()(const std::vector<double> &re, const std::vector<double> &im, const std::vector<double> &w, const std::vector<unsigned> &index) {
            // 构建 problem
            Problem problem;
            std::shared_ptr<P1IDVertex> vertex(new P1IDVertex());

            // 设定待估计参数 Kp, Tp, Td初始值
            vertex->set_parameters(Eigen::Vector3d (_Kp,_Tp,_Td));
            // 将待估计的参数加入最小二乘问题
            problem.add_vertex(vertex);

            for (auto &i : index) {
                // 每个观测对应的残差函数
                std::shared_ptr<P1IDEdge> edge(new P1IDEdge(w[i], re[i], im[i]));
                std::vector<std::shared_ptr<Vertex>> edge_vertex;
                edge_vertex.push_back(vertex);
                edge->set_vertices(edge_vertex);

                // 把这个残差添加到最小二乘问题
                problem.add_edge(edge);
            }

            std::cout<<"\nTest CurveFitting start..."<<std::endl;
            /// 使用 LM 求解
            problem.solve(30);

            std::cout << "-------After optimization, we got these parameters :" << std::endl;
            std::cout << vertex->parameters().transpose() << std::endl;
        }

        void P1IDSolver::operator()(const std::vector<double> &re, const std::vector<double> &im, const std::vector<double> &w) {
            // 构建 problem
            Problem problem;
            std::shared_ptr<P1IDVertex> vertex(new P1IDVertex());

            // 设定待估计参数 Kp, Tp, Td初始值
            vertex->set_parameters(Eigen::Vector3d (_Kp,_Tp,_Td));
            // 将待估计的参数加入最小二乘问题
            problem.add_vertex(vertex);

            for (unsigned i = 0; i < w.size(); ++i) {
                // 每个观测对应的残差函数
                std::shared_ptr<P1IDEdge> edge(new P1IDEdge(w[i], re[i], im[i]));
                std::vector<std::shared_ptr<Vertex>> edge_vertex;
                edge_vertex.push_back(vertex);
                edge->set_vertices(edge_vertex);

                // 把这个残差添加到最小二乘问题
                problem.add_edge(edge);
            }

            std::cout<<"\nTest CurveFitting start..."<<std::endl;
            /// 使用 LM 求解
            problem.solve(30);

            std::cout << "-------After optimization, we got these parameters :" << std::endl;
            std::cout << vertex->parameters().transpose() << std::endl;
        }
    }
}