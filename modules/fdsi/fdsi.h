//
// Created by Cain on 2023/12/29.
//

#ifndef GRAPH_OPTIMIZATION_FDSI_H
#define GRAPH_OPTIMIZATION_FDSI_H

//#include <lib/backend/problem.h>
//#include <lib/backend/eigen_types.h>
#include <vector>
#include "backend/problem.h"
#include "backend/eigen_types.h"
#include "fdsi_vertex.h"
#include "fdsi_edge.h"

namespace system_identification {
    namespace frequency_domain {
        using namespace graph_optimization;

        template<unsigned NP,unsigned NZ,unsigned NI>
        class FDSISolver {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            explicit FDSISolver(const std::vector<double> &parameters) {
                for (unsigned i = 0; i < NP + NZ + 2; ++i) {
                    _parameters(i, 0) = parameters[i];
                }
            }

            void operator()(const std::vector<double> &re, const std::vector<double> &im, const std::vector<double> &w);
            void operator()(const std::vector<double> &re, const std::vector<double> &im, const std::vector<double> &w, const std::vector<unsigned> &index);

        private:
            Eigen::Matrix<double, NP+NZ+2, 1> _parameters;
        };


        template<unsigned NP,unsigned NZ,unsigned NI>
        void FDSISolver<NP, NZ, NI>::operator()(const std::vector<double> &re, const std::vector<double> &im, const std::vector<double> &w, const std::vector<unsigned> &index) {
            // 构建 problem
            Problem problem;
            std::shared_ptr<FDSIVertex<NP, NZ, NI>> vertex(new FDSIVertex<NP, NZ, NI>());

            // 设定待估计参数 Kp, Tp, Td初始值
//                vertex->set_parameters();
            vertex->parameters() = _parameters;
            // 将待估计的参数加入最小二乘问题
            problem.add_vertex(vertex);

            for (auto &i : index) {
                // 每个观测对应的残差函数
                std::shared_ptr<FDSIEdge<NP, NZ, NI>> edge(new FDSIEdge<NP, NZ, NI>(w[i], re[i], im[i]));
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

        template<unsigned NP,unsigned NZ,unsigned NI>
        void FDSISolver<NP, NZ, NI>::operator()(const std::vector<double> &re, const std::vector<double> &im, const std::vector<double> &w) {
            // 构建 problem
            Problem problem;
            std::shared_ptr<FDSIVertex<NP, NZ, NI>> vertex(new FDSIVertex<NP, NZ, NI>());

            // 设定待估计参数 Kp, Tp, Td初始值
//                vertex->set_parameters(_parameters);
            vertex->parameters() = _parameters;
            // 将待估计的参数加入最小二乘问题
            problem.add_vertex(vertex);

            for (unsigned i = 0; i < w.size(); ++i) {
                // 每个观测对应的残差函数
                std::shared_ptr<FDSIEdge<NP, NZ, NI>> edge(new FDSIEdge<NP, NZ, NI>(w[i], re[i], im[i]));
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

#endif //GRAPH_OPTIMIZATION_FDSI_H
