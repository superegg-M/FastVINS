//
// Created by Cain on 2023/12/29.
//

#ifndef GRAPH_OPTIMIZATION_P1ID_H
#define GRAPH_OPTIMIZATION_P1ID_H

#include <lib/backend/problem.h>
#include <lib/backend/eigen_types.h>
#include <vector>

namespace graph_optimization {
    namespace system_identification {
        namespace frequency_domain {
            class P1IDVertex: public Vertex {
            public:
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                P1IDVertex(): Vertex(3) {}  // P1ID: 四个参数， Kp, Tp, Td
                std::string type_info() const override { return "P1ID"; }
            };


            class P1IDEdge: public Edge {
            public:
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                P1IDEdge(double w, double re, double im)
                        : Edge(2,1, std::vector<std::string>{"P1ID"}), _w(w), _re(re), _im(im) {}

                // 计算曲线模型误差
                void compute_residual() override;

                // 计算残差对变量的雅克比
                void compute_jacobians() override;

                // 返回边的类型信息
                std::string type_info() const override { return "P1IDEdge"; }

            private:
                double _w, _re, _im;
            };

            class P1IDSolver {
            public:
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                explicit P1IDSolver(double Kp=200., double Tp=0.02, double Td=0.005) : _Kp(Kp), _Tp(Tp), _Td(Td) {}

                void operator()(const std::vector<double> &re, const std::vector<double> &im, const std::vector<double> &w);
                void operator()(const std::vector<double> &re, const std::vector<double> &im, const std::vector<double> &w, const std::vector<unsigned> &index);

            private:
                double _Kp, _Tp, _Td;
            };
        }
    }
}

#endif //GRAPH_OPTIMIZATION_P1ID_H
