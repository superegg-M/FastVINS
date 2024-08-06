#include<iostream>
#include <random>

#include "backend/problem.h"
#include "backend/vertex.h"
#include "backend/edge.h"


using namespace std;
using namespace graph_optimization;

class VertexCurveFitting : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCurveFitting() : Vertex(3, 3) {}

    std::string type_info() const override { return "VertexCurveFitting"; }

    // void plus(const VecX &delta) override {
    //     VecX &params = parameters();
    //     params += delta;
    // };
};

class EdgeCurveFitting : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeCurveFitting(double x) : Edge(1, 1), _x(x) {}

        /// 返回边的类型信息
        std::string type_info() const override { return "EdgeCurveFitting"; }

        /// 计算残差
        void compute_residual() override {
            const Vec3 abc = _vertices[0]->get_parameters();
            _residual[0] = _observation[0] - std::exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        };

        /// 计算雅可比
        void compute_jacobians() override {
            const Vec3 abc = _vertices[0]->get_parameters();
            double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
            _jacobians[0] = Eigen::Matrix<double, 1, 3>(-_x * _x * y,-_x * y,-y);
        };

public:
    double _x;
};



int main(int argc, char const *argv[])
{
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
    int N = 100;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值

    // 生成高斯噪声
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, w_sigma);

    vector<double> x_data, y_data;      // 数据
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + distribution(generator));
    }

    Problem problem;

    shared_ptr<VertexCurveFitting> vertex(new VertexCurveFitting);
    vertex->set_parameters(Vec3(ae,be,ce));

    problem.add_vertex(vertex);

    // 往图中增加边
    for (int i = 0; i < N; i++) {
        shared_ptr<EdgeCurveFitting> edge(new EdgeCurveFitting(x_data[i]));
        edge->add_vertex(vertex);                // 设置连接的顶点
        edge->set_observation(Vec1(y_data[i]));      // 观测数值
        edge->set_information(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // 信息矩阵：协方差矩阵之逆
        problem.add_edge(edge);
    }

    // 执行优化
    cout << "start optimization" << endl;
    // problem.set_solver_type(graph_optimization::Problem::SolverType::GAUSS_NEWTON);
    problem.solve(10);
    problem.solve(10);


    // 输出优化值
    Vec3 abc_estimate = vertex->get_parameters();
    cout << "estimated model: " << abc_estimate.transpose() << endl;

    return 0;
}
