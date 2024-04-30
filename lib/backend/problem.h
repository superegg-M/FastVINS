//
// Created by Cain on 2023/12/28.
//

#ifndef GRAPH_OPTIMIZATION_PROBLEM_H
#define GRAPH_OPTIMIZATION_PROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>

#include "eigen_types.h"
#include "vertex.h"
#include "edge.h"

namespace graph_optimization {
    class Problem {
    public:
        enum class SolverType {
            STEEPEST_DESCENT,
            GAUSS_NEWTON,
            LEVENBERG_MARQUARDT,
            DOG_LEG,
            LBFGS
        };
        typedef unsigned long ulong;
//    typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
        typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
        typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
        typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    public:
        explicit Problem();
        virtual ~Problem() = default;

        virtual bool add_vertex(const std::shared_ptr<Vertex>& vertex);
        virtual bool remove_vertex(const std::shared_ptr<Vertex>& vertex);
        bool add_edge(const std::shared_ptr<Edge>& edge);
        bool remove_edge(const std::shared_ptr<Edge>& edge);
        void extend_prior_hessian_size(ulong dim);
        bool solve(unsigned long iterations);

    public:
        std::vector<std::shared_ptr<Edge>> get_connected_edges(const std::shared_ptr<Vertex>& vertex);  ///< 获取某个顶点连接到的边
        void get_outlier_edges(std::vector<std::shared_ptr<Edge>> &outlier_edges);  ///< 取得在优化中被判断为outlier部分的边，方便前端去除outlier
        MatXX get_h_prior() const { return _h_prior; }
        VecX get_b_prior() const { return _b_prior; }
        double get_chi2() const { return _chi2; }

    public:
        void set_solver_type(SolverType type) { _solver_type = type; }
        void set_h_prior(const MatXX &h_prior) { _h_prior = h_prior; }
        void set_b_prior(const VecX &b_prior) { _b_prior = b_prior; }

        //test compute prior
        void test_compute_prior();

        void test_marginalize();

    protected:
        virtual void initialize_ordering();    ///< 设置各顶点的ordering_index
        virtual bool check_ordering();  ///< 检查ordering是否正确
        void update_states(const VecX &delta_x);    ///< x_bp = x, x = x + Δx, b_prior_bp = b_prior, b_prior = b_prior - H_prior * Δx
        void rollback_states(const VecX &delta_x);  ///< x = x_bp, b_prior = b_prior_bp
        virtual void update_prior(const VecX &delta_x); ///< b_prior_bp = b_prior, b_prior = b_prior - H_prior * Δx; 在update_states()中运行
        void update_residual(); ///< 计算每条边的残差;      运行顺序必须遵循: update_state -> update_residual
        void update_jacobian(); ///< 计算每条边的残差;      运行顺序必须遵循: update_residual -> update_jacobian
        void update_chi2(); ///< 计算综合的chi2;           运行顺序必须遵循: update_residual -> update_chi2
        void update_gradient();
        void update_hessian();    ///< 计算H, b, J, f;    运行顺序必须遵循: update_jacobian -> update_hessian
        virtual void add_prior_to_hessian();    ///< H = H + H_prior;   在make_hessian()时会运行
        void initialize_lambda();   ///< 计算λ;   运行顺序必须遵循: update_hessian -> initialize_lambda
        virtual VecX multiply_hessian(const VecX &x); ///< 计算: Hx
        virtual double calculate_hessian_norm_square(const VecX &x);    ///< 计算: x'Hx
        virtual bool solve_linear_system(VecX &delta_x);    ///< 解: (H+λ)Δx = b
        bool one_step_steepest_descent(VecX &delta_x);  ///< 计算: h_sd = alpha*g
        bool one_step_gauss_newton(VecX &delta_x);  ///< 计算: h_gn = (H+λ)/g;   运行顺序必须遵循: update_hessian -> one_step_gauss_newton
        bool calculate_steepest_descent(VecX &delta_x, unsigned long iterations=10);
        bool calculate_gauss_newton(VecX &delta_x, unsigned long iterations=10);
        bool calculate_levenberg_marquardt(VecX &delta_x, unsigned long iterations=10);
        bool calculate_dog_leg(VecX &delta_x, unsigned long iterations=10);
        bool calculate_lbfgs(VecX &delta_x, unsigned long iterations=10);


        /// 计算并更新Prior部分
        void compute_prior();

        void logout_vector_size();

        void save_hessian_diagonal_elements();
        void load_hessian_diagonal_elements();

        /// PCG 迭代线性求解器
        VecX PCG_solver(const MatXX &A, const VecX &b, unsigned long max_iter=0);

    protected:
        bool _debug = false;
        SolverType _solver_type {SolverType::DOG_LEG};

        double _t_residual_cost = 0.;
        double _t_chi2_cost = 0.;
        double _t_jacobian_cost = 0.;
        double _t_hessian_cost = 0.;
        double _t_PCG_solve_cost = 0.;

        ulong _ordering_generic = 0;
        double _chi2 {0.};

        VecX _gradient;
        MatXX _hessian;
        VecX _b;
        VecX _hessian_diag;
        VecX _delta_x;
        VecX _delta_x_sd;
        VecX _delta_x_gn;
        VecX _delta_x_lm;
        VecX _delta_x_dl;
        VecX _delta_x_lbfgs;

        MatXX _h_prior;
        VecX _b_prior;
        VecX _b_prior_bp;
        MatXX _jt_prior_inv;
        VecX _err_prior;

        HashVertex _vertices;   ///< 所有的顶点
        HashEdge _edges;    ///< 所有的边
        HashVertexIdToEdge _vertex_to_edge;     ///< pair(顶点id, 与该顶点相连的所有边)
        HashVertex _vertices_marg;  ///< 需要被边缘化的顶点

        // Gauss-Newton or Levenberg-Marquardt
        double _ni {2.};                 //控制 lambda 缩放大小
        VecX _diag_lambda;
        double _current_lambda {0.};
        double _lambda_min {1e-6};
        double _lambda_max {1e6};

        // Dog leg
        double _delta {100.};
        double _delta_min {1e-6};
        double _delta_max {1e6};
    };
}

#endif //GRAPH_OPTIMIZATION_PROBLEM_H
