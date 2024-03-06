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

        /**
         * 问题的类型
         * SLAM问题还是通用的问题
         *
         * 如果是SLAM问题那么pose和landmark是区分开的，Hessian以稀疏方式存储
         * SLAM问题只接受一些特定的Vertex和Edge
         * 如果是通用问题那么hessian是稠密的，除非用户设定某些vertex为marginalized
         */
        enum class ProblemType {
            SLAM_PROBLEM,
            GENERIC_PROBLEM
        };
        enum class SolverType {
            STEEPEST_DESCENT,
            GAUSS_NEWTON,
            LEVENBERG_MARQUARDT,
            DOG_LEG
        };
        typedef unsigned long ulong;
//    typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
        typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
        typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
        typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        explicit Problem(ProblemType problemType);

        ~Problem() = default;

        bool add_vertex(const std::shared_ptr<Vertex>& vertex);

        /**
         * remove a vertex
         * @param vertex_to_remove
         */
        bool remove_vertex(const std::shared_ptr<Vertex>& vertex);

        bool add_edge(const std::shared_ptr<Edge>& edge);

        bool remove_edge(const std::shared_ptr<Edge>& edge);

        /**
         * 取得在优化中被判断为outlier部分的边，方便前端去除outlier
         * @param outlier_edges
         */
        void get_outlier_edges(std::vector<std::shared_ptr<Edge>> &outlier_edges);

        /**
         * 求解此问题
         * @param iterations
         * @return
         */
        bool solve(int iterations);

        /// 边缘化一个frame和以它为host的landmark
        bool marginalize(std::shared_ptr<Vertex> frameVertex,
                         const std::vector<std::shared_ptr<Vertex>> &landmarkVerticies);

        bool marginalize(const std::shared_ptr<Vertex>& vertex_pose, const std::shared_ptr<Vertex>& vertex_motion);

        void extend_prior_hessian_size(ulong dim);

        MatXX get_h_prior() const { return _h_prior; }
        VecX get_b_prior() const { return _b_prior; }

        //test compute prior
        void test_compute_prior();

        void test_marginalize();

    public:

        /// Solve的实现，解通用问题
        bool solve_generic_problem(int iterations);

        /// Solve的实现，解SLAM问题
        bool solve_SLAM_problem(int iterations);

        /// 设置各顶点的ordering_index
        void set_ordering();

        /// set ordering for new vertex in slam problem
        void add_ordering_SLAM(const std::shared_ptr<Vertex>& v);

        /// 构造大H矩阵
        void make_hessian();

        void calculate_jacobian();
        void calculate_negative_gradient();
        void calculate_hessian();
        void calculate_hessian_and_negative_gradient();

        void initialize_lambda();

        double get_robust_chi2() const;

        bool calculate_steepest_descent(VecX &delta_x);
        bool calculate_gauss_newton(VecX &delta_x, unsigned long iterations=10);
        bool calculate_levenberg_marquardt(VecX &delta_x, unsigned long iterations);
        bool calculate_dog_leg(VecX &delta_x, unsigned long iterations);

        /// schur求解SBA
        void schur_SBA(VecX &delta_x);

        /// 解线性方程
        void solve_linear_system(VecX &delta_x);

        /// 更新状态变量
        void update_states(const VecX &delta_x);

        void rollback_states(const VecX &delta_x); // 有时候 update 后残差会变大，需要退回去，重来

        /// 计算并更新Prior部分
        void compute_prior();

        /// 判断一个顶点是否为Pose顶点
        static bool is_pose_vertex(const std::shared_ptr<Vertex>& v);

        /// 判断一个顶点是否为landmark顶点
        static bool is_landmark_vertex(const std::shared_ptr<Vertex>& v);

        /// 在新增顶点后，需要调整几个hessian的大小
        void resize_pose_hessian_when_adding_pose(const std::shared_ptr<Vertex>& v);

        /// 检查ordering是否正确
        bool check_ordering();

        void logout_vector_size();

        /// 获取某个顶点连接到的边
        std::vector<std::shared_ptr<Edge>> get_connected_edges(const std::shared_ptr<Vertex>& vertex);

        /// Levenberg
        /// 计算LM算法的初始Lambda
        void compute_lambda_init_LM();

        /// Hessian 对角线加上或者减去  Lambda
        void add_lambda_to_hessian_LM();

        void remove_lambda_hessian_LM();

        void save_hessian_diagonal_elements();

        void load_hessian_diagonal_elements();

        /// LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
        bool is_good_step_in_LM();

        /// PCG 迭代线性求解器
        static VecX PCG_solver(const MatXX &A, const VecX &b, int maxIter=-1);

        VecX _diag_lambda;
        double _current_lambda {0.};
        double _current_chi {0.};
        double _stop_threshold_LM {0.};    // LM 迭代退出阈值条件
        double _ni {2.};                 //控制 Lambda 缩放大小
        double _lambda_min {1e-6};
        double _lambda_max {1e6};

        double _delta {100.};
        double _delta_max {10000.};

        ProblemType _problem_type;

        /// 整个信息矩阵
        MatXX _hessian;
        VecX _b;
        VecX _delta_x;
        VecX _delta_x_sd;
        VecX _delta_x_gn;
        VecX _delta_x_lm;
        VecX _delta_x_dl;
        VecX _hessian_diag;

        /// 先验部分信息
        MatXX _h_prior;
        VecX _b_prior;
        VecX _b_prior_bp;
        MatXX _jt_prior_inv;
        VecX _err_prior;

        /// SBA的Pose部分
        MatXX _h_pp_schur;
        VecX _b_pp_schur;
        // Heesian 的 Landmark 和 pose 部分
        MatXX _h_pp;
        VecX _b_pp;
        MatXX _h_ll;
        VecX _b_ll;

        /// all vertices
        HashVertex _vertices;

        /// all edges
        HashEdge _edges;

        /// 由vertex id查询edge
        HashVertexIdToEdge _vertex_to_edge;

        /// Ordering related
        ulong _ordering_poses = 0;
        ulong _ordering_landmarks = 0;
        ulong _ordering_generic = 0;
        std::map<unsigned long, std::shared_ptr<Vertex>> _idx_pose_vertices;        // 以ordering排序的pose顶点
        std::map<unsigned long, std::shared_ptr<Vertex>> _idx_landmark_vertices;    // 以ordering排序的landmark顶点

        // vertices need to marg. <Ordering_id_, Vertex>
        HashVertex _verticies_marg;

        bool _debug = false;
        double _t_jacobian_cost = 0.;
        double _t_gradient_cost = 0.;
        double _t_hessian_cost = 0.;
        double _t_PCG_sovle_cost = 0.;
    };
}

#endif //GRAPH_OPTIMIZATION_PROBLEM_H
