//
// Created by Cain on 2023/12/28.
//

#include <iostream>
#include <fstream>

//#include <glog/logging.h>
#include <Eigen/Dense>
#include <lib/tic_toc/tic_toc.h>
#include "problem.h"


using namespace std;

namespace graph_optimization {
    Problem::Problem(ProblemType problem_type) : _problem_type(problem_type) {
        logout_vector_size();
        _verticies_marg.clear();
    }

    bool Problem::add_vertex(const std::shared_ptr<Vertex>& vertex) {
        if (_vertices.find(vertex->id()) != _vertices.end()) {
            // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
            return false;
        }
        _vertices.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->id(), vertex));

        if (_problem_type == ProblemType::SLAM_PROBLEM) {
            if (is_pose_vertex(vertex)) {
                resize_pose_hessian_when_adding_pose(vertex);
            }
        }

        return true;
    }

    bool Problem::remove_vertex(const std::shared_ptr<Vertex>& vertex) {
        //check if the vertex is in map_verticies_
        if (_vertices.find(vertex->id()) == _vertices.end()) {
            // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
            return false;
        }

        // 这里要 remove 该顶点对应的 edge.
        auto &&edges = get_connected_edges(vertex);
        for (auto & edge : edges) {
            remove_edge(edge);
        }

        if (is_pose_vertex(vertex)) {
            _idx_pose_vertices.erase(vertex->id());
        }
        else {
            _idx_landmark_vertices.erase(vertex->id());
        }

        vertex->set_ordering_id(-1);      // used to debug
        _vertices.erase(vertex->id());
        _vertex_to_edge.erase(vertex->id());

        return true;
    }

    bool Problem::add_edge(const shared_ptr<Edge>& edge) {
        if (_edges.find(edge->id()) == _edges.end()) {
            _edges.insert(pair<ulong, std::shared_ptr<Edge>>(edge->id(), edge));
        } else {
            // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
            return false;
        }

        for (auto &vertex: edge->vertices()) {
            _vertex_to_edge.insert(pair<ulong, shared_ptr<Edge>>(vertex->id(), edge));
        }
        return true;
    }

    bool Problem::remove_edge(const std::shared_ptr<Edge>& edge) {
        //check if the edge is in map_edges_
        if (_edges.find(edge->id()) == _edges.end()) {
            // LOG(WARNING) << "The edge " << edge->Id() << " is not in the problem!" << endl;
            return false;
        }

        _edges.erase(edge->id());
        return true;
    }

    double Problem::get_robust_chi2() const {
        double chi2 = 0.;
        for (auto &edge: _edges) {
            chi2 += edge.second->robust_chi2();
        }
        return chi2;
    }

    bool Problem::solve(int iterations) {
        if (_edges.empty() || _vertices.empty()) {
            std::cerr << "\nCannot solve problem without edges or vertices" << std::endl;
            return false;
        }

        TicToc t_solve;
        // 统计优化变量的维数: _ordering_generic，为构建 H 矩阵做准备
        set_ordering();
        // 遍历edge, 构建 H = J^T * J 矩阵
        make_hessian();
        // LM 初始化
        compute_lambda_init_LM();
        // LM 算法迭代求解
        bool stop = false;
        int iter = 0;
        double last_chi = _current_chi;
        while (!stop && (iter < iterations)) {
            bool one_step_success = false;
            int false_cnt = 0;
            while (!one_step_success) {  // 不断尝试 Lambda, 直到成功迭代一步
                // setLambda
//                add_lambda_to_hessian_LM();
                // 第四步，解线性方程 (H + λI) X = B
                solve_linear_system(_delta_x);
                //
//                remove_lambda_hessian_LM();

                // 优化退出条件1： delta_x_ 很小则退出
                if (_delta_x.squaredNorm() <= 1e-12 || false_cnt > 10) {
                    stop = true;
                    break;
                }

                // 更新状态量 X = X+ delta_x
                update_states(_delta_x);
                // 判断当前步是否可行以及 LM 的 lambda 怎么更新
                one_step_success = is_good_step_in_LM();
                // 后续处理，
                if (one_step_success) {
                    // 在新线性化点 构建 hessian
                    make_hessian();
                    // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
//                double b_max = 0.0;
//                for (int i = 0; i < b_.size(); ++i) {
//                    b_max = max(fabs(b_(i)), b_max);
//                }
//                // 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
//                stop = (b_max <= 1e-12);
                    false_cnt = 0;
                } else {
                    false_cnt++;
                    rollback_states(_delta_x);   // 误差没下降，回滚
                }
            }
            iter++;

            // 优化退出条件3： currentChi_ 跟第一次的chi2相比，下降了 1e6 倍则退出
            if (fabs(last_chi - _current_chi) < 1e-3 * last_chi || _current_chi < _stop_threshold_LM) {
                std::cout << "fabs(last_chi_ - currentChi_) < 1e-3 * last_chi_ || currentChi_ < stopThresholdLM_" << std::endl;
                stop = true;
            }
            last_chi = _current_chi;

            std::cout << "iter: " << iter << " , chi= " << _current_chi << " , lambda= " << _current_lambda << std::endl;
        }
        std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
        std::cout << "   makeHessian cost: " << _t_hessian_cost << " ms" << std::endl;
        return true;
    }

    void Problem::set_ordering() {
        // 每次重新计数
        _ordering_poses = 0;
        _ordering_landmarks = 0;
        _ordering_generic = 0;

        // Note:: _vertices 是 map 类型的, 顺序是按照 id 号排序的
        // 统计带估计的所有变量的总维度
        for (auto &vertex: _vertices) {
            _ordering_generic += vertex.second->local_dimension();  // 所有的优化变量总维数

            if (_problem_type == ProblemType::SLAM_PROBLEM) {    // 如果是 slam 问题，还要分别统计 pose 和 landmark 的维数，后面会对他们进行排序
                add_ordering_SLAM(vertex.second);
            }
        }

        if (_problem_type == ProblemType::SLAM_PROBLEM) {
            // 这里要把 landmark 的 ordering 加上 pose 的数量，就保持了 landmark 在后,而 pose 在前
            ulong all_pose_dimension = _ordering_poses;
            for (auto &landmark_vertex : _idx_landmark_vertices) {
                landmark_vertex.second->set_ordering_id(landmark_vertex.second->ordering_id() + all_pose_dimension);
            }
        }
    }

    void Problem::add_ordering_SLAM(const std::shared_ptr<Vertex>& v) {
        if (is_pose_vertex(v)) {
            v->set_ordering_id(_ordering_poses);
            _idx_pose_vertices.insert(pair<ulong, std::shared_ptr<Vertex>>(v->id(), v));
            _ordering_poses += v->local_dimension();
        } else if (is_landmark_vertex(v)) {
            v->set_ordering_id(_ordering_landmarks);
            _idx_landmark_vertices.insert(pair<ulong, std::shared_ptr<Vertex>>(v->id(), v));
            _ordering_landmarks += v->local_dimension();
        }
    }

    void Problem::calculate_jacobian() {
        TicToc t_h;

        for (auto &edge: _edges) {
            edge.second->compute_residual();
            edge.second->compute_jacobians();

//            double drho;
//            MatXX robust_information(edge.second->information().rows(),edge.second->information().cols());
//            edge.second->robust_information(drho, robust_information);
            // TODO: 应该在这里进行所有的information计算
        }

        _t_jacobian_cost += t_h.toc();
    }

    void Problem::calculate_negative_gradient() {
        TicToc t_h;

        ulong size = _ordering_generic;
        VecX b(VecX::Zero(size));       ///< 负梯度

        // TODO:: accelate, accelate, accelate
//#ifdef USE_OPENMP
//#pragma omp parallel for
//#endif

        // 遍历每个残差，并计算他们的雅克比，得到最后的 b = -J^T * f
        for (auto &edge: _edges) {
            auto &&jacobians = edge.second->jacobians();
            auto &&verticies = edge.second->vertices();
            assert(jacobians.size() == verticies.size());
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                double drho;
                MatXX robust_information(edge.second->information().rows(),edge.second->information().cols());
                edge.second->robust_information(drho, robust_information);

                MatXX JtW = jacobian_i.transpose() * robust_information;
                b.segment(index_i, dim_i).noalias() -= drho * JtW * edge.second->residual();
            }
        }
        _b = b;

        // 叠加先验
        if(_b_prior.rows() > 0) {
            VecX b_prior_tmp = _b_prior;

            /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
            /// landmark 没有先验
            for (const auto& vertex: _vertices) {
                if (is_pose_vertex(vertex.second) && vertex.second->is_fixed() ) {
                    ulong idx = vertex.second->ordering_id();
                    ulong dim = vertex.second->local_dimension();
                    b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
            _b.head(_ordering_poses) += b_prior_tmp;
        }

        _t_gradient_cost += t_h.toc();
    }

    void Problem::calculate_hessian() {
        TicToc t_h;

        ulong size = _ordering_generic;
        MatXX H(MatXX::Zero(size, size));       ///< Hessian矩阵

        // TODO:: accelate, accelate, accelate
//#ifdef USE_OPENMP
//#pragma omp parallel for
//#endif

        // 遍历每个残差，并计算他们的雅克比，得到最后的 H = J^T * J
        for (auto &edge: _edges) {
            auto &&jacobians = edge.second->jacobians();
            auto &&verticies = edge.second->vertices();
            assert(jacobians.size() == verticies.size());
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                double drho;
                MatXX robust_information(edge.second->information().rows(),edge.second->information().cols());
                edge.second->robust_information(drho, robust_information);

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto &&v_j = verticies[j];

                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    assert(v_j->ordering_id() != -1);
                    MatXX hessian = JtW * jacobian_j;   // TODO: 这里能继续优化, 因为J'*W*J也是对称矩阵
                    // 所有的信息矩阵叠加起来
                    H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
            }

        }
        _hessian = H;

        // 叠加先验
        if(_h_prior.rows() > 0) {
            MatXX H_prior_tmp = _h_prior;

            /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
            /// landmark 没有先验
            for (const auto& vertex: _vertices) {
                if (is_pose_vertex(vertex.second) && vertex.second->is_fixed() ) {
                    ulong idx = vertex.second->ordering_id();
                    ulong dim = vertex.second->local_dimension();
                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
            _hessian.topLeftCorner(_ordering_poses, _ordering_poses) += H_prior_tmp;
        }

        _t_hessian_cost += t_h.toc();
    }

    void Problem::calculate_hessian_and_negative_gradient() {
        TicToc t_h;

        ulong size = _ordering_generic;
        MatXX H(MatXX::Zero(size, size));       ///< Hessian矩阵
        VecX b(VecX::Zero(size));       ///< 负梯度

        // TODO:: accelate, accelate, accelate
//#ifdef USE_OPENMP
//#pragma omp parallel for
//#endif

        // 遍历每个残差，并计算他们的雅克比，得到最后的 H = J^T * J
        for (auto &edge: _edges) {
            auto &&jacobians = edge.second->jacobians();
            auto &&verticies = edge.second->vertices();
            assert(jacobians.size() == verticies.size());
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                double drho;
                MatXX robust_information(edge.second->information().rows(),edge.second->information().cols());
                edge.second->robust_information(drho, robust_information);

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto &&v_j = verticies[j];

                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    assert(v_j->ordering_id() != -1);
                    MatXX hessian = JtW * jacobian_j;   // TODO: 这里能继续优化, 因为J'*W*J也是对称矩阵
                    // 所有的信息矩阵叠加起来
                    H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                b.segment(index_i, dim_i).noalias() -= drho * JtW * edge.second->residual();
            }
        }
        _hessian = H;
        _b = b;

        // 叠加先验
        if(_h_prior.rows() > 0) {
            MatXX H_prior_tmp = _h_prior;
            VecX b_prior_tmp = _b_prior;

            /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
            /// landmark 没有先验
            for (const auto& vertex: _vertices) {
                if (is_pose_vertex(vertex.second) && vertex.second->is_fixed() ) {
                    ulong idx = vertex.second->ordering_id();
                    ulong dim = vertex.second->local_dimension();
                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
            _hessian.topLeftCorner(_ordering_poses, _ordering_poses) += H_prior_tmp;
            _b.head(_ordering_poses) += b_prior_tmp;
        }

        _t_hessian_cost += t_h.toc();
    }

    void Problem::make_hessian() {
        TicToc t_h;

        ulong size = _ordering_generic;
        MatXX H(MatXX::Zero(size, size));       ///< Hessian矩阵
        VecX b(VecX::Zero(size));       ///< 负梯度

        // TODO:: accelate, accelate, accelate
//#ifdef USE_OPENMP
//#pragma omp parallel for
//#endif

        // 遍历每个残差，并计算他们的雅克比，得到最后的 H = J^T * J
        for (auto &edge: _edges) {
            edge.second->compute_residual();
            edge.second->compute_jacobians();

            auto &&jacobians = edge.second->jacobians();
            auto &&verticies = edge.second->vertices();
            assert(jacobians.size() == verticies.size());
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto &&v_i = verticies[i];
                if (v_i->is_fixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                double drho;
                MatXX robust_information(edge.second->information().rows(),edge.second->information().cols());
                edge.second->robust_information(drho, robust_information);

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto &&v_j = verticies[j];

                    if (v_j->is_fixed()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    assert(v_j->ordering_id() != -1);
                    MatXX hessian = JtW * jacobian_j;   // TODO: 这里能继续优化, 因为J'*W*J也是对称矩阵
                    // 所有的信息矩阵叠加起来
                    H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                b.segment(index_i, dim_i).noalias() -= drho * JtW * edge.second->residual();
            }

        }
        _hessian = H;
        _b = b;
        _t_hessian_cost += t_h.toc();

        // 叠加先验
        if(_h_prior.rows() > 0) {
            MatXX H_prior_tmp = _h_prior;
            VecX b_prior_tmp = _b_prior;

            /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
            /// landmark 没有先验
            for (const auto& vertex: _vertices) {
                if (is_pose_vertex(vertex.second) && vertex.second->is_fixed() ) {
                    ulong idx = vertex.second->ordering_id();
                    ulong dim = vertex.second->local_dimension();
                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
            _hessian.topLeftCorner(_ordering_poses, _ordering_poses) += H_prior_tmp;
            _b.head(_ordering_poses) += b_prior_tmp;
        }

        _delta_x = VecX::Zero(size);  // initial delta_x = 0_n;
    }

    void Problem::initialize_lambda() {
        double max_diagonal = 0.;
        ulong size = _hessian.cols();
        assert(_hessian.rows() == _hessian.cols() && "Hessian is not square");
        for (ulong i = 0; i < size; ++i) {
            max_diagonal = std::max(fabs(_hessian(i, i)), max_diagonal);
        }

        double tau = 1e-5;  // 1e-5
        _current_lambda = tau * max_diagonal;
        _current_lambda = std::min(std::max(_current_lambda, _lambda_min), _lambda_max);

        _diag_lambda = tau * _hessian.diagonal();
        for (int i = 0; i < _hessian.rows(); ++i) {
            _diag_lambda(i) = std::min(std::max(_diag_lambda(i), _lambda_min), _lambda_max);
        }
    }



    void Problem::schur_SBA(VecX &delta_x) {
        if (delta_x.rows() != (_ordering_poses + _ordering_landmarks)) {
            delta_x.resize(_ordering_poses + _ordering_landmarks, 1);
        }

        /*
         * [Hpp Hpl][dxp] = [bp]
         * [Hlp Hll][dxl] = [bl]
         *
         * (Hpp - Hpl * Hll^-1 * Hlp) * dxp = bp - Hpl * Hll^-1 * bl
         * Hll * dxl = bl - Hlp * dxp
         * */
        ulong reserve_size = _ordering_poses;
        ulong marg_size = _ordering_landmarks;

        // 由于叠加了lambda, 所以能够保证Hll可逆
        MatXX Hll = _hessian.block(reserve_size, reserve_size, marg_size, marg_size);
        for (int i = 0; i < marg_size; ++i) {   // LM Method
            // Hll(i, i) += _current_lambda;
            Hll(i, i) += _diag_lambda(i + reserve_size);
        }
//        MatXX Hpl = _hessian.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hlp = _hessian.block(reserve_size, 0, marg_size, reserve_size);
//        VecX bpp = _b.segment(0, reserve_size);
        VecX bll = _b.segment(reserve_size, marg_size);

        MatXX temp_H(MatXX::Zero(marg_size, reserve_size));  // Hll^-1 * Hpl^T
        VecX temp_b(VecX::Zero(marg_size, 1));   // Hll^-1 * bl
        for (const auto& landmark_vertex : _idx_landmark_vertices) {
            ulong idx = landmark_vertex.second->ordering_id() - reserve_size;
            ulong size = landmark_vertex.second->local_dimension();
            if (size == 1) {
                temp_H.row(idx) = Hlp.row(idx) / Hll(idx, idx);
                temp_b(idx) = bll(idx) / Hll(idx, idx);
            } else {
                auto &&Hll_ldlt = Hll.block(idx, idx, size, size).ldlt();
                temp_H.block(idx, 0, size, reserve_size) = Hll_ldlt.solve(Hlp.block(idx, 0, size, reserve_size));
                temp_b.segment(idx, size) = Hll_ldlt.solve(bll.segment(idx, size));
            }
        }

        // (Hpp - Hpl * Hll^-1 * Hlp) * dxp = bp - Hpl * Hll^-1 * bl
        // 这里即使叠加了lambda, 也有可能因为数值精度的问题而导致 _h_pp_schur 不可逆
        _h_pp_schur = _hessian.block(0, 0, reserve_size, reserve_size) - Hlp.transpose() * temp_H;
        for (ulong i = 0; i < _ordering_poses; ++i) {
            // _h_pp_schur(i, i) += _current_lambda;    // LM Method
            _h_pp_schur(i, i) += _diag_lambda(i);
        }
        _b_pp_schur = _b.segment(0, reserve_size) - Hlp.transpose() * temp_b;

        // Solve" Hpp * delta_x = bpp
        VecX delta_x_pp(VecX::Zero(reserve_size));
#ifdef USE_PCG_SOLVER
        auto n_pcg = _h_pp_schur.rows() * 2;                       // 迭代次数
        delta_x_pp = PCG_solver(_h_pp_schur, _b_pp_schur, n_pcg);
#else
        auto &&H_pp_schur_ldlt = _h_pp_schur.ldlt();
        if (H_pp_schur_ldlt.info() != Eigen::Success) {
//            return false;   // H_pp_schur不是正定矩阵
        }
        delta_x_pp =  H_pp_schur_ldlt.solve(_b_pp_schur);
#endif

        _delta_x.head(reserve_size) = delta_x_pp;

        // Hll * dxl = bl - Hlp * dxp
        VecX delta_x_ll(marg_size);
        delta_x_ll = temp_b - temp_H * delta_x_pp;
        _delta_x.tail(marg_size) = delta_x_ll;
    }

    /*
    * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
    */
    void Problem::solve_linear_system(VecX &delta_x) {
        if (_problem_type == ProblemType::GENERIC_PROBLEM) {
            MatXX H = _hessian;
            for (unsigned i = 0; i < _hessian.rows(); ++i) {
                H(i, i) += _current_lambda;
            }
            delta_x = H.fullPivLu().solve(_b);
        } else {
            // SLAM 问题采用舒尔补的计算方式
            schur_SBA(delta_x);
        }
    }

    void Problem::update_states(const VecX &delta_x) {
        for (auto &vertex: _vertices) {
            ulong idx = vertex.second->ordering_id();
            ulong dim = vertex.second->local_dimension();
            VecX delta = delta_x.segment(idx, dim);

            // 所有的参数 x 叠加一个增量  x_{k+1} = x_{k} + delta_x
            vertex.second->plus(delta);
        }

        // update prior
        _b_prior_bp = _b_prior;
        _b_prior -= _h_prior * delta_x.head(_ordering_poses);
    }

    void Problem::rollback_states(const VecX &delta_x) {
        for (auto &vertex: _vertices) {
            ulong idx = vertex.second->ordering_id();
            ulong dim = vertex.second->local_dimension();
            VecX delta = delta_x.segment(idx, dim);

            // 之前的增量加了后使得损失函数增加了，我们应该不要这次迭代结果，所以把之前加上的量减去。
            vertex.second->plus(-delta);
        }

        _b_prior = _b_prior_bp;
    }

    void Problem::add_lambda_to_hessian_LM() {
        ulong size = _hessian.cols();
        assert(_hessian.rows() == _hessian.cols() && "Hessian is not square");
        for (ulong i = 0; i < size; ++i) {
            _hessian(i, i) += _current_lambda;
        }
    }

    void Problem::remove_lambda_hessian_LM() {
        ulong size = _hessian.cols();
        assert(_hessian.rows() == _hessian.cols() && "Hessian is not square");
        // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
        for (ulong i = 0; i < size; ++i) {
            _hessian(i, i) -= _current_lambda;
        }
    }

    void Problem::save_hessian_diagonal_elements() {
        ulong size = _hessian.cols();
        assert(_hessian.rows() == _hessian.cols() && "Hessian is not square");
        _hessian_diag.resize(size);
        for (ulong i = 0; i < size; ++i) {
            _hessian_diag(i) = _hessian(i, i);
        }
    }

    void Problem::load_hessian_diagonal_elements() {
        ulong size = _hessian.cols();
        assert(_hessian.rows() == _hessian.cols() && "Hessian is not square");
        assert(size == _hessian_diag.size() && "Hessian dimension is wrong");
        for (ulong i = 0; i < size; ++i) {
            _hessian(i, i) = _hessian_diag(i);
        }
    }

    /** @brief conjugate gradient with perconditioning
*
*  the jacobi PCG method
*
*/
    VecX Problem::PCG_solver(const MatXX &A, const VecX &b, int maxIter) {
        assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
        int rows = b.rows();
        int n = maxIter < 0 ? rows : maxIter;
        VecX x(VecX::Zero(rows));
        MatXX M_inv = A.diagonal().asDiagonal().inverse();
        VecX r0(b);  // initial r = b - A*0 = b
        VecX z0 = M_inv * r0;
        VecX p(z0);
        VecX w = A * p;
        double r0z0 = r0.dot(z0);
        double alpha = r0z0 / p.dot(w);
        VecX r1 = r0 - alpha * w;
        int i = 0;
        double threshold = 1e-6 * r0.norm();
        while (r1.norm() > threshold && i < n) {
            i++;
            VecX z1 = M_inv * r1;
            double r1z1 = r1.dot(z1);
            double beta = r1z1 / r0z0;
            z0 = z1;
            r0z0 = r1z1;
            r0 = r1;
            p = beta * p + z1;
            w = A * p;
            alpha = r1z1 / p.dot(w);
            x += alpha * p;
            r1 -= alpha * w;
        }
        return x;
    }

//    void Problem::compute_prior() {
//
//    }

    bool Problem::is_pose_vertex(const std::shared_ptr<Vertex>& v) {
        string type = v->type_info();
        return type == string("VertexPose") || type == string("VertexMotion");
    }

    bool Problem::is_landmark_vertex(const std::shared_ptr<Vertex>& v) {
        string type = v->type_info();
        return type == string("VertexPointXYZ") || type == string("VertexInverseDepth");
    }

    void Problem::resize_pose_hessian_when_adding_pose(const std::shared_ptr<Vertex>& v) {
        unsigned size = _h_prior.rows() + v->local_dimension();
        _h_prior.conservativeResize(size, size);
        _b_prior.conservativeResize(size);

        _b_prior.tail(v->local_dimension()).setZero();
        _h_prior.rightCols(v->local_dimension()).setZero();
        _h_prior.bottomRows(v->local_dimension()).setZero();
    }

    void Problem::extend_prior_hessian_size(ulong dim) {
        ulong size = _h_prior.rows() + dim;
        _h_prior.conservativeResize(size, size);
        _b_prior.conservativeResize(size);

        _b_prior.tail(dim).setZero();
        _h_prior.rightCols(dim).setZero();
        _h_prior.bottomRows(dim).setZero();
    }

    bool Problem::check_ordering() {
        if (_problem_type == ProblemType::SLAM_PROBLEM) {
            unsigned long current_ordering = 0;
            for (const auto& v: _idx_pose_vertices) {
                assert(v.second->ordering_id() == current_ordering);
                current_ordering += v.second->local_dimension();
            }

            for (const auto& v: _idx_landmark_vertices) {
                assert(v.second->ordering_id() == current_ordering);
                current_ordering += v.second->local_dimension();
            }
        }
        return true;
    }

    std::vector<std::shared_ptr<Edge>> Problem::get_connected_edges(const std::shared_ptr<Vertex>& vertex) {
        vector<shared_ptr<Edge>> edges;
        auto range = _vertex_to_edge.equal_range(vertex->id());
        for (auto iter = range.first; iter != range.second; ++iter) {

            // 并且这个edge还需要存在，而不是已经被remove了
            if (_edges.find(iter->second->id()) == _edges.end())
                continue;

            edges.emplace_back(iter->second);
        }
        return edges;
    }

    /*
     * marginalize 所有和 frame 相连的 edge: imu factor, projection factor
     * */
    bool Problem::marginalize(const std::shared_ptr<Vertex>& vertex_pose, const std::shared_ptr<Vertex>& vertex_motion) {
        // 重新计算一篇ordering
        set_ordering();
        ulong state_dim = _ordering_poses;

        // 所需被marginalize的edge
        std::vector<shared_ptr<Edge>> marginalized_edges = get_connected_edges(vertex_pose);

        // 所需被marginalize的landmark
        ulong marginalized_landmark_size = 0;
        std::unordered_map<unsigned long, shared_ptr<Vertex>> marginalized_landmark;  // O(1)查找
        for (auto &edge : marginalized_edges) {
            auto vertices_edge = edge->vertices();
            for (auto &vertex : vertices_edge) {
                if (is_landmark_vertex(vertex)
                    && marginalized_landmark.find(vertex->id()) == marginalized_landmark.end()) {
                    // 修改landmark的ordering_id, 方便hessian的计算
                    vertex->set_ordering_id(state_dim + marginalized_landmark_size);
                    marginalized_landmark.insert(make_pair(vertex->id(), vertex));
                    marginalized_landmark_size += vertex->local_dimension();
                }
            }
        }

        // 计算所需marginalize的edge的hessian
        ulong cols = state_dim + marginalized_landmark_size;
        MatXX h_state_landmark(MatXX::Zero(cols, cols));
        VecX b_state_landmark(VecX::Zero(cols));
        for (auto &edge : marginalized_edges) {
            edge->compute_residual();
            edge->compute_jacobians();
            auto &&jacobians = edge->jacobians();
            auto &&vertices = edge->vertices();

            assert(jacobians.size() == vertices.size());
            for (size_t i = 0; i < vertices.size(); ++i) {
                auto v_i = vertices[i];
                auto jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                double drho;
                MatXX robust_information(edge->information().rows(), edge->information().cols());
                edge->robust_information(drho, robust_information);

                for (size_t j = i; j < vertices.size(); ++j) {
                    auto v_j = vertices[j];
                    auto jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    MatXX hessian = jacobian_i.transpose() * robust_information * jacobian_j;

                    assert(hessian.rows() == v_i->local_dimension() && hessian.cols() == v_j->local_dimension());
                    // 所有的信息矩阵叠加起来
                    h_state_landmark.block(index_i, index_j, dim_i, dim_j) += hessian;
                    if (j != i) {
                        // 对称的下三角
                        h_state_landmark.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
                    }
                }
                b_state_landmark.segment(index_i, dim_i) -= drho * jacobian_i.transpose() * edge->information() * edge->residual();
            }
        }

        // marginalize与边连接的landmark
        MatXX h_state_schur;
        VecX b_state_schur;
        if (marginalized_landmark_size > 0) {
//            MatXX Hss = h_state_landmark.block(0, 0, state_dim, state_dim);
            MatXX Hll = h_state_landmark.block(state_dim, state_dim, marginalized_landmark_size, marginalized_landmark_size);
            MatXX Hsl = h_state_landmark.block(0, state_dim, state_dim, marginalized_landmark_size);
//            MatXX Hlp = h_state_landmark.block(state_dim, 0, marginalized_landmark_size, state_dim);
            VecX bss = b_state_landmark.segment(0, state_dim);
            VecX bll = b_state_landmark.segment(state_dim, marginalized_landmark_size);

            MatXX temp_H(MatXX::Zero(marginalized_landmark_size, state_dim));  // Hll^-1 * Hsl^T
            VecX temp_b(VecX::Zero(marginalized_landmark_size, 1));   // Hll^-1 * bl
            for (const auto& landmark_vertex : marginalized_landmark) {
                ulong idx = landmark_vertex.second->ordering_id() - state_dim;
                ulong size = landmark_vertex.second->local_dimension();
                if (size == 1) {
                    temp_H.row(idx) = Hsl.col(idx) / Hll(idx, idx);
                    temp_b(idx) = bll(idx) / Hll(idx, idx);
                } else {
                    auto Hmm_LUP = Hll.block(idx, idx, size, size).fullPivLu();
                    temp_H.block(idx, 0, size, state_dim) = Hmm_LUP.solve(Hsl.block(0, idx, state_dim, size).transpose());
                    temp_b.segment(idx, size) = Hmm_LUP.solve(bll.segment(idx, size));
                }
            }

            // (Hpp - Hsl * Hll^-1 * Hlp) * dxp = bp - Hsl * Hll^-1 * bl
            h_state_schur = h_state_landmark.block(0, 0, state_dim, state_dim) - Hsl * temp_H;
            b_state_schur = bss - Hsl * temp_b;
        }

        // 叠加之前的先验
        if(_h_prior.rows() > 0) {
            h_state_schur += _h_prior;
            b_state_schur += _b_prior;
        }

        // 把需要marginalize的pose和motion的vertices移动到最下面
        ulong marginalized_state_dim = 0;
        auto move_vertex_to_bottom = [&](const std::shared_ptr<Vertex>& vertex) {
            ulong idx = vertex->ordering_id();
            ulong dim = vertex->local_dimension();
            marginalized_state_dim += dim;

            // 将 row i 移动矩阵最下面
            Eigen::MatrixXd temp_rows = h_state_schur.block(idx, 0, dim, state_dim);
            Eigen::MatrixXd temp_botRows = h_state_schur.block(idx + dim, 0, state_dim - idx - dim, state_dim);
            h_state_schur.block(idx, 0, state_dim - idx - dim, state_dim) = temp_botRows;
            h_state_schur.block(state_dim - dim, 0, dim, state_dim) = temp_rows;

            // 将 col i 移动矩阵最右边
            Eigen::MatrixXd temp_cols = h_state_schur.block(0, idx, state_dim, dim);
            Eigen::MatrixXd temp_rightCols = h_state_schur.block(0, idx + dim, state_dim, state_dim - idx - dim);
            h_state_schur.block(0, idx, state_dim, state_dim - idx - dim) = temp_rightCols;
            h_state_schur.block(0, state_dim - dim, state_dim, dim) = temp_cols;

            Eigen::VectorXd temp_b = b_state_schur.segment(idx, dim);
            Eigen::VectorXd temp_btail = b_state_schur.segment(idx + dim, state_dim - idx - dim);
            b_state_schur.segment(idx, state_dim - idx - dim) = temp_btail;
            b_state_schur.segment(state_dim - dim, dim) = temp_b;
        };
        if (vertex_motion) {
            move_vertex_to_bottom(vertex_motion);
        }
        move_vertex_to_bottom(vertex_pose);

        // marginalize与边相连的所有pose和motion顶点
        auto marginalize_bottom_vertex = [&](const std::shared_ptr<Vertex> &vertex) {
            ulong marginalized_size = vertex->local_dimension();
            ulong reserve_size = state_dim - marginalized_size;
//            MatXX Hrr = h_state_schur.block(0, 0, reserve_size, reserve_size);
            MatXX Hmm = h_state_schur.block(reserve_size, reserve_size, marginalized_size, marginalized_size);
            MatXX Hrm = h_state_schur.block(0, reserve_size, reserve_size, marginalized_size);
//            MatXX Hmr = h_state_schur.block(reserve_size, 0, marginalized_size, reserve_size);
            VecX brr = b_state_schur.segment(0, reserve_size);
            VecX bmm = b_state_schur.segment(reserve_size, marginalized_size);

            MatXX temp_H(MatXX::Zero(marginalized_size, reserve_size));  // Hmm^-1 * Hrm^T
            VecX temp_b(VecX::Zero(marginalized_size, 1));   // Hmm^-1 * bm
            ulong size = vertex->local_dimension();
            if (size == 1) {
                temp_H = Hrm.transpose() / Hmm(0, 0);
                temp_b = bmm / Hmm(0, 0);
            } else {
                auto Hmm_LUP = Hmm.fullPivLu();
                temp_H = Hmm_LUP.solve(Hrm.transpose());
                temp_b = Hmm_LUP.solve(bmm);
            }

            // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
            h_state_schur = h_state_schur.block(0, 0, reserve_size, reserve_size) - Hrm * temp_H;
            b_state_schur = brr - Hrm * temp_b;

            state_dim = reserve_size;
        };
        marginalize_bottom_vertex(vertex_pose);
        if (vertex_motion) {
            marginalize_bottom_vertex(vertex_motion);
        }

        _h_prior = h_state_schur;
        _b_prior = b_state_schur;

        // 移除顶点
        remove_vertex(vertex_pose);
        if (vertex_motion) {
            remove_vertex(vertex_motion);
        }

        // 移除路标
        for (auto &landmark : marginalized_landmark) {
            remove_vertex(landmark.second);
        }

        return true;
    }

    void Problem::logout_vector_size() {
        // LOG(INFO) <<l
        //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
        //           " edges:" << edges_.size();
    }

    void Problem::test_marginalize() {
        // Add marg test
        int idx = 1;            // marg 中间那个变量
        int dim = 1;            // marg 变量的维度
        int reserve_size = 3;   // 总共变量的维度
        double delta1 = 0.1 * 0.1;
        double delta2 = 0.2 * 0.2;
        double delta3 = 0.3 * 0.3;

        int cols = 3;
        MatXX H_marg(MatXX::Zero(cols, cols));
        H_marg << 1./delta1, -1./delta1, 0,
                -1./delta1, 1./delta1 + 1./delta2 + 1./delta3, -1./delta3,
                0.,  -1./delta3, 1/delta3;
        std::cout << "---------- TEST Marg: before marg------------"<< std::endl;
        std::cout << H_marg << std::endl;

        // TODO:: home work. 将变量移动到右下角
        /// 准备工作： move the marg pose to the Hmm bottown right
        // 将 row i 移动矩阵最下面
        Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
        Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
        H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
        H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

        // 将 col i 移动矩阵最右边
        Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
        Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
        H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
        H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

        std::cout << "---------- TEST Marg: remove to right bottom ------------"<< std::endl;
        std::cout<< H_marg <<std::endl;

        /// 开始 marg ： schur
        double eps = 1e-8;
        int m2 = dim;
        int n2 = reserve_size - dim;   // 剩余变量的维度
        Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
        Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
                (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
                                  saes.eigenvectors().transpose();

        // TODO:: home work. 完成舒尔补操作
        Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);
        Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
        Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);

        Eigen::MatrixXd tempB = Arm * Amm_inv;
        Eigen::MatrixXd H_prior = Arr - tempB * Amr;

        std::cout << "---------- TEST Marg: after marg------------"<< std::endl;
        std::cout << H_prior << std::endl;
    }
}