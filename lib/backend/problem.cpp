//
// Created by Cain on 2023/12/28.
//

#include <iostream>
#include <fstream>

//#include <glog/logging.h>
#include <Eigen/Dense>
#include <lib/tic_toc/tic_toc.h>
#include "problem.h"

#define MULTIPLY_HESSIAN_USING_SELF_ADJOINT

using namespace std;

namespace graph_optimization {
    Problem::Problem() {
        logout_vector_size();
        _vertices_marg.clear();
    }

    bool Problem::add_vertex(const std::shared_ptr<Vertex>& vertex) {
        if (_vertices.find(vertex->id()) != _vertices.end()) {
            // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
            return false;
        }
        _vertices.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->id(), vertex));

        return true;
    }

    bool Problem::remove_vertex(const std::shared_ptr<Vertex>& vertex) {
        // 如果顶点不在顶点map中, 则返回false
        if (_vertices.find(vertex->id()) == _vertices.end()) {
            // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
            return false;
        }

        // 若删除顶点, 则需要把顶点所连接的edge也删除
        auto &&edges = get_connected_edges(vertex);
        for (auto & edge : edges) {
            remove_edge(edge);
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

    bool Problem::solve(unsigned long iterations) {
        if (_edges.empty() || _vertices.empty()) {
            std::cerr << "\nCannot solve problem without edges or vertices" << std::endl;
            return false;
        }

        TicToc t_solve;
        _t_hessian_cost = 0.;
        _t_jacobian_cost = 0.;
        _t_chi2_cost = 0;
        _t_residual_cost = 0.;

        initialize_ordering(); // 统计优化变量的维数: _ordering_generic，为构建 hessian 矩阵做准备

        bool flag;
        switch (_solver_type) {
            case SolverType::STEEPEST_DESCENT:
                flag = calculate_steepest_descent(_delta_x_sd, iterations);
                break;
            case SolverType::GAUSS_NEWTON:
                flag = calculate_gauss_newton(_delta_x_gn, iterations);
                break;
            case SolverType::LEVENBERG_MARQUARDT:
                flag = calculate_levenberg_marquardt(_delta_x_lm, iterations);
                break;
            case SolverType::DOG_LEG:
                flag = calculate_dog_leg(_delta_x_dl, iterations);
                break;
            default:
                flag = calculate_levenberg_marquardt(_delta_x_lm, iterations);
                break;
        }

        std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
        std::cout << "update_hessian cost: " << _t_hessian_cost << " ms" << std::endl;
        std::cout << "update_jacobian cost: " << _t_jacobian_cost << " ms" << std::endl;
        std::cout << "update_chi2 cost: " << _t_chi2_cost << " ms" << std::endl;
        std::cout << "update_residual cost: " << _t_residual_cost << " ms" << std::endl;

        return flag;
    }

    void Problem::initialize_ordering() {
        // 每次重新计数
        _ordering_generic = 0;

        // Note: _vertices 是 map 类型的, 顺序是按照 id 号排序的
        // 统计带估计的所有变量的总维度
        for (auto &vertex: _vertices) {
            vertex.second->set_ordering_id(_ordering_generic);
            _ordering_generic += vertex.second->local_dimension();  // 所有的优化变量总维数
        }
    }

    void Problem::update_hessian() {
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
//            edge.second->compute_residual();
//            edge.second->compute_jacobians();

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
                b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose() * edge.second->information() * edge.second->residual();
            }

        }
        _hessian = H;
        _b = b;

        // 叠加先验
        add_prior_to_hessian();

//        _delta_x = VecX::Zero(size);  // initial delta_x = 0_n;

        _t_hessian_cost += t_h.toc();
    }

    void Problem::add_prior_to_hessian() {
        if(_h_prior.rows() > 0) {
            MatXX H_prior_tmp = _h_prior;
            VecX b_prior_tmp = _b_prior;

            // 对所有没有被fix的顶点添加先验信息
            for (const auto& vertex: _vertices) {
                if (vertex.second->is_fixed()) {
                    ulong idx = vertex.second->ordering_id();
                    ulong dim = vertex.second->local_dimension();
                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
            _hessian += H_prior_tmp;
            _b += b_prior_tmp;
        }
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

    double Problem::calculate_hessian_norm_square(const VecX &x) {
#ifdef HESSIAN_NORM_SQUARE_USING_GRAPH
        double norm2 = 0.;

        // TODO:: accelate, accelate, accelate
//#ifdef USE_OPENMP
//#pragma omp parallel for
//#endif

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

                VecX Jx = jacobian_i * x.segment(index_i, dim_i);
                norm2 += Jx.transpose() * robust_information * Jx;
            }
        }

        return norm2;
#else
        return x.dot(multiply_hessian(x));
#endif
    }

    VecX Problem::multiply_hessian(const VecX &x) {
#ifdef MULTIPLY_HESSIAN_USING_SELF_ADJOINT
        auto &&hessian = _hessian.selfadjointView<Eigen::Upper>();
        return hessian * x;
#else
        return _hessian * x;
#endif
    }

    /*
    * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
    */
    bool Problem::solve_linear_system(VecX &delta_x) {
        MatXX H = _hessian;
//            for (unsigned i = 0; i < _hessian.rows(); ++i) {
//                H(i, i) += _current_lambda;
//            }
        H += _diag_lambda.asDiagonal();
        auto && H_ldlt = H.ldlt();
        if (H_ldlt.info() == Eigen::Success) {
            delta_x = H_ldlt.solve(_b);
            return true;
        } else {
            return false;
        }
    }

    void Problem::update_states(const VecX &delta_x) {
        for (auto &vertex: _vertices) {
            vertex.second->save_parameters();

            ulong idx = vertex.second->ordering_id();
            ulong dim = vertex.second->local_dimension();
            VecX delta = delta_x.segment(idx, dim);

            // 所有的参数 x 叠加一个增量  x_{k+1} = x_{k} + delta_x
            vertex.second->plus(delta);
        }

        // 利用 delta_x 更新先验信息
        update_prior(delta_x);
    }

    void Problem::update_prior(const VecX &delta_x) {
        if (_b_prior.rows() > 0) {
            _b_prior_bp = _b_prior;
            _b_prior -= _h_prior * delta_x;
        }
    }

    void Problem::update_residual() {
        TicToc t_r;

        for (auto &edge: _edges) {
            edge.second->compute_residual();
        }

        _t_residual_cost += t_r.toc();
    }

    void Problem::update_jacobian() {
        TicToc t_j;

        for (auto &edge: _edges) {
            edge.second->compute_jacobians();
        }

        _t_jacobian_cost += t_j.toc();
    }

    void Problem::update_chi2() {
        TicToc t_c;

        _chi2 = 0.;
        for (auto &edge: _edges) {
            edge.second->compute_chi2();
            _chi2 += edge.second->get_robust_chi2();
        }
        _chi2 *= 0.5;

        _t_chi2_cost += t_c.toc();
    }

    void Problem::rollback_states(const VecX &delta_x) {
        for (auto &vertex: _vertices) {
            ulong idx = vertex.second->ordering_id();
            ulong dim = vertex.second->local_dimension();
            VecX delta = delta_x.segment(idx, dim);

//            // 之前的增量加了后使得损失函数增加了，我们应该不要这次迭代结果，所以把之前加上的量减去。
//            vertex.second->plus(-delta);
            vertex.second->load_parameters();
        }

        _b_prior = _b_prior_bp;
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
    VecX Problem::PCG_solver(const MatXX &A, const VecX &b, unsigned long max_iter) {
        assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
        TicToc t_h;
        unsigned long rows = b.rows();
        unsigned long n = max_iter ? max_iter : rows;
        double threshold = 1e-6 * b.norm();

        VecX x(VecX::Zero(rows));
        VecX M = A.diagonal();
        VecX r(-b);
        VecX y(r.array() / M.array());
        VecX p(-y);
        VecX Ap = A * p;
        double ry = r.dot(y);
        double ry_prev = ry;
        double alpha = ry / p.dot(Ap);
        double beta;
        x += alpha * p;
        r += alpha * Ap;

        unsigned long i = 0;
        while (r.norm() > threshold && ++i < n) {
            y = r.array() / M.array();
            ry = r.dot(y);
            beta = ry / ry_prev;
            ry_prev = ry;
            p = -y + beta * p;

            Ap = A * p;
            alpha = ry / p.dot(Ap);
            x += alpha * p;
            r += alpha * Ap;
        }

        _t_PCG_solve_cost = t_h.toc();
        return x;
    }

//    void Problem::compute_prior() {
//
//    }


    void Problem::extend_prior_hessian_size(ulong dim) {
        ulong size = _h_prior.rows() + dim;
        _h_prior.conservativeResize(size, size);
        _b_prior.conservativeResize(size);

        _b_prior.tail(dim).setZero();
        _h_prior.rightCols(dim).setZero();
        _h_prior.bottomRows(dim).setZero();
    }

    bool Problem::check_ordering() {
        unsigned long current_ordering = 0;
        for (const auto &v : _vertices) {
            if (v.second->ordering_id() != current_ordering) {
                return false;
            }
            current_ordering += v.second->local_dimension();
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