//
// Created by Cain on 2024/3/7.
//

#include "problem_slam.h"
#include "tic_toc/tic_toc.h"
#include <iostream>

// #define USE_PCG_SOLVER


namespace graph_optimization {
    using namespace std;

    /*
        * marginalize 所有和 frame 相连的 edge: imu factor, projection factor
        * */
    bool ProblemSLAM::marginalize(const std::shared_ptr<Vertex>& vertex_pose, const std::shared_ptr<Vertex>& vertex_motion) {
//        TicToc t1;
        // 重新计算一篇ordering
//        initialize_ordering();
        ulong state_dim = _ordering_poses;
//        std::cout << "state_dim = " << state_dim << std::endl;

        // 所需被marginalize的edge
        auto &&marginalized_edges = get_connected_edges(vertex_pose);

        // 所需被marginalize的landmark
        ulong marginalized_landmark_size = 0;
        std::unordered_map<unsigned long, shared_ptr<Vertex>> marginalized_landmark;  // O(1)查找
        for (auto &edge : marginalized_edges) {
            auto vertices_edge = edge->vertices();
            for (auto &vertex : vertices_edge) {
                // TODO: 改为_vertices.find(vertex->id()) == _vertices.end()
                if (vertex->is_ordering_id_invalid()) continue;
                if (is_landmark_vertex(vertex)
                    && marginalized_landmark.find(vertex->id()) == marginalized_landmark.end()) {
                    // 修改landmark的ordering_id, 方便hessian的计算
                    vertex->set_ordering_id(state_dim + marginalized_landmark_size);
                    marginalized_landmark.insert(make_pair(vertex->id(), vertex));
                    marginalized_landmark_size += vertex->local_dimension();
                }
            }
        }

#ifdef USE_OPENMP
        std::vector<std::pair<unsigned long, shared_ptr<Vertex>>> marginalized_landmark_vector;
        marginalized_landmark_vector.reserve(marginalized_landmark.size());
        for (auto &landmark : marginalized_landmark) {
            marginalized_landmark_vector.emplace_back(landmark);
        }
#endif

//        double ms1 = t1.toc();
//        std::cout << "t1 = " << ms1 << std::endl;
////        std::cout << "marginalized_edges.size() = " << marginalized_edges.size() << ", ";
////        std::cout << "marginalized_landmark.size() = " << marginalized_landmark.size() << ", ";
////        std::cout << "marginalized_landmark_dim = " << marginalized_landmark_size << std::endl;
//
//        TicToc t2;

        // 计算所需marginalize的edge的hessian
        ulong cols = state_dim + marginalized_landmark_size;
        MatXX h_state_landmark(MatXX::Zero(cols, cols));
        VecX b_state_landmark(VecX::Zero(cols));
//        std::cout << "marginalized_landmark_size = " << marginalized_landmark_size << std::endl;

#ifdef USE_OPENMP
        MatXX Hs[NUM_THREADS];       ///< Hessian矩阵
        VecX bs[NUM_THREADS];       ///< 负梯度
        for (unsigned int i = 0; i < NUM_THREADS; ++i) {
            Hs[i] = MatXX::Zero(cols, cols);
            bs[i] = VecX::Zero(cols);
        }

#pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t n = 0; n < marginalized_edges.size(); ++n) {
            unsigned int index = omp_get_thread_num();

            auto &&edge = marginalized_edges[n];
            auto &&jacobians = edge->jacobians();
            auto &&vertices = edge->vertices();

            assert(jacobians.size() == vertices.size());
            for (size_t i = 0; i < vertices.size(); ++i) {
                auto v_i = vertices[i];
                if (v_i->is_fixed()) continue;
                if (v_i->is_ordering_id_invalid()) continue;

                auto &&jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                double drho;
                MatXX robust_information(edge->information().rows(), edge->information().cols());
                edge->robust_information(drho, robust_information);

                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < vertices.size(); ++j) {
                    auto &&v_j = vertices[j];
                    if (v_j->is_fixed()) continue;
                    if (v_j->is_ordering_id_invalid()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    MatXX hessian = JtW * jacobian_j;

                    // 所有的信息矩阵叠加起来
                    Hs[index].block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        Hs[index].block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                bs[index].segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose() * edge->information() * edge->residual();
            }
        }

        for (unsigned int i = 0; i < NUM_THREADS; ++i) {
            h_state_landmark += Hs[i];
            b_state_landmark += bs[i];
        }
#else
        for (auto &edge : marginalized_edges) {
            // 若曾经solve problem, 则无需再次计算
            // edge->compute_residual();
            // edge->compute_jacobians();
            auto &&jacobians = edge->jacobians();
            auto &&vertices = edge->vertices();

            assert(jacobians.size() == vertices.size());
            for (size_t i = 0; i < vertices.size(); ++i) {
                auto v_i = vertices[i];
                if (v_i->is_fixed()) continue;
                if (v_i->is_ordering_id_invalid()) continue;

                auto jacobian_i = jacobians[i];
                ulong index_i = v_i->ordering_id();
                ulong dim_i = v_i->local_dimension();

                double drho;
                MatXX robust_information(edge->information().rows(), edge->information().cols());
                edge->robust_information(drho, robust_information);
                MatXX JtW = jacobian_i.transpose() * robust_information;
                for (size_t j = i; j < vertices.size(); ++j) {
                    auto &&v_j = vertices[j];
                    if (v_j->is_fixed()) continue;
                    if (v_j->is_ordering_id_invalid()) continue;

                    auto &&jacobian_j = jacobians[j];
                    ulong index_j = v_j->ordering_id();
                    ulong dim_j = v_j->local_dimension();

                    MatXX hessian = JtW * jacobian_j;

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
#endif

//        double ms2 = t2.toc();
//        std::cout << "t2 = " << ms2 << std::endl;
//
//        TicToc t3;

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
#ifdef USE_OPENMP
            for (size_t n = 0; n < marginalized_landmark_vector.size(); ++n) {
                auto &&landmark_vertex = marginalized_landmark_vector[n];
                ulong idx = landmark_vertex.second->ordering_id() - state_dim;
                ulong size = landmark_vertex.second->local_dimension();
                if (size == 1) {
                    if (Hll(idx, idx) > 1e-12) {
                        temp_H.row(idx).noalias() = Hsl.col(idx) / Hll(idx, idx);
                        temp_b(idx) = bll(idx) / Hll(idx, idx);
                    } else {
                        temp_H.row(idx).setZero();
                        temp_b(idx) = 0.;
                    }
                } else {
                    auto Hmm_ldlt = Hll.block(idx, idx, size, size).ldlt();
                    if (Hmm_ldlt.info() == Eigen::Success) {
                        temp_H.block(idx, 0, size, state_dim).noalias() = Hmm_ldlt.solve(Hsl.block(0, idx, state_dim, size).transpose());
                        temp_b.segment(idx, size).noalias() = Hmm_ldlt.solve(bll.segment(idx, size));
                    } else {
                        temp_H.block(idx, 0, size, state_dim).setZero();
                        temp_b.segment(idx, size).setZero();
                    }
                }
            }
#else
            for (const auto& landmark_vertex : marginalized_landmark) {
                ulong idx = landmark_vertex.second->ordering_id() - state_dim;
                ulong size = landmark_vertex.second->local_dimension();
                if (size == 1) {
                    if (Hll(idx, idx) > 1e-12) {
                        temp_H.row(idx) = Hsl.col(idx) / Hll(idx, idx);
                        temp_b(idx) = bll(idx) / Hll(idx, idx);
                    } else {
                        temp_H.row(idx).setZero();
                        temp_b(idx) = 0.;
                    }
                } else {
                    auto Hmm_ldlt = Hll.block(idx, idx, size, size).ldlt();
                    if (Hmm_ldlt.info() == Eigen::Success) {
                        temp_H.block(idx, 0, size, state_dim) = Hmm_ldlt.solve(Hsl.block(0, idx, state_dim, size).transpose());
                        temp_b.segment(idx, size) = Hmm_ldlt.solve(bll.segment(idx, size));
                    } else {
                        temp_H.block(idx, 0, size, state_dim).setZero();
                        temp_b.segment(idx, size).setZero();
                    }
                }
            }
#endif

            // (Hpp - Hsl * Hll^-1 * Hlp) * dxp = bp - Hsl * Hll^-1 * bl
#ifdef USE_OPENMP
            h_state_schur = MatXX::Zero(state_dim, state_dim);
#pragma omp parallel for num_threads(NUM_THREADS)
            for (ulong i = 0; i < state_dim; ++i) {
                h_state_schur(i, i) = -Hsl.row(i).dot(temp_H.col(i));
                for (ulong j = i + 1; j < state_dim; ++j) {
                    h_state_schur(i, j) = -Hsl.row(i).dot(temp_H.col(j));
                    h_state_schur(j, i) = h_state_schur(i, j);
                }
            }
            h_state_schur += h_state_landmark.block(0, 0, state_dim, state_dim);
#else

            h_state_schur = h_state_landmark.block(0, 0, state_dim, state_dim) - Hsl * temp_H;
            // for (ulong i = 0; i < state_dim; ++i) {
            //     h_state_schur(i, i) -= Hsl.row(i).dot(temp_H.col(i));
            //     for (ulong j = i + 1; j < state_dim; ++j) {
            //         h_state_schur(i, j) -= Hsl.row(i).dot(temp_H.col(j));
            //         h_state_schur(j, i) = h_state_schur(i, j);
            //     }
            // }
#endif
            b_state_schur = bss - Hsl * temp_b;
        } else {
            h_state_schur = h_state_landmark;
            b_state_schur = b_state_landmark;
        }

//        double ms3 = t3.toc();
//        std::cout << "t3 = " << ms3 << std::endl;
//
//        TicToc t4;

        // 叠加之前的先验
        if(_h_prior.rows() > 0) {
//            std::cout << "h_state_schur.size: " << h_state_schur.rows() << ", " << h_state_schur.cols() << std::endl;
//            std::cout << "_h_prior.size: " << _h_prior.rows() << ", " << _h_prior.cols() << std::endl;
//            std::cout << "b_state_schur.size: " << b_state_schur.rows() << std::endl;
//            std::cout << "_b_prior.size: " << _b_prior.rows() << std::endl;
            h_state_schur += _h_prior;
            b_state_schur += _b_prior;
        }

        // 把需要marginalize的pose和motion的vertices移动到最下面
        ulong marginalized_state_dim = 0;
        auto move_vertex_to_bottom = [&](const std::shared_ptr<Vertex>& vertex) {
            if (vertex->is_ordering_id_invalid()) { // TODO: 改为_vertices.find(vertex->id()) == _vertices.end()
                return;
            }
            ulong idx = vertex->ordering_id();
            ulong dim = vertex->local_dimension();
            marginalized_state_dim += dim;

            // 将 row i 移动矩阵最下面
            Eigen::MatrixXd temp_rows = h_state_schur.block(idx, 0, dim, state_dim);
            Eigen::MatrixXd temp_bot_rows = h_state_schur.block(idx + dim, 0, state_dim - idx - dim, state_dim);
            h_state_schur.block(idx, 0, state_dim - idx - dim, state_dim) = temp_bot_rows;
            h_state_schur.block(state_dim - dim, 0, dim, state_dim) = temp_rows;

            // 将 col i 移动矩阵最右边
            Eigen::MatrixXd temp_cols = h_state_schur.block(0, idx, state_dim, dim);
            Eigen::MatrixXd temp_right_cols = h_state_schur.block(0, idx + dim, state_dim, state_dim - idx - dim);
            h_state_schur.block(0, idx, state_dim, state_dim - idx - dim) = temp_right_cols;
            h_state_schur.block(0, state_dim - dim, state_dim, dim) = temp_cols;

            Eigen::VectorXd temp_b = b_state_schur.segment(idx, dim);
            Eigen::VectorXd temp_b_tail = b_state_schur.segment(idx + dim, state_dim - idx - dim);
            b_state_schur.segment(idx, state_dim - idx - dim) = temp_b_tail;
            b_state_schur.segment(state_dim - dim, dim) = temp_b;
        };
        if (vertex_motion) {
            move_vertex_to_bottom(vertex_motion);
        }
        move_vertex_to_bottom(vertex_pose);

//        double ms4 = t4.toc();
//        std::cout << "t4 = " << ms4 << std::endl;
//
//        TicToc t5;

        // marginalize与边相连的所有pose和motion顶点
        auto marginalize_bottom_vertex = [&](const std::shared_ptr<Vertex> &vertex) {
            if (vertex->is_ordering_id_invalid()) { // TODO: 改为_vertices.find(vertex->id()) == _vertices.end()
                return;
            }
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
                if (Hmm(0, 0) > 1e-12) {
                    temp_H = Hrm.transpose() / Hmm(0, 0);
                    temp_b = bmm / Hmm(0, 0);
                } else {
                    temp_H.setZero();
                    temp_b.setZero();
                }

            } else {
                auto Hmm_ldlt = Hmm.ldlt();
                if (Hmm_ldlt.info() == Eigen::Success) {
                    temp_H = Hmm_ldlt.solve(Hrm.transpose());
                    temp_b = Hmm_ldlt.solve(bmm);
                } else {
                    temp_H.setZero();
                    temp_b.setZero();
                }
            }

            // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
#ifdef USE_OPENMP
            MatXX h_state_schur_block = h_state_schur.block(0, 0, reserve_size, reserve_size);
            h_state_schur = MatXX::Zero(reserve_size, reserve_size);
#pragma omp parallel for num_threads(NUM_THREADS)
            for (ulong i = 0; i < reserve_size; ++i) {
                h_state_schur(i, i) = -Hrm.row(i).dot(temp_H.col(i));
                for (ulong j = i + 1; j < reserve_size; ++j) {
                    h_state_schur(i, j) = -Hrm.row(i).dot(temp_H.col(j));
                    h_state_schur(j, i) = h_state_schur(i, j);
                }
            }
            h_state_schur += h_state_schur_block;
#else

            // (Hrr - Hrm * Hmm^-1 * Hmr) * dxp = br - Hrm * Hmm^-1 * bm
            h_state_schur = h_state_schur.block(0, 0, reserve_size, reserve_size) - Hrm * temp_H;
            // for (ulong i = 0; i < reserve_size; ++i) {
            //     h_state_schur(i, i) -= Hrm.row(i).dot(temp_H.col(i));
            //     for (ulong j = i + 1; j < reserve_size; ++j) {
            //         h_state_schur(i, j) -= Hrm.row(i).dot(temp_H.col(j));
            //         h_state_schur(j, i) = h_state_schur(i, j);
            //     }
            // }
#endif
            b_state_schur = brr - Hrm * temp_b;

            state_dim = reserve_size;
        };
        marginalize_bottom_vertex(vertex_pose);
        if (vertex_motion) {
            marginalize_bottom_vertex(vertex_motion);
        }

        _h_prior = h_state_schur;
        _b_prior = b_state_schur;

//        double ms5 = t5.toc();
//        std::cout << "t5 = " << ms5 << std::endl;
//
//        TicToc t6;

        // // 移除顶点
        // remove_vertex(vertex_pose);
        // if (vertex_motion) {
        //     remove_vertex(vertex_motion);
        // }

//        double ms6 = t6.toc();
//        std::cout << "t6 = " << ms6 << std::endl;

        /*
         * 不能在这里移除路标点，因为路标在别的frame中，可能能重新构建重投影误差
         * */
//        // 移除路标
//        for (auto &landmark : marginalized_landmark) {
//            remove_vertex(landmark.second);
//        }

        return true;
    }

    VecX ProblemSLAM::multiply_hessian(const VecX &x) {
        VecX v(VecX::Zero(x.rows(), x.cols()));
        for (unsigned long i = 0; i < _ordering_poses; i++) {
            v(i) += _hessian(i, i) * x(i);  // 计算对角线部分
            for (unsigned long j = i + 1; j < _ordering_generic; j++) { // 计算非对角线部分
                v(i) += _hessian(i, j) * x(j);  // 上三角部分
                v(j) += _hessian(i, j) * x(i);  // 下三角部分
            }
        }
        for (unsigned long i = _ordering_poses; i < _ordering_generic; ++i) {
            v(i) += _hessian(i, i) * x(i);
        }
        return v;
    }

    bool ProblemSLAM::solve_linear_system(VecX &delta_x) {
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

        TicToc t_schur;
        // 由于叠加了lambda, 所以能够保证Hll可逆
        MatXX Hll = _hessian.block(reserve_size, reserve_size, marg_size, marg_size);
        for (ulong i = 0; i < marg_size; ++i) {   // LM Method
            // Hll(i, i) += _current_lambda;
            Hll(i, i) += _diag_lambda(i + reserve_size);
            if (Hll(i, i) < _diag_lambda(i + reserve_size)) {
                Hll(i, i) = _diag_lambda(i + reserve_size);
            }
        }

//        MatXX Hpl = _hessian.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hlp = _hessian.block(reserve_size, 0, marg_size, reserve_size);
//        VecX bpp = _b.segment(0, reserve_size);
        VecX bll = _b.segment(reserve_size, marg_size);
        MatXX temp_H(MatXX::Zero(marg_size, reserve_size));  // Hll^-1 * Hpl^T
        VecX temp_b(VecX::Zero(marg_size, 1));   // Hll^-1 * bl
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t n = 0; n < _idx_landmark_vertices.size(); ++n) {
            ulong idx = _idx_landmark_vertices[n].second->ordering_id() - reserve_size;
            ulong size = _idx_landmark_vertices[n].second->local_dimension();
            if (size == 1) {
                temp_H.row(idx) = Hlp.row(idx) / Hll(idx, idx);
                temp_b(idx) = bll(idx) / Hll(idx, idx);
            } else {
                auto &&Hll_ldlt = Hll.block(idx, idx, size, size).ldlt();
                if (Hll_ldlt.info() == Eigen::Success) {
                    temp_H.block(idx, 0, size, reserve_size) = Hll_ldlt.solve(Hlp.block(idx, 0, size, reserve_size));
                    temp_b.segment(idx, size) = Hll_ldlt.solve(bll.segment(idx, size));
                }
            }
        }
#else
        for (const auto& landmark_vertex : _idx_landmark_vertices) {
            ulong idx = landmark_vertex.second->ordering_id() - reserve_size;
            ulong size = landmark_vertex.second->local_dimension();
            if (size == 1) {
                temp_H.row(idx) = Hlp.row(idx) / Hll(idx, idx);
                temp_b(idx) = bll(idx) / Hll(idx, idx);
            } else {
                auto &&Hll_ldlt = Hll.block(idx, idx, size, size).ldlt();
                if (Hll_ldlt.info() != Eigen::Success) {
                    return false;
                }
                temp_H.block(idx, 0, size, reserve_size) = Hll_ldlt.solve(Hlp.block(idx, 0, size, reserve_size));
                temp_b.segment(idx, size) = Hll_ldlt.solve(bll.segment(idx, size));
            }
        }
#endif

//        std::cout << "_ordering_poses = " << _ordering_poses << std::endl;
//        std::cout << "_ordering_landmarks = " << _ordering_landmarks << std::endl;
//        std::cout << "Hlp: " << "(" <<  Hlp.rows() << ", " << Hlp.cols() << ")" << std::endl;

        // (Hpp - Hpl * Hll^-1 * Hlp) * dxp = bp - Hpl * Hll^-1 * bl
        // 这里即使叠加了lambda, 也有可能因为数值精度的问题而导致 _h_pp_schur 不可逆
#ifdef USE_OPENMP
        _h_pp_schur = MatXX::Zero(reserve_size, reserve_size);
#pragma omp parallel for num_threads(NUM_THREADS)
        for (ulong i = 0; i < reserve_size; ++i) {
            _h_pp_schur(i, i) = -Hlp.col(i).dot(temp_H.col(i));
            for (ulong j = i + 1; j < reserve_size; ++j) {
                _h_pp_schur(i, j) = -Hlp.col(i).dot(temp_H.col(j));
                _h_pp_schur(j, i) = _h_pp_schur(i, j);
            }
        }
        _h_pp_schur += _hessian.block(0, 0, reserve_size, reserve_size);
#else
        _h_pp_schur = _hessian.block(0, 0, reserve_size, reserve_size);// - Hlp.transpose() * temp_H;
        for (ulong i = 0; i < reserve_size; ++i) {
            _h_pp_schur(i, i) -= Hlp.col(i).dot(temp_H.col(i));
            for (ulong j = i + 1; j < reserve_size; ++j) {
                _h_pp_schur(i, j) -= Hlp.col(i).dot(temp_H.col(j));
                _h_pp_schur(j, i) = _h_pp_schur(i, j);
            }
        }
#endif
        _t_schur_cost += t_schur.toc();
        for (ulong i = 0; i < _ordering_poses; ++i) {
            // _h_pp_schur(i, i) += _current_lambda;    // LM Method
            _h_pp_schur(i, i) += _diag_lambda(i);
        }
        _b_pp_schur = _b.segment(0, reserve_size) - Hlp.transpose() * temp_b;

        // Solve: Hpp * delta_x = bpp
        TicToc t_ldlt;
        VecX delta_x_pp(VecX::Zero(reserve_size));
#ifdef USE_PCG_SOLVER
        auto n_pcg = _h_pp_schur.rows();                       // 迭代次数
        delta_x_pp = PCG_solver(_h_pp_schur, _b_pp_schur, n_pcg);
#else
        auto &&H_pp_schur_ldlt = _h_pp_schur.ldlt();
        if (H_pp_schur_ldlt.info() != Eigen::Success) {
            return false;   // H_pp_schur不是正定矩阵
        }
        delta_x_pp =  H_pp_schur_ldlt.solve(_b_pp_schur);
#endif
        delta_x.head(reserve_size) = delta_x_pp;

        // Hll * dxl = bl - Hlp * dxp
        VecX delta_x_ll(marg_size);
        delta_x_ll = temp_b - temp_H * delta_x_pp;
        delta_x.tail(marg_size) = delta_x_ll;
        _t_ldlt_cost += t_ldlt.toc();

        return true;
    }

    void ProblemSLAM::add_prior_to_hessian() {
        if(_h_prior.rows() > 0) {
            MatXX H_prior_tmp = _h_prior;
            VecX b_prior_tmp = _b_prior;

            // 只有没有被fix的pose存在先验, landmark没有先验
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
            for (size_t n = 0; n < _idx_pose_vertices.size(); ++n) {
                auto &&vertex = _idx_pose_vertices[n].second;
                if (vertex->is_fixed() ) {
                    ulong idx = vertex->ordering_id();
                    ulong dim = vertex->local_dimension();
                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
#else
            for (const auto& vertex: _idx_pose_vertices) {
                if (is_pose_vertex(vertex.second) && vertex.second->is_fixed() ) {
                    ulong idx = vertex.second->ordering_id();
                    ulong dim = vertex.second->local_dimension();
                    H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                    H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                    b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
                }
            }
#endif
            _hessian.topLeftCorner(_ordering_poses, _ordering_poses) += H_prior_tmp;
            _b.head(_ordering_poses) += b_prior_tmp;
        }
    }

    void ProblemSLAM::initialize_ordering() {
        _ordering_generic = 0;

        // 分配pose的维度
        _ordering_poses = 0;
        _idx_pose_vertices.clear();
        for (auto &vertex: _vertices) {
            if (is_pose_vertex(vertex.second)) {
                vertex.second->set_ordering_id(_ordering_poses);
                _idx_pose_vertices.emplace_back(vertex.second->id(), vertex.second);
//                _idx_pose_vertices.insert(pair<ulong, std::shared_ptr<Vertex>>(vertex.second->id(), vertex.second));
                _ordering_poses += vertex.second->local_dimension();
            }
        }

        // 分配landmark的维度
        _ordering_landmarks = 0;
        _idx_landmark_vertices.clear();
        for (auto &vertex: _vertices) {
            if (is_landmark_vertex(vertex.second)) {
                vertex.second->set_ordering_id(_ordering_landmarks + _ordering_poses);
                _idx_landmark_vertices.emplace_back(vertex.second->id(), vertex.second);
//                _idx_landmark_vertices.insert(pair<ulong, std::shared_ptr<Vertex>>(vertex.second->id(), vertex.second));
                _ordering_landmarks += vertex.second->local_dimension();
            }
        }

        _ordering_generic = _ordering_poses + _ordering_landmarks;
    }

    bool ProblemSLAM::add_vertex(const std::shared_ptr<Vertex>& vertex) {
        if (Problem::add_vertex(vertex)) {
            if (is_pose_vertex(vertex)) {
                resize_pose_hessian_when_adding_pose(vertex);
            }
            return true;
        }

        return false;
    }

    bool ProblemSLAM::remove_vertex(const std::shared_ptr<Vertex>& vertex) {
        if (Problem::remove_vertex(vertex)) {
//            if (is_pose_vertex(vertex)) {
//                _idx_pose_vertices.erase(vertex->id());
//            }
//            else {
//                _idx_landmark_vertices.erase(vertex->id());
//            }
            return true;
        }

        return false;
    }

    bool ProblemSLAM::is_pose_vertex(const std::shared_ptr<Vertex>& v) {
        string type = v->type_info();
        return type == string("VertexPose") || type == string("VertexMotion");
    }

    bool ProblemSLAM::is_landmark_vertex(const std::shared_ptr<Vertex>& v) {
        string type = v->type_info();
        return type == string("VertexPoint3d") || type == string("VertexInverseDepth");
    }

    void ProblemSLAM::update_prior(const VecX &delta_x) {
        if (_b_prior.rows() > 0) {
            _b_prior_bp = _b_prior;
            _b_prior -= _h_prior * delta_x.head(_ordering_poses);
        }
    }

    bool ProblemSLAM::check_ordering() {
        unsigned long current_ordering = 0;
        for (const auto& v: _idx_pose_vertices) {
            if (v.second->ordering_id() != current_ordering) {
                return false;
            }
            current_ordering += v.second->local_dimension();
        }

        for (const auto& v: _idx_landmark_vertices) {
            if (v.second->ordering_id() != current_ordering) {
                return false;
            }
            current_ordering += v.second->local_dimension();
        }

        return true;
    }

    void ProblemSLAM::resize_pose_hessian_when_adding_pose(const std::shared_ptr<Vertex>& v) {
        unsigned size = _h_prior.rows() + v->local_dimension();
        _h_prior.conservativeResize(size, size);
        _b_prior.conservativeResize(size);

        _b_prior.tail(v->local_dimension()).setZero();
        _h_prior.rightCols(v->local_dimension()).setZero();
        _h_prior.bottomRows(v->local_dimension()).setZero();
    }
}