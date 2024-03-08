//
// Created by Cain on 2024/3/7.
//

#include "problem_slam.h"

using namespace std;

namespace graph_optimization {
    namespace slam {
        /*
         * marginalize 所有和 frame 相连的 edge: imu factor, projection factor
         * */
        bool ProblemSLAM::marginalize(const std::shared_ptr<Vertex>& vertex_pose, const std::shared_ptr<Vertex>& vertex_motion) {
            // 重新计算一篇ordering
            initialize_ordering();
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

        bool ProblemSLAM::solve_linear_system(graph_optimization::VecX &delta_x) {
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

            // (Hpp - Hpl * Hll^-1 * Hlp) * dxp = bp - Hpl * Hll^-1 * bl
            // 这里即使叠加了lambda, 也有可能因为数值精度的问题而导致 _h_pp_schur 不可逆
            _h_pp_schur = _hessian.block(0, 0, reserve_size, reserve_size) - Hlp.transpose() * temp_H;
            for (ulong i = 0; i < _ordering_poses; ++i) {
                // _h_pp_schur(i, i) += _current_lambda;    // LM Method
                _h_pp_schur(i, i) += _diag_lambda(i);
            }
            _b_pp_schur = _b.segment(0, reserve_size) - Hlp.transpose() * temp_b;

            // Solve: Hpp * delta_x = bpp
            VecX delta_x_pp(VecX::Zero(reserve_size));
#ifdef USE_PCG_SOLVER
            auto n_pcg = _h_pp_schur.rows() * 2;                       // 迭代次数
        delta_x_pp = PCG_solver(_h_pp_schur, _b_pp_schur, n_pcg);
#else
            auto &&H_pp_schur_ldlt = _h_pp_schur.ldlt();
            if (H_pp_schur_ldlt.info() != Eigen::Success) {
                return false;   // H_pp_schur不是正定矩阵
            }
            delta_x_pp =  H_pp_schur_ldlt.solve(_b_pp_schur);
#endif
            _delta_x.head(reserve_size) = delta_x_pp;

            // Hll * dxl = bl - Hlp * dxp
            VecX delta_x_ll(marg_size);
            delta_x_ll = temp_b - temp_H * delta_x_pp;
            _delta_x.tail(marg_size) = delta_x_ll;

            return true;
        }

        void ProblemSLAM::add_prior_to_hessian() {
            if(_h_prior.rows() > 0) {
                MatXX H_prior_tmp = _h_prior;
                VecX b_prior_tmp = _b_prior;

                // 只有没有被fix的pose存在先验, landmark没有先验
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
        }

        void ProblemSLAM::initialize_ordering() {
            _ordering_generic = 0;

            // 分配pose的维度
            _ordering_poses = 0;
            for (auto &vertex: _vertices) {
                if (is_pose_vertex(vertex.second)) {
                    vertex.second->set_ordering_id(_ordering_poses);
                    _idx_pose_vertices.insert(pair<ulong, std::shared_ptr<Vertex>>(vertex.second->id(), vertex.second));
                    _ordering_poses += vertex.second->local_dimension();
                }
            }

            // 分配landmark的维度
            _ordering_landmarks = 0;
            for (auto &vertex: _vertices) {
                if (is_landmark_vertex(vertex.second)) {
                    vertex.second->set_ordering_id(_ordering_landmarks + _ordering_poses);
                    _idx_landmark_vertices.insert(pair<ulong, std::shared_ptr<Vertex>>(vertex.second->id(), vertex.second));
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
                if (is_pose_vertex(vertex)) {
                    _idx_pose_vertices.erase(vertex->id());
                }
                else {
                    _idx_landmark_vertices.erase(vertex->id());
                }
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
            return type == string("VertexPointXYZ") || type == string("VertexInverseDepth");
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
}