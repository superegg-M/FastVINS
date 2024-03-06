//
// Created by Cain on 2024/3/6.
//

#include <iostream>
#include <fstream>

//#include <glog/logging.h>
#include <Eigen/Dense>
#include <lib/tic_toc/tic_toc.h>
#include "problem.h"


using namespace std;

namespace graph_optimization {
    bool Problem::calculate_levenberg_marquardt(VecX &delta_x, unsigned long iterations) {
        // 初始化
        compute_lambda_init_LM();   // 这里会计算_current_lambda
        std::cout << "init: " << " , chi= " << _current_chi << " , lambda= " << _current_lambda << std::endl;

        bool stop = false;
        unsigned long iter = 0;
        double last_chi = _current_chi;
        while (!stop && (iter < iterations)) {
            bool one_step_success = false;
            int false_cnt = 0;
            // 不断尝试 Lambda, 直到成功迭代一步
            while (!one_step_success) {
                solve_linear_system(delta_x);  // 解线性方程 (H + λ) x = b

                // 优化退出条件1： delta_x_ 很小则退出
                if (delta_x.squaredNorm() <= 1e-12) {
                    stop = true;
                    break;
                }

                update_states(delta_x);    // 更新状态量 x = x + delta_x
                one_step_success = is_good_step_in_LM();    // 判断当前步是否可行以及 LM 的 lambda 怎么更新, 这里会计算_current_lambda
                if (one_step_success) {
                    make_hessian(); // 在新线性化点 构建 hessian
                    // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
//                double b_max = 0.0;
//                for (int i = 0; i < b_.size(); ++i) {
//                    b_max = max(fabs(b_(i)), b_max);
//                }
//                // 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
//                stop = (b_max <= 1e-12);
                    false_cnt = 0;
                } else {
                    ++false_cnt;
                    rollback_states(delta_x);   // 回滚, x = x - delta_x

                    if (false_cnt > 10) {
                        stop = true;
                        break;
                    }
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
    }

    void Problem::compute_lambda_init_LM() {
        _ni = 2.;
        _current_lambda = 0.;

        _current_chi = 0.;
        for (auto &edge: _edges) {
            _current_chi += edge.second->robust_chi2();
        }
        // TODO: 是否应该叠加先验误差?
        _current_chi *= 0.5;

        _stop_threshold_LM = 1e-8 * _current_chi;          // 迭代条件为 误差下降 1e-8 倍

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

    bool Problem::is_good_step_in_LM() {
        static double eps = 1e-6;
        // double scale = 0.5 * _delta_x.transpose() * (_current_lambda * _delta_x + _b);
        double scale = 0.5 * _delta_x.transpose() * (VecX(_diag_lambda.array() * _delta_x.array()) + _b);
        scale += eps;    // make sure it's non-zero :)

        // recompute residuals after update state
        // 统计所有的残差
        double temp_chi = 0.0;
        for (auto &edge: _edges) {
            edge.second->compute_residual();
            temp_chi += edge.second->chi2();
        }

        double rho = (_current_chi - temp_chi) / scale;
        if (rho > 0 && isfinite(temp_chi)) {    // last step was good, 误差在下降
            double alpha = 1. - pow((2 * rho - 1), 3);
            alpha = std::min(alpha, 2. / 3.);
            double scale_factor = (std::max)(1. / 3., alpha);
            _current_lambda *= scale_factor;
            _diag_lambda *= scale_factor;
            _ni = 2.;
            _current_chi = temp_chi;
            return true;
        } else {
            _current_lambda *= _ni;
            _diag_lambda *= _ni;
            _ni *= 2.;
            return false;
        }
    }
}
