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
    bool Problem::calculate_gauss_newton(VecX &delta_x, unsigned long iterations) {
        static double eps = 1e-12;
        static unsigned long failure_cnt_max = 10;

        // 初始化
        update_residual();
        update_chi2();
        update_jacobian();
        update_hessian();

        double current_chi2 = get_chi2();
        double new_chi2 = current_chi2;
        double stop_threshold = 1e-8 * current_chi2;          // 迭代条件为 误差下降 1e-8 倍
        std::cout << "init: " << " , get_chi2 = " << current_chi2 << std::endl;

        bool is_good_to_stop = false;
        bool is_bad_to_stop = false;
        unsigned long iter = 0;
        do {    // 迭代多次
            ++iter;

            bool is_good_step = false;
            unsigned long failure_cnt = 0;
            double alpha = 1.;
            if (!one_step_gauss_newton(delta_x)) {
                is_bad_to_stop = true;
                std::cout << "Bad: stop iteration due to (one_step_gauss_newton(delta_x) == false)." << std::endl;
                break;
            }
            do {
                // 如果 delta_x 很小则退出
                if (delta_x.squaredNorm() <= eps) {
                    is_good_to_stop = true;
                    std::cout << "Good: stop iteration due to (delta_x.squaredNorm() <= eps)." << std::endl;
                    break;
                }

                // x = x + dx
                update_states(delta_x);
                update_residual();
                update_chi2();

                new_chi2 = get_chi2();
                double nonlinear_gain = current_chi2 - new_chi2;

                if (nonlinear_gain > 0. && isfinite(new_chi2)) {
                    is_good_step = true;

                    // chi2的变化率小于1e-3
                    if (fabs(new_chi2 - current_chi2) < 1e-3 * current_chi2) {
                        is_good_to_stop = true;
                        std::cout << "Good: stop iteration due to (fabs(new_chi2 - current_chi2) < 1e-3 * current_chi2)." << std::endl;
                        break;
                    }
                    // chi2小于最初的chi2一定的倍率
                    if (current_chi2 < stop_threshold) {
                        is_good_to_stop = true;
                        std::cout << "Good: stop iteration due to (current_chi2 < stop_threshold)." << std::endl;
                        break;
                    }

                    current_chi2 = new_chi2;
                    update_jacobian();
                    update_hessian();
                } else {
                    is_good_step = false;
                    ++failure_cnt;

                    // 回退: x = x - dx
                    rollback_states(delta_x);

                    // 线性搜索
                    alpha *= 0.5;
                    delta_x *= alpha;

                    // 若一直找不到合适的delta, 则直接结束迭代
                    if (failure_cnt > failure_cnt_max) {
                        is_bad_to_stop = true;
                        std::cout << "Bad: stop iteration due to (failure_cnt > failure_cnt_max)." << std::endl;
                        break;
                    }
                }
            } while (!is_good_step && !is_bad_to_stop && !is_good_to_stop);

            std::cout << "iter: " << iter << " , get_chi2 = " << current_chi2 << " , alpha = " << alpha << std::endl;
        } while (iter < iterations && !is_bad_to_stop && !is_good_to_stop);

        return !is_bad_to_stop;
    }

    /*!
     * 计算h_gn, 在此前必须先运行update_hessian(), 因为该函数会调用initialize_lambda()
     * @param delta_x
     * @return
     */
    bool Problem::one_step_gauss_newton(VecX &delta_x) {
        static unsigned long iterations = 10;

        initialize_lambda();
        double ni = 2;
        for (unsigned long iter = 0; iter < iterations; ++iter) {
            if (solve_linear_system(delta_x)) {
                return true;
            }

            // 指数式增大lambda
            _current_lambda *= ni;
            _diag_lambda *= ni;
            ni *= 2.;
        }

        return false;
    }
}