//
// Created by Cain on 2024/3/6.
//

#include <iostream>
#include <fstream>

//#include <glog/logging.h>
#include <Eigen/Dense>
#include "tic_toc/tic_toc.h"
#include "problem.h"


using namespace std;

namespace graph_optimization {
    bool Problem::calculate_levenberg_marquardt(VecX &delta_x, unsigned long iterations) {
        static double eps = 1e-12;
        static unsigned long failure_cnt_max = 10;

        // 初始化
        update_residual();
        update_chi2();
        update_jacobian();
        update_hessian();
        initialize_lambda();

        double current_chi2 = get_chi2();
        double new_chi2 = current_chi2;
        double stop_threshold = 1e-8 * current_chi2;          // 迭代条件为 误差下降 1e-8 倍
        std::cout << "init: " << " , get_chi2 = " << current_chi2 << std::endl;

        _ni = 2.;

        bool is_good_to_stop = false;
        bool is_bad_to_stop = false;
        unsigned long iter = 0;
        do {    // 迭代多次
            ++iter;

            bool is_good_step = false;
            unsigned long failure_cnt = 0;
            do {
                // 尝试求解: (H+λ)Δx = b, 若失败则增大λ, 若失败多次, 则直接退出迭代
                if (!solve_linear_system(delta_x)) {
                    std::cout << "Bad: stop iteration due to (solve_linear_system(delta_x) == false)." << std::endl;

                    is_good_step = false;
                    ++failure_cnt;

                    // 若一直找不到合适的delta, 则直接结束迭代
                    if (failure_cnt > failure_cnt_max) {
                        is_bad_to_stop = true;
                        std::cout << "Bad: stop iteration due to (failure_cnt > failure_cnt_max)." << std::endl;
                        break;
                    }

                    _current_lambda *= _ni;
                    _diag_lambda *= _ni;
                    _ni *= 2.;
                    continue;
                }

//                // 如果 delta_x 很小则退出
//                if (delta_x.squaredNorm() <= eps) {
//                    is_good_to_stop = true;
//                    std::cout << "Good: stop iteration due to (delta_x.squaredNorm() <= eps)." << std::endl;
//                    break;
//                }

                // x = x + dx
                update_states(delta_x);
                update_residual();
                update_chi2();

                new_chi2 = get_chi2();
                double nonlinear_gain = current_chi2 - new_chi2;
                double linear_gain = 0.5 * delta_x.transpose() * (VecX(_diag_lambda.array() * delta_x.array()) + _b);
                if (fabs(linear_gain) < eps) {
                    linear_gain = eps;
                }

                double rho = nonlinear_gain / linear_gain;
                if (rho > 0.) {
                    double alpha = 1. - pow((2 * rho - 1), 3);
                    alpha = std::min(alpha, 2. / 3.);
                    double scale_factor = (std::max)(1. / 3., alpha);
                    _current_lambda *= scale_factor;
                    _diag_lambda *= scale_factor;
                    _ni = 2.;
                } else {
                    _current_lambda *= _ni;
                    _diag_lambda *= _ni;
                    _ni *= 2.;
                }

                if (rho > 0. && isfinite(new_chi2)) {    // last step was good, 误差在下降
                    is_good_step = true;
                    failure_cnt = 0;

                    // 如果chi2的减少已经很少了, 则可以认为x已经在最优点, 所以无需在迭代
                    if (rho < eps) {
                        is_good_to_stop = true;
                        std::cout << "Good: stop iteration due to (rho < eps)." << std::endl;
                        break;
                    }
                    // chi2的变化率小于1e-6
                    if (fabs(new_chi2 - current_chi2) < 1e-6 * current_chi2) {
                        is_good_to_stop = true;
                        std::cout << "Good: stop iteration due to (fabs(new_chi2 - current_chi2) < 1e-6 * current_chi2)." << std::endl;
                        break;
                    }
                    // chi2小于最初的chi2一定的倍率
                    if (new_chi2 < stop_threshold) {
                        is_good_to_stop = true;
                        std::cout << "Good: stop iteration due to (current_chi2 < stop_threshold)." << std::endl;
                        break;
                    }

                    current_chi2 = new_chi2;
                    update_jacobian();
                    update_hessian();
                    delta_x.setZero();
                } else {
                    is_good_step = false;
                    ++failure_cnt;

                    // 回退: x = x - dx
                    rollback_states(delta_x);

                    // 若一直找不到合适的delta, 则直接结束迭代
                    if (failure_cnt > failure_cnt_max) {
                        is_bad_to_stop = true;
                        std::cout << "Bad: stop iteration due to (failure_cnt > failure_cnt_max)." << std::endl;
                        break;
                    }
                }
            } while (!is_good_step && !is_bad_to_stop && !is_good_to_stop);

            std::cout << "iter: " << iter << " , get_chi2 = " << current_chi2 << " , lambda = " << _current_lambda << std::endl;
        } while (iter < iterations && !is_bad_to_stop && !is_good_to_stop);

        return !is_bad_to_stop;
    }
}
