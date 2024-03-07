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
    bool Problem::calculate_dog_leg(VecX &delta_x, unsigned long iterations) {
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

            // 计算 h_sd 和 h_gn
            one_step_steepest_descent(_delta_x_sd);
            bool is_gn_valid = one_step_gauss_newton(_delta_x_gn);

            bool is_good_step = false;
            unsigned long failure_cnt = 0;
            do {    // 通过修改delta, 找出合适的delta_x
                bool clip_delta_x = false;      ///< ||delta_x|| == ||delta|| ?

                if (is_gn_valid) {
                    double delta2 = _delta * _delta;
                    double h_gn2 = _delta_x_gn.squaredNorm();
                    if (h_gn2 <= delta2) {
                        delta_x = _delta_x_gn;
                    } else {
                        clip_delta_x = true;
                        double h_sd2 = _delta_x_sd.squaredNorm();
                        if (h_sd2 >= delta2) {
                            delta_x = (_delta / sqrt(h_sd2)) * _delta_x_sd;
                        } else {
                            VecX diff = _delta_x_gn - _delta_x_sd;
                            double diff2 = diff.squaredNorm();
                            if (diff2 < eps) {
                                delta_x = _delta_x_sd;
                            } else {
                                double inner = _delta_x_sd.dot(diff);
                                double beta = (-inner + sqrt(inner * inner + diff2 * (delta2 - h_sd2))) / diff2;
                                delta_x = _delta_x_sd + beta * diff;
                            }
                        }
                    }
                } else {
                    double delta2 = _delta * _delta;
                    double h_sd2 = _delta_x_sd.squaredNorm();
                    if (h_sd2 >= delta2) {
                        clip_delta_x = true;
                        delta_x = (_delta / sqrt(h_sd2)) * _delta_x_sd;
                    } else {
                        delta_x = _delta_x_sd;
                    }
                }

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
                double linear_gain = -0.5 * delta_x.dot(_hessian * delta_x) + delta_x.dot(_b);
                if (fabs(linear_gain) < eps) {
                    linear_gain = eps;
                }

                double rho = nonlinear_gain / linear_gain;
                if (rho < 0.25) {
                    _delta = std::max(0.25 * _delta, _delta_min);
                } else if (rho > 0.75 && clip_delta_x) {
                    _delta = std::min(2. * _delta, _delta_max);
                }

                if (rho > 0. && isfinite(new_chi2)) {
                    is_good_step = true;

                    // 如果chi2的减少已经很少了, 则可以认为x已经在最优点, 所以无需在迭代
                    if (rho < eps) {
                        is_good_to_stop = true;
                        std::cout << "Good: stop iteration due to (rho < eps)." << std::endl;
                        break;
                    }
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

                    // 若一直找不到合适的delta, 则直接结束迭代
                    if (failure_cnt > failure_cnt_max) {
                        is_bad_to_stop = true;
                        std::cout << "Bad: stop iteration due to (failure_cnt > failure_cnt_max)." << std::endl;
                        break;
                    }
                }
            } while (!is_good_step && !is_bad_to_stop && !is_good_to_stop);

            std::cout << "iter: " << iter << " , get_chi2 = " << current_chi2 << " , delta = " << _delta << std::endl;
        } while (iter < iterations && !is_bad_to_stop && !is_good_to_stop);

        return !is_bad_to_stop;
    }
}