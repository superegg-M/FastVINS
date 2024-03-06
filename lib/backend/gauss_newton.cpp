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
        auto hessian_ldlt = _hessian.ldlt();

        if (hessian_ldlt.info() != Eigen::Success) {
            initialize_lambda();
            MatXX H = _hessian;
            for (unsigned long iter = 0; iter < iterations; ++iter) {
                H += _diag_lambda.asDiagonal();
                hessian_ldlt = H.ldlt();
                if (hessian_ldlt.info() == Eigen::Success) {
                    delta_x = hessian_ldlt.solve(_b);
                    return true;
                }
            }
            return false;
        } else {
            delta_x = hessian_ldlt.solve(_b);
            return true;
        }
    }
}