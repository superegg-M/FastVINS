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
    bool Problem::calculate_steepest_descent(VecX &delta_x) {
        static double eps = 1e-12;
        double num = _b.squaredNorm();
        double den = _b.dot(_hessian * _b) + eps;
        double alpha = num / den;
        delta_x = alpha * _b;
        return true;
    }
}