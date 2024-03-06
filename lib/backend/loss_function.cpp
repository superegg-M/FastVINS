//
// Created by Cain on 2024/1/5.
//

#include "loss_function.h"

namespace graph_optimization {
    Vec3 HuberLoss::compute(double error2) const {
        double abs_error = sqrt(error2);
        if (abs_error <= _delta) {
            return {error2, 1., 0.};
        } else {
            Vec3 rho;
            rho[0] = _delta * (2. * abs_error - _delta);
            rho[1] = _delta / abs_error;
            rho[2] = -0.5 * rho[1] / error2;
            return rho;
        }
    }

    Vec3 CauchyLoss::compute(double error2) const {
        double value = 1. + error2 / _c2;
        Vec3 rho;
        rho[0] = _c2 * log(value);
        rho[1] = 1. / value;
        rho[2] = -rho[1] * rho[1] / _c2;
        return rho;
    }

    Vec3 TukeyLoss::compute(double error2) const {
        if (error2 < _c2) {
            double value = 1. - error2 / _c2;
            double value2 = value * value;
            double value3 = value * value2;
            Vec3 rho;
            rho[0] = _c2 / 3. * (1. - value3);
            rho[1] = value2;
            rho[2] = -2. / _c2 * value;
            return rho;
        } else {
            return {_c2 / 3., 0., 0.};
        }
    }
}