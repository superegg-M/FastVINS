//
// Created by Cain on 2024/4/29.
//

#include <iostream>
#include <fstream>
#include <array>

//#include <glog/logging.h>
#include <Eigen/Dense>
#include "tic_toc/tic_toc.h"
#include "problem.h"


using namespace std;

namespace graph_optimization {
    bool Problem::calculate_lbfgs(VecX &delta_x, unsigned long iterations) {
        constexpr static double eps = 1e-6;
        constexpr static double step_min = 1e-6;
        constexpr static double step_max = 1e6;
        constexpr static unsigned int search_cnt_max = 16;
        constexpr static double c1 = 1e-3;
        constexpr static double c2 = 0.9;
        constexpr static unsigned int past = 2;
        constexpr static unsigned int m = 16;
        static array<double, past> pf;
        for (auto& i : pf) {
            i = 0.;
        }

        unsigned int n = _ordering_generic;

        /* Initialize the limited memory. */
        Eigen::VectorXd lm_alpha = Eigen::VectorXd::Zero(m);
        Eigen::MatrixXd lm_s = Eigen::MatrixXd::Zero(n, m);
        Eigen::MatrixXd lm_y = Eigen::MatrixXd::Zero(n, m);
        Eigen::VectorXd lm_ys = Eigen::VectorXd::Zero(m);

        // 初始化
        update_residual();
        update_chi2();
        update_jacobian();
        update_gradient();

        double f = get_chi2();
        pf[0] = f;
        std::cout << "init: " << " , get_chi2 = " << f << std::endl;

        // 初始的下降方向
        VecX d = -_gradient;

        double gnorm_inf = _gradient.cwiseAbs().maxCoeff();
        if (gnorm_inf < eps) {
            return true;
        }

        // 初始步长
        double step = 1. / d.norm();

        VecX gp;
        unsigned int k = 1;
        unsigned int end = 0;
        unsigned int bound = 0;
        while (true) {
            gp = _gradient;
            if (step > step_max) {
                step = 0.5 * step_max;
            }

            // line search lewis-overton
            unsigned int count = 0;
            bool brackt = false, touched = false;
            double mu = 0., nu = step_max;
            double dg_init = d.dot(gp);
            if (dg_init > 0.) {
                std::cout << "dg_init > 0." << std::endl;
                return false;
            }

            double f_init = f;
            double dg_test = c1 * dg_init;
            double ds_test = c2 * dg_init;

            while (true) {
                delta_x = step * d;

                update_states(delta_x);
                update_residual();
                update_chi2();

                f = get_chi2();
                ++count;

                if (isnan(f) || isinf(f)) {
                    std::cout << "isnan(f) || isinf(f)" << std::endl;
                    rollback_states(delta_x);
                    return false;
                }

                /* Check the Armijo condition. */
                if (f > f_init + step * dg_test) {
                    nu = step;
                    brackt = true;
                } else {
                    update_jacobian();
                    update_gradient();

                    /* Check the weak Wolfe condition. */
                    if (_gradient.dot(d) < ds_test) {
                        mu = step;
                    } else {
                        std::cout << "iter: " << k << " , get_chi2 = " << f << std::endl;
                        break;
                    }
                }

                if (count > search_cnt_max) {
                    std::cout << "count > search_cnt_max" << std::endl;
                    rollback_states(delta_x);
                    return false;
                }

                if (brackt && (nu - mu) < eps * nu) {
                    std::cout << "brackt && (nu - mu) < 1e-6 * nu" << std::endl;
                    rollback_states(delta_x);
                    return false;
                }

                if (brackt) {
                    step = 0.5 * (mu + nu);
                } else {
                    step *= 2.;
                }

                if (step < step_min) {
                    std::cout << "step < step_min" << std::endl;
                    rollback_states(delta_x);
                    return false;
                }

                if (step > step_max) {
                    if (touched) {
                        std::cout << "touched" << std::endl;
                        rollback_states(delta_x);
                        return false;
                    } else {
                        touched = true;
                        step = step_max;
                    }
                }
                rollback_states(delta_x);
            }

            /*
            Convergence test.
            The criterion is given by the following formula:
            ||g(x)||_inf / max(1, ||x||_inf) < g_epsilon
            */
            gnorm_inf = _gradient.cwiseAbs().maxCoeff();
            if (gnorm_inf < eps) {
                return true;
            }

            /*
            Test for stopping criterion.
            The criterion is given by the following formula:
            |f(past_x) - f(x)| / max(1, |f(x)|) < \delta.
            */
            if (k >= past) {
                /* The stopping criterion. */
                double rate = abs(pf[k % past] - f) / abs(pf[k % past]);
                if (rate < eps) {
                    std::cout << "rate < 1e-6" << std::endl;
                    return true;
                }
            }

            /* Store the current value of the cost function. */
            pf[k % past] = f;

            if (k >= iterations) {
                std::cout << "k >= iterations" << std::endl;
                return true;
            }

            /* Count the iteration number. */
            ++k;

            /*
            Update vectors s and y:
            s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
            y_{k+1} = g_{k+1} - g_{k}.
            */
            lm_s.col(end) = delta_x;
            lm_y.col(end) = _gradient - gp;

            /*
            Compute scalars ys and yy:
            ys = y^t \cdot s = 1 / \rho.
            yy = y^t \cdot y.
            Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
            */
            double ys = lm_y.col(end).dot(lm_s.col(end));
            double yy = lm_y.col(end).squaredNorm();
            lm_ys(end) = ys;

            /* Compute the negative of gradients. */
            d = -_gradient;

            double cau = lm_s.col(end).squaredNorm() * gp.norm() * eps;
            if (ys > cau) {
                ++bound;
                bound = m < bound ? m : bound;
                end = (end + 1) % m;

                unsigned int j = end;
                for (unsigned int i = 0; i < bound; ++i) {
                    j = (j + m - 1) % m; /* if (--j == -1) j = m-1; */
                    /* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
                    lm_alpha(j) = lm_s.col(j).dot(d) / lm_ys(j);
                    /* q_{i} = q_{i+1} - \alpha_{i} y_{i}. */
                    d += (-lm_alpha(j)) * lm_y.col(j);
                }

                d *= ys / yy;

                for (unsigned int i = 0; i < bound; ++i) {
                    /* \beta_{j} = \rho_{j} y^t_{j} \cdot \gamm_{i}. */
                    double beta = lm_y.col(j).dot(d) / lm_ys(j);
                    /* \gamm_{i+1} = \gamm_{i} + (\alpha_{j} - \beta_{j}) s_{j}. */
                    d += (lm_alpha(j) - beta) * lm_s.col(j);
                    j = (j + 1) % m; /* if (++j == m) j = 0; */
                }
            }

            /* The search direction d is ready. We try step = 1 first. */
            step = 1.0;
        }
    }


}