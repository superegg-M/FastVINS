//
// Created by Cain on 2024/1/5.
//

#ifndef GRAPH_OPTIMIZATION_LOSS_FUNCTION_H
#define GRAPH_OPTIMIZATION_LOSS_FUNCTION_H

#include "eigen_types.h"

namespace graph_optimization {
    /**
     * loss function 即鲁棒核函数
     * loss套在误差之上
     * 假设某条边的残差为r，信息矩阵为I, 那么平方误差为r^T*I*r，令它的开方为e
     * 那么loss就是Compute(e)
     * 在求导时，也必须将loss function放到求导的第一项
     *
     * LossFunction是各核函数的基类，它可以派生出各种Loss
     */
    class LossFunction {
    public:
        enum class Type {
            TRIVIAL,
            HUBER,
            CAUCHY,
            TUKEY
        };

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual ~LossFunction() = default;

        virtual Vec3 compute(double error2) const = 0;
    };

    /**
     * 平凡的Loss，不作任何处理
     * 使用nullptr作为loss function时效果相同
     *
     * TrivalLoss(e) = e^2
     */
    class TrivialLoss : public LossFunction {
    public:
        Vec3 compute(double error2) const override { return {error2, 1., 0.}; }
    };

    /**
     * Huber loss
     *
     * Huber(e) = e^2                      if e <= delta
     * huber(e) = delta*(2*e - delta)      if e > delta
     */
    class HuberLoss : public LossFunction {
    public:
        explicit HuberLoss(double delta=1.345) : _delta(delta) {}

        ~HuberLoss() override = default;

        Vec3 compute(double error2) const override;

    private:
        double _delta;

    };

    /**
     * Cauchy Loss
     *
     * Cauchy(e) = c^2 * log(1 + e^2/c^2)
     */
    class CauchyLoss : public LossFunction {
    public:
        explicit CauchyLoss(double c=2.3849) : _c2(c*c) {}

        ~CauchyLoss() override = default;

        Vec3 compute(double error2) const override;

    private:
        double _c2;
    };

    /**
     * Huber loss
     *
     * Tukey(e) = c^2/3 * (1 - (1 - e^2/c^2)^3)     if e <= c
     * Tukey(e) = c^2/3                             if e > delta
     */
    class TukeyLoss : public LossFunction {
    public:
        explicit TukeyLoss(double c=4.685) : _c2(c*c) {}

        ~TukeyLoss() override = default;

        Vec3 compute(double error2) const override;

    private:
        double _c2;
    };
}

#endif //GRAPH_OPTIMIZATION_LOSS_FUNCTION_H
