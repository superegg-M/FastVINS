//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"

#include "tic_toc/tic_toc.h"
#include "backend/eigen_types.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    void Estimator::process_imu(double dt, const Vec3 &linear_acceleration, const Vec3 &angular_velocity) {
        if (!_imu_integration) {
            // TODO: 把last的初值置为nan, 若为nan时，才进行赋值
            _acc_latest = linear_acceleration;
            _gyro_latest = angular_velocity;
            _imu_integration = new IMUIntegration {_acc_latest, _gyro_latest, _state.ba, _state.bg};
        }

        _imu_integration->push_back(dt, linear_acceleration, angular_velocity);
        Vec3 gyro_corr = 0.5 * (_gyro_latest + angular_velocity) - _state.bg;
        Vec3 acc0_corr = _state.q * (_acc_latest - _state.ba);
        auto delta_q = Sophus::SO3d::exp(gyro_corr * dt);
        _state.q *= delta_q.unit_quaternion();
        _state.q.normalize();
        Vec3 acc1_corr = _state.q * (linear_acceleration - _state.ba);
        Vec3 acc_corr = 0.5 * (acc0_corr + acc1_corr) + _g;
        _state.p += (0.5 * acc_corr * dt + _state.v) * dt;
        _state.v += acc_corr * dt;

        _acc_latest = linear_acceleration;
        _gyro_latest = angular_velocity;
    }
}