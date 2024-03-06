//
// Created by Cain on 2024/1/4.
//

#include "imu_integration.h"

#include <utility>

namespace graph_optimization {
    IMUIntegration::IMUIntegration(Vec3 ba, Vec3 bg) : _ba(std::move(ba)), _bg(std::move(bg)) {
        const Mat33 i3 = Mat33::Identity();
        _noise_measurement.block<3, 3>(0, 0) = (_acc_noise * _acc_noise) * i3;
        _noise_measurement.block<3, 3>(3, 3) = (_gyro_noise * _gyro_noise) * i3;
        _noise_random_walk.block<3, 3>(0, 0) = (_acc_random_walk * _acc_random_walk) * i3;
        _noise_random_walk.block<3, 3>(3, 3) = (_gyro_random_walk * _gyro_random_walk) * i3;
    }

    void IMUIntegration::propagate(double dt, const Vec3 &acc, const Vec3 &gyro) {
        _dt_buf.emplace_back(dt);
        _acc_buf.emplace_back(acc);
        _gyro_buf.emplace_back(gyro);

        // 去偏移
        Vec3 a = acc - _ba;
        Vec3 a_last = _acc_last - _ba;
        Vec3 w = gyro - _bg;
        Vec3 w_last = _gyro_last - _bg;

        // 预积分(中值积分)
        Vec3 w_mid = 0.5 * (w_last + w);
        Vec3 delta_axis_angle = w_mid * dt;
        Sophus::SO3d dR = Sophus::SO3d::exp(delta_axis_angle);
        auto delta_r_last = _delta_r.matrix();
        _delta_r *= dR;

        Vec3 a_mid = 0.5 * (delta_r_last * a_last + _delta_r * a);
        Vec3 delta_vel = a_mid * dt;
        auto delta_v_last = _delta_v;
        _delta_v += delta_vel;

        Vec3 v_mid = 0.5 * (delta_v_last + _delta_v);
        Vec3 delta_pos = v_mid * dt;
        _delta_p += delta_pos;

        _sum_dt += dt;

        // 预积分关于偏移的雅可比
        auto delta_r = _delta_r.matrix();
        auto dr = dR.matrix();
        Mat33 jr_dt = Sophus::SO3d::JacobianR(delta_axis_angle) * dt;
        auto dr_dbg_last = _dr_dbg;
        _dr_dbg = dr.transpose() * _dr_dbg - jr_dt;

        auto delta_r_a_hat_last = delta_r_last * Sophus::SO3d::hat(a_last);
        auto delta_r_a_hat = delta_r * Sophus::SO3d::hat(a);
        auto dv_dba_last = _dv_dba;
        auto dv_dbg_last = _dv_dbg;
        _dv_dba -= (0.5 * dt) * (delta_r_last + delta_r);
        _dv_dbg -= (0.5 * dt) * (delta_r_a_hat_last * dr_dbg_last + delta_r_a_hat * _dr_dbg);

        _dp_dba += (0.5 * dt) * (dv_dba_last + _dv_dba);
        _dp_dbg += (0.5 * dt) * (dv_dbg_last + _dv_dbg);

        // 噪声迭代
        _A.block<3, 3>(0, 0).noalias() = dr.transpose();
        _A.block<3, 3>(3, 0).noalias() = -(0.5 * dt) * (delta_r_a_hat_last + delta_r_a_hat * dr.transpose());
        _A.block<3, 3>(6, 0).noalias() = (0.5 * dt) * _A.block<3, 3>(3, 0);
        _A.block<3, 3>(6, 3).noalias() = Mat33::Identity() * dt;

        _B.block<3, 3>(0, 0).noalias() = jr_dt;
        _B.block<3, 3>(3, 0).noalias() = (-0.5 * dt) * delta_r_a_hat * _B.block<3, 3>(0, 0);
        _B.block<3, 3>(6, 0).noalias() = (0.5 * dt) * _B.block<3, 3>(3, 0);
        _B.block<3, 3>(3, 3).noalias() = delta_r * dt;
        _B.block<3, 3>(6, 3).noalias() = (0.5 * dt) * _B.block<3, 3>(3, 3);

        _covariance_measurement = _A * _covariance_measurement * _A.transpose() + _B * _noise_measurement * _B.transpose();

        // 记录上一时刻数据
        _acc_last = acc;
        _gyro_last = gyro;
    }

    void IMUIntegration::repropagate() {
        reset();
        for (size_t i = 0; i < _dt_buf.size(); ++i) {
            propagate(_dt_buf[i], _acc_buf[i], _gyro_buf[i]);
        }
    }

    void IMUIntegration::correct(const Vec3 &delta_ba, const Vec3 &delta_bg) {
        _delta_r = _delta_r * Sophus::SO3d::exp(_dr_dbg * delta_bg);
        _delta_v += _dv_dba * delta_ba + _dv_dbg * delta_bg;
        _delta_p += _dp_dba * delta_ba + _dp_dbg * delta_bg;
    }

    void IMUIntegration::reset() {
        _sum_dt = 0;
        _delta_r = Sophus::SO3d();  // dR
        _delta_v = Vec3::Zero();    // dv
        _delta_p = Vec3::Zero();    // dp

        // jacobian w.r.t bg and ba
        _dr_dbg = Mat33::Zero();
        _dv_dbg = Mat33::Zero();
        _dv_dba = Mat33::Zero();
        _dp_dbg = Mat33::Zero();
        _dp_dba = Mat33::Zero();

        // noise propagation
        _covariance_measurement = Mat99::Zero();
        _covariance_random_walk = Mat66::Zero();
        _A = Mat99::Identity();
        _B = Mat96::Zero();
    }
}