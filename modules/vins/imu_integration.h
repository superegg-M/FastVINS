//
// Created by Cain on 2024/1/4.
//

#ifndef GRAPH_OPTIMIZATION_IMU_INTEGRATION_H
#define GRAPH_OPTIMIZATION_IMU_INTEGRATION_H

#include <lib/backend/eigen_types.h>
#include <lib/thirdparty/Sophus/sophus/so3.hpp>

namespace vins {
    using namespace graph_optimization;
    class IMUIntegration {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        /**
         * constructor, with initial bias a and bias g
         * @param ba
         * @param bg
         */
        explicit IMUIntegration(Vec3 ba, Vec3 bg);

        ~IMUIntegration() = default;

        /**
         * propage pre-integrated measurements using raw IMU data
         * @param dt
         * @param acc
         * @param gyr_1
         */
        void propagate(double dt, const Vec3 &acc, const Vec3 &gyro);

        /**
         * according to pre-integration, when bias is updated, pre-integration should also be updated using
         * first-order expansion of ba and bg
         *
         * @param delta_ba
         * @param delta_bg
         */
        void correct(const Vec3 &delta_ba, const Vec3 &delta_bg);

        void set_gyro_bias(const Vec3 &bg) { _bg = bg; }

        void set_acc_bias(const Vec3 &ba) { _ba = ba; }

        /// if bias is update by a large value, redo the propagation
        void repropagate();

        /// reset measurements
        /// NOTE ba and bg will not be reset, only measurements and jacobians will be reset!
        void reset();

        /**
         * get the jacobians from r,v,p w.r.t. biases
         * @param _dr_dbg
         * @param _dv_dbg
         * @param _dv_dba
         * @param _dp_dbg
         * @param _dp_dba
         */
        void get_jacobians(Mat33 &dr_dbg, Mat33 &dv_dbg, Mat33 &dv_dba, Mat33 &dp_dbg, Mat33 &dp_dba) const {
            dr_dbg = _dr_dbg;
            dv_dbg = _dv_dbg;
            dv_dba = _dv_dba;
            dp_dbg = _dp_dbg;
            dp_dba = _dp_dba;
        }

        const Mat33 &get_dr_dbg() const { return _dr_dbg; }

        /// get propagated noise covariance
        const Mat99 &get_covariance_measurement() const { return _covariance_measurement; }

        /// get random walk covariance
        Mat66 get_covariance_randomWalk() const { return _noise_random_walk * _sum_dt; }

        /// get sum of time
        double get_sum_dt() const { return _sum_dt; }

        /**
         * get the integrated measurements
         * @param delta_r
         * @param delta_v
         * @param delta_p
         */
        void get_delta_RVP(Sophus::SO3d &delta_r, Vec3 &delta_v, Vec3 &delta_p) const {
            delta_r = _delta_r;
            delta_v = _delta_v;
            delta_p = _delta_p;
        }

        const Vec3 &get_dv() const { return _delta_v; }

        const Vec3 &get_dp() const { return _delta_p; }

        const Sophus::SO3d &get_dr() const { return _delta_r; }

    private:
        // raw data from IMU
        std::vector<double> _dt_buf;
        VecVec3 _acc_buf;
        VecVec3 _gyro_buf;
        Vec3 _acc_last;
        Vec3 _gyro_last;

        // pre-integrated IMU measurements
        double _sum_dt = 0;
        Sophus::SO3d _delta_r;  // dR
        Vec3 _delta_v = Vec3::Zero();    // dv
        Vec3 _delta_p = Vec3::Zero();    // dp

        // gravity, biases
        static Vec3 _gravity;
        Vec3 _bg = Vec3::Zero();    // initial bias of gyro
        Vec3 _ba = Vec3::Zero();    // initial bias of accelerator

        // jacobian w.r.t bg and ba
        Mat33 _dr_dbg = Mat33::Zero();
        Mat33 _dv_dbg = Mat33::Zero();
        Mat33 _dv_dba = Mat33::Zero();
        Mat33 _dp_dbg = Mat33::Zero();
        Mat33 _dp_dba = Mat33::Zero();

        // noise propagation
        Mat99 _covariance_measurement = Mat99::Zero();
        Mat66 _covariance_random_walk = Mat66::Zero();
        Mat99 _A = Mat99::Identity();
        Mat96 _B = Mat96::Zero();

        // raw noise of imu measurement
        Mat66 _noise_measurement = Mat66::Identity();
        Mat66 _noise_random_walk = Mat66::Identity();

        /**@brief accelerometer measurement noise standard deviation*/
        constexpr static double _acc_noise = 0.2;
        /**@brief gyroscope measurement noise standard deviation*/
        constexpr static double _gyro_noise = 0.02;
        /**@brief accelerometer bias random walk noise standard deviation*/
        constexpr static double _acc_random_walk = 0.0002;
        /**@brief gyroscope bias random walk noise standard deviation*/
        constexpr static double _gyro_random_walk = 2.0e-5;
    };
}

#endif //GRAPH_OPTIMIZATION_IMU_INTEGRATION_H
