//
// Created by Cain on 2024/1/4.
//

#ifndef GRAPH_OPTIMIZATION_IMU_INTEGRATION_H
#define GRAPH_OPTIMIZATION_IMU_INTEGRATION_H

//#include <lib/backend/eigen_types.h>
#include "backend/eigen_types.h"
#include "thirdparty/Sophus/sophus/so3.hpp"
//#include <lib/thirdparty/Sophus/sophus/so3.hpp>

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
        explicit IMUIntegration(const Vec3 &acc_init, const Vec3 &gyro_init, const Vec3 &ba, const Vec3 &bg);

        ~IMUIntegration() = default;

        void push_back(double dt, const Vec3 &acc, const Vec3 &gyro);

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
        static void set_gravity(const Vec3 &gravity) { _gravity = gravity; }

        /// if bias is update by a large value, redo the propagation
        void repropagate(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg);

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
        const Mat33 &get_dp_dbg() const { return _dp_dbg; }
        const Mat33 &get_dp_dba() const { return _dp_dba; }
        const Mat33 &get_dv_dbg() const { return _dv_dbg; }
        const Mat33 &get_dv_dba() const { return _dv_dba; }

        const Mat1515 &get_covariance() const { return _covariance; }

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

        const Vec3 &get_delta_v() const { return _delta_v; }
        const Vec3 &get_delta_p() const { return _delta_p; }
        const Sophus::SO3d &get_delta_r() const { return _delta_r; }
        const Qd &get_delta_q() const { return _delta_r.unit_quaternion(); }

        const Vec3 &get_ba() const { return _ba; }
        const Vec3 &get_bg() const { return _bg; }
        static const Vec3 &get_gravity() { return _gravity; }

    protected:
        void calculate_cov(const Mat33 &delta_r, const Mat33 &delta_r_last,
                           const Mat33 &delta_r_a_hat, const Mat33 &delta_r_a_hat_last,
                           const Mat33 &dr, const Mat33 &jr_dt);

    private:
        // raw data from IMU
        std::vector<double> _dt_buf;
        VecVec3 _acc_buf;
        VecVec3 _gyro_buf;
        Vec3 _acc_init;
        Vec3 _gyro_init;
        Vec3 _acc_last;
        Vec3 _gyro_last;

        double _dt = 0.;
        static Vec3 _gravity;
        Vec3 _ba;
        Vec3 _bg;

        // pre-integrated IMU measurements
        double _sum_dt = 0;
        Sophus::SO3d _delta_r;  // dR
        Vec3 _delta_v = Vec3::Zero();    // dv
        Vec3 _delta_p = Vec3::Zero();    // dp

        // jacobian w.r.t bg and ba
        Mat33 _dr_dbg = Mat33::Zero();
        Mat33 _dv_dbg = Mat33::Zero();
        Mat33 _dv_dba = Mat33::Zero();
        Mat33 _dp_dbg = Mat33::Zero();
        Mat33 _dp_dba = Mat33::Zero();

        // noise propagation
        Mat1515 _covariance = Mat1515::Zero();

        Eigen::Matrix<double, 9, 9> _A00 = Eigen::Matrix<double, 9, 9>::Identity();
        Eigen::Matrix<double, 9, 6> _A01 = Eigen::Matrix<double, 9, 6>::Zero();
        Eigen::Matrix<double, 9, 12> _B00 = Eigen::Matrix<double, 9, 12>::Zero();
        Eigen::Matrix<double, 9, 6> _B01 = Eigen::Matrix<double, 9, 6>::Zero();

        // raw noise of imu measurement
        Eigen::Matrix<double, 12, 1> _noise_measurement = Eigen::Matrix<double, 12, 1>::Identity();
        Vec6 _noise_random_walk = Vec6::Identity();

        /**@brief accelerometer measurement noise standard deviation*/
        constexpr static double _acc_noise = 0.08;
        /**@brief gyroscope measurement noise standard deviation*/
        constexpr static double _gyro_noise = 0.004;
        /**@brief accelerometer bias random walk noise standard deviation*/
        constexpr static double _acc_random_walk = 0.00004;
        /**@brief gyroscope bias random walk noise standard deviation*/
        constexpr static double _gyro_random_walk = 2.0e-6;
    };
}

#endif //GRAPH_OPTIMIZATION_IMU_INTEGRATION_H
