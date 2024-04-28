//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"
#include "../vertex/vertex_pose.h"

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

    bool Estimator::structure_from_motion() {
        // 找出第一个与当前imu拥有足够视差的imu, 同时利用对极几何计算t_i_curr, R_i_curr
        unsigned long imu_index;
        Mat33 r_i_curr;
        Vec3 t_i_curr;
        if (!relative_pose(r_i_curr, t_i_curr, imu_index)) {
            cout << "Not enough features or parallax; Move device around" << endl;
            return false;
        }

        // 设置i的位姿
        auto imu_i = _windows[imu_index];
        Vec7 pose_i;
        pose_i << 0., 0., 0., 0., 0., 0., 1.;
        imu_i->vertex_pose->set_parameters(pose_i);

        // 设置curr的位姿
        Qd q_i_curr(r_i_curr);
        Vec7 pose_curr;
        pose_curr << t_i_curr.x(), t_i_curr.y(), t_i_curr.z(),
                q_i_curr.x(), q_i_curr.y(), q_i_curr.z(), q_i_curr.w();
        _imu_node->vertex_pose->set_parameters(pose_curr);

        // 在利用对极几何计算relative_pose时，已经同时得到了global landmark
//        // 利用i和curr进行三角化, 计算特征点的世界坐标
//        global_triangulate_with(imu_i, _imu_node);
//        std::cout << "1" << std::endl;

        /*
         * 1. 对imu_index后面的点进行pnp, 计算R, t.
         * 2. 得到R, t后进行三角化, 计算只有在imu_j到imu_node中才出现的特征点的世界坐标, i < j < curr
         * 3. 利用进行三角化, 计算只有在imu_i到imu_j中才出现的特征点的世界坐标, i < j < curr
         * */
        for (unsigned long j = imu_index + 1; j < _windows.size(); ++j) {
            // 用 j - 1 的位姿作为 j 的初始位姿估计
            auto &&pose_j = _windows[j - 1]->vertex_pose->get_parameters();
            Vec3 t_wj {pose_j(0), pose_j(1), pose_j(2)};
            Qd q_wj {pose_j(6), pose_j(3), pose_j(4), pose_i(5)};

            // pnp
            auto &&imu_j = _windows[j];
            iter_pnp(imu_j, &q_wj, &t_wj);
//            epnp(imu_j);
//            mlpnp(imu_j);
//            dltpnp(imu_j);
//            pnp(imu_j, &q_wj, &t_wj);

            // 三角化
            global_triangulate_with(imu_j, _imu_node);

            // 三角化
            global_triangulate_with(imu_i, imu_j);
        }

        /*
         * 0. 假设imu_index - 1与imu_index有共有的特征点, 并且已求得其世界坐标
         * 1. 对imu_index前面的点进行pnp, 计算R, t.
         * 2. 得到R, t后进行三角化, 计算只有在imu_j到imu_i中才出现的特征点的世界坐标, 0 <= j < i
         * */
        for (unsigned long k = 0; k < imu_index; ++k) {
            unsigned long j = imu_index - k - 1;

            // 用 j + 1 的位姿作为 j 的初始位姿估计
            auto &&pose_j = _windows[j + 1]->vertex_pose->get_parameters();
            Vec3 t_wj {pose_j(0), pose_j(1), pose_j(2)};
            Qd q_wj {pose_j(6), pose_j(3), pose_j(4), pose_j(5)};

            // pnp
            auto &&imu_j = _windows[j];
            iter_pnp(imu_j, &q_wj, &t_wj);
//            epnp(imu_j);
//            mlpnp(imu_j);
//            dltpnp(imu_j);
//            pnp(imu_j, &q_wj, &t_wj);

            // 三角化
            global_triangulate_with(imu_j, imu_i);
        }

//        // 遍历所有特征点, 对没有赋值的特征点进行三角化
//        for (auto &feature_it : _feature_map) {
//            global_triangulate_feature(feature_it.second);
//        }

        // 把所有pose和feature都调整到以第0帧为基准
        Vec7 pose_0 = _windows[0]->vertex_pose->get_parameters();
        Vec3 t_0 {pose_0[0], pose_0[1], pose_0[2]};
        Qd q_0 {pose_0[6], pose_0[3], pose_0[4], pose_0[5]};
        auto &&R_0 = q_0.toRotationMatrix();

        pose_0 << 0., 0., 0., 0., 0., 0., 1.;
        _windows[0]->vertex_pose->set_parameters(pose_0);

        for (unsigned long k = 1; k < _windows.size(); ++k) {
            Vec7 pose_k = _windows[k]->vertex_pose->get_parameters();
            Vec3 t_k {pose_k[0], pose_k[1], pose_k[2]};
            Qd q_k {pose_k[6], pose_k[3], pose_k[4], pose_k[5]};

            t_k = R_0.transpose() * (t_k - t_0);
            q_k = (q_0.inverse() * q_k).normalized();

            pose_k << t_k.x(), t_k.y(), t_k.z(), q_k.x(), q_k.y(), q_k.z(), q_k.w();
            _windows[k]->vertex_pose->set_parameters(pose_k);
        }

        Vec7 pose_k = _imu_node->vertex_pose->get_parameters();
        Vec3 t_k {pose_k[0], pose_k[1], pose_k[2]};
        Qd q_k {pose_k[6], pose_k[3], pose_k[4], pose_k[5]};

        t_k = R_0.transpose() * (t_k - t_0);
        q_k = (q_0.inverse() * q_k).normalized();

        pose_k << t_k.x(), t_k.y(), t_k.z(), q_k.x(), q_k.y(), q_k.z(), q_k.w();
        _imu_node->vertex_pose->set_parameters(pose_k);

        for (auto &feature_it : _feature_map) {
            if (feature_it.second->vertex_point3d) {
                Vec3 p = feature_it.second->vertex_point3d->get_parameters();
                p = R_0.transpose() * (p - t_0);
                feature_it.second->vertex_point3d->set_parameters(p);
            }
        }

        // 总体的优化
        // 固定住不参与优化的点
        vector<shared_ptr<VertexPose>> fixed_poses;
        fixed_poses.emplace_back(_vertex_ext[0]);
        fixed_poses.emplace_back(_windows[imu_index]->vertex_pose);
        fixed_poses.emplace_back(_imu_node->vertex_pose);

        // Global Bundle Adjustment
        global_bundle_adjustment(&fixed_poses);
        // 把特征点从global转为local
        for (auto &feature_it : _feature_map) {
            auto feature_node = feature_it.second;
            feature_node->from_global_to_local(_q_ic, _t_ic);
        }

//        // 把特征点从global转为local
//        for (auto &feature_it : _feature_map) {
//            auto feature_node = feature_it.second;
//            feature_node->from_global_to_local(_q_ic, _t_ic);
//        }
//        // Local Bundle Adjustment
//        local_bundle_adjustment(&fixed_poses);

        return true;
    }

    bool Estimator::relative_pose(Mat33 &r, Vec3 &t, unsigned long &imu_index) {
        TicToc t_r;
//        t = {2.58819, 0.340742, 0.};
//        r = Qd(0.991445, 0., 0., 0.130526);
//        imu_index = 5;
//
//        double scale = t.norm();
//
//        compute_essential_matrix(r, t, _windows[0], _imu_node);
//        imu_index = 0;
//        return true;

        for (unsigned long i = _windows.size() / 2, j = 1; i < _windows.size(); ++i, ++j) {
            if (compute_essential_matrix(r, t, _windows[i], _imu_node)) {
                imu_index = i;
                std::cout << "imu_index = " << imu_index << std::endl;
                std::cout << "find essential_matrix: " << t_r.toc() << std::endl;
                return true;
            }
            if (_windows.size() / 2 < j) {
                continue;
            }
            if (compute_essential_matrix(r, t, _windows[_windows.size() / 2 - j], _imu_node)) {
                imu_index = _windows.size() / 2 - j;
                std::cout << "imu_index = " << imu_index << std::endl;
                std::cout << "find essential_matrix: " << t_r.toc() << std::endl;
                return true;
            }
        }

//        for (unsigned long i = 0; i < _windows.size(); ++i) {
//            if (compute_essential_matrix(r, t, _windows[i], _imu_node)) {
//                imu_index = i;
//                std::cout << "imu_index = " << imu_index << std::endl;
//                return true;
//            }
//        }

//        if (compute_essential_matrix(r, t, _windows[1], _imu_node)) {
//            imu_index = 1;
//            std::cout << "imu_index = " << imu_index << std::endl;
//
//            Qd q(r);
//            std::cout << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << std::endl;
//            std::cout << t.transpose() << std::endl;
//            return true;
//        }

        std::cout << "find essential_matrix: " << t_r.toc() << " ms" << std::endl;
        return false;

//        Problem problem;
//        shared_ptr<VertexQuaternion> vertex_q(new VertexQuaternion);
//        shared_ptr<VertexSpherical> vertex_t(new VertexSpherical);
//        problem.add_vertex(vertex_q);
//        problem.add_vertex(vertex_t);
//        vertex_q->set_parameters(Vec4(0.991445, 0., 0., 0.130526));
//        for (auto &feature_in_cameras : _windows[imu_index]->features_in_cameras) {
//            unsigned long feature_id = feature_in_cameras.first;
//            auto &&it = _imu_node->features_in_cameras.find(feature_id);
//            if (it == _imu_node->features_in_cameras.end()) {
//                continue;
//            }
//
//            shared_ptr<EdgeEpipolar> edge_epipolar(new EdgeEpipolar(feature_in_cameras.second[0].second,
//                                                                    it->second[0].second));
//            edge_epipolar->add_vertex(vertex_q);
//            edge_epipolar->add_vertex(vertex_t);
//            problem.add_edge(edge_epipolar);
//        }
//
//        problem.solve(10);
//
//        /*
//         * 对极几何会存在4个全局最优解
//         * */
//
////        Qd q_c1c2 {vertex_q->get_parameters()[0], vertex_q->get_parameters()[1], vertex_q->get_parameters()[2], vertex_q->get_parameters()[3]};
////        Qd q_12 = _q_ic[0] * q_c1c2 * _q_ic[0].inverse();
////        std::cout << q_12.w() << ", " << q_12.x() << ", " << q_12.y() << ", " << q_12.z() << std::endl;
////
////        Qd qt {vertex_t->get_parameters()[0], vertex_t->get_parameters()[1], vertex_t->get_parameters()[2], vertex_t->get_parameters()[3]};
////        Vec3 t_c1c2_c1 = qt.toRotationMatrix().col(2);
////        Vec3 t_12_1 = _q_ic[0] * t_c1c2_c1 - q_12 * _t_ic[0] + _t_ic[0];
////        std::cout << t_12_1 * t.norm() << std::endl;
//
//        Vec3 tt = _q_ic[0].inverse() * (t + r * _t_ic[0] - _t_ic[0]);
//        Mat33 rr = _q_ic[0].inverse().toRotationMatrix() * r * _q_ic[0].toRotationMatrix();
//
//        std::cout << "ground true: " << std::endl;
//        std::cout << tt << std::endl;
//
//        Qd qt {vertex_t->get_parameters()[0], vertex_t->get_parameters()[1], vertex_t->get_parameters()[2], vertex_t->get_parameters()[3]};
//        Vec3 t_c1c2_c1 = qt.toRotationMatrix().col(2);
//        Qd q_c1c2 {vertex_q->get_parameters()[0], vertex_q->get_parameters()[1], vertex_q->get_parameters()[2], vertex_q->get_parameters()[3]};
//        std::cout << "estimate: " << std::endl;
//        std::cout << t_c1c2_c1 << std::endl;
////        std::cout << -t_c1c2_c1 << std::endl;
////        std::cout << q_c1c2.inverse().toRotationMatrix() * t_c1c2_c1 << std::endl;
////        std::cout << -q_c1c2.inverse().toRotationMatrix() * t_c1c2_c1 << std::endl;
//
//        std::cout << "on 1: " << (q_c1c2.toRotationMatrix() * Vec3(0., 0., 1.) + t_c1c2_c1).transpose() << std::endl;
//        std::cout << "on 2: " << (q_c1c2.inverse().toRotationMatrix() * (Vec3(0., 0., 1.) - t_c1c2_c1)).transpose() << std::endl;
//
////        Mat33 e = Sophus::SO3d::hat(t_c1c2_c1) * q_c1c2.toRotationMatrix();
////        Eigen::JacobiSVD<Eigen::MatrixXf> e_svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);
//
////        Mat33 r1 = e_svd.matrixU() *
//
//
//
////        for (auto &feature_in_cameras : _windows[imu_index]->features_in_cameras) {
////            unsigned long feature_id = feature_in_cameras.first;
////            auto &&it = _imu_node->features_in_cameras.find(feature_id);
////            if (it == _imu_node->features_in_cameras.end()) {
////                continue;
////            }
////
////            std::cout << "ground true : " << feature_in_cameras.second[0].second.dot(tt.cross(rr * it->second[0].second)) << std::endl;
////            std::cout << "estimate : " << feature_in_cameras.second[0].second.dot(t_c1c2_c1.cross(q_c1c2.toRotationMatrix() * it->second[0].second)) << std::endl;
////        }

//        return true;

//        // 遍历windows中的所有imu, 计算哪个imu是第一个与当前imu拥有足够视差的imu
//        for (unsigned long i = 0; i < _windows.size(); ++i) {
//            // 最大可能的匹配对数
//            unsigned long max_count = max(_windows[i]->features_in_cameras.size(), _imu_node->features_in_cameras.size());
//
//            // 获取第i个imu与当前imu的共同特征的像素坐标
//            vector<pair<Vec3, Vec3>> match_pairs;
//            match_pairs.reserve(max_count);
//
//            for (auto &feature_in_cameras : _windows[i]->features_in_cameras) {
//                auto &&it = _imu_node->features_in_cameras.find(feature_in_cameras.first);
//                if (it == _imu_node->features_in_cameras.end()) {
//                    continue;
//                }
//                match_pairs.emplace_back(feature_in_cameras.second[0].second, it->second[0].second);
//            }
//
//            // 特征点数量大于一定值时，计算视差
//            constexpr static unsigned long min_count_lim = 20;
//            if (match_pairs.size() > min_count_lim) {
//                double average_parallax = 0.;
//                for (auto &match_pair : match_pairs) {
//                    double du = corre.first.x() - corre.second.x();
//                    double dv = corre.first.y() - corre.second.y();
//                    average_parallax += sqrt(du * du + dv * dv);
//                }
//                average_parallax /= double(match_pairs.size());
//
//                if (average_parallax * 460. > 30. && m_estimator.solveRelativeRT(corres, r, t)) {
//                    imu_index = i;
//                    //ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
//                    return true;
//                }
//            }
//        }
//
//        return false;
    }
}