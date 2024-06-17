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

    void Estimator::process_image(const unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> &image, double header) {
        std::cout << "process_image(): _windows.size() = " << _windows.size() << std::endl;
        TicToc t_process;

        // 创建imu节点, 需要在marg后加入到windows中
        _imu_node = new ImuNode {_imu_integration};

        // 需要在process_imu重新new
        _imu_integration = nullptr;

        // 设置imu顶点的参数
        Vec7 pose;
        pose << _state.p(0), _state.p(1), _state.p(2),
                _state.q.x(), _state.q.y(), _state.q.z(), _state.q.w();
        Vec9 motion;
        motion << _state.v(0), _state.v(1), _state.v(2),
                _state.ba(0), _state.ba(1), _state.ba(2),
                _state.bg(0), _state.bg(1), _state.bg(2);
        _imu_node->vertex_pose->set_parameters(pose);
        _imu_node->vertex_motion->set_parameters(motion);

        // 把imu顶点加入到problem中
        _problem.add_vertex(_imu_node->vertex_pose);
        _problem.add_vertex(_imu_node->vertex_motion);

        // 计算预积分edge
        if (!_windows.empty() && _imu_node->imu_integration && _imu_node->imu_integration->get_sum_dt() < 10.) {
            shared_ptr<EdgeImu> edge_imu(new EdgeImu(*_imu_node->imu_integration));
            edge_imu->add_vertex(_windows.newest()->vertex_pose);
            edge_imu->add_vertex(_windows.newest()->vertex_motion);
            edge_imu->add_vertex(_imu_node->vertex_pose);
            edge_imu->add_vertex(_imu_node->vertex_motion);
            _problem.add_edge(edge_imu);


//            pose = _windows.newest()->vertex_pose->get_parameters();
//            motion = _windows.newest()->vertex_motion->get_parameters();
//            Qd delta_q = Qd(pose[6], pose[3], pose[4], pose[5]).inverse() * _state.q;
//            Vec3 delta_p = Qd(pose[6], pose[3], pose[4], pose[5]).inverse() *
//                    (_state.p - Vec3(pose[0], pose[1], pose[2]) - Vec3(motion[0], motion[1], motion[2]) * _imu_node->imu_integration->get_sum_dt());
//            Vec3 delta_v = Qd(pose[6], pose[3], pose[4], pose[5]).inverse() * (_state.v - Vec3(motion[0], motion[1], motion[2]));
//            std::cout << "delta_q_gt: " << _imu_node->imu_integration->get_delta_q().w() << ", ";
//            std::cout << _imu_node->imu_integration->get_delta_q().x() << ", ";
//            std::cout << _imu_node->imu_integration->get_delta_q().y() << ", ";
//            std::cout << _imu_node->imu_integration->get_delta_q().z() << std::endl;
//            std::cout << "delta_q_est:" << delta_q.w() << ", " << delta_q.x() << ", " << delta_q.y() << ", " << delta_q.z() << std::endl;
//            std::cout << "delta_p_gt: " << _imu_node->imu_integration->get_delta_p().transpose() << std::endl;
//            std::cout << "delta_p_est: " << delta_p.transpose() << std::endl;
//            std::cout << "delta_v_gt: " << _imu_node->imu_integration->get_delta_v().transpose() << std::endl;
//            std::cout << "delta_v_est: " << delta_v.transpose() << std::endl;
        }

        // 遍历image中的每个feature
        double parallax_sum = 0.;
        unsigned int parallax_num = 0;
        unsigned int last_track_num = 0;
        for (auto &landmark : image) {
            // 特征点必须在双目中都看到
            auto &&cameras = landmark.second;
            if (cameras.size() < 2) {
                continue;
            }

            unsigned long feature_id = landmark.first;
            auto feature_it = _feature_map.find(feature_id);
            FeatureNode *feature_node = nullptr;

            // 计算每个camera的像素坐标
            vector<Vec3> points(cameras.size());
            for (unsigned int i = 0; i < cameras.size(); ++i) {
                points[i] = {cameras[i].second.x() / cameras[i].second.z(), cameras[i].second.y() / cameras[i].second.z(), 1.};
            }

            // 若feature不在feature map中，则需要新建feature_node
            if (feature_it == _feature_map.end()) {
                // 利用多目三角化出特征点深度
                Eigen::VectorXd A(cameras.size());
                Eigen::VectorXd b(cameras.size());
                for (unsigned int i = 1; i < cameras.size(); ++i) {
                    Qd q_c1c2 = (_q_ic[0].inverse() * _q_ic[i]).normalized();
                    Vec3 t_c1c2 = _q_ic[0].inverse() * (_t_ic[i] - _t_ic[0]);
                    Vec3 tmp1 = q_c1c2.inverse() * points[0];
                    Vec3 tmp2 = q_c1c2.inverse() * t_c1c2;
                    A(2 * i - 2) = points[i].x() * tmp1.z() - tmp1.x();
                    A(2 * i - 1) = points[i].y() * tmp1.z() - tmp1.y();
                    b(2 * i - 2) = points[i].x() * tmp2.z() - tmp2.x();
                    b(2 * i - 1) = points[i].y() * tmp2.z() - tmp2.y();
                }
                double depth = A.dot(b) / A.dot(A);
//                std::cout << "depth = " << depth << std::endl;

                if (depth > 0.) {
                    // 把特征点加入到feature_map中
                    feature_node = new FeatureNode(feature_id);
                    _feature_map.emplace(feature_id, feature_node);

                    // 设置逆深度
                    feature_node->vertex_landmark->parameters()[0] = 1. / depth;

                    // 设置标记
                    feature_node->is_triangulated = true;

                    /*
                     * 先不加入到problem中, 在构建重投影误差时才加入problem
                     * */
//                    // 把landmark顶点加入到problem中
//                    _problem.add_vertex(vertex_landmark);
                } else {
                    continue;
                }
            } else {
                feature_node = feature_it->second;
                if (feature_node->is_outlier || !feature_node->is_triangulated) {
                    continue;
                }

                // 记录有几个特征点被跟踪了
                ++last_track_num;
            }

            // 把像素坐标记录到imu_node中
            for (unsigned int i = 0; i < cameras.size(); ++i) {
                _imu_node->features_in_cameras[feature_id].emplace_back(cameras[i].first, points[i]);
            }

            // 先不要把当前的imu node加入到feature的imu deque以及windows中, 等到在slide的时候再添加

            // 构建重投影误差
            if (_windows.is_feature_suitable_to_reproject(feature_id)) {
                // host imu的左目
                auto &&host_imu = feature_node->imu_deque.oldest();   // 第一次看到feature的imu
                auto &&host_cameras = host_imu->features_in_cameras[feature_id];    // imu中，与feature对应的相机信息
                auto &&host_pixel_coord = host_cameras[0].second;    // feature在imu的左目的像素坐标

                // 若特征点还没有被加入到problem中, 则需要遍历所有imu来构建重投影误差
                if (_problem.vertices().find(feature_node->vertex_landmark->id()) == _problem.vertices().end()) {
                    // 把landmark顶点加入到problem中
                    _problem.add_vertex(feature_node->vertex_landmark);

                    // host imu除左目外的cameras
                    for (unsigned long j = 1; j < host_cameras.size(); ++j) {
                        // 构建视觉重投影误差边
                        shared_ptr<EdgeReprojectionOneImuTwoCameras> edge_reproj12(new EdgeReprojectionOneImuTwoCameras(
                                host_pixel_coord,
                                host_cameras[j].second
                        ));
                        edge_reproj12->add_vertex(feature_node->vertex_landmark);
                        edge_reproj12->add_vertex(_vertex_ext[0]);
                        edge_reproj12->add_vertex(_vertex_ext[j]);
                        _problem.add_edge(edge_reproj12);
                    }

                    // deque中除了host的其他imu
                    for (unsigned long i = 1; i < feature_node->imu_deque.size(); ++i) {
                        // 左目
                        auto &&other_imu = feature_node->imu_deque[i];   // 除了第一次看到feature的其他imu
                        auto &&other_cameras = other_imu->features_in_cameras[feature_id];    // imu中，与feature对应的相机信息

                        // 构建视觉重投影误差边
                        shared_ptr<EdgeReprojectionTwoImuOneCameras> edge_reproj(new EdgeReprojectionTwoImuOneCameras (
                                host_pixel_coord,
                                other_cameras[0].second
                        ));
                        edge_reproj->add_vertex(feature_node->vertex_landmark);
                        edge_reproj->add_vertex(host_imu->vertex_pose);
                        edge_reproj->add_vertex(other_imu->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[0]);
                        _problem.add_edge(edge_reproj);

                        // 遍历所有左目外的cameras
                        for (unsigned long j = 1; j < other_cameras.size(); ++j) {
                            // 构建视觉重投影误差边
                            shared_ptr<EdgeReprojectionTwoImuTwoCameras> edge_reproj22(new EdgeReprojectionTwoImuTwoCameras (
                                    host_pixel_coord,
                                    other_cameras[j].second
                            ));
                            edge_reproj22->add_vertex(feature_node->vertex_landmark);
                            edge_reproj22->add_vertex(host_imu->vertex_pose);
                            edge_reproj22->add_vertex(other_imu->vertex_pose);
                            edge_reproj22->add_vertex(_vertex_ext[0]);
                            edge_reproj22->add_vertex(_vertex_ext[j]);
                            _problem.add_edge(edge_reproj22);
                        }
                    }

                    // 当前的imu左目所观测到的像素坐标
                    Vec3 current_pixel_coord = {cameras[0].second.x(), cameras[0].second.y(), cameras[0].second.z()};    // feature在imu的左目的像素坐标

                    // 构建视觉重投影误差边
                    shared_ptr<EdgeReprojectionTwoImuOneCameras> edge_reproj(new EdgeReprojectionTwoImuOneCameras (
                            host_pixel_coord,
                            current_pixel_coord
                    ));
                    edge_reproj->add_vertex(feature_node->vertex_landmark);
                    edge_reproj->add_vertex(host_imu->vertex_pose);
                    edge_reproj->add_vertex(_imu_node->vertex_pose);
                    edge_reproj->add_vertex(_vertex_ext[0]);
                    _problem.add_edge(edge_reproj);

                    // 当前imu除了左目外的cameras
                    for (unsigned int i = 1; i < cameras.size(); ++i) {
                        // 当前imu除了左目外的cameras所观测到的像素坐标
                        current_pixel_coord = {cameras[i].second.x(), cameras[i].second.y(), cameras[i].second.z()};    // feature在imu的左目的像素坐标

                        // 构建视觉重投影误差边
                        shared_ptr<EdgeReprojectionTwoImuTwoCameras> edge_reproj22(new EdgeReprojectionTwoImuTwoCameras (
                                host_pixel_coord,
                                current_pixel_coord
                        ));
                        edge_reproj22->add_vertex(feature_node->vertex_landmark);
                        edge_reproj22->add_vertex(host_imu->vertex_pose);
                        edge_reproj22->add_vertex(_imu_node->vertex_pose);
                        edge_reproj22->add_vertex(_vertex_ext[0]);
                        edge_reproj22->add_vertex(_vertex_ext[i]);
                        _problem.add_edge(edge_reproj22);
                    }
                } else {
                    // 当前的imu左目所观测到的像素坐标
                    Vec3 current_pixel_coord = {cameras[0].second.x(), cameras[0].second.y(), cameras[0].second.z()};    // feature在imu的左目的像素坐标

                    // 构建视觉重投影误差边
                    shared_ptr<EdgeReprojectionTwoImuOneCameras> edge_reproj(new EdgeReprojectionTwoImuOneCameras (
                            host_pixel_coord,
                            current_pixel_coord
                    ));
                    edge_reproj->add_vertex(feature_node->vertex_landmark);
                    edge_reproj->add_vertex(host_imu->vertex_pose);
                    edge_reproj->add_vertex(_imu_node->vertex_pose);
                    edge_reproj->add_vertex(_vertex_ext[0]);
                    _problem.add_edge(edge_reproj);

                    // 当前imu除了左目外的cameras
                    for (unsigned int i = 1; i < cameras.size(); ++i) {
                        // 当前imu除了左目外的cameras所观测到的像素坐标
                        current_pixel_coord = {cameras[i].second.x(), cameras[i].second.y(), cameras[i].second.z()};    // feature在imu的左目的像素坐标

                        // 构建视觉重投影误差边
                        shared_ptr<EdgeReprojectionTwoImuTwoCameras> edge_reproj22(new EdgeReprojectionTwoImuTwoCameras (
                                host_pixel_coord,
                                current_pixel_coord
                        ));
                        edge_reproj22->add_vertex(feature_node->vertex_landmark);
                        edge_reproj22->add_vertex(host_imu->vertex_pose);
                        edge_reproj22->add_vertex(_imu_node->vertex_pose);
                        edge_reproj22->add_vertex(_vertex_ext[0]);
                        edge_reproj22->add_vertex(_vertex_ext[i]);
                        _problem.add_edge(edge_reproj22);
                    }
                }
            }

            /*
             * 计算每个特征点的视差，用于判断是否为keyframe:
             * 若windows中的newest frame是key frame, 则和newest frame计算视差
             * 否则, 和2nd newest frame计算视差，因为2nd newest必为key frame
             * */
            if (!_windows.empty()) {
                ImuNode *ref_imu;
                if (_windows.size() == 1 || _windows.newest()->is_key_frame) {
                    ref_imu = _windows.newest();
                } else {
                    ref_imu = _windows[_windows.size() - 2];
                }
                auto &&ref_cameras_it = ref_imu->features_in_cameras.find(feature_id);
                if (ref_cameras_it != ref_imu->features_in_cameras.end()) {
                    auto &&ref_cameras = ref_cameras_it->second;

                    double u_i = ref_cameras[0].second.x();
                    double v_i = ref_cameras[0].second.y();

                    double u_j = cameras[0].second.x();
                    double v_j = cameras[0].second.y();

                    double du = u_j - u_i;
                    double dv = v_j - v_i;

                    parallax_sum += sqrt(du * du + dv * dv);
                    ++parallax_num;
                }
            }
        }

        // 判断当前帧是否为key frame
        if (_windows.empty()) { // 把第一帧定为keyframe
            _imu_node->is_key_frame = true;
        } else {
            /*
             * 1. 若没有任何一个特征点在上个key frame中出现，则说明当前帧与上个key frame差异很大，所以一定是key frame
             * 2. 若当前帧中大部分的特征点都是新出现的，说明当前帧与历史所有帧差异都很大，所以一定是key frame
             * 其余情况则需通过平均视差值来判断是否为key frame
             * */
            if (parallax_num == 0 || last_track_num < 20) {
                _imu_node->is_key_frame = true;
            } else {
                _imu_node->is_key_frame = parallax_sum / parallax_num >= MIN_PARALLAX;  // 若视差大于一定值，则认为是key frame
            }
        }

        // 若windows中的最新一帧是key frame, 则需要marg最老帧
        if (_windows.empty() || _windows.newest()->is_key_frame) {
            marginalization_flag = MARGIN_OLD;
        } else {
            marginalization_flag = MARGIN_SECOND_NEW;
        }

        auto process_cost = t_process.toc();
        std::cout << "process_cost = " << process_cost << std::endl;

        if (solver_flag == INITIAL) {
            if (_windows.empty()) {
                // TODO: 初始位姿估计
            } else {
                auto &&pose_init = _windows.newest()->vertex_pose->get_parameters();
                Vec3 t_init {pose_init[0], pose_init[1], pose_init[2]};
                Qd q_init {pose_init[6], pose_init[3], pose_init[4], pose_init[5]};
                iter_pnp(_imu_node, &q_init, &t_init);
            }

            if (_windows.full()) {
                if (align_visual_to_imu()) {
                    solver_flag = NON_LINEAR;
                    solve_odometry();
                }

            }
        } else {
//            iter_pnp(_imu_node, &_state.q, &_state.p);

            solve_odometry();
        }


//        if (_windows.empty()) {
//            // TODO: 初始位姿估计
//        } else {
//            auto &&pose_init = _windows.newest()->vertex_pose->get_parameters();
//            Vec3 t_init {pose_init[0], pose_init[1], pose_init[2]};
//            Qd q_init {pose_init[6], pose_init[3], pose_init[4], pose_init[5]};
//            iter_pnp(_imu_node, &q_init, &t_init);
//        }
//
//        if (_windows.full()) {
//            if (solver_flag == INITIAL) {
//                if (align_visual_to_imu()) {
//                    solver_flag = NON_LINEAR;
//                }
//            }
//        }
//
//        if (solver_flag == NON_LINEAR) {
//            solve_odometry();
//        }

        _state.p = _imu_node->get_p();
        _state.q = _imu_node->get_q().normalized();
        _state.v = _imu_node->get_v();
        _state.ba = _imu_node->get_ba();
        _state.bg = _imu_node->get_bg();

        std::cout << "q_est: " << _state.q.w() << ", " << _state.q.x() << ", " << _state.q.y() << ", " << _state.q.z()<< std::endl;
        std::cout << "p_est: " << _state.p.transpose() << std::endl;
        std::cout << "v_est: " << _state.v.transpose() << std::endl;
        std::cout << "ba_est: " << _state.ba.transpose() << std::endl;
        std::cout << "bg_est: " << _state.bg.transpose() << std::endl;

        slide_window();
    }
}