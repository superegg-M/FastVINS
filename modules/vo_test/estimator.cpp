//
// Created by Cain on 2024/1/11.
//

#include "estimator.h"
#include "vertex/vertex_inverse_depth.h"
#include "vertex/vertex_pose.h"
#include "vertex/vertex_motion.h"
#include "edge/edge_reprojection.h"
#include "edge/edge_imu.h"

#include "tic_toc/tic_toc.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    Estimator::Estimator() : _vertex_ext(NUM_OF_CAM), _windows(WINDOW_SIZE) {
        _q_ic.resize(NUM_OF_CAM);
        _t_ic.resize(NUM_OF_CAM);
        _vertex_ext.resize(NUM_OF_CAM);
        _q_ic[0] = {cos(0.5 * double(EIGEN_PI) / 2.), -sin(0.5 * double(EIGEN_PI) / 2.), 0., 0.};
        _t_ic[0] = {0., 0., 0.};
        Vec7 pose;
        pose << 0., 0., 0., -sin(0.5 * double(EIGEN_PI) / 2.), 0., 0., cos(0.5 * double(EIGEN_PI) / 2.);
        _vertex_ext[0] = std::make_shared<VertexPose>();
        _vertex_ext[0]->set_parameters(pose);

        solver_flag = INITIAL;
    }

    void Estimator::process_imu(double dt, const Vec3 &linear_acceleration, const Vec3 &angular_velocity) {
        if (!_imu_integration) {
            // TODO: 把last的初值置为nan, 若为nan时，才进行赋值
            _acc_latest = linear_acceleration;
            _gyro_latest = angular_velocity;
            _imu_integration = new IMUIntegration {_acc_latest, _gyro_latest, _state.ba, _state.bg};
        }

        _imu_integration->push_back(dt, linear_acceleration, angular_velocity);
        Vec3 gyro_corr = 0.5 * (_gyro_latest + angular_velocity) - _state.bg;
        Vec3 acc0_corr = _state.q.toRotationMatrix() * (_acc_latest - _state.ba);
        auto delta_q = Sophus::SO3d::exp(gyro_corr * dt);
        _state.q *= delta_q.unit_quaternion();
        _state.q.normalize();
        Vec3 acc1_corr = _state.q.toRotationMatrix() * (linear_acceleration - _state.ba);
        Vec3 acc_corr = 0.5 * (acc0_corr + acc1_corr) - _g;
        _state.p += (0.5 * acc_corr * dt + _state.v) * dt;
        _state.v += acc_corr * dt;

        _acc_latest = linear_acceleration;
        _gyro_latest = angular_velocity;
    }

    void Estimator::process_image(const unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> &image, double header) {
        std::cout << "_windows.size() = " << _windows.size() << std::endl;
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
        }

        // 遍历image中的每个feature
        double parallax_sum = 0.;
        unsigned int parallax_num = 0;
        unsigned int last_track_num = 0;
        for (auto &landmark : image) {
            unsigned long feature_id = landmark.first;
            auto feature_it = _feature_map.find(feature_id);
            FeatureNode *feature_node = nullptr;

            // 若feature，不在feature map中，则需要新建feature_node
            if (feature_it == _feature_map.end()) {
                feature_node = new FeatureNode(feature_id);
                _feature_map.emplace(feature_id, feature_node);
//                feature_it = _feature_map.find(feature_id);
            } else {
                feature_node = feature_it->second;
                ++last_track_num;   // 记录有几个特征点被跟踪了
            }

            // 先不要把当前的imu node加入到feature的imu deque以及windows中, 等到在slide的时候再添加

            // TODO: 这里假设只有一个双目，后续需要考虑多个双目的情况
            // 遍历看到当前feature的每个camera, 记录feature在camera中的像素坐标
            auto &&cameras = landmark.second;
            for (auto &camera : cameras) {
                Vec3 point {camera.second.x(), camera.second.y(), camera.second.z()};
                _imu_node->features_in_cameras[feature_id].emplace_back(camera.first, point);
            }

            // 判断feature是否能够被用于计算重投影误差
            if (_windows.is_feature_suitable_to_reproject(feature_id)) {
                // 获取feature的参考值
                auto &&host_imu = feature_node->imu_deque.oldest();   // 第一次看到feature的imu
                auto &&host_imu_pose = host_imu->vertex_pose;   // imu的位姿
                auto &&host_cameras = host_imu->features_in_cameras[feature_id];    // imu中，与feature对应的相机信息
                auto &&host_camera_id = host_cameras[0].first;  // camera的id
                auto &&host_pixel_coord = host_cameras[0].second;    // feature在imu的左目的像素坐标

                // 若特征点没有进行过初始化，则需要先进行三角化, 同时对windows中的所有frame计算视觉重投影误差
                if (!feature_node->vertex_landmark) {
                    // 构建landmark顶点
                    shared_ptr<VertexInverseDepth> vertex_landmark(new VertexInverseDepth);
                    feature_node->vertex_landmark = vertex_landmark;

                    // 把landmark顶点加入到problem中
                    _problem.add_vertex(vertex_landmark);

                    // 使用三角化, 对landmark进行初始化
                    unsigned long num_frames = cameras.size();  // 当前imu看到该feature的相机数
                    for (unsigned long i = 0; i < feature_node->imu_deque.size(); ++i) {    // windows中, 看到该feature的相机数
                        num_frames += feature_node->imu_deque[i]->features_in_cameras[feature_id].size();
                    }
                    Eigen::MatrixXd svd_A(2 * num_frames, 4);
                    Eigen::Matrix<double, 3, 4> P;
                    Eigen::Vector3d f;

                    // host imu的左目信息
                    Vec3 p_i {host_imu_pose->get_parameters()(0), host_imu_pose->get_parameters()(1), host_imu_pose->get_parameters()(2)};
                    Qd q_i {host_imu_pose->get_parameters()(6), host_imu_pose->get_parameters()(3), host_imu_pose->get_parameters()(4), host_imu_pose->get_parameters()(5)};
                    Mat33 r_i {q_i.toRotationMatrix()};

                    Eigen::Vector3d t_wci_w = p_i + r_i * _t_ic[host_camera_id];
                    Eigen::Matrix3d r_wci = r_i * _q_ic[host_camera_id];

                    P.leftCols<3>() = Eigen::Matrix3d::Identity();
                    P.rightCols<1>() = Eigen::Vector3d::Zero();

                    f = host_pixel_coord.normalized();
                    svd_A.row(0) = f[0] * P.row(2) - f[2] * P.row(0);
                    svd_A.row(1) = f[1] * P.row(2) - f[2] * P.row(1);

                    unsigned long index = 0;

                    // host imu的其他cameras
                    for (unsigned long j = 1; j < host_cameras.size(); ++j) {
                        ++index;

                        // 获取feature的参考值
                        auto &&other_camera_id = host_cameras[j].first; // camera的id
                        auto &&other_pixel_coord = host_cameras[j].second;    // feature在imu的左目的像素坐标

                        Eigen::Vector3d t_wcj_w = p_i + r_i * _t_ic[other_camera_id];
                        Eigen::Matrix3d r_wcj = r_i * _q_ic[other_camera_id];
                        Eigen::Vector3d t_cicj_ci = r_wci.transpose() * (t_wcj_w - t_wci_w);
                        Eigen::Matrix3d r_cicj = r_wci.transpose() * r_wcj;

                        P.leftCols<3>() = r_cicj.transpose();
                        P.rightCols<1>() = -r_cicj.transpose() * t_cicj_ci;

                        f = other_pixel_coord.normalized();
                        svd_A.row(2 * index) = f[0] * P.row(2) - f[2] * P.row(0);
                        svd_A.row(2 * index + 1) = f[1] * P.row(2) - f[2] * P.row(1);

                        // 构建视觉重投影误差边
                        shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                host_pixel_coord,
                                other_pixel_coord
                        ));
                        edge_reproj->add_vertex(vertex_landmark);
                        edge_reproj->add_vertex(host_imu_pose);
                        edge_reproj->add_vertex(host_imu_pose);
                        edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
                    }

                    // deque中除了host的其他imu
                    for (unsigned long i = 1; i < feature_node->imu_deque.size(); ++i) {
                        auto &&other_imu = feature_node->imu_deque[i];   // 除了第一次看到feature的其他imu
                        auto &&other_imu_pose = other_imu->vertex_pose; // imu的位姿
                        auto &&other_cameras = other_imu->features_in_cameras[feature_id];    // imu中，与feature对应的相机信息

                        // 从vertex中获取位姿
                        Vec3 p_j {other_imu_pose->get_parameters()(0), other_imu_pose->get_parameters()(1), other_imu_pose->get_parameters()(2)};
                        Qd q_j {other_imu_pose->get_parameters()(6), other_imu_pose->get_parameters()(3), other_imu_pose->get_parameters()(4), other_imu_pose->get_parameters()(5)};
                        Mat33 r_j {q_j.toRotationMatrix()};

                        // 遍历所有cameras
                        for (unsigned long j = 0; j < other_cameras.size(); ++j) {
                            ++index;

                            // 获取feature的参考值
                            auto &&other_camera_id = other_cameras[j].first;    // camera的id
                            auto &&other_pixel_coord = other_cameras[j].second;    // feature在imu的左目的像素坐标

                            Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[other_camera_id];
                            Eigen::Matrix3d r_wcj = r_j * _q_ic[other_camera_id];
                            Eigen::Vector3d t_cicj_ci = r_wci.transpose() * (t_wcj_w - t_wci_w);
                            Eigen::Matrix3d r_cicj = r_wci.transpose() * r_wcj;

                            P.leftCols<3>() = r_cicj.transpose();
                            P.rightCols<1>() = -r_cicj.transpose() * t_cicj_ci;

                            f = other_pixel_coord.normalized();
                            svd_A.row(2 * index) = f[0] * P.row(2) - f[2] * P.row(0);
                            svd_A.row(2 * index + 1) = f[1] * P.row(2) - f[2] * P.row(1);

                            // 构建视觉重投影误差边
                            shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                    host_pixel_coord,
                                    other_pixel_coord
                            ));
                            edge_reproj->add_vertex(vertex_landmark);
                            edge_reproj->add_vertex(host_imu_pose);
                            edge_reproj->add_vertex(other_imu_pose);
                            edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
                        }
                    }

                    // 当前的imu
                    auto state_r = _state.q.toRotationMatrix();
                    for (auto &camera : cameras) {
                        unsigned long current_camera_id = camera.first;  // camera的id
                        Vec3 current_pixel_coord = {camera.second.x(), camera.second.y(), camera.second.z()};    // feature在imu的左目的像素坐标

                        Eigen::Vector3d t_wcj_w = _state.p + state_r * _t_ic[current_camera_id];
                        Eigen::Matrix3d r_wcj = state_r * _q_ic[current_camera_id];
                        Eigen::Vector3d t_cicj_ci = r_wci.transpose() * (t_wcj_w - t_wci_w);
                        Eigen::Matrix3d r_cicj = r_wci.transpose() * r_wcj;

                        P.leftCols<3>() = r_cicj.transpose();
                        P.rightCols<1>() = -r_cicj.transpose() * t_cicj_ci;

                        f = current_pixel_coord.normalized();
                        svd_A.row(2 * index) = f[0] * P.row(2) - f[2] * P.row(0);
                        svd_A.row(2 * index + 1) = f[1] * P.row(2) - f[2] * P.row(1);

                        // 构建视觉重投影误差边
                        shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                host_pixel_coord,
                                current_pixel_coord
                        ));
                        edge_reproj->add_vertex(feature_node->vertex_landmark);
                        edge_reproj->add_vertex(host_imu_pose);
                        edge_reproj->add_vertex(_imu_node->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[current_camera_id]);
                    }

                    // 最小二乘计算深度
                    Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
                    double depth = svd_V[2] / svd_V[3];
                    if (depth < 0.1) {
                        depth = INIT_DEPTH;
                    }

                    // 设置landmark顶点的逆深度
                    vertex_landmark->set_parameters(Vec1(1. / depth));
                } else {
                    // 对当前imu下的所有cameras计算视觉重投影误差
                    for (auto &camera : cameras) {
                        unsigned long other_camera_id = camera.first;  // camera的id
                        Vec3 other_pixel_coord = {camera.second.x(), camera.second.y(), camera.second.z()};    // feature在imu的左目的像素坐标

                        shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                host_pixel_coord,
                                other_pixel_coord
                        ));
                        edge_reproj->add_vertex(feature_node->vertex_landmark);
                        edge_reproj->add_vertex(host_imu_pose);
                        edge_reproj->add_vertex(_imu_node->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
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
                    double dv = v_j - v_j;

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
//        std::cout << "5" << std::endl;
//        std::cout << "_windows.size() = " << _windows.size() << std::endl;
//        marginalization_flag = MARGIN_OLD;
//        slide_window();
//        std::cout << "6" << std::endl;
//        std::cout << "_windows.size() = " << _windows.size() << std::endl;

        if (solver_flag == INITIAL) {
            if (_windows.full()) {   // WINDOW已经装满了, 且还有camera frame
                bool is_initialized = initial_structure();
                if (is_initialized) {
                    cout << "Initialization finish!" << endl;

                    // 初始化后进行非线性优化
                    solver_flag = NON_LINEAR;
                    solve_odometry();

                    _state.p = _imu_node->get_p();
                    _state.q = _imu_node->get_q().normalized();
                    _state.v = _imu_node->get_v();
                    _state.ba = _imu_node->get_ba();
                    _state.bg = _imu_node->get_bg();
                }
            }
        } else {
            TicToc t_solve;
            solve_odometry();

            TicToc t_margin;

            _state.p = _imu_node->get_p();
            _state.q = _imu_node->get_q().normalized();
            _state.v = _imu_node->get_v();
            _state.ba = _imu_node->get_ba();
            _state.bg = _imu_node->get_bg();
        }

        slide_window();
    }

    void Estimator::global_bundle_adjustment(vector<shared_ptr<VertexPose>> *fixed_poses) {
        // VO优化
        ProblemSLAM problem;

        // fix住imu
        if (fixed_poses) {
            for (auto &pose : *fixed_poses) {
                pose->set_fixed();
            }
        }

        // 把外参加入到problem中
        problem.add_vertex(_vertex_ext[0]);

        // 把windows中的imu加入到problem中
        for (unsigned long i = 0; i < _windows.size(); ++i) {
            problem.add_vertex(_windows[i]->vertex_pose);
        }

        // 把当前的imu加入到problem中
        problem.add_vertex(_imu_node->vertex_pose);

        for (auto &feature_it : _feature_map) {
            auto &&feature_id = feature_it.first;
            auto &&feature_node = feature_it.second;

            // 只有进行了初始化的特征点才参与计算
            if (feature_node->vertex_point3d) {
                auto &&imu_deque = feature_node->imu_deque;
                auto &&curr_feature_in_cameras = _imu_node->features_in_cameras.find(feature_id);
                bool is_feature_in_curr_imu = curr_feature_in_cameras != _imu_node->features_in_cameras.end();

                if (imu_deque.size() > 1 || (imu_deque.size() == 1 && is_feature_in_curr_imu)) {
                    // 把特征点加入到problem中
                    problem.add_vertex(feature_node->vertex_point3d);

                    // deque中的imu
                    for (unsigned long j = 0; j < imu_deque.size(); ++j) {
                        auto &&imu_node = imu_deque[j];
                        auto &&feature_in_cameras = imu_node->features_in_cameras.find(feature_id);
                        if (feature_in_cameras == imu_node->features_in_cameras.end()) {
                            continue;
                        }

                        // 构建重投影edge
                        auto edge_reproj = std::make_shared<EdgeReprojectionPoint3d>(feature_in_cameras->second[0].second);
                        edge_reproj->add_vertex(feature_node->vertex_point3d);
                        edge_reproj->add_vertex(imu_node->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[0]);

                        // 把edge加入到problem中
                        problem.add_edge(edge_reproj);
                    }

                    // 当前imu
                    if (is_feature_in_curr_imu) {
                        auto &&feature_in_cameras = _imu_node->features_in_cameras.find(feature_id);
                        if (feature_in_cameras == _imu_node->features_in_cameras.end()) {
                            continue;
                        }

                        // 构建重投影edge
                        auto edge_reproj = std::make_shared<EdgeReprojectionPoint3d>(feature_in_cameras->second[0].second);
                        edge_reproj->add_vertex(feature_node->vertex_point3d);
                        edge_reproj->add_vertex(_imu_node->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[0]);

                        // 把edge加入到problem中
                        problem.add_edge(edge_reproj);
                    }
                }
            }
        }

        // 优化
        problem.solve(10);

        // 解锁imu
        if (fixed_poses) {
            for (auto &pose : *fixed_poses) {
                pose->set_fixed(false);
            }
        }
    }

    void Estimator::local_bundle_adjustment(vector<shared_ptr<VertexPose>> *fixed_poses) {
        // VO优化
        ProblemSLAM problem;

        // fix住imu
        if (fixed_poses) {
            for (auto &pose : *fixed_poses) {
                pose->set_fixed();
            }
        }

//        // 把外参加入到problem中
//        problem.add_vertex(_vertex_ext[0]);

        // 把windows中的imu加入到problem中
        for (unsigned long i = 0; i < _windows.size(); ++i) {
            problem.add_vertex(_windows[i]->vertex_pose);
        }

        // 把当前的imu加入到problem中
        problem.add_vertex(_imu_node->vertex_pose);

        // 遍历所有特征点
        for (auto &feature_it : _feature_map) {
            unsigned long feature_id = feature_it.first;
            auto feature_node = feature_it.second;

            // 只有进行了初始化的特征点才参与计算
            if (feature_node->vertex_landmark) {
                // 构建重投影edge
                auto &&imu_deque = feature_node->imu_deque;
                auto &&curr_feature_in_cameras = _imu_node->features_in_cameras.find(feature_id);
                bool is_feature_in_curr_imu = curr_feature_in_cameras != _imu_node->features_in_cameras.end();

                if (imu_deque.size() > 1 || (imu_deque.size() == 1 && is_feature_in_curr_imu)) {
                    // 把特征点加入到problem中
                    problem.add_vertex(feature_node->vertex_landmark);

                    // host imu
                    auto &&host_imu = imu_deque.oldest();
                    auto &&host_feature_in_cameras = host_imu->features_in_cameras.find(feature_id);
                    if (host_feature_in_cameras == host_imu->features_in_cameras.end()) {
                        continue;
                    }
                    auto &&host_cameras = host_feature_in_cameras->second;
                    auto &&host_camera_id = host_cameras[0].first;
                    auto &&host_point_pixel = host_cameras[0].second;
                    host_point_pixel /= host_point_pixel.z();

                    // 其他imu
                    for (unsigned long j = 1; j < imu_deque.size(); ++j) {
                        auto &j_imu = imu_deque[j];
                        auto &&j_feature_in_cameras = j_imu->features_in_cameras.find(feature_id);
                        if (j_feature_in_cameras == j_imu->features_in_cameras.end()) {
                            continue;
                        }
                        auto &&j_cameras = j_feature_in_cameras->second;
                        auto &&j_camera_id = j_cameras[0].first;
                        auto &&j_point_pixel = j_cameras[0].second;
                        j_point_pixel /= j_point_pixel.z();

//                        auto edge_reproj = std::make_shared<EdgeReprojection>(host_point_pixel, j_point_pixel);
                        auto edge_reproj = std::make_shared<EdgeReprojectionLocal>(host_point_pixel, j_point_pixel);
                        edge_reproj->add_vertex(feature_node->vertex_landmark);
                        edge_reproj->add_vertex(host_imu->vertex_pose);
                        edge_reproj->add_vertex(j_imu->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[0]);

                        // 把edge加入到problem中
                        problem.add_edge(edge_reproj);
                    }

                    // 当前imu
                    if (is_feature_in_curr_imu) {
                        auto &&j_cameras = curr_feature_in_cameras->second;
                        auto &&j_camera_id = j_cameras[0].first;
                        auto &&j_point_pixel = j_cameras[0].second;
                        j_point_pixel /= j_point_pixel.z();

//                        auto edge_reproj = std::make_shared<EdgeReprojection>(host_point_pixel, j_point_pixel);
                        auto edge_reproj = std::make_shared<EdgeReprojectionLocal>(host_point_pixel, j_point_pixel);
                        edge_reproj->add_vertex(feature_node->vertex_landmark);
                        edge_reproj->add_vertex(host_imu->vertex_pose);
                        edge_reproj->add_vertex(_imu_node->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[0]);

                        // 把edge加入到problem中
                        problem.add_edge(edge_reproj);
                    }
                }
            }
        }

        // 优化
        problem.solve(5);

        // 解锁imu
        if (fixed_poses) {
            for (auto &pose : *fixed_poses) {
                pose->set_fixed(false);
            }
        }
    }

    bool Estimator::visual_align_to_imu() {
        // 假设ba = 0, bg = 0, 粗略的估算v, alpha, R0

        // ML

        return true;
    }

    bool Estimator::initial_structure() {
        TicToc t_sfm;
        // 通过加速度计的方差判断可观性

//        // 遍历windows
//        Vec3 aver_acc {};
//        for (unsigned long i = 1; i < _windows.size(); ++i) {
//            aver_acc += _windows[i]->imu_integration->get_delta_v() / _windows[i]->imu_integration->get_sum_dt();
//        }
//        aver_acc /= double(_windows.size() - 1);
//
//        double var = 0.;
//        for (unsigned long i = 1; i < _windows.size(); ++i) {
//            Vec3 res = _windows[i]->imu_integration->get_delta_v() / _windows[i]->imu_integration->get_sum_dt() - aver_acc;
//            var += aver_acc.squaredNorm();
//        }
//        var /= double(_windows.size() - 1);
//
//        constexpr static double var_lim = 0.25 * 0.25;
//        if (var < var_lim) {
//            std::cout << "Warning: IMU excitation not enouth" << std::endl;
////            return false;
//        }

        structure_from_motion();

        return true;
    }

    void Estimator::solve_odometry() {
        if (_windows.full() && solver_flag == NON_LINEAR) {
            TicToc t_tri;
//            backend_optimization();
        }
    }

    void Estimator::slide_window() {
        TicToc t_margin;

        std::cout << "_windows.size() = " << _windows.size() << std::endl;

        // 只有当windows满了才进行滑窗操作
        if (_windows.full()) {
            if (marginalization_flag == MARGIN_OLD) {
                std::cout << "MARGIN_OLD" << std::endl;
                ImuNode *imu_oldest {nullptr};
                _windows.pop_oldest(imu_oldest);    // 弹出windows中最老的imu

                // 遍历被删除的imu的所有特征点，在特征点的imu队列中，删除该imu
                for (auto &feature_in_cameras : imu_oldest->features_in_cameras) {
                    auto &&feature_id = feature_in_cameras.first;
                    auto &&feature_it = _feature_map.find(feature_id);
                    if (feature_it == _feature_map.end()) {
                        std::cout << "!!!!!!!! Can't find feature id in feature map when marg oldest !!!!!!!!!" << std::endl;
                        continue;
                    }
                    auto &&feature_node = feature_it->second;
                    auto &&vertex_landmark = feature_node->vertex_landmark;
                    auto &&imu_deque = feature_node->imu_deque;
                    // 应该是不需要进行判断的
//                    if (imu_deque.oldest() == imu_oldest) {
//                        imu_deque.pop_oldest();
//                    }
                    imu_deque.pop_oldest();

                    // TODO: 删除重投影edge与预积分edge

                    // 若特征点的keyframe小于2，则删除该特征点, 否则需要为特征点重新计算深度并且重新构建重投影edge,
                    if (imu_deque.size() < 2) {
                        // 在map中删除特征点
                        _feature_map.erase(feature_id);

                        // 释放特征点node的空间
                        delete feature_node;

                        if (vertex_landmark) {
                            // 在problem中删除特征点
                            _problem.remove_vertex(vertex_landmark);
                        }

                        // TODO: 在新的oldest imu中，把feature删除
                    } else {
                        if (vertex_landmark) {
                            // 曾经的host imu
                            auto &&oldest_cameras = feature_in_cameras.second;    // imu中，与feature对应的相机信息
                            auto &&oldest_imu_pose = imu_oldest->vertex_pose;   // imu的位姿
                            auto &&oldest_camera_id = oldest_cameras[0].first;  // camera的id
                            auto &&oldest_pixel_coord = oldest_cameras[0].second;    // feature在imu的左目的像素坐标

                            Vec3 p_i {oldest_imu_pose->get_parameters()(0), oldest_imu_pose->get_parameters()(1), oldest_imu_pose->get_parameters()(2)};
                            Qd q_i {oldest_imu_pose->get_parameters()(6), oldest_imu_pose->get_parameters()(3), oldest_imu_pose->get_parameters()(4), oldest_imu_pose->get_parameters()(5)};
                            Mat33 r_i {q_i.toRotationMatrix()};

                            Eigen::Vector3d t_wci_w = p_i + r_i * _t_ic[oldest_camera_id];
                            Eigen::Matrix3d r_wci = r_i * _q_ic[oldest_camera_id];

                            // 现在的host imu
                            auto &&host_imu = imu_deque.oldest();
                            auto &&host_cameras = host_imu->features_in_cameras[feature_id];
                            auto &&host_imu_pose = host_imu->vertex_pose;
                            auto &&host_camera_id = host_cameras[0].first;
                            auto &&host_pixel_coord = host_cameras[0].second;

                            Vec3 p_j {host_imu_pose->get_parameters()(0), host_imu_pose->get_parameters()(1), host_imu_pose->get_parameters()(2)};
                            Qd q_j {host_imu_pose->get_parameters()(6), host_imu_pose->get_parameters()(3), host_imu_pose->get_parameters()(4), host_imu_pose->get_parameters()(5)};
                            Mat33 r_j {q_j.toRotationMatrix()};

                            Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[host_camera_id];
                            Eigen::Matrix3d r_wcj = r_j * _q_ic[host_camera_id];

                            // 从i重投影到j
                            Vec3 p_cif_ci = oldest_pixel_coord / vertex_landmark->get_parameters()(0);
                            Vec3 p_wf_w = r_wci * p_cif_ci + t_wci_w;
                            Vec3 p_cjf_cj = r_wcj.transpose() * (p_wf_w - t_wcj_w);
                            double depth = p_cjf_cj.z();
                            if (depth < 0.1) {
                                depth = INIT_DEPTH;
                            }
                            vertex_landmark->set_parameters(Vec1(1. / depth));

                            // 计算重投影误差edge
                            // host imu的其他camera
                            for (unsigned long j = 1; j < host_cameras.size(); ++j) {
                                auto &&other_camera_id = host_cameras[j].first; // camera的id
                                auto &&other_pixel_coord = host_cameras[j].second;    // feature在imu的左目的像素坐标

                                shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                        host_pixel_coord,
                                        other_pixel_coord
                                ));
                                edge_reproj->add_vertex(vertex_landmark);
                                edge_reproj->add_vertex(host_imu_pose);
                                edge_reproj->add_vertex(host_imu_pose);
                                edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
                            }

                            // 其他imu
                            for (unsigned long i = 1; i < imu_deque.size(); ++i) {
                                auto &&other_imu = imu_deque[i];
                                auto &&other_cameras = other_imu->features_in_cameras[feature_id];
                                auto &&other_imu_pose = other_imu->vertex_pose;

                                // 遍历所有imu
                                for (unsigned j = 0; j < other_cameras.size(); ++j) {
                                    auto &&other_camera_id = other_cameras[j].first; // camera的id
                                    auto &&other_pixel_coord = other_cameras[j].second;    // feature在imu的左目的像素坐标

                                    shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                            host_pixel_coord,
                                            other_pixel_coord
                                    ));
                                    edge_reproj->add_vertex(vertex_landmark);
                                    edge_reproj->add_vertex(host_imu_pose);
                                    edge_reproj->add_vertex(other_imu_pose);
                                    edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
                                }
                            }
                        }
                    }
                }
                // 释放imu node的空间
                delete imu_oldest;
            } else {
                std::cout << "MARGIN_NEW" << std::endl;
                ImuNode *imu_newest {nullptr};
                _windows.pop_newest(imu_newest);    // 弹出windows中最新的imu

                // 遍历被删除的imu的所有特征点，在特征点的imu队列中，删除该imu
                for (auto &feature_in_cameras : imu_newest->features_in_cameras) {
                    auto &&feature_id = feature_in_cameras.first;
                    auto &&feature_it = _feature_map.find(feature_id);
                    if (feature_it == _feature_map.end()) {
                        std::cout << "!!!!!!!! Can't find feature id in feature map when marg newest !!!!!!!!!" << std::endl;
                        continue;
                    }
                    auto &&feature_node = feature_it->second;
                    auto &&vertex_landmark = feature_node->vertex_landmark;
                    auto &&imu_deque = feature_node->imu_deque;

                    // 移除feature的imu队列中的最新帧
                    imu_deque.pop_newest();

                    // TODO: 删除重投影edge与预积分edge

                    // 如果feature不在所有的imu中出现了，则需要删除feature
                    if (feature_node->imu_deque.empty()
                        && _imu_node->features_in_cameras.find(feature_id) == _imu_node->features_in_cameras.end()) {
                        // 在problem中删除特征点
                        _problem.remove_vertex(vertex_landmark);

                        // 在map中删除特征点
                        _feature_map.erase(feature_id);

                        // 释放特征点node的空间
                        delete feature_node;
                    }
                }
                // 释放imu node的空间
                delete imu_newest;
            }
        }

        // 把当前imu加入到windows中
        _windows.push_newest(_imu_node);

        // 遍历当前imu的所有特征点，把该imu加入到特征点的队列中
        for (auto &feature_in_cameras : _imu_node->features_in_cameras) {
            auto &&feature_id = feature_in_cameras.first;
            auto &&feature_it = _feature_map.find(feature_id);
            if (feature_it == _feature_map.end()) {
                std::cout << "!!!!!!! ERROR: feature of current imu not in feature map. !!!!!!!" << std::endl;
                continue;
            }
            auto &&feature_node = feature_it->second;
            feature_node->imu_deque.push_newest(_imu_node);
        }
    }
}