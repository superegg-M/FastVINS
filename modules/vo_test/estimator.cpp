//
// Created by Cain on 2024/1/11.
//

#include "estimator.h"
#include "vertex_inverse_depth.h"
#include "vertex_pose.h"
#include "vertex_motion.h"
#include "edge_reprojection.h"
#include "edge_imu.h"

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

    void Estimator::global_triangulate_with(ImuNode *imu_i, ImuNode *imu_j, bool enforce) {
        if (imu_i == imu_j) {
            return;
        }

        // imu i的位姿
        auto &&i_pose = imu_i->vertex_pose;
        Vec3 p_i {i_pose->get_parameters()(0), i_pose->get_parameters()(1), i_pose->get_parameters()(2)};
        Qd q_i {i_pose->get_parameters()(6), i_pose->get_parameters()(3), i_pose->get_parameters()(4), i_pose->get_parameters()(5)};
        Mat33 r_i {q_i.toRotationMatrix()};

        // imu j的位姿
        auto &&j_pose = imu_j->vertex_pose;
        Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
        Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
        Mat33 r_j {q_j.toRotationMatrix()};

        for (auto &feature_in_cameras : imu_i->features_in_cameras) {
            auto &&feature_it = _feature_map.find(feature_in_cameras.first);
            if (feature_it == _feature_map.end()) {
                std::cout << "Error: feature not in feature_map when running global_triangulate_with" << std::endl;
                continue;
            }
            auto &&feature_in_cameras_j = imu_j->features_in_cameras.find(feature_in_cameras.first);
            if (feature_in_cameras_j == imu_j->features_in_cameras.end()) {
                continue;
            }

            if (!feature_it->second->vertex_point3d) {
                shared_ptr<VertexPoint3d> vertex_point3d(new VertexPoint3d);
                feature_it->second->vertex_point3d = vertex_point3d;
            } else if (!enforce) {
                continue;
            }

            Mat43 A;
            Vec4 b;

//            Eigen::MatrixXd svd_A(4, 4);
//            Eigen::Matrix<double, 3, 4> P;
//            Eigen::Vector3d f;

            // imu_i的信息
            auto &&i_cameras = feature_in_cameras.second;    // imu中，与feature对应的相机信息
            auto &&i_camera_id = i_cameras[0].first;  // 左目的id
            auto &&i_pixel_coord = i_cameras[0].second;    // feature在imu的左目的像素坐标

            Eigen::Vector3d t_wci_w = p_i + r_i * _t_ic[i_camera_id];
            Eigen::Matrix3d r_wci = r_i * _q_ic[i_camera_id];

            Vec3 t1 = r_wci.transpose() * t_wci_w;
            A.row(0) = (i_pixel_coord.x() * r_wci.col(2) - r_wci.col(0)).transpose();
            A.row(1) = (i_pixel_coord.y() * r_wci.col(2) - r_wci.col(1)).transpose();
            b(0) = i_pixel_coord.x() * t1.z() - t1.x();
            b(1) = i_pixel_coord.y() * t1.z() - t1.y();

//            P.leftCols<3>() = r_wci.transpose();
//            P.rightCols<1>() = -r_wci.transpose() * t_wci_w;
//
//            f = i_pixel_coord / i_pixel_coord.z();
//            svd_A.row(0) = f[0] * P.row(2) - P.row(0);
//            svd_A.row(1) = f[1] * P.row(2) - P.row(1);

            // imu_j的信息
            auto &&j_cameras = feature_in_cameras_j->second;    // imu中，与feature对应的相机信息
            auto &&j_camera_id = j_cameras[0].first;  // 左目的id
            auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

            Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * _q_ic[j_camera_id];

            Vec3 t2 = r_wcj.transpose() * t_wcj_w;
            A.row(2) = (j_pixel_coord.x() * r_wcj.col(2) - r_wcj.col(0)).transpose();
            A.row(3) = (j_pixel_coord.y() * r_wcj.col(2) - r_wcj.col(1)).transpose();
            b(2) = j_pixel_coord.x() * t2.z() - t2.x();
            b(3) = j_pixel_coord.y() * t2.z() - t2.y();

//            P.leftCols<3>() = r_wcj.transpose();
//            P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;
//
//            f = j_pixel_coord / j_pixel_coord.z();
//            svd_A.row(2) = f[0] * P.row(2) - P.row(0);
//            svd_A.row(3) = f[1] * P.row(2) - P.row(1);

            // 最小二乘计算世界坐标
//            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
//            Vec3 point {svd_V[0] / svd_V[3], svd_V[1] / svd_V[3], svd_V[2] / svd_V[3]};
//            feature_it->second->vertex_point3d->set_parameters(point);

            Mat33 ATA = A.transpose() * A;
            Vec3 ATb = A.transpose() * b;
            auto &&ATA_ldlt = ATA.ldlt();
            Vec3 point = ATA_ldlt.solve(ATb);
            feature_it->second->vertex_point3d->set_parameters(point);

            // 检查深度
            Vec3 p_ci = r_wci.transpose() * (point - t_wci_w);
            if (p_ci.z() < 0.) {
                feature_it->second->vertex_point3d = nullptr;
                continue;
            }
            Vec3 p_cj = r_wcj.transpose() * (point - t_wcj_w);
            if (p_cj.z() < 0.) {
                feature_it->second->vertex_point3d = nullptr;
            }
        }
    }

    void Estimator::local_triangulate_with(ImuNode *imu_i, ImuNode *imu_j, bool enforce) {
        if (imu_i == imu_j) {
            return;
        }

        // imu的位姿
        auto &&i_pose = imu_i->vertex_pose;
        Vec3 p_i {i_pose->get_parameters()(0), i_pose->get_parameters()(1), i_pose->get_parameters()(2)};
        Qd q_i {i_pose->get_parameters()(6), i_pose->get_parameters()(3), i_pose->get_parameters()(4), i_pose->get_parameters()(5)};
        Mat33 r_i {q_i.toRotationMatrix()};

        // imu的位姿
        auto &&j_pose = imu_j->vertex_pose;
        Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
        Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
        Mat33 r_j {q_j.toRotationMatrix()};

        for (auto &feature_in_cameras : imu_i->features_in_cameras) {
            auto &&feature_it = _feature_map.find(feature_in_cameras.first);
            if (feature_it == _feature_map.end()) {
                std::cout << "Error: feature not in feature_map when running local_triangulate_with" << std::endl;
                continue;
            }
            auto &&feature_in_cameras_j = imu_j->features_in_cameras.find(feature_in_cameras.first);
            if (feature_in_cameras_j == imu_j->features_in_cameras.end()) {
                continue;
            }

            if (!feature_it->second->vertex_landmark) {
                shared_ptr<VertexInverseDepth> vertex_inverse_depth(new VertexInverseDepth);
                feature_it->second->vertex_landmark = vertex_inverse_depth;
            } else if (!enforce) {
                continue;
            }

            Eigen::MatrixXd svd_A(4, 4);
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Vector3d f;

            // imu_i的信息
            auto &&i_cameras = feature_in_cameras.second;    // imu中，与feature对应的相机信息
            auto &&i_camera_id = i_cameras[0].first;  // 左目的id
            auto &&i_pixel_coord = i_cameras[0].second;    // feature在imu的左目的像素坐标

            Eigen::Vector3d t_wci_w = p_i + r_i * _t_ic[i_camera_id];
            Eigen::Matrix3d r_wci = r_i * _q_ic[i_camera_id];

            P.leftCols<3>().setIdentity();
            P.rightCols<1>().setZero();

            f = i_pixel_coord / i_pixel_coord.z();
            svd_A.row(0) = f[0] * P.row(2) - P.row(0);
            svd_A.row(1) = f[1] * P.row(2) - P.row(1);

            // imu_j的信息
            auto &&j_cameras = feature_in_cameras_j->second;    // imu中，与feature对应的相机信息
            auto &&j_camera_id = j_cameras[0].first;  // 左目的id
            auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

            Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * _q_ic[j_camera_id];

            P.leftCols<3>() = r_wcj.transpose() * r_wci;
            P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w);

            f = j_pixel_coord / j_pixel_coord.z();
            svd_A.row(2) = f[0] * P.row(2) - P.row(0);
            svd_A.row(3) = f[1] * P.row(2) - P.row(1);

            // 最小二乘计算深度
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            Vec1 inverse_depth {svd_V[3] / svd_V[2]};
            feature_it->second->vertex_landmark->set_parameters(inverse_depth);

            // 检查深度
            if (inverse_depth[0] < 0.) {
                feature_it->second->vertex_landmark = nullptr;
                continue;
            }
            Vec3 p_cj = r_wcj.transpose() * (r_wci * i_pixel_coord / inverse_depth[0] + t_wci_w - t_wcj_w);
            if (p_cj.z() < 0.) {
                feature_it->second->vertex_landmark = nullptr;
            }
        }
    }

    void Estimator::global_triangulate_feature(FeatureNode* feature, bool enforce) {
        if (!feature) {
            return;
        }

        bool is_in_current = _imu_node->features_in_cameras.find(feature->id()) != _imu_node->features_in_cameras.end();

        // 若imu数小于2，则无法进行三角化
        auto &&imu_deque = feature->imu_deque;
        unsigned long num_imu = imu_deque.size() + is_in_current ? 1 : 0;
        if (num_imu < 2) {
            return;
        }

        if (!feature->vertex_point3d) {
            shared_ptr<VertexPoint3d> vertex_point3d(new VertexPoint3d);
            feature->vertex_point3d = vertex_point3d;
        } else if (!enforce) {
            return;
        }

        Eigen::MatrixXd svd_A(2 * num_imu, 4);
        Eigen::Matrix<double, 3, 4> P;
        Eigen::Vector3d f;
        for (unsigned long j = 0; j < imu_deque.size(); ++j) {
            // imu_j的信息
            auto &&imu_j = imu_deque[j];
            auto &&j_pose = imu_j->vertex_pose;   // imu的位姿
            auto &&j_feature_in_cameras = imu_j->features_in_cameras.find(feature->id());
            if (j_feature_in_cameras == imu_j->features_in_cameras.end()) {
                std::cout << "Error: feature not in features_in_cameras when running global_triangulate_feature" << std::endl;
                continue;
            }
            auto &&j_cameras = j_feature_in_cameras->second;    // imu中，与feature对应的相机信息
            auto &&j_camera_id = j_cameras[0].first;  // 左目的id
            auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

            Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
            Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
            Mat33 r_j {q_j.toRotationMatrix()};

            Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * _q_ic[j_camera_id];

            P.leftCols<3>() = r_wcj.transpose();
            P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;

            f = j_pixel_coord / j_pixel_coord.z();
            svd_A.row(2 * j) = f[0] * P.row(2) - P.row(0);
            svd_A.row(2 * j + 1) = f[1] * P.row(2) - P.row(1);
        }

        if (is_in_current) {
            unsigned long j = imu_deque.size();

            // imu_j的信息
            auto &&j_pose = _imu_node->vertex_pose;   // imu的位姿
            auto &&j_feature_in_cameras = _imu_node->features_in_cameras.find(feature->id());
            if (j_feature_in_cameras == _imu_node->features_in_cameras.end()) {
                std::cout << "Error: feature not in features_in_cameras when running global_triangulate_feature" << std::endl;
            } else {
                auto &&j_cameras = j_feature_in_cameras->second;    // imu中，与feature对应的相机信息
                auto &&j_camera_id = j_cameras[0].first;  // 左目的id
                auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

                Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
                Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
                Mat33 r_j {q_j.toRotationMatrix()};

                Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[j_camera_id];
                Eigen::Matrix3d r_wcj = r_j * _q_ic[j_camera_id];

                P.leftCols<3>() = r_wcj.transpose();
                P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;

                f = j_pixel_coord / j_pixel_coord.z();
                svd_A.row(2 * j) = f[0] * P.row(2) - P.row(0);
                svd_A.row(2 * j + 1) = f[1] * P.row(2) - P.row(1);
            }
        }

        // 最小二乘计算世界坐标
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        Vec3 point {svd_V[0] / svd_V[3], svd_V[1] / svd_V[3], svd_V[2] / svd_V[3]};
        feature->vertex_point3d->set_parameters(point);
    }

    void Estimator::local_triangulate_feature(FeatureNode* feature, bool enforce) {
        if (!feature) {
            return;
        }

        bool is_in_current = _imu_node->features_in_cameras.find(feature->id()) != _imu_node->features_in_cameras.end();

        // 若imu数小于2，则无法进行三角化
        auto &&imu_deque = feature->imu_deque;
        unsigned long num_imu = imu_deque.size() + is_in_current ? 1 : 0;
        if (num_imu < 2) {
            return;
        }

        if (!feature->vertex_landmark) {
            shared_ptr<VertexInverseDepth> vertex_inverse_depth(new VertexInverseDepth);
            feature->vertex_landmark = vertex_inverse_depth;
        } else if (!enforce) {
            return;
        }

        Eigen::MatrixXd svd_A(2 * num_imu, 4);
        Eigen::Matrix<double, 3, 4> P;
        Eigen::Vector3d f;

        // imu_i的信息
        auto &&imu_i = imu_deque.oldest();
        auto &&i_pose = imu_i->vertex_pose;   // imu的位姿
        auto &&i_feature_in_cameras = imu_i->features_in_cameras.find(feature->id());
        if (i_feature_in_cameras == imu_i->features_in_cameras.end()) {
            std::cout << "Error: feature not in features_in_cameras when running local_triangulate_feature" << std::endl;
            return;
        }
        auto &&i_cameras = i_feature_in_cameras->second;    // imu中，与feature对应的相机信息
        auto &&i_camera_id = i_cameras[0].first;  // 左目的id
        auto &&i_pixel_coord = i_cameras[0].second;    // feature在imu的左目的像素坐标

        Vec3 p_i {i_pose->get_parameters()(0), i_pose->get_parameters()(1), i_pose->get_parameters()(2)};
        Qd q_i {i_pose->get_parameters()(6), i_pose->get_parameters()(3), i_pose->get_parameters()(4), i_pose->get_parameters()(5)};
        Mat33 r_i {q_i.toRotationMatrix()};

        Eigen::Vector3d t_wci_w = p_i + r_i * _t_ic[i_camera_id];
        Eigen::Matrix3d r_wci = r_i * _q_ic[i_camera_id];

        P.leftCols<3>().setIdentity();
        P.rightCols<1>().setZero();

        f = i_pixel_coord / i_pixel_coord.z();
        svd_A.row(0) = f[0] * P.row(2) - P.row(0);
        svd_A.row(1) = f[1] * P.row(2) - P.row(1);

        for (unsigned long j = 1; j < imu_deque.size(); ++j) {
            // imu_j的信息
            auto &&imu_j = imu_deque[j];
            auto &&j_pose = imu_j->vertex_pose;   // imu的位姿
            auto &&j_feature_in_cameras = imu_j->features_in_cameras.find(feature->id());
            if (j_feature_in_cameras == imu_j->features_in_cameras.end()) {
                std::cout << "Error: feature not in features_in_cameras when running global_triangulate_feature" << std::endl;
                continue;
            }
            auto &&j_cameras = j_feature_in_cameras->second;    // imu中，与feature对应的相机信息
            auto &&j_camera_id = j_cameras[0].first;  // 左目的id
            auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

            Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
            Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
            Mat33 r_j {q_j.toRotationMatrix()};

            Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * _q_ic[j_camera_id];

            P.leftCols<3>() = r_wcj.transpose() * r_wci;
            P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w);

            f = j_pixel_coord / j_pixel_coord.z();
            svd_A.row(2 * j) = f[0] * P.row(2) - P.row(0);
            svd_A.row(2 * j + 1) = f[1] * P.row(2) - P.row(1);
        }

        if (is_in_current) {
            unsigned long j = imu_deque.size();

            // imu_j的信息
            auto &&j_pose = _imu_node->vertex_pose;   // imu的位姿
            auto &&j_feature_in_cameras = _imu_node->features_in_cameras.find(feature->id());
            if (j_feature_in_cameras == _imu_node->features_in_cameras.end()) {
                std::cout << "Error: feature not in features_in_cameras when running global_triangulate_feature" << std::endl;
            } else {
                auto &&j_cameras = j_feature_in_cameras->second;    // imu中，与feature对应的相机信息
                auto &&j_camera_id = j_cameras[0].first;  // 左目的id
                auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

                Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
                Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
                Mat33 r_j {q_j.toRotationMatrix()};

                Eigen::Vector3d t_wcj_w = p_j + r_j * _t_ic[j_camera_id];
                Eigen::Matrix3d r_wcj = r_j * _q_ic[j_camera_id];

                P.leftCols<3>() = r_wcj.transpose() * r_wci;
                P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w);

                f = j_pixel_coord / j_pixel_coord.z();
                svd_A.row(2 * j) = f[0] * P.row(2) - P.row(0);
                svd_A.row(2 * j + 1) = f[1] * P.row(2) - P.row(1);
            }
        }

        // 最小二乘计算深度
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        Vec1 inverse_depth {svd_V[3] / svd_V[2]};
        feature->vertex_landmark->set_parameters(inverse_depth);
    }

    void Estimator::pnp(ImuNode *imu_i, Qd *q_wi_init, Vec3 *t_wi_init) {
        Qd q_wi;
        Vec3 t_wi;
        if (q_wi_init) {
            q_wi = *q_wi_init;
        } else {
            q_wi.setIdentity();
        }
        if (t_wi_init) {
            t_wi = *t_wi_init;
        } else {
            t_wi.setZero();
        }

        // 对imu的pose进行初始化
        Vec7 pose;
        pose << t_wi.x(), t_wi.y(), t_wi.z(),
                q_wi.x(), q_wi.y(), q_wi.z(), q_wi.w();
        imu_i->vertex_pose->set_parameters(pose);

        // 相机外参
        auto &&pose_ext = _vertex_ext[0]->get_parameters();
        Vec3 t_ic {pose_ext(0), pose_ext(1), pose_ext(2)};
        Qd q_ic {pose_ext(6), pose_ext(3), pose_ext(4), pose_ext(5)};
//        Vec3 t_ic = _t_ic[0];
//        Qd q_ic = _q_ic[0];

        Problem problem;
        problem.add_vertex(imu_i->vertex_pose); // 加入imu的位姿
        for (auto &feature_in_cameras : imu_i->features_in_cameras) {
            auto &&feature_it = _feature_map.find(feature_in_cameras.first);
            if (feature_it == _feature_map.end()) {
                std::cout << "Error: feature not in feature_map when running pnp" << std::endl;
                continue;
            }
            if (feature_it->second->vertex_point3d) {
                auto &&point_pixel = feature_in_cameras.second[0].second;
                Vec3 point_world = feature_it->second->vertex_point3d->get_parameters();

                // 重投影edge
                shared_ptr<EdgePnP> edge_pnp(new EdgePnP(point_pixel, point_world));
                edge_pnp->set_translation_imu_from_camera(q_ic, t_ic);
                edge_pnp->add_vertex(imu_i->vertex_pose);

                problem.add_edge(edge_pnp);
            }
        }
        problem.solve(10);
    }

    void Estimator::epnp(ImuNode *imu_i) {
//        // 读取3d, 2d点
//        vector<Vec3> p_w, uv;
//        p_w.reserve(imu_i->features_in_cameras.size());
//        uv.reserve(imu_i->features_in_cameras.size());
//        for (auto &feature_in_cameras : imu_i->features_in_cameras) {
//            auto &&feature_it = _feature_map.find(feature_in_cameras.first);
//            if (feature_it == _feature_map.end()) {
//                std::cout << "Error: feature not in feature_map when running pnp" << std::endl;
//                continue;
//            }
//            if (feature_it->second->vertex_point3d) {
//                p_w.emplace_back(feature_it->second->vertex_point3d->get_parameters());
//                uv.emplace_back(Vec2(feature_in_cameras.second[0].second.x(), feature_in_cameras.second[0].second.y()))
//            }
//        }
//
//        // 计算c_w0
//        Vec3 c_w[4];
//        c_w[0].setZero();
//        unsigned long num_points = p_w.size();
//        for (unsigned long k = 0; k < num_points; ++k) {
//            c_w[0] += p_w[k];
//        }
//        c_w[0] /= double(num_points);
//
//        // 计算c_w1, c_w2, c_w3
//        double sqrt_n = sqrt(double(num_points));
//        MatXX A;
//        A.resize(num_points, 3);
//        for (unsigned long k = 0; k < num_points; ++k) {
//            A.row(k) = (p_w[k] - c_w[0]).transpose();
//        }
//        auto &&A_svd = A.jacobiSvd(Eigen::ComputeThinV);
//        Vec3 s = A_svd.singularValues();
//        Mat33 V = A_svd.matrixV();
//        c_w[1] = c_w[0] + s[0] / sqrt_n * V.col(0);
//        c_w[2] = c_w[0] + s[1] / sqrt_n * V.col(1);
//        c_w[3] = c_w[0] + s[2] / sqrt_n * V.col(2);
//
//        // 控制点到世界坐标的转换矩阵
//        Mat33 Transf;
//        Transf.col(0) = c_w[1] - c_w[0];
//        Transf.col(1) = c_w[2] - c_w[0];
//        Transf.col(2) = c_w[3] - c_w[0];
//
//        // 世界坐标到控制点的转换矩阵
//        auto Transf_lup = Transf.fullPivLu();
//
//        // 特征点在控制点坐标系中的坐标
//        vector<Vec4> alpha;
//        alpha.resize(num_points);
//        for (unsigned long k = 0; k < num_points; ++k) {
//            alpha[k].segment<3>(1) = Transf_lup.solve(p_w[k] - c_w[0]);
//            alpha[k][0] = 1. - alpha[k][1] - alpha[k][2] - alpha[k][3];
//        }
//
//        // M矩阵的计算
//        MatXX M;
//        M.resize(2 * num_points, 12);
//        for (unsigned long k = 0; k < num_points; ++k) {
//            M.row(2 * k) << alpha[k][0], 0., -uv[k][0] * alpha[k][0],
//                            alpha[k][1], 0., -uv[k][0] * alpha[k][1],
//                            alpha[k][2], 0., -uv[k][0] * alpha[k][2],
//                            alpha[k][3], 0., -uv[k][0] * alpha[k][3];
//            M.row(2 * k + 1) << 0., alpha[k][0], -uv[k][1] * alpha[k][0],
//                                0., alpha[k][1], -uv[k][1] * alpha[k][1],
//                                0., alpha[k][2], -uv[k][1] * alpha[k][2],
//                                0., alpha[k][3], -uv[k][1] * alpha[k][3];
//
//        }
//
//        // SVD算零空间
//        auto M_svd = M.jacobiSvd(Eigen::ComputeThinV);
//        auto &&M_kernel = M_svd.matrixV();
//
//        Vec3 dv[4][6];
//        for (unsigned int i = 0; i < 4; ++i) {
//            unsigned int j = 11 - i;
//            dv[i][0] = M_kernel.col(j).segment<3>(0) - M_kernel.col(11).segment<3>(3);
//            dv[i][1] = M_kernel.col(j).segment<3>(0) - M_kernel.col(11).segment<3>(6);
//            dv[i][2] = M_kernel.col(j).segment<3>(0) - M_kernel.col(11).segment<3>(9);
//            dv[i][3] = M_kernel.col(j).segment<3>(3) - M_kernel.col(11).segment<3>(6);
//            dv[i][4] = M_kernel.col(j).segment<3>(3) - M_kernel.col(11).segment<3>(9);
//            dv[i][5] = M_kernel.col(j).segment<3>(6) - M_kernel.col(11).segment<3>(9);
//        }
//
//        Vec3 dc[6];
//        dc[0] = c_w[0] - c_w[1];
//        dc[1] = c_w[0] - c_w[2];
//        dc[2] = c_w[0] - c_w[3];
//        dc[3] = c_w[1] - c_w[2];
//        dc[4] = c_w[1] - c_w[3];
//        dc[5] = c_w[2] - c_w[3];
//
//        // N = 1
//        double num = 0.;
//        double den = 0.;
//        for (unsigned int i = 0; i < 6; ++i) {
//            num += dv[0][i].dot(dc[i]);
//            den += dv[0][i].squaredNorm();
//        }
//        double b_N1 = num / den;
//
//        // N = 2
//
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

        // 把外参加入到problem中
        problem.add_vertex(_vertex_ext[0]);

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

                        auto edge_reproj = std::make_shared<EdgeReprojection>(host_point_pixel, j_point_pixel);
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

                        auto edge_reproj = std::make_shared<EdgeReprojection>(host_point_pixel, j_point_pixel);
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
        problem.solve(10);

        // 解锁imu
        if (fixed_poses) {
            for (auto &pose : *fixed_poses) {
                pose->set_fixed(false);
            }
        }
    }

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
            pnp(imu_j, &q_wj, &t_wj);

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
            pnp(imu_j, &q_wj, &t_wj);

            // 三角化
            global_triangulate_with(imu_j, imu_i);
        }

//        // 遍历所有特征点, 对没有赋值的特征点进行三角化
//        for (auto &feature_it : _feature_map) {
//            global_triangulate_feature(feature_it.second);
//        }

        // 固定住不参与优化的点
        vector<shared_ptr<VertexPose>> fixed_poses;
        fixed_poses.emplace_back(_vertex_ext[0]);
        fixed_poses.emplace_back(_windows[imu_index]->vertex_pose);
        fixed_poses.emplace_back(_imu_node->vertex_pose);

//        // Global Bundle Adjustment
//        global_bundle_adjustment(&fixed_poses);
//        // 把特征点从global转为local
//        for (auto &feature_it : _feature_map) {
//            auto feature_node = feature_it.second;
//            feature_node->from_global_to_local(_q_ic, _t_ic);
//        }

        // 把特征点从global转为local
        for (auto &feature_it : _feature_map) {
            auto feature_node = feature_it.second;
            feature_node->from_global_to_local(_q_ic, _t_ic);
        }
        // Local Bundle Adjustment
        local_bundle_adjustment(&fixed_poses);

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

    bool Estimator::compute_essential_matrix(Mat33 &R, Vec3 &t, ImuNode *imu_i, ImuNode *imu_j,
                                             bool is_init_landmark, unsigned int max_iters) {
        constexpr static double th_e2 = 3.841;
        constexpr static double th_score = 5.991;
        constexpr static unsigned long th_count = 20;

        unsigned long max_num_points = max(imu_i->features_in_cameras.size(), imu_j->features_in_cameras.size());
        vector<pair<const Vec3*, const Vec3*>> match_pairs;
        match_pairs.reserve(max_num_points);
        vector<unsigned long> feature_ids;
        feature_ids.reserve(max_num_points);

        // 找出匹配对
        for (auto &feature_in_cameras : imu_i->features_in_cameras) {
            unsigned long feature_id = feature_in_cameras.first;
            auto &&it = imu_j->features_in_cameras.find(feature_id);
            if (it == imu_j->features_in_cameras.end()) {
                continue;
            }
            match_pairs.emplace_back(&feature_in_cameras.second[0].second, &it->second[0].second);
            feature_ids.emplace_back(feature_id);
        }

        // 匹配对必须大于一定数量
        unsigned long num_points = match_pairs.size();
        if (num_points < th_count) {
            return false;
        }

        // 计算平均视差
        double average_parallax = 0.;
        for (auto &match_pair : match_pairs) {
            double du = match_pair.first->x() - match_pair.second->x();
            double dv = match_pair.first->y() - match_pair.second->y();
            average_parallax += max(abs(du), abs(dv));
        }
        average_parallax /= double(num_points);
        std::cout << "average_parallax = " << average_parallax << std::endl;

        // 平均视差必须大于一定值
        if (average_parallax * 460. < 30.) {
            return false;
        }

        // 归一化变换参数
        Mat33 Ti, Tj;
        double meas_x_i = 0., meas_y_i = 0.;
        double dev_x_i = 0., dev_y_i = 0.;
        double meas_x_j = 0., meas_y_j = 0.;
        double dev_x_j = 0., dev_y_j = 0.;

        // 计算均值
        for (auto &match_pair : match_pairs) {
            meas_x_i += match_pair.first->x();
            meas_y_i += match_pair.first->y();
            meas_x_j += match_pair.second->x();
            meas_y_j += match_pair.second->y();
        }
        meas_x_i /= double(num_points);
        meas_y_i /= double(num_points);
        meas_x_j /= double(num_points);
        meas_y_j /= double(num_points);

        // 计算Dev
        for (auto &match_pair : match_pairs) {
            dev_x_i += abs(match_pair.first->x() - meas_x_i);
            dev_y_i += abs(match_pair.first->y() - meas_y_i);
            dev_x_j += abs(match_pair.second->x() - meas_x_j);
            dev_y_j += abs(match_pair.second->y() - meas_y_j);
        }
        dev_x_i /= double(num_points);
        dev_y_i /= double(num_points);
        dev_x_j /= double(num_points);
        dev_y_j /= double(num_points);

        Ti << 1. / dev_x_i, 0., -meas_x_i / dev_x_i,
              0., 1. / dev_y_i, -meas_y_i / dev_y_i,
              0., 0., 1.;
        Tj << 1. / dev_x_j, 0., -meas_x_j / dev_x_j,
              0., 1. / dev_y_j, -meas_y_j / dev_y_j,
              0., 0., 1.;

        // 归一化后的点
        vector<pair<Vec2, Vec2>> normal_match_pairs(num_points);
        for (unsigned long k = 0; k < num_points; ++k) {
            normal_match_pairs[k].first.x() = (match_pairs[k].first->x() - meas_x_i) / dev_x_i;
            normal_match_pairs[k].first.y() = (match_pairs[k].first->y() - meas_y_i) / dev_y_i;
            normal_match_pairs[k].second.x() = (match_pairs[k].second->x() - meas_x_j) / dev_x_j;
            normal_match_pairs[k].second.y() = (match_pairs[k].second->y() - meas_y_j) / dev_y_j;
        }

        // 构造随机index batch
        std::random_device rd;
        std::mt19937 gen(rd());
        vector<array<unsigned long, 8>> point_indices_set(max_iters);  // TODO: 设为静态变量

        array<unsigned long, 8> local_index_map {};
        vector<unsigned long> global_index_map(num_points);
        for (unsigned long k = 0; k < num_points; ++k) {
            global_index_map[k] = k;
        }

        for (unsigned int n = 0; n < max_iters; ++n) {
            for (unsigned int k = 0; k < 8; ++k) {
                std::uniform_int_distribution<unsigned int> dist(0, global_index_map.size() - 1);
                unsigned int rand_i = dist(gen);
                auto index = global_index_map[rand_i];
                point_indices_set[n][k] = index;
                local_index_map[k] = index;

                global_index_map[rand_i] = global_index_map.back();
                global_index_map.pop_back();
            }

            for (unsigned int k = 0; k < 8; ++k) {
                global_index_map.emplace_back(local_index_map[k]);
            }
        }

        // RANSAC: 计算本质矩阵
        Mat33 best_E;
        double best_score = 0.;
        unsigned int best_iter = 0;
        unsigned long num_outliners = 0;
        vector<bool> is_outliners(num_points, false);
        for (unsigned int n = 0; n < max_iters; ++n) {
            // 八点法算E
            Mat89 D;
            for (unsigned int k = 0; k < 8; ++k) {
                unsigned int index = point_indices_set[n][k];
                double u1 = normal_match_pairs[index].first.x();
                double v1 = normal_match_pairs[index].first.y();
                double u2 = normal_match_pairs[index].second.x();
                double v2 = normal_match_pairs[index].second.y();
                D(k, 0) = u1 * u2;
                D(k, 1) = u1 * v2;
                D(k, 2) = u1;
                D(k, 3) = v1 * u2;
                D(k, 4) = v1 * v2;
                D(k, 5) = v1;
                D(k, 6) = u2;
                D(k, 7) = v2;
                D(k, 8) = 1.;
            }
            Eigen::JacobiSVD<Mat89> D_svd(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Vec9 e = D_svd.matrixV().col(8);
            Mat33 E_raw;
            E_raw << e(0), e(1), e(2),
                     e(3), e(4), e(5),
                     e(6), e(7), e(8);
            Eigen::JacobiSVD<Mat33> E_svd(E_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Vec3 s = E_svd.singularValues();
            s(0) = 0.5 * (s(0) + s(1));
            s(1) = s(0);
            s(2) = 0.;
            Mat33 E = E_svd.matrixU() * s.asDiagonal() * E_svd.matrixV().transpose();
            double e00 = E(0, 0), e01 = E(0, 1), e02 = E(0, 2),
                   e10 = E(1, 0), e11 = E(1, 1), e12 = E(1, 2),
                   e20 = E(2, 0), e21 = E(2, 1), e22 = E(2, 2);

            // 计算分数
   ;        double score = 0.;
            for (unsigned long k = 0; k < num_points; ++k) {
                bool is_outliner = false;

                double u1 = normal_match_pairs[k].first.x();
                double v1 = normal_match_pairs[k].first.y();
                double u2 = normal_match_pairs[k].second.x();
                double v2 = normal_match_pairs[k].second.y();

                double a = u1 * e00 + v1 * e10 + e20;
                double b = u1 * e01 + v1 * e11 + e21;
                double c = u1 * e02 + v1 * e12 + e22;
                double num = a * u2 + b * v2 + c;
                double e2 = num * num / (a * a + b * b);
                if (e2 > th_e2) {
                    is_outliner = true;
                } else {
                    score += th_score - e2;
                }

                a = u2 * e00 + v2 * e01 + e02;
                b = u2 * e10 + v2 * e11 + e12;
                c = u2 * e20 + v2 * e21 + e22;
                num = u1 * a + v1 * b + c;
                e2 = num * num / (a * a + b * b);
                if (e2 > th_e2) {
                    is_outliner = true;
                } else {
                    score += th_score - e2;
                }

                is_outliners[k] = is_outliner;
                if (is_outliner) {
                    ++num_outliners;
                }
            }

            if (score > best_score) {
                best_score = score;
                best_iter = n;
                best_E = E;
            }
        }

        // outliner的点过多
        if (10 * num_outliners > 5 * num_points) {
            return false;
        }

        // 从E中还原出R, t
        best_E = Ti.transpose() * best_E * Tj;
        Eigen::JacobiSVD<Mat33> E_svd(best_E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat33 V = E_svd.matrixV();
        Mat33 U1 = E_svd.matrixU();

        Vec3 t1 = U1.col(2);
        t1 = t1 / t1.norm();
        Vec3 t2 = -t1;

        U1.col(0).swap(U1.col(1));
        Mat33 U2 = U1;
        U1.col(1) *= -1.;
        U2.col(0) *= -1.;

        Mat33 R1 = U1 * V.transpose();
        Mat33 R2 = U2 * V.transpose();
        if (R1.determinant() < 0.) {
            R1 = -R1;
        }
        if (R2.determinant() < 0.) {
            R2 = -R2;
        }

        // 进行三角化，通过深度筛选出正确的R, t
        auto tri = [&](const Vec3 *point_i, const Vec3 *point_j, const Mat33 &R, const Vec3 &t, Vec3 &p) -> bool {
            Vec3 RTt = R.transpose() * t;
            Mat43 A;
            A.row(0) << 1., 0., -point_i->x();
            A.row(1) << 0., 1., -point_i->y();
            A.row(2) = R.col(0).transpose() - point_j->x() * R.col(2).transpose();
            A.row(3) = R.col(1).transpose() - point_j->y() * R.col(2).transpose();
            Vec4 b;
            b << 0., 0., RTt[0] - RTt[2] * point_j->x(), RTt[1] - RTt[2] * point_j->y();
            Mat33 ATA = A.transpose() * A;
            Vec3 ATb = A.transpose() * b;
            auto &&ATA_ldlt = ATA.ldlt();
            if (ATA_ldlt.info() == Eigen::Success) {
                p = ATA_ldlt.solve(ATb);
                return true;
            } else {
                return false;
            }
        };

        auto tri_all_points = [&](const Mat33 &R, const Vec3 &t, vector<pair<bool, Vec3>> &points) -> unsigned long {
            unsigned long succeed_count = 0;
            for (unsigned long k = 0; k < num_points; ++k) {
                if (is_outliners[k]) {
                    continue;
                }

                points[k].first = tri(match_pairs[k].first, match_pairs[k].second, R, t, points[k].second);
                if (!points[k].first) {
                    continue;
                }

                points[k].first = points[k].second[2] > 0.;
                if (!points[k].first) {
                    continue;
                }

                Vec3 pj = R.transpose() * (points[k].second - t);
                points[k].first = pj[2] > 0.;
                if (!points[k].first) {
                    continue;
                }

                ++succeed_count;
            }
            return succeed_count;
        };

        vector<pair<bool, Vec3>> points_w[4];
        unsigned long succeed_points[4];
        points_w[0].resize(num_points);
        points_w[1].resize(num_points);
        points_w[2].resize(num_points);
        points_w[3].resize(num_points);

        succeed_points[0] = tri_all_points(R1, t1, points_w[0]);
        succeed_points[1] = tri_all_points(R1, t2, points_w[1]);
        succeed_points[2] = tri_all_points(R2, t1, points_w[2]);
        succeed_points[3] = tri_all_points(R2, t2, points_w[3]);

        unsigned long max_succeed_points = max(succeed_points[0], max(succeed_points[1], max(succeed_points[2], succeed_points[3])));
        unsigned long min_succeed_points = 9 * (num_points - num_outliners) / 10; // 至少要超过90%的点成功被三角化

        if (max_succeed_points < min_succeed_points) {
            return false;
        }

        unsigned long lim_succeed_points = 7 * max_succeed_points / 10;
        unsigned long num_similar = 0;  // 不允许超过1组解使得70%的点都能三角化
        if (succeed_points[0] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[1] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[2] > lim_succeed_points) {
            ++num_similar;
        }
        if (succeed_points[3] > lim_succeed_points) {
            ++num_similar;
        }
        if (num_similar > 1) {
            return false;
        }

        unsigned int which_case;
        if (succeed_points[0] == max_succeed_points) {
            which_case = 0;
            R = R1;
            t = t1;
        } else if (succeed_points[1] == max_succeed_points) {
            which_case = 1;
            R = R1;
            t = t2;
        } else if (succeed_points[2] == max_succeed_points) {
            which_case = 2;
            R = R2;
            t = t1;
        } else {
            which_case = 3;
            R = R2;
            t = t2;
        }

        // 转到imu坐标系
        Qd q12;
        Vec3 t12;
        q12 = _q_ic[0] * R * _q_ic[0].inverse();
        t12 = _q_ic[0] * t - q12 * _t_ic[0] + _t_ic[0];

        R = q12;
        t = t12;

        // 把三角化的结果赋值给landmark
        if (is_init_landmark) {
            for (unsigned long k = 0; k < num_points; ++k) {
                // 没有outliners以及深度为正的点才会进行赋值
                if (!is_outliners[k] && points_w[which_case][k].first) {
                    auto &&feature_it = _feature_map.find(feature_ids[k]);
                    if (feature_it == _feature_map.end()) {
                        continue;
                    }

                    // 转到imu系
                    Vec3 p = _q_ic[0] * points_w[which_case][k].second + _t_ic[0];

                    feature_it->second->vertex_point3d = std::make_shared<VertexPoint3d>();
                    feature_it->second->vertex_point3d->set_parameters(p);
                }
            }
        }

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
//        compute_essential_matrix(_windows[5], _imu_node);
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