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

#include <iostream>
#include <ostream>
#include <fstream>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    Estimator::Estimator() : f_manager{Rs} {
        for (auto & pre_integration : pre_integrations) {
            pre_integration = nullptr;
        }
        for(auto &it: all_image_frame) {
            it.second.pre_integration = nullptr;
        }
        tmp_pre_integration = nullptr;

        clear_state();
    }

    void Estimator::set_parameter() {
        for (int i = 0; i < NUM_OF_CAM; i++) {
            tic[i] = TIC[i];
            ric[i] = RIC[i];
            // cout << "1 Estimator::setParameter tic: " << tic[i].transpose()
            //     << " ric: " << ric[i] << endl;
        }
        cout << "1 Estimator::setParameter FOCAL_LENGTH: " << FOCAL_LENGTH << endl;
        f_manager.setRic(ric);
        project_sqrt_info_ = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
        td = TD;
    }

    void Estimator::clear_state() {
        for (unsigned i = 0; i < WINDOW_SIZE + 1; i++) {
            Rs[i].setIdentity();
            Ps[i].setZero();
            Vs[i].setZero();
            Bas[i].setZero();
            Bgs[i].setZero();
            dt_buf[i].clear();
            linear_acceleration_buf[i].clear();
            angular_velocity_buf[i].clear();

            if (pre_integrations[i] != nullptr)
                delete pre_integrations[i];
            pre_integrations[i] = nullptr;
        }

        for (int i = 0; i < NUM_OF_CAM; i++) {
            tic[i] = Vector3d::Zero();
            ric[i] = Matrix3d::Identity();
        }

        for (auto &it : all_image_frame) {
            if (it.second.pre_integration != nullptr) {
                delete it.second.pre_integration;
                it.second.pre_integration = nullptr;
            }
        }

        solver_flag = INITIAL;
        first_imu = false,
        sum_of_back = 0;
        sum_of_front = 0;
        frame_count = 0;
        solver_flag = INITIAL;
        initial_timestamp = 0;
        all_image_frame.clear();
        td = TD;

        if (tmp_pre_integration != nullptr)
            delete tmp_pre_integration;

        tmp_pre_integration = nullptr;

        last_marginalization_parameter_blocks.clear();

        f_manager.clear_feature();

        failure_occur = false;
        relocalization_info = false;

        drift_correct_r = Matrix3d::Identity();
        drift_correct_t = Vector3d::Zero();
    }

    void Estimator::process_imu(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity) {
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
        Vec3 acc_corr = 0.5 * (acc0_corr + acc1_corr) - g;
        _state.p += (0.5 * acc_corr * dt + _state.v) * dt;
        _state.v += acc_corr * dt;

        _acc_latest = linear_acceleration;
        _gyro_latest = angular_velocity;
    }

    void Estimator::process_image(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header) {
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
        if (!_windows.empty() && _imu_node->imu_integration->get_sum_dt() < 10.) {
            shared_ptr<EdgeImu> edge_imu(new EdgeImu(*_imu_node->imu_integration));
            edge_imu->set_vertex(_windows.newest()->vertex_pose, 0);
            edge_imu->set_vertex(_windows.newest()->vertex_motion, 1);
            edge_imu->set_vertex(_imu_node->vertex_pose, 2);
            edge_imu->set_vertex(_imu_node->vertex_motion, 3);
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
                _feature_map.emplace(make_pair(feature_id, feature_node));
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

                    Eigen::Vector3d t_wci_w = p_i + r_i * tic[host_camera_id];
                    Eigen::Matrix3d r_wci = r_i * ric[host_camera_id];

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

                        Eigen::Vector3d t_wcj_w = p_i + r_i * tic[other_camera_id];
                        Eigen::Matrix3d r_wcj = r_i * ric[other_camera_id];
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
                        edge_reproj->set_vertex(vertex_landmark, 0);
                        edge_reproj->set_vertex(host_imu_pose, 1);
                        edge_reproj->set_vertex(host_imu_pose, 2);
                        edge_reproj->set_vertex(_vertex_ext[other_camera_id], 3);
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

                            Eigen::Vector3d t_wcj_w = p_j + r_j * tic[other_camera_id];
                            Eigen::Matrix3d r_wcj = r_j * ric[other_camera_id];
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
                            edge_reproj->set_vertex(vertex_landmark, 0);
                            edge_reproj->set_vertex(host_imu_pose, 1);
                            edge_reproj->set_vertex(other_imu_pose, 2);
                            edge_reproj->set_vertex(_vertex_ext[other_camera_id], 3);
                        }
                    }

                    // 当前的imu
                    auto state_r = _state.q.toRotationMatrix();
                    for (auto &camera : cameras) {
                        unsigned long current_camera_id = camera.first;  // camera的id
                        Vec3 current_pixel_coord = {camera.second.x(), camera.second.y(), camera.second.z()};    // feature在imu的左目的像素坐标

                        Eigen::Vector3d t_wcj_w = _state.p + state_r * tic[current_camera_id];
                        Eigen::Matrix3d r_wcj = state_r * ric[current_camera_id];
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
                        edge_reproj->set_vertex(feature_node->vertex_landmark, 0);
                        edge_reproj->set_vertex(host_imu_pose, 1);
                        edge_reproj->set_vertex(_imu_node->vertex_pose, 2);
                        edge_reproj->set_vertex(_vertex_ext[current_camera_id], 3);
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
                        edge_reproj->set_vertex(feature_node->vertex_landmark, 0);
                        edge_reproj->set_vertex(host_imu_pose, 1);
                        edge_reproj->set_vertex(_imu_node->vertex_pose, 2);
                        edge_reproj->set_vertex(_vertex_ext[other_camera_id], 3);
                    }
                }
            }

            /*
             * 计算每个特征点的视差，用于判断是否为keyframe:
             * 若windows中的newest frame是key frame, 则和newest frame计算视差
             * 否则, 和2nd newest frame计算视差，因为2nd newest必为key frame
             * */
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

        Headers[frame_count] = header;

        ImageFrame imageframe(image, header);
        imageframe.pre_integration = tmp_pre_integration;
        all_image_frame.insert(make_pair(header, imageframe));
//        tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};



        if (ESTIMATE_EXTRINSIC == 2)
        {
            cout << "calibrating extrinsic param, rotation movement is needed" << endl;
            if (frame_count != 0)
            {
                vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
                Matrix3d calib_ric;
                if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
                {
                    // ROS_WARN("initial extrinsic rotation calib success");
                    // ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                    //    << calib_ric);
                    ric[0] = calib_ric;
                    RIC[0] = calib_ric;
                    ESTIMATE_EXTRINSIC = 1;
                }
            }
        }

        if (solver_flag == INITIAL) {
            if (_windows.full()) {   // WINDOW已经装满了, 且还有camera frame
                bool is_initialized = false;
                if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1) {
                    // cout << "1 initialStructure" << endl;
//                    is_initialized = initialStructure();
                    is_initialized = initial_structure();
                    initial_timestamp = header;
                }
                if (is_initialized) {
                    cout << "Initialization finish!" << endl;

                    // 初始化后进行非线性优化
                    solver_flag = NON_LINEAR;
                    solve_odometry();
                    slide_window();
                    f_manager.removeFailures();

                    _state.p = _imu_node->get_p();
                    _state.q = _imu_node->get_q().normalized();
                    _state.v = _imu_node->get_v();
                    _state.ba = _imu_node->get_ba();
                    _state.bg = _imu_node->get_bg();
                } else {
                    slide_window();
                }
            }
//            else {
//                ++frame_count;
//            }
        } else {
            TicToc t_solve;
            solve_odometry();
            //ROS_DEBUG("solver costs: %fms", t_solve.toc());

            if (failure_detection()) {
                // ROS_WARN("failure detection!");
                failure_occur = 1;
                clear_state();
                set_parameter();
                // ROS_WARN("system reboot!");
                return;
            }

            TicToc t_margin;
            slide_window();
            f_manager.remove_failures();

            // prepare output of VINS
            key_poses.clear();
            for (int i = 0; i <= WINDOW_SIZE; i++)
                key_poses.push_back(Ps[i]);

            _state.p = _imu_node->get_p();
            _state.q = _imu_node->get_q().normalized();
            _state.v = _imu_node->get_v();
            _state.ba = _imu_node->get_ba();
            _state.bg = _imu_node->get_bg();
        }
    }

    void Estimator::global_triangulate_with(ImuNode *imu_i, ImuNode *imu_j, bool enforce) {
        if (imu_i == imu_j) {
            return;
        }

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

            Eigen::MatrixXd svd_A(4, 4);
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Vector3d f;

            // imu_i的信息
            auto &&i_pose = imu_i->vertex_pose;   // imu的位姿
            auto &&i_cameras = feature_in_cameras.second;    // imu中，与feature对应的相机信息
            auto &&i_camera_id = i_cameras[0].first;  // 左目的id
            auto &&i_pixel_coord = i_cameras[0].second;    // feature在imu的左目的像素坐标

            Vec3 p_i {i_pose->get_parameters()(0), i_pose->get_parameters()(1), i_pose->get_parameters()(2)};
            Qd q_i {i_pose->get_parameters()(6), i_pose->get_parameters()(3), i_pose->get_parameters()(4), i_pose->get_parameters()(5)};
            Mat33 r_i {q_i.toRotationMatrix()};

            Eigen::Vector3d t_wci_w = p_i + r_i * tic[i_camera_id];
            Eigen::Matrix3d r_wci = r_i * ric[i_camera_id];

            P.leftCols<3>() = r_wci.transpose();
            P.rightCols<1>() = -r_wci.transpose() * t_wci_w;

            f = i_pixel_coord / i_pixel_coord.z();
            svd_A.row(0) = f[0] * P.row(2) - P.row(0);
            svd_A.row(1) = f[1] * P.row(2) - P.row(1);

            // imu_j的信息
            auto &&j_pose = imu_j->vertex_pose;   // imu的位姿
            auto &&j_cameras = feature_in_cameras_j->second;    // imu中，与feature对应的相机信息
            auto &&j_camera_id = j_cameras[0].first;  // 左目的id
            auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

            Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
            Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
            Mat33 r_j {q_j.toRotationMatrix()};

            Eigen::Vector3d t_wcj_w = p_j + r_j * tic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * ric[j_camera_id];

            P.leftCols<3>() = r_wcj.transpose();
            P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;

            f = j_pixel_coord / j_pixel_coord.z();
            svd_A.row(2) = f[0] * P.row(2) - P.row(0);
            svd_A.row(3) = f[1] * P.row(2) - P.row(1);

            // 最小二乘计算世界坐标
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            Vec3 point {svd_V[0] / svd_V[3], svd_V[1] / svd_V[3], svd_V[2] / svd_V[3]};
            feature_it->second->vertex_point3d->set_parameters(point);
        }
    }

    void Estimator::local_triangulate_with(ImuNode *imu_i, ImuNode *imu_j, bool enforce) {
        if (imu_i == imu_j) {
            return;
        }

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
            auto &&i_pose = imu_i->vertex_pose;   // imu的位姿
            auto &&i_cameras = feature_in_cameras.second;    // imu中，与feature对应的相机信息
            auto &&i_camera_id = i_cameras[0].first;  // 左目的id
            auto &&i_pixel_coord = i_cameras[0].second;    // feature在imu的左目的像素坐标

            Vec3 p_i {i_pose->get_parameters()(0), i_pose->get_parameters()(1), i_pose->get_parameters()(2)};
            Qd q_i {i_pose->get_parameters()(6), i_pose->get_parameters()(3), i_pose->get_parameters()(4), i_pose->get_parameters()(5)};
            Mat33 r_i {q_i.toRotationMatrix()};

            Eigen::Vector3d t_wci_w = p_i + r_i * tic[i_camera_id];
            Eigen::Matrix3d r_wci = r_i * ric[i_camera_id];

            P.leftCols<3>().setIdentity();
            P.rightCols<1>().setZero();

            f = i_pixel_coord / i_pixel_coord.z();
            svd_A.row(0) = f[0] * P.row(2) - P.row(0);
            svd_A.row(1) = f[1] * P.row(2) - P.row(1);

            // imu_j的信息
            auto &&j_pose = imu_j->vertex_pose;   // imu的位姿
            auto &&j_cameras = feature_in_cameras_j->second;    // imu中，与feature对应的相机信息
            auto &&j_camera_id = j_cameras[0].first;  // 左目的id
            auto &&j_pixel_coord = j_cameras[0].second;    // feature在imu的左目的像素坐标

            Vec3 p_j {j_pose->get_parameters()(0), j_pose->get_parameters()(1), j_pose->get_parameters()(2)};
            Qd q_j {j_pose->get_parameters()(6), j_pose->get_parameters()(3), j_pose->get_parameters()(4), j_pose->get_parameters()(5)};
            Mat33 r_j {q_j.toRotationMatrix()};

            Eigen::Vector3d t_wcj_w = p_j + r_j * tic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * ric[j_camera_id];

            P.leftCols<3>() = r_wcj.transpose() * r_wci;
            P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w);

            f = j_pixel_coord / j_pixel_coord.z();
            svd_A.row(2) = f[0] * P.row(2) - P.row(0);
            svd_A.row(3) = f[1] * P.row(2) - P.row(1);

            // 最小二乘计算深度
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            Vec1 inverse_depth {svd_V[3] / svd_V[2]};
            feature_it->second->vertex_landmark->set_parameters(inverse_depth);
        }
    }

    void Estimator::global_triangulate_feature(FeatureNode* feature, bool enforce) {
        if (!feature) {
            return;
        }

        // 若imu数小于2，则无法进行三角化
        auto &&imu_deque = feature->imu_deque;
        if (imu_deque.size() < 2) {
            return;
        }

        if (!feature->vertex_point3d) {
            shared_ptr<VertexPoint3d> vertex_point3d(new VertexPoint3d);
            feature->vertex_point3d = vertex_point3d;
        } else if (!enforce) {
            return;
        }

        Eigen::MatrixXd svd_A(2 * imu_deque.size(), 4);
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

            Eigen::Vector3d t_wcj_w = p_j + r_j * tic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * ric[j_camera_id];

            P.leftCols<3>() = r_wcj.transpose();
            P.rightCols<1>() = -r_wcj.transpose() * t_wcj_w;

            f = j_pixel_coord / j_pixel_coord.z();
            svd_A.row(2 * j) = f[0] * P.row(2) - P.row(0);
            svd_A.row(2 * j + 1) = f[1] * P.row(2) - P.row(1);
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

        // 若imu数小于2，则无法进行三角化
        auto &&imu_deque = feature->imu_deque;
        if (imu_deque.size() < 2) {
            return;
        }

        if (!feature->vertex_landmark) {
            shared_ptr<VertexInverseDepth> vertex_inverse_depth(new VertexInverseDepth);
            feature->vertex_landmark = vertex_inverse_depth;
        } else if (!enforce) {
            return;
        }

        Eigen::MatrixXd svd_A(2 * imu_deque.size(), 4);
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

        Eigen::Vector3d t_wci_w = p_i + r_i * tic[i_camera_id];
        Eigen::Matrix3d r_wci = r_i * ric[i_camera_id];

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

            Eigen::Vector3d t_wcj_w = p_j + r_j * tic[j_camera_id];
            Eigen::Matrix3d r_wcj = r_j * ric[j_camera_id];

            P.leftCols<3>() = r_wcj.transpose() * r_wci;
            P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w);

            f = j_pixel_coord / j_pixel_coord.z();
            svd_A.row(2 * j) = f[0] * P.row(2) - P.row(0);
            svd_A.row(2 * j + 1) = f[1] * P.row(2) - P.row(1);
        }

        // 最小二乘计算深度
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        Vec1 inverse_depth {svd_V[3] / svd_V[2]};
        feature->vertex_landmark->set_parameters(inverse_depth);
    }

    bool Estimator::structure_from_motion() {
        // 找出第一个与当前imu拥有足够视差的imu, 同时利用对极几何计算t_i_curr, R_i_curr
        unsigned long imu_index;
        Matrix3d r_i_curr;
        Vector3d t_i_curr;
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

        // 利用i和curr进行三角化, 计算特征点的世界坐标
        global_triangulate_with(imu_i, _imu_node);

        /*
         * 1. 对imu_index后面的点进行pnp, 计算R, t.
         * 2. 得到R, t后进行三角化, 计算只有在imu_j到imu_node中才出现的特征点的世界坐标, i < j < curr
         * 3. 利用进行三角化, 计算只有在imu_i到imu_j中才出现的特征点的世界坐标, i < j < curr
         * */
        for (unsigned long j = imu_index + 1; _windows.size(); ++j) {
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

        // 遍历所有特征点, 对没有赋值的特征点进行三角化
        for (auto &feature_it : _feature_map) {
            global_triangulate_feature(feature_it.second);
        }

        // VO优化
        ProblemSLAM problem;

        // 把imu加入到problem中
        for (unsigned long i = 0; i < _windows.size(); ++i) {
            problem.add_vertex(_windows[i]->vertex_pose);
        }

        // 把特征点的世界坐标转换成深度，同时把特征点加入到problem中
        for (auto &feature_it : _feature_map) {
            unsigned long feature_id = feature_it.first;
            auto feature_node = feature_it.second;
            feature_node->from_global_to_local(_q_ic, _t_ic);

            if (feature_node->vertex_landmark) {
                // 把特征点加入到problem中
                problem.add_vertex(feature_node->vertex_landmark);

                // 构建重投影edge
                auto &&imu_deque = feature_node->imu_deque;
                if (imu_deque.size() > 1) {
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

                        auto edge_reproj = shared_ptr<EdgeReprojection>(new EdgeReprojection(host_point_pixel, j_point_pixel));
                        edge_reproj->set_vertex(feature_node->vertex_landmark, 0);
                        edge_reproj->set_vertex(host_imu->vertex_pose, 1);
                        edge_reproj->set_vertex(j_imu->vertex_pose, 2);
                        edge_reproj->set_vertex(_vertex_ext[0], 3);

                        // 把edge加入到problem中
                        problem.add_edge(edge_reproj);

                        // 外参不参与优化
                        _vertex_ext[0]->is_fixed();
                    }
                }
            }
        }




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
                auto edge_pnp = shared_ptr<EdgePnP>(new EdgePnP(point_pixel, point_world));
                edge_pnp->set_translation_imu_from_camera(q_ic, t_ic);
                edge_pnp->set_vertex(imu_i->vertex_pose, 0);

                problem.add_edge(edge_pnp);
            }
        }
        problem.solve(10);
    }

    bool Estimator::initial_structure() {
        TicToc t_sfm;
        // 通过加速度计的方差判断可观性

        // 遍历windows
        Vector3d aver_acc {};
        for (unsigned long i = 1; i < _windows.size(); ++i) {
            aver_acc += _windows[i]->imu_integration->get_delta_v() / _windows[i]->imu_integration->get_sum_dt();
        }
        aver_acc /= double(_windows.size() - 1);

        double var = 0.;
        for (unsigned long i = 1; i < _windows.size(); ++i) {
            Vec3 res = _windows[i]->imu_integration->get_delta_v() / _windows[i]->imu_integration->get_sum_dt() - aver_acc;
            var += aver_acc.squaredNorm();
        }
        var /= double(_windows.size() - 1);

        constexpr static double var_lim = 0.25 * 0.25;
        if (var < var_lim) {
            std::cout << "Warning: IMU excitation not enouth" << std::endl;
//            return false;
        }

        // global sfm
        Quaterniond Q[frame_count + 1];
        Vector3d T[frame_count + 1];
        map<int, Vector3d> sfm_tracked_points;
        vector<SFMFeature> sfm_f;
        for (auto &it_per_id : f_manager.feature)
        {
            int imu_j = it_per_id.start_frame - 1;
            SFMFeature tmp_feature;
            tmp_feature.state = false;
            tmp_feature.id = it_per_id.feature_id;
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                Vector3d pts_j = it_per_frame.point;
                tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
            }
            sfm_f.push_back(tmp_feature);
        }

        // 找出第一个与当前imu拥有足够视差的imu
        unsigned long imu_index;
        Matrix3d relative_R;
        Vector3d relative_T;
        if (!relative_pose(relative_R, relative_T, imu_index)) {
            cout << "Not enough features or parallax; Move device around" << endl;
            return false;
        }

        GlobalSFM sfm;
        if (!sfm.construct(frame_count + 1, Q, T, l,
                           relative_R, relative_T,
                           sfm_f, sfm_tracked_points))
        {
            cout << "global SFM failed!" << endl;
            marginalization_flag = MARGIN_OLD;
            return false;
        }

        //solve pnp for all frame
        map<double, ImageFrame>::iterator frame_it;
        map<int, Vector3d>::iterator it;
        frame_it = all_image_frame.begin();
        for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
        {
            // provide initial guess
            cv::Mat r, rvec, t, D, tmp_r;
            if ((frame_it->first) == Headers[i])
            {
                frame_it->second.is_key_frame = true;
                frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
                frame_it->second.T = T[i];
                i++;
                continue;
            }
            if ((frame_it->first) > Headers[i])
            {
                i++;
            }
            Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
            Vector3d P_inital = -R_inital * T[i];
            cv::eigen2cv(R_inital, tmp_r);
            cv::Rodrigues(tmp_r, rvec);
            cv::eigen2cv(P_inital, t);

            frame_it->second.is_key_frame = false;
            vector<cv::Point3f> pts_3_vector;
            vector<cv::Point2f> pts_2_vector;
            for (auto &id_pts : frame_it->second.points)
            {
                int feature_id = id_pts.first;
                for (auto &i_p : id_pts.second)
                {
                    it = sfm_tracked_points.find(feature_id);
                    if (it != sfm_tracked_points.end())
                    {
                        Vector3d world_pts = it->second;
                        cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                        pts_3_vector.push_back(pts_3);
                        Vector2d img_pts = i_p.second.head<2>();
                        cv::Point2f pts_2(img_pts(0), img_pts(1));
                        pts_2_vector.push_back(pts_2);
                    }
                }
            }
            cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
            if (pts_3_vector.size() < 6)
            {
                cout << "Not enough points for solve pnp pts_3_vector size " << pts_3_vector.size() << endl;
                return false;
            }
            if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
            {
                cout << " solve pnp fail!" << endl;
                return false;
            }
            cv::Rodrigues(rvec, r);
            MatrixXd R_pnp, tmp_R_pnp;
            cv::cv2eigen(r, tmp_R_pnp);
            R_pnp = tmp_R_pnp.transpose();
            MatrixXd T_pnp;
            cv::cv2eigen(t, T_pnp);
            T_pnp = R_pnp * (-T_pnp);
            frame_it->second.R = R_pnp * RIC[0].transpose();
            frame_it->second.T = T_pnp;
        }
        if (visualInitialAlign())
            return true;
        else
        {
            cout << "misalign visual structure with IMU" << endl;
            return false;
        }
    }

    bool Estimator::visualInitialAlign()
    {
        TicToc t_g;
        VectorXd x;
        //solve scale
        bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
        if (!result)
        {
            //ROS_DEBUG("solve g failed!");
            return false;
        }

        // change state
        for (int i = 0; i <= frame_count; i++)
        {
            Matrix3d Ri = all_image_frame[Headers[i]].R;
            Vector3d Pi = all_image_frame[Headers[i]].T;
            Ps[i] = Pi;
            Rs[i] = Ri;
            all_image_frame[Headers[i]].is_key_frame = true;
        }

        VectorXd dep = f_manager.getDepthVector();
        for (int i = 0; i < dep.size(); i++)
            dep[i] = -1;
        f_manager.clearDepth(dep);

        //triangulat on cam pose , no tic
        Vector3d TIC_TMP[NUM_OF_CAM];
        for (int i = 0; i < NUM_OF_CAM; i++)
            TIC_TMP[i].setZero();
        ric[0] = RIC[0];
        f_manager.setRic(ric);
        f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

        double s = (x.tail<1>())(0);
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        }
        for (int i = frame_count; i >= 0; i--)
            Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
        int kv = -1;
        map<double, ImageFrame>::iterator frame_i;
        for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
        {
            if (frame_i->second.is_key_frame)
            {
                kv++;
                Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
            }
        }
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            it_per_id.estimated_depth *= s;
        }

        Matrix3d R0 = Utility::g2R(g);
        double yaw = Utility::R2ypr(R0 * Rs[0]).x();
        R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        g = R0 * g;
        //Matrix3d rot_diff = R0 * Rs[0].transpose();
        Matrix3d rot_diff = R0;
        for (int i = 0; i <= frame_count; i++)
        {
            Ps[i] = rot_diff * Ps[i];
            Rs[i] = rot_diff * Rs[i];
            Vs[i] = rot_diff * Vs[i];
        }
        //ROS_DEBUG_STREAM("g0     " << g.transpose());
        //ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

        return true;
    }

    bool Estimator::relative_pose(Matrix3d &r, Vector3d &t, unsigned long &imu_index) {
        // 遍历windows中的所有imu, 计算哪个imu是第一个与当前imu拥有足够视差的imu
        for (unsigned long i = 0; i < _windows.size(); ++i) {
            // 获取第i个imu与当前imu的共同特征的像素坐标
            vector<pair<Vector3d, Vector3d>> corres;
            for (auto &feature_in_cameras : _windows[i]->features_in_cameras) {
                auto &&it = _imu_node->features_in_cameras.find(feature_in_cameras.first);
                if (it == _imu_node->features_in_cameras.end()) {
                    continue;
                }
                corres.emplace_back(feature_in_cameras.second[0].second, it->second[0].second);
            }

            // 特征点数量大于一定值时，计算视差
            constexpr static unsigned long corres_count_lim = 20;
            if (corres.size() > corres_count_lim) {
                double average_parallax = 0.;
                for (auto &corre : corres) {
                    double du = corre.first.x() - corre.second.x();
                    double dv = corre.first.y() - corre.second.y();
                    average_parallax += sqrt(du * du + dv * dv);
                }
                average_parallax /= double(corres.size());

                if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, r, t)) {
                    imu_index = i;
                    //ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                    return true;
                }
            }
        }

        return false;
    }

    void Estimator::solve_odometry() {
        if (_windows.full() && solver_flag == NON_LINEAR) {
            TicToc t_tri;
            backend_optimization();
        }
    }

    /*!
     * 把先验参数赋给后端
     */
    void Estimator::vector2double() {
        for (int i = 0; i <= WINDOW_SIZE; i++) {
            para_Pose[i][0] = Ps[i].x();
            para_Pose[i][1] = Ps[i].y();
            para_Pose[i][2] = Ps[i].z();
            Quaterniond q{Rs[i]};
            para_Pose[i][3] = q.x();
            para_Pose[i][4] = q.y();
            para_Pose[i][5] = q.z();
            para_Pose[i][6] = q.w();

            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
        for (int i = 0; i < NUM_OF_CAM; i++) {
            para_Ex_Pose[i][0] = tic[i].x();
            para_Ex_Pose[i][1] = tic[i].y();
            para_Ex_Pose[i][2] = tic[i].z();
            Quaterniond q{ric[i]};
            para_Ex_Pose[i][3] = q.x();
            para_Ex_Pose[i][4] = q.y();
            para_Ex_Pose[i][5] = q.z();
            para_Ex_Pose[i][6] = q.w();
        }

        auto &&dep = f_manager.get_depth_vector();
        for (int i = 0; i < dep.rows(); ++i) {
            para_Feature[i][0] = dep(i);
        }
        if (ESTIMATE_TD) {
            para_Td[0][0] = td;
        }
    }

    /*!
     * 把后端参数赋值到先验参数上, 并且对齐第一个位姿
     */
    void Estimator::double2vector() {
        Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
        Vector3d origin_P0 = Ps[0];

        if (failure_occur) {
            origin_R0 = Utility::R2ypr(last_R0);
            origin_P0 = last_P0;
            failure_occur = 0;
        }
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                         para_Pose[0][3],
                                                         para_Pose[0][4],
                                                         para_Pose[0][5])
                                                     .toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
            //ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5])
                    .toRotationMatrix()
                    .transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++) {
            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                        para_Pose[i][1] - para_Pose[0][1],
                                        para_Pose[i][2] - para_Pose[0][2]) +
                    origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                              para_SpeedBias[i][4],
                              para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                              para_SpeedBias[i][7],
                              para_SpeedBias[i][8]);
        }

        for (int i = 0; i < NUM_OF_CAM; i++) {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5])
                    .toRotationMatrix();
        }

        VectorXd dep = VectorXd::Zero(f_manager.get_feature_count(), 1);
        for (int i = 0; i < dep.size(); i++) {
            dep(i) = para_Feature[i][0];
        }
        f_manager.set_depth(dep);
        if (ESTIMATE_TD) {
            td = para_Td[0][0];
        }

        // relative info between two loop frame
        if (relocalization_info) {
            Matrix3d relo_r;
            Vector3d relo_t;
            relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
            relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                         relo_Pose[1] - para_Pose[0][1],
                                         relo_Pose[2] - para_Pose[0][2]) +
                     origin_P0;
            double drift_correct_yaw;
            drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
            drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
            drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
            relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
            relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
            relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
            //cout << "vins relo " << endl;
            //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
            //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
            relocalization_info = 0;
        }
    }

    bool Estimator::failureDetection()
    {
        if (f_manager.last_track_num < 2)
        {
            //ROS_INFO(" little feature %d", f_manager.last_track_num);
            //return true;
        }
        if (Bas[WINDOW_SIZE].norm() > 2.5)
        {
            //ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
            return true;
        }
        if (Bgs[WINDOW_SIZE].norm() > 1.0)
        {
            //ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
            return true;
        }
        /*
        if (tic(0) > 1)
        {
            //ROS_INFO(" big extri param estimation %d", tic(0) > 1);
            return true;
        }
        */
        Vector3d tmp_P = Ps[WINDOW_SIZE];
        if ((tmp_P - last_P).norm() > 5)
        {
            //ROS_INFO(" big translation");
            return true;
        }
        if (abs(tmp_P.z() - last_P.z()) > 1)
        {
            //ROS_INFO(" big z translation");
            return true;
        }
        Matrix3d tmp_R = Rs[WINDOW_SIZE];
        Matrix3d delta_R = tmp_R.transpose() * last_R;
        Quaterniond delta_Q(delta_R);
        double delta_angle;
        delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
        if (delta_angle > 50)
        {
            //ROS_INFO(" big delta_angle ");
            //return true;
        }
        return false;
    }

    void Estimator::marg_old_frame() {
        backend::LossFunction *lossfunction;
        lossfunction = new backend::CauchyLoss(1.0);

        // step1. 构建 problem
        backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
        vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
        vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
        int pose_dim = 0;

        // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
        shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
        {
            Eigen::VectorXd pose(7);
            pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
            vertexExt->SetParameters(pose);
            problem.AddVertex(vertexExt);
            pose_dim += vertexExt->LocalDimension();
        }

        for (int i = 0; i < WINDOW_SIZE + 1; i++)
        {
            shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
            Eigen::VectorXd pose(7);
            pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
            vertexCam->SetParameters(pose);
            vertexCams_vec.push_back(vertexCam);
            problem.AddVertex(vertexCam);
            pose_dim += vertexCam->LocalDimension();

            shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
            Eigen::VectorXd vb(9);
            vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
                    para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
                    para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
            vertexVB->SetParameters(vb);
            vertexVB_vec.push_back(vertexVB);
            problem.AddVertex(vertexVB);
            pose_dim += vertexVB->LocalDimension();
        }

        // IMU
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(pre_integrations[1]));
                std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
                edge_vertex.push_back(vertexCams_vec[0]);
                edge_vertex.push_back(vertexVB_vec[0]);
                edge_vertex.push_back(vertexCams_vec[1]);
                edge_vertex.push_back(vertexVB_vec[1]);
                imuEdge->SetVertex(edge_vertex);
                problem.AddEdge(imuEdge);
            }
        }

        // Visual Factor
        {
            int feature_index = -1;
            // 遍历每一个特征
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
                VecX inv_d(1);
                inv_d << para_Feature[feature_index][0];
                verterxPoint->SetParameters(inv_d);
                problem.AddVertex(verterxPoint);

                // 遍历所有的观测
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;

                    std::shared_ptr<backend::EdgeReprojection> edge(new backend::EdgeReprojection(pts_i, pts_j));
                    std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
                    edge_vertex.push_back(verterxPoint);
                    edge_vertex.push_back(vertexCams_vec[imu_i]);
                    edge_vertex.push_back(vertexCams_vec[imu_j]);
                    edge_vertex.push_back(vertexExt);

                    edge->SetVertex(edge_vertex);
                    edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);

                    edge->SetLossFunction(lossfunction);
                    problem.AddEdge(edge);
                }
            }
        }

        // 先验
        {
            // 已经有 Prior 了
            if (Hprior_.rows() > 0)
            {
                problem.SetHessianPrior(Hprior_); // 告诉这个 problem
                problem.SetbPrior(bprior_);
                problem.SetErrPrior(errprior_);
                problem.SetJtPrior(Jprior_inv_);
                problem.ExtendHessiansPriorSize(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
            }
            else
            {
                Hprior_ = MatXX(pose_dim, pose_dim);
                Hprior_.setZero();
                bprior_ = VecX(pose_dim);
                bprior_.setZero();
                problem.SetHessianPrior(Hprior_); // 告诉这个 problem
                problem.SetbPrior(bprior_);
            }
        }

        std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
        marg_vertex.push_back(vertexCams_vec[0]);
        marg_vertex.push_back(vertexVB_vec[0]);
        problem.Marginalize(marg_vertex, pose_dim);
        Hprior_ = problem.GetHessianPrior();
        bprior_ = problem.GetbPrior();
        errprior_ = problem.GetErrPrior();
        Jprior_inv_ = problem.GetJtPrior();
    }
    void Estimator::MargNewFrame()
    {

        // step1. 构建 problem
        backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
        vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
        vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
        //    vector<backend::Point3d> points;
        int pose_dim = 0;

        // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
        shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
        {
            Eigen::VectorXd pose(7);
            pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
            vertexExt->SetParameters(pose);
            problem.AddVertex(vertexExt);
            pose_dim += vertexExt->LocalDimension();
        }

        for (int i = 0; i < WINDOW_SIZE + 1; i++)
        {
            shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
            Eigen::VectorXd pose(7);
            pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
            vertexCam->SetParameters(pose);
            vertexCams_vec.push_back(vertexCam);
            problem.AddVertex(vertexCam);
            pose_dim += vertexCam->LocalDimension();

            shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
            Eigen::VectorXd vb(9);
            vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
                    para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
                    para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
            vertexVB->SetParameters(vb);
            vertexVB_vec.push_back(vertexVB);
            problem.AddVertex(vertexVB);
            pose_dim += vertexVB->LocalDimension();
        }

        // 先验
        {
            // 已经有 Prior 了
            if (Hprior_.rows() > 0)
            {
                problem.SetHessianPrior(Hprior_); // 告诉这个 problem
                problem.SetbPrior(bprior_);
                problem.SetErrPrior(errprior_);
                problem.SetJtPrior(Jprior_inv_);

                problem.ExtendHessiansPriorSize(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
            }
            else
            {
                Hprior_ = MatXX(pose_dim, pose_dim);
                Hprior_.setZero();
                bprior_ = VecX(pose_dim);
                bprior_.setZero();
            }
        }

        std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
        // 把窗口倒数第二个帧 marg 掉
        marg_vertex.push_back(vertexCams_vec[WINDOW_SIZE - 1]);
        marg_vertex.push_back(vertexVB_vec[WINDOW_SIZE - 1]);
        problem.Marginalize(marg_vertex, pose_dim);
        Hprior_ = problem.GetHessianPrior();
        bprior_ = problem.GetbPrior();
        errprior_ = problem.GetErrPrior();
        Jprior_inv_ = problem.GetJtPrior();
    }

    void Estimator::problem_solve() {
        backend::LossFunction *lossfunction;
        lossfunction = new backend::CauchyLoss(1.0);
        //    lossfunction = new backend::TukeyLoss(1.0);

        // step1. 构建 problem
        _problem = slam::ProblemSLAM();
        vector<shared_ptr<VertexPose>> vertex_pose_buff;
        vector<shared_ptr<VertexMotion>> vertex_motion_buff;
        unsigned long pose_dim = 0;

        // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
        shared_ptr<VertexPose> vertex_extrinsic(new VertexPose());
        Eigen::Matrix<double, 7, 1> pose_extrinsic;
        pose_extrinsic << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertex_extrinsic->set_parameters(pose_extrinsic);
        if (!ESTIMATE_EXTRINSIC) {
            //ROS_DEBUG("fix extinsic param");
            // TODO:: set Hessian prior to zero
            vertex_extrinsic->set_fixed();
        } else{
            //ROS_DEBUG("estimate extinsic param");
        }
        _problem.add_vertex(vertex_extrinsic);
        pose_dim += vertex_extrinsic->local_dimension();

        // 设置顶点 (p, R, v, ba, bg)
        for (unsigned long i = 0; i < WINDOW_SIZE + 1; ++i) {
            // p, R
            shared_ptr<VertexPose> vertex_pose(new VertexPose());
            Eigen::Matrix<double, 7, 1> pose_imu;
            pose_imu << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
            vertex_pose->set_parameters(pose_imu);
            vertex_pose_buff.emplace_back(vertex_pose);
            _problem.add_vertex(vertex_pose);
            pose_dim += vertex_pose->local_dimension();

            // v, ba, bg
            shared_ptr<VertexMotion> vertex_motion(new VertexMotion());
            Eigen::Matrix<double, 9, 1> motion;
            motion << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
                      para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
                      para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
            vertex_motion->set_parameters(motion);
            vertex_motion_buff.emplace_back(vertex_motion);
            _problem.add_vertex(vertex_motion);
            pose_dim += vertex_motion->local_dimension();
        }

        // 设置IMU误差
        vector<shared_ptr<Vertex>> vertices_of_imu(4);
        for (unsigned long i = 0; i < WINDOW_SIZE; ++i) {
            unsigned long j = i + 1;
            if (pre_integrations[j]->get_sum_dt() > 10.0) {
                continue;
            }

            vertices_of_imu.emplace_back(vertex_pose_buff[i]);
            vertices_of_imu.emplace_back(vertex_motion_buff[i]);
            vertices_of_imu.emplace_back(vertex_pose_buff[j]);
            vertices_of_imu.emplace_back(vertex_motion_buff[j]);

            std::shared_ptr<graph_optimization::EdgeImu> edge_imu(new graph_optimization::EdgeImu(pre_integrations[j]));
            edge_imu->set_vertices(vertices_of_imu);
            _problem.add_edge(edge_imu);
        }

        // 设置顶点(landmark)以及视觉重投影误差
        vector<shared_ptr<VertexInverseDepth>> vertex_feature_buff;
        vector<shared_ptr<Vertex>> vertices_of_reproj(4);
        unsigned long feature_index = 0;
        // 遍历每一个特征
        for (auto &feature : f_manager.features_map) {
            if (feature.second.is_suitable_to_reprojection()) {
                // 设置landmark顶点
                shared_ptr<VertexInverseDepth> vertex_feature(new VertexInverseDepth());
                vertex_feature->set_parameters(Eigen::Matrix<double, 1, 1>(para_Feature[feature_index][0]));
                _problem.add_vertex(vertex_feature);
                vertex_feature_buff.emplace_back(vertex_feature);

                // 计算视觉重投影误差
                unsigned long imu_i = feature.second.start_frame_id;
                Vector3d pt_i = feature.second.feature_local_infos[0].point;    //
                for (unsigned long index = 1; index < feature.second.feature_local_infos.size(); ++index) {
                    unsigned long imu_j = imu_i + index;
                    Vector3d pt_j = feature.second.feature_local_infos[index].point;

                    vertices_of_reproj[0] = vertex_feature;
                    vertices_of_reproj[1] = vertex_pose_buff[imu_i];
                    vertices_of_reproj[2] = vertex_pose_buff[imu_j];
                    vertices_of_reproj[3] = vertex_extrinsic;

                    shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection(pt_i, pt_j));
                    edge_reproj->set_vertices(vertices_of_reproj);
                    edge_reproj->set_information(project_sqrt_info_.transpose() * project_sqrt_info_);
                    edge_reproj->set_loss_function(loss_function);

                    _problem.add_edge(edge_reproj);
                }
                ++feature_index;
            }
        }

        // 先验
        if (Hprior_.rows() > 0) {
            _problem.set_hessian_prior(Hprior_);
            _problem.set_b_prior(bprior_);
            _problem.set_err_Prior(errprior_);
            _problem.set_Jt_prior(Jprior_inv_);
            _problem.extend_prior_hessian_size(15); // 因为每次都会marg掉一个状态, 所以要给先验hessian扩维到当前维度
        }

        _problem.solve(10);

        // update bprior_,  Hprior_ do not need update
        if (Hprior_.rows() > 0)
        {
            std::cout << "----------- update bprior -------------\n";
            std::cout << "             before: " << bprior_.norm() << std::endl;
            std::cout << "                     " << errprior_.norm() << std::endl;
            bprior_ = problem.GetbPrior();
            errprior_ = problem.GetErrPrior();
            std::cout << "             after: " << bprior_.norm() << std::endl;
            std::cout << "                    " << errprior_.norm() << std::endl;
        }

        // update parameter
        for (int i = 0; i < WINDOW_SIZE + 1; i++)
        {
            auto &&p = vertex_pose_buff[i]->get_parameters();
            for (int j = 0; j < 7; ++j){
                para_Pose[i][j] = p[j];
            }

            auto &&vb = vertex_motion_buff[i]->get_parameters();
            for (int j = 0; j < 9; ++j) {
                para_SpeedBias[i][j] = vb[j];
            }
        }

        // 遍历每一个特征
        for (int i = 0; i < vertex_feature_buff.size(); ++i) {
            auto &&f = vertex_feature_buff[i]->get_parameters();
            para_Feature[i][0] = f[0];
        }
    }

    void Estimator::backend_optimization() {
        TicToc t_solver;
        // 借助 vins 框架，维护变量
        vector2double();
        // 构建求解器
        problem_solve();
        // 优化后的变量处理下自由度
        double2vector();
        //ROS_INFO("whole time for solver: %f", t_solver.toc());

        // 维护 marg
        TicToc t_whole_marginalization;
        if (marginalization_flag == MARGIN_OLD) {
            vector2double();

            marg_old_frame();

            std::unordered_map<long, double *> addr_shift; // prior 中对应的保留下来的参数地址
            for (int i = 1; i <= WINDOW_SIZE; i++)
            {
                addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
        }
        else
        {
            if (Hprior_.rows() > 0)
            {

                vector2double();

                MargNewFrame();

                std::unordered_map<long, double *> addr_shift;
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    if (i == WINDOW_SIZE - 1)
                        continue;
                    else if (i == WINDOW_SIZE)
                    {
                        addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                    }
                    else
                    {
                        addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                    }
                }
                for (int i = 0; i < NUM_OF_CAM; i++)
                    addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
                if (ESTIMATE_TD)
                {
                    addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
                }
            }
        }

    }


    void Estimator::slide_window() {
        TicToc t_margin;

        // 只有当windows满了才进行滑窗操作
        if (_windows.full()) {
            if (marginalization_flag == MARGIN_OLD) {
                ImuNode *imu_oldest;
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
                        // 在problem中删除特征点
                        _problem.remove_vertex(vertex_landmark);

                        // 在map中删除特征点
                        _feature_map.erase(feature_id);

                        // 释放特征点node的空间
                        delete feature_node;

                        // TODO: 在新的oldest imu中，把feature删除
                    } else {
                        // 曾经的host imu
                        auto &&oldest_cameras = feature_in_cameras.second;    // imu中，与feature对应的相机信息
                        auto &&oldest_imu_pose = imu_oldest->vertex_pose;   // imu的位姿
                        auto &&oldest_camera_id = oldest_cameras[0].first;  // camera的id
                        auto &&oldest_pixel_coord = oldest_cameras[0].second;    // feature在imu的左目的像素坐标

                        Vec3 p_i {oldest_imu_pose->get_parameters()(0), oldest_imu_pose->get_parameters()(1), oldest_imu_pose->get_parameters()(2)};
                        Qd q_i {oldest_imu_pose->get_parameters()(6), oldest_imu_pose->get_parameters()(3), oldest_imu_pose->get_parameters()(4), oldest_imu_pose->get_parameters()(5)};
                        Mat33 r_i {q_i.toRotationMatrix()};

                        Eigen::Vector3d t_wci_w = p_i + r_i * tic[oldest_camera_id];
                        Eigen::Matrix3d r_wci = r_i * ric[oldest_camera_id];

                        // 现在的host imu
                        auto &&host_imu = imu_deque.oldest();
                        auto &&host_cameras = host_imu->features_in_cameras[feature_id];
                        auto &&host_imu_pose = host_imu->vertex_pose;
                        auto &&host_camera_id = host_cameras[0].first;
                        auto &&host_pixel_coord = host_cameras[0].second;

                        Vec3 p_j {host_imu_pose->get_parameters()(0), host_imu_pose->get_parameters()(1), host_imu_pose->get_parameters()(2)};
                        Qd q_j {host_imu_pose->get_parameters()(6), host_imu_pose->get_parameters()(3), host_imu_pose->get_parameters()(4), host_imu_pose->get_parameters()(5)};
                        Mat33 r_j {q_j.toRotationMatrix()};

                        Eigen::Vector3d t_wcj_w = p_j + r_j * tic[host_camera_id];
                        Eigen::Matrix3d r_wcj = r_j * ric[host_camera_id];

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
                            edge_reproj->set_vertex(vertex_landmark, 0);
                            edge_reproj->set_vertex(host_imu_pose, 1);
                            edge_reproj->set_vertex(host_imu_pose, 2);
                            edge_reproj->set_vertex(_vertex_ext[other_camera_id], 3);
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
                                edge_reproj->set_vertex(vertex_landmark, 0);
                                edge_reproj->set_vertex(host_imu_pose, 1);
                                edge_reproj->set_vertex(other_imu_pose, 2);
                                edge_reproj->set_vertex(_vertex_ext[other_camera_id], 3);
                            }
                        }
                    }
                }
                // 释放imu node的空间
                delete imu_oldest;
            } else {
                ImuNode *imu_newest;
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