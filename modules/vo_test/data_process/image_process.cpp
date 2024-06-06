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
        std::cout << "_windows.size() = " << _windows.size() << std::endl;
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
        }

        // 遍历image中的每个feature
        double parallax_sum = 0.;
        unsigned int parallax_num = 0;
        unsigned int last_track_num = 0;
        for (auto &landmark : image) {
            unsigned long feature_id = landmark.first;
            auto feature_it = _feature_map.find(feature_id);
            FeatureNode *feature_node = nullptr;

            // 若feature不在feature map中，则需要新建feature_node
            if (feature_it == _feature_map.end()) {
                feature_node = new FeatureNode(feature_id);
                _feature_map.emplace(feature_id, feature_node);
//                feature_it = _feature_map.find(feature_id);

                // TODO: 对于新的特征点，其实不需要进行重投影误差的计算
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

            // 利用windows中的信息，判断feature是否能够被用于计算重投影误差
            if (_windows.is_feature_suitable_to_reproject(feature_id)) {
                // 获取feature的参考值
                auto &&host_imu = feature_node->imu_deque.oldest();   // 第一次看到feature的imu
                auto &&host_imu_pose = host_imu->vertex_pose;   // imu的位姿
                auto &&host_cameras = host_imu->features_in_cameras[feature_id];    // imu中，与feature对应的相机信息
                auto &&host_camera_id = host_cameras[0].first;  // camera的id
                auto &&host_pixel_coord = host_cameras[0].second;    // feature在imu的左目的像素坐标

                // TODO: 不使用指针是否为nullptr来判断，而是用is_triangulated来判断
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

                    f = host_pixel_coord / host_pixel_coord.z();
                    svd_A.row(0) = f[0] * P.row(2) - P.row(0);
                    svd_A.row(1) = f[1] * P.row(2) - P.row(1);

                    unsigned long index = 0;

                    // host imu的其他cameras
                    for (unsigned long j = 1; j < host_cameras.size(); ++j) {
                        ++index;

                        // 获取feature的参考值
                        auto &&other_camera_id = host_cameras[j].first; // camera的id
                        auto &&other_pixel_coord = host_cameras[j].second;    // feature在imu的左目的像素坐标

                        Eigen::Vector3d t_wcj_w = p_i + r_i * _t_ic[other_camera_id];
                        Eigen::Matrix3d r_wcj = r_i * _q_ic[other_camera_id];

                        P.leftCols<3>() = r_wcj.transpose() * r_wci;    // r_cjci
                        P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w); // t_cjci_cj

                        f = other_pixel_coord / other_pixel_coord.z();
                        svd_A.row(2 * index) = f[0] * P.row(2) - P.row(0);
                        svd_A.row(2 * index + 1) = f[1] * P.row(2) - P.row(1);

                        // 构建视觉重投影误差边
                        shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                host_pixel_coord,
                                other_pixel_coord
                        ));
                        edge_reproj->add_vertex(vertex_landmark);
                        edge_reproj->add_vertex(host_imu_pose);
                        edge_reproj->add_vertex(host_imu_pose);
                        edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
                        _problem.add_edge(edge_reproj);
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

                            P.leftCols<3>() = r_wcj.transpose() * r_wci;    // r_cjci;
                            P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w); // t_cjci_cj

                            f = other_pixel_coord / other_pixel_coord.z();
                            svd_A.row(2 * index) = f[0] * P.row(2) - P.row(0);
                            svd_A.row(2 * index + 1) = f[1] * P.row(2) - P.row(1);

                            // 构建视觉重投影误差边
                            shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                    host_pixel_coord,
                                    other_pixel_coord
                            ));
                            edge_reproj->add_vertex(vertex_landmark);
                            edge_reproj->add_vertex(host_imu_pose);
                            edge_reproj->add_vertex(other_imu_pose);
                            edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
                            _problem.add_edge(edge_reproj);
                        }
                    }

                    // 当前的imu
                    auto state_r = _state.q.toRotationMatrix();
                    for (auto &camera : cameras) {
                        ++index;

                        unsigned long current_camera_id = camera.first;  // camera的id
                        Vec3 current_pixel_coord = {camera.second.x(), camera.second.y(), camera.second.z()};    // feature在imu的左目的像素坐标

                        Eigen::Vector3d t_wcj_w = _state.p + state_r * _t_ic[current_camera_id];
                        Eigen::Matrix3d r_wcj = state_r * _q_ic[current_camera_id];

                        P.leftCols<3>() = r_wcj.transpose() * r_wci;    // r_cjci;
                        P.rightCols<1>() = r_wcj.transpose() * (t_wci_w - t_wcj_w); // t_cjci_cj;

                        f = current_pixel_coord / current_pixel_coord.z();
                        svd_A.row(2 * index) = f[0] * P.row(2) - P.row(0);
                        svd_A.row(2 * index + 1) = f[1] * P.row(2) - P.row(1);

                        // 构建视觉重投影误差边
                        shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                host_pixel_coord,
                                current_pixel_coord
                        ));
                        edge_reproj->add_vertex(feature_node->vertex_landmark);
                        edge_reproj->add_vertex(host_imu_pose);
                        edge_reproj->add_vertex(_imu_node->vertex_pose);
                        edge_reproj->add_vertex(_vertex_ext[current_camera_id]);
                        _problem.add_edge(edge_reproj);
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
                    // 把landmark顶点加入到problem中
                    _problem.add_vertex(feature_node->vertex_landmark);

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
                        _problem.add_edge(edge_reproj);
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
//        std::cout << "5" << std::endl;
//        std::cout << "_windows.size() = " << _windows.size() << std::endl;
//        marginalization_flag = MARGIN_OLD;
//        slide_window();
//        std::cout << "6" << std::endl;
//        std::cout << "_windows.size() = " << _windows.size() << std::endl;

        if (solver_flag == INITIAL) {
            if (_windows.full()) {   // WINDOW已经装满了, 且还有camera frame
                bool is_initialized = initialize();
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

                    std::cout << "q_est: " << _state.q.w() << ", " << _state.q.x() << ", " << _state.q.y() << ", " << _state.q.z()<< std::endl;
                    std::cout << "p_est: " << _state.p.transpose() << std::endl;
                    std::cout << "v_est: " << _state.v.transpose() << std::endl;
                    std::cout << "ba_est: " << _state.ba.transpose() << std::endl;
                    std::cout << "bg_est: " << _state.bg.transpose() << std::endl;
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

            std::cout << "q_est: " << _state.q.w() << ", " << _state.q.x() << ", " << _state.q.y() << ", " << _state.q.z()<< std::endl;
            std::cout << "p_est: " << _state.p.transpose() << std::endl;
            std::cout << "v_est: " << _state.v.transpose() << std::endl;
            std::cout << "ba_est: " << _state.ba.transpose() << std::endl;
            std::cout << "bg_est: " << _state.bg.transpose() << std::endl;
        }

        slide_window();
    }
}