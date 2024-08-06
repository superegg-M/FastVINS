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
//        _q_ic[0] = {cos(-0.5 * double(EIGEN_PI) * 0.5), 0., sin(-0.5 * double(EIGEN_PI) * 0.5), 0.};
        _q_ic[0] = {cos(-0.5 * double(EIGEN_PI) * 0.5), sin(-0.5 * double(EIGEN_PI) * 0.5), 0., 0.};
        _t_ic[0] = {0., 0., 0.};
        Vec7 pose;
        pose << _t_ic[0].x(), _t_ic[0].y(), _t_ic[0].z(), _q_ic[0].x(), _q_ic[0].y(), _q_ic[0].z(), _q_ic[0].w();
        _vertex_ext[0] = std::make_shared<VertexPose>();
        _vertex_ext[0]->set_parameters(pose);

        solver_flag = INITIAL;

        _problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
    }

    bool Estimator::initialize() {
        std::cout << "running initialize" << std::endl;
        TicToc t_sfm;

        if (structure_from_motion()) {
            std::cout << "done structure_from_motion" << std::endl;

            // 移除outlier的landmarks
            remove_outlier_landmarks();

            // 移除为三角化的landmarks
            remove_untriangulated_landmarks();

            if (align_visual_to_imu()) {
                std::cout << "done align_visual_to_imu" << std::endl;

                // 把当前imu的信息赋值到state中
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

                return true;
            }
        }

        return false;
    }

    graph_optimization::ProblemSLAM Estimator::optimization(const unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> &image){
        std::cout << "_windows.size() = " << _windows.size() << std::endl;
        TicToc t_process;        
        
        // 创建SLAM问题
        graph_optimization::ProblemSLAM problem;

        // 首先添加滑窗和旧地图中的顶点和边

        // 遍历滑窗,添加imu顶点
        for(size_t i=0; i<_windows.size(); ++i){
            problem.add_vertex(_windows[i]->vertex_pose);
            problem.add_vertex(_windows[i]->vertex_motion);
        }

        // 添加预积分边
        for(size_t j=1; j<_windows.size(); ++j){
            size_t i=j-1;
            if (_windows[j]->imu_integration && _windows[j]->imu_integration->get_sum_dt() < 10.) {
                shared_ptr<EdgeImu> edge_imu(new EdgeImu(*_windows[j]->imu_integration));
                edge_imu->add_vertex(_windows[i]->vertex_pose);
                edge_imu->add_vertex(_windows[i]->vertex_motion);
                edge_imu->add_vertex(_windows[j]->vertex_pose);
                edge_imu->add_vertex(_windows[j]->vertex_motion);
                problem.add_edge(edge_imu);
            }
        }


#ifdef USE_OPENMP   
        std::vector<std::pair<const unsigned long, vins::FeatureNode *>> _feature_map_vector;
        _feature_map_vector.reserve(_feature_map.size());
        for (auto &feature : _feature_map) {
            _feature_map_vector.emplace_back(feature);
        }

#pragma omp parallel for num_threads(4)
        for(size_t i=0;i<_feature_map_vector.size();++i){
            unsigned long feature_id = _feature_map_vector[i].first;
            FeatureNode *feature_node = nullptr;
            feature_node = _feature_map_vector[i].second;

            // 进行特征点有效性的检查
            if (!_windows.is_feature_suitable_to_reproject(feature_id))
                continue;

            // 三角化特征点并检查有效性
            local_triangulate_feature(feature_node);
            if(!feature_node->is_triangulated)
                continue;

#pragma omp critical
{
            // 添加landmark顶点
            problem.add_vertex(feature_node->vertex_landmark);
}

            // 获取feature的参考值
            auto &&host_imu = feature_node->imu_deque.oldest();   // 第一次看到feature的imu
            auto &&host_imu_pose = host_imu->vertex_pose;   // imu的位姿
            auto &&host_cameras = host_imu->features_in_cameras[feature_id];    // imu中，与feature对应的相机信息
            // auto &&host_camera_id = host_cameras[0].first;  // camera的id
            auto &&host_pixel_coord = host_cameras[0].second;    // feature在imu的左目的像素坐标

            // 遍历看到feature的所有imu，添加重投影误差边
            // 从1开始，0是host_imu
            for(size_t imu_i=1; imu_i<feature_node->imu_deque.size();++imu_i){
                // imu中，与feature对应的相机信息
                auto &cameras=feature_node->imu_deque[imu_i]->features_in_cameras[feature_id] ;

                // 对当前imu下的所有cameras计算视觉重投影误差
                for (auto &camera : cameras) {
                    unsigned long other_camera_id = camera.first;  // camera的id
                    Vec3 other_pixel_coord = {camera.second[0], camera.second[1], camera.second[2]};    // feature在imu的左目的像素坐标

                    shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                            host_pixel_coord,
                            other_pixel_coord
                    ));
                    edge_reproj->add_vertex(feature_node->vertex_landmark);
                    edge_reproj->add_vertex(host_imu_pose);
                    edge_reproj->add_vertex(feature_node->imu_deque[imu_i]->vertex_pose);
                    edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
#pragma omp critical
{
                    problem.add_edge(edge_reproj);
}
 
                }

            }

        }

#else
        // 遍历所有feature
        for(auto &feature:_feature_map){
            unsigned long feature_id = feature.first;
            FeatureNode *feature_node = nullptr;
            feature_node = feature.second;

            // 进行特征点有效性的检查
            if (!_windows.is_feature_suitable_to_reproject(feature_id))
                continue;

            // 三角化特征点并检查有效性
            local_triangulate_feature(feature_node);
            if(!feature_node->is_triangulated)
                continue;

            // 添加landmark顶点
            problem.add_vertex(feature_node->vertex_landmark);

            // 获取feature的参考值
            auto &&host_imu = feature_node->imu_deque.oldest();   // 第一次看到feature的imu
            auto &&host_imu_pose = host_imu->vertex_pose;   // imu的位姿
            auto &&host_cameras = host_imu->features_in_cameras[feature_id];    // imu中，与feature对应的相机信息
            // auto &&host_camera_id = host_cameras[0].first;  // camera的id
            auto &&host_pixel_coord = host_cameras[0].second;    // feature在imu的左目的像素坐标

            // 遍历看到feature的所有imu，添加重投影误差边
            // 从1开始，0是host_imu
            for(size_t imu_i=1; imu_i<feature_node->imu_deque.size();++imu_i){
                // imu中，与feature对应的相机信息
                auto &cameras=feature_node->imu_deque[imu_i]->features_in_cameras[feature_id] ;

                // 对当前imu下的所有cameras计算视觉重投影误差
                for (auto &camera : cameras) {
                    unsigned long other_camera_id = camera.first;  // camera的id
                    Vec3 other_pixel_coord = {camera.second[0], camera.second[1], camera.second[2]};    // feature在imu的左目的像素坐标

                    shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                            host_pixel_coord,
                            other_pixel_coord
                    ));
                    edge_reproj->add_vertex(feature_node->vertex_landmark);
                    edge_reproj->add_vertex(host_imu_pose);
                    edge_reproj->add_vertex(feature_node->imu_deque[imu_i]->vertex_pose);
                    edge_reproj->add_vertex(_vertex_ext[other_camera_id]);
                    problem.add_edge(edge_reproj);
                }

            }

        }
#endif

        // 然后添加最新帧构成的顶点和边

        // 把imu顶点加入到problem中
        problem.add_vertex(_imu_node->vertex_pose);
        problem.add_vertex(_imu_node->vertex_motion);

        // 计算预积分edge
        if (!_windows.empty() && _imu_node->imu_integration && _imu_node->imu_integration->get_sum_dt() < 10.) {
            shared_ptr<EdgeImu> edge_imu(new EdgeImu(*_imu_node->imu_integration));
            edge_imu->add_vertex(_windows.newest()->vertex_pose);
            edge_imu->add_vertex(_windows.newest()->vertex_motion);
            edge_imu->add_vertex(_imu_node->vertex_pose);
            edge_imu->add_vertex(_imu_node->vertex_motion);
            problem.add_edge(edge_imu);
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
                    problem.add_vertex(vertex_landmark);

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
                        problem.add_edge(edge_reproj);
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
                            problem.add_edge(edge_reproj);
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
                        problem.add_edge(edge_reproj);
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
                    problem.add_vertex(feature_node->vertex_landmark);

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
                        problem.add_edge(edge_reproj);
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
        std::cout << "process_cost = " << process_cost << " ms" << std::endl;

        return problem;

    }

    void Estimator::solve_odometry() {
        if (_windows.full() && solver_flag == NON_LINEAR) {
            TicToc t_tri;

            // fix住最老的pose, 以保证可观
            _windows.oldest()->vertex_pose->set_fixed(true);

            _problem.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);

            // 求解非线性最小二乘问题
            _problem.solve(5);

            // 解锁最老的pose
            _windows.oldest()->vertex_pose->set_fixed(false);
        }

        remove_outlier_landmarks();
    }

    void Estimator::slide_window() {
        TicToc t_margin;

        std::cout << "_windows.size() = " << _windows.size() << std::endl;

        // TODO: 若初始化失败, 不能够直接进行margin, 因为chi2和jacobian都没有计算
        // 只有当windows满了才进行滑窗操作
        if (_windows.full()) {
            if (marginalization_flag == MARGIN_OLD) {
                std::cout << "MARGIN_OLD" << std::endl;

                // 弹出windows中最老的imu
                ImuNode *imu_oldest {nullptr};
                _windows.pop_oldest(imu_oldest);

                // 边缘化掉oldest imu。在margin时会把pose和motion的顶点从problem中删除，同时与pose和motion相关的边也会被全部删除
                _problem.marginalize(imu_oldest->vertex_pose, imu_oldest->vertex_motion);

                // 遍历被删除的imu的所有特征点，在特征点的imu队列中，删除该imu
                for (auto &feature_in_cameras : imu_oldest->features_in_cameras) {
                    auto &&feature_id = feature_in_cameras.first;
                    auto &&feature_it = _feature_map.find(feature_id);
                    if (feature_it == _feature_map.end()) {
//                        std::cout << "!!!!!!!! Can't find feature id in feature map when marg oldest !!!!!!!!!" << std::endl;
                        continue;
                    }

                    auto feature_node = feature_it->second;
                    auto vertex_landmark = feature_node->vertex_landmark;
                    auto &&imu_deque = feature_node->imu_deque;
                    // 应该是不需要进行判断的
//                    if (imu_deque.oldest() == imu_oldest) {
//                        imu_deque.pop_oldest();
//                    }
                    imu_deque.pop_oldest();

                    // 若特征点的keyframe小于2，则删除该特征点, 否则需要为特征点重新计算深度并且重新构建重投影edge,
                    if (imu_deque.size() < 2) {
                        // 在map中删除特征点
                        _feature_map.erase(feature_id);

                        if (vertex_landmark) {
                            // 在problem中删除特征点, 同时与之相关的边也会被删除
                            _problem.remove_vertex(vertex_landmark);
                        }

                        // 在新的oldest imu的features表中把当前feature删除, 不然feature依然会存在于oldest imu中, 但不在feature_map中
                        _windows.oldest()->features_in_cameras.erase(feature_id);

                        // 释放特征点node的空间
                        delete feature_node;
                    } else {
                        if (vertex_landmark) {
                            /*
                             * 1. 从旧的host imu还原出landmark的基于world系的3d坐标
                             * 2. 再投影到当前的host imu上，计算出landmark的新的逆深度
                             * 3. 把landmark所关联的重投影误差的旧的host imu修改为当前的host imu
                             * */

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

//                            // 对所有关于该特征点的edge的host imu进行修改
//                            auto &&edges_of_landmark = _problem.get_connected_edges(vertex_landmark);
//                            for (auto &edge : edges_of_landmark) {
//                                if (edge->type_info() == "EdgeReprojection") {
//                                    if (edge->vertices()[1] == edge->vertices()[2]) {   // 双目重投影
//                                        edge->vertices()[1] = host_imu_pose;
//                                        edge->vertices()[2] = host_imu_pose;
//                                    } else {    // 单目重投影
//                                        edge->vertices()[1] = host_imu_pose;
//                                    }
//
//                                    // 修改host_imu的像素坐标
//                                    auto edge_reproj = (EdgeReprojection *)edge.get();
//                                    edge_reproj->set_pt_i(host_pixel_coord);
//                                }
//                            }

                            // 删除landmark及其相关的edge
                            _problem.remove_vertex(vertex_landmark);

                            // 重新把landmark加入到problem中
                            _problem.add_vertex(vertex_landmark);

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
                                _problem.add_edge(edge_reproj);
                            }

                            // windows中的imu
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
                                    _problem.add_edge(edge_reproj);
                                }
                            }

                            // 当前imu
                            auto &&curr_cameras = _imu_node->features_in_cameras[feature_id];
                            auto &&curr_imu_pose = _imu_node->vertex_pose;

                            // 遍历所有imu
                            for (unsigned j = 0; j < curr_cameras.size(); ++j) {
                                auto &&curr_camera_id = curr_cameras[j].first; // camera的id
                                auto &&curr_pixel_coord = curr_cameras[j].second;    // feature在imu的左目的像素坐标

                                shared_ptr<EdgeReprojection> edge_reproj(new EdgeReprojection (
                                        host_pixel_coord,
                                        curr_pixel_coord
                                ));
                                edge_reproj->add_vertex(vertex_landmark);
                                edge_reproj->add_vertex(host_imu_pose);
                                edge_reproj->add_vertex(curr_imu_pose);
                                edge_reproj->add_vertex(_vertex_ext[curr_camera_id]);
                                _problem.add_edge(edge_reproj);
                            }
                        }
                    }
                }
                // 释放imu node的空间
                delete imu_oldest;
            } else {
                std::cout << "MARGIN_NEW" << std::endl;

                // 弹出windows中最新的imu
                ImuNode *imu_newest {nullptr};
                _windows.pop_newest(imu_newest);

                // 边缘化掉旧的newest, 同时在problem中删除其pose和motion顶点以及与之相关的预积分与重投影误差边
                _problem.marginalize(imu_newest->vertex_pose, imu_newest->vertex_motion);

                /* 1. 把_imu_node中的预积分值叠加到旧的newest的预积分中
                 * 2. 把_imu_node与旧的newest中的预积分指针进行交换
                 * 3. 构建新的预积分edge, 连接新的newest与_imu_node
                 * */
                if (_imu_node->imu_integration && imu_newest->imu_integration) {
                    const auto &curr_dt_buff = _imu_node->imu_integration->get_dt_buf();
                    const auto &curr_acc_buff = _imu_node->imu_integration->get_acc_buf();
                    const auto &curr_gyro_buff = _imu_node->imu_integration->get_gyro_buf();

                    // 把_imu_node中的预积分值叠加到旧的newest的预积分中
                    for (size_t i = 0; i < _imu_node->imu_integration->size(); ++i) {
                        imu_newest->imu_integration->push_back(curr_dt_buff[i], curr_acc_buff[i], curr_gyro_buff[i]);
                    }

                    // 把_imu_node与旧的newest中的预积分指针进行交换, 使得旧的newest的析构函数删除的是没交换前_imu_node的预积分
                    auto curr_imu_integration = _imu_node->imu_integration;
                    _imu_node->imu_integration = imu_newest->imu_integration;
                    imu_newest->imu_integration = curr_imu_integration;

                    // 构建新的预积分edge, 连接新的newest与_imu_node
                    if (_imu_node->imu_integration->get_sum_dt() < 10.) {
                        shared_ptr<EdgeImu> edge_imu(new EdgeImu(*_imu_node->imu_integration));
                        edge_imu->add_vertex(_windows.newest()->vertex_pose);
                        edge_imu->add_vertex(_windows.newest()->vertex_motion);
                        edge_imu->add_vertex(_imu_node->vertex_pose);
                        edge_imu->add_vertex(_imu_node->vertex_motion);
                        _problem.add_edge(edge_imu);
                    }
                }

                // 遍历被删除的imu的所有特征点，在特征点的imu队列中删除该imu
                for (auto &feature_in_cameras : imu_newest->features_in_cameras) {
                    auto &&feature_id = feature_in_cameras.first;
                    auto &&feature_it = _feature_map.find(feature_id);
                    if (feature_it == _feature_map.end()) {
                        std::cout << "!!!!!!!! Can't find feature id in feature map when marg newest !!!!!!!!!" << std::endl;
                        continue;
                    }
                    auto feature_node = feature_it->second;
                    auto vertex_landmark = feature_node->vertex_landmark;
                    auto &&imu_deque = feature_node->imu_deque;

                    // 移除feature的imu队列中的最新帧
                    imu_deque.pop_newest();

                    // 如果feature不在所有的imu中出现了，则需要删除feature
                    if (imu_deque.size() < 2
                        && _imu_node->features_in_cameras.find(feature_id) == _imu_node->features_in_cameras.end()) {

                        if (vertex_landmark) {
                            // 在problem中删除特征点, 同时把与之相关的edge删除
                            _problem.remove_vertex(vertex_landmark);
                        }

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

        auto margin_cost = t_margin.toc();
        std::cout << "margin_cost = " << margin_cost << " ms" << std::endl;
    }

    void Estimator::slide_window_only() {
        TicToc t_margin;

        std::cout << "_windows.size() = " << _windows.size() << std::endl;

        // TODO: 若初始化失败, 不能够直接进行margin, 因为chi2和jacobian都没有计算
        // 只有当windows满了才进行滑窗操作
        if (_windows.full()) {
            if (marginalization_flag == MARGIN_OLD) {
                std::cout << "MARGIN_OLD" << std::endl;

                // 弹出windows中最老的imu
                ImuNode *imu_oldest {nullptr};
                _windows.pop_oldest(imu_oldest);

                // 遍历被删除的imu的所有特征点，在特征点的imu队列中，删除该imu
                for (auto &feature_in_cameras : imu_oldest->features_in_cameras) {
                    auto &&feature_id = feature_in_cameras.first;
                    auto &&feature_it = _feature_map.find(feature_id);
                    if (feature_it == _feature_map.end()) {
//                        std::cout << "!!!!!!!! Can't find feature id in feature map when marg oldest !!!!!!!!!" << std::endl;
                        continue;
                    }

                    auto feature_node = feature_it->second;
                    auto vertex_landmark = feature_node->vertex_landmark;
                    auto &&imu_deque = feature_node->imu_deque;
                    // 应该是不需要进行判断的
//                    if (imu_deque.oldest() == imu_oldest) {
//                        imu_deque.pop_oldest();
//                    }
                    imu_deque.pop_oldest();

                    // 若特征点的keyframe小于2，则删除该特征点, 否则需要为特征点重新计算深度并且重新构建重投影edge,
                    if (imu_deque.size() < 2) {
                        // 在map中删除特征点
                        _feature_map.erase(feature_id);

                        // 在新的oldest imu的features表中把当前feature删除, 不然feature依然会存在于oldest imu中, 但不在feature_map中
                        _windows.oldest()->features_in_cameras.erase(feature_id);

                        // 释放特征点node的空间
                        delete feature_node;
                    }
                }
                // 释放imu node的空间
                delete imu_oldest;
            } else {
                std::cout << "MARGIN_NEW" << std::endl;

                // 弹出windows中最新的imu
                ImuNode *imu_newest {nullptr};
                _windows.pop_newest(imu_newest);

                /* 1. 把_imu_node中的预积分值叠加到旧的newest的预积分中
                 * 2. 把_imu_node与旧的newest中的预积分指针进行交换
                 * */
                if (_imu_node->imu_integration && imu_newest->imu_integration) {
                    const auto &curr_dt_buff = _imu_node->imu_integration->get_dt_buf();
                    const auto &curr_acc_buff = _imu_node->imu_integration->get_acc_buf();
                    const auto &curr_gyro_buff = _imu_node->imu_integration->get_gyro_buf();

                    // 把_imu_node中的预积分值叠加到旧的newest的预积分中
                    for (size_t i = 0; i < _imu_node->imu_integration->size(); ++i) {
                        imu_newest->imu_integration->push_back(curr_dt_buff[i], curr_acc_buff[i], curr_gyro_buff[i]);
                    }

                    // 把_imu_node与旧的newest中的预积分指针进行交换, 使得旧的newest的析构函数删除的是没交换前_imu_node的预积分
                    auto curr_imu_integration = _imu_node->imu_integration;
                    _imu_node->imu_integration = imu_newest->imu_integration;
                    imu_newest->imu_integration = curr_imu_integration;
                }


                // 遍历被删除的imu的所有特征点，在特征点的imu队列中删除该imu
                for (auto &feature_in_cameras : imu_newest->features_in_cameras) {
                    auto &&feature_id = feature_in_cameras.first;
                    auto &&feature_it = _feature_map.find(feature_id);
                    if (feature_it == _feature_map.end()) {
                        std::cout << "!!!!!!!! Can't find feature id in feature map when marg newest !!!!!!!!!" << std::endl;
                        continue;
                    }
                    auto feature_node = feature_it->second;
                    auto vertex_landmark = feature_node->vertex_landmark;
                    auto &&imu_deque = feature_node->imu_deque;

                    // 移除feature的imu队列中的最新帧
                    imu_deque.pop_newest();

                    // 如果feature不在所有的imu中出现了，则需要删除feature
                    if (imu_deque.size() < 2
                        && _imu_node->features_in_cameras.find(feature_id) == _imu_node->features_in_cameras.end()) {
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

        auto margin_cost = t_margin.toc();
        std::cout << "margin_cost = " << margin_cost << " ms" << std::endl;
    }

    bool Estimator::remove_outlier_landmarks() {
        std::vector<unsigned long> id_delete;
        id_delete.reserve(_feature_map.size());
        for (auto &feature_it : _feature_map) {
            if (feature_it.second->vertex_landmark && feature_it.second->vertex_landmark->get_parameters()[0] < 0.) {
                feature_it.second->is_outlier = true;
            }

            if (feature_it.second->is_outlier) {
                id_delete.emplace_back(feature_it.first);
            }
        }

        for (auto &id : id_delete) {
            if (_feature_map[id]->vertex_landmark) {
                _problem.remove_vertex(_feature_map[id]->vertex_landmark);
            }
            for (unsigned long i = 0; i < _windows.size(); ++i) {
                _windows[i]->features_in_cameras.erase(id);
            }
            _imu_node->features_in_cameras.erase(id);
            _feature_map.erase(id);

            std::cout << "remove outlier landmark: " << id << std::endl;
        }

        return true;

//        double chi2_th = 3.841;
//        unsigned int cnt_outlier, cnt_inlier;
//        for (unsigned int i = 0; i < iteration; ++i) {
//            cnt_outlier = 0;
//            cnt_inlier = 0;
//            for (const auto &edge : _problem.edges()) {
//                if (edge.second->type_info() == "EdgeReprojection") {
//                    if (edge.second->get_chi2() > chi2_th) {
//                        ++cnt_outlier;
//                    } else {
//                        ++cnt_inlier;
//                    }
//                }
//            }
//
//            double inlier_ratio = double(cnt_inlier) / double(cnt_outlier + cnt_inlier);
//            if (inlier_ratio > 0.5) {
//                break;
//            } else {
//                chi2_th *= 2.;
//            }
//        }
//
//        for (const auto &edge : _problem.edges()) {
//            if (edge.second->type_info() == "EdgeReprojection") {
//                if (edge.second->get_chi2() > chi2_th) {
//                    edge.second->vertices()[0]->id()
//                } else {
//                    ++cnt_inlier;
//                }
//            }
//        }
    }

    bool Estimator::remove_untriangulated_landmarks() {
        std::vector<unsigned long> id_delete;
        id_delete.reserve(_feature_map.size());
        for (auto &feature_it : _feature_map) {
            if (feature_it.second->vertex_landmark && !feature_it.second->is_triangulated) {
                id_delete.emplace_back(feature_it.first);
            }
        }

        for (auto &id : id_delete) {
            _problem.remove_vertex(_feature_map[id]->vertex_landmark);
            for (unsigned long i = 0; i < _windows.size(); ++i) {
                _windows[i]->features_in_cameras.erase(id);
            }
            _imu_node->features_in_cameras.erase(id);
            _feature_map.erase(id);

            std::cout << "remove untriangulated landmark: " << id << std::endl;
        }

        return true;
    }
}