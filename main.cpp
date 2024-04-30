#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include "modules/fdsi/p1id.h"
#include "modules/fdsi/fdsi.h"
//#include "modules/test_ba/vertex_pose.h"
//#include "modules/test_ba/vertex_inverse_depth.h"
//#include "modules/test_ba/edge_reprojection.h"
//#include "modules/vins/imu_integration.h"
#include "modules/vo_test/estimator.h"
#include "modules/vo_test/edge/edge_align.h"
#include "modules/vo_test/edge/edge_align_linear.h"
#include "modules/vo_test/vertex/vertex_vec1.h"
#include "modules/vo_test/vertex/vertex_scale.h"
#include "modules/vo_test/vertex/vertex_spherical.h"
#include "modules/vo_test/vertex/vertex_velocity.h"
#include "modules/vo_test/vertex/vertex_bias.h"
#include "lib/backend/loss_function.h"

using namespace graph_optimization;
using namespace std;

///*
// * FDSI
// * */
//int main() {
//    vector<double> w_buff {
//            6.28318530717959,	12.5663706143592,	18.8495559215388,	25.1327412287183,	31.4159265358979,
//            37.6991118430775,	43.9822971502571,	50.2654824574367,	56.5486677646163,	62.8318530717959,
//            69.1150383789754,	75.3982236861550,	81.6814089933346,	87.9645943005142,	94.2477796076938,
//            100.530964914873,	106.814150222053,	113.097335529233,	119.380520836412,	125.663706143592
//    };
//    vector<double> re_buff {
//            15.665461,	-6.879626,	-6.8117366,	-6.023029,	-5.014349,
//            -3.2953122,	-3.451121,	-3.0044954,	-2.3530235,	-1.9938891,
//            -1.6838235,	-1.5613447,	-1.3195838,	-1.1730264,	-1.0157101,
//            -0.8913852,	-0.72248876,	-0.6360515,	-0.46929023,	-0.23781322
//    };
//    vector<double> im_buff {
//            4.8768,	-20.388437,	-11.834236,	-6.9196825,	-4.2112164,
//            -1.6616025,	-1.6983799,	-1.1227595,	-0.8433019,	-0.42252827,
//            -0.27069244,	-0.16111729,	-0.032739427,	0.101014294,	0.11733885,
//            0.14699747,	0.27201325,	0.2992155,	0.3367225,	0.4298026
//    };
//    vector<unsigned > index_buff {
//        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
//    };
//
//    system_identification::frequency_domain::P1IDSolver p1id_solver;
//    p1id_solver(re_buff, im_buff, w_buff, index_buff);
//
//    vector<double> parameters {0.02, 200., 0.005};
//    system_identification::frequency_domain::FDSISolver<1, 0, 1> fdsi_solver(parameters);
//    fdsi_solver(re_buff, im_buff, w_buff, index_buff);
//
//    return 0;
//}


///*
// * 预积分
// * */
//int main() {
//    vins::IMUIntegration imu_integration(Vec3::Zero(), Vec3::Zero());
//    double dt = 0.005;
//    Vec3 a {0.1, 0.2, 0.3};
//    Vec3 w {-0.1, -0.2, -0.3};
//
//    imu_integration.propagate(dt, a, w);
//    imu_integration.propagate(dt, a, w);
//    imu_integration.propagate(dt, a, w);
//    imu_integration.propagate(dt, a, w);
//    imu_integration.propagate(dt, a, w);
//
//    cout << imu_integration.get_delta_r().matrix() << endl;
//    cout << imu_integration.get_delta_p() << endl;
//    cout << imu_integration.get_delta_v() << endl;
//
//    Vec3 ba {0.01, 0.02, 0.03};
//    Vec3 bg {-0.01, -0.02, -0.03};
//    imu_integration.correct(ba, bg);
//    cout << imu_integration.get_delta_r().matrix() << endl;
//    cout << imu_integration.get_delta_p() << endl;
//    cout << imu_integration.get_delta_v() << endl;
//
//    return 0;
//}

///*
// * SLAM
// * */
//
///*
// * Frame : 保存每帧的姿态和观测
// */
//struct Frame {
//    Frame(const Eigen::Matrix3d& R, const Eigen::Vector3d &t) : Rwc(R), qwc(R), twc(t) {};
//    Eigen::Matrix3d Rwc;
//    Eigen::Quaterniond qwc;
//    Eigen::Vector3d twc;
//
//    unordered_map<int, Eigen::Vector3d> featurePerId; // 该帧观测到的特征以及特征id
//};
//
//class FeatureLocalInfo {
//public:
//    explicit FeatureLocalInfo(Vec2 pix) : uv(pix) {}
//
//    Vec3 point;
//    Vec2 uv;
//};
//
//class FeatureGlobalInfo {
//public:
//    explicit FeatureGlobalInfo(unsigned long id) : feature_id(id) {}
//
//    const unsigned long feature_id;
//    unsigned long start_frame{};
//    vector<FeatureLocalInfo> local_infos;
//};
//
///*
// * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
// */
//void GetSimDataInWordFrame(vector<Frame> &cameraPoses, vector<Eigen::Vector3d> &points) {
//    int featureNums = 20;  // 特征数目，假设每帧都能观测到所有的特征
//    int poseNums = 4;     // 相机数目
//
//    double radius = 8;
//    for (int n = 0; n < poseNums; ++n) {
//        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 圆弧
//        // 绕 z轴 旋转
//        Eigen::Matrix3d R;
//        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
//        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
//        cameraPoses.emplace_back(R, t);
//    }
//
//    // 随机数生成三维特征点
//    std::default_random_engine generator;
//    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
//    for (int j = 0; j < featureNums; ++j) {
//        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
//        std::uniform_real_distribution<double> z_rand(4., 8.);
//
//        Eigen::Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
//        points.push_back(Pw);
//
//        // 在每一帧上的观测量
//        for (int i = 0; i < poseNums; ++i) {
//            Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
//            Pc = Pc / Pc.z();  // 归一化图像平面
//            Pc[0] += noise_pdf(generator);
//            Pc[1] += noise_pdf(generator);
//            cameraPoses[i].featurePerId.insert(make_pair(j, Pc));
//        }
//    }
//}
//
//void generate_sim_data(vector<Frame> &cameraPoses, vector<Eigen::Vector3d> &points,
//                        vector<FeatureGlobalInfo> &feature_global_buff) {
//    int featureNums = 20;  // 特征数目，假设每帧都能观测到所有的特征
//    int poseNums = 4;     // 相机数目
//
//    double radius = 8;
//    for (int n = 0; n < poseNums; ++n) {
//        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 圆弧
//        // 绕 z轴 旋转
//        Eigen::Matrix3d R;
//        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
//        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
//        cameraPoses.emplace_back(R, t);
//    }
//
//    // 随机数生成三维特征点
//    std::default_random_engine generator;
//    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
//    for (int j = 0; j < featureNums; ++j) {
//        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
//        std::uniform_real_distribution<double> z_rand(4., 8.);
//
//        Eigen::Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
//        points.push_back(Pw);
//
//        feature_global_buff.emplace_back(j);
//        feature_global_buff[j].start_frame = 2 * (j / 5);
//    }
//
//    // 在每一帧上的观测量
//    int j = 0;
//    for (int i = 0; i < poseNums; ++i) {
//        if (i % 2 == 1) {
//            ++j;
//        }
//
//        for (int k = j; k < featureNums && k < j + 5; ++k) {
//            Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose() * (points[k] - cameraPoses[i].twc);
//            Vec2 Pc_xy = {Pc[0] / Pc[2], Pc[1] / Pc[2]};    // 归一化图像平面
//            Pc_xy[0] += noise_pdf(generator);
//            Pc_xy[1] += noise_pdf(generator);
//
//            cameraPoses[i].featurePerId.insert(make_pair(k, Vec3(Pc_xy[0], Pc_xy[1], 1.)));
//
//            feature_global_buff[k].local_infos.emplace_back(Pc_xy);
//            feature_global_buff[k].local_infos[feature_global_buff[k].local_infos.size() - 1].point = Pc;
//        }
//
//    }
//}
//
//int main() {
//    // 准备数据
//    vector<Frame> cameras;
//    vector<Eigen::Vector3d> points;
//    vector<FeatureGlobalInfo> feature_global_buff;
//    GetSimDataInWordFrame(cameras, points);
////    generate_sim_data(cameras, points, feature_global_buff);
//    Eigen::Quaterniond qic(1, 0, 0, 0);
//    Eigen::Vector3d tic(0, 0, 0);
//
//    // 构建 problem
//    Problem problem(Problem::ProblemType::SLAM_PROBLEM);
//
//    // 噪声分布
//    std::default_random_engine generator;
//    std::normal_distribution<double> noise_pdf(0, 1.);
//
//    // 所有 Pose
//    vector<shared_ptr<VertexPose> > vertexCams_vec;
//    for (size_t i = 0; i < cameras.size(); ++i) {
//        shared_ptr<VertexPose> vertexCam(new VertexPose());
//        Vec7 pose;
//        pose << cameras[i].twc, cameras[i].qwc.x(), cameras[i].qwc.y(), cameras[i].qwc.z(), cameras[i].qwc.w();
////        pose << 0., 0., 0., 0., 0., 0., 1.;
//        vertexCam->set_parameters(pose);
//
//        if(i < 2) {
//            vertexCam->set_fixed();
//        }
//
//        problem.add_vertex(vertexCam);
//        vertexCams_vec.push_back(vertexCam);
//    }
//
//    // 所有 Point 及 edge
//    double noise = 0;
//    vector<double> noise_invd;
//    vector<shared_ptr<VertexInverseDepth> > allPoints;
//    for (size_t i = 0; i < points.size(); ++i) {
//        //假设所有特征点的起始帧为第0帧， 逆深度容易得到
//        Eigen::Vector3d Pw = points[i];
//        Eigen::Vector3d Pc = cameras[0].Rwc.transpose() * (Pw - cameras[0].twc);
//        noise = noise_pdf(generator);
//        double inverse_depth = 1. / (Pc.z() + noise);
////        double inverse_depth = 1. / Pc.z();
//        noise_invd.push_back(inverse_depth);
//
//        // 初始化特征 vertex
//        shared_ptr<VertexInverseDepth> verterxPoint(new VertexInverseDepth());
//        VecX inv_d(1);
//        inv_d << inverse_depth;
//        verterxPoint->set_parameters(inv_d);
//        problem.add_vertex(verterxPoint);
//        allPoints.push_back(verterxPoint);
//
//        // 每个特征对应的投影误差, 第 0 帧为起始帧
//        if (i < 10) {
//            for (size_t j = 1; j < cameras.size(); ++j) {
//                Eigen::Vector3d pt_i = cameras[0].featurePerId.find(i)->second;
//                Eigen::Vector3d pt_j = cameras[j].featurePerId.find(i)->second;
//                shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
//                edge->set_translation_imu_from_camera(qic, tic);
//
//                std::vector<std::shared_ptr<Vertex> > edge_vertex;
//                edge_vertex.push_back(verterxPoint);
//                edge_vertex.push_back(vertexCams_vec[0]);
//                edge_vertex.push_back(vertexCams_vec[j]);
//                edge->set_vertices(edge_vertex);
//
//                problem.add_edge(edge);
//            }
//        } else {
//            for (size_t j = 2; j < cameras.size(); ++j) {
//                Eigen::Vector3d pt_i = cameras[1].featurePerId.find(i)->second;
//                Eigen::Vector3d pt_j = cameras[j].featurePerId.find(i)->second;
//                shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
//                edge->set_translation_imu_from_camera(qic, tic);
//
//                std::vector<std::shared_ptr<Vertex> > edge_vertex;
//                edge_vertex.push_back(verterxPoint);
//                edge_vertex.push_back(vertexCams_vec[1]);
//                edge_vertex.push_back(vertexCams_vec[j]);
//                edge->set_vertices(edge_vertex);
//
//                problem.add_edge(edge);
//            }
//        }
//
//    }
//
////    Eigen::Vector3d pt_i;
////    Eigen::Vector3d pt_j;
////    for (std::size_t j = 0; j < 5; ++j) {
////        pt_i = cameras[0].featurePerId.find(j)->second;
////        pt_j = cameras[1].featurePerId.find(j)->second;
////        shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
////        edge->set_translation_imu_from_camera(qic, tic);
////        std::vector<std::shared_ptr<Vertex> > edge_vertex;
////        edge_vertex.push_back(allPoints[j]);
////        edge_vertex.push_back(vertexCams_vec[0]);
////        edge_vertex.push_back(vertexCams_vec[1]);
////        edge->set_vertices(edge_vertex);
////        problem.add_edge(edge);
////
////        if (j > 0) {
////            pt_i = cameras[0].featurePerId.find(j)->second;
////            pt_j = cameras[2].featurePerId.find(j)->second;
////            shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
////            edge->set_translation_imu_from_camera(qic, tic);
////            std::vector<std::shared_ptr<Vertex> > edge_vertex;
////            edge_vertex.push_back(allPoints[j]);
////            edge_vertex.push_back(vertexCams_vec[0]);
////            edge_vertex.push_back(vertexCams_vec[2]);
////            edge->set_vertices(edge_vertex);
////            problem.add_edge(edge);
////        }
////
////        if (j > 0) {
////            pt_i = cameras[0].featurePerId.find(j)->second;
////            pt_j = cameras[3].featurePerId.find(j)->second;
////            shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
////            edge->set_translation_imu_from_camera(qic, tic);
////            std::vector<std::shared_ptr<Vertex> > edge_vertex;
////            edge_vertex.push_back(allPoints[j]);
////            edge_vertex.push_back(vertexCams_vec[0]);
////            edge_vertex.push_back(vertexCams_vec[3]);
////            edge->set_vertices(edge_vertex);
////            problem.add_edge(edge);
////        }
////    }
////    pt_i = cameras[2].featurePerId.find(5)->second;
////    pt_j = cameras[3].featurePerId.find(5)->second;
////    shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
////    edge->set_translation_imu_from_camera(qic, tic);
////    std::vector<std::shared_ptr<Vertex> > edge_vertex;
////    edge_vertex.push_back(allPoints[5]);
////    edge_vertex.push_back(vertexCams_vec[2]);
////    edge_vertex.push_back(vertexCams_vec[3]);
////    edge->set_vertices(edge_vertex);
////    problem.add_edge(edge);
//
//    problem.solve(10);
//
//    std::cout << "\nCompare MonoBA results after opt..." << std::endl;
//    for (size_t k = 0; k < 6; k+=1) {
//        std::cout << "after opt, point " << k << " : gt " << 1. / points[k].z() << " ,noise "
//                  << noise_invd[k] << " ,opt " << allPoints[k]->get_parameters() << std::endl;
//    }
//    std::cout<<"------------ pose translation ----------------"<<std::endl;
//    for (int i = 0; i < vertexCams_vec.size(); ++i) {
//        std::cout<<"translation after opt: "<< i <<" :"<< vertexCams_vec[i]->get_parameters().head(3).transpose() << " || gt: "<<cameras[i].twc.transpose()<<std::endl;
//    }
//    /// 优化完成后，第一帧相机的 pose 平移（x,y,z）不再是原点 0,0,0. 说明向零空间发生了漂移。
//    /// 解决办法： fix 第一帧和第二帧，固定 7 自由度。 或者加上非常大的先验值。
//
//    problem.marginalize(vertexCams_vec[0], nullptr);
//
//    cout << problem.get_h_prior() << endl;
//
//    auto &&edge_buff = problem.get_connected_edges(vertexCams_vec[1]);
//    for (auto &edge : edge_buff) {
//        for (auto &vertex : edge->vertices()) {
//            cout << vertex->id() << endl;
//        }
//    }
//
//    problem.test_marginalize();
//
//    Eigen::LDLT<Eigen::MatrixXd> ldlt(problem.get_h_prior());
//    ldlt.vectorD().asDiagonal() * ldlt.matrixL().;
//
//    return 0;
//}


/*
 * VO
 * */

static unordered_map<unsigned long, Vec3> landmark_map;

constexpr static unsigned int num_steps = 20;
static Qd q_per_imu[num_steps];
static Vec3 t_per_imu[num_steps];
static unordered_map<unsigned long, vector<pair<unsigned long, Vec3>>> landmarks_per_imu[num_steps];
static unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> f_per_imu[num_steps];

constexpr static unsigned int nun_cameras = 1;
static Qd q_ic[nun_cameras];
static Vec3 t_ic[nun_cameras];

static void sim_data() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-3., 3.);

    q_ic[0] = {cos(0.5 * double(EIGEN_PI) / 2.), -sin(0.5 * double(EIGEN_PI) / 2.), 0., 0.};
    t_ic[0] = {0., 0., 0.};

    double r = 10.;
    int ang_int = 3600;
    for (unsigned int i = 0; i < num_steps; ++i) {
        // 产生位姿
        double theta = double(ang_int) / 10. / 180. * double(EIGEN_PI);
        q_per_imu[i] = {cos(0.5 * theta), 0., 0., sin(0.5 * theta)};
        t_per_imu[i] = {r * sin(theta), r - r * cos(theta), 0.};

        for (int j = -200; j <= 200; ++j) {
            // 产生特征点
            auto ang_int_tmp = ang_int + j;
            unsigned long landmark_index = ang_int_tmp;
            Vec3 p_w;
            auto landmark_it = landmark_map.find(landmark_index);
            if (landmark_it == landmark_map.end()) {
                double ang_tmp = double(ang_int_tmp) / 10. / 180. * double(EIGEN_PI);
                double z = (landmark_index % 2 == 0) ? 1. : -1.;
//                double z = 0.;
                p_w = {-r * sin(ang_tmp) + dist(gen), r + r * cos(ang_tmp) + dist(gen),  dist(gen)};
                landmark_map.insert(make_pair(landmark_index, p_w));
            } else {
                p_w = landmark_it->second;
            }

            // 记录每帧的特征点
            landmarks_per_imu[i][landmark_index].emplace_back(0, p_w);

            Vec3 p_i = q_per_imu[i].inverse() * (p_w - t_per_imu[i]);
            Vec3 p_c = q_ic[0].inverse() * (p_i - t_ic[0]);
            p_c /= p_c.z();
            Vec7 f;
            f << p_c.x(), p_c.y(), 1., 0., 0., 0., 0.;
            f_per_imu[i][landmark_index].emplace_back(0, f);

//            std::cout << "feature: " << p_c << std::endl;
        }
        ang_int += 30;

//        std::cout << "angle = " << theta / double(EIGEN_PI) * 180. << std::endl;
    }
}

class ImuAlignTest {
public:
    explicit ImuAlignTest(double dt, double w=0.5, double r=10.) : _dt(dt), _w(w), _r(r) {
        _ba.setZero();
        _bg.setZero();
    }

    void generate_data(unsigned int num_data) {
        _theta_buff.resize(num_data);
        _p_buff.resize(num_data);
        _v_buff.resize(num_data);
        _a_buff.resize(num_data);
        _w_buff.resize(num_data);
        for (unsigned int i = 0; i < num_data; ++i) {
            _theta_buff[i] = double(i) * _dt * _w;

            _p_buff[i].x() = _r * cos(_theta_buff[i]);
            _p_buff[i].y() = _r * sin(_theta_buff[i]);
            _p_buff[i].z() = 0.;

            _v_buff[i].x() = -_r * _w * sin(_theta_buff[i]);
            _v_buff[i].y() = _r * _w * cos(_theta_buff[i]);
            _v_buff[i].z() = 0.;

            _a_buff[i].x() = -_r * _w * _w + _ba.x();
            _a_buff[i].y() = 0. + _ba.y();
            _a_buff[i].z() = 9.8 + _ba.z();

            _w_buff[i].x() = 0. + _bg.x();
            _w_buff[i].y() = 0. + _bg.y();
            _w_buff[i].z() = _w + _bg.z();
        }
    }

public:
    vector<double> _theta_buff;
    vector<Vec3> _p_buff;
    vector<Vec3> _v_buff;
    vector<Vec3> _a_buff;
    vector<Vec3> _w_buff;

public:
    Vec3 _ba;
    Vec3 _bg;

public:
    double _dt;
    double _w;
    double _r;

};

int main() {
    sim_data();

    unsigned int ref_index = 5;

    Qd q_wc[num_steps];
    Vec3 t_wc[num_steps];
    for (unsigned int i = 0; i < num_steps; ++i) {
        q_wc[i] = q_per_imu[i] * q_ic[0];
        t_wc[i] = t_per_imu[i] + q_per_imu[i] * t_ic[0];
    }
    Vec3 t_5_to_10_on_5 = q_per_imu[ref_index].inverse() * (t_per_imu[10] - t_per_imu[ref_index]);
    Qd q_5_10 = q_per_imu[ref_index].inverse() * q_per_imu[10];
    std::cout << t_5_to_10_on_5 << std::endl;
    std::cout << q_5_10.w() << ", " << q_5_10.x() << ", " << q_5_10.y() << ", " << q_5_10.z() << std::endl;

    vins::Estimator estimator;

    for (unsigned int i = 0; i < estimator._windows.capacity() + 1; ++i) {
        estimator.process_image(f_per_imu[i], 0.);
    }

    for (unsigned int i = 0; i < estimator._windows.size(); ++i) {
        Vec7 pose = estimator._windows[i]->vertex_pose->get_parameters();
        Vec3 t {pose(0), pose(1), pose(2)};
        Qd q {pose(6), pose(3), pose(4), pose(5)};
//        t = t_per_imu[ref_index] + q_per_imu[ref_index] * t * (t_per_imu[ref_index] - t_per_imu[estimator._windows.capacity()]).norm();
//        q = q_per_imu[ref_index] * q;
        t = t_per_imu[0] + q_per_imu[0] * t * (t_per_imu[ref_index] - t_per_imu[estimator._windows.capacity()]).norm();
        q = q_per_imu[0] * q;

        std::cout << "ground truth: " << std::endl;
        std::cout << "t = " << t_per_imu[i].transpose() << std::endl;
        std::cout << "q = " << q_per_imu[i].w() << ", "  << q_per_imu[i].x() << ", "  << q_per_imu[i].y() << ", "  << q_per_imu[i].z() << std::endl;
        std::cout << "estimate: " << std::endl;
        std::cout << "t = " << t.transpose() << std::endl;
        std::cout << "q = " << q.w() << ", "  << q.x() << ", "  << q.y() << ", "  << q.z() << std::endl;
    }

//    for (auto &feature_in_cameras : estimator._windows[5]->features_in_cameras) {
//        auto &&feature_it = estimator._feature_map.find(feature_in_cameras.first);
//        if (feature_it == estimator._feature_map.end()) {
//            continue;
//        }
//        if (feature_it->second->vertex_point3d) {
//            Vec3 p_est = feature_it->second->vertex_point3d->get_parameters();
//            p_est = t_per_imu[5] + q_per_imu[5] * p_est;
//            Vec3 p_gt = landmark_map[feature_it->first];
//
//            std::cout << "gt : " << p_gt.transpose() << std::endl;
//            std::cout << "est: " << p_est.transpose() << std::endl;
//        }
//    }

    static ImuAlignTest imu_align(0.005);
    imu_align.generate_data(201);

    Vec3 ba_init {0., 0., 0.};
    Vec3 bg_init {0., 0., 0.};
    double scale = 10.;
    Qd q0 {cos(0.5 * 0.2), sin(0.5 * 0.2), 0., 0.};

    vector<Vec3> t;
    vector<Qd> q;
    vector<Vec3> v;

    Problem problem;
    shared_ptr<VertexScale> vertex_scale = make_shared<VertexScale>();
    shared_ptr<VertexSpherical> vertex_q_wb0 = make_shared<VertexSpherical>();
    shared_ptr<VertexBias> vertex_ba = make_shared<VertexBias>();
    shared_ptr<VertexBias> vertex_bg = make_shared<VertexBias>();
    vector<shared_ptr<VertexVelocity>> vertices_v;
    shared_ptr<vins::IMUIntegration> imu_integration = nullptr;

    problem.add_vertex(vertex_scale);
    problem.add_vertex(vertex_q_wb0);
    problem.add_vertex(vertex_ba);
    problem.add_vertex(vertex_bg);

    Problem problem_linear;
    shared_ptr<VertexVec1> vertex_linear_scale = make_shared<VertexVec1>();
    shared_ptr<VertexBias> vertex_g_b0 = make_shared<VertexBias>();
    vertex_linear_scale->parameters().setZero();
//    vertex_linear_scale->set_parameters(Vec1(1.));
    vertex_g_b0->parameters().setZero();
//    vertex_g_b0->set_parameters(imu_integration->get_gravity());
    problem_linear.add_vertex(vertex_linear_scale);
    problem_linear.add_vertex(vertex_g_b0);
    shared_ptr<TrivialLoss> trivial_loss = make_shared<TrivialLoss>();

    Eigen::Matrix<double, 37, 37> A;
    Eigen::Matrix<double, 37, 1> b;
    for (unsigned int i = 0; i < 201; ++i) {
        if (imu_integration) {
//            std::cout << "1: " << imu_integration->get_delta_p().transpose() << std::endl;
//            std::cout << "2: " << imu_integration->get_delta_v().transpose() << std::endl;
            imu_integration->push_back(imu_align._dt, imu_align._a_buff[i], imu_align._w_buff[i]);
        }

        if (i % 20 == 0) {
            t.emplace_back(imu_align._p_buff[i]);
            q.emplace_back(cos(0.5 * imu_align._theta_buff[i]), 0., 0., sin(0.5 * imu_align._theta_buff[i]));
            v.emplace_back(imu_align._v_buff[i]);

            Vec3 v_init = imu_align._v_buff[i];
            vertices_v.emplace_back(make_shared<VertexVelocity>());
            vertices_v.back()->parameters().setZero();
//            vertices_v.back()->set_parameters(v_init);

            // 非线性
            problem.add_vertex(vertices_v.back());

            // 线性
            problem_linear.add_vertex(vertices_v.back());

            if (imu_integration) {
                Vec3 ti = q0.inverse() * t[t.size() - 2];
                Vec3 tj = q0.inverse() * t[t.size() - 1];
                Qd qi = (q0.inverse() * q[q.size() - 2]).normalized();
                Qd qj = (q0.inverse() * q[q.size() - 1]).normalized();

                Vec3 tij = (tj - ti) / scale;
                Qd qij = (qi.inverse() * qj).normalized();

                // 非线性
//                Vec3 v_gauss = tij / imu_integration->get_sum_dt();
//                vertices_v[vertices_v.size() - 2]->set_parameters(v_gauss);
//                vertices_v[vertices_v.size() - 1]->set_parameters(v_gauss);
                shared_ptr<EdgeAlign> edge = make_shared<EdgeAlign>(*imu_integration,
                                                                    tij,
                                                                    qij,
                                                                    qi);
                edge->add_vertex(vertex_scale);
                edge->add_vertex(vertex_q_wb0);
                edge->add_vertex(vertices_v[vertices_v.size() - 2]);
                edge->add_vertex(vertices_v[vertices_v.size() - 1]);
                edge->add_vertex(vertex_ba);
                edge->add_vertex(vertex_bg);
                problem.add_edge(edge);

                // 线性
                shared_ptr<EdgeAlignLinear> edge_linear = make_shared<EdgeAlignLinear>(*imu_integration,
                                                                                       tij,
                                                                                       qij,
                                                                                       qi);
                edge_linear->set_loss_function(trivial_loss);
                edge_linear->add_vertex(vertex_linear_scale);
                edge_linear->add_vertex(vertex_g_b0);
                edge_linear->add_vertex(vertices_v[vertices_v.size() - 2]);
                edge_linear->add_vertex(vertices_v[vertices_v.size() - 1]);
                problem_linear.add_edge(edge_linear);

//                {
//                    Vec9 residual;
//                    Vec3 v_i = v[v.size() - 2];
//                    Vec3 v_j = v[v.size() - 1];
//                    double dt = imu_integration->get_sum_dt();
//                    Vec3 g_b0 = q0.inverse() * imu_integration->get_gravity();
//                    residual.head<3>() = qi.inverse() * (scale * tij - (v_i + 0.5 * dt * g_b0) * dt) - imu_integration->get_delta_p();
//                    residual.segment<3>(3) = 2. * (imu_integration->get_delta_q().inverse() * qij).vec();
//                    residual.tail<3>() = qi.inverse() * (v_j - (v_i + g_b0 * dt)) - imu_integration->get_delta_v();
//                    std::cout << "residual: " << residual << std::endl;
//                }

                imu_integration = make_shared<vins::IMUIntegration>(imu_align._a_buff[i - 1],
                                                                    imu_align._w_buff[i - 1],
                                                                    ba_init,
                                                                    bg_init);
            } else {
                imu_integration = make_shared<vins::IMUIntegration>(imu_align._a_buff[i],
                                                                    imu_align._w_buff[i],
                                                                    ba_init,
                                                                    bg_init);
            }
        }
    }

    // 线性
    problem_linear.set_solver_type(graph_optimization::Problem::SolverType::LEVENBERG_MARQUARDT);
    problem_linear.solve(15);

    Vec3 g_b0 = vertex_g_b0->get_parameters();
//    Vec3 v_nav_est = (q0.inverse() * q[0]).normalized() * g_b0;
    Vec3 v_nav_est = g_b0;

    double norm2 = sqrt(v_nav_est.squaredNorm() * imu_integration->get_gravity().squaredNorm());
    double cos_psi = v_nav_est.dot(imu_integration->get_gravity());
    Vec3 sin_psi = v_nav_est.cross(imu_integration->get_gravity());
    Qd q0_est(norm2 + cos_psi, sin_psi(0), sin_psi(1), sin_psi(2));
    q0_est.normalize();

    std::cout << "linear scale: " << vertex_linear_scale->get_parameters().transpose() << std::endl;
    std::cout << "linear g_b0: " << vertex_g_b0->get_parameters().transpose() << std::endl;
    std::cout << "linear q_wb0: " << q0_est.w() << ", " << q0_est.x() << ", " << q0_est.y() << ", " << q0_est.z() << std::endl;

    // 线性转非线性
    vertex_scale->set_parameters(vertex_linear_scale->get_parameters());
    Vec4 q_wb0_est {q0_est.w(), q0_est.x(), q0_est.y(), q0_est.z()};
    vertex_q_wb0->set_parameters(q_wb0_est);
    for (auto &vertex : vertices_v) {
        vertex->set_parameters(vertex->get_parameters() / vertex_linear_scale->get_parameters()[0]);
    }

    // 非线性
    problem.set_solver_type(graph_optimization::Problem::SolverType::LBFGS);
    vertex_ba->set_fixed();
    vertex_bg->set_fixed();
    problem.solve(100);
    std::cout << "nonlinear scale: " << vertex_scale->get_parameters().transpose() << std::endl;
    std::cout << "nonlinear q_wb0: " << vertex_q_wb0->get_parameters().transpose() << std::endl;
    std::cout << "nonlinear ba: " << vertex_ba->get_parameters().transpose() << std::endl;
    std::cout << "nonlinear bg: " << vertex_bg->get_parameters().transpose() << std::endl;

    // Ground truth
    std::cout << "ground truth scale: " << scale << std::endl;
    std::cout << "ground truth q_wb0: " << q0.w() << ", " << q0.x() << ", " << q0.y() << ", " << q0.z() << std::endl;

    return 0;
}