#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "modules/fdsi/p1id.h"
#include "modules/fdsi/fdsi.h"
#include "modules/test_ba/vertex_pose.h"
#include "modules/test_ba/vertex_inverse_depth.h"
#include "modules/test_ba/edge_reprojection.h"

using namespace graph_optimization;
using namespace std;

/*
 * FDSI
 * */
int main() {
    vector<double> w_buff {
            6.28318530717959,	12.5663706143592,	18.8495559215388,	25.1327412287183,	31.4159265358979,
            37.6991118430775,	43.9822971502571,	50.2654824574367,	56.5486677646163,	62.8318530717959,
            69.1150383789754,	75.3982236861550,	81.6814089933346,	87.9645943005142,	94.2477796076938,
            100.530964914873,	106.814150222053,	113.097335529233,	119.380520836412,	125.663706143592
    };
    vector<double> re_buff {
            15.665461,	-6.879626,	-6.8117366,	-6.023029,	-5.014349,
            -3.2953122,	-3.451121,	-3.0044954,	-2.3530235,	-1.9938891,
            -1.6838235,	-1.5613447,	-1.3195838,	-1.1730264,	-1.0157101,
            -0.8913852,	-0.72248876,	-0.6360515,	-0.46929023,	-0.23781322
    };
    vector<double> im_buff {
            4.8768,	-20.388437,	-11.834236,	-6.9196825,	-4.2112164,
            -1.6616025,	-1.6983799,	-1.1227595,	-0.8433019,	-0.42252827,
            -0.27069244,	-0.16111729,	-0.032739427,	0.101014294,	0.11733885,
            0.14699747,	0.27201325,	0.2992155,	0.3367225,	0.4298026
    };
    vector<unsigned > index_buff {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    };

    system_identification::frequency_domain::P1IDSolver p1id_solver;
    p1id_solver(re_buff, im_buff, w_buff, index_buff);

    vector<double> parameters {0.02, 200., 0.005};
    system_identification::frequency_domain::FDSISolver<1, 0, 1> fdsi_solver(parameters);
    fdsi_solver(re_buff, im_buff, w_buff, index_buff);

    return 0;
}

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