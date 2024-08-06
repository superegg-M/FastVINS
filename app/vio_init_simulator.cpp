//
// Created by Cain on 2024/5/22.
//

#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <map>

#include "../modules/vo_test/estimator.h"
#include "../modules/vo_test/edge/edge_align.h"
#include "../modules/vo_test/edge/edge_align_linear.h"
#include "../modules/vo_test/vertex/vertex_vec1.h"
#include "../modules/vo_test/vertex/vertex_scale.h"
#include "../modules/vo_test/vertex/vertex_spherical.h"
#include "../modules/vo_test/vertex/vertex_velocity.h"
#include "../modules/vo_test/vertex/vertex_bias.h"
#include "../lib/backend/loss_function.h"


using namespace graph_optimization;
using namespace std;


struct MotionData1
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double timestamp;
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;
    Eigen::Vector3d imu_acc;
    Eigen::Vector3d imu_gyro;

    Eigen::Vector3d imu_gyro_bias;
    Eigen::Vector3d imu_acc_bias;

    Eigen::Vector3d imu_velocity;
};

struct FeatureData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int feature_id;
    double x;
    double y;
    double z = 1;
    double p_u;
    double p_v;
    double velocity_x = 0;
    double velocity_y = 0;
};

struct GtData {
    double timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond rotation;
    Eigen::Vector3d velocity;
    Eigen::Vector3d bias_gyr;
    Eigen::Vector3d bias_acc;
};

unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> LoadFeature(std::string filename)
{

    std::ifstream f;
    f.open(filename.c_str());

    int feature_id=0;
    unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> landmarks_map;

    if(!f.is_open())
    {
        std::cerr << " can't open LoadFeatures file "<<std::endl;
        // return ;
    }

    while (!f.eof()) {

        std::string s;
        std::getline(f,s);


        if(! s.empty())
        {
            std::stringstream ss;
            ss << s;

            FeatureData data;
            Eigen::Vector4d point;
            Eigen::Vector4d obs;

            ss>>point[0];
            ss>>point[1];
            ss>>point[2];
            ss>>point[3];
            ss>>obs[0];
            ss>>obs[1];

            // 行号即id
            data.feature_id=feature_id;
            feature_id++;

            data.x=obs[0];
            data.y=obs[1];
            data.z=1;
            data.p_u=(obs[0]*460 + 255);
            data.p_v=(obs[1]*460 + 255);
            // feature.push_back(data);
            Vec7 f;
            f<<data.x,data.y,data.z,data.p_u,data.p_v,0,0;
            landmarks_map[data.feature_id].emplace_back(0,f);

        }
    }

    return landmarks_map;

}

void LoadPose(std::string filename, std::vector<MotionData1>& pose, vector<GtData>& gt)
{

    std::ifstream f;
    f.open(filename.c_str());

    if(!f.is_open())
    {
        std::cerr << " can't open LoadPoses file "<<std::endl;
        return;
    }

    while (!f.eof()) {

        std::string s;
        std::getline(f,s);

        if(! s.empty())
        {
            std::stringstream ss;
            ss << s;

            MotionData1 data;
            GtData gtdata;
            double time;
            Eigen::Quaterniond q;
            Eigen::Vector3d t;
            Eigen::Vector3d gyro;
            Eigen::Vector3d acc;

            ss>>time;
            ss>>q.w();
            ss>>q.x();
            ss>>q.y();
            ss>>q.z();
            ss>>t(0);
            ss>>t(1);
            ss>>t(2);
            ss>>gyro(0);
            ss>>gyro(1);
            ss>>gyro(2);
            ss>>acc(0);
            ss>>acc(1);
            ss>>acc(2);


            data.timestamp = time;;
            data.imu_gyro = gyro;
            data.imu_acc = acc;
            data.twb = t;
            data.Rwb = Eigen::Matrix3d(q);
            pose.push_back(data);

            gtdata.timestamp=time;
            gtdata.position=t;
            gtdata.rotation=q;
            gtdata.velocity=Eigen::Vector3d::Zero();
            gtdata.bias_acc=Eigen::Vector3d::Zero();
            gtdata.bias_gyr=Eigen::Vector3d::Zero();

            gt.push_back(gtdata);

        }
    }

}


class Simulator {
public:
    explicit Simulator(double dt, double w=0.5, double r=10.) : _dt(dt), _w(w), _r(r) {
        _ba.setZero();
        _bg.setZero();

        // landmarks生成
        double deg2rad = EIGEN_PI / 180.;
        for (int i = 0; i < 360; ++i) {
            double angle = double(i % 360) * deg2rad;
            double cos_ang = cos(angle);
            double sin_ang = sin(angle);
            // 轴向
            for (int j = 0; j < 5; ++j) {
                double l = r + double(j);
                for (int k = 0; k < 5; ++k) {
                    /*
                     * 把 p = (0, l, k), 旋转R
                     * 其中,
                     * R = [cos(theta) -sin(theta) 0
                     *      sin(theta) cos(theta) 0
                     *      0 0 1]
                     * */
                    landmarks[i][j][k] = {-l * sin_ang, l * cos_ang, double(k) - 2.};
                }
            }
        }
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

    unsigned long get_landmark_id(unsigned int i, unsigned int j, unsigned int k) {
        return i + j * 1000 + k * 10000;
    }

    unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> get_landmarks_per_pose(double theta, const Vec3 &t_wi) {
        static double rad2deg = 180. / EIGEN_PI;
        Qd q_wi {cos(0.5 * theta), 0., 0., sin(0.5 * theta)};

        Vec3 p_i, p_c;
        Vec7 f;
        unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> landmarks_map;

        int ang = (int(theta * rad2deg) + 90 + 360) % 360;
        for (int i = -10; i <= 10; ++i) {
            int index = (ang + i + 360) % 360;
            for (int j = 0; j < 5; ++j) {
                for (int k = 0; k < 5; ++k) {
                    p_i = q_wi.inverse() * (landmarks[index][j][k] - t_wi);
                    p_c = q_ic.inverse() * (p_i - t_ic);
                    f << p_c.x() / p_c.z(), p_c.y() / p_c.z(), 1., 0., 0., 0., 0.;
                    landmarks_map[get_landmark_id(index, j, k)].emplace_back(0, f);
                }
            }
        }

        return landmarks_map;
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

public:
    Vec3 landmarks[360][5][5];

public:
    Qd q_ic {cos(-0.5 * EIGEN_PI * 0.5), 0., sin(-0.5 * EIGEN_PI * 0.5), 0.};
    Vec3 t_ic {0., 0., 0.};
};

int main() {
    Eigen::Matrix3d RIC;
    RIC<<0.0, 0.0, -1.0,
         -1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;
    Vec3 TIC;
    TIC<<0.05,0.04,0.03;

    vins::Estimator estimator;

    double dt = 0.005;
    // unsigned int num_data = 6000 + 1;
    // static Simulator simulator(dt);
    // simulator.generate_data(num_data);
    
    std::vector<MotionData1> cam_poses;
    std::vector<GtData> cam_gt;
    LoadPose("../simulator_data/cam_pose.txt",cam_poses,cam_gt);
    for(auto &gt : cam_gt){
        gt.rotation=gt.rotation*RIC.inverse();
        gt.position=gt.position-gt.rotation*TIC;
    }

    std::vector<MotionData1> imu_poses;
    std::vector<GtData> imu_gt;
    LoadPose("../simulator_data/imu_pose.txt",imu_poses,imu_gt);


    for (unsigned int n = 0; n < imu_poses.size(); ++n) {
        estimator.process_imu(dt, imu_poses[n].imu_acc, imu_poses[n].imu_gyro);
        if (n % 30 == 0) {
            std::cout << "p_gt: " << (cam_gt[n/30].position).transpose() << std::endl;
            std::stringstream feature_filename;
            feature_filename<<"../simulator_data/keyframe/all_points_"<<(n/30)<<".txt";
            auto &&f_per_imu = LoadFeature(feature_filename.str());
            estimator.process_image(f_per_imu, dt * double(n));
        }

        // 只测试初始化部分代码
        if (estimator.solver_flag == vins::Estimator::NON_LINEAR) {
            break;
        }
    }


    return 0;
}