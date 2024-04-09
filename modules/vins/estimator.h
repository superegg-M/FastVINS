//
// Created by Cain on 2024/1/11.
//

#ifndef GRAPH_OPTIMIZATION_ESTIMATOR_H
#define GRAPH_OPTIMIZATION_ESTIMATOR_H

#include "parameters.h"
#include "feature_manager.h"
#include "data_manager.h"
//#include "utility/utility.h"
//#include "utility/tic_toc.h"
//#include "initial/solve_5pts.h"
//#include "initial/initial_sfm.h"
//#include "initial/initial_alignment.h"
//#include "initial/initial_ex_rotation.h"

#include "backend/eigen_types.h"
#include "backend/problem.h"
#include "backend/problem_slam.h"


#include <unordered_map>
#include <queue>
//#include <opencv2/core/eigen.hpp>

#include "imu_integration.h"
#include "edge_imu.h"
#include "edge_reprojection.h"
#include "edge_pnp.h"
#include "vertex_inverse_depth.h"
#include "vertex_pose.h"
#include "vertex_motion.h"

namespace vins {
    using namespace graph_optimization;

    class Estimator {
    public:
        Estimator();

        void set_parameter();

        // interface
        void process_imu(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);

        void process_image(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header);
        void set_relo_frame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

        // internal
        void clear_state();
        bool initial_structure();
        void global_triangulate_with(ImuNode *imu_i, ImuNode *imu_j, bool enforce=false);
        void local_triangulate_with(ImuNode *imu_i, ImuNode *imu_j, bool enforce=false);
        void global_triangulate_between(unsigned long i, unsigned long j, bool enforce=false);
        void local_triangulate_between(unsigned long i, unsigned long j, bool enforce=false);
        void global_triangulate_feature(FeatureNode* feature, bool enforce=false);
        void local_triangulate_feature(FeatureNode* feature, bool enforce=false);
        void pnp(ImuNode *imu_i, Qd *q_wi_init=nullptr, Vec3 *t_wi_init=nullptr);
        bool structure_from_motion();
        bool visual_initial_align();
        bool relative_pose(Matrix3d &r, Vector3d &t, unsigned long &imu_index);
        void slide_window();
        void solve_odometry();
        void slide_window_new();
        void slide_window_old();
        void optimization();
        void backend_optimization();

        void problem_solve();
        void marg_old_frame();
        void marg_new_frame();

        void vector2double();
        void double2vector();
        bool failure_detection();


        enum SolverFlag {
            INITIAL,
            NON_LINEAR
        };

        enum MarginalizationFlag {
            MARGIN_OLD = 0,
            MARGIN_SECOND_NEW = 1
        };
//////////////// OUR SOLVER ///////////////////
        MatXX Hprior_;
        VecX bprior_;
        VecX errprior_;
        MatXX Jprior_inv_;

        Eigen::Matrix2d project_sqrt_info_;
//////////////// OUR SOLVER //////////////////
        SolverFlag solver_flag;
        MarginalizationFlag  marginalization_flag;
        Vector3d g;
        MatrixXd Ap[2], backup_A;
        VectorXd bp[2], backup_b;

        Matrix3d ric[NUM_OF_CAM];
        Vector3d tic[NUM_OF_CAM];

        Vector3d Ps[(WINDOW_SIZE + 1)];
        Vector3d Vs[(WINDOW_SIZE + 1)];
        Matrix3d Rs[(WINDOW_SIZE + 1)];
        Vector3d Bas[(WINDOW_SIZE + 1)];
        Vector3d Bgs[(WINDOW_SIZE + 1)];
        double td;

        Matrix3d back_R0, last_R, last_R0;
        Vector3d back_P0, last_P, last_P0;
        double Headers[(WINDOW_SIZE + 1)];

        IMUIntegration *pre_integrations[(WINDOW_SIZE + 1)];
        Vector3d acc_0, gyr_0;

        vector<double> dt_buf[(WINDOW_SIZE + 1)];
        vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
        vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

        int frame_count;
        int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

        FeatureManager f_manager;
        MotionEstimator m_estimator;
        InitialEXRotation initial_ex_rotation;

        bool first_imu;
        bool is_valid, is_key;
        bool failure_occur;

        vector<Vector3d> point_cloud;
        vector<Vector3d> margin_cloud;
        vector<Vector3d> key_poses;
        double initial_timestamp;

        double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
        double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
        double para_Feature[NUM_OF_F][SIZE_FEATURE];
        double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
        double para_Retrive_Pose[SIZE_POSE];
        double para_Td[1][1];
        double para_Tr[1][1];

        int loop_window_index;

        // MarginalizationInfo *last_marginalization_info;
        vector<double *> last_marginalization_parameter_blocks;

        map<double, ImageFrame> all_image_frame;
        IMUIntegration *tmp_pre_integration;

        //relocalization variable
        bool relocalization_info;
        double relo_frame_stamp;
        double relo_frame_index;
        int relo_frame_local_index;
        vector<Vector3d> match_points;
        double relo_Pose[SIZE_POSE];
        Matrix3d drift_correct_r;
        Vector3d drift_correct_t;
        Vector3d prev_relo_t;
        Matrix3d prev_relo_r;
        Vector3d relo_relative_t;
        Quaterniond relo_relative_q;
        double relo_relative_yaw;

    private:
        State _state;
        graph_optimization::ProblemSLAM _problem;

        shared_ptr<VertexPose> _vertex_ext[NUM_OF_CAM]; // 相机的外参
        vector<Qd> _q_ic {NUM_OF_CAM};
        vector<Vec3> _t_ic{NUM_OF_CAM};

        IMUIntegration *_imu_integration {nullptr};
        Vec3 _acc_latest {};
        Vec3 _gyro_latest {};

        unordered_map<unsigned long, FeatureNode*> _feature_map;
        unordered_map<unsigned long, FrameNode*> _feature_based_frame;

        ImuNode *_imu_node;
        ImuWindows _windows;
    };
}

#endif //GRAPH_OPTIMIZATION_ESTIMATOR_H
