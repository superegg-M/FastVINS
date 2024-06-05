//
// Created by Cain on 2024/1/11.
//

#ifndef GRAPH_OPTIMIZATION_ESTIMATOR_H
#define GRAPH_OPTIMIZATION_ESTIMATOR_H

#include "parameters.h"
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
#include "edge/edge_imu.h"
#include "edge/edge_reprojection.h"
#include "edge/edge_pnp.h"
#include "edge/edge_pnp_sim3.h"
#include "edge/edge_epipolar.h"
#include "vertex/vertex_inverse_depth.h"
#include "vertex/vertex_pose.h"
#include "vertex/vertex_motion.h"
#include "vertex/vertex_scale.h"
#include "vertex/vertex_quaternion.h"
#include "vertex/vertex_spherical.h"

namespace vins {
    using namespace graph_optimization;

    class Estimator {
    public:
        Estimator();

        // interface
        void process_imu(double t, const Vec3 &linear_acceleration, const Vec3 &angular_velocity);

        void process_image(const unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> &image, double header);

        // initialize
        bool initialize();

        // visual initialize
        bool search_relative_pose(Mat33 &r, Vec3 &t, unsigned long &imu_index);
        bool structure_from_motion();

        // inertial initialize
        bool align_visual_to_imu();

        // 2D-2D
        bool compute_essential_matrix(Mat33 &R, Vec3 &t, ImuNode *imu_i, ImuNode *imu_j, bool is_init_landmark=true, unsigned int max_iters=30);
        bool compute_homography_matrix(Mat33 &R, Vec3 &t, ImuNode *imu_i, ImuNode *imu_j, bool is_init_landmark=true, unsigned int max_iters=30);

        // 2D-3D
        void global_triangulate_with(ImuNode *imu_i, ImuNode *imu_j, bool enforce=false);
        void local_triangulate_with(ImuNode *imu_i, ImuNode *imu_j, bool enforce=false);
        void global_triangulate_between(unsigned long i, unsigned long j, bool enforce=false);
        void local_triangulate_between(unsigned long i, unsigned long j, bool enforce=false);
        void global_triangulate_feature(FeatureNode* feature, bool enforce=false);
        void local_triangulate_feature(FeatureNode* feature, bool enforce=false);

        // 3D-2D
        void pnp(ImuNode *imu_i, Qd *q_wi_init=nullptr, Vec3 *t_wi_init=nullptr);
        void epnp(ImuNode *imu_i);
        void mlpnp(ImuNode *imu_i, unsigned int batch_size=36, unsigned int num_batches=30);
        void dltpnp(ImuNode *imu_i, unsigned int batch_size=36, unsigned int num_batches=30);
        bool iter_pnp(ImuNode *imu_i, Qd *q_wi_init=nullptr, Vec3 *t_wi_init=nullptr);

        // bundle adjustment
        void global_bundle_adjustment(vector<shared_ptr<VertexPose>> *fixed_poses=nullptr);
        void local_bundle_adjustment(vector<shared_ptr<VertexPose>> *fixed_poses=nullptr);

        void slide_window();
        void solve_odometry();

        void optimization();

        void problem_solve();

        bool failure_detection(unsigned int iteration=5);

        enum SolverFlag {
            INITIAL,
            NON_LINEAR
        };

        enum MarginalizationFlag {
            MARGIN_OLD = 0,
            MARGIN_SECOND_NEW = 1
        };

        bool _is_visual_initialized {false};
        bool _is_visual_aligned_to_imu {false};

//////////////// OUR SOLVER ///////////////////
        MatXX Hprior_;
        VecX bprior_;
        VecX errprior_;
        MatXX Jprior_inv_;

        Eigen::Matrix2d project_sqrt_info_;
//////////////// OUR SOLVER //////////////////
        SolverFlag solver_flag {INITIAL};
        MarginalizationFlag  marginalization_flag {MARGIN_OLD};

    public:
        Vec3 _g {0., 0., -9.8};
        State _state;
        graph_optimization::ProblemSLAM _problem;

        vector<shared_ptr<VertexPose>> _vertex_ext; // 相机的外参
        vector<Qd> _q_ic {NUM_OF_CAM};
        vector<Vec3> _t_ic{NUM_OF_CAM};

        IMUIntegration *_imu_integration {nullptr};
        Vec3 _acc_latest {};
        Vec3 _gyro_latest {};

        unordered_map<unsigned long, FeatureNode*> _feature_map;
        unordered_map<unsigned long, FrameNode*> _feature_based_frame;

        ImuNode *_imu_node {nullptr};
        ImuWindows _windows;
    };
}

#endif //GRAPH_OPTIMIZATION_ESTIMATOR_H
