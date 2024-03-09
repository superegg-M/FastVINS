//
// Created by Cain on 2024/1/10.
//

#ifndef GRAPHO_PTIMIZATION_FEATURE_MANAGER_H
#define GRAPHO_PTIMIZATION_FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <map>
#include <unordered_map>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// #include <ros/console.h>
// #include <ros/assert.h>

#include "parameters.h"

namespace vins {
    /*！
     * 特征点在每个frame中的信息
     */
    class FeatureLocalInfo {
    public:
        struct State {
            Eigen::Matrix<double, 3, 1> point;
            Eigen::Matrix<double, 2, 1> pixel;
            Eigen::Matrix<double, 2, 1> pixel_rate;
        };

    public:
        explicit FeatureLocalInfo(const State &state, double td) {
            point = state.point;
            uv = state.pixel;
            velocity = state.pixel_rate;
            cur_td = td;
        }

    public:
        double cur_td;
        Vector3d point;
        Vector2d uv;
        Vector2d velocity;
        double z{};
        double dep_gradient{};
        double parallax{};
        MatrixXd A;
        VectorXd b;
        bool is_used{false};
    };

    /*!
     * 特征点的全局信息
     */
    class FeatureGlobalInfo {
    public:
        enum class Flag : unsigned char {
            unused = 0,
            success = 1,
            failure = 2
        };

    public:
        explicit FeatureGlobalInfo(unsigned long feature_index, unsigned long start_frame_index)
        : feature_id(feature_index), start_frame_id(start_frame_index) {
        }

        /*!
         * 判断该特征点是否能够用于计算重投影误差
         * 由于WINDOW的最后一帧(WINDOW_SIZE - 1)不一定是key_frame,
         * 所以最后一帧的key_frame为WINDOW中的倒数第二帧(WINDOW_SIZE - 2).
         * 而要用与计算重投影误差至少需要被2个key_frame观测到,
         * 所以start_frame < WINDOW_SIZE - 2
         * @return
         */
        bool is_suitable_to_reprojection() const { return get_used_num() >= 2 && start_frame_id + 2 < WINDOW_SIZE; }
        bool is_start_in_key_frame() const { return start_frame_id + 2 <= WINDOW_SIZE; }
        bool is_end_in_camera_frame() const { return get_end_frame_id() == WINDOW_SIZE; }
        bool is_end_in_newest_frame() const { return get_end_frame_id() == WINDOW_SIZE - 1; }
        unsigned long get_used_num() const { return feature_local_infos.size(); }
        unsigned long get_end_frame_id() const { return start_frame_id + feature_local_infos.size() - 1; }

    public:
        const unsigned long feature_id; // 每个feature对应一个独一无二的id
        unsigned long start_frame_id;   // start_frame_id ∈ [0, WINDOW_SIZE)
//        unsigned long used_num {0};   // 通过feature_local_infos.size()获取

        double estimated_depth {-1.};
        Vector3d gt_p;

        vector<FeatureLocalInfo> feature_local_infos;   // TODO: 换成map

        bool is_outlier{false};
        bool is_margin{false};
        Flag solve_flag{Flag::unused};
    };

    class FeatureManager {
    public:
        explicit FeatureManager(Matrix3d *r_wi);

        void set_ric(Matrix3d *r_ic);

        void clear_feature();

        unsigned int get_feature_count();

        /*!
         *
         * @param frame_count
         * @param image {feature_id, {{camera_id, local_feature_state}, {camera_id, local_feature_state}, ...}}
         * @param td
         * @return
         */
        bool add_feature_and_check_latest_frame_parallax(unsigned long frame_count, const map<unsigned long, vector<pair<unsigned long, FeatureLocalInfo::State>>> &image, double td);
        void debug_show();
        vector<pair<Vector3d, Vector3d>> get_corresponding(unsigned long frame_count_l, unsigned long frame_count_r);

        //void update_depth(const VectorXd &x);
        void set_depth(const VectorXd &x);
        void remove_outlier();
        void clear_depth(const VectorXd &x);
        VectorXd get_depth_vector();
        void triangulate(Vector3d p_imu[], Vector3d t_ic[], Matrix3d r_ic[]);

        /// @brief 滑窗中移除了最老帧，更新frame index，并将持有特征的深度信息转移给次老帧
        void remove_back_shift_depth(const Eigen::Matrix3d& marg_R, const Eigen::Vector3d& marg_P, Eigen::Matrix3d new_R, const Eigen::Vector3d& new_P);

        /// @brief 在滑窗中移除最旧的一帧
        void remove_back();

        /// @brief 在滑窗中移除倒数第二新的帧
        void remove_front(unsigned long frame_count);
        void remove_failures();
//        list<FeatureGlobalInfo> features;
        unordered_map<unsigned long, FeatureGlobalInfo> features_map;
        vector<unsigned long> feature_id_erase;
        unsigned long last_track_num;

    private:
        static double compensated_parallax2(const FeatureGlobalInfo &feature, unsigned long frame_count);
        const Matrix3d *_r_wi;          ///< imu到world的旋转
        Matrix3d _r_ic[NUM_OF_CAM];     ///< 相机到imu的旋转
    };
}



#endif //GRAPH_OPTIMIZATION_FEATURE_MANAGER_H
