//
// Created by Cain on 2024/6/12.
//

#ifndef GRAPH_OPTIMIZATION_FRAME_H
#define GRAPH_OPTIMIZATION_FRAME_H

#include <unordered_map>
#include <map>
#include <memory>
#include "backend/eigen_types.h"
#include "thirdparty/Sophus/sophus/so3.hpp"

namespace vins {
    struct MapPoint;
    struct Feature;

    /**
     * 帧
     * 每一帧分配独立id，关键帧分配关键帧ID
     */
    struct Frame {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr;

        unsigned long id = 0;           // id of this frame
        unsigned long keyframe_id = 0;  // id of key frame
        bool is_keyframe = false;       // 是否为关键帧
        double time_stamp;              // 时间戳，暂不使用
        SE3 pose;                       // Tcw 形式Pose
//        std::mutex pose_mutex_;          // Pose数据锁
//        cv::Mat left_img_, right_img_;   // stereo images

        // extracted features in left image
        std::vector<std::shared_ptr<Feature>> features_left;
        // corresponding features in right image, set to nullptr if no corresponding
        std::vector<std::shared_ptr<Feature>> features_right;

    public:  // data members
        Frame() {}

        Frame(unsigned long id, double time_stamp, const SE3 &pose, const Mat &left,
              const Mat &right);

        // set and get pose, thread safe
        SE3 Pose() {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            return pose_;
        }

        void SetPose(const SE3 &pose) {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            pose_ = pose;
        }

        /// 设置关键帧并分配并键帧id
        void SetKeyFrame();

        /// 工厂构建模式，分配id
        static std::shared_ptr<Frame> CreateFrame();
    };
}

#endif //GRAPH_OPTIMIZATION_FRAME_H
