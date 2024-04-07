//
// Created by Cain on 2024/4/2.
//

#include "data_manager.h"

namespace vins {
    ImuNode::ImuNode(IMUIntegration *imu_integration_pt, unsigned int num_cameras)
    : imu_integration(imu_integration_pt), features(2 * num_cameras) {
    }

    ImuNode::~ImuNode() {
        delete imu_integration;
    }


    FrameNode::FrameNode(unsigned long id, ImuNode *imu) : _camera_id(id), _imu_pt(imu) {

    }


    FeatureNode::FeatureNode(unsigned long id) : imu_deque(WINDOW_SIZE), _feature_id(id) {

    }


    ImuWindows::ImuWindows(unsigned long n) : Deque<ImuNode *>(n) {

    }

    bool ImuWindows::is_feature_in_newest(unsigned long feature_id) const {
        if (empty()) {
            return false;
        }
//            if (_imu_deque.back()) {
//                _imu_deque.back().
//            }
    }
    bool ImuWindows::is_feature_suitable_to_reproject(unsigned long feature_id) const {
        if (size() > 2) {
            auto feature_it = operator[](size() - 3)->features.find(feature_id);
            if (feature_it == operator[](size() - 3)->features.end()) {
                return false;
            } else {
                return true;
            }
        }
        return false;
    }
}