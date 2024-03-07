//
// Created by Cain on 2024/3/7.
//

#ifndef GRAPH_OPTIMIZATION_PROBLEM_SLAM_H
#define GRAPH_OPTIMIZATION_PROBLEM_SLAM_H

#include <iostream>
#include <fstream>

//#include <glog/logging.h>
#include <Eigen/Dense>
#include <lib/tic_toc/tic_toc.h>
#include "problem.h"

namespace graph_optimization {
    namespace slam {
        class ProblemSLAM : public Problem {
        public:
            ProblemSLAM() = default;

        public:
            static bool is_pose_vertex(const std::shared_ptr<Vertex>& v);   ///< 判断一个顶点是否为Pose顶点
            static bool is_landmark_vertex(const std::shared_ptr<Vertex>& v);   ///< 判断一个顶点是否为landmark顶点

        };
    }
}

#endif //GRAPH_OPTIMIZATION_PROBLEM_SLAM_H
