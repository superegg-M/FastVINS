//
// Created by Cain on 2024/3/7.
//

#include "problem_slam.h"

namespace graph_optimization {
    namespace slam {
        bool ProblemSLAM::is_pose_vertex(const std::shared_ptr<Vertex>& v) {
            string type = v->type_info();
            return type == string("VertexPose") || type == string("VertexMotion");
        }

        bool ProblemSLAM::is_landmark_vertex(const std::shared_ptr<Vertex>& v) {
            string type = v->type_info();
            return type == string("VertexPointXYZ") || type == string("VertexInverseDepth");
        }
    }
}