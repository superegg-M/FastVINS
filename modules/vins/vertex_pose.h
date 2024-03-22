//
// Created by Cain on 2024/1/2.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_POSE_H
#define GRAPH_OPTIMIZATION_VERTEX_POSE_H

//#include <lib/backend/vertex.h>
#include <lib/thirdparty/Sophus/sophus/so3.hpp>
#include "backend/vertex.h"

namespace graph_optimization {
    class VertexPose : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexPose() : Vertex(7, 6) {}

        void plus(const VecX &delta) override;

        std::string type_info() const override { return "VertexPose"; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_POSE_H
