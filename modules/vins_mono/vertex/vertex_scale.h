//
// Created by Cain on 2024/4/11.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_SCALE_H
#define GRAPH_OPTIMIZATION_VERTEX_SCALE_H

#include "thirdparty/Sophus/sophus/so3.hpp"
#include "backend/vertex.h"

namespace graph_optimization {
    class VertexScale : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexScale();

        void plus(const VecX &delta) override;

        std::string type_info() const override { return "VertexScale"; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_SCALE_H
