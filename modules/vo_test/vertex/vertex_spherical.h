//
// Created by Cain on 2024/4/12.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_SPHERICAL_H
#define GRAPH_OPTIMIZATION_VERTEX_SPHERICAL_H

#include "thirdparty/Sophus/sophus/so3.hpp"
#include "backend/vertex.h"

namespace graph_optimization {
    class VertexSpherical : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexSpherical();

        void plus(const VecX &delta) override;

        std::string type_info() const override { return "VertexSpherical"; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_SPHERICAL_H
