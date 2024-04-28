//
// Created by Cain on 2024/4/24.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_BIAS_H
#define GRAPH_OPTIMIZATION_VERTEX_BIAS_H

#include "backend/vertex.h"

namespace graph_optimization {
    class VertexBias : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexBias() : Vertex(3) {}

        std::string type_info() const override { return "VertexBias"; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_BIAS_H
