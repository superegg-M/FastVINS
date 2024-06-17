//
// Created by Cain on 2024/4/24.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_ACC_BIAS_H
#define GRAPH_OPTIMIZATION_VERTEX_ACC_BIAS_H

#include "backend/vertex.h"

namespace graph_optimization {
    class VertexAccBias : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexAccBias() : Vertex(3) {}

        std::string type_info() const override { return "VertexAccBias"; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_ACC_BIAS_H
