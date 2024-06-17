//
// Created by Cain on 2024/4/25.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_VEC1_H
#define GRAPH_OPTIMIZATION_VERTEX_VEC1_H

#include "backend/vertex.h"

namespace graph_optimization {
    class VertexVec1 : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexVec1() : Vertex(1) {}

        std::string type_info() const override { return "VertexVec1"; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_VEC1_H
