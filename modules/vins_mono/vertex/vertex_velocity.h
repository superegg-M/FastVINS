//
// Created by Cain on 2024/4/24.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_VELOCITY_H
#define GRAPH_OPTIMIZATION_VERTEX_VELOCITY_H

#include "backend/vertex.h"

namespace graph_optimization {
    class VertexVelocity : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexVelocity() : Vertex(3) {}

        std::string type_info() const override { return "VertexVelocity"; }
    };
}


#endif //GRAPH_OPTIMIZATION_VERTEX_VELOCITY_H
