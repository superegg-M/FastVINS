//
// Created by Cain on 2024/1/2.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_MOTION_H
#define GRAPH_OPTIMIZATION_VERTEX_MOTION_H

//#include <lib/backend/vertex.h>
#include "backend/vertex.h"

namespace graph_optimization {
    class VertexMotion : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexMotion() : Vertex(9) {} // v, ba, bg

        std::string type_info() const override { return "VertexMotion"; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_MOTION_H
