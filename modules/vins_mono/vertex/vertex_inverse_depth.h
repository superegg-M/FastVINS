//
// Created by Cain on 2024/1/2.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_INVERSE_DEPTH_H
#define GRAPH_OPTIMIZATION_VERTEX_INVERSE_DEPTH_H

//#include <lib/backend/vertex.h>
#include "backend/vertex.h"

namespace graph_optimization {
    class VertexInverseDepth : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexInverseDepth() : Vertex(1) {
            _parameters[0] = 1.;
        }

        std::string type_info() const override { return "VertexInverseDepth"; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_INVERSE_DEPTH_H
