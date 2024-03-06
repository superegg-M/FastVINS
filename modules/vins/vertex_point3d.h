//
// Created by Cain on 2024/1/2.
//

#ifndef GRAPH_OPTIMIZATION_VERTEX_POINT_3D_H
#define GRAPH_OPTIMIZATION_VERTEX_POINT_3D_H

#include <lib/backend/vertex.h>

namespace graph_optimization {
    class VertexPoint3d : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexPoint3d() : Vertex(3) {}

        std::string type_info() const override { return "VertexPoint3d"; }
    };
}

#endif //GRAPH_OPTIMIZATION_VERTEX_POINT_3D_H
