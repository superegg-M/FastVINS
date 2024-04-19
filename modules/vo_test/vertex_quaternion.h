//
// Created by Cain on 2024/4/12.
//

#ifndef GRAPHO_PTIMIZATION_VERTEX_QUATERNION_H
#define GRAPHO_PTIMIZATION_VERTEX_QUATERNION_H

#include "thirdparty/Sophus/sophus/so3.hpp"
#include "backend/vertex.h"

namespace graph_optimization {
    class VertexQuaternion : public Vertex {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexQuaternion();

        void plus(const VecX &delta) override;

        std::string type_info() const override { return "VertexQuaternion"; }
    };
}

#endif //GRAPHO_PTIMIZATION_VERTEX_QUATERNION_H
