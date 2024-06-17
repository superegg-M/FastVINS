//
// Created by Cain on 2024/4/11.
//

#include "vertex_scale.h"

graph_optimization::VertexScale::VertexScale() : Vertex(1, 1) {
    _parameters[0] = 1.;
}

void graph_optimization::VertexScale::plus(const graph_optimization::VecX &delta) {
    VecX &params = parameters();
    params[0] *= exp(delta[0]);
}