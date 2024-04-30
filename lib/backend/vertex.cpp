//
// Created by Cain on 2023/2/20.
//

#include "vertex.h"

namespace graph_optimization {
    unsigned long Vertex::_global_vertex_id = 0;

    Vertex::Vertex(unsigned long num_dimension, unsigned long local_dimension) {
        _parameters.resize(num_dimension, 1);
        _parameters.setZero();
        _local_dimension = local_dimension ? local_dimension : num_dimension;
        _id = _global_vertex_id++;
    }

    void Vertex::save_parameters() {
        saved_parameters = true;
        _parameters_backup = _parameters;
    }

    bool Vertex::load_parameters() {
        if (saved_parameters) {
            saved_parameters = false;
            _parameters = _parameters_backup;
            return true;
        }
        return false;
    }

}

