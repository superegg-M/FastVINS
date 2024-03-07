//
// Created by Cain on 2024/3/7.
//

#ifndef GRAPH_OPTIMIZATION_FDSI_VERTEX_H
#define GRAPH_OPTIMIZATION_FDSI_VERTEX_H

#include <lib/backend/problem.h>
#include <lib/backend/eigen_types.h>
#include <vector>

namespace system_identification {
    namespace frequency_domain {
        using namespace graph_optimization;

        template<unsigned NP,unsigned NZ,unsigned NI>
        class FDSIVertex: public Vertex {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            FDSIVertex(): Vertex(NP + NZ + 2) {}
            std::string type_info() const override { return "FDSI"; }
        };
    }
}

#endif //GRAPH_OPTIMIZATION_FDSI_VERTEX_H
