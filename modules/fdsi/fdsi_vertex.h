//
// Created by Cain on 2024/3/7.
//

#ifndef GRAPH_OPTIMIZATION_FDSI_VERTEX_H
#define GRAPH_OPTIMIZATION_FDSI_VERTEX_H

//#include <lib/backend/problem.h>
//#include <lib/backend/eigen_types.h>
#include <vector>
#include "backend/problem.h"
#include "backend/eigen_types.h"

namespace system_identification {
    namespace frequency_domain {
        using namespace graph_optimization;

        template<unsigned NP,unsigned NZ,unsigned NI>
        class FDSIVertex: public Vertex {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            FDSIVertex(): Vertex(NP + NZ + 2) {}
            std::string type_info() const override { return "FDSI"; }

//            void plus(const graph_optimization::VecX &delta) override {
//                _parameters.topRows<NP + NZ + 1>() += delta.topRows<NP + NZ + 1>();
//                _parameters[NP + NZ + 1] *= exp(delta[NP + NZ + 1]);
//            }
        };
    }
}

#endif //GRAPH_OPTIMIZATION_FDSI_VERTEX_H
