//
// Created by Cain on 2024/1/10.
//

#ifndef GRAPHO_PTIMIZATION_PARAMETERS_H
#define GRAPHO_PTIMIZATION_PARAMETERS_H

namespace vins {
    static constexpr unsigned int NUM_OF_CAM = 1;
    static constexpr unsigned NUM_OF_F = 1000;
    static constexpr unsigned int WINDOW_SIZE = 10;
    static constexpr double MIN_PARALLAX = 0.01;
    static constexpr double INIT_DEPTH = 0.1;
    static constexpr unsigned int SIZE_POSE = 7;
    static constexpr unsigned int SIZE_SPEEDBIAS = 9;
    static constexpr unsigned int SIZE_FEATURE = 1;
}

#endif //GRAPH_OPTIMIZATION_PARAMETERS_H
