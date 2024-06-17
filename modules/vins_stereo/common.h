//
// Created by Cain on 2024/5/23.
//

#ifndef GRAPH_OPTIMIZATION_COMMON_H
#define GRAPH_OPTIMIZATION_COMMON_H

namespace vins {
    union MarginFlags {
        struct {
            bool margin_old : 1;
            bool margin_new : 1;
        } flags;
        unsigned char value {0};
    };
}

#endif //GRAPH_OPTIMIZATION_COMMON_H
