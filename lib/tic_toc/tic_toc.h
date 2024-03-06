//
// Created by Cain on 2023/12/29.
//

#ifndef GRAPHOPTIMIZATION_TIC_TOC_H
#define GRAPHOPTIMIZATION_TIC_TOC_H

#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

namespace graph_optimization {
    class TicToc {
    public:
        TicToc() {
            tic();
        }

        void tic() {
            start = std::chrono::system_clock::now();
        }

        double toc() {
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            return elapsed_seconds.count() * 1000;
        }

    private:
        std::chrono::time_point<std::chrono::system_clock> start, end;
    };
}

#endif //GRAPHOPTIMIZATION_TIC_TOC_H
