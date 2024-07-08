//
// Created by Cain on 2024/5/22.
//

#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include "../modules/fdsi/p1id.h"
#include "../modules/fdsi/fdsi.h"

using namespace graph_optimization;
using namespace std;

/*
 * FDSI
 * */
int main() {
    vector<double> w_buff {
            6.28318530717959,	12.5663706143592,	18.8495559215388,	25.1327412287183,	31.4159265358979,
            37.6991118430775,	43.9822971502571,	50.2654824574367,	56.5486677646163,	62.8318530717959,
            69.1150383789754,	75.3982236861550,	81.6814089933346,	87.9645943005142,	94.2477796076938,
            100.530964914873,	106.814150222053,	113.097335529233,	119.380520836412,	125.663706143592
    };
    vector<double> re_buff {
            15.665461,	-6.879626,	-6.8117366,	-6.023029,	-5.014349,
            -3.2953122,	-3.451121,	-3.0044954,	-2.3530235,	-1.9938891,
            -1.6838235,	-1.5613447,	-1.3195838,	-1.1730264,	-1.0157101,
            -0.8913852,	-0.72248876,	-0.6360515,	-0.46929023,	-0.23781322
    };
    vector<double> im_buff {
            4.8768,	-20.388437,	-11.834236,	-6.9196825,	-4.2112164,
            -1.6616025,	-1.6983799,	-1.1227595,	-0.8433019,	-0.42252827,
            -0.27069244,	-0.16111729,	-0.032739427,	0.101014294,	0.11733885,
            0.14699747,	0.27201325,	0.2992155,	0.3367225,	0.4298026
    };

    // TODO: 使用RANSAC选择频段
    vector<unsigned > index_buff {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    };

    system_identification::frequency_domain::P1IDSolver p1id_solver;
    p1id_solver(re_buff, im_buff, w_buff, index_buff);

    vector<double> parameters {0.02, 200., 0.005};
    system_identification::frequency_domain::FDSISolver<1, 0, 1> fdsi_solver(parameters);
    fdsi_solver(re_buff, im_buff, w_buff, index_buff);

    vector<double> parameters2 {0.02, 0.0004, 200., 0.005};
    system_identification::frequency_domain::FDSISolver<2, 1, 1> fdsi_solver2(parameters2, false);
    fdsi_solver2(re_buff, im_buff, w_buff, index_buff);

    return 0;
}