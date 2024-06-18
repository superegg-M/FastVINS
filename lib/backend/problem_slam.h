//
// Created by Cain on 2024/3/7.
//

#ifndef GRAPH_OPTIMIZATION_PROBLEM_SLAM_H
#define GRAPH_OPTIMIZATION_PROBLEM_SLAM_H

//#include <iostream>
//#include <fstream>
//
////#include <glog/logging.h>
//#include <Eigen/Dense>
//#include <lib/tic_toc/tic_toc.h>
#include "problem.h"

namespace graph_optimization {
    class ProblemSLAM : public Problem {
    public:
        ProblemSLAM() = default;

        bool add_vertex(const std::shared_ptr<Vertex>& vertex) override;
        bool remove_vertex(const std::shared_ptr<Vertex>& vertex) override;
        bool marginalize(const std::shared_ptr<Vertex>& vertex_pose, const std::shared_ptr<Vertex>& vertex_motion); ///< 边缘化pose和与之相关的edge

    public:
        static bool is_pose_vertex(const std::shared_ptr<Vertex>& v);   ///< 判断一个顶点是否为Pose顶点
        static bool is_landmark_vertex(const std::shared_ptr<Vertex>& v);   ///< 判断一个顶点是否为landmark顶点

    protected:
        void initialize_ordering() override;
        bool check_ordering() override;
        void add_prior_to_hessian() override;
        VecX multiply_hessian(const VecX &x) override;
        bool solve_linear_system(VecX &delta_x) override;
        void update_prior(const VecX &delta_x) override;
        void resize_pose_hessian_when_adding_pose(const std::shared_ptr<Vertex>& v);    ///< 在新增顶点后，需要调整几个hessian的大小

    protected:
        ulong _ordering_poses = 0;
        ulong _ordering_landmarks = 0;
//        std::unordered_map<unsigned long, std::shared_ptr<Vertex>> _idx_pose_vertices;        // 以ordering排序的pose顶点
//        std::unordered_map<unsigned long, std::shared_ptr<Vertex>> _idx_landmark_vertices;    // 以ordering排序的landmark顶点
        std::vector<std::pair<unsigned long, std::shared_ptr<Vertex>>> _idx_pose_vertices;        // 以ordering排序的pose顶点
        std::vector<std::pair<unsigned long, std::shared_ptr<Vertex>>> _idx_landmark_vertices;    // 以ordering排序的landmark顶点

        // 使用schur补求解线性方程组时的过程量
        MatXX _h_pp_schur;
        VecX _b_pp_schur;
//            MatXX _h_pp;
//            VecX _b_pp;
//            MatXX _h_ll;
//            VecX _b_ll;
    };
}

#endif //GRAPH_OPTIMIZATION_PROBLEM_SLAM_H
