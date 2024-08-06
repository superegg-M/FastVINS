//
// Created by Cain on 2024/4/28.
//

#include "../estimator.h"

#include "tic_toc/tic_toc.h"
#include "backend/eigen_types.h"

#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <ostream>
#include <fstream>

namespace vins {
    using namespace graph_optimization;
    using namespace std;

    void Estimator::process_image(const unordered_map<unsigned long, vector<pair<unsigned long, Vec7>>> &image, double header) {
        TicToc t_image;
        
        // 创建imu节点, 需要在marg后加入到windows中
        _imu_node = new ImuNode {_imu_integration};

        // 需要在process_imu重新new
        _imu_integration = nullptr;

        // 设置imu顶点的参数
        Vec7 pose;
        pose << _state.p(0), _state.p(1), _state.p(2),
                _state.q.x(), _state.q.y(), _state.q.z(), _state.q.w();
        Vec9 motion;
        motion << _state.v(0), _state.v(1), _state.v(2),
                _state.ba(0), _state.ba(1), _state.ba(2),
                _state.bg(0), _state.bg(1), _state.bg(2);
        _imu_node->vertex_pose->set_parameters(pose);
        _imu_node->vertex_motion->set_parameters(motion);
        
        // 创建图优化问题
        // _problem=nullptr;
        _problem = optimization(image);

        // 添加先验信息
        _problem.set_h_prior(Hprior_);
        _problem.set_b_prior(bprior_);

        if (solver_flag == INITIAL) {
            if (_windows.full()) {   // WINDOW已经装满了, 且还有camera frame
                bool is_initialized = initialize();
                if (is_initialized) {
                    cout << "Initialization finish!" << endl;

                    // 初始化后进行非线性优化
                    solver_flag = NON_LINEAR;
                    solve_odometry();

                    _state.p = _imu_node->get_p();
                    _state.q = _imu_node->get_q().normalized();
                    _state.v = _imu_node->get_v();
                    _state.ba = _imu_node->get_ba();
                    _state.bg = _imu_node->get_bg();

                    std::cout << "q_est: " << _state.q.w() << ", " << _state.q.x() << ", " << _state.q.y() << ", " << _state.q.z()<< std::endl;
                    std::cout << "p_est: " << _state.p.transpose() << std::endl;
                    std::cout << "v_est: " << _state.v.transpose() << std::endl;
                    std::cout << "ba_est: " << _state.ba.transpose() << std::endl;
                    std::cout << "bg_est: " << _state.bg.transpose() << std::endl;

//                    assert(-1 > 0);
                }
            }
        } else {
            TicToc t_solve;
            solve_odometry();

            TicToc t_margin;

            _state.p = _imu_node->get_p();
            _state.q = _imu_node->get_q().normalized();
            _state.v = _imu_node->get_v();
            _state.ba = _imu_node->get_ba();
            _state.bg = _imu_node->get_bg();

            std::cout << "q_est: " << _state.q.w() << ", " << _state.q.x() << ", " << _state.q.y() << ", " << _state.q.z()<< std::endl;
            std::cout << "p_est: " << _state.p.transpose() << std::endl;
            std::cout << "v_est: " << _state.v.transpose() << std::endl;
            std::cout << "ba_est: " << _state.ba.transpose() << std::endl;
            std::cout << "bg_est: " << _state.bg.transpose() << std::endl;
        }

        // 只有当windows满了才进行边缘化操作
        if (_windows.full()) {
            std::cout << "MARGINALIZATION" << std::endl;
            if(marginalization_flag == MARGIN_OLD){
                std::cout << "MARGIN_OLD" << std::endl;
                // 查找windows中最老的imu
                ImuNode *imu_oldest {nullptr};
                imu_oldest=_windows.oldest();
                // 边缘化掉oldest imu
                _problem.marginalize(imu_oldest->vertex_pose, imu_oldest->vertex_motion);
                // 保存先验信息
                Hprior_=_problem.get_h_prior();
                bprior_=_problem.get_b_prior();
                // 调整先验信息的维度
                unsigned size = Hprior_.rows() + 15;
                Hprior_.conservativeResize(size, size);
                bprior_.conservativeResize(size);

                bprior_.tail(15).setZero();
                Hprior_.rightCols(15).setZero();
                Hprior_.bottomRows(15).setZero();                          
            }else{
                std::cout << "MARGIN_NEW" << std::endl;
                // 查找windows中最新的imu
                ImuNode *imu_newest {nullptr};
                imu_newest=_windows.newest();
                // 边缘化掉旧的newest
                _problem.marginalize(imu_newest->vertex_pose, imu_newest->vertex_motion);
                // 保存先验信息
                Hprior_=_problem.get_h_prior();
                bprior_=_problem.get_b_prior();
                // 调整先验信息的维度
                unsigned size = Hprior_.rows() + 15;
                Hprior_.conservativeResize(size, size);
                bprior_.conservativeResize(size);

                bprior_.tail(15).setZero();
                Hprior_.rightCols(15).setZero();
                Hprior_.bottomRows(15).setZero();
            }
        }

        slide_window_only();
        
        auto image_cost=t_image.toc();
        cout<<"image_process_cost:"<<image_cost<< " ms"<<endl;
    }
}