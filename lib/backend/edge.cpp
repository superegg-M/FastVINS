//
// Created by Cain on 2023/2/20.
//

#include "edge.h"
#include "vertex.h"
#include <iostream>

namespace graph_optimization {
    unsigned long Edge::_global_edge_id = 0;

    Edge::Edge(unsigned long residual_dimension, unsigned long num_vertices, const std::vector<std::string> &vertices_types, unsigned loss_function_type) {
        _residual.resize(residual_dimension, 1);
        _verticies.reserve(num_vertices);
        if (!vertices_types.empty()) {
            _vertices_types = vertices_types;
        }
        _jacobians.resize(num_vertices);
        _id = _global_edge_id++;

        _information.resize(residual_dimension, residual_dimension);
        _information.setIdentity();

        _sqrt_information = _information;

        switch (loss_function_type) {
            case 1:
                _loss_function = new HuberLoss;
                break;
            case 2:
                _loss_function = new CauchyLoss;
                break;
            case 3:
                _loss_function = new TukeyLoss;
                break;
            default:
                _loss_function = new TrivialLoss;
                break;
        }
    }

    double Edge::chi2() const {
        // TODO::  we should not Multiply information here, because we have computed Jacobian = sqrt_info * Jacobian
        return _residual.transpose() * _information * _residual;
//        return _residual.squaredNorm();   // 当计算 residual 的时候已经乘以了 sqrt_info, 这里不要再乘
    }

    double Edge::robust_chi2() const {
        double e2 = chi2();
        auto &&rho = _loss_function->compute(e2);
        return rho[0];
    }

    void Edge::robust_information(double &drho, MatXX &info) const {
        double e2 = chi2();
        auto &&rho = _loss_function->compute(e2);
        VecX error = _sqrt_information * _residual;

        MatXX robust_info(_information.rows(), _information.cols());
        robust_info.setIdentity();
        robust_info *= rho[1];
        if(rho[1] + 2 * rho[2] * e2 > 0.) {
            robust_info += 2 * rho[2] * error * error.transpose();
        }

        info = robust_info * _information;
        drho = rho[1];
    }

    bool Edge::check_valid() {
        if (!_vertices_types.empty()) {
            // check type info
            for (size_t i = 0; i < _verticies.size(); ++i) {
                if (_vertices_types[i] != _verticies[i]->type_info()) {
                    std::cout << "Vertex type does not match, should be " << _vertices_types[i] <<
                              ", but set to " << _verticies[i]->type_info() << std::endl;
                    return false;
                }
            }
        }
        return true;
    }
}

