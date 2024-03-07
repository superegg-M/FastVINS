//
// Created by Cain on 2024/3/7.
//

#ifndef GRAPH_OPTIMIZATION_FDSI_EDGE_H
#define GRAPH_OPTIMIZATION_FDSI_EDGE_H

#include <lib/backend/problem.h>
#include <lib/backend/eigen_types.h>
#include <vector>

namespace system_identification {
    namespace frequency_domain {
        using namespace graph_optimization;


        template<unsigned NP,unsigned NZ,unsigned NI>
        class FDSIEdge: public Edge {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            FDSIEdge(double w, double re, double im)
                    : Edge(2, 1, std::vector<std::string>{"FDSI"}), _w(w), _re(re), _im(im) {}

            // 计算曲线模型误差
            void compute_residual() override;

            // 计算残差对变量的雅克比
            void compute_jacobians() override;

            // 返回边的类型信息
            std::string type_info() const override { return "FDSIEdge"; }

            std::vector<double> get_den() const;
            std::vector<double> get_num() const;
            double get_Td() const;
            void calculate_wr_wi(std::vector<double> &wr, std::vector<double> &wi) const;

        private:
            double _w, _re, _im;
        };


        template<unsigned NP,unsigned NZ,unsigned NI>
        void FDSIEdge<NP, NZ, NI>::compute_residual() {
            float wi = pow(_w, NI);

            auto &&Td = get_Td();
            auto &&den = get_den();
            auto &&num = get_num();
            std::vector<double> den_wr(den.size(), 0.);
            std::vector<double> den_wi(den.size(), 0.);
            std::vector<double> num_wr(num.size(), 0.);
            std::vector<double> num_wi(num.size(), 0.);
            calculate_wr_wi(den_wr, den_wi);
            calculate_wr_wi(num_wr, num_wi);

            double AR = 0., AI = 0., BR = 0., BI = 0.;
            for (unsigned i = 0; i < den.size(); ++i) {
                AR += den[i] * den_wr[i];
                AI += den[i] * den_wi[i];
            }
            for (unsigned i = 0; i < num.size(); ++i) {
                BR += num[i] * num_wr[i];
                BI += num[i] * num_wi[i];
            }

            double A_norm2 = AI * AI + AR * AR;
            double cos_Tw = cos(Td * _w);
            double sin_Tw = sin(Td * _w);

            double X = ((AI*BI + AR*BR)*cos_Tw + (AR*BI - AI*BR)*sin_Tw)/ A_norm2;
            double Y = ((AR*BI - AI*BR)*cos_Tw - (AI*BI + AR*BR)*sin_Tw)/ A_norm2;

            unsigned task = NI % 4;
            switch (task) {
                case 0:
                    _residual(0) = X/wi - _re;
                    _residual(1) = Y/wi - _im;
                    break;
                case 1:
                    _residual(0) = Y/wi - _re;
                    _residual(1) = -X/wi - _im;
                    break;
                case 2:
                    _residual(0) = -X/wi - _re;
                    _residual(1) = -Y/wi - _im;
                    break;
                case 3:
                    _residual(0) = -Y/wi - _re;
                    _residual(1) = X/wi - _im;
                    break;
                default:
                    break;
            }
        }

        template<unsigned NP,unsigned NZ,unsigned NI>
        void FDSIEdge<NP, NZ, NI>::compute_jacobians() {
            float wi = pow(_w, NI);

            auto &&Td = get_Td();
            auto &&den = get_den();
            auto &&num = get_num();
            std::vector<double> den_wr(den.size(), 0.);
            std::vector<double> den_wi(den.size(), 0.);
            std::vector<double> num_wr(num.size(), 0.);
            std::vector<double> num_wi(num.size(), 0.);
            calculate_wr_wi(den_wr, den_wi);
            calculate_wr_wi(num_wr, num_wi);

            double AR = 0., AI = 0., BR = 0., BI = 0.;
            for (unsigned i = 0; i < den.size(); ++i) {
                AR += den[i] * den_wr[i];
                AI += den[i] * den_wi[i];
            }
            for (unsigned i = 0; i < num.size(); ++i) {
                BR += num[i] * num_wr[i];
                BI += num[i] * num_wi[i];
            }

            double AI2 = AI * AI;
            double AR2 = AR * AR;
            double A_norm2 = AI * AI + AR * AR;
            double A_norm4 = A_norm2 * A_norm2;
            double cos_Tw = cos(Td * _w);
            double sin_Tw = sin(Td * _w);

            double dX_dAR = ((-2.*AI*AR*BI + AI2*BR - AR2*BR)*cos_Tw + (AI2*BI - AR2*BI + 2*AI*AR*BR)*sin_Tw)/A_norm4;
            double dX_dAI = ((-(AI2*BI) + AR2*BI - 2.*AI*AR*BR)*cos_Tw + (-2.*AI*AR*BI + AI2*BR - AR2*BR)*sin_Tw)/A_norm4;
            double dX_dBR = (AR*cos_Tw - AI*sin_Tw)/(A_norm2);
            double dX_dBI = (AI*cos_Tw + AR*sin_Tw)/(A_norm2);
            double dX_dTd = (_w*((AR*BI - AI*BR)*cos_Tw - (AI*BI + AR*BR)*sin_Tw))/(A_norm2);

            double dY_dAR = ((AI2*BI - AR2*BI + 2.*AI*AR*BR)*cos_Tw + (2.*AI*AR*BI - AI2*BR + AR2*BR)*sin_Tw)/A_norm4;
            double dY_dAI = ((-2.*AI*AR*BI + AI2*BR - AR2*BR)*cos_Tw + (AI2*BI - AR2*BI + 2.*AI*AR*BR)*sin_Tw)/A_norm4;
            double dY_dBR = -((AI*cos_Tw + AR*sin_Tw)/(A_norm2));
            double dY_dBI = (AR*cos_Tw - AI*sin_Tw)/(A_norm2);
            double dY_dTd = -((_w*((AI*BI + AR*BR)*cos_Tw + (AR*BI - AI*BR)*sin_Tw))/(A_norm2));

            Eigen::Matrix<double, 2, 5> part_j;
            unsigned task = NI % 4;
            switch (task) {
                case 0:
                    part_j(0, 0) = dX_dAR/wi;
                    part_j(0, 1) = dX_dAI/wi;
                    part_j(0, 2) = dX_dBR/wi;
                    part_j(0, 3) = dX_dBI/wi;
                    part_j(0, 4) = dX_dTd/wi;
                    part_j(1, 0) = dY_dAR/wi;
                    part_j(1, 1) = dY_dAI/wi;
                    part_j(1, 2) = dY_dBR/wi;
                    part_j(1, 3) = dY_dBI/wi;
                    part_j(1, 4) = dY_dTd/wi;
                    break;
                case 1:
                    part_j(0, 0) = dY_dAR/wi;
                    part_j(0, 1) = dY_dAI/wi;
                    part_j(0, 2) = dY_dBR/wi;
                    part_j(0, 3) = dY_dBI/wi;
                    part_j(0, 4) = dY_dTd/wi;
                    part_j(1, 0) = -dX_dAR/wi;
                    part_j(1, 1) = -dX_dAI/wi;
                    part_j(1, 2) = -dX_dBR/wi;
                    part_j(1, 3) = -dX_dBI/wi;
                    part_j(1, 4) = -dX_dTd/wi;
                    break;
                case 2:
                    part_j(0, 0) = -dX_dAR/wi;
                    part_j(0, 1) = -dX_dAI/wi;
                    part_j(0, 2) = -dX_dBR/wi;
                    part_j(0, 3) = -dX_dBI/wi;
                    part_j(0, 4) = -dX_dTd/wi;
                    part_j(1, 0) = -dY_dAR/wi;
                    part_j(1, 1) = -dY_dAI/wi;
                    part_j(1, 2) = -dY_dBR/wi;
                    part_j(1, 3) = -dY_dBI/wi;
                    part_j(1, 4) = -dY_dTd/wi;
                    break;
                case 3:
                    part_j(0, 0) = -dY_dAR/wi;
                    part_j(0, 1) = -dY_dAI/wi;
                    part_j(0, 2) = -dY_dBR/wi;
                    part_j(0, 3) = -dY_dBI/wi;
                    part_j(0, 4) = -dY_dTd/wi;
                    part_j(1, 0) = dX_dAR/wi;
                    part_j(1, 1) = dX_dAI/wi;
                    part_j(1, 2) = dX_dBR/wi;
                    part_j(1, 3) = dX_dBI/wi;
                    part_j(1, 4) = dX_dTd/wi;
                    break;
                default:
                    break;
            }

            Eigen::Matrix<double, 2, NP + NZ + 2> jacobians;
            for (unsigned i = 0; i < NP; ++i) {
                jacobians(0, i) = part_j(0, 0) * den_wr[i+1] + part_j(0, 1) * den_wi[i+1];
                jacobians(1, i) = part_j(1, 0) * den_wr[i+1] + part_j(1, 1) * den_wi[i+1];
            }
            for (unsigned i = 0; i < NZ + 1; ++i) {
                jacobians(0, NP + i) = part_j(0, 2) * num_wr[i] + part_j(0, 3) * num_wi[i];
                jacobians(1, NP + i) = part_j(1, 2) * num_wr[i] + part_j(1, 3) * num_wi[i];
            }
            jacobians(0, NP + NZ + 1) = part_j(0, 4);
            jacobians(1, NP + NZ + 1) = part_j(1, 4);

            _jacobians[0] = jacobians;
        }

        template<unsigned NP,unsigned NZ,unsigned NI>
        std::vector<double> FDSIEdge<NP, NZ, NI>::get_den() const {
            std::vector<double> den(NP + 1, 1.);
            for (unsigned i = 0; i < NP; ++i) {
                den[i + 1] = _vertices[0]->parameters()(i);
            }
            return den;
        }

        template<unsigned NP,unsigned NZ,unsigned NI>
        std::vector<double> FDSIEdge<NP, NZ, NI>::get_num() const {
            std::vector<double> num(NZ + 1, 0.);
            for (unsigned i = 0; i < NZ + 1; ++i) {
                num[i] = _vertices[0]->parameters()(NP + i);
            }
            return num;
        }

        template<unsigned NP,unsigned NZ,unsigned NI>
        double FDSIEdge<NP, NZ, NI>::get_Td() const {
            return _vertices[0]->parameters()(NP + NZ + 1);
        }

        template<unsigned NP,unsigned NZ,unsigned NI>
        void FDSIEdge<NP, NZ, NI>::calculate_wr_wi(std::vector<double> &wr, std::vector<double> &wi) const {
            double w_freq = 1;
            for (unsigned i = 0; i < wr.size(); ++i) {
                unsigned task = i % 4;
                switch (task) {
                    case 0:
                        wr[i] = w_freq;
                        wi[i] = 0.;
                        break;
                    case 1:
                        wr[i] = 0.;
                        wi[i] = w_freq;
                        break;
                    case 2:
                        wr[i] = -w_freq;
                        wi[i] = 0.;
                        break;
                    case 3:
                        wr[i] = 0.;
                        wi[i] = -w_freq;
                        break;
                    default:
                        break;
                }
                w_freq = w_freq * _w;
            }
        }
    }
}

#endif //GRAPH_OPTIMIZATION_FDSI_EDGE_H
