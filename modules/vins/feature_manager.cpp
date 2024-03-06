//
// Created by Cain on 2024/1/10.
//

#include "feature_manager.h"

namespace vins {
    FeatureManager::FeatureManager(Matrix3d *r_wi) : _r_wi(r_wi) {
        for (auto & r : _r_ic) {
            r.setIdentity();
        }
    }

    void FeatureManager::set_ric(Matrix3d *r_ic) {
        for (int i = 0; i < NUM_OF_CAM; i++) {
            _r_ic[i] = r_ic[i];
        }
    }

    void FeatureManager::clear_feature() {
        features_map.clear();
    }

    unsigned int FeatureManager::get_feature_count() {
        unsigned int cnt = 0;
        for (auto &feature : features_map) {
            feature.second.used_num = feature.second.feature_local_infos.size();
            if (feature.second.used_num >= 2 && feature.second.start_frame_id + 2 < WINDOW_SIZE) {
                ++cnt;
            }
        }
        return cnt;
    }

    bool FeatureManager::add_feature_check_parallax(unsigned long frame_id,
                                                    const map<unsigned long, vector<pair<unsigned long, FeatureLocalInfo::State>>> &image,
                                                    double td) {
        double parallax_sum = 0;
        int parallax_num = 0;
        last_track_num = 0;
        for (auto &feature_global_info : image) {   // pair<unsigned long, vector<pair<unsigned long, FeatureLocalInfo::State>>>
            FeatureLocalInfo feature_local_info_start_frame(feature_global_info.second[0].second, td);  // 左目

            unsigned long feature_id = feature_global_info.first;
            if (features_map.find(feature_id) == features_map.end()) {  // 新的feature
                features_map.emplace(pair<unsigned long, FeatureGlobalInfo>(feature_id, FeatureGlobalInfo(feature_id, frame_id)));
                features_map[feature_id].feature_local_infos.push_back(feature_local_info_start_frame);
            } else {    // 已存在的feature
                features_map[feature_id].feature_local_infos.push_back(feature_local_info_start_frame);
                ++last_track_num;
            }
        }

        if (frame_id < 2 || last_track_num < 20) {  // 头两帧 or 该帧的所有特征点被追踪的次数小于20次
            return true;
        }

        for (auto &features : features_map) {
            if (features.second.start_frame_id + 2 <= frame_id && features.second.get_end_frame_id() + 1 >= frame_id) {
                parallax_sum += compensated_parallax2(features.second, frame_id);
                ++parallax_num;
            }
        }

        if (parallax_num == 0) {
            return true;
        }
        else {
            return parallax_sum / parallax_num >= MIN_PARALLAX;
        }
    }

    vector<pair<Vector3d, Vector3d>> FeatureManager::get_corresponding(unsigned long frame_id_l, unsigned long frame_id_r) {
        vector<pair<Vector3d, Vector3d>> corres;
        for (auto &feature : features_map) {
            if (feature.second.start_frame_id <= frame_id_l && feature.second.get_end_frame_id() >= frame_id_r) {
                Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
                unsigned long local_id_l = frame_id_l - feature.second.start_frame_id;
                unsigned long local_id_r = frame_id_r - feature.second.start_frame_id;

                a = feature.second.feature_local_infos[local_id_l].point;
                b = feature.second.feature_local_infos[local_id_r].point;

                corres.emplace_back(a, b);
            }
        }
        return corres;
    }

    void FeatureManager::set_depth(const VectorXd &x) {
        unsigned long feature_index = 0;
        for (auto &feature : features_map) {
            feature.second.used_num = feature.second.feature_local_infos.size();
            if (!(feature.second.used_num >= 2 && feature.second.start_frame_id + 2 < WINDOW_SIZE)) {
                continue;
            }

            feature.second.estimated_depth = 1. / x[feature_index++];
            if (feature.second.estimated_depth < 0) {
                feature.second.solve_flag = FeatureGlobalInfo::Flag::failure;
            } else {
                feature.second.solve_flag = FeatureGlobalInfo::Flag::success;
            }
        }
    }

    void FeatureManager::remove_failures() {
        for (auto it = features_map.begin(); it != features_map.end(); ++it) {
            if (it->second.solve_flag == FeatureGlobalInfo::Flag::failure) {
                features_map.erase(it);
            }
        }
    }

    void FeatureManager::clear_depth(const VectorXd &x) {
        unsigned long feature_index = 0;
        for (auto &feature : features_map) {
            feature.second.used_num = feature.second.feature_local_infos.size();
            if (!(feature.second.used_num >= 2 && feature.second.start_frame_id + 2 < WINDOW_SIZE)) {
                continue;
            }
            feature.second.estimated_depth = 1. / x[feature_index++];
        }
    }

    VectorXd FeatureManager::get_depth_vector() {
        VectorXd dep_vec(get_feature_count());
        unsigned long feature_index = 0;
        for (auto &feature : features_map) {
            feature.second.used_num = feature.second.feature_local_infos.size();
            if (!(feature.second.used_num >= 2 && feature.second.start_frame_id + 2 < WINDOW_SIZE)) {
                continue;
            }
            dep_vec[feature_index++] = 1. / feature.second.estimated_depth;
        }
        return dep_vec;
    }

    void FeatureManager::triangulate(Vector3d p_imu[], Vector3d t_ic[], Matrix3d r_ic[]) {
        for (auto &feature : features_map) {
            feature.second.used_num = feature.second.feature_local_infos.size();
            if (!(feature.second.used_num >= 2 && feature.second.start_frame_id + 2 < WINDOW_SIZE)) {
                continue;
            }

            // 已经进行过三角化
            if (feature.second.estimated_depth > 0) {
                continue;
            }
            unsigned long frame_i = feature.second.start_frame_id;
            unsigned long frame_j;

            assert(NUM_OF_CAM == 1);

            Eigen::MatrixXd svd_A(2 * feature.second.feature_local_infos.size(), 4);
            Eigen::Matrix<double, 3, 4> P;
            Eigen::Vector3d f;

            Eigen::Vector3d t_wci_w = p_imu[frame_i] + _r_wi[frame_i] * t_ic[0];
            Eigen::Matrix3d r_wci = _r_wi[frame_i] * r_ic[0];

            P.leftCols<3>() = Eigen::Matrix3d::Identity();
            P.rightCols<1>() = Eigen::Vector3d::Zero();

            f = feature.second.feature_local_infos[0].point.normalized();
            svd_A.row(0) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(1) = f[1] * P.row(2) - f[2] * P.row(1);

            for (unsigned long j = 1; j < feature.second.feature_local_infos.size(); ++j) {
                frame_j = frame_i + j;

                Eigen::Vector3d t_wcj_w = p_imu[frame_j] + _r_wi[frame_j] * t_ic[0];
                Eigen::Matrix3d r_wcj = _r_wi[frame_j] * r_ic[0];
                Eigen::Vector3d t_cicj_ci = r_wci.transpose() * (t_wcj_w - t_wci_w);
                Eigen::Matrix3d r_cicj = r_wci.transpose() * r_wcj;

                P.leftCols<3>() = r_cicj.transpose();
                P.rightCols<1>() = -r_cicj.transpose() * t_cicj_ci;

                f = feature.second.feature_local_infos[j].point.normalized();
                svd_A.row(2 * j) = f[0] * P.row(2) - f[2] * P.row(0);
                svd_A.row(2 * j + 1) = f[1] * P.row(2) - f[2] * P.row(1);
            }

            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double depth = svd_V[2] / svd_V[3];
            feature.second.estimated_depth = depth;

            if (feature.second.estimated_depth < 0.1) {
                feature.second.estimated_depth = INIT_DEPTH;
            }
        }
    }

    void FeatureManager::remove_outlier() {
//        int i = -1;
//        for (auto it = feature.begin(), it_next = feature.begin();
//             it != feature.end(); it = it_next)
//        {
//            it_next++;
//            i += it->used_num != 0;
//            if (it->used_num != 0 && it->is_outlier == true)
//            {
//                feature.erase(it);
//            }
//        }
    }

    void FeatureManager::remove_back_shift_depth(const Eigen::Matrix3d& marg_R, const Eigen::Vector3d& marg_P,
                                                 Eigen::Matrix3d new_R, const Eigen::Vector3d& new_P) {
        for (auto &feature : features_map) {
            if (feature.second.start_frame_id != 0) {
                --feature.second.start_frame_id;
            } else {
                Eigen::Vector3d uv_i = feature.second.feature_local_infos[0].point;
                feature.second.feature_local_infos.erase(feature.second.feature_local_infos.begin());
                if (feature.second.feature_local_infos.size() < 2) {
                    features_map.erase(feature.first);
                    continue;
                } else {
                    Eigen::Vector3d p_feature_c = uv_i * feature.second.estimated_depth;
                    Eigen::Vector3d p_feature_w = marg_R * p_feature_c + marg_P;
                    Eigen::Vector3d p_feature_c_new = new_R.transpose() * (p_feature_w - new_P);
                    double depth_new = p_feature_c_new[2];
                    if (depth_new > 0) {
                        feature.second.estimated_depth = depth_new;
                    } else {
                        feature.second.estimated_depth = INIT_DEPTH;
                    }
                }
            }
            // remove tracking-lost feature after marginalize
            /*
            if (it->endFrame() < WINDOW_SIZE - 1)
            {
                feature.erase(it);
            }
            */
        }
    }

    void FeatureManager::remove_back() {
        for (auto &feature : features_map) {
            if (feature.second.start_frame_id != 0) {
                --feature.second.start_frame_id;
            }
            else {
                feature.second.feature_local_infos.erase(feature.second.feature_local_infos.begin());
                if (feature.second.feature_local_infos.empty()) {
                    features_map.erase(feature.first);
                }
            }
        }
    }

    void FeatureManager::remove_front(unsigned long frame_id) {
        for (auto &feature : features_map) {
            if (feature.second.start_frame_id == frame_id) {
                --feature.second.start_frame_id;
            } else if (feature.second.get_end_frame_id() + 1 >= frame_id) {
                // feature.second.feature_local_infos.begin() + j 对应倒数第二帧
                unsigned long j = WINDOW_SIZE - 1 - feature.second.start_frame_id;
                feature.second.feature_local_infos.erase(feature.second.feature_local_infos.begin() + j);
                if (feature.second.feature_local_infos.empty()) {
                    features_map.erase(feature.first);
                }
            }
        }
    }

    double FeatureManager::compensated_parallax2(const FeatureGlobalInfo &feature, unsigned long frame_count) {
        /*
         * 假设两帧之间没有旋转, 计算两帧之间的基线
         * z = f*b/d  =>  b = z*d/f
         * */

        //check the second last frame is keyframe or not
        //parallax betwwen second last frame and third last frame
        const FeatureLocalInfo &frame_i = feature.feature_local_infos[frame_count - 2 - feature.start_frame_id]; // 倒数第三
        const FeatureLocalInfo &frame_j = feature.feature_local_infos[frame_count - 1 - feature.start_frame_id]; // 倒数第二

        double ans = 0;
        Vector3d p_j = frame_j.point;

        double u_j = p_j(0);
        double v_j = p_j(1);

        Vector3d p_i = frame_i.point;
        Vector3d p_i_comp;

        //int r_i = frame_count - 2;
        //int r_j = frame_count - 1;
        //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
        p_i_comp = p_i;
        double dep_i = p_i(2);
        double u_i = p_i(0) / dep_i;
        double v_i = p_i(1) / dep_i;
        double du = u_i - u_j, dv = v_i - v_j;

        double dep_i_comp = p_i_comp(2);
        double u_i_comp = p_i_comp(0) / dep_i_comp;
        double v_i_comp = p_i_comp(1) / dep_i_comp;
        double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

        ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

        return ans;
    }
}