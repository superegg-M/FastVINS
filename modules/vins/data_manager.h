//
// Created by Cain on 2024/4/2.
//

#ifndef VINS_DATA_MANAGER_H
#define VINS_DATA_MANAGER_H

#include <iostream>
#include <unordered_map>
#include "parameters.h"
#include "imu_integration.h"
#include "edge_imu.h"
#include "edge_reprojection.h"
#include "vertex_inverse_depth.h"
#include "vertex_point3d.h"
#include "vertex_pose.h"
#include "vertex_motion.h"

namespace vins {
    using namespace std;
    using namespace graph_optimization;

    template<typename T>
    class Queue {
    public:
        explicit Queue(unsigned long n) : N(n), _head(n - 1) {
            _data = new T[N];
        }

        explicit Queue(unsigned long n, const T &value) : N(n), _head(n - 1) {
            _data = new T[N];
            for (const T &v : _data) {
                v = value;
            }
        }

        virtual ~Queue() {
            delete [] _data;
        }

        T &operator[](unsigned long index) {
            unsigned long res = N - _tail;
            index %= _size;
            index = (index < res) ? (_tail + index) : (index - res);
            return _data[index];
        }

        const T&operator[](unsigned long index) const {
            unsigned long res = N - _tail;
            index %= _size;
            index = (index < res) ? (_tail + index) : (index - res);
            return _data[index];
        }

        bool empty() const { return _size == 0; };
        bool full() const { return _size == N; };
        const T &newest() const { return _data[_head]; };
        const T &oldest() const { return _data[_tail]; };
        unsigned long newest_index() const { return _head; };
        unsigned long oldest_index() const { return _tail; };
        unsigned long size() const { return _size; };
        unsigned long capacity() const { return N; };
        void clear() { _size = 0; _head = N - 1; _tail = 0; }

        void push(const T &value) {
            if (++_head == N) {
                _head = 0;
            }
            _data[_head] = value;

            if (++_size > N) {
                _size = N;
                if (++_tail == N) {
                    _tail = 0;
                }
            }
        }

        bool pop(T & value) {
            if (_size > 0) {
                value = _data[_tail];

                if (++_tail == N) {
                    _tail = 0;
                }

                --_size;

                return true;
            } else {
                return false;
            }
        }

        bool pop() {
            if (_size > 0) {

                if (++_tail == N) {
                    _tail = 0;
                }

                --_size;

                return true;
            } else {
                return false;
            }
        }

    protected:
        const unsigned long N;
        T *_data {nullptr};
        unsigned long _size {0};
        unsigned long _head;
        unsigned long _tail {0};
    };

    template<typename T>
    class Deque : public Queue<T> {
    public:
        explicit Deque(unsigned long n) : Queue<T>(n) {
        }

        explicit Deque(unsigned long n, const T &value) : Queue<T>(n, value) {
        }

        void push_newest(const T &value) {
            Queue<T>::push(value);
        }

        void push_oldest(const T &value) {
            if (Queue<T>::_tail-- == 0) {
                Queue<T>::_tail = Queue<T>::N - 1;
            }
            Queue<T>::_data[Queue<T>::_tail] = value;

            if (Queue<T>::_size++ == Queue<T>::N) {
                Queue<T>::_size = Queue<T>::N;
                if (Queue<T>::_head-- == 0) {
                    Queue<T>::_head = Queue<T>::N - 1;
                }
            }
        }

        bool pop_oldest(T & value) {
            return Queue<T>::pop(value);
        }

        bool pop_oldest() {
            return Queue<T>::pop();
        }

        bool pop_newest(T &value) {
            if (Queue<T>::_size > 0) {
                value = Queue<T>::_data[Queue<T>::_head];

                if (Queue<T>::_head-- == 0) {
                    Queue<T>::_head = Queue<T>::N - 1;
                }

                --Queue<T>::_size;

                return true;
            } else {
                return false;
            }
        }

        bool pop_newest() {
            if (Queue<T>::_size > 0) {

                if (Queue<T>::_head-- == 0) {
                    Queue<T>::_head = Queue<T>::N - 1;
                }

                --Queue<T>::_size;

                return true;
            } else {
                return false;
            }
        }
    };

    class ImuNode;
    class FrameNode;
    class FeatureNode;

    class ImuNode {
    public:
        explicit ImuNode(IMUIntegration *imu_integration_pt, unsigned int num_features=NUM_OF_F);
        ~ImuNode();

        Vec3 get_p() const { return {vertex_pose->get_parameters()(0), vertex_pose->get_parameters()(1), vertex_pose->get_parameters()(2)};}
        Qd get_q() const { return {vertex_pose->get_parameters()(6), vertex_pose->get_parameters()(3), vertex_pose->get_parameters()(4), vertex_pose->get_parameters()(5)}; }
        Vec3 get_v() const { return {vertex_motion->get_parameters()(0), vertex_motion->get_parameters()(1), vertex_motion->get_parameters()(2)}; }
        Vec3 get_ba() const { return {vertex_motion->get_parameters()(3), vertex_motion->get_parameters()(4), vertex_motion->get_parameters()(5)}; }
        Vec3 get_bg() const { return {vertex_motion->get_parameters()(6), vertex_motion->get_parameters()(7), vertex_motion->get_parameters()(8)}; }

    public:
        bool is_key_frame {false};

        shared_ptr<VertexPose> vertex_pose {new VertexPose};
        shared_ptr<VertexMotion> vertex_motion {new VertexMotion};

//        vector<pair<unsigned long, FrameNode*>> frames; ///< 每个相机的frame
        unordered_map<unsigned long, vector<pair<unsigned long, Vec3>>> features_in_cameras;

        IMUIntegration *imu_integration;
    };

//    class FrameNode {
//    public:
//        explicit FrameNode(unsigned long id, ImuNode *imu);
//        unsigned long id() const { return _camera_id; }
//        ImuNode *imu_pt() const { return _imu_pt; }
//        bool is_feature_in_frame(unsigned long feature_id) const { return features.find(feature_id) != features.end(); }
//
//    public:
//        unordered_map<unsigned long, pair<FeatureNode*, Vec3>> features;
//
//    private:
//        const unsigned long _camera_id;
//        ImuNode *_imu_pt;
//    };

    class FeatureNode {
    public:
        explicit FeatureNode(unsigned long id);

        unsigned long id() const { return _feature_id; }
//        pair<unsigned long, FrameNode *> get_reference_frame() const { return imu_deque[0]->frames[0]; }
        void from_global_to_local(const std::vector<Qd> &q_ic, const vector<Vec3> &t_ic);
        void from_local_to_global(const std::vector<Qd> &q_ic, const vector<Vec3> &t_ic);

    public:
        bool is_triangulated {false};
        Vec3 point;

        shared_ptr<VertexInverseDepth> vertex_landmark {nullptr};
        shared_ptr<VertexPoint3d> vertex_point3d {nullptr};
        Deque<ImuNode *> imu_deque;

    private:
        const unsigned long _feature_id;
    };

    class ImuWindows : public Deque<ImuNode *>{
    public:
        explicit ImuWindows(unsigned long n);
        bool is_feature_in_newest(unsigned long feature_id) const;
        bool is_feature_suitable_to_reproject(unsigned long feature_id) const;
    };

    class State {
    public:
        Vec3 p {};
        Qd q {};
        Vec3 v {};
        Vec3 ba {};
        Vec3 bg {};
    };
}

#endif //VINS_DATA_MANAGER_H
