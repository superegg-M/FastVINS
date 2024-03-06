add_library(test_ba
        vertex_pose.cpp
        vertex_motion.h
        vertex_point3d.h
        vertex_inverse_depth.h
        edge_reprojection.cpp
        imu_integration.cpp)
target_link_libraries(test_ba backend)