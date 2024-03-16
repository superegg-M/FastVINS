add_library(vins
        vertex_pose.cpp
        vertex_motion.h
        vertex_point3d.h
        vertex_inverse_depth.h
        edge_imu.cpp
        edge_reprojection.cpp
        imu_integration.cpp
        feature_manager.cpp
        )
target_link_libraries(vins backend)