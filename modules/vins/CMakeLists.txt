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

find_library(BACKEND_LIB backend
        ../../lib/backend
)

target_link_libraries(vins ${BACKEND_LIB})
