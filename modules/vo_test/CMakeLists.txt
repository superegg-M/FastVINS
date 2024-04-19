add_library(vo_test
        vertex_pose.cpp
        vertex_motion.h
        vertex_point3d.h
        vertex_inverse_depth.h
        vertex_scale.cpp
        vertex_quaternion.cpp
        vertex_spherical.cpp
        edge_imu.cpp
        edge_reprojection.cpp
        edge_pnp.cpp
        edge_pnp_sim3.cpp
        edge_epipolar.cpp
        imu_integration.cpp
        data_manager.cpp
        estimator.cpp
)

#find_library(BACKEND_LIB backend
#        ../../lib/backend
#)

#target_link_libraries(vo_test ${BACKEND_LIB})