add_library(vo_test
        vertex/vertex_pose.cpp
        vertex/vertex_motion.h
        vertex/vertex_point3d.h
        vertex/vertex_inverse_depth.h
        vertex/vertex_scale.cpp
        vertex/vertex_quaternion.cpp
        vertex/vertex_spherical.cpp
        vertex/vertex_vec1.h

        edge/edge_imu.cpp
        edge/edge_reprojection.cpp
        edge/edge_pnp.cpp
        edge/edge_pnp_sim3.cpp
        edge/edge_epipolar.cpp
        edge/edge_align.cpp
        edge/edge_align_linear.cpp

        2d_to_2d/essential.cpp
        2d_to_2d/homography.cpp

        2d_to_3d/local_triangulate.cpp
        2d_to_3d/global_triangulate.cpp

        3d_to_2d/pnp.cpp
        3d_to_2d/dlt_pnp.cpp
        3d_to_2d/ml_pnp.cpp
        3d_to_2d/e_pnp.cpp
        3d_to_2d/iter_pnp.cpp

        initialize/visual_initialize.cpp
        initialize/inertial_initialize.cpp

        imu_integration.cpp
        data_manager.cpp
        estimator.cpp
)

#find_library(BACKEND_LIB backend
#        ../../lib/backend
#)

#target_link_libraries(vo_test ${BACKEND_LIB})