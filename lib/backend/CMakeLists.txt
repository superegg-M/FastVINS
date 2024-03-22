add_library(backend
        eigen_types.h
        vertex.cpp
        edge.cpp
        problem.cpp
        steepest_descent.cpp
        gauss_newton.cpp
        levenberg_marquardt.cpp
        dog_leg.cpp
        loss_function.cpp
        problem_slam.cpp)

set_target_properties(backend PROPERTIES LINKER_LANGUAGE CXX)

#set_target_properties(backend PROPERTIES
#        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#)
#set_target_properties(backend PROPERTIES
#        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
#)

#find_library(TIC_TOC_LIB tic_toc
#        ${CMAKE_BINARY_DIR}/lib/tic_toc
#)
#find_library(SO_PLUS_LIB so_plus
#        ${CMAKE_BINARY_DIR}/lib/thirdparty/Sophus/sophus
#)
#target_link_libraries(backend ${TIC_TOC_LIB} ${SO_PLUS_LIB})
target_link_libraries(backend tic_toc so_plus)

#target_include_directories(backend PUBLIC ${GRAPH_OPTIMIZATION_SOURCE_DIR})

#target_link_libraries(backend PUBLIC eigen3)
#target_include_directories(backend PUBLIC ${LM_SOURCE_DIR})