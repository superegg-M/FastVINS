add_library(backend eigen_types.h vertex.cpp edge.cpp problem.cpp steepest_descent.cpp gauss_newton.cpp levenberg_marquardt.cpp dog_leg.cpp loss_function.cpp)

target_include_directories(backend PUBLIC ${GRAPH_OPTIMIZATION_SOURCE_DIR})

#target_link_libraries(backend PUBLIC eigen3)
#target_include_directories(backend PUBLIC ${LM_SOURCE_DIR})