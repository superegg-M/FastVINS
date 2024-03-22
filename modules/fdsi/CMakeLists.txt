add_library(fdsi p1id.cpp fdsi.cpp)

#find_library(BACKEND_LIB backend
#        ${CMAKE_BINARY_DIR}/lib/backend
#)
#target_link_libraries(fdsi ${BACKEND_LIB})
target_link_libraries(fdsi backend tic_toc so_plus)