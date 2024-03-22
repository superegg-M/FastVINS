add_library(tic_toc tic_toc.h)
#target_include_directories(tic_toc PUBLIC ${GRAPH_OPTIMIZATION_SOURCE_DIR})
set_target_properties(tic_toc PROPERTIES LINKER_LANGUAGE CXX)