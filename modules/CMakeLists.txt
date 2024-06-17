#add_library(LM_src rate_model_fit.cpp)
#
#target_include_directories(LM_src PUBLIC ${LM_SOURCE_DIR})

#target_link_libraries(LM_src PRIVATE LM)

add_subdirectory(fdsi)
add_subdirectory(vo_test)
#add_subdirectory(vins_mono)
add_subdirectory(vins_stereo)
#add_subdirectory(test_ba)
#add_subdirectory(vins)