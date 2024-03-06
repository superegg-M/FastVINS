add_library(so_plus SHARED so3.hpp se3.hpp sophus.hpp)
set_target_properties(so_plus PROPERTIES LINKER_LANGUAGE CXX)