cmake_minimum_required(VERSION 3.10)

#set(CMAKE_BUILD_TYPE Release)

project(main)
set(CMAKE_C_COMPILER /usr/bin/gcc CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER /usr/bin/g++ CACHE PATH "" FORCE)

#include_directories("/usr/local/include")

#ceres
find_package(Ceres REQUIRED)

#add_compile_options( -Wall -Wextra)

#set(Targets test_arithmetic_operator test_neural_network test_matrix  
#	test_eigen_helper test_init_var test_losses test_IO test_distributions)
#set(CeresTragets test_ceres_optimizer)
#set(Targets test_arithmetic_operator test_distributions)
set(Targets test_arithmetic_operator)
foreach(tgt ${Targets})
	add_executable(${tgt} ${tgt}.cpp)
	target_compile_features(${tgt} PUBLIC cxx_std_20)
	#target_include_directories(${tgt} PUBLIC ${CERES_INCLUDE_DIRS})
#target_link_libraries(${tgt}  Ceres::ceres gflags)

endforeach()
