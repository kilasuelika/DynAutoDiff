cmake_minimum_required(VERSION 3.10)
project(main)
set(CMAKE_C_COMPILER /usr/bin/gcc CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER /usr/bin/g++ CACHE PATH "" FORCE)

include_directories("/usr/local/include")
include_directories("../../DynAutoDiff")

add_executable(mp src.cpp)
target_compile_features(mp PUBLIC cxx_std_20)

#ceres
find_package(Ceres REQUIRED)
target_include_directories(mp PUBLIC ${CERES_INCLUDE_DIRS})
target_link_libraries(mp  Ceres::ceres gflags)

#openmp
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
