cmake_minimum_required(VERSION 3.10)
project(main)
set(CMAKE_C_COMPILER /usr/bin/gcc CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER /usr/bin/g++ CACHE PATH "" FORCE)

include_directories("/usr/local/include")
#eigen3
find_package(Eigen3)
include_directories(EIGEN3_INCLUDE_DIR)

add_executable(mp src.cpp)
target_compile_features(mp PUBLIC cxx_std_20)

#nlopt
find_package(NLopt REQUIRED)
target_link_libraries(mp nlopt)