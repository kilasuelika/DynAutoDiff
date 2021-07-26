cmake_minimum_required(VERSION 3.10)

#set(CMAKE_BUILD_TYPE Release)

project(main)

include_directories("/usr/local/include")
#eigen3
find_package(Eigen3)
include_directories(EIGEN3_INCLUDE_DIR)

#add_compile_options( -Wall -Wextra)

set(Targets  hello)
add_executable(hello "hello.cpp")
target_compile_features(hello PUBLIC cxx_std_20)
