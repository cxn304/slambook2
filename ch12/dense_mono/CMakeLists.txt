cmake_minimum_required(VERSION 3.0)
project(dense_monocular)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++14 -march=native -O3")

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES})

add_executable(dense_mapping dense_mapping.cpp)
target_link_libraries(dense_mapping ${THIRD_PARTY_LIBS} fmt::fmt)