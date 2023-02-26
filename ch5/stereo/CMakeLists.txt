cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE "Debug")
# 添加c++ 11标准支持
#set(CMAKE_CXX_FLAGS "-std=c++14 -O2")

# Eigen
include_directories("/usr/include/eigen3")

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Pangolin REQUIRED)

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})