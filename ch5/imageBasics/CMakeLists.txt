cmake_minimum_required(VERSION 3.0)
project(imageBasics)
set(CMAKE_BUILD_TYPE "Debug")
# 添加c++ 11标准支持
#set(CMAKE_CXX_FLAGS "-std=c++14 -O2")

# Eigen
include_directories("/usr/include/eigen3")

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(imageBasics imageBasics.cpp)
# 链接OpenCV库
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortImage undistortImage.cpp)
# 链接OpenCV库
target_link_libraries(undistortImage ${OpenCV_LIBS})
