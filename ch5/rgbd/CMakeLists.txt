cmake_minimum_required(VERSION 3.0)
project(joinMap)
set(CMAKE_BUILD_TYPE "Debug")

# Eigen
include_directories("/usr/include/eigen3")

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

add_executable(joinMap joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
target_link_libraries(joinMap Sophus::Sophus)