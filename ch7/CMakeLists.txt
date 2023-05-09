cmake_minimum_required(VERSION 3.5)
project(vo1)

set(CMAKE_BUILD_TYPE "Debug")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 ${SSE_FLAGS} -msse4")#增加simd指令集支持
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH "/home/g2o/cmake_modules")

find_package(OpenCV REQUIRED)
# G2O
find_package(G2O REQUIRED)

find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)
include_directories(${FMT_INCLUDE_DIRS})
find_package(FMT REQUIRED)

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

#add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})
#target_link要连接glog
# add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2
add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d
        g2o_core g2o_stuff glog
        ${OpenCV_LIBS})
add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d
        g2o_core g2o_stuff glog fmt::fmt
        ${OpenCV_LIBS})
