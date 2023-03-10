cmake_minimum_required(VERSION 3.0)
project(ch8)

set(CMAKE_BUILD_TYPE "Debug")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++14 ${SSE_FLAGS} -g -O3 -march=native")

include_directories(${FMT_INCLUDE_DIRS})
find_package(FMT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${Pangolin_INCLUDE_DIRS}
)

# add_executable(optical_flow optical_flow.cpp)
# target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
target_link_libraries(direct_method ${THIRD_PARTY_LIBS} fmt::fmt)
