cmake_minimum_required(VERSION 2.8)
project(graph)

ADD_DEFINITIONS(-std=c++11 -DGLOG_NO_ABBREVIATED_SEVERITIES)

find_package(OpenCV REQUIRED)
message("version is ${OPENCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp graph.cpp)
target_link_libraries(main ${OpenCV_LIBS})