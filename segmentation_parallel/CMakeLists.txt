cmake_minimum_required(VERSION 2.8)
project(graph)

find_package(CUDA QUIET REQUIRED)
set (CUDA_PROPAGATE_HOST_FLAGS ON)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
set (CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
message("version is ${OPENCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

#find_package(Boost)
#if(Boost_FOUND)
#    include_directories(${Boost_INCLUDE_DIRS}) 
#endif()

add_executable(main main.cpp graph.cpp selective.cpp)
target_link_libraries(main ${OpenCV_LIBS})

add_executable(boru Boruvka.cpp selective.cpp)
target_link_libraries(boru ${OpenCV_LIBS})