cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(/usr/local/cuda/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(/usr/local/cuda/lib64)
link_directories(${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_C_FLAGS "-std=c++11")

find_package(OpenCV)

add_subdirectory(src)
add_subdirectory(sample)
