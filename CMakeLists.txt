cmake_minimum_required(VERSION 3.10)
project(SpatialSubdivision)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O0")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS ${OpenCV_LIBS})

add_executable(SpatialSubdivision main.cpp)
target_link_libraries(SpatialSubdivision spsub_div)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
