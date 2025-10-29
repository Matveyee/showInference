cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0081 NEW)
project(daheng)

include_directories("./include")
include_directories("./include/rga")
include_directories( /usr/include/libdrm)
find_package(HailoRT REQUIRED)



find_library(GX_LIB gxiapi HINTS ${CMAKE_SOURCE_DIR}/libs)
find_library(DRM_LIB drm HINTS /usr/lib /usr/local/lib)
find_library(RGA_LIB rga HINTS ${CMAKE_SOURCE_DIR}/libs)

message(STATUS "GX_LIB = ${GX_LIB}")
message(STATUS "DRM_LIB = ${DRM_LIB}")
message(STATUS "RGA_LIB = ${RGA_LIB}")

file(GLOB SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(main ${SRC_FILES})

target_link_libraries(main PUBLIC ${GX_LIB} ${DRM_LIB} ${RGA_LIB} HailoRT::libhailort -pthread )