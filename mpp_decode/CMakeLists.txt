cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0081 NEW)
project(daheng)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_BUILD_RPATH "${CMAKE_SOURCE_DIR}/libs")
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/libs")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_library(MPP_LIB rockchip_mpp HINTS ${CMAKE_SOURCE_DIR}/libs)
message(STATUS "MPP_LIB = ${MPP_LIB}")

file(GLOB SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(main ${SRC_FILES})

target_link_libraries(main PUBLIC ${MPP_LIB})