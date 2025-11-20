
project(rtsp)
cmake_minimum_required(VERSION 3.16)
set( ARCH "X86" STRING)


#Найти OpenCV
find_package(OpenCV REQUIRED)

include_directories("/usr/local/include/opencv4")
include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(".include/")
include_directories("/usr/local/include/rockchip")
include_directories("/home/user/hailort-4.23.0/hailort/libhailort/include/")

include_directories(${OpenCV_INCLUDE_DIRS})
find_library(HAILORT hailort HINTS ${CMAKE_SOURCE_DIR}/libs)
find_library(RGA rga HINTS ${CMAKE_SOURCE_DIR}/libs)

link_directories("/usr/local/lib")
if(ARCH STREQUAL "X86")
	link_directories("/usr/lib/x86_64-linux-gnu")
else()
	link_directories("/usr/lib/aarch64-linux-gnu")
endif()

add_executable(main MultithreadInference_mp4_file_640_READY.cpp )

target_link_libraries(main
    -pthread
    -lssl
    -lcrypto
    -lavformat
    -lavcodec
    -lavutil
    -lswscale
    -lswresample
    -lz
    -llzma
    -lm
    
    -ldl
    ${HAILORT}
    ${RGA}
)
