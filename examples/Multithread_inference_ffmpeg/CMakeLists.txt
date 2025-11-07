
project(rtsp)
cmake_minimum_required(VERSION 3.16)
set( ARCH "X86" STRING)


#Найти OpenCV
find_package(OpenCV REQUIRED)

include_directories("/usr/local/include/opencv4")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(".include/")
include_directories("/home/user/hailort-4.23.0/hailort/libhailort/include/")
include_directories("/usr/local/include/rockchip")
find_library(HAILORT hailort HINTS ${CMAKE_SOURCE_DIR}/libs)
find_library(RGA rga HINTS ${CMAKE_SOURCE_DIR}/libs)

link_directories("/usr/local/lib")
if(ARCH STREQUAL "X86")
	link_directories("/usr/lib/x86_64-linux-gnu")
else()
	link_directories("/usr/lib/aarch64-linux-gnu")
endif()
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
add_executable(main MultithreadInference_ffmpeg_file.cpp )

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
