
project(rtsp)
cmake_minimum_required(VERSION 3.16)
set( ARCH "X86" STRING)


#Найти OpenCV
find_package(OpenCV REQUIRED)

include_directories("/usr/local/include/opencv4")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories("/home/user/OnlineDetection/include")
include_directories("include/")
include_directories("include/rga")
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(HailoRT REQUIRED)
link_directories(${CMAKE_SOURCE_DIR}/libs )

link_directories("/usr/local/lib")
if(ARCH STREQUAL "X86")
	link_directories("/usr/lib/x86_64-linux-gnu")
else()
	link_directories("/usr/lib/aarch64-linux-gnu")
endif()
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
add_executable(main src/MultithreadInference_yuyv_rga_file.cpp )

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
    -lrga
    -ldrm
    -ldl
    HailoRT::libhailort
    ${OpenCV_LIBS}
    -lrockchip_mpp
)
target_link_libraries(main HailoRT::libhailort)
