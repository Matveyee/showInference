cmake_minimum_required(VERSION 3.18)
project(inference_rknn_file)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/libs )
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(
    "/usr/include/libdrm"
    "/usr/include/")


add_executable(inference_rknn_cam_drm ${CMAKE_SOURCE_DIR}/src/MultithreadInference_rknn_cam_drm.cpp  ${CMAKE_SOURCE_DIR}/src/postprocess.cpp ${CMAKE_SOURCE_DIR}/src/yolov6.cpp ${CMAKE_SOURCE_DIR}/src/file_utils.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp)
target_link_libraries(inference_rknn_cam_drm ${OpenCV_LIBS} rga rknnrt pthread -ldrm)
