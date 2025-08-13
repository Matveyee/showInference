project(inference_rknn_file)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/libs )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(inference_rknn_file ${CMAKE_SOURCE_DIR}/src/MultithreadInference_rknn_file.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp)
target_link_libraries(inference_rknn_file ${OpenCV_LIBS} rga rknnrt pthread)
