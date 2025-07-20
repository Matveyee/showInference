cmake_minimum_required(VERSION 3.10)
project(VideoPlayerWithSDL2AndGStreamer)



# Найти OpenCV
#find_package(OpenCV REQUIRED)

# Найти GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)

# Найти  Hailo
find_package(HailoRT REQUIRED)

# Найти SDL2
#pkg_check_modules(SDL2 REQUIRED sdl2)



# Включить заголовки
include_directories(
    "./include"
#    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
#    ${SDL2_INCLUDE_DIRS}
    "/usr/include/libdrm"
    "/usr/include/"
)

# Добавить исполняемый файл
add_executable(main src/inference_gst_cam_mjpg.cpp src/utils.cpp)

# Линковка с библиотеками
target_link_libraries(main
    HailoRT::libhailort PkgConfig::GST PkgConfig::GST_APP -pthread -ldrm -lrockchip_mpp -lavformat -lavcodec -lavutil -lz 
)