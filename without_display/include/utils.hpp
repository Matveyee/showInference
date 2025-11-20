#ifndef UTILS_H
#define UTILS_H
extern "C" {
//    #include <gst/gst.h>
//    #include <gst/app/gstappsink.h>

    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/ioctl.h>
    #include <sys/mman.h>
    #include <xf86drm.h>
    #include <xf86drmMode.h>
    #include <drm/drm_mode.h>
}
#include <iostream>
#include <vector>
#include <mutex>
#include "opencv2/opencv.hpp"
#include <unistd.h> // для Unix систем
#include "hailo/hailort.hpp"
#include <chrono>
#include <bits/stdc++.h>

struct NamedBbox {
    hailo_bbox_float32_t bbox;
    size_t class_id;
};

// Парсинг данных NMS из буфера
std::vector<NamedBbox> parse_nms_data(uint8_t* data, size_t max_class_count);

void draw_rect(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch);

void draw_line(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch);

void draw_bounding_boxes(uint8_t* map, const std::vector<NamedBbox>& bboxes, int width, int height, uint32_t pitch);



#endif