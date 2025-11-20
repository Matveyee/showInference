#include "../include/utils.hpp"
#include <cmath>
std::vector<NamedBbox> parse_nms_data(uint8_t* data, size_t max_class_count) {
    std::vector<NamedBbox> bboxes;
    size_t offset = 0;

    for (size_t class_id = 0; class_id < max_class_count; class_id++) {
        auto det_count = static_cast<uint32_t>(*reinterpret_cast<float32_t*>(data + offset));
        offset += sizeof(float32_t);

        for (size_t j = 0; j < det_count; j++) {
            hailo_bbox_float32_t bbox_data = *reinterpret_cast<hailo_bbox_float32_t*>(data + offset);
            offset += sizeof(hailo_bbox_float32_t);

            NamedBbox named_bbox;
            named_bbox.bbox = bbox_data;
            named_bbox.class_id = class_id + 1;
            bboxes.push_back(named_bbox);
        }
    }
    return bboxes;
}

void draw_line(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch) {
    uint32_t pixel = (0xFF << 24) | (255 << 16) | (0 << 8) | 0;
    if (x == x1) {
        for (int i = y; i <= y1; i++) {
            ((uint32_t*)(map + i * pitch))[x] = pixel;
        }
    } else if (y == y1) {
        for (int i = x; i <= x1; i++) {
            ((uint32_t*)(map + y * pitch))[i] = pixel;
        }
    }
    
}

void draw_rect(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch) {
    draw_line(map,x,y,x1,y, pitch);
    draw_line(map,x,y1,x1,y1, pitch);
    draw_line(map,x,y,x,y1, pitch);
    draw_line(map,x1,y,x1,y1, pitch);
}


void draw_bounding_boxes(uint8_t* map, const std::vector<NamedBbox>& bboxes, int width, int height, uint32_t pitch) {


    for (const auto& named_bbox : bboxes) {
        hailo_bbox_float32_t bbox = named_bbox.bbox;
        int x = static_cast<int>(bbox.x_min * width);
        int y = static_cast<int>(bbox.y_min * height);
        int x1 = x + static_cast<int>((bbox.x_max - bbox.x_min) * width);
        int y1 = y + static_cast<int>((bbox.y_max - bbox.y_min) * height);
//        std::cout << "x = " << x << ", y = " << y << ", x1 = " << x1 << ", y1 = " << y1 << std::endl;
        if (x1 > 0 && x > 0 && y > 0 && y1 > 0) {
            draw_rect(map, x , y, x1 ,y1, pitch);
        }
    }
}
