#ifndef UTILS_HPP
#define UTILS_HPP

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <libdrm/drm_mode.h>
#include <string>
#include <iostream>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <mutex>
#include <vector>
#if defined(__unix__)
#include <sys/mman.h>
#endif
#include <unistd.h>
// #include "hailo/hailort.hpp"


extern drmModeRes* res;
extern drmModeConnector* conn;
extern uint32_t conn_id;
extern drmModeCrtc* old_crtc;
extern uint8_t* map;
extern uint32_t fb_id;
extern uint32_t handle;
extern uint32_t pitch;
extern uint64_t size;
extern struct drm_mode_fb_cmd fb;


uint16_t modbus_crc16( const unsigned char *buf, unsigned int len );

class Vec {
private:
    int x;
    int y;

public:
    Vec(int p_x, int p_y);
    Vec();

    void init(int p_x, int p_y);
    int getx();
    int gety();
    Vec& operator+=(Vec& other);
    void print();
};

Vec operator+(Vec& first, Vec& second);

class Projection {
private:
    uint8_t* data;
    int offsetX;
    int offsetY;
    int sourceW;
    int sourceH;
    int w;
    int h;

public:
    Projection();
    Projection(uint8_t* data_ptr, int offset_x, int offset_y, int source_w, int source_h, int width, int height);

    uint8_t operator[](int index);
    void init(uint8_t* data_ptr, int offset_x, int offset_y, int source_w, int source_h, int width, int height);
    uint8_t get(int x, int y);
    uint8_t get(Vec vec);
    int getx();
    int gety();
    int getW();
    int getH();
};



void log(std::string message);


void draw_line(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch);
void draw_rect(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch);


void specifyVectors(Vec& R, Vec& G1, Vec& G2, Vec& B, Vec r);

uint8_t* toRGB(Projection input);

void drawPicture(Projection* proj, uint8_t* data);

void drm_init(int drm_fd);
void drm_destroy(int drm_fd);

#endif // UTILS_HPP
