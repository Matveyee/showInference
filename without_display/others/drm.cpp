
// Обновлённый пример: видео через GStreamer (kmssink), overlay через drmModeSetPlane
// Требуется root-доступ. Рисуем поверх видео на overlay plane

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <thread>
#include <atomic>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>

static std::atomic<bool> running(true);

// --- DRM Plane Overlay --------------------------------------------------

int drm_fd;
uint32_t plane_id = 0, crtc_id = 0, conn_id = 0, fb_id = 0;
uint32_t handle = 0, pitch = 0;
void *fb_mem = nullptr;
int WIDTH = 640;
int HEIGHT = 640;
size_t SIZE = 0;

bool drm_plane_overlay_init() {
    drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (drm_fd < 0) return false;

    drmModeRes *res = drmModeGetResources(drm_fd);
    drmModePlaneRes *pres = drmModeGetPlaneResources(drm_fd);
    if (!res || !pres) return false;

    conn_id = res->connectors[0];
    crtc_id = res->crtcs[0];

    // Найдём overlay plane
    for (uint32_t i = 0; i < pres->count_planes; i++) {
        drmModePlane *plane = drmModeGetPlane(drm_fd, pres->planes[i]);
        if ((plane->possible_crtcs & (1 << 0)) && !(plane->plane_id == crtc_id)) {
            plane_id = plane->plane_id;
            drmModeFreePlane(plane);
            break;
        }
        drmModeFreePlane(plane);
    }

    drmModeFreePlaneResources(pres);
    drmModeFreeResources(res);

    if (!plane_id) return false;

    struct drm_mode_create_dumb create = {};
    create.width = WIDTH;
    create.height = HEIGHT;
    create.bpp = 32;
    if (drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create) < 0) return false;

    handle = create.handle;
    pitch = create.pitch;
    SIZE = create.size;

    struct drm_mode_map_dumb map = { .handle = handle };
    if (drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map) < 0) return false;

    fb_mem = mmap(0, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, map.offset);
    if (!fb_mem) return false;

    memset(fb_mem, 0, SIZE);
    if (drmModeAddFB(drm_fd, WIDTH, HEIGHT, 24, 32, pitch, handle, &fb_id)) return false;

    return true;
}

void drm_draw_overlay_rect(int x, int y, int w, int h, uint32_t color) {
    uint32_t *pixels = (uint32_t *)fb_mem;
    for (int j = y; j < y + h; ++j)
        for (int i = x; i < x + w; ++i)
            if (i >= 0 && i < WIDTH && j >= 0 && j < HEIGHT)
                pixels[j * WIDTH + i] = color;
}

void drm_update_plane() {
    drmModeSetPlane(drm_fd, plane_id, crtc_id, fb_id, 0,
                    0, 0, WIDTH, HEIGHT,
                    0 << 16, 0 << 16, WIDTH << 16, HEIGHT << 16);
}

// --- GStreamer ----------------------------------------------------------

GstFlowReturn on_new_sample(GstAppSink *, gpointer) {
    int box_x = WIDTH / 4;
    int box_y = HEIGHT / 4;
    int box_w = WIDTH / 2;
    int box_h = HEIGHT / 2;
    memset(fb_mem, 0, SIZE);
    drm_draw_overlay_rect(box_x, box_y, box_w, 3, 0xFF00FF00);
    drm_draw_overlay_rect(box_x, box_y + box_h - 3, box_w, 3, 0xFF00FF00);
    drm_draw_overlay_rect(box_x, box_y, 3, box_h, 0xFF00FF00);
    drm_draw_overlay_rect(box_x + box_w - 3, box_y, 3, box_h, 0xFF00FF00);
    drm_update_plane();
    return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    if (!drm_plane_overlay_init()) {
        fprintf(stderr, "Failed to init DRM overlay plane\n");
        return 1;
    }

    GstElement *pipeline = gst_parse_launch(
        "v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720 ! "
        "jpegparse ! mppjpegdec ! "
        "rgaconvert ! video/x-raw,format=RGB,width=640,height=640 ! "
        "kmssink sync=false",
        nullptr);

    GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline), "kmssink0");
    GstPad *sinkpad = gst_element_get_static_pad(sink, "sink");
    gst_pad_add_probe(sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
                      [](GstPad *, GstPadProbeInfo *, gpointer) -> GstPadProbeReturn {
                          on_new_sample(nullptr, nullptr);
                          return GST_PAD_PROBE_OK;
                      },
                      nullptr, nullptr);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    std::thread loop([] { while (running) g_usleep(100000); });

    printf("Press Enter to quit...\n");
    std::cin.get();
    running = false;

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}
