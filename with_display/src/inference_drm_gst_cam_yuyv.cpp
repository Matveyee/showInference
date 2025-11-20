#include "../include/utils.hpp"




struct drm_mode_fb_cmd fb = {};
int drm_fd;
uint8_t* map = nullptr;
uint32_t fb_id;
uint32_t handle;
uint32_t pitch;
uint64_t size;
drmModeCrtc *old_crtc = nullptr;
int running = 1;
int frames = 0;

static inline uint8_t clamp(int val) {
    return std::max(0, std::min(255, val));
}

static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size)
{
#if defined(__unix__)
    auto addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
#elif defined(_MSC_VER)
    auto addr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [](void *addr){ VirtualFree(addr, 0, MEM_RELEASE); });
#else
#pragma error("Aligned alloc not supported")
#endif
}

void sigint_handler(int) {
    running = 0;
}

using namespace hailort;

struct YourContextStruct {
    uint8_t* map;
    ConfiguredInferModel& configured_infer_model;
    std::shared_ptr<InferModel> infer_model;
};
void debug( std::string message) {
    std::cout << "DEBUG: " << message << std::endl;
}

void firstBytes(guint8* data) {
    for (int i = 0; i <= 20; i++) {
        std::cout << (int)data[i] << ", ";
    }
    std::cout << std::endl;
}
GstFlowReturn on_new_sample(GstAppSink *appsink, gpointer user_data) {
    auto *context = reinterpret_cast<YourContextStruct *>(user_data);

    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_OK;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    
    }
    int image_width = 640;
    int image_height = 480;

    int min_width = std::min((int)fb.width, image_width);
    int min_height = std::min((int)fb.height, image_height);

    for (int y = 0; y < min_height; ++y) {
        for (int x = 0; x < min_width; x += 2) {
            // 👉 Правильный offset: вся строка = image_width * 2 байт
            int src_offset = y * image_width * 2 + x * 2;

            uint8_t y0 = map_info.data[src_offset + 0];
            uint8_t u  = map_info.data[src_offset + 1];
            uint8_t y1 = map_info.data[src_offset + 2];
            uint8_t v  = map_info.data[src_offset + 3];

            int d = u - 128;
            int e = v - 128;
            int c0 = y0 - 16;
            int c1 = y1 - 16;

            uint8_t r0 = clamp((298 * c0 + 409 * e + 128) >> 8);
            uint8_t g0 = clamp((298 * c0 - 100 * d - 208 * e + 128) >> 8);
            uint8_t b0 = clamp((298 * c0 + 516 * d + 128) >> 8);

            uint8_t r1 = clamp((298 * c1 + 409 * e + 128) >> 8);
            uint8_t g1 = clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);
            uint8_t b1 = clamp((298 * c1 + 516 * d + 128) >> 8);

            uint32_t pixel0 = (0xFF << 24) | (r0 << 16) | (g0 << 8) | b0;
            uint32_t pixel1 = (0xFF << 24) | (r1 << 16) | (g1 << 8) | b1;

            ((uint32_t*)(map + y * pitch))[x] = pixel0;
            ((uint32_t*)(map + y * pitch))[x + 1] = pixel1;
        }
    }
    //     auto &infer_model = context->configured_infer_model;
    //     auto bindings = infer_model.create_bindings().expect("Failed to create bindings");

    //     for (const auto &input_name : context->infer_model->get_input_names()) {
    //         size_t input_frame_size = context->infer_model->input(input_name)->get_frame_size();
    //         bindings.input(input_name)->set_buffer(MemoryView(map_info.data, input_frame_size));
    //     }

    //     std::shared_ptr<uint8_t> output_buffer;
    //     for (const auto &output_name : context->infer_model->get_output_names()) {
    //         size_t output_frame_size = context->infer_model->output(output_name)->get_frame_size();
    //         output_buffer = page_aligned_alloc(output_frame_size);
    //         bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
    //     }

    // auto job = infer_model.run_async(bindings,[&](const AsyncInferCompletionInfo & info){
    //     auto bboxes = parse_nms_data(output_buffer.get(), 80);

    //     draw_bounding_boxes(context->map, bboxes, 640, 480, pitch);
    //     gst_buffer_unmap(buffer, &map_info);
    //     gst_sample_unref(sample);
    //     frames++;
    // }).expect("Failed to start async infer job");
    gst_buffer_unmap(buffer, &map_info);
    gst_sample_unref(sample);
    
    return GST_FLOW_OK;


}

int main(int argc, char* argv[]) {
    std::string hef_path = argv[1];
    std::string source = argv[2];
    auto vdevice = VDevice::create().expect("Failed to create vdevice");
    auto infer_model = vdevice->create_infer_model(hef_path).expect("Failed to create infer model");
    auto configured_infer_model = infer_model->configure().expect("Failed to configure model");

    signal(SIGINT, sigint_handler);
    gst_init(&argc, &argv);

    drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (drm_fd < 0) {
        perror("open");
        return 1;
    }

    drmModeRes* res = drmModeGetResources(drm_fd);
    drmModeConnector* conn = nullptr;
    drmModeModeInfo mode;
    uint32_t conn_id = 0;

    for (int i = 0; i < res->count_connectors; ++i) {
        conn = drmModeGetConnector(drm_fd, res->connectors[i]);
        if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
            mode = conn->modes[0];
            conn_id = conn->connector_id;
            break;
        }
        drmModeFreeConnector(conn);
    }

    if (!conn_id) {
        std::cerr << "No connected display found\n";
        return 1;
    }

    drmModeEncoder* enc = drmModeGetEncoder(drm_fd, conn->encoder_id);
    uint32_t crtc_id = enc->crtc_id;
    old_crtc = drmModeGetCrtc(drm_fd, crtc_id);

    struct drm_mode_create_dumb create = {};
    create.width = mode.hdisplay;
    create.height = mode.vdisplay;
    create.bpp = 32;
    ioctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create);
    handle = create.handle;
    pitch = create.pitch;
    size = create.size;

    struct drm_mode_map_dumb map_dumb = {};
    map_dumb.handle = handle;
    ioctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_dumb);
    map = (uint8_t*)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, map_dumb.offset);

    
    fb.width = mode.hdisplay;
    fb.height = mode.vdisplay;
    fb.pitch = pitch;
    fb.bpp = 32;
    fb.depth = 24;
    fb.handle = handle;
    drmModeAddFB(drm_fd, fb.width, fb.height, fb.depth, fb.bpp, pitch, handle, &fb_id);

    drmModeSetCrtc(drm_fd, crtc_id, fb_id, 0, 0, &conn_id, 1, &mode);

    // GStreamer: pipeline with appsink
    std::string pipeline_desc =
    "v4l2src device=" + source +
    " ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 "
    " ! appsink name=sink";

    GstElement *pipeline = gst_parse_launch(pipeline_desc.c_str(), nullptr);
    GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    g_object_set(appsink, "emit-signals", TRUE, NULL);

    YourContextStruct context = {
        map,
        configured_infer_model,
        infer_model
    };
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), &context);
    auto start = std::chrono::high_resolution_clock::now();
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    while (running == 1) {
        //usleep(10000);
    } 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Average FPS = " << frames / (duration.count() / 1000) << std::endl;
    if (old_crtc) {
        drmModeSetCrtc(drm_fd, old_crtc->crtc_id, old_crtc->buffer_id,
                       old_crtc->x, old_crtc->y,
                       &conn_id, 1, &old_crtc->mode);
        drmModeFreeCrtc(old_crtc);
    }

    munmap(map, size);
    drmModeRmFB(drm_fd, fb_id);

    struct drm_mode_destroy_dumb destroy = {0};
    destroy.handle = handle;
    drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);

    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(drm_fd);

    return 0;
}
