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
std::vector<double> times;
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

    int min_width = std::min((int)fb.width, 640);
    int min_height = std::min((int)fb.height, 640);
    for (int y = 0; y < min_height; ++y) {
        for (int x = 0; x < min_width; ++x) {
            int src_offset = y * min_width * 3 + x * 3;
            uint8_t r = map_info.data[src_offset + 0];
            uint8_t g = map_info.data[src_offset + 1];
            uint8_t b = map_info.data[src_offset + 2];

            uint32_t pixel = (0xFF << 24) | (r << 16) | (g << 8) | b;

            ((uint32_t*)(context->map + y * pitch))[x] = pixel;
        }

    }
    auto &infer_model = context->configured_infer_model;
    auto bindings = infer_model.create_bindings().expect("Failed to create bindings");

    for (const auto &input_name : context->infer_model->get_input_names()) {
        size_t input_frame_size = context->infer_model->input(input_name)->get_frame_size();
        bindings.input(input_name)->set_buffer(MemoryView(map_info.data, input_frame_size));
    }

    std::shared_ptr<uint8_t> output_buffer;
    for (const auto &output_name : context->infer_model->get_output_names()) {
        size_t output_frame_size = context->infer_model->output(output_name)->get_frame_size();
        output_buffer = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto job = infer_model.run_async(bindings,[&, start](const AsyncInferCompletionInfo & info){
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());
        std::cout <<  1 / (duration.count() / 1000) << std::endl;
        auto bboxes = parse_nms_data(output_buffer.get(), 80);

        draw_bounding_boxes(context->map, bboxes, 640, 640, pitch);
        gst_buffer_unmap(buffer, &map_info);
        gst_sample_unref(sample);
        frames++;
    }).expect("Failed to start async infer job");
    
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
    " ! image/jpeg, width=640, height=480, framerate=30/1 "
    " ! jpegparse ! mppjpegdec ! rgaconvert "
    " ! video/x-raw,format=RGB,width=640,height=640 ! appsink name=sink";

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
    double sum = 0;
    for (int i = 0; i < times.size(); i++) {
        sum += times[i];
    }
    std::cout << "Average inference FPS = " << times.size() / (sum / 1000) << std::endl;
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
