#include "utils.hpp"




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
        auto bboxes = parse_nms_data(output_buffer.get(), 80);

       // draw_bounding_boxes(context->map, bboxes, 640, 640, pitch);
        gst_buffer_unmap(buffer, &map_info);
        gst_sample_unref(sample);
       // std::cout << frames << ";" << 1 / (duration.count() / 1000) << std::endl;
        frames++;
        if (frames % 100 == 0) {
            //std::cout << "FPS " << 1 / (duration.count() / 1000) << std::endl;
            std::cout << frames << std::endl;
        }
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
        std::cout << i << ";" << 1 / (times[i] / 1000) << std::endl;
    }
    std::cout << "Frames count = " << times.size() << std::endl;
    std::cout << "Average inference FPS = " << times.size() / (sum / 1000) << std::endl;
    std::cout << "Average FPS = " << frames / (duration.count() / 1000) << std::endl;
    

    return 0;
}
