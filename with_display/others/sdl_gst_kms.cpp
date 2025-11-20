
// Пример: MPP Video Decode + RGA + NPU (заглушка) + kmssink + SDL отрисовка поверх
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <thread>
#include <atomic>
#include <iostream>
#include <SDL2/SDL.h>
#include <hailo/hailort.hpp>
#include <SDL2/SDL.h>
#include "utils.hpp"

#if defined(__unix__)
#include <sys/mman.h>
#endif
using namespace hailort;
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


std::atomic<bool> running(true);

struct Detection {
    int x, y, w, h;
};

struct YourContextStruct {
    SDL_Renderer *renderer;
    ConfiguredInferModel& configured_infer_model;
    std::shared_ptr<InferModel> infer_model;
    
};


void run_inference(YourContextStruct* context, GstMapInfo& map, GstBuffer *buffer, GstSample *sample) {
    auto renderer = context->renderer;
    auto &infer_model = context->configured_infer_model;
    auto bindings = infer_model.create_bindings().expect("Failed to create bindings");

    for (const auto &input_name : context->infer_model->get_input_names()) {
        size_t input_frame_size = context->infer_model->input(input_name)->get_frame_size();
        bindings.input(input_name)->set_buffer(MemoryView(map.data, input_frame_size));
    }
    
    std::shared_ptr<uint8_t> output_buffer;
    for (const auto &output_name : context->infer_model->get_output_names()) {
        size_t output_frame_size = context->infer_model->output(output_name)->get_frame_size();
        output_buffer = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
    }
        
    auto job = infer_model.run_async(bindings,[&](const AsyncInferCompletionInfo & info){
        auto bboxes = parse_nms_data(output_buffer.get(), 80);

        
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0); // semi-transparent green
        SDL_RenderClear(renderer);
        draw_bounding_boxes(context->renderer, bboxes, 640, 640);
        SDL_RenderPresent(renderer);
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        
    });
}

GstFlowReturn on_new_sample(GstAppSink *appsink, gpointer user_data) {
    auto *context = reinterpret_cast<YourContextStruct *>(user_data);

    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_ERROR;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *s = gst_caps_get_structure(caps, 0);

    int width, height;
    gst_structure_get_int(s, "width", &width);
    gst_structure_get_int(s, "height", &height);

    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        run_inference(context, map, buffer, sample);
    }


    return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {

    std::string hef_path = argv[1];
    auto vdevice = VDevice::create().expect("Failed to create vdevice");
    auto infer_model = vdevice->create_infer_model(hef_path).expect("Failed to create infer model");
    auto configured_infer_model = infer_model->configure().expect("Failed to configure model");

    gst_init(&argc, &argv);
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Overlay",
    SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
    640, 640,
    SDL_WINDOW_ALWAYS_ON_TOP | SDL_WINDOW_SKIP_TASKBAR |
    SDL_WINDOW_BORDERLESS );

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    SDL_SetWindowOpacity(window, 0.0f);  // Сделать фон прозрачным

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    GstElement *pipeline = gst_parse_launch(
        "v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720 ! "
        "jpegparse ! mppjpegdec ! "
        "rgaconvert ! video/x-raw,format=RGB,width=640,height=640 ! "
        "tee name=t "
        "t. ! queue ! ximagesink sync=false "
        "t. ! queue ! appsink name=appsink emit-signals=true max-buffers=1 drop=true",
        nullptr);
        // "v4l2src device=/dev/video0 ! "
        // "image/jpeg, width=1280, height=720 ! jpegparse ! mppjpegdec ! "
        // "rgaconvert ! video/x-raw, format=RGB, width=640, height=640 ! "
        // "ximagesink", nullptr);

    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "appsink");

    YourContextStruct context = {
    renderer,
    configured_infer_model,
    infer_model
    };

    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), &context);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    std::cout << "Running... Press Enter to quit\n";
    std::cin.get();

    running = false;
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
