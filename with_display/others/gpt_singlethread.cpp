#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <SDL2/SDL.h>
#include <hailo/hailort.hpp>
#include <iostream>
#include <chrono>
#include "../include/utils.hpp"
#include <cstdint>
#include <algorithm>

#if defined(__unix__)
#include <sys/mman.h>
#endif

using namespace hailort;

#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_RESET   "\033[0m"



int WINDOW_WIDTH =  640;
int WINDOW_HEIGHT = 480;
int infer = 1;
int rgb = 0;
int frame_counter = 0;
void debug(std::string message) {
    std::cout << COLOR_YELLOW <<"DEBUG: "  <<message << COLOR_RESET  << std::endl;
}

inline uint8_t clamp(int value) {
    return static_cast<uint8_t>(std::max(0, std::min(255, value)));
}


void yuy2_to_rgb(const uint8_t *yuy2, uint8_t *rgb, int width, int height) {
    for (int i = 0; i < width * height * 2; i += 4) {
        uint8_t y0 = yuy2[i + 0];
        uint8_t u  = yuy2[i + 1];
        uint8_t y1 = yuy2[i + 2];
        uint8_t v  = yuy2[i + 3];

        int c0 = y0 - 16;
        int c1 = y1 - 16;
        int d = u - 128;
        int e = v - 128;

        int r0 = (298 * c0 + 409 * e + 128) >> 8;
        int g0 = (298 * c0 - 100 * d - 208 * e + 128) >> 8;
        int b0 = (298 * c0 + 516 * d + 128) >> 8;

        int r1 = (298 * c1 + 409 * e + 128) >> 8;
        int g1 = (298 * c1 - 100 * d - 208 * e + 128) >> 8;
        int b1 = (298 * c1 + 516 * d + 128) >> 8;

        *rgb++ = clamp(r0);
        *rgb++ = clamp(g0);
        *rgb++ = clamp(b0);

        *rgb++ = clamp(r1);
        *rgb++ = clamp(g1);
        *rgb++ = clamp(b1);
    }
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

struct YourContextStruct {
    SDL_Renderer *renderer;
    SDL_Texture *texture;
    int width, height;

    ConfiguredInferModel& configured_infer_model;
    std::shared_ptr<InferModel> infer_model;
};


// Callback для получения новых кадров
GstFlowReturn on_new_sample(GstAppSink *appsink, gpointer user_data) {
    auto *context = reinterpret_cast<YourContextStruct *>(user_data);
    debug("on_new_sample entered");
    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_OK;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    if (infer == 1) {
        //----------- HAILO INFERENCE ------------
        
        auto &infer_model = context->configured_infer_model;
        auto bindings = infer_model.create_bindings().expect("Failed to create bindings");
        if (rgb == 3) {
            const uint8_t* yuy2_data = map.data;
            std::vector<uint8_t> rgb_buffer(context->width * context->height * 3);
            yuy2_to_rgb(yuy2_data, rgb_buffer.data(), context->width, context->height);

            for (const auto &input_name : context->infer_model->get_input_names()) {
                size_t input_frame_size = context->infer_model->input(input_name)->get_frame_size();
                bindings.input(input_name)->set_buffer(MemoryView(rgb_buffer.data(), input_frame_size));
            }
        }else {
            for (const auto &input_name : context->infer_model->get_input_names()) {
                size_t input_frame_size = context->infer_model->input(input_name)->get_frame_size();
                bindings.input(input_name)->set_buffer(MemoryView(map.data, input_frame_size));
            }
        }
        std::shared_ptr<uint8_t> output_buffer;
        for (const auto &output_name : context->infer_model->get_output_names()) {
            size_t output_frame_size = context->infer_model->output(output_name)->get_frame_size();
            output_buffer = page_aligned_alloc(output_frame_size);
            bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
        }
        
        auto job = infer_model.run_async(bindings,[&](const AsyncInferCompletionInfo & info){
        auto bboxes = parse_nms_data(output_buffer.get(), 80);

        if (rgb == 0) {
            uint8_t* y_plane = map.data;
            uint8_t* u_plane = y_plane + context->width * context->height;
            uint8_t* v_plane = u_plane + (context->width * context->height) / 4;

            SDL_UpdateYUVTexture(
            context->texture,
            nullptr,
            y_plane, context->width,
            u_plane, context->width / 2,
            v_plane, context->width / 2
            );
            
        }else if (rgb == 1){

            SDL_UpdateTexture(
            context->texture,
            nullptr,                       // обновить весь кадр
            map.data,                      // данные из GstBuffer (RGB)
            context->width * 3                      // шаг по строке: 3 байта * ширина
            );
        }
        else {
            SDL_UpdateTexture(
                context->texture,
                nullptr,             // весь кадр
                map.data,            // данные из GstBuffer
                context->width * 2            // pitch: 2 байта на пиксель (YUY2 = 16bpp)
            );
        }
        SDL_RenderClear(context->renderer);
        SDL_RenderCopy(context->renderer, context->texture, nullptr, nullptr);
        draw_bounding_boxes(context->renderer, bboxes, context->width, context->height);
        SDL_RenderPresent(context->renderer);

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        frame_counter++;
            
    } ).expect("Failed to start async infer job");

    } else {
        if (rgb == 0) {
            uint8_t* y_plane = map.data;
            uint8_t* u_plane = y_plane + context->width * context->height;
            uint8_t* v_plane = u_plane + (context->width * context->height) / 4;

            SDL_UpdateYUVTexture(
            context->texture,
            nullptr,
            y_plane, context->width,
            u_plane, context->width / 2,
            v_plane, context->width / 2
            );
            
        }else if (rgb == 1){

            SDL_UpdateTexture(
            context->texture,
            nullptr,                       // обновить весь кадр
            map.data,                      // данные из GstBuffer (RGB)
            context->width * 3                      // шаг по строке: 3 байта * ширина
            );
        }
        else {
            SDL_UpdateTexture(
                context->texture,
                nullptr,             // весь кадр
                map.data,            // данные из GstBuffer
                context->width * 2            // pitch: 2 байта на пиксель (YUY2 = 16bpp)
            );
        }
        SDL_RenderClear(context->renderer);
        SDL_RenderCopy(context->renderer, context->texture, nullptr, nullptr);
        //draw_bounding_boxes(context->renderer, bboxes, context->width, context->height);
        SDL_RenderPresent(context->renderer);

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        frame_counter++;
    }
    // ----------- SDL DRAW -------------------
    // SDL_UpdateYUVTexture(context->texture, nullptr,
    //                      map.data, context->width,
    //                      map.data + context->width * context->height, context->width / 2,
    //                      map.data + context->width * context->height * 5 / 4, context->width / 2);
    
    return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
    std::string hef_path = argv[1];
    infer = std::stoi(argv[2]);
    rgb = std::stoi(argv[3]);
    WINDOW_HEIGHT = std::stoi(argv[4]);
    auto vdevice = VDevice::create().expect("Failed to create vdevice");
    auto infer_model = vdevice->create_infer_model(hef_path).expect("Failed to create infer model");
    auto configured_infer_model = infer_model->configure().expect("Failed to configure model");

    gst_init(&argc, &argv);
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow("GStreamer + SDL2",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *texture;
    if (rgb == 0) {
    texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_IYUV,  //SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STREAMING,
        WINDOW_WIDTH, WINDOW_HEIGHT);
    }else if (rgb == 1){
        texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_RGB24,  //SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STREAMING,
        WINDOW_WIDTH, WINDOW_HEIGHT);
    }else {
        texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_YUY2,  
        SDL_TEXTUREACCESS_STREAMING,
        WINDOW_WIDTH, WINDOW_HEIGHT);
    }
    void *userdata[2] = { renderer, texture };
    std::string pipeline_str;
    if (rgb == 0) {
        //pipeline_str =
        //"v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! videoscale ! video/x-raw,format=YUY2,width=640,height=640 ! appsink name=sink";
            //"v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! appsink name=sink";
            //"filesrc location=/home/user/showInference/test/build/" + source_path + " ! decodebin ! video/x-raw,format=I420 ! appsink name=sink";
    }else if (rgb == 1) {
        pipeline_str =
        //"v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=640 ! appsink name=sink";
        "v4l2src device=/dev/video0 ! image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegparse ! mppjpegdec ! "
        "rgaconvert ! video/x-raw,format=RGB,width=640,height=640 ! appsink name=sink";
    }else {
        pipeline_str =
        //"v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! videoscale ! video/x-raw,format=YUY2,width=640,height=640 ! appsink name=sink";
        "v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! appsink name=sink";
    }
    GError *error = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline) {
        std::cerr << "Failed to create pipeline: " << error->message << std::endl;
        return -1;
    }

    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    YourContextStruct context = {
    renderer,
    texture,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    configured_infer_model,
    infer_model
    };
    g_object_set(appsink, "emit-signals", TRUE, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), &context);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    SDL_Event event;
    
    bool quit = false;
    auto start = std::chrono::high_resolution_clock::now();
    while (!quit) {
       // debug("Cycle endered");
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                quit = true;
        }
        SDL_Delay(10);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Average FPS: " << frame_counter / (duration.count() / 1000) << std::endl;


    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
