#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <SDL2/SDL.h>
#include <thread>
#include "utils.hpp"
#include "hailo/hailort.hpp"
#include <bits/stdc++.h>
#include <unistd.h> // для Unix сис
#if defined(__unix__)
#include <sys/mman.h>
#endif
using namespace hailort;

#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_RESET   "\033[0m"

std::string source;
std::string HEF_FILE;
std::string SOURCE_PATH;
int DELAY;
//cv::VideoCapture cap;
std::unique_ptr<hailort::VDevice> vdevice;
std::shared_ptr<hailort::InferModel> infer_model;
hailort::ConfiguredInferModel configured_infer_model;

int fps;
GstElement *pipeline;

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

typedef struct Duplet {
    GstBuffer* buf;
    GstSample* sample;
    int number;
    Duplet(GstBuffer* a, GstSample* b, int n) : buf(a), sample(b), number(n){}
} Dup;
std::vector<Dup> arr;
int acqIndex = -1;

void debug(std::string message) {
    std::cout << COLOR_YELLOW <<"DEBUG: "  <<message << COLOR_RESET  << std::endl;
}

int exited = 0;
int prev = 0;
void startGetting(GstAppSink* sink) {
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    while (acqIndex < 400) {
        std::cout <<"Frame counter : " <<acqIndex << std::endl;
        if (exited == 1) {
            break;
        }
        
    auto start = std::chrono::high_resolution_clock::now();
   // debug("startGetting: before sample");
    GstSample *sample = gst_app_sink_try_pull_sample(sink, GST_SECOND);
    if (!sample ) {
         //gst_sample_unref(sample);
         debug("!sample");
         if (prev == 1) {
            exited = 1;
            break;
         }
         prev = 1;
         //exited = 1;
         continue;
    }
    GstSample* sample_copy = gst_sample_ref(sample);
    //debug("startGetting: before buffer");
    GstBuffer *buf  = gst_sample_get_buffer(sample);
    if (!buf ) {
        gst_sample_unref(sample);
        debug("!buf");
        break;
    }
    GstBuffer *buf_copy = gst_buffer_ref(buf);
    Dup two(buf_copy, sample_copy, acqIndex);
    arr.push_back(two);
    acqIndex++;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    //std::this_thread::sleep_for(std::chrono::milliseconds((int)fps - (int)duration.count()));
    std::cout << "DELAY : " << (int)(1.0 / fps * 1000) << std::endl;
    auto next_frame_time = std::chrono::steady_clock::now() + std::chrono::milliseconds( (int)(1.0 / fps * 1000) - (int)duration.count());
    std::this_thread::sleep_until(next_frame_time); 
    //std::cout << "Acq index= " << acqIndex << std::endl;
    }
    exited = 1;

}
void printBytes(guint8* data) {
    for (int i = 0; i < 15; i++) {
        std::cout << (int)data[i];
    }
    std::cout << std::endl;
}
int main(int argc, char* argv[])
{   
///SDL init

    HEF_FILE = argv[1];
    SOURCE_PATH = argv[2];
    //RESIZED = std::stoi(argv[4]);
    vdevice = VDevice::create().expect("Failed create vdevice");
    infer_model = vdevice->create_infer_model(HEF_FILE).expect("Failed to create infer model");
    configured_infer_model = infer_model->configure().expect("Failed to create configured infer model");

    if (argc < 2) {
        std::cerr << "Укажите путь к видеофайлу, например:\n"
                  << argv[0] << " video.mp4\n";
        return 1;
    }


    int width = static_cast<int>(640);
    int height = static_cast<int>(640);

    // Инициализация SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "Ошибка SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Видео с SDL2",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width, height, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Ошибка создания окна: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        std::cerr << "Ошибка создания renderer: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Texture* texture = SDL_CreateTexture(renderer,
    SDL_PIXELFORMAT_IYUV,// SDL_PIXELFORMAT_YUY2, // или SDL_PIXELFORMAT_NV12
    SDL_TEXTUREACCESS_STREAMING,
    width, height);
    if (!texture) {
        std::cerr << "Ошибка создания texture: " << SDL_GetError() << "\n";
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Event event;
    bool running = true;
/// SDL

    gst_init(&argc, &argv);
    fps = std::stoi(argv[3]);
    
    std::cout << fps << std::endl;

    std::string pipeline_str =
    "filesrc location=/home/user/showInference/test/build/" + SOURCE_PATH + " !"
    //"filesrc location=/home/user/showInference/test/build/resized2.mp4 !"
    //"v4l2src device=/dev/video0 ! "
    "decodebin! "
    //"video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 !"
    "videorate ! video/x-raw,framerate="+ std::to_string(fps) + "/1 !"
    "appsink name=sink drop=false max-buffers=3 sync=true";

    GError *err = nullptr;
    pipeline = gst_parse_launch(pipeline_str.c_str(), &err);
    if (!pipeline) { std::cerr << err->message << '\n'; g_error_free(err); return -1; }

    auto sink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline), "sink"));
    //gst_debug_set_default_threshold(GST_LEVEL_WARNING); // или GST_LEVEL_INFO / GST_LEVEL_LOG
    int i = 0;
    auto startt = std::chrono::high_resolution_clock::now();
    std::thread getting(startGetting, sink);
    while (i < 400) {
       //i++;
        auto start = std::chrono::high_resolution_clock::now();
        debug("Cycle entered");

        debug("Before while");
        while (acqIndex < i + 1) {
            if (exited == 1) {
                break;
            }
        }
        if (exited == 1) {
                break;
        }
        debug("Before current");
        std::cout<< arr[i].number << std::endl;
        Dup Current = arr[i];
        debug("After current"); 

        GstBuffer *buf = Current.buf;
        GstSample *sample = Current.sample;

        GstMapInfo map;
        if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            i++;
            debug("!map");
            break;
        }
        debug("Frame acquired");
        //std::cout << (double)map.size  / width / height<< std::endl;
        if (!map.data || map.size < width*height*1.5) {
            i++;
            debug("map size");
            break;
        }
        std::cout << map.size << std::endl;
        
        

        auto pre_start = std::chrono::high_resolution_clock::now();
        printBytes(map.data);
    auto bindings = configured_infer_model.create_bindings().expect("Failed to create infer bindings");
    for (const auto &input_name : infer_model->get_input_names()) {
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
        // input_buffer = page_aligned_alloc(input_frame_size);
        //std::unique_lock<std::mutex> lock(queue_mutex);
        debug("Setting input buffer");
        auto status = bindings.input(input_name)->set_buffer(MemoryView(map.data, input_frame_size));
        debug("Buffer has been set");
        //lock.unlock();
        if (HAILO_SUCCESS != status) {
            throw hailort_error(status, "Failed to set infer input buffer");
        }

    }
    // memcpy( input_buffer.get(), data.data() , data.size());

    std::shared_ptr<uint8_t> output_buffer;
    for (const auto &output_name : infer_model->get_output_names()) {
        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
        output_buffer = page_aligned_alloc(output_frame_size);
        auto status = bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
        if (HAILO_SUCCESS != status) {
            throw hailort_error(status, "Failed to set infer output buffer");
        }

    }

    //auto start = std::chrono::high_resolution_clock::now();
    
    //auto job = configured_infer_model.run_async(bindings,[&buf, &sample, &height, &width, &map,&texture,&renderer,&pre_start,&output_buffer, &start](const AsyncInferCompletionInfo & info){
    std::cout << "Before inference : " << i << std::endl;
   // auto job = configured_infer_model.run_async(bindings,[&](const AsyncInferCompletionInfo & info){
//    printFirstBytes(output_buffer.get());
  //  auto end = std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double, std::milli> duration = end - start;
    //times.push_back(duration.count() / 1000);
  //  std::cout << "Inference time: " <<  ( duration.count() / 1000);
  //  auto preproc_start = std::chrono::high_resolution_clock::now();
   // auto bboxes = parse_nms_data(output_buffer.get(), 80);

    uint8_t* y_plane = map.data;
        uint8_t* u_plane = y_plane + width*height;
        uint8_t* v_plane = u_plane + width*height/4;
        SDL_UpdateYUVTexture(texture, nullptr,
        y_plane, width,
        u_plane, width / 2,
        v_plane, width / 2);

    
    // SDL_UpdateTexture(texture, nullptr, map.data, width * 3);
    debug("Texture updated");
    SDL_RenderClear(renderer);
    debug("Render cleared");
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    debug("Render copyied");
    //draw_bounding_boxes(renderer, bboxes, width, height);
    SDL_RenderPresent(renderer);
    gst_buffer_unmap(buf, &map);
    gst_buffer_unref(buf);
    gst_sample_unref(sample);
    
   // } ).expect("Failed to start async infer job");


        // Рисуем поверх (красный крест)
        // SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
        // SDL_RenderDrawLine(renderer, 0, 0, 639, 639);
        // SDL_RenderDrawLine(renderer, 639, 0, 0, 639);
        // SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);

        
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT ||
               (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
                running = false;
        }
        //auto next_frame_time = std::chrono::steady_clock::now() + std::chrono::milliseconds((int)fps - (int)duration.count());
        //std::this_thread::sleep_until(next_frame_time); 
        i++;
    }
    getting.join();
    auto endd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = endd - startt;
    std::cout << "Total FPS" <<400 / (duration.count() / 1000) << std::endl;
    exited = 1;
    
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return 0;
}
