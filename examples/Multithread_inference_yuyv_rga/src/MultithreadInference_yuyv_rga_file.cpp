#include <iostream>
#include <vector>
#include <mutex>
#include <unistd.h> // для Unix систем
#include "hailo/hailort.hpp"
#include <chrono>
#include <bits/stdc++.h>
#include <cmath>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
}
#include <csignal>
#include <sys/mman.h>
#include "hailo/hailort.hpp"

#include "rga/RgaUtils.h"
#include "rga/im2d.h"
#include "rga/rga.h"

using namespace hailort;
//variables 

bool running = true;
void sigint_handler(int) { running = false; }
std::string HEF_FILE;
std::string SOURCE_PATH;
int FRAMES;
int FPS;
//cv::VideoCapture cap;
std::unique_ptr<hailort::VDevice> vdevice;
std::shared_ptr<hailort::InferModel> infer_model;
hailort::ConfiguredInferModel configured_infer_model;
std::vector<double> times;
std::mutex queue_mutex;
int captured_index = 0;
int processed_index = 0;

//
#if defined(__unix__)
#include <sys/mman.h>
#endif
void debug(std::string message) {
    std::cout << "DEBUG: " << message << std::endl;

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
struct NamedBbox {
    hailo_bbox_float32_t bbox;
    size_t class_id;
};

class PtrQueue {

public:

    PtrQueue() : size(0){
       // debug("Queue created");
    }


    std::vector<char*> arr;

    void push(char* ptr) {
    
        std::lock_guard<std::mutex> lock(queue_mutex);
        arr.push_back(ptr);

        size++;

    }
    int size;
    char* read() {
        return arr.front();
    }

    void pop_front() {
        free(arr.front());
        arr.erase(arr.begin());
        size--;
    }


};
void printBytes(uint8_t* data) {
    for(int i = 0; i < 20; i++) {
        std::cout << (int)data[i] << ", " ;
    }
    std::cout << std::endl;
}
PtrQueue queue;

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

void frameProc() {
    auto &infer_model1 = configured_infer_model;
    auto bindings = infer_model1.create_bindings().expect("Failed to create bindings");
    if (queue.size == 0) {
       // std::cout<< "waiting..." << std::endl;
    }
    while (queue.size == 0) {}
    //std::cout<< "ready" << std::endl;
    for (const auto &input_name : infer_model->get_input_names()) {
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
      //  printBytes(queue.read()->data);
        bindings.input(input_name)->set_buffer(MemoryView(queue.read(), input_frame_size));
    }

    std::shared_ptr<uint8_t> output_buffer;
    for (const auto &output_name : infer_model->get_output_names()) {
        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
        output_buffer = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
    }
    //debug("before inference");
    auto start = std::chrono::high_resolution_clock::now();
    auto job = infer_model1.run_async(bindings,[&](const AsyncInferCompletionInfo & info){
       //debug("Frame processed");
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back( duration.count() / 1000);
        processed_index++;
       // auto bboxes = parse_nms_data(output_buffer.get(), 80);
       // std::cout << "Frame processed" << std::endl;
        std::unique_lock<std::mutex> lock(queue_mutex);
        //debug("Entering pop");
        queue.pop_front();
        lock.unlock();
    }).expect("Failed to start async infer job");

}




 

void inference() {
    for(int i = 0; i < FRAMES; i++) {
        frameProc();
        
    }
}

int main(int argc, char* argv[]) {

    if (argc == 0) {
        std::cout << "Usage: " << std::endl;
        std::cout << argv[0] << " {path to HEF file} {path to avi yuyv 480x640 file} {count of frames to process} {input FPS}" << std::endl;
        return 0;
    }
    HEF_FILE = argv[1];
    if (HEF_FILE == "--help") {
        std::cout << "Usage: " << std::endl;
        std::cout << argv[0] << " {path to HEF file} {path to avi yuyv 480x640 file} {count of frames to process} {input FPS}" << std::endl;
        return 0;
    }
    SOURCE_PATH = argv[2];
    FRAMES = std::stoi(argv[3]);
    FPS = std::stoi(argv[4]);
    

    vdevice = VDevice::create().expect("Failed create vdevice");
    infer_model = vdevice->create_infer_model(HEF_FILE).expect("Failed to create infer model");
    configured_infer_model = infer_model->configure().expect("Failed to create configured infer model");

//    av_register_all();

    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, SOURCE_PATH.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Failed to open file\n";
        return 1;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "Failed to get stream info\n";
        return 1;
    }

    // Find the video stream
    int video_stream_idx = -1;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }

    if (video_stream_idx == -1) {
        std::cerr << "No video stream found\n";
        return 1;
    }

    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_idx]->codecpar;
    if (codecpar->format != AV_PIX_FMT_YUYV422) {
        std::cerr << "Warning: Expected YUYV422, but got format = " << codecpar->format << std::endl;
    }

    int width = codecpar->width;
    int height = codecpar->height;
    size_t frame_size = av_image_get_buffer_size((AVPixelFormat)codecpar->format, width, height, 1);

    std::cout << "[info] Frame size: " << frame_size << " bytes (" << width << "x" << height << ", YUYV)\n";

    
    signal(SIGINT, sigint_handler);
    int i = 0;
    std::thread infer(inference);
    auto gen_start = std::chrono::high_resolution_clock::now();
    while (running && i < FRAMES) {
        auto start = std::chrono::high_resolution_clock::now();
        AVPacket* pkt = av_packet_alloc();
        av_read_frame(fmt_ctx, pkt);
        rga_buffer_t src;
        rga_buffer_t dst;
        rga_buffer_t dst1;
        im_rect      src_rect;
        im_rect      dst_rect;
        rga_buffer_handle_t src_handle;
        rga_buffer_handle_t dst_handle;
        rga_buffer_handle_t dst_handle1;
        memset(&src_rect, 0, sizeof(src_rect));
        memset(&dst_rect, 0, sizeof(dst_rect));
        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));
    //    char* src_buf = (char*)malloc(480*640*get_bpp_from_format(RK_FORMAT_YVYU_422));
        char* dst_buf = (char*)malloc(640*640*get_bpp_from_format(RK_FORMAT_YUYV_422));
        char* dst_buf1 = (char*)malloc(640*640*get_bpp_from_format(RK_FORMAT_RGB_888));
        src = wrapbuffer_virtualaddr(pkt->data, 640, 480, RK_FORMAT_YUYV_422);
        dst = wrapbuffer_virtualaddr(dst_buf, 640, 640, RK_FORMAT_YUYV_422);
        dst1 = wrapbuffer_virtualaddr(dst_buf1, 640, 640, RK_FORMAT_RGB_888);
        // src = wrapbuffer_handle(src_handle, 640, 480, RK_FORMAT_YUYV_422);
        // dst = wrapbuffer_handle(dst_handle, 640, 640, RK_FORMAT_YUYV_422);
        // dst1 = wrapbuffer_handle(dst_handle1, 640, 640, RK_FORMAT_RGB_888);

        IM_STATUS STATUS = imresize(src, dst);

        IM_STATUS STATUS2 = imcvtcolor(dst, dst1, RK_FORMAT_YUYV_422, RK_FORMAT_RGB_888);
        free(dst_buf);

        queue.push(dst_buf1);

        
        captured_index++;
 //       }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
       // captured_index++;
        if (captured_index % 10 == 0) {
            std::cout << processed_index << ";" << captured_index << std::endl;
        }
        
       // std::cout << "Time = " <<(int)(1.0/FPS * 1000) - (int)duration.count() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds( (int)(1.0/FPS * 1000) - (int)duration.count() ));
        i++;
    }
   // STOP_CAP = 0;

    infer.join();

    

    auto gen_end = std::chrono::high_resolution_clock::now();
   double sum = 0;
   double sum4 = 0;
    for (int i = 0; i < FRAMES; i++) {
	sum += times[i];
    }
   std::cout << "Average inference FPS = " <<1 /( sum / FRAMES) << std::endl;
    std::cout << "Average full FPS= " << FRAMES / (std::chrono::duration<double, std::milli>(gen_end - gen_start).count() / 1000 ) << std::endl;

}
