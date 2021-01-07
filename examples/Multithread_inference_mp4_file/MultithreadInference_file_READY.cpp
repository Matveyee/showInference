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
#include <libswscale/swscale.h>
}
#include <csignal>
#include <sys/mman.h>
#include "hailo/hailort.hpp"
// #include "../include/utils.hpp"

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

typedef struct duplet {
    AVPacket* pkt;
    AVFrame* frame;
} Duplet;

class PtrQueue {

public:

    PtrQueue() : size(0){
       // debug("Queue created");
    }


    std::vector<Duplet> arr;

    void push(Duplet ptr) {
    
        std::lock_guard<std::mutex> lock(queue_mutex);
        arr.push_back(ptr);

        size++;

    }
    int size;
    AVFrame* read() {
        return arr.front().frame;
    }

    void pop_front() {
        av_packet_unref( arr.front().pkt );
        av_packet_free( &arr.front().pkt );
        av_frame_free( &arr.front().frame );
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


void frameProc() {
   // debug("Frame proc entered");
    auto &infer_model1 = configured_infer_model;
   // debug("Infer model created");
    auto bindings = infer_model1.create_bindings().expect("Failed to create bindings");
    if (queue.size == 0) {
       // std::cout<< "waiting..." << std::endl;
    }
    while (queue.size == 0) {}
    //std::cout<< "ready" << std::endl;
   // debug("bindings created");
    for (const auto &input_name : infer_model->get_input_names()) {
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
      //  printBytes(queue.read()->data);
        bindings.input(input_name)->set_buffer(MemoryView(queue.read()->data, input_frame_size));
    }

    std::shared_ptr<uint8_t> output_buffer;
    for (const auto &output_name : infer_model->get_output_names()) {
        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
        output_buffer = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
    }
    //debug("before inference");
    auto start = std::chrono::high_resolution_clock::now();
   // debug("Before inference");
    auto job = infer_model1.run_async(bindings,[&](const AsyncInferCompletionInfo & info){
     //   debug("After inference");
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
     //   debug("Queue pop");
        lock.unlock();
    }).expect("Failed to start async infer job");

}




 

void inference() {
    for(int i = 0; i < FRAMES; i++) {
        if (!running) {
            break;
        }
        frameProc();
        
    }
}

int main(int argc, char* argv[]) {

    if (argc == 0) {
        std::cout << "Usage: " << std::endl;
        std::cout << argv[0] << " {path to HEF file} {path to mp4 480x640 file} {count of frames to process} {input FPS}" << std::endl;
        return 0;
    }
    HEF_FILE = argv[1];
    if (HEF_FILE == "--help") {
        std::cout << "Usage: " << std::endl;
        std::cout << argv[0] << " {path to HEF file} {path to mp4 480x640 file} {count of frames to process} {input FPS}" << std::endl;
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
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        std::cerr << "Unsupported codec\n";
        return 1;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codecpar);
    avcodec_open2(codec_ctx, codec, nullptr);

    

    int width = codec_ctx->width;
    int height = codec_ctx->height;

    SwsContext* sws_ctx = sws_getContext(width, height, codec_ctx->pix_fmt,
                                         width, height, AV_PIX_FMT_BGR24,
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);

    


    signal(SIGINT, sigint_handler);
    int i = 0;
    std::thread infer(inference);
    auto gen_start = std::chrono::high_resolution_clock::now();
    while (running && i < FRAMES) {
        auto start = std::chrono::high_resolution_clock::now();
        AVFrame* frame = av_frame_alloc();
        AVFrame* rgb_frame = av_frame_alloc();
        AVPacket* pkt = av_packet_alloc();
        int rgb_buf_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1);
        uint8_t* rgb_buf = (uint8_t*)av_malloc(rgb_buf_size * sizeof(uint8_t));
        av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, rgb_buf,
                            AV_PIX_FMT_BGR24, width, height, 1);
      //  debug("Packet allocated");
        av_read_frame(fmt_ctx, pkt);
      //  debug("Frame read");
        avcodec_send_packet(codec_ctx, pkt);
       // debug("Packet sent");
        avcodec_receive_frame(codec_ctx, frame);
       // debug("Packet received");
        sws_scale(sws_ctx, frame->data, frame->linesize, 0, height,
                              rgb_frame->data, rgb_frame->linesize);
        //debug("Frame scaled");
        av_frame_free(&frame);
 //       if (pkt->convergence_duration == video_stream_idx) {
            //debug("Before push");
            Duplet dup;
            dup.pkt = pkt;
            dup.frame = rgb_frame;
           // debug("Frame pushed");
            queue.push(dup);
            captured_index++;
  //      }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
       // captured_index++;
        std::cout << processed_index << ";" << captured_index << std::endl;
       // std::cout << "Time = " <<(int)(1.0/FPS * 1000) - (int)duration.count() << std::endl;
       std::cout << "Current FPS " << 1 / (duration.count() / 1000) << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds( (int)(1.0/FPS * 1000) - (int)duration.count() ));
        i++;
    }
   // STOP_CAP = 0;

    infer.join();

    

    auto gen_end = std::chrono::high_resolution_clock::now();
   double sum = 0;
   double sum4 = 0;
    for (int i = 0; i < times.size(); i++) {
	sum += times[i];
    }
   std::cout << "Average inference FPS = " <<1 /( sum / times.size()) << std::endl;
    std::cout << "Average full FPS= " << FRAMES / (std::chrono::duration<double, std::milli>(gen_end - gen_start).count() / 1000 ) << std::endl;

}
