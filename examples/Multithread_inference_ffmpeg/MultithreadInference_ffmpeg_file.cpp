// g++ -o mpp_rga_async main.cpp -I/usr/include/mpp -I/usr/include/rga -lrga -lmpp -lpthread -lavformat -lavcodec -lavutil -lswscale -lhailort -ldrm

#include <iostream>
#include <fstream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>
#include <cstring>
#include <csignal>
#include <chrono>
#include <cmath>

#include "/home/user/hailort-4.23.0/hailort/libhailort/include/hailo/hailort.hpp"

extern "C" {
// #include "rk_mpi.h"
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

using namespace hailort;

std::atomic<bool> running(true);
AVBufferRef *hw_device_ctx = nullptr;
std::shared_ptr<hailort::VDevice> vdevice;
std::shared_ptr<hailort::InferModel> infer_model;
hailort::ConfiguredInferModel configured_infer_model;
std::vector<double> times;
std::vector<double> times_cap;
std::queue<AVFrame*> frame_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
int processed = 0;
int captured = 0;

void debug(std::string message) {
    std::cout << "DEBUG : " << message << std::endl;
}

#if defined(__unix__)
#include <sys/mman.h>
#endif
static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size) {
    auto addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
}

// int init_hw_device() {
//     if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_RKMPP, nullptr, nullptr, 0) < 0) {
//         std::cerr << "Failed to create RKMPP hwdevice context" << std::endl;
//         return -1;
//     }
//     return 0;
// }

void inference_loop(int width, int height) {
    SwsContext* sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_NV12,
        640, 640, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    while (running) {
        AVFrame *frame = nullptr;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [] { return !frame_queue.empty() || !running; });
            if (!running) break;
            frame = frame_queue.front();
            frame_queue.pop();
        }

        auto bindings = configured_infer_model.create_bindings().expect("Failed to create bindings");

        std::shared_ptr<uint8_t> input_buf = page_aligned_alloc(640 * 640 * 3);
        uint8_t* dst_data[4];
        int dst_linesize[4];
        av_image_fill_arrays(dst_data, dst_linesize, input_buf.get(), AV_PIX_FMT_BGR24, 640, 640, 1);

        sws_scale(sws_ctx, frame->data, frame->linesize, 0, height, dst_data, dst_linesize);

        for (const auto &input_name : infer_model->get_input_names()) {
            bindings.input(input_name)->set_buffer(MemoryView(input_buf.get(), 640 * 640 * 3));
        }

        std::shared_ptr<uint8_t> output_buf;
        for (const auto &output_name : infer_model->get_output_names()) {
            size_t output_size = infer_model->output(output_name)->get_frame_size();
            output_buf = page_aligned_alloc(output_size);
            bindings.output(output_name)->set_buffer(MemoryView(output_buf.get(), output_size));
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto job = configured_infer_model.run_async(bindings, [&](const AsyncInferCompletionInfo & info){
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            times.push_back(duration.count() / 1000);
            //debug("Inference happened");
            processed++;
        }).expect("Failed async infer");

        av_frame_free(&frame);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <hef_file> <input.mp4> <FPS> <FRAMES COUNT>" << std::endl;
        return 1;
    }

    std::string hef_file = argv[1];
    std::string input_path = argv[2];
    int FPS = std::stoi(argv[3]);
    int COUNT = std::stoi(argv[4]);

    vdevice = VDevice::create().expect("Failed to create vdevice");
    infer_model = vdevice->create_infer_model(hef_file).expect("Failed to create infer model");
    configured_infer_model = infer_model->configure().expect("Failed to configure model");

    std::cout << "Hailo skip" << std::endl;
 
    // if (init_hw_device() != 0) return -1;

    AVFormatContext *fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, input_path.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Cannot open input file" << std::endl;
        return -1;
    }
    avformat_find_stream_info(fmt_ctx, nullptr);

    int video_stream_idx = -1;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }
    if (video_stream_idx == -1) {
        std::cerr << "Video stream not found" << std::endl;
        return -1;
    }

    AVCodecParameters *codecpar = fmt_ctx->streams[video_stream_idx]->codecpar;
    const AVCodec *decoder = avcodec_find_decoder(codecpar->codec_id);
    AVCodecContext *codec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(codec_ctx, codecpar);
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    avcodec_open2(codec_ctx, decoder, nullptr);

    int width = codec_ctx->width;
    int height = codec_ctx->height;

    std::thread infer_thread(inference_loop, width, height);

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    AVFrame *sw_frame = av_frame_alloc();
    int i = 0;
    auto g_start = std::chrono::high_resolution_clock::now();
    std::chrono::_V2::system_clock::time_point start = std::chrono::high_resolution_clock::now();
    while (av_read_frame(fmt_ctx, pkt) >= 0 && i < COUNT) {
        if (pkt->stream_index == video_stream_idx) {
            avcodec_send_packet(codec_ctx, pkt);
            while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                if (frame->format == AV_PIX_FMT_DRM_PRIME) {
                    av_hwframe_transfer_data(sw_frame, frame, 0);
                } else {
                    av_frame_ref(sw_frame, frame);
                }
                {
                    if (i != 0) {
                       auto end = std::chrono::high_resolution_clock::now();
                       std::chrono::duration<double, std::milli> duration = end - start;
                       std::this_thread::sleep_for(std::chrono::milliseconds( (int)(1.0/FPS * 1000) - (int)duration.count() ));
                    }
                    start = std::chrono::high_resolution_clock::now();
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    frame_queue.push(av_frame_clone(sw_frame));
                    
                    captured++;
                    i++;
                    std::cout << processed << ", " << captured << std::endl;
                }
                queue_cv.notify_one();
                av_frame_unref(sw_frame);
                av_frame_unref(frame);
            }
        }
        av_packet_unref(pkt);
    }

    running = false;
    queue_cv.notify_all();
    infer_thread.join();
    auto g_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration  = g_end - g_start;
    double sum = 0;
    for (const auto &t : times) sum += t;
    std::cout << "Average inference FPS: " << (times.empty() ? 0 : 1.0 / (sum / times.size())) << std::endl;
    std::cout << "Average full FPS: " << captured / (duration.count() / 1000) << std::endl;

    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    av_packet_free(&pkt);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    return 0;
}
