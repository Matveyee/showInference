#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <string>

#include <unistd.h>
#include <sys/mman.h>

#include "/home/user/hailort-4.23.0/hailort/libhailort/include/hailo/hailort.hpp"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using namespace hailort;

static bool running = true;
void sigint_handler(int) { running = false; }

std::string HEF_FILE;
std::string SOURCE_PATH;
int FRAMES = 0;
int FPS = 0;
int REALTIME = 0;

std::unique_ptr<hailort::VDevice> vdevice;
std::shared_ptr<hailort::InferModel> infer_model;
hailort::ConfiguredInferModel configured_infer_model;

std::vector<double> times;
std::vector<int> diff_counts;

int captured_index = 0;
int processed_index = 0;

static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size)
{
#if defined(__unix__)
    void *addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (addr == MAP_FAILED) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr),
        [size](void *p){ munmap(p, size); });
#else
    void *addr = aligned_alloc(4096, ((size + 4095) / 4096) * 4096);
    if (!addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), free);
#endif
}

// ---------- Потокобезопасная очередь ----------
class FrameQueue {
public:
    void push(AVFrame *f) {
        std::lock_guard<std::mutex> lk(m_);
        q_.push_back(f);
        cv_.notify_one();
    }
    AVFrame* pop() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return !q_.empty() || !running; });
        if (q_.empty()) return nullptr;
        AVFrame *f = q_.front();
        q_.erase(q_.begin());
        return f;
    }
    size_t size() const {
        std::lock_guard<std::mutex> lk(m_);
        return q_.size();
    }
private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::vector<AVFrame*> q_;
};

FrameQueue frame_queue;

// ---------- Инференс ----------
void frameProcOnce() {
    auto &infer_model1 = configured_infer_model;
    auto bindings = infer_model1.create_bindings().expect("Failed to create bindings");

    AVFrame *rgb_frame = frame_queue.pop();
    if (!rgb_frame) return;

    for (const auto &input_name : infer_model->get_input_names()) {
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
        bindings.input(input_name)->set_buffer(MemoryView(rgb_frame->data[0], input_frame_size));
    }

    std::shared_ptr<uint8_t> output_buffer;
    for (const auto &output_name : infer_model->get_output_names()) {
        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
        output_buffer = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
    }

    auto start = std::chrono::high_resolution_clock::now();
    infer_model1.run_async(bindings, [start](const AsyncInferCompletionInfo & info){
        (void)info;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d = end - start;
        times.push_back(d.count());
        processed_index++;
        if (processed_index % 5 == 0) {
          //  std::cout << "[INFER] Processed: " << processed_index << std::endl;
        }
    }).expect("Failed to start async infer job");

    av_freep(&rgb_frame->data[0]);
    av_frame_free(&rgb_frame);
}

void inference() {
    std::cout << "[THREAD] Inference started" << std::endl;
    for (int i = 0; i < FRAMES && running; i++) {
        frameProcOnce();
    }
    std::cout << "[THREAD] Inference finished" << std::endl;
}

// ---------- MAIN ----------
int main(int argc, char* argv[]) {
    std::cout << "[START] Program init" << std::endl;
    if (argc < 6) {
        std::cout << "Usage: " << argv[0]
                  << " {hef} {input.mp4} {frames} {fps} {REALTIME 1/0}\n";
        return 0;
    }

    HEF_FILE = argv[1];
    SOURCE_PATH = argv[2];
    FRAMES = std::stoi(argv[3]);
    FPS = std::stoi(argv[4]);
    REALTIME = std::stoi(argv[5]);
    signal(SIGINT, sigint_handler);

    std::cout << "[INFO] HEF=" << HEF_FILE << " VIDEO=" << SOURCE_PATH
              << " FRAMES=" << FRAMES << " FPS=" << FPS << " REALTIME=" << REALTIME << std::endl;

    // Hailo init
    std::cout << "[STAGE] Init HailoRT" << std::endl;
    vdevice = VDevice::create().expect("Failed create vdevice");
    infer_model = vdevice->create_infer_model(HEF_FILE).expect("Failed to create infer model");
    configured_infer_model = infer_model->configure().expect("Failed to create configured infer model");

    // FFmpeg init
    std::cout << "[STAGE] Open video" << std::endl;
    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, SOURCE_PATH.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "[ERR] Failed to open file\n";
        return 1;
    }
    avformat_find_stream_info(fmt_ctx, nullptr);

    int video_stream_idx = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = (int)i;
            break;
        }
    }
    if (video_stream_idx == -1) {
        std::cerr << "[ERR] No video stream found\n";
        return 1;
    }

    std::cout << "[INFO] Video stream index: " << video_stream_idx << std::endl;

    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder_by_name("h264");
    if (!codec) codec = avcodec_find_decoder(codecpar->codec_id);
    std::cout << "[INFO] Decoder: " << codec->name << std::endl;

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codecpar);
    avcodec_open2(codec_ctx, codec, nullptr);
    std::cout << "[INFO] Codec opened, pix_fmt=" << av_get_pix_fmt_name(codec_ctx->pix_fmt)
              << " " << codec_ctx->width << "x" << codec_ctx->height << std::endl;

    AVBSFContext *bsf_ctx = nullptr;
    const AVBitStreamFilter *filter = av_bsf_get_by_name("h264_mp4toannexb");
    av_bsf_alloc(filter, &bsf_ctx);
    avcodec_parameters_copy(bsf_ctx->par_in, codecpar);
    av_bsf_init(bsf_ctx);
    std::cout << "[INFO] Bitstream filter: " << bsf_ctx->filter->name << std::endl;

    const int DST_W = 640, DST_H = 640;
    SwsContext* sws_ctx = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
                                         DST_W, DST_H, AV_PIX_FMT_BGR24,
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
    std::cout << "[STAGE] Swscale init OK" << std::endl;

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    auto gen_start = std::chrono::high_resolution_clock::now();
    
    if (REALTIME == 1) {
        std::thread infer_thr;
        infer_thr = std::thread(inference);
    

        std::cout << "[LOOP] Capture start" << std::endl;
        int i = 0;
        while (running && i < FRAMES) {
            if (av_read_frame(fmt_ctx, pkt) < 0) {
                std::cout << "[INFO] EOF" << std::endl;
                break;
            }
            auto start = std::chrono::high_resolution_clock::now();

            if (pkt->stream_index != video_stream_idx) {
                av_packet_unref(pkt);
                continue;
            }

            av_bsf_send_packet(bsf_ctx, pkt);
            while (av_bsf_receive_packet(bsf_ctx, pkt) == 0) {
                int ret = avcodec_send_packet(codec_ctx, pkt);
                av_packet_unref(pkt);
                if (ret < 0 && ret != AVERROR(EAGAIN)) {
                    std::cerr << "[WARN] send_packet err=" << ret << std::endl;
                    continue;
                }

                while (true) {
                    ret = avcodec_receive_frame(codec_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                        break;
                    if (ret < 0) {
                        std::cerr << "[ERR] receive_frame=" << ret << std::endl;
                        break;
                    }

                    if (frame->data[0]) {
                        AVFrame *rgb_frame = av_frame_alloc();
                        int rgb_buf_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, DST_W, DST_H, 1);
                        uint8_t *rgb_buf = (uint8_t*)av_malloc(rgb_buf_size);
                        av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, rgb_buf,
                                            AV_PIX_FMT_BGR24, DST_W, DST_H, 1);

                        sws_scale(sws_ctx, frame->data, frame->linesize, 0, codec_ctx->height,
                                rgb_frame->data, rgb_frame->linesize);
                        
                        auto end = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> duration = end - start;
                        std::this_thread::sleep_for(std::chrono::milliseconds((int)(1.0 / FPS * 1000) - (int)duration.count()));
                        frame_queue.push(rgb_frame);
                        
                        captured_index++;
                        i++;

                        if (i % 20 == 0)
                            std::cout << "[FRAME] status:" << captured_index << " ; " << processed_index << " queue=" << frame_queue.size() << std::endl;
                    }
                    av_frame_unref(frame);
                    if (i >= FRAMES) break;
                }
            }
        }
        infer_thr.join();
    }
    if (REALTIME == 0) {
        std::cout << "[LOOP] Capture start" << std::endl;
        int i = 0;
        while (running && i < FRAMES) {
            if (av_read_frame(fmt_ctx, pkt) < 0) {
                std::cout << "[INFO] EOF" << std::endl;
                break;
            }

            if (pkt->stream_index != video_stream_idx) {
                av_packet_unref(pkt);
                continue;
            }

            av_bsf_send_packet(bsf_ctx, pkt);
            while (av_bsf_receive_packet(bsf_ctx, pkt) == 0) {
                int ret = avcodec_send_packet(codec_ctx, pkt);
                av_packet_unref(pkt);
                if (ret < 0 && ret != AVERROR(EAGAIN)) {
                    std::cerr << "[WARN] send_packet err=" << ret << std::endl;
                    continue;
                }

                while (true) {
                    ret = avcodec_receive_frame(codec_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                        break;
                    if (ret < 0) {
                        std::cerr << "[ERR] receive_frame=" << ret << std::endl;
                        break;
                    }

                    if (frame->data[0]) {
                        AVFrame *rgb_frame = av_frame_alloc();
                        int rgb_buf_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, DST_W, DST_H, 1);
                        uint8_t *rgb_buf = (uint8_t*)av_malloc(rgb_buf_size);
                        av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, rgb_buf,
                                            AV_PIX_FMT_BGR24, DST_W, DST_H, 1);

                        sws_scale(sws_ctx, frame->data, frame->linesize, 0, codec_ctx->height,
                                rgb_frame->data, rgb_frame->linesize);
                        frame_queue.push(rgb_frame);
                        captured_index++;
                        i++;

                        if (i % 5 == 0)
                            std::cout << "[FRAME] status:" << captured_index << " ; " << processed_index << " queue=" << frame_queue.size() << std::endl;
                    }
                    av_frame_unref(frame);
                    if (i >= FRAMES) break;
                }
            }
        }
        std::thread infer_thr;
        infer_thr = std::thread(inference);
        infer_thr.join();
    }

    running = false;


    auto gen_end = std::chrono::high_resolution_clock::now();
    double sum = 0.0;
    for (double t : times) sum += t;
    if (!times.empty()) {
        std::cout << "[STATS] Inference FPS=" << (1.0 / (sum / times.size())) << std::endl;
    }
    double total_s = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count() / 1000.0;
    std::cout << "[STATS] Full FPS=" << (FRAMES / std::max(0.001, total_s)) << std::endl;

    av_frame_free(&frame);
    av_packet_free(&pkt);
    if (bsf_ctx) av_bsf_free(&bsf_ctx);
    if (sws_ctx) sws_freeContext(sws_ctx);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    std::cout << "[DONE] Finished cleanly" << std::endl;
}
