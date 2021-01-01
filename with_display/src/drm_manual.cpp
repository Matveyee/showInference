extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
}

#include <iostream>
#include <csignal>
#include <sys/mman.h>
#include "hailo/hailort.hpp"
#include <chrono>
using namespace hailort;


bool running = true;
void sigint_handler(int) { running = false; }

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
void debug(std::string message) {
    std::cout << "DEBUG : " << message << std::endl;
}

int main(int argc, char* argv[]) {

    std::string hef_path = argv[1];
    auto vdevice = VDevice::create().expect("Failed to create vdevice");
    auto infer_model = vdevice->create_infer_model(hef_path).expect("Failed to create infer model");
    auto configured_infer_model = infer_model->configure().expect("Failed to configure model");


    if (argc < 2) {
        std::cerr << "Usage: ./read_yuyv_raw_avi input.avi\n";
        return 1;
    }

    const char* filename = argv[2];
    av_register_all();

    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, filename, nullptr, nullptr) < 0) {
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

    AVPacket pkt;
    signal(SIGINT, sigint_handler);
    int i = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (running && av_read_frame(fmt_ctx, &pkt) >= 0 && i < 200) {
        if (pkt.stream_index == video_stream_idx) {
            // 🚨 Здесь pkt.data содержит указатель на кадр YUYV
            uint8_t* ptr = pkt.data;
            size_t raw_size = pkt.size;

            //std::cout << "[frame] Got raw YUYV frame, size = " << raw_size << "\n";

            auto &infer_model1 = configured_infer_model;
                    auto bindings = infer_model1.create_bindings().expect("Failed to create bindings");

                   //std::shared_ptr<uint8_t> input_buffer;
                    for (const auto &input_name : infer_model->get_input_names()) {
                        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
                        //std::cout << "Input frame size =" << input_frame_size << std::endl;
                        //input_buffer = page_aligned_alloc(input_frame_size);
                        //memcpy(input_buffer.get(), ptr, input_frame_size);
                        bindings.input(input_name)->set_buffer(MemoryView(ptr, input_frame_size));
                    }

                    std::shared_ptr<uint8_t> output_buffer;
                    for (const auto &output_name : infer_model->get_output_names()) {
                        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
                        output_buffer = page_aligned_alloc(output_frame_size);
                        bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
                    }
                    //debug("before inference");
                    auto job = infer_model1.run_async(bindings,[&](const AsyncInferCompletionInfo & info){
                        auto bboxes = parse_nms_data(output_buffer.get(), 80);
                        std::cout << "Frame processed" << std::endl;
                    }).expect("Failed to start async infer job");

            // Пример: доступ к первым 4 пикселям (8 байт)
            // for (int i = 0; i < 8 && i < raw_size; ++i) {
            //     std::cout << std::hex << (int)raw_frame_ptr[i] << " ";
            // }
            // std::cout << std::dec << "\n";

            // Тут можешь отправить raw_frame_ptr в Hailo / DRM и т.д.
        }
        
        av_packet_unref(&pkt);
        i++;
    }
    auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Average Full FPS = " << 200 / (duration.count() / 1000) << std::endl;
    avformat_close_input(&fmt_ctx);
    std::cout << "Done.\n";
    return 0;
}