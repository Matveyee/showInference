#include <iostream>
#include <vector>
#include <mutex>
#include "opencv2/opencv.hpp"
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
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm/drm_mode.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include "../include/utils.hpp"
#include <rga/RgaApi.h>
#include <rga/RgaUtils.h>

using namespace hailort;
//variables 

void *rgaCtx = nullptr;
struct drm_mode_fb_cmd fb = {};
int drm_fd;
uint8_t* map = nullptr;
uint32_t fb_id;
uint32_t handle;
uint32_t pitch;
uint64_t size;
drmModeCrtc *old_crtc = nullptr;
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
int dma_fd;

//
#if defined(__unix__)
#include <sys/mman.h>
#endif
void debug(std::string message) {
    std::cout << "DEBUG: " << message << std::endl;

}

static inline uint8_t clamp(int val) {
    return std::max(0, std::min(255, val));
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
void printBytes(uint8_t* data, int count) {
    for( int i = 0; i < count; i++) {
        std::cout << std::hex << (int*)data[i] << ", ";
            
    }
    std::cout << std::endl;
}
#include <rga/RgaApi.h>
#include <rga/rga.h>

bool convert_yuyv_to_xrgb_rga(uint8_t* yuyv_ptr, uint8_t* drm_ptr, int width, int height) {
    debug("Printing yuyv");
    
    printBytes(yuyv_ptr, 10);

    rga_info_t src, dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));


    bo_t src_bo = {}, dst_bo = {};
    c_RkRgaGetAllocBufferCache(&src_bo, width, height, 32);   
    c_RkRgaGetAllocBufferCache(&dst_bo, width, height, 32); 

    c_RkRgaGetMmap(&src_bo);
    c_RkRgaGetMmap(&dst_bo);

    if (!src_bo.ptr) {
        std::cerr << "src_bo.ptr is null!" << std::endl;
    }
   // std::cout << "dst_bo.size = " << dst_bo.size << std::endl;
   // std::cout << "src_bo.size = " << src_bo.size << std::endl;
    // Копируй данные из AVPacket в src_bo.vir_addr
    memcpy(src_bo.ptr, yuyv_ptr, width * height * 2);
    debug("Printing src_bo.ptr");
    printBytes((uint8_t*)src_bo.ptr, 10);
    debug("Printing dst_bo.ptr");
    printBytes((uint8_t*)dst_bo.ptr, 10);

    src.virAddr = src_bo.ptr;
    src.mmuFlag = 1;
    src.format = RK_FORMAT_YUYV_422;
    src.bufferSize = width * height * 4;
    src.rect.xoffset = 0;
    src.rect.yoffset = 0;
    src.rect.width = width;
    src.rect.height = height;
    src.rect.wstride  = width;   
    src.rect.hstride  = height; 

 //   dst.fd = dma_fd;
    dst.virAddr = dst_bo.ptr;
    dst.mmuFlag = 1;
    dst.format = RK_FORMAT_RGBA_8888;
    dst.bufferSize = width * height * 4;
    dst.rect.xoffset = 0;
    dst.rect.yoffset = 0;
    dst.rect.width = width;
    dst.rect.height = height;
    dst.rect.wstride  = width;   
    dst.rect.hstride  = height;

    

   // printf("src.format = 0x%x, dst.format = 0x%x\n", src.format, dst.format);
   // printf("src.bufferSize = %d, dst.bufferSize = %d\n", src.bufferSize, dst.bufferSize);
   // printf("src.virAddr = %p\n", src.virAddr);
   // printf("dst.virAddr = %p\n", dst.virAddr);
    // Выполнить копирование с цветовой конверсией
    int ret = c_RkRgaBlit(&src, &dst, NULL);

    debug("Printing dst.virAddr");
    printBytes((uint8_t*)dst.virAddr, 10);
    debug("Printing src.virAddr");
    printBytes((uint8_t*)src.virAddr, 10);

    if (ret != 0) {
        fprintf(stderr, "RGA conversion failed: %d\n", ret);
        return false;
    }else {
       // std::cout << "RGA Succeed" << std::endl;
        
        memcpy(drm_ptr, dst.virAddr, width * height * 4);
        debug("Printing drmr");
        printBytes((uint8_t*)drm_ptr, 10);
        return true;
    }

    
}

class PtrQueue {

public:

    PtrQueue() : size(0){
       // debug("Queue created");
    }


    std::vector<AVPacket*> arr;

    void push(AVPacket* ptr) {
    
        std::lock_guard<std::mutex> lock(queue_mutex);
        arr.push_back(ptr);

        size++;

    }
    int size;
    AVPacket* read() {
        return arr.front();
    }

    void pop_front() {
        av_packet_unref( arr.front() );
        av_packet_free( &arr.front());
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


bool frameProc() {
    auto &infer_model1 = configured_infer_model;
    auto bindings = infer_model1.create_bindings().expect("Failed to create bindings");
    if (queue.size == 0) {
       // std::cout<< "waiting..." << std::endl;
    }
    while (queue.size == 0) {}
    //std::cout<< "ready" << std::endl;
    uint8_t* current = queue.read()->data;
    for (const auto &input_name : infer_model->get_input_names()) {
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
      //  printBytes(queue.read()->data);
        bindings.input(input_name)->set_buffer(MemoryView(current, input_frame_size));
    }

    std::shared_ptr<uint8_t> output_buffer;
    for (const auto &output_name : infer_model->get_output_names()) {
        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
        output_buffer = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
    }
    //debug("before inference");
    int min_width = std::min((int)fb.width, 640);
    int min_height = std::min((int)fb.height, 480);

    if (!convert_yuyv_to_xrgb_rga(current,map,640,480) ) {
        return false;
    }else {
        return true;
    }

    // auto start = std::chrono::high_resolution_clock::now();
    // auto job = infer_model1.run_async(bindings,[&](const AsyncInferCompletionInfo & info){
    //    //debug("Frame processed");
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double, std::milli> duration = end - start;
    //     times.push_back( duration.count() / 1000);
    //     processed_index++;
    //     auto bboxes = parse_nms_data(output_buffer.get(), 80);
    //     draw_bounding_boxes(map, bboxes, 640, 480, pitch);
    //     std::unique_lock<std::mutex> lock(queue_mutex);
    //     //debug("Entering pop");
    //     queue.pop_front();
    //     lock.unlock();
    // }).expect("Failed to start async infer job");
   // return true;

}
bool cond = true;
void inference() {
    for(int i = 0; i < FRAMES; i++) {
        if (!frameProc()) {
            cond = false;
            break;
        }
        if (i == 1) {
            break;
        }
        
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
    

    
    if (RgaInit(&rgaCtx) != 0) {
        std::cerr << "RGA init failed" << std::endl;
        return -1;
    }

    /// DRM 
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

    
    drmPrimeHandleToFD(drm_fd, handle, DRM_CLOEXEC | DRM_RDWR, &dma_fd);

    drmModeAddFB(drm_fd, fb.width, fb.height, fb.depth, fb.bpp, pitch, handle, &fb_id);

    drmModeSetCrtc(drm_fd, crtc_id, fb_id, 0, 0, &conn_id, 1, &mode);
    ///

    vdevice = VDevice::create().expect("Failed create vdevice");
    infer_model = vdevice->create_infer_model(HEF_FILE).expect("Failed to create infer model");
    configured_infer_model = infer_model->configure().expect("Failed to create configured infer model");

    av_register_all();

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
    while (running && i < 1) {
        if (!cond) {
            break;
        }
        auto start = std::chrono::high_resolution_clock::now();
        AVPacket* pkt = av_packet_alloc();
        av_read_frame(fmt_ctx, pkt);
        if (pkt->convergence_duration == video_stream_idx) {
            //debug("Before push");
            queue.push(pkt);
            captured_index++;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
       // captured_index++;
        std::cout << processed_index << ";" << captured_index << std::endl;
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
    RgaDeInit(rgaCtx);

}
