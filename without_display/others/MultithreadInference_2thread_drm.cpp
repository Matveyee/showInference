#include <iostream>
#include <vector>
#include <mutex>
#include "opencv2/opencv.hpp"
#include <unistd.h> // для Unix систем
#include "hailo/hailort.hpp"
#include <chrono>
#include <bits/stdc++.h>
#include <cmath>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm/drm_mode.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
using namespace hailort;


#if defined(__unix__)
#include <sys/mman.h>
#endif
void debug(std::string message) {
    std::cout << "DEBUG: " << message << std::endl;

}

struct drm_mode_fb_cmd fb = {};
int drm_fd;
uint8_t* map = nullptr;
uint32_t fb_id;
uint32_t handle;
uint32_t pitch;
uint64_t size;
drmModeCrtc *old_crtc = nullptr;
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
std::string HEF_FILE;
std::string SOURCE_PATH;
int DELAY;
int FPS;
//cv::VideoCapture cap;
std::unique_ptr<hailort::VDevice> vdevice;
std::shared_ptr<hailort::InferModel> infer_model;
hailort::ConfiguredInferModel configured_infer_model;
std::vector<double> times;
std::vector<double> full_times;
std::vector<double> post_times;
std::vector<double> cap_times; 
std::atomic<int> STOP_CAP(1);
std::mutex queue_mutex;
int captured_index = 0;
int processed_index = 0;
int RESIZED;
class MatQueue {

public:

    MatQueue() : size(0){
        debug("Queue created");
    }


    std::vector<cv::Mat> arr;
    std::vector<cv::Mat> arr_original;

    void push(cv::VideoCapture* cap) {
        auto start = std::chrono::high_resolution_clock::now();
       // debug("Push entered ");
        cv::Mat original;
        //debug("Frame allocated");
        (*cap) >> original;
       // debug("Frame captured");
        cv::Mat frame;
        cv::resize(original, frame, cv::Size(640, 640));
        auto end = std::chrono::high_resolution_clock::now();
        cap_times.push_back( std::chrono::duration<double, std::milli>(end-start).count() / 1000 );
       // debug("Frame resized");
        std::lock_guard<std::mutex> lock(queue_mutex);
        arr.emplace_back( std::move(frame) );
        arr_original.emplace_back( std::move(original) );
        //debug("Frame emplaced");
        size++;

    }
    int size;
    cv::Mat& read() {
        return arr.front();
    }
    cv::Mat& read_orig() {
        return arr_original.front();
    }

    void pop_front() {
        arr_original.erase(arr_original.begin());
        arr.erase(arr.begin());
        size--;
    }


};
MatQueue queue;

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

void draw_line(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch) {
    uint32_t pixel = (0xFF << 24) | (255 << 16) | (0 << 8) | 0;
    if (x == x1) {
        for (int i = y; i <= y1; i++) {
            ((uint32_t*)(map + i * pitch))[x] = pixel;
        }
    } else if (y == y1) {
        for (int i = x; i <= x1; i++) {
            ((uint32_t*)(map + y * pitch))[i] = pixel;
        }
    }
    
}

void draw_rect(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch) {
    draw_line(map,x,y,x1,y, pitch);
    draw_line(map,x,y1,x1,y1, pitch);
    draw_line(map,x,y,x,y1, pitch);
    draw_line(map,x1,y,x1,y1, pitch);
}

void draw_bounding_boxes(uint8_t* map, const std::vector<NamedBbox>& bboxes, int width, int height, uint32_t pitch) {


    for (const auto& named_bbox : bboxes) {
        hailo_bbox_float32_t bbox = named_bbox.bbox;
        int x = static_cast<int>(bbox.x_min * width);
        int y = static_cast<int>(bbox.y_min * width);
        int x1 = x + static_cast<int>((bbox.x_max - bbox.x_min) * width);
        int y1 = y + static_cast<int>((bbox.y_max - bbox.y_min) * width);
//        std::cout << "x = " << x << ", y = " << y << ", x1 = " << x1 << ", y1 = " << y1 << std::endl;
        if (x1 > 0 && x > 0 && y > 0 && y1 > 0) {
            draw_rect(map, x , y, x1 ,y1, pitch);
        }
    }
}


void printFirstBytes(uint8_t* data) {
    for(int i = 0; i < 20; i++) {
        std::cout <<(int)data[i] << ", ";
    }
    std::cout << std::endl;
}

   
    void doGetNextFrame() {
        //debug("doGetNextFrame entered");


   

    //std::vector<uint8_t> input_buffer(frame.data, frame.data + frame.total() * frame.elemSize());
  //  auto buff_start = std::chrono::high_resolution_clock::now();
    // std::shared_ptr<uint8_t> input_buffer;

    while (queue.size == 0){
       // debug("In cycle");
    }
    
    auto pre_start = std::chrono::high_resolution_clock::now();
    auto bindings = configured_infer_model.create_bindings().expect("Failed to create infer bindings");
    for (const auto &input_name : infer_model->get_input_names()) {
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
        // input_buffer = page_aligned_alloc(input_frame_size);
        std::unique_lock<std::mutex> lock(queue_mutex);
       // debug("Setting input buffer");
        auto status = bindings.input(input_name)->set_buffer(MemoryView(queue.read().data, input_frame_size));
       // debug("Buffer has been set");
        lock.unlock();
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
   // auto buff_end = std::chrono::high_resolution_clock::now();
   // std::chrono::duration<double, std::milli> duration1 = buff_end - buff_start;
   // std::cout << "Buffers time: " <<  (duration1.count() / 1000) << ", ";
//    std::cout << "Running inference..." << std::endl;
    // Run the async infer job
    auto start = std::chrono::high_resolution_clock::now();
    auto job = configured_infer_model.run_async(bindings,[&pre_start,&output_buffer, &start](const AsyncInferCompletionInfo & info){
//    printFirstBytes(output_buffer.get());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    times.push_back(duration.count() / 1000);
    std::cout << "Inference time: " << duration.count() / 1000 << std::endl;
    // processed_index++;
    // std::unique_lock<std::mutex> lock(queue_mutex);
    // queue.pop_front();
    // lock.unlock();

    auto preproc_start = std::chrono::high_resolution_clock::now();
    auto bboxes = parse_nms_data(output_buffer.get(), 80);
    uchar* pic = queue.read().data;
    if (RESIZED == 1) {
        int min_width = std::min((int)fb.width, 640);
        int min_height = std::min((int)fb.height, 640);
        for (int y = 0; y < min_height; ++y) {
            for (int x = 0; x < min_width; ++x) {
                int src_offset = y * min_width * 3 + x * 3;
                uint8_t r = pic[src_offset + 0];
                uint8_t g = pic[src_offset + 1];
                uint8_t b = pic[src_offset + 2];

                uint32_t pixel = (0xFF << 24) | (r << 16) | (g << 8) | b;

                ((uint32_t*)(map + y * pitch))[x] = pixel;
            }

        }
        draw_bounding_boxes(map, bboxes, 640, 640, pitch);
     //   debug("Showing image");
        processed_index++;
        //std::cout << processed_index << ";" << captured_index << std::endl;
    }else {
        
        draw_bounding_boxes(queue.read_orig().data, bboxes, 640, 640, pitch);
     //   debug("Showing image");

        processed_index++;
        //std::cout << processed_index << ";" << captured_index << std::endl; 
    }
    
   // debug("Shown image");
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue.pop_front();
    lock.unlock();
    
    auto after_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration2 = after_end - preproc_start;
    std::chrono::duration<double, std::milli> duration3 = after_end - pre_start;
   // std::cout << ", Postprocess time: " <<  ( duration2.count() / 1000) <<std::endl;
   // std::cout << ", Full time: " <<  ( duration3.count() / 1000) <<std::endl;
    full_times.push_back(duration3.count() / 1000);
    post_times.push_back(duration2.count() / 1000);
    //cv::imshow("STREAM", original);
    //cv::waitKey(0);
    } ).expect("Failed to start async infer job");
    // auto status = job.wait(std::chrono::milliseconds(DELAY));
    // if (HAILO_SUCCESS != status) {
    //     throw hailort_error(status, "Failed to wait for infer to finish");
    // }else {
    //     std::cout << "Inference succeed" << std::endl;

    // }
    // auto bboxes = parse_nms_data(output_buffer.get(), 80);
    // draw_bounding_boxes(original, bboxes);
    // cv::imshow("STREAM", original);

    }

void capture(cv::VideoCapture* cap) {
    while (STOP_CAP == 1) {
     //   debug("Cycle entered");
        auto start = std::chrono::high_resolution_clock::now();
        queue.push(cap);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        captured_index++;
        std::cout << processed_index << ";" << captured_index << std::endl; 
        std::this_thread::sleep_for(std::chrono::milliseconds( (int)(1.0/FPS * 1000) - (int)duration.count() ));
        

    }

    
}
void inference() {
  //  std::cout << "delay : " << DELAY << std::endl;
    for(int i = 0; i < DELAY; i++) {
	    doGetNextFrame();
    }
}

int main(int argc, char* argv[]) {
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
    drmModeAddFB(drm_fd, fb.width, fb.height, fb.depth, fb.bpp, pitch, handle, &fb_id);

    drmModeSetCrtc(drm_fd, crtc_id, fb_id, 0, 0, &conn_id, 1, &mode);
    
    HEF_FILE = argv[1];
    SOURCE_PATH = argv[2];
    DELAY = std::stoi(argv[3]);
    RESIZED = std::stoi(argv[4]);
    FPS = std::stoi(argv[5]);

    vdevice = VDevice::create().expect("Failed create vdevice");
    infer_model = vdevice->create_infer_model(HEF_FILE).expect("Failed to create infer model");
    configured_infer_model = infer_model->configure().expect("Failed to create configured infer model");
    cv::VideoCapture cap1;
    cap1.open(SOURCE_PATH);
  //  debug("Camera started");
    std::thread camera(capture, &cap1);
    auto gen_start = std::chrono::high_resolution_clock::now();
  //  debug("Inference started");
    std::thread infer(inference);
    infer.join();
    //debug("Inference ended");
    STOP_CAP = 0;
    camera.join();
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
    //debug("Camera ended");
    auto gen_end = std::chrono::high_resolution_clock::now();
   double sum = 0;
  //  double sum1 = 0;
  //  double sum2 = 0;
   double sum4 = 0;
    for (int i = 0; i < DELAY; i++) {
	sum += times[i];
//	sum1 += full_times[i];
  //  sum2 += post_times[i];
    sum4 += cap_times[i];
    }
   // std::cout << "Average capture FPS = " << DELAY / sum4 << std::endl;
   std::cout << "Average inference time = " <<1 /( sum / DELAY) << std::endl;
  //  std::cout << "Average post process time = " << sum2 / DELAY << std::endl;
   // std::cout << "Average each FPS= " << DELAY / sum1 << std::endl;
    std::cout << "Average full FPS= " << DELAY / (std::chrono::duration<double, std::milli>(gen_end - gen_start).count() / 1000 ) << std::endl;
    cv::destroyAllWindows();

}
