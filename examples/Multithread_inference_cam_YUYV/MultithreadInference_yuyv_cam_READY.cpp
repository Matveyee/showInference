#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <iostream>
#include <csignal>
#include <mutex>
#include <vector>
#include "hailo/hailort.hpp"
#include <bits/stdc++.h>

#define WIDTH 640
#define HEIGHT 480
#define BUFFER_COUNT 6
using namespace hailort;

void debug(std::string message) {
    std::cout << "DEBUG: " << message << std::endl;

}

//// global variables ////

int fd;
std::string HEF_FILE;
std::string DEVICE;
int FRAMES;
int FPS;
bool running = true;
std::mutex queue_mutex;
int processed_frames = 0;
int captured_frames = 0;
std::unique_ptr<hailort::VDevice> vdevice;
std::shared_ptr<hailort::InferModel> infer_model;
hailort::ConfiguredInferModel configured_infer_model;

std::vector<int> times;
////                  ////

//// types ////

struct buffer {
    void* start;
    size_t length;
};
struct buffer buffers[BUFFER_COUNT];
class BufQueue {

public:

    BufQueue() : size(0){}


    std::vector<v4l2_buffer*> arr;

    void push(v4l2_buffer* ptr) {
    
        std::lock_guard<std::mutex> lock(queue_mutex);
        arr.push_back(ptr);

        size++;

    }
    int size;
    v4l2_buffer* read() {
        return arr.front();
    }

    void pop_front() {
        if (ioctl(fd, VIDIOC_QBUF, arr.front()) < 0) {
            perror("VIDIOC_QBUF");
        }
        delete arr.front();
        arr.erase(arr.begin());
        size--;
    }


};
BufQueue queue;
////       ////

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



void sigint_handler(int) { running = false; }
void inference() {
    //debug("Inference entered");
    auto &infer_model1 = configured_infer_model;
    //debug("infer model created");
    auto bindings = infer_model1.create_bindings().expect("Failed to create bindings");
    //debug("bindings created");
    if (queue.size == 0) {
       // std::cout<< "waiting..." << std::endl;
    }
    while (queue.size == 0) {}
   // debug("Queue size waited");
    //std::cout<< "ready" << std::endl;
    for (const auto &input_name : infer_model->get_input_names()) {
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
      //  printBytes(queue.read()->data);
        bindings.input(input_name)->set_buffer(MemoryView((uint8_t*)buffers[queue.read()->index].start, input_frame_size));
    }

    std::shared_ptr<uint8_t> output_buffer;
    for (const auto &output_name : infer_model->get_output_names()) {
        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
        output_buffer = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
    }
  //  debug("before inference");
    auto start = std::chrono::high_resolution_clock::now();
    auto job = infer_model1.run_async(bindings,[&](const AsyncInferCompletionInfo & info){
   //    debug("Frame processed");
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back( duration.count() );
        processed_frames++;
        
       // auto bboxes = parse_nms_data(output_buffer.get(), 80);
       // std::cout << "Frame processed" << std::endl;
        std::unique_lock<std::mutex> lock(queue_mutex);
    //    debug("Entering pop");
        queue.pop_front();
        lock.unlock();
    }).expect("Failed to start async infer job");

}


void startInference() {
    while(running) {
        inference();
    }
}
int main(int argc, char* argv[]) {

    if (argc == 0) {
        std::cout << "Usage: " << std::endl;
        std::cout << argv[0] << " {path to HEF file} {/dev/video* file} {input FPS}" << std::endl;
        return 0;
    }
    HEF_FILE = argv[1];
    if (HEF_FILE == "--help") {
        std::cout << "Usage: " << std::endl;
        std::cout << argv[0] << " {path to HEF file} {/dev/video* file} {input FPS}" << std::endl;
        return 0;
    }
    vdevice = VDevice::create().expect("Failed create vdevice");
    infer_model = vdevice->create_infer_model(HEF_FILE).expect("Failed to create infer model");
    configured_infer_model = infer_model->configure().expect("Failed to create configured infer model");
    DEVICE = argv[2];
    FPS = std::stoi(argv[3]);

    fd = open(DEVICE.c_str(), O_RDWR);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    // Установим формат
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("VIDIOC_S_FMT");
        return 1;
    }

    // Запросим буферы
    struct v4l2_requestbuffers req = {0};
    req.count = BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("VIDIOC_REQBUFS");
        return 1;
    }

    // Отобразим буферы
    
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("VIDIOC_QUERYBUF");
            return 1;
        }

        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED) {
            perror("mmap");
            return 1;
        }

        // Подготовим к захвату
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
            return 1;
        }
    }

    // Запускаем поток
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        perror("VIDIOC_STREAMON");
        return 1;
    }
    signal(SIGINT, sigint_handler);
    auto start = std::chrono::high_resolution_clock::now();
    std::thread inf(startInference);
   // debug("Before running");
    while (running) {
      //  debug("Buf creating");
        struct v4l2_buffer* buf = new struct v4l2_buffer();
      //  debug("Buf created");
        memset(buf, 0, sizeof(struct v4l2_buffer));
        buf->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf->memory = V4L2_MEMORY_MMAP;
       // debug("Before dqbuf");
        if (ioctl(fd, VIDIOC_DQBUF, buf) < 0) {
            perror("VIDIOC_DQBUF");
            return 1;
        }
     //   debug("After dqbuf");

        // Доступ к кадру — zero-copy через mapped буфер
        // uint8_t* data = (uint8_t*)buffers[buf.index].start;
        queue.push(buf);
        captured_frames++;
        std::cout << processed_frames << ", " << captured_frames << std::endl;

        
    }
    
    // Остановим поток
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0) {
        perror("VIDIOC_STREAMOFF");
        return 1;
    }
    inf.join();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    int sum = 0;
    for (int i = 0; i < times.size() ; i++) {
        sum += times[i];
    }
    std::cout << "Average inference FPS = " << processed_frames / (sum / 1000) << std::endl;
    std::cout << "Average full FPS = " << processed_frames / (duration.count() / 1000) << std::endl;
    // Очистим память
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }

    close(fd);
    return 0;
}
