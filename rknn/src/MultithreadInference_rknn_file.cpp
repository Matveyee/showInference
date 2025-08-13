#include "utils.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include "opencv2/opencv.hpp"
#include "rga/RgaUtils.h"
#include "rga/im2d.h"
#include "rga/rga.h"
#include "rknn_api.h"
#include <chrono>
#include <mutex>

#define ALIGN 8
typedef std::chrono::duration<double, std::milli> dur;

rknn_context ctx;
rknn_tensor_attr* input_attrs_r;
rknn_tensor_attr* output_attrs_r;;
rknn_input_output_num io_num;
int processed_index = 0;
int captured_index = 0;
int img_input_height = 0;
int img_input_width  = 0;
int channel          = 0;
int model_in_height  = 0;
int model_in_width   = 0;
int req_channel      = 0;
int wstride;
int hstride;
int FRAMES_COUNT;
int FPS;
std::vector<double> times;
std::mutex queue_mutex;
class RknnQueue {
    public:
        std::vector<rknn_tensor_mem*> arr;
        int size;

        RknnQueue(): size(0) {}

        void push(cv::VideoCapture cap) {
            // Init rga
            rga_buffer_t src;
            rga_buffer_t dst;
            im_rect      src_rect;
            im_rect      dst_rect;
            memset(&src_rect, 0, sizeof(src_rect));
            memset(&dst_rect, 0, sizeof(dst_rect));
            memset(&src, 0, sizeof(src));
            memset(&dst, 0, sizeof(dst));
            cv::Mat frame;
            cap >> frame;
            
            src = wrapbuffer_virtualaddr((void*)frame.data, 640, 480, RK_FORMAT_BGR_888); // wstride, hstride,
            auto buf = rknn_create_mem(ctx, input_attrs_r[0].size_with_stride);

            dst         = wrapbuffer_fd_t(buf->fd, model_in_width, model_in_height, wstride, hstride, RK_FORMAT_RGB_888); // wstride, hstride,
            IM_STATUS STATUS = imresize(src, dst);
            queue_mutex.lock();
            arr.push_back( buf );
            queue_mutex.unlock();
            size++;
        }
        
        rknn_tensor_mem* read() {
            return arr.front();
        }
        void pop() {
            rknn_destroy_mem(ctx, arr.front());
            queue_mutex.lock();
            arr.erase(arr.begin());
            queue_mutex.unlock();
            size--;
        }


};
RknnQueue queue;

auto simple_now() {
    return std::chrono::high_resolution_clock::now();
}
void inference() {
    
    rknn_tensor_mem* output_mems[io_num.n_output];
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        int output_size = output_attrs_r[i].n_elems * sizeof(float);
        // default output type is depend on model, this require float32 to compute top5
        //output_attrs_r[i].type = RKNN_TENSOR_FLOAT32;
        output_mems[i]       = rknn_create_mem(ctx, output_size);
    }
        // Set input tensor memory
    while(queue.size == 0) {}
    int ret = rknn_set_io_mem(ctx, queue.read() , &input_attrs_r[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
    }
    
    // Set output tensor memory
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs_r[i]);
        if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        }
    }

   // printf("Begin perf ...\n");
 //   int64_t start_us  = getCurrentTimeUs();
    auto start = simple_now();
    ret               = rknn_run(ctx, NULL);
    auto end = simple_now();
    dur duration = end - start;
    times.push_back(duration.count());
    std::cout << "FPS = " << 1 / (duration.count() / 1000) << std::endl;
  //  int64_t elapse_us = getCurrentTimeUs() - start_us;
    if (ret < 0) {
        printf("rknn run error %d\n", ret);
    }
 //   printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", processed_index, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
    processed_index++;
    if (captured_index >= FRAMES_COUNT) {
        std::cout << "processing " << processed_index << std::endl;
    }
    queue.pop();
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        rknn_destroy_mem(ctx, output_mems[i]);
    }


}

void start_inference() { 
    for (int i = 0; i < FRAMES_COUNT; i++) {
        inference();
    }

}

int main(int argc, char* argv[]) {
    // Initializing rknn
    if (argc < 3) {
        printf("Usage:%s model_path input_path frames_cout fps\n", argv[0]);
        return -1;
    }

    char* model_path = argv[1];
    char* input_path = argv[2];
    FRAMES_COUNT = std::stoi(argv[3]);
    FPS = std::stoi(argv[4]);
    cv::VideoCapture cap(input_path);

    ctx = 0;
    

    int            model_len = 0;
    unsigned char* model     = load_model(model_path, &model_len);
    int            ret       = rknn_init(&ctx, model, model_len, 0 , NULL);
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    int ret1 = rknn_set_core_mask(ctx, core_mask);
    std::cout << "NPU connected: " << ret1 << std::endl;
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }



    // Get Model Input Output Info
    
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    input_attrs_r = input_attrs;
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
        printf("rknn_init error! ret=%d\n", ret);
        return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    switch (input_attrs[0].fmt) {
    case RKNN_TENSOR_NHWC:
        model_in_height = input_attrs[0].dims[1];
        model_in_width  = input_attrs[0].dims[2];
        req_channel     = input_attrs[0].dims[3];
        break;
    case RKNN_TENSOR_NCHW:
        model_in_height = input_attrs[0].dims[2];
        model_in_width  = input_attrs[0].dims[3];
        req_channel     = input_attrs[0].dims[1];
        break;
    default:
        printf("meet unsupported layout\n");
    }
    int wstride = model_in_width + (ALIGN - model_in_width % ALIGN) % ALIGN;
    int hstride = model_in_height;


    
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    output_attrs_r = output_attrs;
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
        }
        dump_tensor_attr(&output_attrs[i]);
    }
    
    // start capture
    for (int i = 0; i < FRAMES_COUNT; i++) {
        auto start = simple_now();
        queue.push(cap);
        auto end = simple_now();
        captured_index++;
        std::cout << processed_index << "; " << captured_index << std::endl;
        dur duration = end - start;
        std::this_thread::sleep_for(std::chrono::milliseconds(  (int) (  (1.0 / FPS) * 1000 - duration.count()) ));
    }
    std::thread inf(start_inference);
    inf.join();
    double sum = 0;
    for (int i = 0; i < times.size() ; i++) {
        sum += times[i];
    }
    std::cout << "Average inference FPS = " << FRAMES_COUNT/ (sum / 1000) << std::endl;
}