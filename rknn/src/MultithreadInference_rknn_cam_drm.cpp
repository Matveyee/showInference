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
#include "postprocess.hpp"
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <libdrm/drm_mode.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#define ALIGN 8

typedef std::chrono::duration<double, std::milli> dur;
struct drm_mode_fb_cmd fb = {};
int drm_fd;
uint8_t* map = nullptr;
uint32_t fb_id;
uint32_t handle;
uint32_t pitch;
uint64_t size;
drmModeCrtc *old_crtc = nullptr;
rknn_context ctx;
rknn_tensor_attr* input_attrs_r;
rknn_tensor_attr* output_attrs_r;;
rknn_input_output_num io_num;
int processed_index = 0;
int captured_index = 0;
int current_index = 0;
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
int RT_MODE;
std::vector<double> times;
std::vector<int> diff;
std::mutex queue_mutex;
std::mutex start_mutex;
std::mutex out_mutex;
int cond = 0;


void debug(std::string message) {
    std::cout << "DEBUG: " << message << std::endl;
}


class RknnQueue {
    public:
        std::vector<rknn_tensor_mem*> arr;
        int size;

        RknnQueue(): size(0) {}

        void push(cv::VideoCapture& cap) {
        //    debug("entered push");
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
           // uint8_t* conv = (uint8_t*)malloc(640 * 640 * 3);
            src = wrapbuffer_virtualaddr((void*)frame.data, 640, 480, RK_FORMAT_BGR_888); // wstride, hstride,
            //dst = wrapbuffer_virtualaddr((void*)conv, 640, 640, RK_FORMAT_RGB_888); // wstride, hstride,
            // std::cout << "SIZE " << rknn_app_ctx.input_attrs[0].size_with_stride  <<std::endl;
            // std::cout << "WIDTH " << rknn_app_ctx.model_width  <<std::endl;
            // std::cout << "HEIGHT " << rknn_app_ctx.model_height  <<std::endl;
            // std::cout << "w stride " << rknn_app_ctx.input_attrs[0].w_stride  <<std::endl;
            // std::cout << "h stride " << rknn_app_ctx.input_attrs[0].h_stride  <<std::endl;
            auto buf = rknn_create_mem(ctx, input_attrs_r[0].size_with_stride );

            dst         = wrapbuffer_fd_t(buf->fd, model_in_height, model_in_height, model_in_width + (8 - model_in_height % 8) % 8, model_in_height, RK_FORMAT_RGB_888); // wstride, hstride,
        //    debug("Before resize");
        //    std::cout << "Check " << imcheck(src,dst,src_rect, dst_rect) << std::endl;
            IM_STATUS STATUS = imresize(src, dst);
           // std::cout << "STATUS " << STATUS << std::endl;
        //    debug("After resize");
            queue_mutex.lock();
            arr.push_back( buf );
            queue_mutex.unlock();
            size++;
            
        }
        
        rknn_tensor_mem* read() {
            return arr.front();
        }
        void pop() {
            while( size == 0) {}
        //    debug("enter queue pop");
            queue_mutex.lock();
        //    debug("try dextroy mem");
           // std::cout << "Queue size " << size << std::endl;
            int stat = rknn_destroy_mem(ctx, arr.front());
        //    debug("mem destroyed");
           // free(arr.front());
            arr.erase(arr.begin());
            current_index--;
            size--;
            queue_mutex.unlock();
            
        }


};
RknnQueue queue;



class OutQueue {

    public:

        std::vector<rknn_output*> arr;
        int size;
        OutQueue() : size(0) {}

        void push(rknn_output* ptr) {

            out_mutex.lock();
        //    debug("oqueue try push");
            arr.push_back(ptr);
         //   debug("oqueue pushed");
            size++;
            out_mutex.unlock();
        }

        rknn_output* read() {
            return arr.front();
        }

        void pop() {
           // while (size == 0) {} 
            out_mutex.lock();
            rknn_outputs_release(ctx, io_num.n_output, read());
         //   std::cout << "RELEASED" << std::endl;
        //    rknn_outputs_release(ctx, io_num.n_output, arr.front());
        //    debug("try free oqueue");
          //  free(arr.front()->buf);
         //   debug(" free oqueue");
            arr.erase(arr.begin());
            size--;
            out_mutex.unlock();
        }


};
OutQueue oqueue;

auto simple_now() {
    return std::chrono::high_resolution_clock::now();
}
int max(int a, int b) {
    if (a > b) {
        return b;
    } else {
        return a;
    }
}
void draw_line(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch) {
    
    uint32_t pixel = (0xFF << 24) | (255 << 16) | (0 << 8) | 0;
    if (x == x1) {
        int i = y;
        while (i != y1) {
            if (y1 >= y) {
                i++;
            } else {
                i--;
            }

            ((uint32_t*)(map + max(i, 639) * pitch))[max(x, 639)] = pixel;
        }

    } else if (y == y1) {
        int i = x;
        while (i != x1) {
            if (x1 >= x ) {
                i++;
            } else {
                i--;
            }
            ((uint32_t*)(map + max(y, 639) * pitch))[max(i, 639)] = pixel;
        }

    }
    
}

void draw_rect(uint8_t* map, int x, int y, int x1, int y1) {
    draw_line(map,x,y,x1,y, pitch);
    draw_line(map,x,y1,x1,y1, pitch);
    draw_line(map,x,y,x,y1, pitch);
    draw_line(map,x1,y,x1,y1, pitch);
}
void printBytes(void* data) {
    for (int i = 0; i < 30; i++) {
        std::cout <<  (int) ((uint8_t*)data) [i] << " ";
    }
    std::cout << std::endl;
}
void draw() {
 //   debug("entered draw");
    while (oqueue.size == 0) {}
  //  debug("start draw");
    queue_mutex.lock();
    uint8_t* data = (uint8_t*)queue.read()->virt_addr;
    queue_mutex.unlock();

    for (int y = 0; y < 640; ++y) {
        for (int x = 0; x < 640; ++x) {
           int src_offset = y * 640 * 3  + x * 3;
            
            uint8_t r = data[src_offset + 0];
            uint8_t g = data[src_offset + 1];
            uint8_t b = data[src_offset + 2];

            uint32_t pixel = (0xFF << 24) | (r << 16) | (g << 8) | b;
          //  debug("before screen");
            ((uint32_t*)(map + y * pitch))[x] = pixel;
        }

    }
 //  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
 //  debug("SHOWN NORMAL");
//     debug("draw end");
     
     queue.pop();
//    while (oqueue.size == 0) {}
  // debug("try out read");
  //  debug("Start post process");
    rknn_output* outputs = oqueue.read();
  //  debug("outs read");
    
    object_detect_result_list od_results;
    letterbox_t letter_box;
    letter_box.x_pad = 0;
    letter_box.y_pad = 0;
    letter_box.scale = 1;
    rknn_app_context_t app_ctx;
    app_ctx.input_attrs = input_attrs_r;
    app_ctx.output_attrs = output_attrs_r;
    app_ctx.io_num = io_num;
    app_ctx.model_channel = req_channel;
    app_ctx.rknn_ctx = ctx;
    app_ctx.model_width = model_in_width;
    app_ctx.model_height = model_in_height;
    app_ctx.is_quant = 0;
   // printBytes(outputs[0].buf);
    post_process(&app_ctx, outputs, &letter_box, BOX_THRESH, NMS_THRESH, &od_results);


    for (int i = 0; i < 10; i++) {
     //   std::cout << "BOX " << od_results.results[i].box.left << ", " << od_results.results[i].box.bottom <<  ", " << od_results.results[i].box.right << ", " << od_results.results[i].box.top << std::endl;
        draw_rect(map, od_results.results[i].box.right,  od_results.results[i].box.top, od_results.results[i].box.left,  od_results.results[i].box.bottom);
    }
   // draw_rect(map, 100, 100, 200, 200);

   // debug("End post process");
    oqueue.pop();

}

void startDrawing() {
  //  debug("enter start draw");
    while (true)
    {
        draw();
        if (cond == 1) {
            break;
        }
    }
    
}
void inference(rknn_input* input) {
    
    // rknn_tensor_mem* output_mems[io_num.n_output];
    // for (uint32_t i = 0; i < io_num.n_output; ++i) {
    //     int output_size = output_attrs_r[i].n_elems * sizeof(float);
    //   //  std::cout << "Output size = " << output_size << std::endl;
    //     // default output type is depend on model, this require float32 to compute top5
    //     //output_attrs_r[i].type = RKNN_TENSOR_FLOAT32;
    //     output_mems[i]       = rknn_create_mem(ctx, output_size);
    // }
        // Set input tensor memory
    while(queue.size <= current_index + 1) {}
   // std::cout << "size " << queue.size << std::endl;
   // std::cout << "curr " << current_index << std::endl;
    if (processed_index != 0) {
        queue_mutex.lock();
   //     debug("try read");
        input->buf =  queue.arr[current_index]->virt_addr;
        int ret = rknn_inputs_set(ctx, io_num.n_input, input);
  //      debug("inf read");
        queue_mutex.unlock();
    }
    
  //  int ret = rknn_set_io_mem(ctx, queue.read() , &input_attrs_r[0]);
   // int ret = rknn_inputs_set(ctx, io_num.n_input, &input);
    // if (ret < 0) {
    //     printf("rknn_set_io_mem fail! ret=%d\n", ret);
    // }
    
    // Set output tensor memory
    // for (uint32_t i = 0; i < io_num.n_output; ++i) {
    //     ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs_r[i]);
    //     if (ret < 0) {
    //     printf("rknn_set_io_mem fail! ret=%d\n", ret);
    //     }
    // }
    

    auto start = simple_now();
  //  debug("inference");
    int ret               = rknn_run(ctx, NULL);
  //  std::cout << "RET " << ret <<std::endl;
  //  debug("inferenced");
    auto end = simple_now();
  //  //std::cout << "Outs " << io_num.n_output << std::endl;
//debug("try ouyputs get");
   // rknn_output outputs[io_num.n_output];
    rknn_output* outputs = (rknn_output*)calloc(io_num.n_output, sizeof(rknn_output));
  //  memset(outputs, 0, io_num.n_output * sizeof(rknn_output));
     //   std::cout << "n outputs " << io_num.n_output << std::endl;
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
        // int output_size = output_attrs_r[i].n_elems * sizeof(float);
            outputs[i].want_float  = 1;
            outputs[i].index       = i;
            // outputs[i].is_prealloc = 1;
            // outputs[i].buf = (void*)malloc(output_attrs_r[i].size * 1.85);
            // outputs[i].size = output_attrs_r[i].size;
        //    dump_tensor_attr(&output_attrs_r[i]);
        }
    rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    oqueue.push(outputs);

    

    // printBytes(outputs[0].buf);
  //  debug("outputs got");
    

    if (ret < 0) {
        printf("rknn run error %d\n", ret);
    }

    processed_index++;
    queue_mutex.lock();
    current_index++;
    queue_mutex.unlock();
//    if (captured_index >= FRAMES_COUNT) {
//       std::cout << "processing " << processed_index << std::endl;
//    }
    if (processed_index % 10 == 0 ) {
	std::cout << "processed : " << processed_index << std::endl;
    }
//    queue.pop();
    // for (uint32_t i = 0; i < io_num.n_output; ++i) {
    //     rknn_destroy_mem(ctx, output_mems[i]);
    // }
    
    dur duration = end - start;
    times.push_back(duration.count());

  //  debug("inference end");
    


}

void start_inference() { 
    start_mutex.lock();
    rknn_input input;
    input.index = 0;
    input.pass_through = 0;
    input.type = (rknn_tensor_type)RKNN_TENSOR_UINT8;
    input.fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;
    while(queue.size == 0) {}
    input.buf =  queue.read()->virt_addr;
    input.size = 640 * 640 * 3;
    int ret = rknn_inputs_set(ctx, io_num.n_input, &input);
    // std::cout << "RET INPUTS " << ret << std::endl;
    // rknn_output outputs[io_num.n_output];
    // for (uint32_t i = 0; i < io_num.n_output; ++i) {
    //    // int output_size = output_attrs_r[i].n_elems * sizeof(float);
    //     outputs[i].want_float  = 0;
    //     outputs[i].index       = i;
    //     outputs[i].is_prealloc = 1;
    //     outputs[i].buf = (void*)malloc(output_attrs_r[i].size );
    //     outputs[i].size = output_attrs_r[i].size;
    // }
    int i = 0;
    while(true) {

         inference(&input);
  //      debug("start_inference cycle");
        
        
	i++;
	if ( FRAMES_COUNT != 0 ) {
		if (i == FRAMES_COUNT) {break;}
	}
	if (cond == 1 ) {break;}
    }
    start_mutex.unlock();
    
}

void sign_handler(int signal) {
	cond = 1;
}
int main(int argc, char* argv[]) {
    // Initializing rknn
    if (argc < 3) {
        printf("Usage:%s model_path input_path frames_cout fps realtime_mode\n", argv[0]);
        return -1;
    }

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


    char* model_path = argv[1];
    char* input_path = argv[2];
    FRAMES_COUNT = std::stoi(argv[3]);
    FPS = std::stoi(argv[4]);
    RT_MODE = std::stoi(argv[5]);
    std::cout << "input path " << input_path << std::endl;
    cv::VideoCapture cap(input_path);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    ctx = 0;
    
    
    int            model_len = 0;
    unsigned char* model     = load_model(model_path, &model_len);
    int            ret       = rknn_init(&ctx, model, model_len, 0 , NULL);
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    int ret1 = rknn_set_core_mask(ctx, core_mask);
    std::cout << "NPU connected: " << ret << std::endl;
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
    // rknn_tensor_attr input_attrs[io_num.n_input];
    rknn_tensor_attr* input_attrs = (rknn_tensor_attr*)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
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
    wstride = model_in_width + (ALIGN - model_in_width % ALIGN) % ALIGN;
    hstride = model_in_height;


    
    printf("output tensors:\n");
    // rknn_tensor_attr output_attrs[io_num.n_output];
    rknn_tensor_attr* output_attrs = (rknn_tensor_attr*)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
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
    if (RT_MODE == 0) {
        start_mutex.lock();
        std::cout << "locked" << std::endl;
    }
    std::thread inf(start_inference);
    std::thread drw(startDrawing);
    //std::thread inf(start_inference);
    int i = 0;
    debug("before cycle");
    while (true) {
        auto start = simple_now();
        queue.push(cap);
        auto end = simple_now();
        debug("captured");
        captured_index++;
        // std::cout << processed_index << "; " << captured_index << std::endl;
        if (captured_index % 10 == 0) {
//            std::cout << "captured " << captured_index << std::endl;
        }
        if (RT_MODE == 3) {
            diff.push_back(captured_index - processed_index);
        }
        
        dur duration = end - start;
        if (RT_MODE != 0) { 
            std::this_thread::sleep_for(std::chrono::milliseconds(  (int) (  (1.0 / FPS) * 1000 - duration.count()) ));
        }
	i++;
	if ( FRAMES_COUNT != 0) {
		if ( i == FRAMES_COUNT ) {
			break;
		}
	}
	if (cond == 1) {
		break;
	}
    }
    if (RT_MODE == 0) {
        start_mutex.unlock();
    }
    
    inf.join();
    drw.join();
    double sum = 0;
    for (int i = 0; i < times.size() ; i++) {
        sum += times[i];
        if (RT_MODE == 3) {
            std::cout << i << ";" << diff[i] << std::endl;
        }else {
            std::cout << i << ";" << 1 / times[i] * 1000 << std::endl;
        }
        
    }
    std::cout << "Average inference FPS = " << FRAMES_COUNT/ (sum / 1000) << std::endl;
}
