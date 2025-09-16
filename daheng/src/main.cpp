#include "../include/GxIAPI.h"
#include <vector>
#include <mutex>
#include "rga/RgaUtils.h"
#include "rga/im2d.h"
#include "rga/rga.h"
#include <bits/stdc++.h>
#include "hailo/hailort.hpp"
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <libdrm/drm_mode.h>
#include <sys/ioctl.h>
#include <fcntl.h>

#define H 640
#define W 640
using namespace hailort;

static volatile std::sig_atomic_t stop = 1;
GX_DEV_HANDLE hDevice = NULL;
bool running = true;
int drm_fd;
struct drm_mode_fb_cmd fb = {};
uint8_t* map = nullptr;
uint32_t fb_id;
uint32_t handle;
uint32_t pitch;
uint64_t size;
drmModeCrtc *old_crtc = nullptr;
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
std::mutex empty_mutex;
std::mutex queue_mutex1;
std::mutex empty_mutex1;
int captured_index = 0;
int processed_index = 0;
std::mutex main_mutex;


#if defined(__unix__)
#include <sys/mman.h>
#endif
void debug(std::string message) {
    std::cout << "DEBUG: " << message << std::endl;
}
void statu(std::string message, int status) {

    std::cout << "STATUS: " << message << ": " << status << std::endl;

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
class Vec {

    private:
        
        int x;
        int y;

    public:

        Vec(int p_x, int p_y) : x(p_x), y(p_y) {}
        Vec() {}

        void init(int p_x, int p_y) {
            x = p_x;
            y = p_y;
        }

        int getx() {
            return x;
        }

        int gety() {
            return y;
        }
        Vec& operator += (Vec& other) {
            x += other.getx();
            y += other.gety();
            return *this;
        }
        void print() {
            std::cout << "( " << x << "," << y << ")";
        }
       
};

Vec operator +(Vec& first, Vec& second) {

    Vec vec(first.getx() + second.getx(), first.gety() + second.gety());
    return vec;

}
class Projection {

    private:

        uint8_t* data;
        int offsetX;
        int offsetY;
        int sourceW;
        int sourceH;
        int w;
        int h;

        
        
    public:
        Projection() {}

        Projection(uint8_t* data_ptr, int offset_x, int offset_y, int source_w, int source_h, int width, int height) : data(data_ptr), offsetX(offset_x), offsetY(offset_y), w(width), h(height), sourceW(source_w), sourceH(source_h) {}

        uint8_t operator [](int index) {
            return data[(offsetY + index / w) * sourceW + offsetX + index % w];
        }
        void init(uint8_t* data_ptr , int offset_x, int offset_y, int source_w, int source_h, int width, int height) {
            data = data_ptr;
            offsetX = offset_x;
            offsetY = offset_y;
            w = width;
            h = height;
            sourceW = source_w;
            sourceH = source_h;
        }
        uint8_t get(int x, int y) {
            if (x >= w || y >= h) {
                return 0;
            } else {
                return (*this)[x + y * w];
            }
            
        }

        uint8_t get(Vec vec) {
            return get(vec.getx(), vec.gety());
        }

        int getx() {
            return offsetX;
        }
        
        int gety() {
            return offsetY;
        }
        
        int getW() {
            return w;
        }

        int getH() {
            return h;
        }


};


void specifyVectors(Vec& R, Vec& G1, Vec& G2, Vec& B, Vec r) {
    
    if ( (r.getx() % 2 == 0) && (r.gety() % 2 == 0)) {

        R.init(0,0);
        G1.init(1,0);
        G2.init(0,1);
        B.init(1,1);
    }else if ((r.getx() % 2 != 0) && (r.gety() % 2 == 0)) {

        G1.init(0,0);
        R.init(0,1);
        B.init(1,0);
        G2.init(1,1);
    }else if ((r.getx() % 2 == 0) && (r.gety() % 2 != 0)) {

        G2.init(0,0);
        B.init(1,0);
        R.init(0,1);
        G1.init(1,1);
    } else {

        B.init(0,0);
        G2.init(1,0);
        G1.init(0,1);
        R.init(1,1);
    }


}
uint8_t* toRGB(Projection input) {
//    debug("to rgb entered");
    uint8_t* rgbData = new uint8_t[ input.getW() * input.getH() * 3];
    Vec r;
    Vec r0;
    Vec R;
    Vec G1;
    Vec G2;
    Vec B;
    r.init(0,0);
    r0.init(input.getx(), input.gety());
   // debug("before cycle");
    for (int i = 0; i < (input.getW() * input.getH()); i++ ) {
        if(stop == 0) {
            break;
        }
        Vec d = r0 + r;
        specifyVectors(R, G1, G2, B, d);
        rgbData[i * 3 + 0] = input.get(r + R);
        rgbData[i * 3 + 1] = ( input.get(r + G1) + input.get(r + G2) ) / 2;
        rgbData[i * 3 + 2] = input.get(r + B);
        

        Vec delta;
        if ( (i + 1) % input.getW() == 0 ) {
            delta.init(1 - input.getW(), 1);
        } else {
            delta.init(1,0);
        }
        r += delta;
    }
   // debug("before return");
   return rgbData;
}

class PtrQueue {

public:

    PtrQueue() : size(0){
       // debug("Queue created");
       empty_mutex.lock();
    }


    std::vector<PGX_FRAME_BUFFER> arr;

    void push(PGX_FRAME_BUFFER ptr) {
    
        std::lock_guard<std::mutex> lock(queue_mutex);
        arr.push_back(ptr);

        size++;
        empty_mutex.unlock();

    }
    int size;
    PGX_FRAME_BUFFER read() {
        return arr.front();
    }

    void pop_front() {
    //    debug("queue pop front entered");
        queue_mutex.lock();
     //   debug("queue mutex locked");
        int status = GXQBuf(hDevice, arr.front());
    //    std::cout << "Status " << status << std::endl;
    //    debug("buf released");
        arr.erase(arr.begin());
        queue_mutex.unlock();
    //    debug("buf erased");
        size--;
        if (size == 0) {
          //  debug("try to lock empty mutex");
            empty_mutex.lock();
           // debug("empty mutex locked");
        }
       // debug("end of pop");
    }


};
PtrQueue queue;

class DecodedQueue {

public:

    DecodedQueue() : size(0){
       // debug("Queue created");
       empty_mutex1.lock();
    }


    std::vector<uint8_t*> arr;

    void push(uint8_t* ptr) {
    
        std::lock_guard<std::mutex> lock(queue_mutex1);
        arr.push_back(ptr);

        size++;
        empty_mutex1.unlock();

    }
    int size;
    uint8_t* read() {
        return arr.front();
    }

    void pop_front() {
        queue_mutex1.lock();
      //  GXQBuf(hDevice, arr.front());
        free(arr.front());
        arr.erase(arr.begin());
        queue_mutex1.unlock();
        size--;
        if (size == 0) {
            empty_mutex1.lock();
        }
    }


};
DecodedQueue dqueue;


void signal_handler(int signal) {
    std::cout << "Signal recieved" << std::endl;
//    int status = GXSendCommand(hDevice, GX_COMMAND_ACQUISITION_STOP);
   main_mutex.unlock();
    stop = 0;
}


void frameProc() {
    auto &infer_model1 = configured_infer_model;
    auto bindings = infer_model1.create_bindings().expect("Failed to create bindings");
    empty_mutex1.lock();
    //std::cout<< "ready" << std::endl;
    for (const auto &input_name : infer_model->get_input_names()) {
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
      //  printBytes(queue.read()->data);
        bindings.input(input_name)->set_buffer(MemoryView(dqueue.read(), input_frame_size));
    }

    std::shared_ptr<uint8_t> output_buffer;
    for (const auto &output_name : infer_model->get_output_names()) {
        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
        output_buffer = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto job = infer_model1.run_async(bindings,[&](const AsyncInferCompletionInfo & info){
        debug("Inferenced!");
        // auto end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> duration = end - start;
        // times.push_back( duration.count() / 1000);
        processed_index++;

        dqueue.pop_front();
    }).expect("Failed to start async infer job");

}

void decode() {
    while(stop == 1) {
      //  debug("before empty mutex");
        empty_mutex.lock();
        uint8_t* ptr = (uint8_t*)queue.read()->pImgBuf;
        Projection block1(ptr, 0, 0, 2592, 1944, W, H);
        Projection block2(ptr, 976, 0, 2592, 1944, W, H);
        Projection block3(ptr, 0, 652, 2592, 1944, W, H);
        Projection block4(ptr, 976, 652, 2592, 1944, W, H);
        uint8_t* pic1 = toRGB(block1);
        uint8_t* pic2 = toRGB(block2);
        uint8_t* pic3 = toRGB(block3);
        uint8_t* pic4 = toRGB(block4);
       //  debug("after rgb");
        queue.pop_front();
      //  debug("after queue pop");
        dqueue.push(pic1);
        dqueue.push(pic2);
        dqueue.push(pic3);
        dqueue.push(pic4);
      //  debug("after dqueue push");
        
        
    }
}

void startInference() {
    while (stop == 1) {
        frameProc();
    }
    
}

int main(int argc, char* argv[]) {
    debug("Enter main");
    main_mutex.lock();
    debug("Mutex locked");
    std::signal(SIGINT, signal_handler);
    debug("Handler added");
   HEF_FILE = argv[1];

    vdevice = VDevice::create().expect("Failed to create vdevice");
    infer_model = vdevice->create_infer_model(HEF_FILE).expect("Failed to create infer model");
    configured_infer_model = infer_model->configure().expect("Failed to configure model");


//    drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
//     debug("Drm opened");
//     if (drm_fd < 0) {
//         perror("open");
//         return 1;
//     }
//     drmModeRes* res = drmModeGetResources(drm_fd);
//     drmModeConnector* conn = nullptr;
//     drmModeModeInfo mode;
//     uint32_t conn_id = 0;
//     for (int i = 0; i < res->count_connectors; ++i) {
//         conn = drmModeGetConnector(drm_fd, res->connectors[i]);
//         if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
//             mode = conn->modes[0];
//             conn_id = conn->connector_id;
//             break;
//         }
//         drmModeFreeConnector(conn);
//     }
//     debug("Drm connector");

//     if (!conn_id) {
//         std::cerr << "No connected display found\n";
//         return 1;
//     }

//     drmModeEncoder* enc = drmModeGetEncoder(drm_fd, conn->encoder_id);
//     uint32_t crtc_id = enc->crtc_id;
//     old_crtc = drmModeGetCrtc(drm_fd, crtc_id);
//     debug("Drm encoder");
//     struct drm_mode_create_dumb create = {};
//     create.width = mode.hdisplay;
//     create.height = mode.vdisplay;
//     create.bpp = 32;
//     ioctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create);
//     handle = create.handle;
//     pitch = create.pitch;
//     size = create.size;

//     struct drm_mode_map_dumb map_dumb = {};
//     map_dumb.handle = handle;
//     ioctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_dumb);
//     map = (uint8_t*)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, map_dumb.offset);
//     debug("Drm mmap");
    
    // fb.width = mode.hdisplay;
    // fb.height = mode.vdisplay;
    // fb.pitch = pitch;
    // fb.bpp = 32;
    // fb.depth = 24;
    // fb.handle = handle;
    // drmModeAddFB(drm_fd, fb.width, fb.height, fb.depth, fb.bpp, pitch, handle, &fb_id);

    // drmModeSetCrtc(drm_fd, crtc_id, fb_id, 0, 0, &conn_id, 1, &mode);

    // vdevice = VDevice::create().expect("Failed create vdevice");
    // infer_model = vdevice->create_infer_model(HEF_FILE).expect("Failed to create infer model");
    // configured_infer_model = infer_model->configure().expect("Failed to create configured infer model");

    debug("Drm end");
    GX_STATUS status = GX_STATUS_SUCCESS;
    
    uint32_t nDeviceNum = 0;
    //Initializes the library.
    status = GXInitLib();
    debug("Gx inited");
    if (status != GX_STATUS_SUCCESS) {
        debug("Init failed");
        std::cout << status << std::endl;
        return 0;
    }
    //Updates the enumeration list for the devices.
    status = GXUpdateDeviceList(&nDeviceNum, 1000);
    if ((status != GX_STATUS_SUCCESS) || (nDeviceNum <= 0)) {
        debug("Update list failed");
        return 0;
    }
    //Opens the device.
    status = GXOpenDeviceByIndex(1, &hDevice);

    if (status != GX_STATUS_SUCCESS) {
        debug("Open failed");
        return 0;
    }

    debug("Start inference");
   
    statu("aq mode on", GXSetEnum(hDevice, GX_ENUM_ACQUISITION_FRAME_RATE_MODE , GX_ACQUISITION_FRAME_RATE_MODE_ON));
    

    status = GXStreamOn(hDevice);
    std::thread inference(startInference);
    std::thread dec(decode);
    while (stop == 1) {
        PGX_FRAME_BUFFER pFrameBuffer;

        status = GXDQBuf(hDevice, &pFrameBuffer, 1000);
       // std::this_thread::sleep_for(std::chrono::microseconds(35));

        queue.push(pFrameBuffer);
    }
    
    // if (old_crtc) {
    //     drmModeSetCrtc(drm_fd, old_crtc->crtc_id, old_crtc->buffer_id,
    //                    old_crtc->x, old_crtc->y,
    //                    &conn_id, 1, &old_crtc->mode);
    //     drmModeFreeCrtc(old_crtc);
    // }

    // munmap(map, size);
    // drmModeRmFB(drm_fd, fb_id);

    // struct drm_mode_destroy_dumb destroy = {0};
    // destroy.handle = handle;
    // drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);

    // drmModeFreeConnector(conn);
    // drmModeFreeResources(res);
    // close(drm_fd);
    inference.join();
    dec.join();
    status = GXCloseDevice(hDevice);
    status = GXCloseLib();
    return 0;
}