#include "../include/GxIAPI.h"
#include <vector>
#include <mutex>
#include "rga/RgaUtils.h"
#include "rga/im2d.h"
#include "rga/rga.h"
#include <bits/stdc++.h>

std::mutex queue_mutex;

class PtrQueue {

public:

    PtrQueue() : size(0){
       // debug("Queue created");
    }


    std::vector<char*> arr;

    void push(char* ptr) {
    
        std::lock_guard<std::mutex> lock(queue_mutex);
        arr.push_back(ptr);

        size++;

    }
    int size;
    char* read() {
        return arr.front();
    }

    void pop_front() {
        free(arr.front());
        arr.erase(arr.begin());
        size--;
    }


};
PtrQueue queue;
void debug(std::string message) {
    std::cout << "DEBUG: " << message << std::endl;
}

int main(int argc, char* argv[])
{
GX_STATUS status = GX_STATUS_SUCCESS;
GX_DEV_HANDLE hDevice = NULL;
uint32_t nDeviceNum = 0;
//Initializes the library.
status = GXInitLib();
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
GX_STATUS stat = GXSetEnum(hDevice, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_RGB8_PLANAR);
std::cout << "set format " << stat << std::endl;
if (status != GX_STATUS_SUCCESS) {
    debug("Open failed");
    return 0;
}
if (status == GX_STATUS_SUCCESS) {
    //Define the incoming parameters of GXDQBuf.
    PGX_FRAME_BUFFER pFrameBuffer;
    //Stream On.
    status = GXStreamOn(hDevice);
    if (status == GX_STATUS_SUCCESS) {
        //Calls GXDQBuf to get a frame of image.
        status = GXDQBuf(hDevice, &pFrameBuffer, 1000);


        //Calls GXQBuf to put the image buffer back into the library
        //and continue acquiring.
        status = GXQBuf(hDevice, pFrameBuffer);
        std::cout << "Format " << std::hex << pFrameBuffer->nPixelFormat << std::endl;
        printf( "Height - %d\n", pFrameBuffer->nHeight );
        printf( "Width - %d\n", pFrameBuffer->nWidth );
        printf( "Size - %d\n", pFrameBuffer->nImgSize);


//         rga_buffer_t src;
//         rga_buffer_t dst;
//         im_rect      src_rect;
//         im_rect      dst_rect;
//         rga_buffer_handle_t src_handle;
//         rga_buffer_handle_t dst_handle;
//         memset(&src_rect, 0, sizeof(src_rect));
//         memset(&dst_rect, 0, sizeof(dst_rect));
//         memset(&src, 0, sizeof(src));
//         memset(&dst, 0, sizeof(dst));
       char* src_buf = (char*)malloc(480*640*get_bpp_from_format(RK_FORMAT_YVYU_422));
//         char* dst_buf = (char*)malloc(640*640*get_bpp_from_format(RK_FORMAT_RGB_888));

//         src_handle = importbuffer_virtualaddr(pFrameBuffer->pImgBuf, 2592, 1944, RK_FORMAT_YVYU_422);
//         dst_handle = importbuffer_virtualaddr(dst_buf, 640, 640, RK_FORMAT_RGB_888);
//         src = wrapbuffer_handle(src_handle, 640, 480, RK_FORMAT_YVYU_422);
//         dst = wrapbuffer_handle(dst_handle, 640, 640, RK_FORMAT_RGB_888);
        IM_STATUS STATUS = imresize(src, dst);
//  //       if (pkt->convergence_duration == video_stream_idx) {
//             //debug("Before push");
//         queue.push(dst_buf);
//         releasebuffer_handle(src_handle);
//         releasebuffer_handle(dst_handle);

    }
    //Sends a stop acquisition command.
    status = GXStreamOff(hDevice);
}
status = GXCloseDevice(hDevice);
status = GXCloseLib();
return 0;
}