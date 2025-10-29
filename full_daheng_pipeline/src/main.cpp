#include "../include/camera.hpp"
#include "../include/relay.hpp"
// #include "../include/modbus.hpp"
#include "../include/GxIAPI.h"
#include "../include/hailoNN.hpp"
#include "../include/rockChipNN.hpp"
#include "../include/utils.hpp"
#include "rga/RgaUtils.h"
#include "rga/im2d.h"
#include "rga/rga.h"
#include "../include/gpio.hpp"
#include <thread>

std::mutex main_mutex;
AbstractNNBase* NN = NULL;
void* context = NULL;
int stop = 0;
GPIO gpio;

void sigint_handler(int) { 
    main_mutex.unlock();
    stop = 1;
}



void controlRelay(Relay* relay) {

    Request req(8);

    req.address(0x01).read_input(0, 2);

    int snap = 0;

    while( true ) {

        relay->send(req);

        // std::cout << std::endl;
        usleep(1000 * 60);
        uint8_t* buffer = (uint8_t*) malloc(15 * sizeof(uint8_t));
        int n = relay->receive(&buffer);
        for(int i=0; i<15; i++) {
            printf("%02X ", buffer[i]);
            
        }

        std::cout << std::endl;
        if (buffer[3] != 0x00 && (snap == 0) ) {
            log("trigger");
            gpio.trigger();
            snap = 1;
        }

        if (buffer[3] == 0x00 && (snap == 1) ) {
    
            snap = 0;
        }



        if (stop ==1) {
            break;
        }

        free(buffer);


    }
    


}


void callBack(GX_FRAME_CALLBACK_PARAM* pFrame) {
    std::cout << "Call Back" << std::endl;

    uint8_t* ptr = (uint8_t*)pFrame->pImgBuf;

    Projection block1(ptr, 0, 0, 1280, 1280, 640, 640);

    standart_inference_ctx ctx;
    uint8_t* data = toRGB(block1);
    drawPicture(&block1, data);
    ctx.input_buffer = data;
    log("Transformed to rgb");
    ctx.proj = block1;
    context = &ctx;
    NN->inference(context);
}



int main(int argc, char* argv[]) {
    
    main_mutex.lock();

    
    int drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    drm_init(drm_fd);

    Relay relay("/dev/ttyUSB0");
    gpio.init(39);

    AICamera cam;
    cam.setHeight(1280);
    cam.setWidth(1280);
    cam.setOffsetX(656);
    cam.setOffsetY(332);
    if ( std::string(argv[2]) == "hailo") {
        HailoNN* hailo = new HailoNN(argv[1]);
        NN = hailo;

    }
    if ( std::string(argv[2]) == "rockchip") {
        RockChipNN* rockchip = new RockChipNN(argv[1]);
        NN = rockchip;

    }
    std::thread contol(controlRelay, &relay);

    cam.setCallBack(callBack);
    cam.startCapture();
    main_mutex.lock();

    cam.stopCapture();

    drm_destroy(drm_fd);
 
    
    

    return 0;
}

