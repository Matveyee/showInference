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

void debug(std::string message) {
    std::cout << "DEBUG: " << message << std::endl;
}

int main() {

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

    

    status = GXStreamOn(hDevice);


}